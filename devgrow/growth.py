"""Function-preserving growth of a GPT (see model.py) plus Adam-state transfer.

Everything is expressed as per-axis *source index maps*. For every parameter tensor
and every axis, a LongTensor says "new position i comes from old position map[i]"
(-1 = brand-new position). The same maps are applied to the weights and to the
Adam moments, so optimizer state follows the weights automatically.

Roles of a tensor decide how brand-new entries are filled:
  reader : reads from the residual stream / hidden units (q,k,v,gate,up)  -> small random
  writer : writes into the residual stream (o, down, embedding)           -> ZERO
  norm   : RMSNorm gain                                                   -> 1
Reader-random + writer-zero is what keeps the function exactly unchanged while
still letting gradient reach the new parameters (writer-zero alone is a dead start;
both-zero is a saddle).

width_init:
  net2net : duplicate random existing heads / MLP units, split the writer weights
            among the copies (Chen et al. 2016); tiny noise breaks the symmetry.
  zero    : new units = random readers + zero writers.
  random  : new units fully random (NOT function preserving; control condition).
d_model growth always zero-pads the residual stream and rescales RMSNorm gains by
sqrt(d_old / d_new) so that normalisation statistics are unchanged.

depth_init (for the *new* layers):
  copy_zero  : copy a neighbouring layer, zero its output projections -> identity
  fresh_zero : fresh random layer, zero output projections            -> identity
  copy       : plain copy (Gong et al. stacking)                       -> NOT preserving
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from configs import ModelConfig, plan_depth
from model import GPT

# key -> (role, axis kinds per tensor dimension)
LAYER_SPEC = {
    "attn_norm.weight": ("norm", ("dm",)),
    "attn.q.weight": ("reader", ("head", "dm")),
    "attn.k.weight": ("reader", ("head", "dm")),
    "attn.v.weight": ("reader", ("head", "dm")),
    "attn.o.weight": ("writer", ("dm", "head")),
    "mlp_norm.weight": ("norm", ("dm",)),
    "mlp.gate.weight": ("reader", ("ff", "dm")),
    "mlp.up.weight": ("reader", ("ff", "dm")),
    "mlp.down.weight": ("writer", ("dm", "ff")),
}
OUT_PROJ_KEYS = {"attn.o.weight", "mlp.down.weight"}


@dataclass
class GrowthState:
    moments: Dict[str, Tuple[torch.Tensor, torch.Tensor]]  # name -> (exp_avg, exp_avg_sq), CPU
    step: float


# ----------------------------------------------------------------------------
# index-map helpers
# ----------------------------------------------------------------------------
def _unit_map(old_n: int, new_n: int, mode: str, g: torch.Generator) -> torch.Tensor:
    base = torch.arange(old_n)
    extra = new_n - old_n
    if extra == 0:
        return base
    if mode == "net2net":
        add = torch.randint(0, old_n, (extra,), generator=g)
    else:
        add = torch.full((extra,), -1, dtype=torch.long)
    return torch.cat([base, add])


def _unit_scale(m: torch.Tensor, old_n: int) -> torch.Tensor:
    """1/(number of copies of the source unit) for positions that have a source."""
    valid = m >= 0
    counts = torch.bincount(m[valid], minlength=old_n).float()
    s = torch.ones(m.shape[0])
    s[valid] = 1.0 / counts[m[valid]]
    return s


def _expand_heads(m: torch.Tensor, hd: int) -> torch.Tensor:
    rows = m.unsqueeze(1) * hd + torch.arange(hd).unsqueeze(0)
    rows[m < 0] = -1
    return rows.reshape(-1)


def _gather(t: torch.Tensor, maps: List[Optional[torch.Tensor]]):
    out = t
    valid = torch.ones((), dtype=torch.bool)
    for d, m in enumerate(maps):
        if m is None:
            continue
        out = out.index_select(d, m.clamp(min=0))
        shp = [1] * t.dim()
        shp[d] = -1
        valid = valid & (m >= 0).view(shp)
    return out, valid.expand_as(out)


def grow_tensor(old, m1, m2, maps, scales, role, std, g, fresh=False,
                zero_out=False, preserve=True, noise_slices=(), noise=0.0):
    """Return (new_weight, new_exp_avg, new_exp_avg_sq) for one tensor."""
    out, valid = _gather(old, maps)
    shape = out.shape

    if role == "norm":
        fill = torch.ones(shape)
    elif role == "writer" and preserve:
        fill = torch.zeros(shape)
    else:
        fill = torch.randn(shape, generator=g) * std

    if fresh:  # brand-new identity layer: random readers, zero writers, unit norms
        w = torch.zeros(shape) if role == "writer" else fill.clone()
    else:
        wv = out.clone()
        for d, s in enumerate(scales):
            if s is not None:
                shp = [1] * old.dim()
                shp[d] = -1
                wv = wv * s.view(shp)
        w = torch.where(valid, wv, fill)
        if zero_out and role == "writer":
            w = torch.zeros(shape)
        if role == "reader" and noise > 0:
            ref = float(old.std()) if old.numel() > 1 else std
            for d, start in noise_slices:
                if w.shape[d] > start:
                    sl = w.narrow(d, start, w.shape[d] - start)
                    sl.add_(torch.randn(sl.shape, generator=g) * noise * ref)

    if m1 is None:
        return w, None, None
    g1, _ = _gather(m1, maps)
    g2, _ = _gather(m2, maps)
    if fresh:
        n1 = torch.zeros(shape)
        n2 = torch.full(shape, float(m2.mean()))
    else:
        n1 = torch.where(valid, g1, torch.zeros_like(g1))
        n2 = torch.where(valid, g2, torch.full_like(g2, float(m2.mean())))
    return w, n1, n2


# ----------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------
@torch.no_grad()
def grow(model: GPT, new_cfg: ModelConfig, opt: Optional[torch.optim.Optimizer] = None,
         depth_place: str = "spread", depth_init: str = "copy_zero",
         width_init: str = "net2net", seed: int = 0, dup_noise: float = 1e-3):
    oc, nc = model.cfg, new_cfg
    assert nc.head_dim == oc.head_dim and nc.vocab_size == oc.vocab_size
    assert nc.n_layers >= oc.n_layers and nc.d_model >= oc.d_model
    assert nc.n_heads >= oc.n_heads and nc.d_ff >= oc.d_ff
    assert depth_init in ("copy_zero", "fresh_zero", "copy")
    assert width_init in ("net2net", "zero", "random")

    device = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    preserve_width = width_init != "random"
    std = nc.init_std
    hd = oc.head_dim

    old = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
    old_m: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    step = 0.0
    if opt is not None:
        ok = True
        for n, p in model.named_parameters():
            st = opt.state.get(p, {})
            if "exp_avg" not in st:
                ok = False
                break
            old_m[n] = (st["exp_avg"].detach().cpu().float(), st["exp_avg_sq"].detach().cpu().float())
            step = float(st["step"])
        if not ok:
            old_m = {}

    def mom(name):
        return old_m.get(name, (None, None))

    new_sd: Dict[str, torch.Tensor] = {}
    states: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def put(name, res):
        w, n1, n2 = res
        new_sd[name] = w
        if n1 is not None:
            states[name] = (n1, n2)

    dm_map = torch.cat([torch.arange(oc.d_model), torch.full((nc.d_model - oc.d_model,), -1, dtype=torch.long)])
    norm_scale = torch.full((nc.d_model,), math.sqrt(oc.d_model / nc.d_model))

    # ---- embedding (tied head) and final norm --------------------------------
    m1, m2 = mom("tok_emb.weight")
    put("tok_emb.weight", grow_tensor(old["tok_emb.weight"], m1, m2, [None, dm_map], [None, None],
                                      "writer", std, g, preserve=preserve_width))
    m1, m2 = mom("norm_f.weight")
    put("norm_f.weight", grow_tensor(old["norm_f.weight"], m1, m2, [dm_map], [norm_scale], "norm", std, g))

    # ---- transformer blocks --------------------------------------------------
    plan = plan_depth(oc.n_layers, nc.n_layers, depth_place)
    for l, (s, is_copy) in enumerate(plan):
        fresh = is_copy and depth_init == "fresh_zero"
        zero_out = is_copy and depth_init == "copy_zero"
        head_map = _unit_map(oc.n_heads, nc.n_heads, width_init, g)
        ff_map = _unit_map(oc.d_ff, nc.d_ff, width_init, g)
        head_rows = _expand_heads(head_map, hd)
        head_scale = _unit_scale(head_map, oc.n_heads).repeat_interleave(hd)
        ff_scale = _unit_scale(ff_map, oc.d_ff)

        for key, (role, axes) in LAYER_SPEC.items():
            maps, scales, nslices = [], [], []
            for d, ax in enumerate(axes):
                if ax == "dm":
                    maps.append(dm_map)
                    scales.append(norm_scale if role == "norm" else None)
                elif ax == "head":
                    maps.append(head_rows)
                    scales.append(head_scale if role == "writer" else None)
                    if width_init == "net2net":
                        nslices.append((d, oc.n_heads * hd))
                else:  # ff
                    maps.append(ff_map)
                    scales.append(ff_scale if role == "writer" else None)
                    if width_init == "net2net":
                        nslices.append((d, oc.d_ff))
            name_old = f"blocks.{s}.{key}"
            name_new = f"blocks.{l}.{key}"
            m1, m2 = mom(name_old)
            put(name_new, grow_tensor(
                old[name_old], m1, m2, maps, scales, role, std, g,
                fresh=fresh, zero_out=zero_out and key in OUT_PROJ_KEYS,
                preserve=preserve_width, noise_slices=nslices, noise=dup_noise))

    new_model = GPT(nc)
    new_model.load_state_dict(new_sd, strict=True)
    new_model.to(device)
    states = {k: (a.to(device), b.to(device)) for k, (a, b) in states.items()}
    return new_model, GrowthState(states if old_m else {}, step)


def attach_optimizer_state(new_opt: torch.optim.Optimizer, new_model: GPT, gs: GrowthState):
    """Install transferred Adam moments into a freshly built AdamW (same param grouping)."""
    if not gs.moments:
        return
    for name, p in new_model.named_parameters():
        if name in gs.moments:
            m1, m2 = gs.moments[name]
            new_opt.state[p] = {"step": torch.tensor(gs.step, dtype=torch.float32),
                                "exp_avg": m1.to(p.device, p.dtype).contiguous(),
                                "exp_avg_sq": m2.to(p.device, p.dtype).contiguous()}


@torch.no_grad()
def preservation_gap(old_model: GPT, new_model: GPT, x: torch.Tensor) -> float:
    """max |logits_new - logits_old| on a batch (should be ~0 for preserving growth)."""
    old_model.eval(); new_model.eval()
    a, _ = old_model(x)
    b, _ = new_model(x)
    return float((a.float() - b.float()).abs().max())

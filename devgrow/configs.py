"""Model config, size ladder, FLOPs accounting and the experimental conditions.

This file is deliberately torch-free so the schedule / FLOPs logic can be
inspected and unit-tested without a GPU stack.
"""
from dataclasses import dataclass, replace, asdict, field
from typing import List, Tuple


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8192
    n_layers: int = 8
    d_model: int = 512
    n_heads: int = 8
    head_dim: int = 64          # kept FIXED across growth (RoPE + head growth stay trivial)
    d_ff: int = 1408            # SwiGLU hidden size (~8/3 * d_model)
    max_seq_len: int = 256
    rope_base: float = 10000.0
    init_std: float = 0.02

    @property
    def d_attn(self) -> int:
        return self.n_heads * self.head_dim

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------
# Size ladder. Stage 0 -> 1 -> 2. FINAL is the fixed-size baseline (condition B).
# Edit these three dicts to scale the whole study up or down.
# --------------------------------------------------------------------------
SMALL = dict(n_layers=2, d_model=256, n_heads=4, d_ff=704)
MID = dict(n_layers=4, d_model=384, n_heads=6, d_ff=1024)
FINAL = dict(n_layers=8, d_model=512, n_heads=8, d_ff=1408)
LADDER = [SMALL, MID, FINAL]

DEPTH_KEYS = {"n_layers"}
WIDTH_KEYS = {"d_model", "n_heads", "d_ff"}
ALL_KEYS = DEPTH_KEYS | WIDTH_KEYS


def nonembedding_params(c: ModelConfig) -> int:
    per_layer = 4 * c.d_model * c.d_attn + 3 * c.d_model * c.d_ff + 2 * c.d_model
    return c.n_layers * per_layer + c.d_model


def flops_per_token(c: ModelConfig, seq_len: int) -> float:
    """Training FLOPs / token (fwd+bwd). 6N for matmuls (incl. tied LM head)
    plus the attention score/value term 12 * L * d_attn * T (causal not halved)."""
    n = nonembedding_params(c) + c.vocab_size * c.d_model
    return 6.0 * n + 12.0 * c.n_layers * c.d_attn * seq_len


def plan_depth(old_n: int, new_n: int, place: str) -> List[Tuple[int, bool]]:
    """For each new layer index return (source_old_layer, is_new_copy).

    spread: new layers are placed right after the layer they copy, spread evenly
            through the stack (e.g. 2->4 gives [0, 0', 1, 1']).
    stack : original layers stay in place, copies of layers 0..k are appended on
            top (Gong et al. 2019 "stacking").
    """
    assert new_n >= old_n, "can only grow"
    if place == "spread":
        src = [min(old_n - 1, (i * old_n) // new_n) for i in range(new_n)]
        return [(s, i > 0 and src[i] == src[i - 1]) for i, s in enumerate(src)]
    if place == "stack":
        return [(i, False) for i in range(old_n)] + [(t % old_n, True) for t in range(new_n - old_n)]
    raise ValueError(place)


@dataclass
class RunPlan:
    name: str
    stages: List[Tuple[int, ModelConfig]]   # (start_step, config); first start_step == 0
    total_steps: int
    depth_place: str = "spread"
    depth_init: str = "copy_zero"           # copy_zero | fresh_zero | copy
    width_init: str = "net2net"             # net2net | zero | random
    notes: str = ""


def _cfg(base: ModelConfig, grow_keys, stage_idx: int) -> ModelConfig:
    d = {k: (LADDER[stage_idx][k] if k in grow_keys else FINAL[k]) for k in ALL_KEYS}
    return replace(base, **d)


def plan_flops(plan: RunPlan, tokens_per_step: int, seq_len: int) -> float:
    total = 0.0
    bounds = [s for s, _ in plan.stages] + [plan.total_steps]
    for i, (start, cfg) in enumerate(plan.stages):
        total += (bounds[i + 1] - start) * tokens_per_step * flops_per_token(cfg, seq_len)
    return total


CONDITIONS = [
    "A_small",         # fixed, initial size
    "B_large",         # fixed, final size  (main baseline)
    "C_depth",         # depth growth, width fixed at final
    "D_width",         # width growth (d_model, heads, d_ff), depth fixed at final
    "E_both",          # depth + width growth
    "F_nopreserve",    # like E but NOT function preserving (copy init / random width init)
    "G_flops_matched", # fixed large model trained for the same FLOPs as E_both
]


def build_plan(cond: str, total_steps: int, tokens_per_step: int, seq_len: int,
               vocab_size: int, fracs=(0.0, 0.25, 0.5), depth_place: str = "spread",
               width_init: str = "net2net") -> RunPlan:
    base = ModelConfig(vocab_size=vocab_size, max_seq_len=seq_len)
    starts = [int(f * total_steps) for f in fracs]
    assert starts[0] == 0

    if cond == "A_small":
        return RunPlan(cond, [(0, _cfg(base, ALL_KEYS, 0))], total_steps)
    if cond == "B_large":
        return RunPlan(cond, [(0, _cfg(base, set(), 0))], total_steps)

    grow_keys = {"C_depth": DEPTH_KEYS, "D_width": WIDTH_KEYS,
                 "E_both": ALL_KEYS, "F_nopreserve": ALL_KEYS}.get(cond)
    if grow_keys is not None:
        stages = [(starts[i], _cfg(base, grow_keys, i)) for i in range(len(starts))]
        if cond == "F_nopreserve":
            return RunPlan(cond, stages, total_steps, depth_place, "copy", "random")
        return RunPlan(cond, stages, total_steps, depth_place, "copy_zero", width_init)

    if cond == "G_flops_matched":
        e = build_plan("E_both", total_steps, tokens_per_step, seq_len, vocab_size, fracs, depth_place, width_init)
        target = plan_flops(e, tokens_per_step, seq_len)
        large = _cfg(base, set(), 0)
        steps = int(target / (flops_per_token(large, seq_len) * tokens_per_step))
        return RunPlan(cond, [(0, large)], steps,
                       notes=f"fewer steps ({steps} vs {total_steps}) to match E_both FLOPs")
    raise ValueError(f"unknown condition {cond}; choose from {CONDITIONS}")

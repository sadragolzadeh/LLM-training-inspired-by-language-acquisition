"""Requires torch + pytest. Run: pytest tests/ -q"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest, torch
from configs import ModelConfig
from model import GPT
from growth import grow, attach_optimizer_state, preservation_gap

OLD = ModelConfig(vocab_size=100, n_layers=2, d_model=32, n_heads=2, head_dim=16, d_ff=64, max_seq_len=16)
GROWS = {
    "depth": dict(n_layers=4),
    "heads": dict(n_heads=3),
    "ff": dict(d_ff=96),
    "dmodel": dict(d_model=48),
    "all": dict(n_layers=5, d_model=48, n_heads=4, d_ff=96),
}

def _x():
    return torch.randint(0, 100, (3, 16))

@pytest.mark.parametrize("kind,place,dinit,winit", list(itertools.product(
    GROWS, ["spread", "stack"], ["copy_zero", "fresh_zero"], ["net2net", "zero"])))
def test_preserving(kind, place, dinit, winit):
    torch.manual_seed(0)
    m = GPT(OLD)
    new, _ = grow(m, ModelConfig(**{**OLD.to_dict(), **GROWS[kind]}), None, place, dinit, winit, dup_noise=0.0)
    tol = 2e-2 if "d_model" in GROWS[kind] else 1e-4   # RMSNorm eps shifts slightly when padding d_model
    assert preservation_gap(m, new, _x()) < tol

def test_non_preserving_controls():
    torch.manual_seed(0)
    m = GPT(OLD)
    new, _ = grow(m, ModelConfig(**{**OLD.to_dict(), **GROWS["depth"]}), None, "spread", "copy", "zero")
    assert preservation_gap(m, new, _x()) > 1e-3
    new, _ = grow(m, ModelConfig(**{**OLD.to_dict(), **GROWS["ff"]}), None, "spread", "copy_zero", "random")
    assert preservation_gap(m, new, _x()) > 1e-4

def test_optimizer_transfer_and_trainability():
    torch.manual_seed(0)
    m = GPT(OLD)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(3):
        _, l = m(_x(), _x()); opt.zero_grad(); l.backward(); opt.step()
    new_cfg = ModelConfig(**{**OLD.to_dict(), **GROWS["all"]})
    new, gs = grow(m, new_cfg, opt, "spread", "copy_zero", "zero")
    nopt = torch.optim.AdamW(new.parameters(), lr=1e-3)
    attach_optimizer_state(nopt, new, gs)
    for p in new.parameters():
        assert nopt.state[p]["exp_avg"].shape == p.shape and nopt.state[p]["exp_avg_sq"].shape == p.shape
    x, y = _x(), _x()
    _, l = new(x, y); l.backward()
    # new heads' output columns (zero-init writers) must receive gradient -> growth is not a dead start
    assert new.blocks[0].attn.o.weight.grad[:, OLD.n_heads * OLD.head_dim:].abs().sum() > 0
    nopt.step()
    assert all(torch.isfinite(p).all() for p in new.parameters())

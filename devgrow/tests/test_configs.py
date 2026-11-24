import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs import *

def test_plan_depth():
    assert plan_depth(2, 4, "spread") == [(0, False), (0, True), (1, False), (1, True)]
    assert plan_depth(2, 2, "spread") == [(0, False), (1, False)]
    assert plan_depth(2, 4, "stack") == [(0, False), (1, False), (0, True), (1, True)]
    p = plan_depth(4, 8, "spread")
    assert sorted(s for s, c in p if not c) == [0, 1, 2, 3]

def test_flops_matched():
    kw = dict(total_steps=1000, tokens_per_step=16384, seq_len=256, vocab_size=8192)
    e, g, b = build_plan("E_both", **kw), build_plan("G_flops_matched", **kw), build_plan("B_large", **kw)
    fe, fg, fb = (plan_flops(p, 16384, 256) for p in (e, g, b))
    assert g.total_steps < 1000 and abs(fg - fe) / fe < 0.01 and fe < fb

def test_stages():
    kw = dict(total_steps=1000, tokens_per_step=16384, seq_len=256, vocab_size=8192)
    c = build_plan("C_depth", **kw)
    assert [s.n_layers for _, s in c.stages] == [2, 4, 8] and all(s.d_model == 512 for _, s in c.stages)
    d = build_plan("D_width", **kw)
    assert [s.d_model for _, s in d.stages] == [256, 384, 512] and all(s.n_layers == 8 for _, s in d.stages)
    assert [st for st, _ in build_plan("E_both", **kw).stages] == [0, 250, 500]

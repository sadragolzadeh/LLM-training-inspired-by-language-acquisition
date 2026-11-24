"""Linguistic evaluations, all based on sentence log-probability.

  * eval_blimp : accuracy on BLiMP minimal pairs (dir of *.jsonl with sentence_good /
                 sentence_bad), macro-averaged over phenomena.
  * eval_morph : past-tense probe. For irregular verbs compares the correct form
                 ("went") with the over-regularised form ("goed"). Tracking
                 `irreg_acc` across checkpoints is the U-shaped-learning diagnostic:
                 look for a dip after an early plateau. Regular verbs are a control.

Caveat: sentences are scored by the SUM of token log-probs (as in BLiMP), so a form that
tokenises into more pieces is penalised. Read `irreg_margin` together with `irreg_acc`.
"""
import glob
import json
import os

import torch

IRREGULAR = [  # (base, correct past, over-regularised)
    ("go", "went", "goed"), ("eat", "ate", "eated"), ("run", "ran", "runned"),
    ("come", "came", "comed"), ("take", "took", "taked"), ("give", "gave", "gived"),
    ("make", "made", "maked"), ("say", "said", "sayed"), ("fall", "fell", "falled"),
    ("break", "broke", "breaked"), ("bring", "brought", "bringed"), ("buy", "bought", "buyed"),
    ("catch", "caught", "catched"), ("draw", "drew", "drawed"), ("drink", "drank", "drinked"),
    ("drive", "drove", "drived"), ("find", "found", "finded"), ("fly", "flew", "flyed"),
    ("get", "got", "getted"), ("grow", "grew", "growed"), ("hold", "held", "holded"),
    ("know", "knew", "knowed"), ("leave", "left", "leaved"), ("sit", "sat", "sitted"),
    ("sleep", "slept", "sleeped"), ("speak", "spoke", "speaked"), ("swim", "swam", "swimmed"),
    ("tell", "told", "telled"), ("think", "thought", "thinked"), ("throw", "threw", "throwed"),
    ("write", "wrote", "writed"), ("win", "won", "winned"), ("wear", "wore", "weared"),
]
REGULAR = [("walk", "walked"), ("jump", "jumped"), ("play", "played"), ("look", "looked"),
           ("want", "wanted"), ("help", "helped"), ("call", "called"), ("watch", "watched"),
           ("open", "opened"), ("start", "started"), ("work", "worked"), ("talk", "talked")]
TEMPLATES = ["Yesterday, the boy {} there.", "Yesterday, she {} home.", "Last week, we {} it."]


@torch.no_grad()
def score_sentences(model, tok, sentences, device, batch_size=64):
    """Sum of token log-probs for each sentence (an <eos> is used as BOS)."""
    model.eval()
    max_len = model.cfg.max_seq_len
    ids = [([tok.eos_id] + tok.encode(s))[:max_len] for s in sentences]
    out = []
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if "cuda" in str(device) \
        else torch.autocast(device_type="cpu", enabled=False)
    for i in range(0, len(ids), batch_size):
        chunk = ids[i:i + batch_size]
        L = max(len(s) for s in chunk)
        x = torch.full((len(chunk), L), tok.eos_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), L))
        for j, s in enumerate(chunk):
            x[j, :len(s)] = torch.tensor(s)
            mask[j, :len(s)] = 1
        x, mask = x.to(device), mask.to(device)
        with ctx:
            logits, _ = model(x)
        logp = torch.log_softmax(logits.float(), dim=-1)[:, :-1]
        lp = logp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
        out.extend((lp * mask[:, 1:]).sum(1).tolist())
    return out


def eval_blimp(model, tok, blimp_dir, device, max_per_uid=200):
    per = {}
    for f in sorted(glob.glob(os.path.join(blimp_dir, "*.jsonl"))):
        rows = [json.loads(l) for l in open(f)][:max_per_uid]
        good = score_sentences(model, tok, [r["sentence_good"] for r in rows], device)
        bad = score_sentences(model, tok, [r["sentence_bad"] for r in rows], device)
        per[os.path.basename(f)[:-6]] = sum(g > b for g, b in zip(good, bad)) / len(rows)
    if not per:
        return {}
    return {"blimp": sum(per.values()) / len(per), "blimp_per_uid": per}


def eval_morph(model, tok, device):
    ir_correct, ir_margin, ir_total = 0, 0.0, 0
    rg_correct, rg_total = 0, 0
    for t in TEMPLATES:
        good = score_sentences(model, tok, [t.format(c) for _, c, _ in IRREGULAR], device)
        over = score_sentences(model, tok, [t.format(o) for _, _, o in IRREGULAR], device)
        for g, o in zip(good, over):
            ir_correct += g > o
            ir_margin += g - o
            ir_total += 1
        good = score_sentences(model, tok, [t.format(p) for _, p in REGULAR], device)
        stem = score_sentences(model, tok, [t.format(b) for b, _ in REGULAR], device)
        for g, s in zip(good, stem):
            rg_correct += g > s
            rg_total += 1
    return {"irreg_acc": ir_correct / ir_total, "irreg_margin": ir_margin / ir_total,
            "reg_acc": rg_correct / rg_total}

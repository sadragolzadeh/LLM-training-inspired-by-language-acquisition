"""Train one condition (fixed or growing) and log everything to metrics.jsonl.

Example:
  python train.py --cond E_both --data_dir data/babylm100M --blimp_dir blimp \
      --total_tokens 100000000 --batch_size 64 --seq_len 256 --seed 0 --out_dir runs
"""
import argparse
import json
import math
import os
import time

import torch

from configs import CONDITIONS, build_plan, flops_per_token, nonembedding_params
from data import TokenData, load_tokenizer
from evals import eval_blimp, eval_morph
from growth import attach_optimizer_state, grow, preservation_gap
from model import GPT


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=CONDITIONS)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--blimp_dir", default=None)
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total_tokens", type=int, default=100_000_000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--regrow_warmup", type=int, default=200, help="LR re-warmup steps after a growth event")
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--fracs", type=str, default="0,0.25,0.5", help="stage start fractions of training")
    ap.add_argument("--depth_place", default="spread", choices=["spread", "stack"])
    ap.add_argument("--width_init", default="net2net", choices=["net2net", "zero"])
    ap.add_argument("--dup_noise", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_iters", type=int, default=20)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--device", default="auto")
    return ap.parse_args()


def make_optimizer(model, args):
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=args.lr, betas=(0.9, 0.95), eps=1e-8)


def lr_at(step, total, last_growth, a):
    if step < a.warmup:
        lr = a.lr * (step + 1) / a.warmup
    else:
        prog = (step - a.warmup) / max(1, total - a.warmup)
        lr = a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))
    if last_growth > 0:  # re-warm after growth: absorbs Adam-moment mismatch of new parameters
        lr *= 0.1 + 0.9 * min(1.0, (step - last_growth) / max(1, a.regrow_warmup))
    return lr


def main():
    a = get_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    torch.manual_seed(a.seed)

    data = TokenData(a.data_dir)
    tok = load_tokenizer(a.data_dir)
    vocab = data.meta["vocab_size"]
    tokens_per_step = a.batch_size * a.grad_accum * a.seq_len
    total_steps = a.total_tokens // tokens_per_step
    fracs = tuple(float(x) for x in a.fracs.split(","))
    plan = build_plan(a.cond, total_steps, tokens_per_step, a.seq_len, vocab, fracs,
                      a.depth_place, a.width_init)
    total_steps = plan.total_steps  # differs from the nominal value for G_flops_matched

    run_dir = os.path.join(a.out_dir, a.cond, f"seed{a.seed}")
    os.makedirs(run_dir, exist_ok=True)
    logf = open(os.path.join(run_dir, "metrics.jsonl"), "w")

    def log(rec):
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

    events = {s: c for s, c in plan.stages}
    model = GPT(plan.stages[0][1]).to(device)
    opt = make_optimizer(model, a)
    cum_flops, last_growth = 0.0, 0
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" \
        else torch.autocast(device_type="cpu", enabled=False)
    val_batches = [data.get_batch("val", i, a.batch_size, a.seq_len, a.seed, device) for i in range(a.eval_iters)]

    @torch.no_grad()
    def val_loss(m):
        m.eval()
        tot = 0.0
        for x, y in val_batches:
            with amp:
                _, l = m(x, y)
            tot += float(l)
        m.train()
        return tot / len(val_batches)

    def do_eval(step):
        rec = {"type": "eval", "step": step, "tokens": step * tokens_per_step, "flops": cum_flops,
               "val_loss": val_loss(model), "n_params": nonembedding_params(model.cfg)}
        if a.blimp_dir:
            rec.update(eval_blimp(model, tok, a.blimp_dir, device))
        rec.update(eval_morph(model, tok, device))
        model.train()
        log(rec)
        print(f"[{a.cond}] step {step}/{total_steps} val {rec['val_loss']:.3f} "
              f"blimp {rec.get('blimp', float('nan')):.3f} irreg {rec['irreg_acc']:.2f}", flush=True)

    log({"type": "plan", "cond": a.cond, "total_steps": total_steps, "tokens_per_step": tokens_per_step,
         "stages": [(s, c.to_dict()) for s, c in plan.stages], "depth_init": plan.depth_init,
         "width_init": plan.width_init, "depth_place": plan.depth_place, "notes": plan.notes,
         "args": vars(a)})
    model.train()
    run_loss, run_n, t0 = 0.0, 0, time.time()

    for step in range(total_steps):
        # ---------------- growth event ----------------
        if step > 0 and step in events:
            new_cfg = events[step]
            probe_x, _ = val_batches[0]
            before = val_loss(model)
            old_cfg = model.cfg
            new_model, gs = grow(model, new_cfg, opt, plan.depth_place, plan.depth_init,
                                 plan.width_init, seed=a.seed * 1000 + step, dup_noise=a.dup_noise)
            gap = preservation_gap(model, new_model, probe_x)
            new_opt = make_optimizer(new_model, a)
            attach_optimizer_state(new_opt, new_model, gs)
            model, opt = new_model, new_opt
            model.train()
            last_growth = step
            after = val_loss(model)
            log({"type": "growth", "step": step, "tokens": step * tokens_per_step, "flops": cum_flops,
                 "from": old_cfg.to_dict(), "to": new_cfg.to_dict(),
                 "val_before": before, "val_after": after, "max_logit_gap": gap})
            print(f"[{a.cond}] GROW @ {step}: {new_cfg.n_layers}L d{new_cfg.d_model} h{new_cfg.n_heads} "
                  f"ff{new_cfg.d_ff} | val {before:.3f}->{after:.3f} gap {gap:.2e}", flush=True)
            do_eval(step)

        # ---------------- train step ----------------
        lr = lr_at(step, total_steps, last_growth, a)
        for gparam in opt.param_groups:
            gparam["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for micro in range(a.grad_accum):
            x, y = data.get_batch("train", step * a.grad_accum + micro, a.batch_size, a.seq_len, a.seed, device)
            with amp:
                _, loss = model(x, y)
            (loss / a.grad_accum).backward()
            run_loss += float(loss)
            run_n += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        cum_flops += tokens_per_step * flops_per_token(model.cfg, a.seq_len)

        if (step + 1) % a.log_every == 0:
            log({"type": "train", "step": step + 1, "tokens": (step + 1) * tokens_per_step,
                 "flops": cum_flops, "loss": run_loss / run_n, "lr": lr,
                 "n_params": nonembedding_params(model.cfg), "sec": time.time() - t0})
            run_loss, run_n = 0.0, 0
        if (step + 1) % a.eval_every == 0 and step + 1 < total_steps:
            do_eval(step + 1)

    do_eval(total_steps)
    torch.save({"cfg": model.cfg.to_dict(), "state_dict": model.state_dict()},
               os.path.join(run_dir, "final.pt"))


if __name__ == "__main__":
    main()

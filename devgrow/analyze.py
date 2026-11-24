"""Aggregate runs/<cond>/seed*/metrics.jsonl into plots and a summary table."""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(runs):
    out = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(runs, "*", "seed*", "metrics.jsonl"))):
        cond, seed = f.split(os.sep)[-3], f.split(os.sep)[-2]
        out[cond][seed] = [json.loads(l) for l in open(f)]
    return out


def series(recs, typ, key):
    r = [x for x in recs if x["type"] == typ and key in x]
    return np.array([x["tokens"] for x in r]), np.array([x["flops"] for x in r]), np.array([x[key] for x in r])


def plot(runs_data, key, xaxis, path, ylabel):
    plt.figure(figsize=(7, 4.5))
    for cond, seeds in runs_data.items():
        xs, ys = [], []
        for recs in seeds.values():
            t, f, y = series(recs, "eval", key)
            if len(y):
                xs.append(t if xaxis == "tokens" else f); ys.append(y)
        if not ys:
            continue
        n = min(len(y) for y in ys)
        x = np.mean([a[:n] for a in xs], 0); m = np.mean([a[:n] for a in ys], 0)
        s = np.std([a[:n] for a in ys], 0)
        plt.plot(x, m, label=cond); plt.fill_between(x, m - s, m + s, alpha=0.15)
    plt.xlabel(xaxis); plt.ylabel(ylabel); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = load(a.runs)
    for key, lab in [("val_loss", "val loss"), ("blimp", "BLiMP acc"),
                     ("irreg_acc", "irregular past: correct > over-regularised"),
                     ("irreg_margin", "irregular margin (logp)")]:
        for xa in ("tokens", "flops"):
            plot(d, key, xa, os.path.join(a.out, f"{key}_vs_{xa}.png"), lab)
    print(f"{'cond':18s} {'val_loss':>14s} {'blimp':>14s} {'irreg_acc':>14s} {'PFLOPs':>8s}")
    for cond, seeds in d.items():
        fin = [[x for x in r if x["type"] == "eval"][-1] for r in seeds.values()]
        f = lambda k: (np.mean([x.get(k, np.nan) for x in fin]), np.std([x.get(k, np.nan) for x in fin]))
        print(f"{cond:18s} {f('val_loss')[0]:7.3f}±{f('val_loss')[1]:.3f} {f('blimp')[0]:7.3f}±{f('blimp')[1]:.3f} "
              f"{f('irreg_acc')[0]:7.3f}±{f('irreg_acc')[1]:.3f} {np.mean([x['flops'] for x in fin])/1e15:8.2f}")
    print("\ngrowth events (val loss before -> after, max logit gap):")
    for cond, seeds in d.items():
        for seed, recs in seeds.items():
            for g in [x for x in recs if x["type"] == "growth"]:
                print(f"  {cond}/{seed} step {g['step']}: {g['val_before']:.3f}->{g['val_after']:.3f} gap {g['max_logit_gap']:.2e}")


if __name__ == "__main__":
    main()

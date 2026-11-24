# devgrow: growing vs fixed-size LMs on BabyLM-scale data

## Setup
    pip install torch numpy tokenizers matplotlib pytest
    pytest tests/ -q            # FIRST: checks function preservation + optimizer-state transfer
    python prepare_data.py --input_dir babylm/train_100M --val_dir babylm/dev --out_dir data/babylm100M
    # BLiMP: put the *.jsonl files (one per phenomenon) in ./blimp
    ./run_all.sh                # 7 conditions x 3 seeds, then analyze.py

Single run: `python train.py --cond E_both --data_dir data/babylm100M --blimp_dir blimp --seed 0`

## Conditions (configs.py)
A_small, B_large (fixed) | C_depth, D_width, E_both (growth, function-preserving) |
F_nopreserve (growth, NOT preserving: control) | G_flops_matched (large model, same FLOPs as E).
Size ladder: 2L/d256 -> 4L/d384 -> 8L/d512 (~25.7M non-embedding params); edit SMALL/MID/FINAL to rescale.
Growth happens at 25% and 50% of training (--fracs). All conditions see the same token windows.

## Logged (runs/<cond>/seed<k>/metrics.jsonl)
train loss, val loss, BLiMP, past-tense probe (irreg_acc = U-shape diagnostic), cumulative FLOPs,
and per growth event: val loss before/after and max logit gap (≈0 means preserved).

## Known caveats
* One LR for all sizes (no muP); small stages may prefer a higher LR.
* d_model growth shifts RMSNorm eps slightly (negligible at trained activation scales).
* Past-tense probe uses summed log-prob, so tokenisation length biases it; read margin with accuracy.
* Net2Net duplicates need tiny noise (--dup_noise) to break symmetry; set 0 for exact preservation.

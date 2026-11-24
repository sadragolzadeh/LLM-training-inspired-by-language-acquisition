#!/usr/bin/env bash
# Run every condition for several seeds. Edit DATA/BLIMP/TOKENS to taste.
set -e
DATA=${DATA:-data/babylm100M}
BLIMP=${BLIMP:-blimp}
TOKENS=${TOKENS:-100000000}
for seed in 0 1 2; do
  for cond in A_small B_large C_depth D_width E_both F_nopreserve G_flops_matched; do
    python train.py --cond $cond --data_dir $DATA --blimp_dir $BLIMP \
      --total_tokens $TOKENS --seed $seed --out_dir runs
  done
done
python analyze.py --runs runs --out figures

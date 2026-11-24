"""Tokenise a BabyLM-style corpus into uint16 .bin shards.

Usage:
  python prepare_data.py --input_dir babylm/train_100M --out_dir data/babylm100M \
         --vocab_size 8192 --tokenizer bpe [--val_dir babylm/dev]

Every non-empty line is treated as one utterance/document and followed by <eos>.
If --val_dir is not given, the last --val_frac of the lines of every file is held out.
--tokenizer byte needs no extra packages (vocab 257); bpe needs `pip install tokenizers`.
"""
import argparse
import glob
import json
import os

import numpy as np


def read_lines(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [l.strip() for l in f if l.strip()]


def split_files(files, val_frac):
    train, val = [], []
    for f in files:
        lines = read_lines(f)
        k = max(1, int(len(lines) * val_frac))
        train.extend(lines[:-k])
        val.extend(lines[-k:])
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--val_dir", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--glob", default="*")
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--tokenizer", choices=["bpe", "byte"], default="bpe")
    ap.add_argument("--val_frac", type=float, default=0.01)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(f for f in glob.glob(os.path.join(args.input_dir, args.glob)) if os.path.isfile(f))
    assert files, f"no files in {args.input_dir}"
    if args.val_dir:
        train_lines = [l for f in files for l in read_lines(f)]
        vfiles = sorted(f for f in glob.glob(os.path.join(args.val_dir, args.glob)) if os.path.isfile(f))
        val_lines = [l for f in vfiles for l in read_lines(f)]
    else:
        train_lines, val_lines = split_files(files, args.val_frac)
    print(f"{len(train_lines):,} train lines, {len(val_lines):,} val lines")

    if args.tokenizer == "byte":
        vocab, eos = 257, 256

        def encode(lines):
            return [list(l.encode("utf-8")) for l in lines]
    else:
        from tokenizers import ByteLevelBPETokenizer
        tok = ByteLevelBPETokenizer()
        tok.train_from_iterator(train_lines, vocab_size=args.vocab_size, min_frequency=2,
                                special_tokens=["<eos>"])
        tok.save(os.path.join(args.out_dir, "tokenizer.json"))
        vocab, eos = tok.get_vocab_size(), tok.token_to_id("<eos>")

        def encode(lines):
            return [e.ids for e in tok.encode_batch(lines)]

    def write(lines, path):
        n = 0
        with open(path, "wb") as f:
            for i in range(0, len(lines), 20000):
                ids = []
                for seq in encode(lines[i:i + 20000]):
                    ids.extend(seq)
                    ids.append(eos)
                arr = np.asarray(ids, dtype=np.uint16)
                f.write(arr.tobytes())
                n += len(arr)
        print(f"{path}: {n:,} tokens")
        return n

    n_tr = write(train_lines, os.path.join(args.out_dir, "train.bin"))
    n_va = write(val_lines, os.path.join(args.out_dir, "val.bin"))
    json.dump({"tokenizer": args.tokenizer, "vocab_size": vocab, "eos_id": eos,
               "train_tokens": n_tr, "val_tokens": n_va},
              open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()

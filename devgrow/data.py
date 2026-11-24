"""Deterministic batch stream + tokenizer loading.

get_batch(split, step, ...) is a pure function of (seed, split, step), so EVERY
condition sees exactly the same token windows at the same step -> clean comparison.
"""
import json
import os

import numpy as np
import torch


class ByteTokenizer:
    eos_id = 256
    vocab_size = 257

    def encode(self, s):
        return list(s.encode("utf-8"))


class BPETokenizer:
    def __init__(self, path, eos_id, vocab_size):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(path)
        self.eos_id, self.vocab_size = eos_id, vocab_size

    def encode(self, s):
        return self.tok.encode(s).ids


def load_meta(data_dir):
    return json.load(open(os.path.join(data_dir, "meta.json")))


def load_tokenizer(data_dir):
    meta = load_meta(data_dir)
    if meta["tokenizer"] == "byte":
        return ByteTokenizer()
    return BPETokenizer(os.path.join(data_dir, "tokenizer.json"), meta["eos_id"], meta["vocab_size"])


class TokenData:
    def __init__(self, data_dir):
        self.meta = load_meta(data_dir)
        self.train = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
        self.val = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

    def get_batch(self, split, step, batch_size, seq_len, seed, device="cpu"):
        arr = self.train if split == "train" else self.val
        rng = np.random.default_rng([seed, 0 if split == "train" else 1, step])
        ix = rng.integers(0, len(arr) - seq_len - 1, size=batch_size)
        x = np.stack([arr[i:i + seq_len] for i in ix]).astype(np.int64)
        y = np.stack([arr[i + 1:i + 1 + seq_len] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

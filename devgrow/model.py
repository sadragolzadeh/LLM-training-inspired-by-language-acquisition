"""Small GPT-style decoder built to be growable.

Design choices that make growth clean:
  * pre-RMSNorm residual blocks  -> a block with zeroed output projections is an exact identity
  * RoPE + FIXED head_dim        -> adding heads / widening d_model never touches positional code
  * SwiGLU MLP, no biases        -> every parameter is a plain matrix, easy to index/gather
  * tied input/output embeddings -> the LM head is not a separate parameter
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).type_as(x)


def _rope(x, cos, sin):
    # x: (B, H, T, hd); cos/sin: (T, hd/2)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos.to(x.dtype)[None, None]
    sin = sin.to(x.dtype)[None, None]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.h, self.hd = c.n_heads, c.head_dim
        self.q = nn.Linear(c.d_model, c.d_attn, bias=False)
        self.k = nn.Linear(c.d_model, c.d_attn, bias=False)
        self.v = nn.Linear(c.d_model, c.d_attn, bias=False)
        self.o = nn.Linear(c.d_attn, c.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.hd).transpose(1, 2)
        q, k = _rope(q, cos[:T], sin[:T]), _rope(k, cos[:T], sin[:T])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, self.h * self.hd))


class MLP(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.up = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.down = nn.Linear(c.d_ff, c.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(c.d_model)
        self.attn = Attention(c)
        self.mlp_norm = RMSNorm(c.d_model)
        self.mlp = MLP(c)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.cfg = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.norm_f = RMSNorm(c.d_model)

        inv = 1.0 / (c.rope_base ** (torch.arange(0, c.head_dim, 2).float() / c.head_dim))
        ang = torch.outer(torch.arange(c.max_seq_len).float(), inv)
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)

        self.apply(self._init)
        for b in self.blocks:  # GPT-2 style residual scaling
            for w in (b.attn.o.weight, b.mlp.down.weight):
                nn.init.normal_(w, 0.0, c.init_std / math.sqrt(2 * c.n_layers))

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, self.cfg.init_std)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        x = self.norm_f(x)
        logits = F.linear(x, self.tok_emb.weight)  # tied head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

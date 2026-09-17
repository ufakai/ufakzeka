"""ufakzeka decoder-only transformer.

Design (all choices are the 2025-2026 consensus for models under 1B parameters):
- pre-RMSNorm, SwiGLU MLP, rotary position embeddings, grouped-query attention
- QK-Norm (RMSNorm on q and k per head) for stability at high learning rates
- tied input and output embeddings (embedding matrix is a large share of a 150M model)
- zero-initialised output projections of attention and MLP (residual branches start as identity)
- logit soft-capping at 30 as a cheap guard, z-loss handled in the training loop
- optional value embeddings and U-net style skip connections are left out of v1 to keep the
  exported checkpoint loadable by standard Llama code.

The parameter layout matches LlamaForCausalLM so the trained weights export to
Hugging Face transformers with a plain key rename."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention


@dataclass
class ModelConfig:
    vocab_size: int = 40960
    n_layer: int = 24
    d_model: int = 768
    n_head: int = 12
    n_kv_head: int = 4
    d_ff: int = 2048
    seq_len: int = 2048
    rope_theta: float = 100_000.0
    norm_eps: float = 1e-5
    softcap: float = 30.0
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).type_as(x)


def rope_cache(seq_len: int, head_dim: int, theta: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    rot = torch.cat([-x2, x1], dim=-1)
    return (x * cos + rot * sin).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        hd = cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_head * hd, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * hd, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * hd, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * hd, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(hd, cfg.norm_eps)
        self.k_norm = RMSNorm(hd, cfg.norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, block_mask: BlockMask | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        hd, H, KV = self.cfg.head_dim, self.cfg.n_head, self.cfg.n_kv_head
        q = self.q_proj(x).view(B, T, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, KV, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, KV, hd).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if block_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            # flex attention only touches the blocks the mask keeps, so a document mask costs no more memory
            # than plain causal attention (a dense attn_mask would materialise B x H x T x T floats per layer)
            y = flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, H * hd))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, block_mask=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, block_mask)
        return x + self.mlp(self.post_attention_layernorm(x))


def document_block_mask(idx: torch.Tensor, eot_id: int) -> BlockMask:
    """Causal mask where packed documents cannot see each other. Documents end with eot_id and the eot
    token belongs to the document it closes. Call this outside the compiled forward, once per batch."""
    B, T = idx.shape
    is_eot = (idx == eot_id).to(torch.int32)
    doc = torch.cumsum(is_eot, dim=1) - is_eot

    def mask_mod(b, h, q_idx, kv_idx):
        return (doc[b, q_idx] == doc[b, kv_idx]) & (q_idx >= kv_idx)

    return create_block_mask(mask_mod, B, None, T, T, device=str(idx.device), _compile=idx.is_cuda)


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        # NEFTune (arXiv 2310.05914): uniform noise on the input embeddings during finetuning only.
        # Set by the SFT loop; 0 disables it, which is what pretraining and inference want.
        self.neftune_alpha = 0.0
        self.embed_dropout = 0.0
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)
        self.apply(self._init)
        for blk in self.layers:  # zero init residual outputs
            nn.init.zeros_(blk.self_attn.o_proj.weight)
            nn.init.zeros_(blk.mlp.down_proj.weight)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _rope(self, T: int, device):
        if self.rope_cos is None or self.rope_cos.shape[0] < T or self.rope_cos.device != device:
            cos, sin = rope_cache(max(T, self.cfg.seq_len), self.cfg.head_dim, self.cfg.rope_theta, device)
            self.rope_cos, self.rope_sin = cos, sin
        return self.rope_cos[:T], self.rope_sin[:T]

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, z_loss: float = 0.0,
                block_mask: BlockMask | None = None, token_weights: torch.Tensor | None = None):
        """block_mask: optional flex attention mask, see document_block_mask. Build it outside torch.compile."""
        B, T = idx.shape
        cos, sin = self._rope(T, idx.device)
        x = self.embed_tokens(idx)
        if self.training and self.neftune_alpha > 0:
            mag = self.neftune_alpha / math.sqrt(T * self.cfg.d_model)
            x = x + torch.empty_like(x).uniform_(-mag, mag)
        if self.training and self.embed_dropout > 0:
            x = F.dropout(x, self.embed_dropout)  # LFM2 (arXiv 2511.23404) drops 10 percent of input embeddings in SFT
        for blk in self.layers:
            x = blk(x, cos, sin, block_mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        if self.cfg.softcap > 0:
            logits = self.cfg.softcap * torch.tanh(logits / self.cfg.softcap)
        if targets is None:
            return logits, None
        logits = logits.float()
        if token_weights is not None:
            # prompt tokens at a small weight instead of masked: best in 81 percent of 525 runs (WIT, arXiv 2507.07817)
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1, reduction="none")
            w = token_weights.reshape(-1).to(ce.dtype) * (targets.reshape(-1) != -1)
            loss = (ce * w).sum() / w.sum().clamp(min=1.0)
        else:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        if z_loss > 0:
            lse = torch.logsumexp(logits, dim=-1)
            loss = loss + z_loss * (lse ** 2).mean()
        return logits, loss

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed_tokens.weight.numel()
        return n

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int, temperature: float = 0.0, top_k: int = 50) -> torch.Tensor:
        for _ in range(max_new):
            logits, _ = self(idx[:, -self.cfg.seq_len:])
            logits = logits[:, -1]
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], 1)
        return idx

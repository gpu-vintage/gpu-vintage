"""A minimal decoder-only transformer for the Qwen3 (and Qwen2) architectures.

The same code runs on every GPU of the panel, so that the hardware is the only
thing that changes between measurements. Weights are read from the Hugging Face
safetensors of a checkpoint; the query/key/value projections and the gate/up
projections are fused into one matrix each, as production inference engines do.

Numerics follow the Hugging Face implementation (RMSNorm in fp32, rotary
embedding with the half-split convention, attention scale 1/sqrt(head_dim)),
which tests/test_model.py checks logit by logit on a small random model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    arch: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    qkv_bias: bool
    o_bias: bool
    qk_norm: bool

    @classmethod
    def from_hf(cls, d: dict) -> "Config":
        arch = d.get("model_type")
        if arch not in ("qwen3", "qwen2"):
            raise ValueError(f"unsupported model_type: {arch!r}")
        if d.get("rope_scaling"):
            raise ValueError("rope_scaling is not supported")
        head_dim = d.get("head_dim") or d["hidden_size"] // d["num_attention_heads"]
        if arch == "qwen3":
            bias = bool(d.get("attention_bias", False))
            qkv_bias, o_bias, qk_norm = bias, bias, True
        else:  # qwen2: biased q, k, v; unbiased o
            qkv_bias, o_bias, qk_norm = True, False, False
        return cls(
            arch=arch,
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_layers=d["num_hidden_layers"],
            num_heads=d["num_attention_heads"],
            num_kv_heads=d["num_key_value_heads"],
            head_dim=head_dim,
            rms_norm_eps=d["rms_norm_eps"],
            rope_theta=float(d.get("rope_theta", 10000.0)),
            tie_word_embeddings=bool(d.get("tie_word_embeddings", False)),
            qkv_bias=qkv_bias,
            o_bias=o_bias,
            qk_norm=qk_norm,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def rms_norm(x: torch.Tensor, weight2: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm in two kernels instead of eight.

    The layer norm of [x, -x] has mean zero and variance mean(x^2), so its first
    half is x / sqrt(mean(x^2) + eps), RMSNorm. layer_norm is one fused kernel
    that accumulates in fp32, as the Hugging Face RMSNorm does. weight2 is the
    norm's weight repeated twice (see norm_weight2)."""
    d = x.shape[-1]
    y = F.layer_norm(torch.cat((x, -x), dim=-1), (2 * d,), weight2, None, eps)
    return y[..., :d]


def norm_weight2(w: torch.Tensor) -> torch.Tensor:
    return torch.cat((w, w)).contiguous()


def rope(x: torch.Tensor, cos: torch.Tensor, sin_signed: torch.Tensor) -> torch.Tensor:
    """Rotary embedding, Hugging Face's half-split convention, in three kernels:
    rotate_half(x) = cat(-x2, x1) = roll(x, d/2) * cat(-1, +1), and the sign is
    folded into sin_signed."""
    return torch.addcmul(x * cos, torch.roll(x, x.shape[-1] // 2, dims=-1), sin_signed)


class Layer:
    """One decoder layer's weights (fused projections)."""

    __slots__ = ("ln1", "wqkv", "bqkv", "q_norm", "k_norm", "wo", "bo", "ln2", "w13", "w2")
    # ln1, ln2, q_norm and k_norm hold the norm weights repeated twice (rms_norm)

    def __init__(self) -> None:
        for s in self.__slots__:
            setattr(self, s, None)


class Model:
    """Weights plus the two forward passes the benchmark needs."""

    def __init__(self, cfg: Config, device: torch.device, dtype: torch.dtype) -> None:
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.embed: torch.Tensor | None = None
        self.lm_head: torch.Tensor | None = None
        self.norm: torch.Tensor | None = None
        self.layers = [Layer() for _ in range(cfg.num_layers)]
        self.q_size = cfg.num_heads * cfg.head_dim
        self.kv_size = cfg.num_kv_heads * cfg.head_dim
        self.group = cfg.num_heads // cfg.num_kv_heads
        self.scale = 1.0 / math.sqrt(cfg.head_dim)
        self._rope_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, path: str | Path, device: str | torch.device = "cuda",
                        dtype: torch.dtype = torch.float16) -> "Model":
        """Load a checkpoint directory (config.json and *.safetensors)."""
        from safetensors import safe_open

        path = Path(path)
        cfg = Config.from_hf(json.loads((path / "config.json").read_text()))
        device = torch.device(device)
        m = cls(cfg, device, dtype)
        files = sorted(path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no .safetensors in {path}")
        pending: dict[int, dict[str, torch.Tensor]] = {i: {} for i in range(cfg.num_layers)}
        for f in files:
            with safe_open(str(f), framework="pt", device="cpu") as fh:
                for name in fh.keys():
                    t = fh.get_tensor(name).to(device=device, dtype=dtype)
                    m._place(name, t, pending)
        for i, parts in pending.items():
            if parts:
                raise ValueError(f"layer {i}: unused weights {sorted(parts)}")
        if m.lm_head is None:
            if not cfg.tie_word_embeddings:
                raise ValueError("checkpoint has no lm_head and does not tie embeddings")
            m.lm_head = m.embed
        m._check_complete()
        return m

    def _place(self, name: str, t: torch.Tensor, pending: dict[int, dict[str, torch.Tensor]]) -> None:
        if name == "model.embed_tokens.weight":
            self.embed = t
            return
        if name == "model.norm.weight":
            self.norm = norm_weight2(t)
            return
        if name == "lm_head.weight":
            self.lm_head = t
            return
        if not name.startswith("model.layers."):
            raise ValueError(f"unexpected weight {name}")
        rest = name[len("model.layers."):]
        idx_s, sub = rest.split(".", 1)
        i = int(idx_s)
        L = self.layers[i]
        simple = {
            "input_layernorm.weight": "ln1",
            "post_attention_layernorm.weight": "ln2",
            "self_attn.q_norm.weight": "q_norm",
            "self_attn.k_norm.weight": "k_norm",
            "self_attn.o_proj.weight": "wo",
            "self_attn.o_proj.bias": "bo",
            "mlp.down_proj.weight": "w2",
        }
        if sub in simple:
            if simple[sub] in ("ln1", "ln2", "q_norm", "k_norm"):
                t = norm_weight2(t)
            setattr(L, simple[sub], t)
            return
        parts = pending[i]
        parts[sub] = t
        qkv_w = ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight")
        qkv_b = ("self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias")
        gu = ("mlp.gate_proj.weight", "mlp.up_proj.weight")
        if all(k in parts for k in qkv_w):
            L.wqkv = torch.cat([parts.pop(k) for k in qkv_w], dim=0).contiguous()
        if all(k in parts for k in qkv_b):
            L.bqkv = torch.cat([parts.pop(k) for k in qkv_b], dim=0).contiguous()
        if all(k in parts for k in gu):
            L.w13 = torch.cat([parts.pop(k) for k in gu], dim=0).contiguous()

    def _check_complete(self) -> None:
        cfg = self.cfg
        for i, L in enumerate(self.layers):
            need = ["ln1", "wqkv", "wo", "ln2", "w13", "w2"]
            if cfg.qkv_bias:
                need.append("bqkv")
            if cfg.o_bias:
                need.append("bo")
            if cfg.qk_norm:
                need += ["q_norm", "k_norm"]
            missing = [n for n in need if getattr(L, n) is None]
            if missing:
                raise ValueError(f"layer {i}: missing {missing}")
        if self.embed is None or self.norm is None:
            raise ValueError("missing embeddings or final norm")

    def num_parameters(self) -> int:
        seen, n = set(), 0
        tensors = [self.embed, self.lm_head, self.norm]
        for L in self.layers:
            tensors += [getattr(L, s) for s in Layer.__slots__]
        doubled = {id(self.norm)} | {id(getattr(L, a)) for L in self.layers
                                     for a in ("ln1", "ln2", "q_norm", "k_norm")}
        for t in tensors:
            if t is not None and t.data_ptr() not in seen:
                seen.add(t.data_ptr())
                n += t.numel() // 2 if id(t) in doubled else t.numel()
        return n

    def weight_bytes(self) -> int:
        return self.num_parameters() * torch.finfo(self.dtype).bits // 8

    # ------------------------------------------------------------------ rope
    def rope_tables(self, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """cos and signed sin (see rope) of shape [max_len, head_dim], computed
        in fp32 as HF does, then cast."""
        hit = self._rope_cache.get(max_len)
        if hit is not None:
            return hit
        d = self.cfg.head_dim
        inv_freq = 1.0 / (self.cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.int64,
                                                                device=self.device).float() / d))
        pos = torch.arange(max_len, device=self.device, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        sign = torch.cat((-torch.ones(d // 2, device=self.device), torch.ones(d // 2, device=self.device)))
        tables = (emb.cos().to(self.dtype), (emb.sin() * sign).to(self.dtype))
        self._rope_cache[max_len] = tables
        return tables

    # ------------------------------------------------------------------ cache
    def new_cache(self, batch: int, max_len: int) -> torch.Tensor:
        """Static KV cache, [layers, 2, batch, kv_heads, max_len, head_dim]."""
        c = self.cfg
        return torch.zeros((c.num_layers, 2, batch, c.num_kv_heads, max_len, c.head_dim),
                           device=self.device, dtype=self.dtype)

    def cache_bytes(self, batch: int, max_len: int) -> int:
        c = self.cfg
        return (c.num_layers * 2 * batch * c.num_kv_heads * max_len * c.head_dim
                * torch.finfo(self.dtype).bits // 8)

    # ------------------------------------------------------------------ blocks
    def _qkv(self, L: Layer, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Projections, q/k norm and rotary embedding. h: [b, t, hidden]."""
        c = self.cfg
        b, t, _ = h.shape
        qkv = F.linear(h, L.wqkv, L.bqkv)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(b, t, c.num_heads, c.head_dim)
        k = k.view(b, t, c.num_kv_heads, c.head_dim)
        v = v.view(b, t, c.num_kv_heads, c.head_dim)
        if c.qk_norm:
            q = rms_norm(q, L.q_norm, c.rms_norm_eps)
            k = rms_norm(k, L.k_norm, c.rms_norm_eps)
        cos = cos[None, :, None]  # [1, t, 1, d]
        sin = sin[None, :, None]
        q = rope(q, cos, sin)  # [b, t, heads, d], contiguous
        k = rope(k, cos, sin)
        return q, k, v

    def _mlp(self, L: Layer, x: torch.Tensor) -> torch.Tensor:
        """x is updated in place: the residual add rides on the down-projection GEMM."""
        h = rms_norm(x, L.ln2, self.cfg.rms_norm_eps)
        g, u = F.linear(h, L.w13).chunk(2, dim=-1)
        a = F.silu(g) * u
        x.view(-1, x.shape[-1]).addmm_(a.view(-1, a.shape[-1]), L.w2.t())
        return x

    def _out(self, L: Layer, x: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        """x += o @ wo^T (+ bo), in place, the residual add riding on the GEMM."""
        x2 = x.view(-1, x.shape[-1])
        x2.addmm_(o.reshape(-1, o.shape[-1]), L.wo.t())
        if L.bo is not None:
            x2.add_(L.bo)
        return x

    def logits(self, x_last: torch.Tensor) -> torch.Tensor:
        """x_last: [b, hidden] -> logits [b, vocab]."""
        return F.linear(rms_norm(x_last, self.norm, self.cfg.rms_norm_eps), self.lm_head)

    # ------------------------------------------------------------------ passes
    @torch.inference_mode()
    def prefill(self, tokens: torch.Tensor, cache: torch.Tensor, row0: int = 0) -> torch.Tensor:
        """Process whole prompts from position 0; write their keys and values into
        rows row0:row0+b of the cache. tokens: [b, t]. Returns last-position logits."""
        c = self.cfg
        b, t = tokens.shape
        cos_t, sin_t = self.rope_tables(cache.shape[4])
        cos, sin = cos_t[:t], sin_t[:t]
        x = F.embedding(tokens, self.embed)
        for i, L in enumerate(self.layers):
            h = rms_norm(x, L.ln1, c.rms_norm_eps)
            q, k, v = self._qkv(L, h, cos, sin)  # [b, t, heads, d]
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            cache[i, 0, row0:row0 + b, :, :t] = k
            cache[i, 1, row0:row0 + b, :, :t] = v
            if self.group > 1:
                k = k.repeat_interleave(self.group, dim=1)
                v = v.repeat_interleave(self.group, dim=1)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
            x = self._out(L, x, o.transpose(1, 2).reshape(b, t, self.q_size))
            x = self._mlp(L, x)
        return self.logits(x[:, -1])

    @torch.inference_mode()
    def decode_step(self, tok: torch.Tensor, pos: torch.Tensor, cache: torch.Tensor,
                    span: int) -> torch.Tensor:
        """One decoding step for every row of the cache.

        tok: [B] token ids; pos: one-element int64 tensor, the position being
        written; span: how many cache positions attention reads (>= pos + 1).
        Everything that depends on the position goes through tensors, so the
        step can be captured once in a CUDA graph and replayed for any pos below
        span. Returns logits [B, vocab].
        """
        c = self.cfg
        B = tok.shape[0]
        cos_t, sin_t = self.rope_tables(cache.shape[4])
        cos = cos_t.index_select(0, pos)  # [1, d]
        sin = sin_t.index_select(0, pos)
        keep = (torch.arange(span, device=tok.device) <= pos).view(1, 1, 1, span)
        x = F.embedding(tok[:, None], self.embed)  # [B, 1, hidden]
        for i, L in enumerate(self.layers):
            h = rms_norm(x, L.ln1, c.rms_norm_eps)
            q, k, v = self._qkv(L, h, cos, sin)  # [B, 1, heads, d]
            cache[i, 0].index_copy_(2, pos, k.transpose(1, 2))
            cache[i, 1].index_copy_(2, pos, v.transpose(1, 2))
            K = cache[i, 0, :, :, :span]  # [B, Hkv, span, d], a view
            V = cache[i, 1, :, :, :span]
            # the query heads of one KV group, taken as that group's query rows:
            # one fused attention call instead of a repeat of the cache
            qg = q.view(B, c.num_kv_heads, self.group, c.head_dim)
            o = F.scaled_dot_product_attention(qg, K, V, attn_mask=keep, scale=self.scale)
            x = self._out(L, x, o.reshape(B, 1, self.q_size))
            x = self._mlp(L, x)
        return self.logits(x[:, 0])

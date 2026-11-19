"""Dense pre-norm decoder-only transformer with the CompleteP parameterization (alpha = 1).

Architecture: RMSNorm, multi-head attention with rotary positions and normalized queries and keys, SwiGLU MLP,
untied embedding and unembedding, no biases. With m_N = width / base width and m_L = depth / base depth
(base: the 2.2M size):

  forward    residual branches are multiplied by 1/m_L, logits by 1/m_N, attention scores by 1/head width;
  init       embedding and unembedding std 0.02, hidden matrices std 0.02 * m_N^-1/2, norm gains 1;
  AdamW      hidden matrices: learning rate / m_N, weight decay * m_N; epsilon scaled by 1/m_N (and 1/m_L in blocks).

A token stream packs documents as ``document + separator``. The separator is read as the first token of the next
document: attention never crosses a separator and positions restart at it, so the prediction at a token depends
only on the prefix of its own document.

Attention backends compute the same function. ``"varlen"`` (CUDA, used for training) packs the documents of a
batch for variable-length flash attention; ``"sdpa"`` builds a dense mask and runs anywhere.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from thinning.sizes import BASE, HEAD_DIM, Size

SIGMA_BASE = 0.02
EPS_BASE = 1e-16
ROPE_THETA = 10000.0


def segment_ids(idx: torch.Tensor, sep_id: int) -> torch.Tensor:
    """(B, T) index of the document each token belongs to within its row; it increments at a separator."""
    return torch.cumsum((idx == sep_id).to(torch.long), dim=1)


def positions_in_segment(idx: torch.Tensor, sep_id: int) -> torch.Tensor:
    """(B, T) position of each token within its document (0 at the separator)."""
    B, T = idx.shape
    ar = torch.arange(T, device=idx.device).expand(B, T)
    seg_start = torch.cummax(torch.where(idx == sep_id, ar, torch.zeros_like(ar)), dim=1).values
    return ar - seg_start


def dense_mask(idx: torch.Tensor, sep_id: int) -> torch.Tensor:
    """(B, 1, T, T) boolean mask, True where attention is allowed: causal and within one document."""
    seg = segment_ids(idx, sep_id)
    T = idx.shape[1]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=idx.device))
    return (causal & (seg[:, :, None] == seg[:, None, :])).unsqueeze(1)


def doc_bounds(idx: torch.Tensor, sep_id: int) -> tuple[torch.Tensor, int]:
    """Cumulative document boundaries over the flattened batch (int32) and the longest document, as
    variable-length attention expects. Every row start is a boundary, so attention never crosses rows."""
    B, T = idx.shape
    is_start = (idx.reshape(-1) == sep_id).clone()
    is_start[0::T] = True
    starts = torch.nonzero(is_start, as_tuple=False).flatten()
    total = torch.full((1,), B * T, dtype=torch.int64, device=idx.device)
    cu = torch.cat([starts, total]).to(torch.int32)
    return cu, int((cu[1:] - cu[:-1]).max())


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


def rope_cos_sin(positions: torch.Tensor, dtype):
    """(B, T) positions -> cos and sin of the rotary angles, each of shape (B, 1, T, HEAD_DIM / 2)."""
    inv_freq = 1.0 / ROPE_THETA ** (
        torch.arange(0, HEAD_DIM, 2, device=positions.device, dtype=torch.float32) / HEAD_DIM)
    freqs = positions[..., None].to(torch.float32) * inv_freq
    return freqs.cos().to(dtype)[:, None], freqs.sin().to(dtype)[:, None]


def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class Attention(nn.Module):
    def __init__(self, size: Size):
        super().__init__()
        d = size.d_model
        self.n_heads = size.n_heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)

    def forward(self, x, cos, sin, mask_info, impl):
        B, T, d = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, HEAD_DIM).transpose(1, 2)
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)
        scale = 1.0 / HEAD_DIM
        if impl == "varlen":
            z = self._varlen(q, k, v, mask_info, scale).view(B, T, self.n_heads, HEAD_DIM)
        else:
            z = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_info, scale=scale).transpose(1, 2)
        return self.o_proj(z.reshape(B, T, d))

    def _varlen(self, q, k, v, bounds, scale):
        """Causal attention over the packed documents of the batch. Queries and keys leave the norm and the
        rotary step in fp32 under autocast; the kernel needs one half-precision dtype."""
        from torch.nn.attention.varlen import varlen_attn

        B, H, T, hd = q.shape
        cu, max_len = bounds
        dt = v.dtype if v.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        qf, kf, vf = (t.to(dt).transpose(1, 2).reshape(B * T, H, hd) for t in (q, k, v))
        return varlen_attn(qf, kf, vf, cu, cu, max_len, max_len, scale=scale, window_size=(-1, 0))


class SwiGLU(nn.Module):
    def __init__(self, size: Size):
        super().__init__()
        self.gate_proj = nn.Linear(size.d_model, size.d_ff, bias=False)
        self.up_proj = nn.Linear(size.d_model, size.d_ff, bias=False)
        self.down_proj = nn.Linear(size.d_ff, size.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, size: Size, resid_mult: float):
        super().__init__()
        self.attn_norm = RMSNorm(size.d_model)
        self.attn = Attention(size)
        self.mlp_norm = RMSNorm(size.d_model)
        self.mlp = SwiGLU(size)
        self.resid_mult = resid_mult

    def forward(self, x, cos, sin, mask_info, impl):
        x = x + self.resid_mult * self.attn(self.attn_norm(x), cos, sin, mask_info, impl)
        return x + self.resid_mult * self.mlp(self.mlp_norm(x))


class Decoder(nn.Module):
    def __init__(self, size: Size, vocab_size: int, sep_id: int, attn_impl: str = "sdpa"):
        super().__init__()
        if attn_impl not in ("varlen", "sdpa"):
            raise ValueError(f"attn_impl must be 'varlen' or 'sdpa', got {attn_impl!r}")
        self.size = size
        self.vocab_size = int(vocab_size)
        self.sep_id = int(sep_id)
        self.attn_impl = attn_impl
        self.width_mult = size.d_model / BASE.d_model
        self.depth_mult = size.n_layers / BASE.n_layers
        self.embed = nn.Embedding(self.vocab_size, size.d_model)
        self.layers = nn.ModuleList(Block(size, 1.0 / self.depth_mult) for _ in range(size.n_layers))
        self.final_norm = RMSNorm(size.d_model)
        self.lm_head = nn.Linear(size.d_model, self.vocab_size, bias=False)

    def hidden(self, idx):
        """Residual stream after the final norm, (B, T, width)."""
        x = self.embed(idx)
        cos, sin = rope_cos_sin(positions_in_segment(idx, self.sep_id), x.dtype)
        if self.attn_impl == "varlen":
            mask_info = doc_bounds(idx, self.sep_id)
        else:
            mask_info = dense_mask(idx, self.sep_id)
        for block in self.layers:
            x = block(x, cos, sin, mask_info, self.attn_impl)
        return self.final_norm(x)

    def unembed(self, h):
        return self.lm_head(h) / self.width_mult

    def forward(self, idx):
        return self.unembed(self.hidden(idx))


def _param_kind(name: str) -> str:
    if name == "embed.weight":
        return "embedding"
    if name == "lm_head.weight":
        return "lm_head"
    if name.endswith("norm.weight"):
        return "norm"
    return "hidden_matrix"


def init_weights(model: Decoder, seed: int) -> None:
    """Initialize from a dedicated generator: the weights depend on the shape, the vocabulary and the seed only."""
    gen = torch.Generator().manual_seed(seed)
    hidden_std = SIGMA_BASE * model.width_mult**-0.5
    with torch.no_grad():
        for name, p in model.named_parameters():
            kind = _param_kind(name)
            if kind == "norm":
                p.fill_(1.0)
            else:
                std = hidden_std if kind == "hidden_matrix" else SIGMA_BASE
                p.copy_(torch.empty(p.shape).normal_(0.0, std, generator=gen))


def build_optimizer(model: Decoder, base_lr: float, base_wd: float = 0.0):
    """AdamW with one parameter group per kind. Norm gains have no weight decay. Each group stores its peak
    learning rate as ``peak_lr``; schedules set ``lr = peak_lr * multiplier``."""
    m_n, m_l = model.width_mult, model.depth_mult
    groups = {k: [] for k in ("embedding", "hidden_matrix", "norm", "lm_head")}
    for name, p in model.named_parameters():
        groups[_param_kind(name)].append(p)
    spec = {
        "embedding": dict(lr=base_lr, weight_decay=base_wd, eps=EPS_BASE / m_n),
        "hidden_matrix": dict(lr=base_lr / m_n, weight_decay=base_wd * m_n, eps=EPS_BASE / (m_n * m_l)),
        "norm": dict(lr=base_lr, weight_decay=0.0, eps=EPS_BASE / (m_n * m_l)),
        "lm_head": dict(lr=base_lr, weight_decay=base_wd, eps=EPS_BASE / m_n),
    }
    param_groups = [{"params": groups[k], "kind": k, "peak_lr": spec[k]["lr"], **spec[k]} for k in groups]
    fused = next(model.parameters()).is_cuda
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.95), fused=fused)

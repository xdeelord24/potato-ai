"""
Mamba-2 / Structured State Space Duality (SSD) block.

Based on: "Transformers are SSMs: Generalized Models and Efficient Algorithms
Through Structured State Space Duality" (Dao & Gu, 2024)
Reference: https://github.com/tommyip/mamba2-minimal
"""

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from .config import PotatoConfig


class InferenceCache(NamedTuple):
    """Hidden state for constant-time per-token inference."""

    conv_state: torch.Tensor  # (batch, d_inner + 2*d_state, d_conv)
    ssm_state: torch.Tensor  # (batch, nheads, headdim, d_state)

    @staticmethod
    def alloc(batch_size: int, config: PotatoConfig, device: Optional[torch.device] = None) -> "InferenceCache":
        return InferenceCache(
            torch.zeros(
                batch_size,
                config.d_inner + 2 * config.d_state,
                config.d_conv,
                device=device,
            ),
            torch.zeros(
                batch_size,
                config.nheads,
                config.headdim,
                config.d_state,
                device=device,
            ),
        )


def _segsum(x: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """Stable segment sum for 1-semiseparable matrix (scalar SSM)."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def _ssd(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    initial_states: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Structured State Space Duality (SSD) - core of Mamba-2.

    Args:
        x: (batch, seqlen, n_heads, d_head)
        A: (batch, seqlen, n_heads)
        B: (batch, seqlen, n_heads, d_state)
        C: (batch, seqlen, n_heads, d_state)

    Returns:
        y: (batch, seqlen, n_heads, d_head)
        final_state: for next chunk
    """
    assert x.shape[1] % chunk_size == 0

    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size)
        for m in (x, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Intra-chunk (diagonal blocks)
    L = torch.exp(_segsum(A, device=device))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # 2. State for each intra-chunk
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # 3. Inter-chunk SSM recurrence
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(
        _segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device)
    )
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. State -> output (off-diagonal blocks)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state


def _silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation."""
    return x * F.sigmoid(x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with optional gating."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        if z is not None:
            x = x * _silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class GatedFFN(nn.Module):
    """SwiGLU-style gated feed-forward network."""

    def __init__(self, config: PotatoConfig):
        super().__init__()
        hidden = config.d_model * config.ffn_mult
        self.w1 = nn.Linear(config.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, hidden, bias=False)
        self.dropout = nn.Dropout(config.ffn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(_silu(self.w1(x)) * self.w3(x)))


class Mamba2Block(nn.Module):
    """Single Mamba-2 block: SSM mixer + gated FFN."""

    def __init__(self, config: PotatoConfig):
        super().__init__()
        self.config = config

        # Mamba-2 mixer
        d_in_proj = 2 * config.d_inner + 2 * config.d_state + config.nheads
        self.in_proj = nn.Linear(config.d_model, d_in_proj, bias=False)

        conv_dim = config.d_inner + 2 * config.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=config.d_conv,
            groups=conv_dim,
            padding=config.d_conv - 1,
        )

        self.dt_bias = nn.Parameter(torch.empty(config.nheads))
        self.A_log = nn.Parameter(torch.empty(config.nheads))
        self.D = nn.Parameter(torch.empty(config.nheads))
        self.norm = RMSNorm(config.d_inner)
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=False)

        # Gated FFN
        self.ffn = GatedFFN(config)
        self.ffn_norm = RMSNorm(config.d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.dt_bias, -0.1, 0.1)
        nn.init.uniform_(self.A_log, -5.0, -1.0)
        nn.init.uniform_(self.D, -0.1, 0.1)

    def forward(
        self,
        u: torch.Tensor,
        h: Optional[InferenceCache] = None,
    ) -> tuple[torch.Tensor, InferenceCache]:
        """
        Args:
            u: (batch, seqlen, d_model)
            h: inference cache for step mode

        Returns:
            y: (batch, seqlen, d_model)
            h: updated cache
        """
        if h is not None:
            return self._step(u, h)

        # Full sequence forward
        A = -torch.exp(self.A_log)
        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.config.d_inner,
                self.config.d_inner + 2 * self.config.d_state,
                self.config.nheads,
            ],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)

        # Conv (xBC_raw is input to conv, needed for conv_state in step mode)
        xBC_raw = xBC
        xBC = _silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :]
        )
        x, B, C = torch.split(
            xBC,
            [self.config.d_inner, self.config.d_state, self.config.d_state],
            dim=-1,
        )
        x = rearrange(x, "b l (h p) -> b l h p", p=self.config.headdim)

        y, ssm_state = _ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.config.chunk_size,
            device=u.device,
        )
        y = y + x * self.D.unsqueeze(-1)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        # Residual + FFN
        u = u + y
        u = u + self.ffn(self.ffn_norm(u))

        # conv_state: last d_conv inputs (before conv) for step mode
        seqlen = u.shape[1]
        if seqlen >= self.config.d_conv:
            conv_state = rearrange(
                xBC_raw[:, -self.config.d_conv :, :], "b l d -> b d l"
            ).clone()
        else:
            padded = F.pad(
                xBC_raw,
                (0, 0, self.config.d_conv - seqlen, 0),
            )
            conv_state = rearrange(padded, "b l d -> b d l")
        h = InferenceCache(conv_state, ssm_state)
        return u, h

    def _step(self, u: torch.Tensor, h: InferenceCache) -> tuple[torch.Tensor, InferenceCache]:
        """Single-token inference step."""
        if u.shape[1] != 1:
            raise ValueError(
                f"Step mode requires seqlen=1, got {u.shape[1]}. "
                "When using cache, pass only the next token (batch, 1)."
            )
        zxbcdt = self.in_proj(u.squeeze(1))
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.config.d_inner,
                self.config.d_inner + 2 * self.config.d_state,
                self.config.nheads,
            ],
            dim=-1,
        )

        # Conv step
        h_conv = torch.roll(h.conv_state, shifts=-1, dims=-1)
        h_conv = h_conv.clone()
        h_conv[:, :, -1] = xBC
        xBC = torch.sum(
            h_conv * rearrange(self.conv1d.weight, "d 1 w -> d w"),
            dim=-1,
        )
        xBC += self.conv1d.bias
        xBC = _silu(xBC)

        x, B, C = torch.split(
            xBC,
            [self.config.d_inner, self.config.d_state, self.config.d_state],
            dim=-1,
        )
        A = -torch.exp(self.A_log)
        dt = F.softplus(dt + self.dt_bias)
        dA = torch.exp(dt * A)
        x = rearrange(x, "b (h p) -> b h p", p=self.config.headdim)
        dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
        ssm_state = h.ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx
        y = torch.einsum("bhpn, bn -> bhp", ssm_state, C)
        y = y + rearrange(self.D, "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        u = u + y
        u = u + self.ffn(self.ffn_norm(u))

        h_new = InferenceCache(h_conv, ssm_state)
        return u, h_new

# src/kmw/models/model.py

"""
KMW model components.

This file contains the learnable model stack for the current KMW design.

Contents
--------
1. LogSinkhorn
   - differentiable soft permutation solver used during training

2. LogicalReindexBranch
   - learns a soft reordering of logical qubit slots

3. HardwareReindexBranch
   - learns a soft reordering of hardware qubit slots

4. SoftPermutationReindexer
   - combines the two reindex branches
   - provides tensor reordering helpers

5. HardwareTokenEncoder
   - builds latent hardware tokens T_hw* from reordered hardware tensors

6. CrossAttentionBlock
   - 4-head cross-attention from spatial feature map -> hardware tokens

7. AttnUNet27
   - the locked shallow interpolation-based attention U-Net mapper

8. Decode / training assignment helpers
   - decode latent logits back to native frame
   - Sinkhorn assignment during training

9. Small trainer-side helper
   - optionally detach reindex outputs for mapper-only updates

10. KMWModel
   - convenience wrapper that runs:
       native tensors
         -> reindexer
         -> reordered tensors
         -> hardware token encoder
         -> U-Net mapper
         -> latent logits S*

Important architectural note
----------------------------
This file intentionally follows the locked implementation decisions for v1.1.1 / v1.2:

- Mapper backbone authority is Attn_Unet_27_SPEC.md.
- The mapper uses interpolation-based resizing, NOT stride-2 conv / transposed-conv.
- The mapper uses:
    Interpolate -> Conv3x3 -> attention
  at the resize stages defined by the spec.
- The final mapper output is Conv3x3(128 -> 1).
- No normalization layers are used inside the spatial mapper backbone.
- LayerNorm is kept only in token/reindexer MLP-style components.
- Hungarian hard assignment is intentionally NOT kept here;
  that belongs later in evaluation / inference code.
- Whether reindex outputs are detached or not is a training-policy decision,
  so the actual decision stays in trainer.py.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Small helper functions
# =============================================================================

def _assert_rank(x: torch.Tensor, expected_rank: int, name: str) -> None:
    """Raise a clear error if a tensor does not have the expected number of axes."""
    if x.ndim != expected_rank:
        raise ValueError(
            f"{name} must have rank {expected_rank}, but got shape {tuple(x.shape)}"
        )


def _assert_last_dims(x: torch.Tensor, expected_last_dims: Tuple[int, ...], name: str) -> None:
    """Check trailing dimensions of a tensor."""
    got = tuple(x.shape[-len(expected_last_dims):])
    if got != expected_last_dims:
        raise ValueError(
            f"{name} must end with shape {expected_last_dims}, but got {tuple(x.shape)}"
        )


def _assert_finite(x: torch.Tensor, name: str) -> None:
    """Fail loudly if a tensor contains NaN or Inf."""
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf.")


def _ensure_batched_square_matrix(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    Accept either:
        (N, N)       -> convert to (1, N, N)
        (B, N, N)    -> keep as is
    """
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim == 3:
        return x
    raise ValueError(f"{name} must have shape (N,N) or (B,N,N), got {tuple(x.shape)}")


def _ensure_batched_vector(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    Accept either:
        (N,)      -> convert to (1, N)
        (B, N)    -> keep as is
    """
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim == 2:
        return x
    raise ValueError(f"{name} must have shape (N,) or (B,N), got {tuple(x.shape)}")


# =============================================================================
# 1) Log-domain Sinkhorn
# =============================================================================

class LogSinkhorn(nn.Module):
    """
    Log-domain Sinkhorn normalization.

    What it does
    ------------
    Takes a score matrix and turns it into a soft permutation matrix
    (approximately doubly stochastic: rows sum to 1 and columns sum to 1).

    Why log-domain?
    ---------------
    Naive Sinkhorn often exponentiates raw scores first, which can become
    numerically unstable. Log-domain Sinkhorn performs repeated row/column
    normalization in log-space using logsumexp.

    Input:
        logits: (B, N, N) or (N, N)

    Output:
        P:      (B, N, N) or (N, N)
    """

    def __init__(self, num_iters: int = 20, eps: float = 1e-8):
        super().__init__()
        self.num_iters = num_iters
        self.eps = eps

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        was_unbatched = logits.ndim == 2
        log_alpha = _ensure_batched_square_matrix(logits, "logits")

        _assert_finite(log_alpha, "logits before Sinkhorn")

        for _ in range(self.num_iters):
            # Normalize rows in log-space
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            # Normalize columns in log-space
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)

        P = torch.exp(log_alpha)
        _assert_finite(P, "Sinkhorn output")

        if was_unbatched:
            return P[0]
        return P


# =============================================================================
# 2) Shared MLP builder
# =============================================================================

def build_mlp(in_dim: int, hidden_dim: int, dropout: float = 0.1) -> nn.Sequential:
    """
    Reusable MLP pattern used in:
    - logical reindex branch
    - hardware reindex branch
    - hardware token encoder

    Architectural note:
    -------------------
    LayerNorm is intentionally used here because normalization was explicitly
    retained in token/reindexer MLP-style paths. In contrast, the spatial
    U-Net backbone intentionally uses no normalization layers.
    """
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
    )


# =============================================================================
# 3) Logical reindex branch
# =============================================================================

class LogicalReindexBranch(nn.Module):
    """
    Learns a soft permutation R_L over logical slots.

    Input features per logical slot u:
        feat_L[u] = [
            log(1 + sum_v A[u, v]),
            top1_offdiag(A[u, :]),
            top2_offdiag(A[u, :]),
            m[u]
        ]

    Output:
        R_L of shape (B, N, N)

    Orientation convention
    ----------------------
    R_L[t, u] means:
        "how much original logical slot u is assigned to latent slot t"

    Therefore:
        latent_vector = R_L @ native_vector
        latent_matrix = R_L @ native_matrix @ R_L^T
    """

    def __init__(
        self,
        n: int = 27,
        hidden_dim: int = 128,
        sinkhorn_iters: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n = n
        self.hidden_dim = hidden_dim

        self.mlp = build_mlp(in_dim=4, hidden_dim=hidden_dim, dropout=dropout)

        # Learnable prototypes for the latent logical slots.
        self.slot_prototypes = nn.Parameter(torch.randn(n, hidden_dim) * 0.02)

        self.sinkhorn = LogSinkhorn(num_iters=sinkhorn_iters)

    def build_features(self, A: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Build logical features from:
            A: (B, N, N)
            m: (B, N)

        Returns:
            feat_L: (B, N, 4)
        """
        _assert_rank(A, 3, "A")
        _assert_rank(m, 2, "m")

        _, N, N2 = A.shape
        if N != self.n or N2 != self.n:
            raise ValueError(f"A must have shape (B,{self.n},{self.n}), got {tuple(A.shape)}")
        if m.shape[1] != self.n:
            raise ValueError(f"m must have shape (B,{self.n}), got {tuple(m.shape)}")

        row_sum = A.sum(dim=-1)

        # Zero out the diagonal before taking top-2 off-diagonal values.
        eye = torch.eye(self.n, device=A.device, dtype=A.dtype).unsqueeze(0)
        offdiag = A * (1.0 - eye)

        top2_vals, _ = torch.topk(offdiag, k=2, dim=-1)
        top1 = top2_vals[..., 0]
        top2 = top2_vals[..., 1]

        feat = torch.stack(
            [
                torch.log1p(row_sum),
                top1,
                top2,
                m,
            ],
            dim=-1,
        )

        _assert_finite(feat, "logical features")
        return feat

    def forward(self, A: torch.Tensor, m: torch.Tensor, tau_r: float) -> torch.Tensor:
        """
        Compute soft logical permutation matrix R_L.

        Pipeline:
            features -> MLP -> score vs latent prototypes -> Sinkhorn
        """
        feat = self.build_features(A, m)
        H = self.mlp(feat)

        # scores: (B, original_u, latent_t)
        scores = torch.matmul(H, self.slot_prototypes.t())

        # Re-orient into (B, latent_t, original_u)
        scores_tilde = scores.transpose(1, 2)

        R_L = self.sinkhorn(scores_tilde / tau_r)
        _assert_finite(R_L, "R_L")
        return R_L


# =============================================================================
# 4) Hardware reindex branch
# =============================================================================

class HardwareReindexBranch(nn.Module):
    """
    Learns a soft permutation R_H over hardware slots.

    Input features per hardware qubit j:
        feat_H[j] = [
            c1[j],
            deg(j),
            mean_c2(j),
            min_c2(j),
            mean_D(j)
        ]

    where:
        deg(j)     = sum_k B[j, k]
        mean_c2(j) = mean of c2[j, k] over valid neighbors
        min_c2(j)  = min  of c2[j, k] over valid neighbors
        mean_D(j)  = mean of D[j, :]
    """

    def __init__(
        self,
        n: int = 27,
        hidden_dim: int = 128,
        sinkhorn_iters: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n = n
        self.hidden_dim = hidden_dim

        self.mlp = build_mlp(in_dim=5, hidden_dim=hidden_dim, dropout=dropout)
        self.slot_prototypes = nn.Parameter(torch.randn(n, hidden_dim) * 0.02)
        self.sinkhorn = LogSinkhorn(num_iters=sinkhorn_iters)

    def build_features(
        self,
        Bmat: torch.Tensor,
        c1: torch.Tensor,
        c2: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build hardware features from:
            Bmat: (B, N, N)
            c1:   (B, N)
            c2:   (B, N, N)
            D:    (B, N, N)

        Returns:
            feat_H: (B, N, 5)

        Architectural note:
        -------------------
        The hardware branch uses row-wise c2[j, k] features, matching the
        locked interpretation in the combined design.
        """
        _assert_rank(Bmat, 3, "B")
        _assert_rank(c1, 2, "c1")
        _assert_rank(c2, 3, "c2")
        _assert_rank(D, 3, "D")

        _, N, N2 = Bmat.shape
        if (N, N2) != (self.n, self.n):
            raise ValueError(f"B must have shape (B,{self.n},{self.n}), got {tuple(Bmat.shape)}")
        if c1.shape[1] != self.n:
            raise ValueError(f"c1 must have shape (B,{self.n}), got {tuple(c1.shape)}")
        if c2.shape[-2:] != (self.n, self.n):
            raise ValueError(f"c2 must have shape (B,{self.n},{self.n}), got {tuple(c2.shape)}")
        if D.shape[-2:] != (self.n, self.n):
            raise ValueError(f"D must have shape (B,{self.n},{self.n}), got {tuple(D.shape)}")

        deg = Bmat.sum(dim=-1)

        nbr_mask = Bmat > 0

        c2_masked_sum = (c2 * nbr_mask.to(c2.dtype)).sum(dim=-1)
        nbr_count = nbr_mask.sum(dim=-1).clamp(min=1)
        mean_c2 = c2_masked_sum / nbr_count

        inf = torch.tensor(float("inf"), device=c2.device, dtype=c2.dtype)
        c2_for_min = torch.where(nbr_mask, c2, inf)
        min_c2 = c2_for_min.min(dim=-1).values
        min_c2 = torch.where(torch.isinf(min_c2), torch.zeros_like(min_c2), min_c2)

        mean_D = D.mean(dim=-1)

        feat = torch.stack(
            [
                c1,
                deg,
                mean_c2,
                min_c2,
                mean_D,
            ],
            dim=-1,
        )

        _assert_finite(feat, "hardware features")
        return feat

    def forward(
        self,
        Bmat: torch.Tensor,
        c1: torch.Tensor,
        c2: torch.Tensor,
        D: torch.Tensor,
        tau_r: float,
    ) -> torch.Tensor:
        feat = self.build_features(Bmat, c1, c2, D)
        H = self.mlp(feat)

        scores = torch.matmul(H, self.slot_prototypes.t())
        scores_tilde = scores.transpose(1, 2)

        R_H = self.sinkhorn(scores_tilde / tau_r)
        _assert_finite(R_H, "R_H")
        return R_H


# =============================================================================
# 5) Soft permutation reindexer
# =============================================================================

class SoftPermutationReindexer(nn.Module):
    """
    Wrapper around:
        - LogicalReindexBranch
        - HardwareReindexBranch

    Reordering conventions
    ----------------------
    If R has shape (latent, original), then:
        vector* = R @ vector
        matrix* = R @ matrix @ R^T
    """

    def __init__(
        self,
        n: int = 27,
        hidden_dim: int = 128,
        sinkhorn_iters: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n = n

        self.logical_branch = LogicalReindexBranch(
            n=n,
            hidden_dim=hidden_dim,
            sinkhorn_iters=sinkhorn_iters,
            dropout=dropout,
        )
        self.hardware_branch = HardwareReindexBranch(
            n=n,
            hidden_dim=hidden_dim,
            sinkhorn_iters=sinkhorn_iters,
            dropout=dropout,
        )

    @staticmethod
    def reorder_vector(R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Reorder a vector: x* = R @ x"""
        return torch.matmul(R, x.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def reorder_matrix(R: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Reorder a square matrix: M* = R @ M @ R^T"""
        return torch.matmul(torch.matmul(R, M), R.transpose(1, 2))

    def reorder_all(
        self,
        A: torch.Tensor,
        m: torch.Tensor,
        Bmat: torch.Tensor,
        c1: torch.Tensor,
        c2: torch.Tensor,
        D: torch.Tensor,
        R_L: torch.Tensor,
        R_H: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Reorder every native-frame tensor into the latent frame.
        """
        A_star = self.reorder_matrix(R_L, A)
        m_star = self.reorder_vector(R_L, m)

        B_star = self.reorder_matrix(R_H, Bmat)
        c1_star = self.reorder_vector(R_H, c1)
        c2_star = self.reorder_matrix(R_H, c2)
        D_star = self.reorder_matrix(R_H, D)

        for name, tensor in {
            "A_star": A_star,
            "m_star": m_star,
            "B_star": B_star,
            "c1_star": c1_star,
            "c2_star": c2_star,
            "D_star": D_star,
        }.items():
            _assert_finite(tensor, name)

        return {
            "A_star": A_star,
            "m_star": m_star,
            "B_star": B_star,
            "c1_star": c1_star,
            "c2_star": c2_star,
            "D_star": D_star,
        }

    def forward(
        self,
        A: torch.Tensor,
        m: torch.Tensor,
        Bmat: torch.Tensor,
        c1: torch.Tensor,
        c2: torch.Tensor,
        D: torch.Tensor,
        tau_r: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Full reindexer forward pass.
        """
        R_L = self.logical_branch(A, m, tau_r=tau_r)
        R_H = self.hardware_branch(Bmat, c1, c2, D, tau_r=tau_r)

        reordered = self.reorder_all(A, m, Bmat, c1, c2, D, R_L, R_H)

        return {
            "R_L": R_L,
            "R_H": R_H,
            **reordered,
        }


# =============================================================================
# 6) Hardware token encoder
# =============================================================================

class HardwareTokenEncoder(nn.Module):
    """
    Build hardware tokens T_hw* from reordered hardware tensors.

    Locked token rule
    -----------------
    For each latent hardware slot j:

        x_hw*[j] = concat(B*[j, :], c2*[j, :], c1*[j])

    Dimensions:
        B*[j, :]  -> 27
        c2*[j, :] -> 27
        c1*[j]    -> 1
        total     -> 55

    Then an MLP maps 55 -> 128.
    """

    def __init__(self, n: int = 27, token_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.n = n
        self.token_dim = token_dim

        in_dim = 2 * n + 1  # 55 when n=27
        self.mlp = build_mlp(in_dim=in_dim, hidden_dim=token_dim, dropout=dropout)

    def build_token_inputs(
        self,
        B_star: torch.Tensor,
        c2_star: torch.Tensor,
        c1_star: torch.Tensor,
    ) -> torch.Tensor:
        """
        Construct x_hw* for every latent hardware slot.

        Inputs:
            B_star  : (B, N, N)
            c2_star : (B, N, N)
            c1_star : (B, N)

        Output:
            x_hw : (B, N, 2N + 1)
        """
        _assert_rank(B_star, 3, "B_star")
        _assert_rank(c2_star, 3, "c2_star")
        _assert_rank(c1_star, 2, "c1_star")

        _, N, N2 = B_star.shape
        if (N, N2) != (self.n, self.n):
            raise ValueError(f"B_star must have shape (B,{self.n},{self.n})")
        if c2_star.shape[-2:] != (self.n, self.n):
            raise ValueError(f"c2_star must have shape (B,{self.n},{self.n})")
        if c1_star.shape[1] != self.n:
            raise ValueError(f"c1_star must have shape (B,{self.n})")

        c1_expanded = c1_star.unsqueeze(-1)
        x_hw = torch.cat([B_star, c2_star, c1_expanded], dim=-1)

        _assert_finite(x_hw, "hardware token inputs")
        return x_hw

    def forward(
        self,
        B_star: torch.Tensor,
        c2_star: torch.Tensor,
        c1_star: torch.Tensor,
    ) -> torch.Tensor:
        x_hw = self.build_token_inputs(B_star, c2_star, c1_star)
        T_hw = self.mlp(x_hw)
        _assert_finite(T_hw, "T_hw_star")
        return T_hw


# =============================================================================
# 7) Cross-attention block
# =============================================================================

class CrossAttentionBlock(nn.Module):
    """
    Cross-attention from spatial feature map -> hardware tokens.

    Input:
        Fmap : (B, C, H, W)
        T_hw : (B, 27, 128)

    Locked design notes
    -------------------
    - 4 heads
    - attention dimension = 128
    - hardware tokens provide K and V
    - spatial feature map provides Q
    - residual-gated attention update:
          F_out = F + alpha_attn * DeltaF
    - alpha_attn starts at 0.0 so attention is initially a no-op
    """

    def __init__(
        self,
        in_channels: int,
        token_dim: int = 128,
        attn_dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()

        if attn_dim % num_heads != 0:
            raise ValueError("attn_dim must be divisible by num_heads")

        self.in_channels = in_channels
        self.token_dim = token_dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads

        self.q_proj = nn.Linear(in_channels, attn_dim)
        self.k_proj = nn.Linear(token_dim, attn_dim)
        self.v_proj = nn.Linear(token_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, in_channels)

        self.alpha_attn = nn.Parameter(torch.tensor(0.0))

    def forward(self, Fmap: torch.Tensor, T_hw: torch.Tensor) -> torch.Tensor:
        _assert_rank(Fmap, 4, "Fmap")
        _assert_rank(T_hw, 3, "T_hw")

        Bsz, C, H, W = Fmap.shape
        if T_hw.shape[0] != Bsz:
            raise ValueError("Batch size mismatch between Fmap and T_hw")
        if T_hw.shape[1] != 27:
            raise ValueError("T_hw must have 27 hardware tokens")
        if T_hw.shape[2] != self.token_dim:
            raise ValueError(
                f"T_hw last dimension must be {self.token_dim}, got {T_hw.shape[2]}"
            )

        # Flatten spatial feature map into a sequence of HW spatial tokens.
        F_flat = Fmap.flatten(start_dim=2).transpose(1, 2)  # (B, HW, C)

        Q = self.q_proj(F_flat)  # (B, HW, 128)
        K = self.k_proj(T_hw)    # (B, 27, 128)
        V = self.v_proj(T_hw)    # (B, 27, 128)

        Q = Q.view(Bsz, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(Bsz, 27, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(Bsz, 27, self.num_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)

        attn_out = attn_out.transpose(1, 2).contiguous().view(Bsz, H * W, self.attn_dim)
        delta_flat = self.out_proj(attn_out)
        delta = delta_flat.transpose(1, 2).contiguous().view(Bsz, C, H, W)

        out = Fmap + self.alpha_attn * delta
        _assert_finite(out, "CrossAttention output")
        return out


# =============================================================================
# 8) Locked shallow attention U-Net mapper
# =============================================================================

class AttnUNet27(nn.Module):
    """
    Authoritative mapper backbone for v1.1.1 / v1.2.

    Locked structure
    ----------------
    Stage 1: down1
        A* -> Conv3x3(1->64) -> ReLU

    Stage 2: down2
        Interpolate to 14x14
        Conv3x3(64->128) -> ReLU
        CrossAttn

    Stage 3: bottleneck
        Interpolate to 7x7
        Conv3x3(128->256) -> ReLU
        CrossAttn

    Stage 4: up1
        Interpolate to 14x14
        Conv3x3(256->128) -> ReLU
        Add skip from d2
        CrossAttn

    Stage 5: head
        Interpolate to 27x27
        Conv3x3(128->1)
        no activation

    Architectural decision note
    ---------------------------
    This mapper intentionally does NOT use:
    - stride-2 downsampling convolutions
    - transposed convolutions
    - BatchNorm / LayerNorm / GroupNorm in spatial blocks

    Those were explicitly left out to match the locked Attn_Unet_27 design
    and the current clarified implementation decisions.
    """

    def __init__(self, token_dim: int = 128, num_heads: int = 4):
        super().__init__()

        # Stage 1
        self.conv_down1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True)

        # Stage 2
        self.conv_down2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.attn_down2 = CrossAttentionBlock(
            in_channels=128,
            token_dim=token_dim,
            attn_dim=128,
            num_heads=num_heads,
        )

        # Stage 3
        self.conv_bottleneck = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.attn_bottleneck = CrossAttentionBlock(
            in_channels=256,
            token_dim=token_dim,
            attn_dim=128,
            num_heads=num_heads,
        )

        # Stage 4
        self.conv_up1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.attn_up1 = CrossAttentionBlock(
            in_channels=128,
            token_dim=token_dim,
            attn_dim=128,
            num_heads=num_heads,
        )

        # Stage 5
        self.conv_head = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=True)

    @staticmethod
    def _resize(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """
        Locked interpolation policy:
            mode = "bilinear"
            align_corners = False
        """
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, A_star_spatial: torch.Tensor, T_hw_star: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the locked mapper backbone.

        Inputs:
            A_star_spatial : (B, 1, 27, 27)
            T_hw_star      : (B, 27, 128)

        Output:
            S_star         : (B, 1, 27, 27)
        """
        _assert_rank(A_star_spatial, 4, "A_star_spatial")
        _assert_rank(T_hw_star, 3, "T_hw_star")
        _assert_last_dims(A_star_spatial, (27, 27), "A_star_spatial")
        if A_star_spatial.shape[1] != 1:
            raise ValueError("A_star_spatial must have exactly 1 channel")
        if T_hw_star.shape[1:] != (27, 128):
            raise ValueError(
                f"T_hw_star must have shape (B,27,128), got {tuple(T_hw_star.shape)}"
            )

        # Stage 1: down1
        d1 = F.relu(self.conv_down1(A_star_spatial))      # (B, 64, 27, 27)

        # Stage 2: down2 + attention
        d1_ds = self._resize(d1, size=(14, 14))           # (B, 64, 14, 14)
        d2_pre = F.relu(self.conv_down2(d1_ds))           # (B, 128, 14, 14)
        d2 = self.attn_down2(d2_pre, T_hw_star)           # (B, 128, 14, 14)

        # Stage 3: bottleneck + attention
        d2_ds = self._resize(d2, size=(7, 7))             # (B, 128, 7, 7)
        b_pre = F.relu(self.conv_bottleneck(d2_ds))       # (B, 256, 7, 7)
        b = self.attn_bottleneck(b_pre, T_hw_star)        # (B, 256, 7, 7)

        # Stage 4: up1 + one additive skip + attention
        b_us = self._resize(b, size=(14, 14))             # (B, 256, 14, 14)
        u1_pre = F.relu(self.conv_up1(b_us))              # (B, 128, 14, 14)

        # Locked skip rule: additive skip from d2 only.
        u1_skip = u1_pre + d2
        u1 = self.attn_up1(u1_skip, T_hw_star)            # (B, 128, 14, 14)

        # Stage 5: head
        u1_us = self._resize(u1, size=(27, 27))           # (B, 128, 27, 27)
        S_star = self.conv_head(u1_us)                    # (B, 1, 27, 27)

        _assert_finite(S_star, "S_star")
        return S_star


# =============================================================================
# 9) Decode / training assignment helpers
# =============================================================================

def decode_to_native(
    S_star: torch.Tensor,
    R_L: torch.Tensor,
    R_H: torch.Tensor,
) -> torch.Tensor:
    """
    Decode latent-frame logits back to the native frame.

    Locked decode rule:
        S_nat = R_L^T @ S* @ R_H

    Shapes:
        S_star : (B, 1, N, N) or (B, N, N)
        R_L    : (B, N, N)
        R_H    : (B, N, N)

    Returns:
        S_nat  : (B, N, N)
    """
    if S_star.ndim == 4:
        if S_star.shape[1] != 1:
            raise ValueError("S_star with rank 4 must have channel dimension = 1")
        S_star_mat = S_star[:, 0]
    elif S_star.ndim == 3:
        S_star_mat = S_star
    else:
        raise ValueError("S_star must have shape (B,1,N,N) or (B,N,N)")

    S_nat = torch.matmul(torch.matmul(R_L.transpose(1, 2), S_star_mat), R_H)
    _assert_finite(S_nat, "S_nat")
    return S_nat


def sinkhorn_assignment(
    S_nat: torch.Tensor,
    tau_m: float = 0.30,
    num_iters: int = 20,
) -> torch.Tensor:
    """
    Training-time soft assignment:
        P_map = Sinkhorn(S_nat / tau_m)

    Note
    ----
    This helper stays in model.py because it is part of the differentiable
    training path. Hard Hungarian assignment is intentionally deferred to
    evaluation/inference code.
    """
    sinkhorn = LogSinkhorn(num_iters=num_iters)
    return sinkhorn(S_nat / tau_m)


# =============================================================================
# 10) Small trainer helper
# =============================================================================

def maybe_detach_reindex_outputs(
    reindex_outputs: Dict[str, torch.Tensor],
    detach: bool,
) -> Dict[str, torch.Tensor]:
    """
    Small helper for trainer readability.

    Why this exists
    ---------------
    The actual detach decision belongs to trainer.py because it is a training
    policy choice, not a model-architecture choice.

    Example usage in trainer.py
    ---------------------------
        reidx = model.reindexer(...)
        reidx_for_mapper = maybe_detach_reindex_outputs(reidx, detach=True)

    Behavior
    --------
    - if detach=False: returns the original tensors unchanged
    - if detach=True:  returns a new dict with all tensor values detached

    Non-tensor values, if ever present, are passed through unchanged.
    """
    if not detach:
        return reindex_outputs

    out = {}
    for key, value in reindex_outputs.items():
        if torch.is_tensor(value):
            out[key] = value.detach()
        else:
            out[key] = value
    return out


# =============================================================================
# 11) Full KMW model wrapper
# =============================================================================

class KMWModel(nn.Module):
    """
    Convenience wrapper that chains the current model components.

    Native-frame inputs:
        A   : (B, 27, 27)
        m   : (B, 27)
        B   : (B, 27, 27)
        c1  : (B, 27)
        c2  : (B, 27, 27)
        D   : (B, 27, 27)

    Forward pipeline:
        native tensors
          -> soft reindexer
          -> reordered tensors
          -> hardware token encoder
          -> attention U-Net mapper
          -> latent logits S*

    Important
    ---------
    - This wrapper stops at differentiable outputs needed for training.
    - Hard inference assignment is intentionally NOT handled here.
    """

    def __init__(
        self,
        n: int = 27,
        d_r: int = 128,
        d_tok: int = 128,
        sinkhorn_iters: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n = n
        self.d_r = d_r
        self.d_tok = d_tok

        self.reindexer = SoftPermutationReindexer(
            n=n,
            hidden_dim=d_r,
            sinkhorn_iters=sinkhorn_iters,
            dropout=dropout,
        )
        self.token_encoder = HardwareTokenEncoder(
            n=n,
            token_dim=d_tok,
            dropout=dropout,
        )
        self.mapper = AttnUNet27(
            token_dim=d_tok,
            num_heads=4,  # locked by design
        )

    def forward(
        self,
        A: torch.Tensor,
        m: torch.Tensor,
        Bmat: torch.Tensor,
        c1: torch.Tensor,
        c2: torch.Tensor,
        D: torch.Tensor,
        tau_r: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Full latent-frame forward pass.

        Returns:
            {
                "R_L", "R_H",
                "A_star", "m_star", "B_star", "c1_star", "c2_star", "D_star",
                "T_hw_star",
                "S_star",
            }
        """
        # 1) Soft reindexing
        reidx = self.reindexer(
            A=A,
            m=m,
            Bmat=Bmat,
            c1=c1,
            c2=c2,
            D=D,
            tau_r=tau_r,
        )

        # 2) Hardware token encoding
        T_hw_star = self.token_encoder(
            B_star=reidx["B_star"],
            c2_star=reidx["c2_star"],
            c1_star=reidx["c1_star"],
        )

        # 3) Mapper backbone
        A_star_spatial = reidx["A_star"].unsqueeze(1)  # (B, 1, 27, 27)
        S_star = self.mapper(A_star_spatial, T_hw_star)

        return {
            **reidx,
            "T_hw_star": T_hw_star,
            "S_star": S_star,
        }

    @staticmethod
    def decode_to_native(
        S_star: torch.Tensor,
        R_L: torch.Tensor,
        R_H: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience wrapper around the standalone helper."""
        return decode_to_native(S_star=S_star, R_L=R_L, R_H=R_H)

    @staticmethod
    def training_assignment(
        S_nat: torch.Tensor,
        tau_m: float = 0.30,
        num_iters: int = 20,
    ) -> torch.Tensor:
        """Training-time soft assignment helper."""
        return sinkhorn_assignment(S_nat=S_nat, tau_m=tau_m, num_iters=num_iters)


# =============================================================================
# 12) Public export list
# =============================================================================

__all__ = [
    "LogSinkhorn",
    "LogicalReindexBranch",
    "HardwareReindexBranch",
    "SoftPermutationReindexer",
    "HardwareTokenEncoder",
    "CrossAttentionBlock",
    "AttnUNet27",
    "decode_to_native",
    "sinkhorn_assignment",
    "maybe_detach_reindex_outputs",
    "KMWModel",
]


# =============================================================================
# notes on implementation:
# =============================================================================
# I kept the mapper backbone exactly in the interpolation-based form you locked, 
# not the stride-2 / transpose-conv variant.

# I kept normalization out of the spatial conv backbone and only used LayerNorm inside the MLP-style components, 
# matching your locked interpretation. That choice came from your clarification decision plus the MLP specs for the reindexer/token encoders.

# I used c2[j, k] row-wise in the hardware features and concat(B*[j,:], c2*[j,:], c1*[j]) for mapper hardware tokens, 
# which is consistent with the mapper/reindexer sections of the combined plan.
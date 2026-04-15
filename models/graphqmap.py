"""GraphQMap: Full model combining all components.

Pipeline:
  [Circuit Graph] → Circuit GNN → C (l×d)
  [Hardware Graph] → Hardware GNN → H (h×d)
  → Cross-Attention (2 layers) → C' (l×d), H' (h×d)
  → Score Head → S (l×h)
  → Row-wise Softmax(S/τ) → P (l×h)
  [Training] P used for loss
  [Inference] Hungarian(P) → discrete layout
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data

from models.cross_attention import CrossAttentionModule
from models.gnn_encoder import GNNEncoder
from models.hungarian import hungarian_decode_batch
from models.score_head import ScoreHead
from models.sinkhorn import SinkhornLayer, SoftmaxNorm


class GraphQMap(nn.Module):
    """Hardware-agnostic quantum circuit qubit mapping model.

    Args:
        circuit_node_dim: Circuit graph node feature dimension.
        circuit_edge_dim: Circuit graph edge feature dimension.
        hardware_node_dim: Hardware graph node feature dimension.
        hardware_edge_dim: Hardware graph edge feature dimension.
        embedding_dim: Shared embedding dimension (d).
        gnn_layers: Number of GATv2 layers.
        gnn_heads: Number of GATv2 attention heads.
        gnn_dropout: GNN dropout rate.
        cross_attn_layers: Number of cross-attention layers.
        cross_attn_heads: Number of cross-attention heads.
        cross_attn_ffn_dim: FFN hidden dimension in cross-attention.
        cross_attn_dropout: Cross-attention dropout rate.
        score_d_k: Score head projection dimension.
        noise_bias_dim: Hardware feature dim for score head bias (0 to disable).
        sinkhorn_max_iter: Maximum Sinkhorn iterations.
        sinkhorn_tol: Sinkhorn convergence tolerance.
        score_norm: Score normalization type ("softmax" or "sinkhorn").
    """

    def __init__(
        self,
        circuit_node_dim: int = 4,
        circuit_edge_dim: int = 3,
        hardware_node_dim: int = 5,
        hardware_edge_dim: int = 1,
        embedding_dim: int = 64,
        gnn_layers: int = 3,
        gnn_heads: int = 4,
        gnn_dropout: float = 0.1,
        cross_attn_layers: int = 2,
        cross_attn_heads: int = 4,
        cross_attn_ffn_dim: int = 128,
        cross_attn_dropout: float = 0.1,
        score_d_k: int = 64,
        noise_bias_dim: int = 0,
        sinkhorn_max_iter: int = 20,
        sinkhorn_tol: float = 1e-6,
        score_norm: str = "softmax",
        refine_iterations: int = 0,
        refine_lambda: float = 1.0,
        refine_beta: float = 0.9,
        bypass_cross_attn: bool = False,
        cross_attn_gate: bool = False,
        zero_score_init: bool = False,
    ) -> None:
        super().__init__()
        self.refine_iterations = refine_iterations
        self.refine_beta = refine_beta
        self.bypass_cross_attn = bypass_cross_attn
        self.zero_score_init = zero_score_init
        if refine_iterations > 0:
            self.refine_lambda = nn.Parameter(torch.tensor(refine_lambda))
        else:
            self.refine_lambda = None

        # When zero_score_init=True, skip all neural components.
        # S^(0) = 0 (fixed), only refine_lambda is learned.
        # This tests whether the GNN contributes anything beyond
        # what iterative QAP refinement provides analytically.
        if not zero_score_init:
            # Dual GNN encoders (independent, no shared parameters)
            self.circuit_gnn = GNNEncoder(
                node_input_dim=circuit_node_dim,
                edge_input_dim=circuit_edge_dim,
                embedding_dim=embedding_dim,
                num_layers=gnn_layers,
                num_heads=gnn_heads,
                dropout=gnn_dropout,
            )
            self.hardware_gnn = GNNEncoder(
                node_input_dim=hardware_node_dim,
                edge_input_dim=hardware_edge_dim,
                embedding_dim=embedding_dim,
                num_layers=gnn_layers,
                num_heads=gnn_heads,
                dropout=gnn_dropout,
            )

            # Cross-attention (skipped entirely when bypass_cross_attn=True)
            if not bypass_cross_attn:
                self.cross_attention = CrossAttentionModule(
                    d_model=embedding_dim,
                    num_layers=cross_attn_layers,
                    num_heads=cross_attn_heads,
                    d_ff=cross_attn_ffn_dim,
                    dropout=cross_attn_dropout,
                    use_gate=cross_attn_gate,
                )
            else:
                self.cross_attention = None

            # Score head
            self.score_head = ScoreHead(
                d_model=embedding_dim, d_k=score_d_k, noise_bias_dim=noise_bias_dim,
            )
        else:
            self.circuit_gnn = None
            self.hardware_gnn = None
            self.cross_attention = None
            self.score_head = None

        # Score normalization
        if score_norm == "sinkhorn":
            self.sinkhorn = SinkhornLayer(
                max_iter=sinkhorn_max_iter, tol=sinkhorn_tol,
            )
        else:
            self.sinkhorn = SoftmaxNorm()

    def encode(
        self,
        circuit_data: Data,
        hardware_data: Data,
        batch_size: int,
        num_logical: int,
        num_physical: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode circuit and hardware graphs into embeddings.

        Args:
            circuit_data: Batched PyG Data for circuit graphs.
            hardware_data: Batched PyG Data for hardware graphs.
            batch_size: Number of samples in batch.
            num_logical: Number of logical qubits per sample.
            num_physical: Number of physical qubits per sample.

        Returns:
            C: (batch, l, d) circuit embeddings.
            H: (batch, h, d) hardware embeddings.
        """
        # GNN forward
        c_emb = self.circuit_gnn(
            circuit_data.x, circuit_data.edge_index, circuit_data.edge_attr,
        )  # (total_circuit_nodes, d)
        h_emb = self.hardware_gnn(
            hardware_data.x, hardware_data.edge_index, hardware_data.edge_attr,
        )  # (total_hardware_nodes, d)

        # Reshape to (batch, nodes_per_sample, d)
        C = c_emb.view(batch_size, num_logical, -1)
        H = h_emb.view(batch_size, num_physical, -1)

        return C, H

    def forward(
        self,
        circuit_data: Data,
        hardware_data: Data,
        batch_size: int,
        num_logical: int,
        num_physical: int,
        tau: float = 0.05,
        hw_node_features: torch.Tensor | None = None,
        c_eff: torch.Tensor | None = None,
        circuit_adj: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full forward pass: GNN → Cross-Attention → Score → [Iterative Refinement] → P.

        When refine_iterations > 0 and c_eff/circuit_adj are provided, applies
        iterative QAP mirror descent refinement after initial score computation:
            S^(t+1) = S^(0) - λ · Ã_c · P^(t) · C_eff
        This feeds the QAP gradient back into the score, improving layout quality
        beyond what the GNN alone achieves.

        Args:
            circuit_data: Batched PyG Data for circuit graphs.
            hardware_data: Batched PyG Data for hardware graphs.
            batch_size: Number of samples in batch.
            num_logical: Number of logical qubits per sample.
            num_physical: Number of physical qubits per sample.
            tau: Sinkhorn temperature.
            hw_node_features: (h, feat_dim) raw hardware features for noise bias.
            c_eff: (h, h) effective cost matrix for iterative refinement.
            circuit_adj: (l, l) gate-count weighted adjacency for refinement.

        Returns:
            P: Row-stochastic matrix (batch, l, h).
        """
        # Zero-score mode: S^(0) = 0, skip all neural components
        if self.zero_score_init:
            device = c_eff.device if c_eff is not None else circuit_adj.device
            S = torch.zeros(batch_size, num_logical, num_physical, device=device)
            S_init = S
        else:
            # Encode
            C, H = self.encode(
                circuit_data, hardware_data, batch_size, num_logical, num_physical,
            )

            # Cross-attention interaction (skipped when bypass_cross_attn=True)
            if self.cross_attention is not None:
                C_prime, H_prime = self.cross_attention(C, H)
            else:
                C_prime, H_prime = C, H

            # Score matrix (with optional noise bias)
            S = self.score_head(C_prime, H_prime, hw_node_features)  # (batch, l, h)

        # Iterative refinement (QAP mirror descent)
        if (self.refine_iterations > 0
                and c_eff is not None and circuit_adj is not None):
            c_eff_dev = c_eff.to(device=S.device, dtype=S.dtype)
            A_c = circuit_adj.to(device=S.device, dtype=S.dtype)

            # Ensure A_c is batched (B, l, l) for per-sample refinement
            if A_c.dim() == 2:
                A_c = A_c.unsqueeze(0).expand(batch_size, -1, -1)

            if self.zero_score_init:
                # S_init is already 0, no normalization needed
                pass
            else:
                # Normalize S^(0) so that feedback term is at comparable scale.
                S_std = S.std().detach().clamp(min=1e-6)
                S_mean = S.mean().detach()
                S_init = (S - S_mean) / S_std  # normalized to ~N(0,1)

            # Tau schedule: start high, land on epoch tau at the last iteration.
            # tau_start = tau / beta^(T-1), so after T-1 multiplications by beta
            # the final iteration uses exactly the epoch-level tau.
            T = self.refine_iterations
            tau_t = tau / (self.refine_beta ** (T - 1))

            S_current = S_init
            for _t in range(T):
                # Soft assignment at current temperature
                P_t = self.sinkhorn(S_current, num_logical, num_physical, tau_t)
                if P_t.shape[1] != num_logical:
                    P_t = P_t[:, :num_logical, :]

                # Z = P @ C_eff: expected cost landscape
                Z = torch.matmul(P_t, c_eff_dev)  # (B, l, h)

                # QAP gradient feedback: Ã_c @ Z — per-sample adjacency
                feedback = torch.bmm(A_c, Z)  # (B, l, h)

                # Update score: subtract fidelity-improving direction
                S_current = S_init - self.refine_lambda * feedback

                # Decay temperature for next iteration
                tau_t = tau_t * self.refine_beta

            # Final P is from the last iteration — no extra Sinkhorn needed.
            # The last iteration used tau_t = tau (epoch-level), matching the
            # training schedule. This follows the paper's structure exactly.
            P = P_t
            return P

        # No refinement: single Sinkhorn pass
        P = self.sinkhorn(S, num_logical, num_physical, tau)

        # Sinkhorn returns (batch, h, h) with dummy rows; extract logical rows
        if P.shape[1] != num_logical:
            P = P[:, :num_logical, :]  # (batch, l, h)

        return P

    @torch.no_grad()
    def predict(
        self,
        circuit_data: Data,
        hardware_data: Data,
        batch_size: int,
        num_logical: int,
        num_physical: int,
        tau: float = 0.05,
        hw_node_features: torch.Tensor | None = None,
        c_eff: torch.Tensor | None = None,
        circuit_adj: torch.Tensor | None = None,
    ) -> list[dict[int, int]]:
        """Inference: forward pass + Hungarian decoding.

        Args:
            circuit_data: Batched PyG Data for circuit graphs.
            hardware_data: Batched PyG Data for hardware graphs.
            batch_size: Number of samples in batch.
            num_logical: Number of logical qubits per sample.
            num_physical: Number of physical qubits per sample.
            tau: Sinkhorn temperature.
            hw_node_features: (h, feat_dim) raw hardware features for noise bias.
            c_eff: (h, h) effective cost matrix for iterative refinement.
            circuit_adj: (l, l) gate-count weighted adjacency for refinement.

        Returns:
            List of layout dicts {logical_qubit: physical_qubit}.
        """
        self.eval()
        P = self.forward(
            circuit_data, hardware_data, batch_size, num_logical, num_physical,
            tau, hw_node_features, c_eff, circuit_adj,
        )
        return hungarian_decode_batch(P, num_logical)

    @classmethod
    def from_config(cls, cfg: object) -> "GraphQMap":
        """Create model from a Config object.

        Args:
            cfg: Config with model.* attributes.

        Returns:
            Initialized GraphQMap model.
        """
        # Auto-compute circuit node_input_dim from feature config
        node_features = getattr(cfg.model.circuit_gnn, "node_features", None)
        rwpe_k = getattr(cfg.model.circuit_gnn, "rwpe_k", 0)
        if node_features is not None:
            circuit_node_dim = len(node_features) + rwpe_k
        else:
            circuit_node_dim = getattr(cfg.model.circuit_gnn, "node_input_dim", 4)

        # Iterative refinement config
        refine_cfg = getattr(cfg.model, "iterative_refinement", None)
        refine_iterations = getattr(refine_cfg, "iterations", 0) if refine_cfg else 0
        refine_lambda = getattr(refine_cfg, "lambda_init", 1.0) if refine_cfg else 1.0
        refine_beta = getattr(refine_cfg, "beta", 0.9) if refine_cfg else 0.9

        # Cross-attention options
        cross_attn_cfg = cfg.model.cross_attention
        bypass_cross_attn = getattr(cross_attn_cfg, "bypass", False)
        cross_attn_gate = getattr(cross_attn_cfg, "use_gate", False)

        # Zero-score init: S^(0)=0, no neural components, refinement only
        zero_score_init = getattr(cfg.model, "zero_score_init", False)

        return cls(
            circuit_node_dim=circuit_node_dim,
            circuit_edge_dim=cfg.model.circuit_gnn.edge_input_dim,
            hardware_node_dim=cfg.model.hardware_gnn.node_input_dim,
            hardware_edge_dim=cfg.model.hardware_gnn.edge_input_dim,
            embedding_dim=cfg.model.embedding_dim,
            gnn_layers=cfg.model.circuit_gnn.num_layers,
            gnn_heads=cfg.model.circuit_gnn.num_heads,
            gnn_dropout=cfg.model.circuit_gnn.dropout,
            cross_attn_layers=cross_attn_cfg.num_layers,
            cross_attn_heads=cross_attn_cfg.num_heads,
            cross_attn_ffn_dim=cross_attn_cfg.ffn_hidden_dim,
            cross_attn_dropout=cross_attn_cfg.dropout,
            score_d_k=cfg.model.score_head.d_k,
            noise_bias_dim=getattr(cfg.model.score_head, "noise_bias_dim", 0),
            sinkhorn_max_iter=cfg.sinkhorn.max_iter,
            sinkhorn_tol=cfg.sinkhorn.tolerance,
            score_norm=getattr(cfg.sinkhorn, "score_norm", "softmax"),
            refine_iterations=refine_iterations,
            refine_lambda=refine_lambda,
            refine_beta=refine_beta,
            bypass_cross_attn=bypass_cross_attn,
            cross_attn_gate=cross_attn_gate,
            zero_score_init=zero_score_init,
        )

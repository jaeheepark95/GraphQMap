"""GraphQMap: Full model combining all components.

Pipeline:
  [Circuit Graph] → Circuit GNN → C (l×d)
  [Hardware Graph] → Hardware GNN → H (h×d)
  → Cross-Attention (2 layers) → C' (l×d), H' (h×d)
  → Score Head → S (l×h)
  → Dummy Padding + Log-domain Sinkhorn → P (h×h)
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
from models.sinkhorn import SinkhornLayer


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
        sinkhorn_max_iter: Maximum Sinkhorn iterations.
        sinkhorn_tol: Sinkhorn convergence tolerance.
    """

    def __init__(
        self,
        circuit_node_dim: int = 8,
        circuit_edge_dim: int = 3,
        hardware_node_dim: int = 6,
        hardware_edge_dim: int = 2,
        embedding_dim: int = 64,
        gnn_layers: int = 3,
        gnn_heads: int = 4,
        gnn_dropout: float = 0.1,
        cross_attn_layers: int = 2,
        cross_attn_heads: int = 4,
        cross_attn_ffn_dim: int = 128,
        cross_attn_dropout: float = 0.1,
        score_d_k: int = 64,
        sinkhorn_max_iter: int = 20,
        sinkhorn_tol: float = 1e-6,
    ) -> None:
        super().__init__()

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

        # Cross-attention
        self.cross_attention = CrossAttentionModule(
            d_model=embedding_dim,
            num_layers=cross_attn_layers,
            num_heads=cross_attn_heads,
            d_ff=cross_attn_ffn_dim,
            dropout=cross_attn_dropout,
        )

        # Score head
        self.score_head = ScoreHead(d_model=embedding_dim, d_k=score_d_k)

        # Sinkhorn
        self.sinkhorn = SinkhornLayer(
            max_iter=sinkhorn_max_iter,
            tol=sinkhorn_tol,
        )

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
    ) -> torch.Tensor:
        """Full forward pass: GNN → Cross-Attention → Score → Sinkhorn → P.

        Args:
            circuit_data: Batched PyG Data for circuit graphs.
            hardware_data: Batched PyG Data for hardware graphs.
            batch_size: Number of samples in batch.
            num_logical: Number of logical qubits per sample.
            num_physical: Number of physical qubits per sample.
            tau: Sinkhorn temperature.

        Returns:
            P: Doubly stochastic matrix (batch, h, h).
        """
        # Encode
        C, H = self.encode(
            circuit_data, hardware_data, batch_size, num_logical, num_physical,
        )

        # Cross-attention interaction
        C_prime, H_prime = self.cross_attention(C, H)

        # Score matrix
        S = self.score_head(C_prime, H_prime)  # (batch, l, h)

        # Sinkhorn with dummy padding
        P = self.sinkhorn(S, num_logical, num_physical, tau)  # (batch, h, h)

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
    ) -> list[dict[int, int]]:
        """Inference: forward pass + Hungarian decoding.

        Args:
            circuit_data: Batched PyG Data for circuit graphs.
            hardware_data: Batched PyG Data for hardware graphs.
            batch_size: Number of samples in batch.
            num_logical: Number of logical qubits per sample.
            num_physical: Number of physical qubits per sample.
            tau: Sinkhorn temperature.

        Returns:
            List of layout dicts {logical_qubit: physical_qubit}.
        """
        self.eval()
        P = self.forward(
            circuit_data, hardware_data, batch_size, num_logical, num_physical, tau,
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
        return cls(
            circuit_node_dim=cfg.model.circuit_gnn.node_input_dim,
            circuit_edge_dim=cfg.model.circuit_gnn.edge_input_dim,
            hardware_node_dim=cfg.model.hardware_gnn.node_input_dim,
            hardware_edge_dim=cfg.model.hardware_gnn.edge_input_dim,
            embedding_dim=cfg.model.embedding_dim,
            gnn_layers=cfg.model.circuit_gnn.num_layers,
            gnn_heads=cfg.model.circuit_gnn.num_heads,
            gnn_dropout=cfg.model.circuit_gnn.dropout,
            cross_attn_layers=cfg.model.cross_attention.num_layers,
            cross_attn_heads=cfg.model.cross_attention.num_heads,
            cross_attn_ffn_dim=cfg.model.cross_attention.ffn_hidden_dim,
            cross_attn_dropout=cfg.model.cross_attention.dropout,
            score_d_k=cfg.model.score_head.d_k,
            sinkhorn_max_iter=cfg.sinkhorn.max_iter,
            sinkhorn_tol=cfg.sinkhorn.tolerance,
        )

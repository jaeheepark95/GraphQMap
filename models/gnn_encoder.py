"""GATv2-based GNN Encoder for GraphQMap.

Dual independent GNN encoders for circuit and hardware graphs.
Architecture per GNN:
  Input → Linear(input_dim, d) →
  GATv2 Layer 1 → BatchNorm → ELU →
  GATv2 Layer 2 → BatchNorm → ELU → Residual →
  GATv2 Layer 3 → BatchNorm → ELU → Residual →
  Linear(d, d) → Final embedding
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class GNNEncoder(nn.Module):
    """GATv2-based graph neural network encoder.

    Args:
        node_input_dim: Dimension of input node features.
        edge_input_dim: Dimension of input edge features.
        embedding_dim: Hidden and output embedding dimension (d).
        num_layers: Number of GATv2 layers.
        num_heads: Number of attention heads per layer.
        dropout: Dropout rate.
        use_batchnorm: Whether to use BatchNorm after each layer.
        residual_layers: List of 1-indexed layer numbers that use residual connections.
    """

    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        residual_layers: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.residual_layers = set(residual_layers or [2, 3])
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim

        # Input projection
        self.input_proj = nn.Linear(node_input_dim, embedding_dim)

        # Edge feature projection for GATv2
        self.edge_proj = nn.Linear(edge_input_dim, embedding_dim)

        # GATv2 layers
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            # Each head outputs embedding_dim // num_heads dimensions
            # so concatenated output = embedding_dim
            head_dim = embedding_dim // num_heads
            gat = GATv2Conv(
                in_channels=embedding_dim,
                out_channels=head_dim,
                heads=num_heads,
                edge_dim=embedding_dim,
                dropout=dropout,
                concat=True,  # concatenate head outputs
            )
            self.gat_layers.append(gat)

            if use_batchnorm:
                self.batch_norms.append(nn.BatchNorm1d(embedding_dim))
            else:
                self.batch_norms.append(nn.Identity())

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features (N, node_input_dim).
            edge_index: Edge indices (2, E).
            edge_attr: Edge features (E, edge_input_dim).

        Returns:
            Node embeddings (N, embedding_dim).
        """
        # Slice input features to expected dim. Hardware/circuit graph builders
        # may produce more features than the model is configured for (e.g. after
        # adding new optional features); we honor node_input_dim/edge_input_dim
        # by taking the leading slice. Trailing features are dropped.
        if x.size(-1) > self.node_input_dim:
            x = x[:, : self.node_input_dim]
        if edge_attr.numel() > 0 and edge_attr.size(-1) > self.edge_input_dim:
            edge_attr = edge_attr[:, : self.edge_input_dim]

        # Project inputs to embedding space
        h = self.input_proj(x)
        edge_emb = self.edge_proj(edge_attr)

        # GATv2 layers
        for i in range(self.num_layers):
            h_in = h
            h = self.gat_layers[i](h, edge_index, edge_attr=edge_emb)
            h = self.batch_norms[i](h)
            h = self.activation(h)

            # Residual connection on specified layers (1-indexed)
            if (i + 1) in self.residual_layers:
                h = h + h_in

        # Output projection
        h = self.output_proj(h)
        return h

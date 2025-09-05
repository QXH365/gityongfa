# graph_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class GINEConv(MessagePassing):
    """
    Graph Isomorphism Network with Edge Features (GINE) convolution layer.
    (This class remains unchanged as its logic is correct)
    """
    def __init__(self, node_dim):
        super(GINEConv, self).__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2),
            nn.ReLU(),
            nn.Linear(node_dim * 2, node_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        message = F.relu(x_j + edge_attr)
        return message

    def update(self, aggr_out, x):
        return self.mlp(aggr_out + x)

class MolecularGraphEncoder(nn.Module):
    """
    A complete molecular graph encoder with dynamically sized embeddings.
    """
    def __init__(self, num_atom_types, num_edge_types, node_dim=128, num_layers=4):
        """
        Args:
            num_atom_types (int): The number of unique atom types in the dataset (max_atomic_number + 1).
            num_edge_types (int): The number of unique edge types in the dataset (max_edge_type + 1).
            node_dim (int): The hidden dimension size for node and edge features.
            num_layers (int): The number of GINEConv layers.
        """
        super().__init__()

        # --- FIX: Dynamic Embedding Sizes ---
        # The embedding size is now passed in, not hardcoded.
        self.atom_embedding = nn.Embedding(num_atom_types, node_dim)
        self.edge_embedding = nn.Embedding(num_edge_types, node_dim)
        # --- End of FIX ---

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GINEConv(node_dim))
            self.batch_norms.append(nn.BatchNorm1d(node_dim))

    def forward(self, atom_type, edge_index, edge_type):
        """
        Forward pass for the encoder.
        (This method remains unchanged as its logic is correct)
        """
        node_features = self.atom_embedding(atom_type)
        edge_features = self.edge_embedding(edge_type)

        for conv, bn in zip(self.convs, self.batch_norms):
            residual = node_features
            node_features = conv(node_features, edge_index, edge_features)
            node_features = bn(node_features)
            node_features = F.relu(node_features)
            node_features = node_features + residual

        return node_features
# denoising_network.py

import torch
import torch.nn as nn
import math

class GaussianSmearing(nn.Module):
    """
    Expands a scalar distance value into a vector representation using Gaussian basis functions.
    This is a standard technique for creating smooth and stable distance features.
    """
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        # dist shape: [E, 1]
        dist = dist.view(-1, 1) - self.offset.view(1, -1) # [E, num_gaussians]
        return torch.exp(self.coeff * torch.pow(dist, 2)) # [E, num_gaussians]
    
class SinusoidalTimestepEmbedding(nn.Module):
    """Module for creating sinusoidal embeddings for the diffusion timestep."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
class EGNNLayer(nn.Module):
    """A single layer of an E(n)-Equivariant Graph Neural Network with stable distance encoding."""
    def __init__(self, node_dim, hidden_dim=128, num_dist_basis=32):
        super().__init__()
        
        # --- FIX: Use Gaussian Smearing for distance ---
        self.distance_expansion = GaussianSmearing(stop=10.0, num_gaussians=num_dist_basis)
        
        # The input size of the message MLP is updated to reflect the smeared distance vector
        mlp_input_dim = node_dim * 2 + num_dist_basis
        # --- End of FIX ---

        self.message_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        self.coord_update_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h, pos, edge_index):
        row, col = edge_index
        
        rel_pos = pos[row] - pos[col]
        sq_dist = (rel_pos**2).sum(dim=1, keepdim=True)
        
        # --- FIX: Apply Gaussian Smearing to the distance ---
        dist_emb = self.distance_expansion(torch.sqrt(sq_dist + 1e-8)) # Add epsilon for sqrt stability
        
        # Use the smeared distance embedding instead of the raw scalar distance
        message_input = torch.cat([h[row], h[col], dist_emb], dim=1)
        # --- End of FIX ---

        messages = self.message_mlp(message_input)
        
        aggr_messages = torch.zeros(h.size(0), messages.size(1), device=h.device).index_add_(0, row, messages)
        
        coord_multipliers = self.coord_update_mlp(messages)
        coord_updates = rel_pos * coord_multipliers
        aggr_coord_updates = torch.zeros_like(pos).index_add_(0, row, coord_updates)
        pos = pos + aggr_coord_updates
        
        node_update_input = torch.cat([h, aggr_messages], dim=1)
        h_updates = self.node_update_mlp(node_update_input)
        h = h + h_updates
        
        return h, pos

class DenoisingNetwork(nn.Module):
    """The main denoising model, now with stable EGNN layers."""
    def __init__(self, node_dim, spec_dim, num_layers=6, final_out_dim=3):
        super().__init__()
        
        self.timestep_embedder = SinusoidalTimestepEmbedding(dim=node_dim)
        self.condition_mlp = nn.Sequential(
            nn.Linear(node_dim + spec_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        
        # --- FIX: Pass the correct dimension to the updated EGNNLayer ---
        self.egnn_layers = nn.ModuleList([EGNNLayer(node_dim=node_dim) for _ in range(num_layers)])
        # --- End of FIX ---

        self.output_head = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, final_out_dim)
        )

    def forward(self, noisy_pos, node_features, spec_condition, timestep, batch_map, edge_index):
        t_emb = self.timestep_embedder(timestep)
        combined_cond = torch.cat([spec_condition, t_emb], dim=1)
        final_cond = self.condition_mlp(combined_cond)
        node_cond = final_cond[batch_map]
        h = node_features + node_cond
        
        pos = noisy_pos
        for egnn_layer in self.egnn_layers:
            h, pos = egnn_layer(h, pos, edge_index)
            
        predicted_noise = self.output_head(h)
        return predicted_noise
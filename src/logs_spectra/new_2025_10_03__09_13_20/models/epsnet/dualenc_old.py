import torch
from torch import nn
import numpy as np
from tqdm.auto import tqdm

from ..common import (MultiLayerPerceptron, assemble_atom_pair_feature,
                      extend_graph_order_radius)
from ..encoder import SchNetEncoder, GINEncoder, get_edge_encoder
from ..geometry import get_distance, eq_transform

from .spectrum_encoder import SpectrumEncoder


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    # ... (This function remains unchanged) ...
    def sigmoid(x): return 1 / (np.exp(-x) + 1)
    if beta_schedule == "quad": betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == "linear": betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    else: raise NotImplementedError(beta_schedule)
    return betas

class SinusoidalPosEmb(nn.Module):
    # ... (This class remains unchanged) ...
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DualEncoderEpsNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.spectrum_encoder = SpectrumEncoder(config)

        time_dim = config.hidden_dim * 4
        self.timestep_embedder = nn.Sequential(
            SinusoidalPosEmb(config.hidden_dim),
            nn.Linear(config.hidden_dim, time_dim), nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.condition_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim + time_dim, time_dim), nn.GELU(),
            nn.Linear(time_dim, config.hidden_dim)
        )

        self.edge_encoder_global = get_edge_encoder(config)
        self.edge_encoder_local = get_edge_encoder(config)

        self.encoder_global = SchNetEncoder(
            hidden_channels=config.hidden_dim, num_filters=config.hidden_dim,
            num_interactions=config.num_convs, edge_channels=self.edge_encoder_global.out_channels,
            cutoff=config.cutoff, smooth=config.smooth_conv,
        )
        self.encoder_local = GINEncoder(
            hidden_dim=config.hidden_dim, num_convs=config.num_convs_local,
        )

        self.grad_global_dist_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1],
            activation=config.mlp_act
        )
        self.grad_local_dist_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1],
            activation=config.mlp_act
        )
        
        betas = get_beta_schedule(
            beta_schedule=config.beta_schedule, beta_start=config.beta_start,
            beta_end=config.beta_end, num_diffusion_timesteps=config.num_diffusion_timesteps,
        )
        betas = torch.from_numpy(betas).float()
        self.betas = nn.Parameter(betas, requires_grad=False)
        alphas = (1. - betas).cumprod(dim=0)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.num_timesteps = self.betas.size(0)

    def forward(self, batch, pos_perturbed, time_step):
        # ... (This method remains unchanged from the previous version) ...
        atom_type, bond_index, bond_type, batch_idx = batch.atom_type, batch.edge_index, batch.edge_type, batch.batch
        spec_condition = self.spectrum_encoder(batch)
        time_condition = self.timestep_embedder(time_step)
        combined_cond = torch.cat([spec_condition, time_condition], dim=1)
        final_cond = self.condition_mlp(combined_cond)
        node_condition = final_cond[batch_idx]
        edge_index, edge_type = extend_graph_order_radius(
            num_nodes=atom_type.size(0), pos=pos_perturbed, edge_index=bond_index,
            edge_type=bond_type, batch=batch_idx, order=self.config.edge_order,
            cutoff=self.config.cutoff,
        )
        edge_length = get_distance(pos_perturbed, edge_index).unsqueeze(-1)
        local_edge_mask = (edge_type > 0)
        
        edge_attr_global = self.edge_encoder_global(edge_length=edge_length, edge_type=edge_type)
        node_attr_global = self.encoder_global(z=atom_type, edge_index=edge_index, edge_length=edge_length, edge_attr=edge_attr_global)
        node_attr_global = node_attr_global + node_condition
        h_pair_global = assemble_atom_pair_feature(node_attr=node_attr_global, edge_index=edge_index, edge_attr=edge_attr_global)
        edge_inv_global = self.grad_global_dist_mlp(h_pair_global)

        edge_attr_local = self.edge_encoder_local(edge_length=edge_length, edge_type=edge_type)
        node_attr_local = self.encoder_local(z=atom_type, edge_index=edge_index[:, local_edge_mask], edge_attr=edge_attr_local[local_edge_mask])
        node_attr_local = node_attr_local + node_condition
        h_pair_local = assemble_atom_pair_feature(node_attr=node_attr_local, edge_index=edge_index[:, local_edge_mask], edge_attr=edge_attr_local[local_edge_mask])
        edge_inv_local = self.grad_local_dist_mlp(h_pair_local)

        return edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask

    def get_loss(self, batch, anneal_power=2.0):
        """
        **Corrected Loss Function**
        Calculates the diffusion loss based on positional gradients, as in original GeoDiff.
        """
        pos_true, node2graph, num_graphs = batch.pos, batch.batch, batch.num_graphs
        
        time_step = torch.randint(0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=pos_true.device)
        time_step = torch.cat([time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        a = self.alphas.index_select(0, time_step)
        
        a_pos = a.index_select(0, node2graph).unsqueeze(-1)
        pos_noise = torch.randn_like(pos_true)
        pos_perturbed = pos_true * a_pos.sqrt() + pos_noise * (1.0 - a_pos).sqrt()
        
        # Get model predictions
        edge_inv_global, edge_inv_local, edge_index, _, edge_length, local_edge_mask = self.forward(
            batch=batch, pos_perturbed=pos_perturbed, time_step=time_step
        )
        
        # True score is the noise scaled by the variance
        target_pos_noise = -pos_noise / (1.0 - a_pos).sqrt()

        # Convert edge-wise predictions to node-wise gradients (scores)
        node_eq_global = eq_transform(edge_inv_global, pos_perturbed, edge_index, edge_length)
        node_eq_local = eq_transform(edge_inv_local, pos_perturbed, edge_index[:, local_edge_mask], edge_length[local_edge_mask])

        # Calculate loss on the positional gradients
        loss_global = (node_eq_global - target_pos_noise)**2
        loss_global = torch.sum(loss_global, dim=-1, keepdim=True)
        
        loss_local = (node_eq_local - target_pos_noise)**2
        loss_local = torch.sum(loss_local, dim=-1, keepdim=True)
        
        loss_pos = loss_global + loss_local
        
        # Apply annealing weight
        loss = loss_pos * (a_pos.squeeze(-1) ** anneal_power)

        return loss.mean(), loss_global.mean(), loss_local.mean()

    def langevin_dynamics_sample(self, batch, n_steps, eta=1.0):
        # ... (This is a simplified sampling loop for demonstration, can be replaced with the one in test.py) ...
        # A full implementation should follow the logic in test.py for DDIM/DDPM sampling
        pos = torch.randn_like(batch.pos)
        pos_traj = []
        
        with torch.no_grad():
            skip = self.num_timesteps // n_steps
            seq = range(0, self.num_timesteps, skip)
            
            for i in tqdm(reversed(seq), desc="Sampling", total=len(seq)):
                t = torch.full(size=(batch.num_graphs,), fill_value=i, dtype=torch.long, device=pos.device)
                
                # Predict score
                pred = self.forward(batch, pos, t)
                edge_inv_global, edge_inv_local, edge_index, _, edge_length, local_edge_mask = pred
                
                # Convert to positional gradients
                score_pos_local = eq_transform(edge_inv_local, pos, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
                score_pos_global = eq_transform(edge_inv_global, pos, edge_index, edge_length)
                score_pos = score_pos_local + score_pos_global
                
                # DDPM update step
                # ... (A more complex DDPM/DDIM sampler from test.py should be used here) ...
                # This is a simplified placeholder
                alpha = self.alphas[i]
                alpha_bar_prev = self.alphas[i-skip] if i-skip >= 0 else torch.tensor(1.0)
                
                noise = torch.randn_like(pos) if i > 0 else 0.
                pos = 1. / torch.sqrt(alpha) * (pos - (1-alpha)/torch.sqrt(1-self.alphas[i]) * score_pos) + torch.sqrt(1-alpha) * noise
                
                pos_traj.append(pos.cpu())
                
        return pos, pos_traj
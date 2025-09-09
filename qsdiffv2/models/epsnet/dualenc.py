import torch
from torch import nn
from torch_scatter import scatter_add, scatter_mean
from torch_scatter import scatter
from torch_geometric.data import Data, Batch
import numpy as np
from tqdm.auto import tqdm
from utils.chem import BOND_TYPES
from ..common import (MultiLayerPerceptron, assemble_atom_pair_feature,
                      extend_graph_order_radius)
from ..encoder import SchNetEncoder, GINEncoder, get_edge_encoder
from ..geometry import get_distance, eq_transform
from .spectrum_encoder import SpectrumEncoder
from easydict import EasyDict

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    # ... (Unchanged) ...
    def sigmoid(x): return 1 / (np.exp(-x) + 1)
    if beta_schedule == "quad": betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == "linear": betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    else: raise NotImplementedError(beta_schedule)
    return betas

class SinusoidalPosEmb(nn.Module):
    # ... (Unchanged) ...
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
        # ... (Initialization is unchanged from the previous correct version) ...
        super().__init__()
        self.config = config
        # The config object passed to SpectrumEncoder is already config
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
        edge_encoder_config = EasyDict({
            'edge_encoder': config.edge_encoder,
            'hidden_dim': config.hidden_dim,
            'mlp_act': config.mlp_act
        })
        self.edge_encoder_global = get_edge_encoder(edge_encoder_config)
        self.edge_encoder_local = get_edge_encoder(edge_encoder_config)
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
        # ... (Forward pass is unchanged from the previous correct version) ...
        atom_type, bond_index, bond_type, batch_idx = batch.atom_type, batch.bond_edge_index, batch.bond_edge_type, batch.batch
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
        # ... (Loss function is unchanged from the previous correct version) ...
        pos_true, node2graph, num_graphs = batch.pos, batch.batch, batch.num_graphs
        time_step = torch.randint(0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=pos_true.device)
        time_step = torch.cat([time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        a = self.alphas.index_select(0, time_step)
        a_pos = a.index_select(0, node2graph).unsqueeze(-1)
        pos_noise = torch.randn_like(pos_true)
        pos_perturbed = pos_true * a_pos.sqrt() + pos_noise * (1.0 - a_pos).sqrt()
        edge_inv_global, edge_inv_local, edge_index, _, edge_length, local_edge_mask = self.forward(
            batch=batch, pos_perturbed=pos_perturbed, time_step=time_step
        )
        target_pos_noise = -pos_noise / (1.0 - a_pos).sqrt()
        node_eq_global = eq_transform(edge_inv_global, pos_perturbed, edge_index, edge_length)
        node_eq_local = eq_transform(edge_inv_local, pos_perturbed, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
        pred_pos_noise = node_eq_global + node_eq_local
        loss_pos = (pred_pos_noise - target_pos_noise)**2
        loss_pos = torch.sum(loss_pos, dim=-1, keepdim=True)
        loss_global = torch.sum((node_eq_global - target_pos_noise)**2, dim=-1, keepdim=True)
        loss_local = torch.sum((node_eq_local - target_pos_noise)**2, dim=-1, keepdim=True)
        loss = loss_pos * (a_pos.squeeze(-1) ** anneal_power)
        return loss.mean(), loss_global.mean(), loss_local.mean()

    # in models/epsnet/dualenc.py -> DualEncoderEpsNetwork class

    def compute_alpha(self, beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        idx = torch.clamp(t + 1, 0, beta.size(0) - 1)
        a = (1 - beta).cumprod(dim=0).index_select(0, idx)
        return a

    @torch.no_grad()
    def langevin_dynamics_sample(
        self, batch, n_steps, eta=1.0, w_global=0.5,
        global_start_sigma=float('inf'), clip=1000.0, clip_local=None
    ):
        # 1. Initialization: Start from pure Gaussian noise
        pos = torch.randn_like(batch.pos)
        pos_traj = []
        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()

        # 2. Define the reverse diffusion timesteps
        # This matches the original's logic for n_steps
        seq = range(self.num_timesteps - n_steps, self.num_timesteps)
        seq_next = [-1] + list(seq[:-1])

        # 3. The main sampling loop
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc="Sampling", total=len(seq), leave=False):
            t = torch.full(size=(batch.num_graphs,), fill_value=i, dtype=torch.long, device=pos.device)

            # 4. Get the score prediction from our conditioned forward pass
            pred = self.forward(batch, pos, t)
            edge_inv_global, edge_inv_local, edge_index, _, edge_length, local_edge_mask = pred
            
            # 5. Convert edge scores to node-wise gradients (forces)
            node_eq_local = eq_transform(edge_inv_local, pos, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
            if clip_local is not None:
                node_eq_local = clip_norm(node_eq_local, limit=clip_local)

            if sigmas[i] < global_start_sigma:
                non_local_mask = (1.0 - local_edge_mask.view(-1, 1).float())
                edge_inv_global = edge_inv_global * non_local_mask
                node_eq_global = eq_transform(edge_inv_global, pos, edge_index, edge_length)
                node_eq_global = clip_norm(node_eq_global, limit=clip)
            else:
                node_eq_global = 0
            
            # This is the model's prediction for the noise, eps_theta
            eps_pos = node_eq_local + node_eq_global * w_global

            # 6. DDPM Update Step
            # This section meticulously follows the 'ddpm_noisy' logic from the original code
            t_torch = torch.full((1,), i, device=pos.device)
            next_t_torch = torch.full((1,), j, device=pos.device)

            at = self.compute_alpha(self.betas, t_torch)
            at_next = self.compute_alpha(self.betas, next_t_torch)
            
            # The original code uses `e = -eps_pos`, let's be explicit
            e = -eps_pos
            
            # Predict x0 from xt and the predicted noise `e`
            pos0_from_e = (1.0 / at).sqrt() * pos - (1.0 / at - 1).sqrt() * e
            
            # Calculate the mean of the posterior q(x_{t-1} | xt, x0)
            # This is the core of the DDPM update step
            beta_t = 1 - at / at_next
            mean_eps = (
                (at_next.sqrt() * beta_t) * pos0_from_e + ((1 - beta_t).sqrt() * (1 - at_next)) * pos
            ) / (1.0 - at)

            mean = mean_eps
            # Add noise only if not the last step (t > 0)
            mask = (1 - (t_torch == 0).float()).squeeze()
            logvar = beta_t.log()
            
            noise = torch.randn_like(pos) if j >= 0 else 0.
            pos_next = mean + mask * torch.exp(0.5 * logvar) * noise
            
            pos = pos_next
            pos = center_pos(pos, batch.batch)
            pos_traj.append(pos.clone().cpu())
            
        return pos, pos_traj


def is_bond(edge_type):
    return torch.logical_and(edge_type < len(BOND_TYPES), edge_type > 0)


def is_angle_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 1 - 1


def is_dihedral_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 2 - 1


def is_radius_edge(edge_type):
    return edge_type == 0


def is_local_edge(edge_type):
    return edge_type > 0


def is_train_edge(edge_index, is_sidechain):
    if is_sidechain is None:
        return torch.ones(edge_index.size(1), device=edge_index.device).bool()
    else:
        is_sidechain = is_sidechain.bool()
        return torch.logical_or(is_sidechain[edge_index[0]], is_sidechain[edge_index[1]])


def regularize_bond_length(edge_type, edge_length, rng=5.0):
    mask = is_bond(edge_type).float().reshape(-1, 1)
    d = -torch.clamp(edge_length - rng, min=0.0, max=float('inf')) * mask
    return d


def center_pos(pos, batch):
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center


def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom

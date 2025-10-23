# diffusion/models/epsnet/dualenc_v3.py

import torch
from torch import nn
from torch_scatter import scatter_add, scatter_mean, scatter
from torch_geometric.data import Data, Batch
import numpy as np
from tqdm.auto import tqdm
import math

from utils.chem import BOND_TYPES
from ..common import (MultiLayerPerceptron, assemble_atom_pair_feature,
                      extend_graph_order_radius)
from ..encoder import SchNetEncoder, GINEncoder, get_edge_encoder
# !! 导入 geometry 中的新函数 !!
from ..geometry import get_distance, eq_transform, get_dihedral, wrap_angles, compute_dihedral_gradients
# !! 导入解耦光谱编码器 !!
from .spectrum_encoder import SpectrumEncoderDisentangled, kl_divergence_gaussian
from easydict import EasyDict
import torch.nn.functional as F
# --- FiLM 层定义 (保持不变) ---
class FiLMLayer(nn.Module):
    def __init__(self, feature_dim, condition_dim):
        super().__init__()
        hidden_cond_dim = (condition_dim + feature_dim * 2) // 2
        self.condition_projector = nn.Sequential(
            nn.Linear(condition_dim, hidden_cond_dim),
            nn.SiLU(),
            nn.Linear(hidden_cond_dim, feature_dim * 2),
        )
    def forward(self, x, c):
        gamma_beta = self.condition_projector(c)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return gamma * x + beta

# --- get_beta_schedule 和 SinusoidalPosEmb (保持不变) ---
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x): return 1 / (np.exp(-x) + 1)
    if beta_schedule == "quad": betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == "linear": betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else: raise NotImplementedError(beta_schedule)
    return betas

class SinusoidalPosEmb(nn.Module):
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

# --- 主模型 ---
class DualEncoderEpsNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 光谱编码器
        self.spectrum_encoder = SpectrumEncoderDisentangled(config)
        spec_embed_dim = config.model.get('spec_embed_dim', 128)
        gnn_hidden_dim = config.model.hidden_dim

        # 时间嵌入器
        time_dim = config.model.hidden_dim * 4
        self.timestep_embedder = nn.Sequential(
            SinusoidalPosEmb(config.model.hidden_dim),
            nn.Linear(config.model.hidden_dim, time_dim), nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # 分子统计量编码器
        self.mol_stats_dim = config.model.get('mol_stats_dim', 32)
        self.mol_stats_encoder = nn.Sequential(
            nn.Linear(1, self.mol_stats_dim), nn.SiLU(),
            nn.Linear(self.mol_stats_dim, self.mol_stats_dim)
        )

        # FiLM 层
        film_condition_dim = time_dim + self.mol_stats_dim
        self.film1 = FiLMLayer(feature_dim=spec_embed_dim, condition_dim=film_condition_dim)
        self.film2 = FiLMLayer(feature_dim=spec_embed_dim, condition_dim=film_condition_dim)

        # 交叉注意力模块
        num_heads = config.model.get('num_attn_heads', 8)
        self.cross_attention_global = nn.MultiheadAttention(
            embed_dim=gnn_hidden_dim, kdim=spec_embed_dim, vdim=spec_embed_dim,
            num_heads=num_heads, batch_first=True
        )
        self.norm_global = nn.LayerNorm(gnn_hidden_dim)
        self.cross_attention_local = nn.MultiheadAttention(
            embed_dim=gnn_hidden_dim, kdim=spec_embed_dim, vdim=spec_embed_dim,
            num_heads=num_heads, batch_first=True
        )
        self.norm_local = nn.LayerNorm(gnn_hidden_dim)

        # GNN 编码器
        edge_encoder_config = EasyDict({
            'edge_encoder': config.model.edge_encoder, 'hidden_dim': gnn_hidden_dim, 'mlp_act': config.model.mlp_act
        })
        self.edge_encoder_global = get_edge_encoder(edge_encoder_config)
        self.edge_encoder_local = get_edge_encoder(edge_encoder_config)
        self.encoder_global = SchNetEncoder(
            hidden_channels=gnn_hidden_dim, num_filters=gnn_hidden_dim, num_interactions=config.model.num_convs,
            edge_channels=self.edge_encoder_global.out_channels, cutoff=config.model.cutoff, smooth=config.model.smooth_conv,
        )
        self.encoder_local = GINEncoder(
            hidden_dim=gnn_hidden_dim, num_convs=config.model.num_convs_local, activation=config.model.mlp_act # Pass activation
        )

        # 主要 MLP 预测头
        self.grad_global_dist_mlp = MultiLayerPerceptron(
            gnn_hidden_dim + self.edge_encoder_global.out_channels, # 输入包含节点对特征和边特征
            [gnn_hidden_dim, gnn_hidden_dim // 2, 1], activation=config.model.mlp_act
        )
        self.grad_local_dist_mlp = MultiLayerPerceptron(
            gnn_hidden_dim + self.edge_encoder_local.out_channels, # 输入包含节点对特征和边特征
            [gnn_hidden_dim, gnn_hidden_dim // 2, 1], activation=config.model.mlp_act
        )

        # --- 新增：二面角预测头 ---
        dihedral_head_input_dim = (4 * gnn_hidden_dim + spec_embed_dim + film_condition_dim)
        dihedral_head_hidden_dims = config.model.get('dihedral_head_hidden_dims', [128, 64])
        self.dihedral_correction_head = MultiLayerPerceptron(
            input_dim=dihedral_head_input_dim, hidden_dims=dihedral_head_hidden_dims + [1],
            activation=config.model.mlp_act
        )
        # (可选) z2 聚合方式，先用平均池化
        self.pool_z2_for_dihedral = lambda z: z.mean(dim=1)

        # 扩散参数
        betas = get_beta_schedule(
            beta_schedule=config.model.beta_schedule, beta_start=config.model.beta_start,
            beta_end=config.model.beta_end, num_diffusion_timesteps=config.model.num_diffusion_timesteps,
        )
        betas = torch.from_numpy(betas).float()
        self.betas = nn.Parameter(betas, requires_grad=False)
        alphas = (1. - betas).cumprod(dim=0)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.num_timesteps = self.betas.size(0)

        # 损失权重
        self.kl_loss_weight = config.train.get('kl_loss_weight', 0.01)
        self.dihedral_loss_weight = config.train.get('dihedral_loss_weight', 0.1)
        self.dihedral_weight_train = config.train.get('dihedral_weight_train', 0.5)
        self.dihedral_weight_sample = config.model.get('dihedral_weight_sample',
                                             config.sampling.get('dihedral_weight_sample', 0.5) if hasattr(config, 'sampling') else 0.5)


    def forward(self, batch, pos_perturbed, time_step):
        atom_type, bond_index, bond_type, batch_idx = batch.atom_type, batch.bond_edge_index, batch.bond_edge_type, batch.batch
        num_graphs = batch.num_graphs

        # 1. 光谱编码 + KL损失
        z1, z2, kl_separation_loss = self.spectrum_encoder(batch)

        # 2. 时间条件
        time_condition = self.timestep_embedder(time_step)

        # 3. 分子统计量条件
        num_nodes_per_graph = batch.ptr[1:] - batch.ptr[:-1]
        mol_stats_input = torch.log1p(num_nodes_per_graph.float().unsqueeze(-1))
        mol_stats_embedding = self.mol_stats_encoder(mol_stats_input)
        combined_film_condition = torch.cat([time_condition, mol_stats_embedding], dim=-1)

        # 4. FiLM 调制光谱特征
        modulated_z1 = self.film1(z1, combined_film_condition)
        modulated_z2 = self.film2(z2, combined_film_condition)

        # 5. 准备交叉注意力上下文
        context_sequence = torch.cat([modulated_z1, modulated_z2], dim=1)
        context_expanded = context_sequence[batch_idx]

        # 6. 构建图结构
        edge_index, edge_type = extend_graph_order_radius(
            num_nodes=atom_type.size(0), pos=pos_perturbed, edge_index=bond_index,
            edge_type=bond_type, batch=batch_idx, order=self.config.model.edge_order,
            cutoff=self.config.model.cutoff,
        )
        edge_length = get_distance(pos_perturbed, edge_index).unsqueeze(-1)
        local_edge_mask = (edge_type > 0)
        sigma_edge = torch.ones(size=(edge_index.size(1), 1), device=pos_perturbed.device)

        # 7. GNN + 交叉注意力 + 主要 MLP 预测
        # Global
        edge_attr_global = self.edge_encoder_global(edge_length=edge_length, edge_type=edge_type)
        node_attr_global = self.encoder_global(z=atom_type, edge_index=edge_index, edge_length=edge_length, edge_attr=edge_attr_global)
        attn_output_global, _ = self.cross_attention_global(query=node_attr_global.unsqueeze(1), key=context_expanded, value=context_expanded)
        attn_output_global = attn_output_global.squeeze(1)
        node_attr_global = self.norm_global(node_attr_global + attn_output_global)
        h_pair_global = assemble_atom_pair_feature(node_attr=node_attr_global, edge_index=edge_index, edge_attr=edge_attr_global)
        edge_inv_global = self.grad_global_dist_mlp(h_pair_global) * (1.0 / sigma_edge)

        # Local
        edge_attr_local = self.edge_encoder_local(edge_length=edge_length[local_edge_mask], edge_type=edge_type[local_edge_mask])
        # GINEncoder 可能需要 edge_attr 的维度匹配 hidden_dim，如果 get_edge_encoder 返回不同维度，可能需要调整
        # 假设 edge_encoder_local.out_channels == gnn_hidden_dim
        node_attr_local = self.encoder_local(z=atom_type, edge_index=edge_index[:, local_edge_mask], edge_attr=edge_attr_local)
        attn_output_local, _ = self.cross_attention_local(query=node_attr_local.unsqueeze(1), key=context_expanded, value=context_expanded)
        attn_output_local = attn_output_local.squeeze(1)
        node_attr_local = self.norm_local(node_attr_local + attn_output_local)
        # 重新组装 h_pair_local 时需要使用正确的 edge_attr_local
        h_pair_local = assemble_atom_pair_feature(node_attr=node_attr_local, edge_index=edge_index[:, local_edge_mask], edge_attr=edge_attr_local)
        edge_inv_local = self.grad_local_dist_mlp(h_pair_local) * (1.0 / sigma_edge[local_edge_mask])


        # 8. 二面角修正头预测
        pred_dih_correction = None
        dihedral_index = getattr(batch, 'dihedral_index', None)

        if dihedral_index is not None and dihedral_index.numel() > 0:
            num_dihedrals = dihedral_index.shape[1]
            idx_i, idx_j, idx_k, idx_l = dihedral_index[0], dihedral_index[1], dihedral_index[2], dihedral_index[3]

            # !! 使用融合后的 node_attr_global 作为原子特征 !!
            feat_i, feat_j, feat_k, feat_l = node_attr_global[idx_i], node_attr_global[idx_j], node_attr_global[idx_k], node_attr_global[idx_l]
            node_features_dih = torch.cat([feat_i, feat_j, feat_k, feat_l], dim=-1)

            dihedral_graph_idx = batch_idx[idx_i]
            dih_z2_context = modulated_z2[dihedral_graph_idx]
            pooled_dih_z2 = self.pool_z2_for_dihedral(dih_z2_context)
            dih_condition = combined_film_condition[dihedral_graph_idx]

            head_input = torch.cat([node_features_dih, pooled_dih_z2, dih_condition], dim=-1)
            pred_dih_correction = self.dihedral_correction_head(head_input)

        return (edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length,
                local_edge_mask, kl_separation_loss, pred_dih_correction, dihedral_index,
                node_attr_global, node_attr_local, combined_film_condition, modulated_z2 # 返回用于计算梯度的中间结果
               )

    def get_loss(self, batch, anneal_power=2.0):
        pos_true, node2graph, num_graphs = batch.pos, batch.batch, batch.num_graphs
        time_step = torch.randint(0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=pos_true.device)
        time_step = torch.cat([time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        a = self.alphas.index_select(0, time_step)
        a_pos = a.index_select(0, node2graph).unsqueeze(-1)
        pos_noise = torch.randn_like(pos_true)
        # VP-SDE Noise Addition
        sigma_t = ((1.0 - a_pos) / (a_pos + 1e-8)).sqrt()
        pos_perturbed = pos_true + pos_noise * sigma_t

        # Call forward
        (edge_inv_global, edge_inv_local, edge_index, _, edge_length,
         local_edge_mask, kl_separation_loss, pred_dih_correction, dihedral_index,
         node_attr_global, _, _, _) = self.forward( # 忽略 forward 中不需要的返回值
            batch=batch, pos_perturbed=pos_perturbed, time_step=time_step
        )

        # --- Dihedral Loss ---
        loss_dih = torch.tensor(0.0, device=pos_true.device)
        node_eq_dih_correction = torch.zeros_like(pos_perturbed) # Initialize gradient contribution

        if dihedral_index is not None and dihedral_index.numel() > 0 and pred_dih_correction is not None:
            true_dihedrals = get_dihedral(pos_true, dihedral_index)
            perturbed_dihedrals = get_dihedral(pos_perturbed, dihedral_index)
            # Check for NaNs immediately after calculation
            if torch.isnan(true_dihedrals).any() or torch.isnan(perturbed_dihedrals).any():
                print(f"!!! NaN detected in get_dihedral output !!!")
                print(f"pos_true stats: min={pos_true.min()}, max={pos_true.max()}")
                print(f"pos_perturbed stats: min={pos_perturbed.min()}, max={pos_perturbed.max()}")
                # You might want to save the problematic batch here for offline analysis
                # torch.save(batch.to_data_list(), "nan_batch.pt")

            diff = true_dihedrals - perturbed_dihedrals
            target_dih_correction_wrapped = wrap_angles(diff)

            print(f"Iter {time_step[0].item() if time_step.numel() > 0 else 'N/A'}: Target Dih Range: [{target_dih_correction_wrapped.min():.4f}, {target_dih_correction_wrapped.max():.4f}]")
            print(f"Iter {time_step[0].item() if time_step.numel() > 0 else 'N/A'}: Pred Dih Range: [{pred_dih_correction.min():.4f}, {pred_dih_correction.max():.4f}]")

            loss_dih = F.mse_loss(pred_dih_correction, target_dih_correction_wrapped)
            print(f"Iter {time_step[0].item() if time_step.numel() > 0 else 'N/A'}: Loss Dih: {loss_dih.item()}")

            # --- Compute and Aggregate Dihedral Gradient Contribution ---
            # 使用自动梯度计算，需要确保 pos_perturbed 有梯度
            pos_perturbed_grad = pos_perturbed.detach().clone().requires_grad_(True)
            # 重新计算前向传播中需要梯度的部分（主要是二面角计算和梯度本身）
            with torch.enable_grad():
                current_dihedrals_for_grad = get_dihedral(pos_perturbed_grad, dihedral_index)
                # 计算梯度 d(phi)/d(pos_perturbed)
                # 对每个二面角分别计算梯度可能更精确，但计算量大
                # 这里采用对总和求梯度，然后提取的方式
                total_angle = current_dihedrals_for_grad.sum()
                grads_all_atoms = torch.autograd.grad(total_angle, pos_perturbed_grad, create_graph=False)[0] # 不创建图，因为我们只需要梯度值
                grads_all_atoms = torch.nan_to_num(grads_all_atoms)

            # 提取每个二面角对其四个原子的梯度贡献
            idx_i, idx_j, idx_k, idx_l = dihedral_index[0], dihedral_index[1], dihedral_index[2], dihedral_index[3]
            grad_p_i = grads_all_atoms[idx_i]
            grad_p_j = grads_all_atoms[idx_j]
            grad_p_k = grads_all_atoms[idx_k]
            grad_p_l = grads_all_atoms[idx_l]
            grad_phi = torch.stack([grad_p_i, grad_p_j, grad_p_k, grad_p_l], dim=1) # [NumDih, 4, 3]

            # 计算梯度贡献
            pos_correction_grad = grad_phi * pred_dih_correction.detach().unsqueeze(-1) # Detach prediction,梯度只通过 loss_dih 传回 head

            # 聚合到原子
            indices = dihedral_index.T.reshape(-1)
            grads_flat = pos_correction_grad.reshape(-1, 3)
            node_eq_dih_correction = scatter_add(grads_flat, indices, dim=0, dim_size=pos_perturbed.size(0))

        # --- Diffusion Loss ---
        # Main score components
        node_eq_global = eq_transform(edge_inv_global, pos_perturbed, edge_index, edge_length)
        node_eq_local = eq_transform(edge_inv_local, pos_perturbed, edge_index[:, local_edge_mask], edge_length[local_edge_mask])

        # Target score for VP-SDE: -noise / sigma_t
        target_pos_score = -pos_noise / (sigma_t + 1e-8)

        # Combined predicted score (with dihedral correction)
        pred_total_pos_score = node_eq_global + node_eq_local + self.dihedral_weight_train * node_eq_dih_correction

        # Calculate diffusion loss (comparing scores)
        diffusion_loss_per_node = torch.sum((pred_total_pos_score - target_pos_score)**2, dim=-1) # (N,)
        diffusion_loss = scatter_mean(diffusion_loss_per_node, node2graph, dim=0).mean() # Mean over nodes, then mean over batch

        # --- Total Loss ---
        total_loss = diffusion_loss + self.kl_loss_weight * kl_separation_loss + self.dihedral_loss_weight * loss_dih

        # 返回各项损失
        return total_loss, diffusion_loss, loss_dih, kl_separation_loss

    # --- _get_score_pred for sampling ---
    def _get_score_pred(self, batch, pos, t_tensor, w_global, global_start_sigma, clip, clip_local):
        sigmas = (1.0 - self.alphas).sqrt() / (self.alphas.sqrt() + 1e-8)

        # Call forward, ignore kl_loss
        (edge_inv_global, edge_inv_local, edge_index, _, edge_length,
         local_edge_mask, _, pred_dih_correction, dihedral_index,
         _, _, _, _) = self.forward( # Ignore kl_loss and intermediate features
            batch=batch, pos_perturbed=pos, time_step=t_tensor
        )

        # Calculate main score components
        node_eq_local = torch.zeros_like(pos)
        if local_edge_mask.sum() > 0:
            node_eq_local = eq_transform(edge_inv_local, pos, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
            if clip_local is not None: node_eq_local = clip_norm(node_eq_local, limit=clip_local)

        node_eq_global = torch.zeros_like(pos)
        current_sigma_val = sigmas[t_tensor[0].item()] if t_tensor[0].item() < len(sigmas) else sigmas[-1]
        if current_sigma_val < global_start_sigma:
            non_local_mask = ~local_edge_mask
            if non_local_mask.sum() > 0:
                node_eq_global = eq_transform(edge_inv_global[non_local_mask], pos, edge_index[:, non_local_mask], edge_length[non_local_mask])
                node_eq_global = clip_norm(node_eq_global, limit=clip)

        # Calculate dihedral correction gradient contribution
        node_eq_dih_correction = torch.zeros_like(pos)
        if dihedral_index is not None and dihedral_index.numel() > 0 and pred_dih_correction is not None:
             grad_phi = compute_dihedral_gradients(pos, dihedral_index) # Gradient at current noisy position
             pos_correction_grad = grad_phi * pred_dih_correction.unsqueeze(-1)
             indices = dihedral_index.T.reshape(-1)
             grads_flat = pos_correction_grad.reshape(-1, 3)
             node_eq_dih_correction = scatter_add(grads_flat, indices, dim=0, dim_size=pos.size(0))

        # Combine final score for sampling
        score_pred = node_eq_local + w_global * node_eq_global + self.dihedral_weight_sample * node_eq_dih_correction

        return score_pred

    # --- Other methods (compute_alpha, sampling, helpers) ---
    def compute_alpha(self, beta, t):
        # (保持不变)
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        idx = torch.clamp(t, 0, beta.size(0) - 1)
        a = (1 - beta).cumprod(dim=0).index_select(0, idx)
        return a

    @torch.no_grad()
    def langevin_dynamics_sample(
        self, batch, w_global=0.5, global_start_sigma=float('inf'),
        clip=10.0, clip_local=5.0, temperature=1.0, debug=False, n_steps=None # 添加 n_steps
    ):
        device = self.betas.device
        batch_size = batch.num_graphs
        pos = torch.randn_like(batch.pos) * temperature
        pos = center_pos(pos, batch.batch)
        pos_traj = []

        if n_steps is None:
            n_steps = self.num_timesteps
            timesteps = list(range(self.num_timesteps))[::-1]
            skip_type = 'none'
        else:
             # Use linspace for skipping steps if n_steps < num_timesteps
             timesteps = torch.linspace(self.num_timesteps - 1, 0, n_steps, device=device).long().tolist()
             skip_type = 'linspace'

        alphas_cumprod = self.alphas

        for i, t in enumerate(tqdm(timesteps, desc="Sampling", leave=False)):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)
            # score_pred 内部调用 forward 并整合二面角梯度
            score_pred = self._get_score_pred(batch, pos, t_tensor, w_global, global_start_sigma, clip, clip_local)

            beta_t = self.betas[t]
            alpha_t = 1. - beta_t
            pos_mean = (1. / alpha_t.sqrt()) * (pos + beta_t * score_pred) # Drift term based on VP SDE solution

            if i < len(timesteps) - 1:
                # Get alpha_cumprod_prev based on skip type
                if skip_type == 'linspace':
                    t_prev = timesteps[i+1]
                else: # skip_type == 'none'
                    t_prev = t-1
                alpha_cumprod_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
                alpha_cumprod_t = alphas_cumprod[t]

                posterior_variance = (1. - alpha_cumprod_prev) / (1. - alpha_cumprod_t) * beta_t
                noise = torch.randn_like(pos) * temperature
                pos = pos_mean + torch.sqrt(posterior_variance.clamp(min=1e-8)) * noise # Add noise (diffusion term)
            else:
                pos = pos_mean # Last step has no noise

            pos = center_pos(pos, batch.batch)
            if torch.isnan(pos).any():
                print(f"NaN detected at step {i}, t={t}. Stopping.")
                break # Early stopping if NaN occurs
            pos_traj.append(pos.clone().cpu())
            if debug and i % (len(timesteps) // 10) == 0:
                 print(f"Step {i}/{len(timesteps)}, t={t}, pos_std={pos.std():.4f}, score_norm={score_pred.norm():.4f}")

        return pos, pos_traj

    @torch.no_grad()
    def langevin_dynamics_sample_ode(
        self, batch, n_steps=100, w_global=0.5,
        global_start_sigma=float('inf'), clip=10.0, clip_local=5.0,
    ):
        device = self.betas.device
        batch_size = batch.num_graphs
        sigmas = (1.0 - self.alphas).sqrt() / (self.alphas.sqrt() + 1e-8)
        pos = torch.randn_like(batch.pos) * sigmas[-1] # Start from noise level sigma_T
        pos = center_pos(pos, batch.batch)
        pos_traj = []
        timesteps = torch.linspace(self.num_timesteps - 1, 0, n_steps + 1, device=device).long()

        for i in tqdm(range(n_steps), desc="ODE Sampling", leave=False):
            t_current = timesteps[i]
            t_next = timesteps[i+1]
            t_tensor = t_current.expand(batch_size)
            # score_pred 内部调用 forward 并整合二面角梯度
            score_pred = self._get_score_pred(batch, pos, t_tensor, w_global, global_start_sigma, clip, clip_local)

            # VP SDE Probability Flow ODE step
            # dx = [f(x,t) - 0.5 * g(t)^2 * score] dt
            # For VP SDE: f(x,t) = -0.5 * beta(t) * x, g(t)^2 = beta(t)
            # dt is negative here (t_next - t_current)
            # dx = [-0.5*beta(t)*x - 0.5*beta(t)*score] dt
            # dx = -0.5 * beta(t) * (x + score) * dt
            # Simplified using sigma: dx = -0.5 * d(sigma^2)/dt * score * dt = -0.5 * d(sigma^2) * score
            # pos_next = pos + (-0.5) * (sigma_next_sq - sigma_current_sq) * score_pred
            # Note: sigma decreases as t decreases. sigma_next < sigma_current, so (sigma_next_sq - sigma_current_sq) is negative.

            sigma_current_sq = sigmas[t_current]**2
            sigma_next_sq = sigmas[t_next]**2 if t_next >= 0 else torch.tensor(0.0, device=device)
            drift = -0.5 * (sigma_next_sq - sigma_current_sq) * score_pred # The minus sign is important
            pos = pos + drift

            pos = center_pos(pos, batch.batch)
            pos_traj.append(pos.clone().cpu())
        return pos, pos_traj


# --- 辅助函数 ---
def clip_norm(vec, limit, p=2):
    # (保持不变)
    if limit is None or limit <= 0: return vec
    norm = torch.norm(vec, dim=-1, p=p, keepdim=True)
    norm = torch.clamp(norm, min=1e-12)
    scale = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * scale

def center_pos(pos, batch):
    # (保持不变)
    try:
        pos_center = scatter_mean(pos, batch, dim=0)[batch]
        result = pos - pos_center
        if torch.isnan(result).any() or torch.isinf(result).any():
            print("Warning: NaN/Inf in center_pos, returning original")
            return pos
        return result
    except Exception as e:
        print(f"Error in center_pos: {e}, returning original pos")
        return pos

def is_bond(edge_type):
    # (保持不变)
    return torch.logical_and(edge_type < len(BOND_TYPES), edge_type > 0)

# ... is_angle_edge, is_dihedral_edge, etc. (保持不变) ...
def is_angle_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 1 - 1
def is_dihedral_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 2 - 1
def is_radius_edge(edge_type):
    return edge_type == 0
def is_local_edge(edge_type):
    return edge_type > 0
def clip_norm(vec, limit, p=2):
    if limit is None or limit <= 0: return vec
    norm = torch.norm(vec, dim=-1, p=p, keepdim=True)
    norm = torch.clamp(norm, min=1e-12)
    scale = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * scale

def center_pos(pos, batch):
    try:
        from torch_scatter import scatter_mean # 确保导入
        pos_center = scatter_mean(pos, batch, dim=0)[batch]
        result = pos - pos_center
        if torch.isnan(result).any() or torch.isinf(result).any():
            print("Warning: NaN/Inf in center_pos, returning original")
            return pos
        return result
    except Exception as e:
        print(f"Error in center_pos: {e}, returning original pos")
        return pos

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


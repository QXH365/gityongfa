# models/epsnet/spectroformer_ib.py

import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from torch_geometric.data import Batch
import numpy as np
from tqdm.auto import tqdm

# 导入项目相关的模块
from utils.chem import BOND_TYPES
from ..common import (MultiLayerPerceptron, assemble_atom_pair_feature,
                      extend_graph_order_radius)
from ..encoder import SchNetEncoder, GINEncoder, get_edge_encoder
from ..geometry import get_distance, eq_transform
from .spectrum_encoder import Spectroformer # 导入我们第一步创建的新光谱编码器
from easydict import EasyDict

# --- 以下辅助函数与 dualenc_v3.py 保持一致 ---

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

# --- 主要模型：SpectroformerIB ---

class SpectroformerIB(nn.Module):
    def __init__(self, config, training_phase='finetune'):
        super().__init__()
        self.config = config
        self.training_phase = training_phase

        # 1. 光谱编码器 (仅在微调阶段实例化)
        if self.training_phase == 'finetune':
            self.spectrum_encoder = Spectroformer(config.model)
        
        # 2. 时间步编码器 (所有阶段都需要)
        time_dim = config.model.hidden_dim * 4
        self.timestep_embedder = nn.Sequential(
            SinusoidalPosEmb(config.model.hidden_dim),
            nn.Linear(config.model.hidden_dim, time_dim), nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.GELU(),
            nn.Linear(time_dim, config.model.hidden_dim)
        )

        # 3. 几何骨干网络 (与 dualenc_v3 相同)
        edge_encoder_config = EasyDict({
            'edge_encoder': config.model.edge_encoder,
            'hidden_dim': config.model.hidden_dim,
            'mlp_act': config.model.mlp_act
        })
        self.edge_encoder_global = get_edge_encoder(edge_encoder_config)
        self.edge_encoder_local = get_edge_encoder(edge_encoder_config)
        
        self.encoder_global = SchNetEncoder(
            hidden_channels=config.model.hidden_dim, num_filters=config.model.hidden_dim,
            num_interactions=config.model.num_convs, edge_channels=self.edge_encoder_global.out_channels,
            cutoff=config.model.cutoff, smooth=config.model.smooth_conv,
        )
        self.encoder_local = GINEncoder(
            hidden_dim=config.model.hidden_dim, num_convs=config.model.num_convs_local,
        )
        
        # 4. Atom-to-Concept Attention 融合模块 (仅在微调阶段需要)
        if self.training_phase == 'finetune':
            self.atom_to_concept_attention = nn.MultiheadAttention(
                embed_dim=config.model.hidden_dim,
                num_heads=config.model.spec_num_heads,
                batch_first=False  # PyTorch 默认期望 (SeqLen, Batch, Dim)
            )
            self.norm_fusion = nn.LayerNorm(config.model.hidden_dim)

        # 5. 输出MLP (与 dualenc_v3 相同)
        self.grad_global_dist_mlp = MultiLayerPerceptron(
            2 * config.model.hidden_dim,
            [config.model.hidden_dim, config.model.hidden_dim // 2, 1],
            activation=config.model.mlp_act
        )
        self.grad_local_dist_mlp = MultiLayerPerceptron(
            2 * config.model.hidden_dim,
            [config.model.hidden_dim, config.model.hidden_dim // 2, 1],
            activation=config.model.mlp_act
        )
        
        # 6. 扩散过程参数 (与 dualenc_v3 相同)
        betas = get_beta_schedule(
            beta_schedule=config.model.beta_schedule, beta_start=config.model.beta_start,
            beta_end=config.model.beta_end, num_diffusion_timesteps=config.model.num_diffusion_timesteps,
        )
        betas = torch.from_numpy(betas).float()
        self.betas = nn.Parameter(betas, requires_grad=False)
        alphas = (1. - betas).cumprod(dim=0)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.num_timesteps = self.betas.size(0)

        # 用于存储KL损失的临时变量
        self.kl_loss = None

    def forward(self, batch, pos_perturbed, time_step):
        atom_type, bond_index, bond_type, batch_idx = batch.atom_type, batch.bond_edge_index, batch.bond_edge_type, batch.batch
        
        # --- 几何路径 (所有阶段都运行) ---
        time_emb = self.timestep_embedder(time_step)
        time_vec = self.time_mlp(time_emb)
        node_time_emb = time_vec[batch_idx]

        edge_index, edge_type = extend_graph_order_radius(
            num_nodes=atom_type.size(0), pos=pos_perturbed, edge_index=bond_index,
            edge_type=bond_type, batch=batch_idx, order=self.config.model.edge_order,
            cutoff=self.config.model.cutoff,
        )
        edge_length = get_distance(pos_perturbed, edge_index).unsqueeze(-1)
        local_edge_mask = (edge_type > 0)

        # 全局编码器
        edge_attr_global = self.edge_encoder_global(edge_length=edge_length, edge_type=edge_type)
        node_attr_global = self.encoder_global(z=atom_type, edge_index=edge_index, edge_length=edge_length, edge_attr=edge_attr_global)
        node_attr_global = node_attr_global + node_time_emb # 注入时间信息
        
        # 局部编码器
        edge_attr_local = self.edge_encoder_local(edge_length=edge_length[local_edge_mask], edge_type=edge_type[local_edge_mask])
        node_attr_local = self.encoder_local(z=atom_type, edge_index=edge_index[:, local_edge_mask], edge_attr=edge_attr_local)
        node_attr_local = node_attr_local + node_time_emb # 注入时间信息

        # --- 条件融合路径 (只在微调阶段运行) ---
        if self.training_phase == 'finetune':
            # 1. 生成图级别的条件向量 C_graph 用于 CIB
            graph_condition_vector = scatter_mean(node_attr_global, batch_idx, dim=0)
            
            # 2. 从 Spectroformer 获取光谱概念和 KL 损失
            spectral_concepts, kl_loss = self.spectrum_encoder(batch, graph_condition_vector)
            self.kl_loss = kl_loss  # 暂存 KL 损失以便 get_loss 函数访问
            
            # 3. Atom-to-Concept Attention 融合
            # Query: (N, H) -> (N, 1, H)
            query = node_attr_global.unsqueeze(1)
            
            # 将谱概念从 (B, NumConcepts, H) 扩展到 (N, NumConcepts, H)
            concepts_expanded = spectral_concepts[batch_idx]
            
            # MultiheadAttention 输入需要 (SeqLen, Batch, Dim)，我们调整输入以匹配
            # 这里我们将 N 视为 "Batch" 维度
            attn_output, _ = self.atom_to_concept_attention(
                query=query.transpose(0, 1),             # Query: (1, N, H)
                key=concepts_expanded.transpose(0, 1),   # Key: (NumConcepts, N, H)
                value=concepts_expanded.transpose(0, 1)  # Value: (NumConcepts, N, H)
            ) # Output: (1, N, H)
            
            # 融合特征
            node_attr_global = self.norm_fusion(node_attr_global + attn_output.squeeze(0))
            # (对 local 分支也可以做类似操作，为简化暂时省略)

        # --- 输出 MLP 计算 ---
        h_pair_global = assemble_atom_pair_feature(node_attr=node_attr_global, edge_index=edge_index, edge_attr=edge_attr_global)
        edge_inv_global = self.grad_global_dist_mlp(h_pair_global)

        h_pair_local = assemble_atom_pair_feature(node_attr=node_attr_local, edge_index=edge_index[:, local_edge_mask], edge_attr=edge_attr_local)
        edge_inv_local = self.grad_local_dist_mlp(h_pair_local)

        return edge_inv_global, edge_inv_local, edge_index, edge_length, local_edge_mask
    
    def get_loss(self, batch, anneal_power=2.0):
        pos_true, node2graph, num_graphs = batch.pos, batch.batch, batch.num_graphs
        
        time_step = torch.randint(0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=pos_true.device)
        time_step = torch.cat([time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        
        a = self.alphas.index_select(0, time_step)
        a_pos = a.index_select(0, node2graph).unsqueeze(-1)
        
        pos_noise = torch.randn_like(pos_true)
        pos_perturbed = pos_true + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()

        # 调用 self.forward
        edge_inv_global, edge_inv_local, edge_index, edge_length, local_edge_mask = self.forward(
            batch=batch, pos_perturbed=pos_perturbed, time_step=time_step
        )

        # --- 计算几何损失 (与 dualenc_v2/v3 相同) ---
        edge2graph = node2graph.index_select(0, edge_index[0])
        a_edge = a.index_select(0, edge2graph).unsqueeze(-1)
        d_gt = get_distance(pos_true, edge_index).unsqueeze(-1)
        d_target = (d_gt - edge_length) / (1.0 - a_edge).sqrt() * a_edge.sqrt()

        # 全局损失
        global_mask = torch.logical_and(
            edge_length <= self.config.model.cutoff,
            ~local_edge_mask.unsqueeze(-1)
        )
        target_d_global = torch.where(global_mask, d_target, torch.zeros_like(d_target))
        edge_inv_global = torch.where(global_mask, edge_inv_global, torch.zeros_like(edge_inv_global))
        target_pos_global = eq_transform(target_d_global, pos_perturbed, edge_index, edge_length)
        node_eq_global = eq_transform(edge_inv_global, pos_perturbed, edge_index, edge_length)
        loss_global = (node_eq_global - target_pos_global)**2
        loss_global = 2 * torch.sum(loss_global, dim=-1, keepdim=True)

        # 局部损失
        target_pos_local = eq_transform(d_target[local_edge_mask], pos_perturbed, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
        node_eq_local = eq_transform(edge_inv_local, pos_perturbed, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
        loss_local = (node_eq_local - target_pos_local)**2
        loss_local = 2 * torch.sum(loss_local, dim=-1, keepdim=True)

        geom_loss = (loss_global + loss_local).mean()
        
        # --- 添加 CIB 的 KL 散度损失 (仅在微调阶段) ---
        total_loss = geom_loss
        if self.training_phase == 'finetune' and self.kl_loss is not None:
            kl_weight = self.config.train.get('kl_weight', 1e-4) # 从配置获取权重
            total_loss = total_loss + self.kl_loss * kl_weight
            self.kl_loss = None # 清除已使用的损失

        return total_loss, loss_global.mean(), loss_local.mean()
    def compute_alpha(self, beta, t):
        """修复的 compute_alpha 函数"""
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        # 修复索引问题：不要 +1，直接使用 t
        idx = torch.clamp(t, 0, beta.size(0) - 1)
        a = (1 - beta).cumprod(dim=0).index_select(0, idx)
        return a
    @torch.no_grad()
    def langevin_dynamics_sample(
        self,
        batch,
        w_global=0.5,
        global_start_sigma=float('inf'),
        clip=10.0,
        clip_local=5.0,
        temperature=1.0, # 控制初始噪声和每步噪声的温度
        debug=False
    ):
        """
        修复后的采样函数，参考GeoDiff实现，使用正确的逆向SDE求解步骤。
        此函数直接使用模型预测的分数(score)，解决了训练与采样的不匹配问题。
        """
        device = self.betas.device
        batch_size = batch.num_graphs
        
        # 1. 初始化坐标为高斯噪声
        pos = torch.randn_like(batch.pos) * temperature
        pos = center_pos(pos, batch.batch)
        pos_traj = []

        # 2. 定义时间步序列，从 T-1 到 0
        timesteps = list(range(self.num_timesteps))[::-1]
        
        # 预计算用于计算sigma的alpha累积乘积
        alphas_cumprod = self.alphas
        
        # 3. 逆向扩散循环
        for i, t in enumerate(tqdm(timesteps, desc="Sampling", leave=False)):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)
            
            # --- 模型前向传播，得到分数预测 ---
            edge_inv_global, edge_inv_local, edge_index, _, edge_length, local_edge_mask = self.forward(
                batch=batch, pos_perturbed=pos, time_step=t_tensor
            )
            
            # --- 将距离空间的预测转换为坐标空间的分数 ---
            # 局部（化学键）分数
            if local_edge_mask.sum() > 0:
                node_eq_local = eq_transform(
                    edge_inv_local, pos, 
                    edge_index[:, local_edge_mask], 
                    edge_length[local_edge_mask]
                )
                if clip_local is not None:
                    node_eq_local = clip_norm(node_eq_local, limit=clip_local)
            else:
                node_eq_local = torch.zeros_like(pos)

            # 全局（非键）分数，在噪声水平较低时启用
            current_sigma = ((1 - alphas_cumprod[t]) / alphas_cumprod[t]).sqrt()
            if current_sigma < global_start_sigma:
                non_local_mask = ~local_edge_mask
                if non_local_mask.sum() > 0:
                    node_eq_global = eq_transform(
                        edge_inv_global[non_local_mask], pos,
                        edge_index[:, non_local_mask],
                        edge_length[non_local_mask]
                    )
                    node_eq_global = clip_norm(node_eq_global, limit=clip)
                else:
                    node_eq_global = torch.zeros_like(pos)
            else:
                node_eq_global = torch.zeros_like(pos)

            # 组合局部和全局分数
            score_pred = node_eq_local + w_global * node_eq_global

            # --- 正确的逆向SDE更新步骤 (祖先采样) ---
            # 获取当前时间步的扩散参数
            beta_t = self.betas[t]
            alpha_t = 1. - beta_t
            
            # 1. 计算均值 (Drift Term)
            # 这个公式直接使用了分数 score_pred，是修复的关键
            pos_mean = (1. / alpha_t.sqrt()) * (pos + beta_t * score_pred)
            
            # 2. 添加噪声 (Diffusion Term)
            if i < len(timesteps) - 1: # 除了最后一步，都添加噪声
                # 计算后验方差
                alpha_cumprod_prev = alphas_cumprod[timesteps[i+1]] if i + 1 < len(timesteps) else torch.tensor(1.0, device=device)
                posterior_variance = (1. - alpha_cumprod_prev) / (1. - alphas_cumprod[t]) * beta_t
                
                noise = torch.randn_like(pos) * temperature
                pos = pos_mean + torch.sqrt(posterior_variance) * noise
            else:
                # 最后一步不加噪声
                pos = pos_mean

            # --- 后处理，增强稳定性 ---
            pos = center_pos(pos, batch.batch)
            if torch.isnan(pos).any():
                print(f"NaN detected at step {i}, t={t}. Stopping.")
                break # 如果出现NaN，提前终止
                
            pos_traj.append(pos.clone().cpu())
            
            if debug and i % (len(timesteps) // 10) == 0:
                 print(f"Step {i}/{len(timesteps)}, t={t}, pos_std={pos.std():.4f}, score_norm={score_pred.norm():.4f}")

        return pos, pos_traj

    def langevin_dynamics_sample_ode(
        self,
        batch,
        n_steps=100,  # ODE求解器可以使用较少的步数
        w_global=0.5,
        global_start_sigma=float('inf'),
        clip=10.0,
        clip_local=5.0,
    ):
        """
        基于概率流常微分方程(Probability Flow ODE)的快速、确定性采样器。

        此方法将扩散过程视为一个确定性的轨迹，从纯噪声平滑地演变到数据。
        因为它没有随机步骤，所以速度更快，并且对于给定的初始噪声，结果是完全可复现的。
        非常适合需要快速生成构象的应用。

        Args:
            n_steps (int): 采样的总步数。通常50-200步即可获得很好的结果。
        """
        device = self.betas.device
        batch_size = batch.num_graphs

        # 1. 初始化坐标为高斯噪声
        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        pos = torch.randn_like(batch.pos) * sigmas[-1]
        pos = center_pos(pos, batch.batch)
        pos_traj = []

        # 2. 生成跳步采样的时间步序列 (从 T-1 到 0)
        timesteps = torch.linspace(self.num_timesteps - 1, 0, n_steps + 1, device=device).long()

        # 3. ODE 求解循环 (使用简单的欧拉法)
        for i in tqdm(range(n_steps), desc="ODE Sampling", leave=False):
            t_current = timesteps[i]
            t_next = timesteps[i+1]
            
            t_tensor = t_current.expand(batch_size)

            # --- 模型前向传播，得到分数预测 ---
            score_pred = self._get_score_pred(batch, pos, t_tensor, w_global, global_start_sigma, clip, clip_local)
            
            # --- 核心: 概率流ODE更新步骤 ---
            sigma_current_sq = sigmas[t_current]**2
            sigma_next_sq = sigmas[t_next]**2
            
            # 漂移项(Drift): ODE的更新只包含漂移项
            drift = 0.5 * (sigma_current_sq - sigma_next_sq) * score_pred
            
            # 更新坐标 (无噪声项)
            pos = pos + drift
            
            # --- 后处理 ---
            pos = center_pos(pos, batch.batch)
            pos_traj.append(pos.clone().cpu())

        return pos, pos_traj
    
    
    
    def _get_score_pred(self, batch, pos, t_tensor, w_global, global_start_sigma, clip, clip_local):
        """辅助函数，用于计算组合后的分数预测。"""
        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        
        # 模型前向传播
        edge_inv_global, edge_inv_local, edge_index, _, edge_length, local_edge_mask = self.forward(
            batch=batch, pos_perturbed=pos, time_step=t_tensor
        )
        
        # 局部（化学键）分数
        if local_edge_mask.sum() > 0:
            node_eq_local = eq_transform(edge_inv_local, pos, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
            if clip_local is not None: node_eq_local = clip_norm(node_eq_local, limit=clip_local)
        else:
            node_eq_local = torch.zeros_like(pos)
            
        # 全局（非键）分数
        current_sigma = sigmas[t_tensor[0].item()]
        if current_sigma < global_start_sigma:
            non_local_mask = ~local_edge_mask
            if non_local_mask.sum() > 0:
                node_eq_global = eq_transform(edge_inv_global[non_local_mask], pos, edge_index[:, non_local_mask], edge_length[non_local_mask])
                node_eq_global = clip_norm(node_eq_global, limit=clip)
            else:
                node_eq_global = torch.zeros_like(pos)
        else:
            node_eq_global = torch.zeros_like(pos)
        
        # 组合分数
        score_pred = node_eq_local + w_global * node_eq_global
        return score_pred


def clip_norm(vec, limit, p=2):
    """改进的梯度裁剪"""
    if limit is None or limit <= 0:
        return vec
    
    norm = torch.norm(vec, dim=-1, p=p, keepdim=True)
    norm = torch.clamp(norm, min=1e-12)  # 避免除零
    scale = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * scale


def center_pos(pos, batch):
    """改进的位置中心化"""
    try:
        from torch_scatter import scatter_mean
        pos_center = scatter_mean(pos, batch, dim=0)[batch]
        result = pos - pos_center
        
        # 检查结果
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



# models/epsnet/dualenc_v4.py

import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from torch_geometric.data import Batch
import numpy as np
from tqdm.auto import tqdm

# 导入项目相关的模块
from utils.chem import BOND_TYPES
from ..common import MultiLayerPerceptron, assemble_atom_pair_feature, extend_graph_order_radius
from ..encoder import SchNetEncoder, GINEncoder, get_edge_encoder
from ..geometry import get_distance, eq_transform

try:
    from .spectrum_encoder import Spectroformer
except ImportError:
    Spectroformer = None
from easydict import EasyDict

# --- 辅助模块与函数 ---


class GatedFusionBlock(nn.Module):
    """
    一个封装了“原子到概念注意力”和“门控融合”的模块。
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.atom_to_concept_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=False  # 注意力模块的输入需要 (SeqLen, Batch, Dim)
        )
        # 门控网络：接收拼接后的特征，输出一个0-1之间的门控值
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim), nn.Sigmoid()
        )
        self.norm_geom = nn.LayerNorm(embed_dim)
        self.norm_spec = nn.LayerNorm(embed_dim)
        self.norm_final = nn.LayerNorm(embed_dim)

    def forward(self, geom_features, spectral_concepts, batch_idx):
        # 1. Atom-to-Concept Attention
        query = geom_features.unsqueeze(0)  # (1, NumAtoms, Dim)
        concepts_expanded = spectral_concepts[batch_idx]  # (NumAtoms, NumConcepts, Dim)

        # PyTorch MHA期望 (SeqLen, Batch, Dim), 这里我们将NumAtoms视为Batch, NumConcepts视为SeqLen
        spec_context, _ = self.atom_to_concept_attention(
            query=query,  # (1, NumAtoms, Dim)
            key=concepts_expanded.transpose(0, 1),  # (NumConcepts, NumAtoms, Dim)
            value=concepts_expanded.transpose(0, 1),  # (NumConcepts, NumAtoms, Dim)
        )
        spec_context = spec_context.squeeze(0)  # (NumAtoms, Dim)

        # 2. Gated Fusion
        # 归一化可以使门控更稳定
        geom_features_norm = self.norm_geom(geom_features)
        spec_context_norm = self.norm_spec(spec_context)

        # 计算门控值
        gate_input = torch.cat([geom_features_norm, spec_context_norm], dim=-1)
        gate = self.gate_mlp(gate_input)

        # 应用门控融合
        fused_features = (1 - gate) * geom_features + gate * spec_context

        # 3. 最终归一化
        return self.norm_final(fused_features)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = np.linspace(beta_start**0.5, beta_end**0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
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
    def __init__(self, config, training_phase):
        super().__init__()
        self.config = config

        if training_phase not in ["pretrain", "finetune"]:
            raise ValueError(f"Invalid training_phase: {training_phase}. Must be 'pretrain' or 'finetune'.")
        self.training_phase = training_phase
        print(f"--- SpectroformerIB model initialized in '{self.training_phase}' mode. ---")

        # 1. 光谱编码器 (仅在微调阶段实例化)
        if self.training_phase == "finetune":
            if Spectroformer is None:
                raise ImportError("Spectroformer could not be imported, which is required for finetuning.")
            self.spectrum_encoder = Spectroformer(config)

        # 2. 时间步编码器
        time_dim = config.model.hidden_dim * 4
        self.timestep_embedder = nn.Sequential(
            SinusoidalPosEmb(config.model.hidden_dim),
            nn.Linear(config.model.hidden_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.GELU(), nn.Linear(time_dim, config.model.hidden_dim)
        )

        # 3. 几何骨干网络
        edge_encoder_config = EasyDict(
            {
                "edge_encoder": config.model.edge_encoder,
                "hidden_dim": config.model.hidden_dim,
                "mlp_act": config.model.mlp_act,
            }
        )
        self.edge_encoder_global = get_edge_encoder(edge_encoder_config)
        self.edge_encoder_local = get_edge_encoder(edge_encoder_config)

        self.encoder_global = SchNetEncoder(
            hidden_channels=config.model.hidden_dim,
            num_filters=config.model.hidden_dim,
            num_interactions=config.model.num_convs,
            edge_channels=self.edge_encoder_global.out_channels,
            cutoff=config.model.cutoff,
            smooth=config.model.smooth_conv,
        )
        self.encoder_local = GINEncoder(
            hidden_dim=config.model.hidden_dim,
            num_convs=config.model.num_convs_local,
        )

        # 4. 【核心修改】为全局和局部路径分别实例化门控融合模块
        if self.training_phase == "finetune":
            self.fusion_global = GatedFusionBlock(
                embed_dim=config.model.hidden_dim, num_heads=config.model.spec_num_heads
            )
            self.fusion_local = GatedFusionBlock(
                embed_dim=config.model.hidden_dim, num_heads=config.model.spec_num_heads
            )

        # 5. 输出MLP
        self.grad_global_dist_mlp = MultiLayerPerceptron(
            2 * config.model.hidden_dim,
            [config.model.hidden_dim, config.model.hidden_dim // 2, 1],
            activation=config.model.mlp_act,
        )
        self.grad_local_dist_mlp = MultiLayerPerceptron(
            2 * config.model.hidden_dim,
            [config.model.hidden_dim, config.model.hidden_dim // 2, 1],
            activation=config.model.mlp_act,
        )

        # 6. 扩散过程参数
        betas = get_beta_schedule(
            beta_schedule=config.model.beta_schedule,
            beta_start=config.model.beta_start,
            beta_end=config.model.beta_end,
            num_diffusion_timesteps=config.model.num_diffusion_timesteps,
        )
        betas = torch.from_numpy(betas).float()
        self.betas = nn.Parameter(betas, requires_grad=False)
        alphas = (1.0 - betas).cumprod(dim=0)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.num_timesteps = self.betas.size(0)

        self.kl_loss = None

    def forward(self, batch, pos_perturbed, time_step):
        atom_type, bond_index, bond_type, batch_idx = (
            batch.atom_type,
            batch.bond_edge_index,
            batch.bond_edge_type,
            batch.batch,
        )

        time_emb = self.timestep_embedder(time_step)
        time_vec = self.time_mlp(time_emb)
        node_time_emb = time_vec[batch_idx]

        edge_index, edge_type = extend_graph_order_radius(
            num_nodes=atom_type.size(0),
            pos=pos_perturbed,
            edge_index=bond_index,
            edge_type=bond_type,
            batch=batch_idx,
            order=self.config.model.edge_order,
            cutoff=self.config.model.cutoff,
        )
        edge_length = get_distance(pos_perturbed, edge_index).unsqueeze(-1)
        local_edge_mask = edge_type > 0

        # --- 几何特征提取 ---
        edge_attr_global = self.edge_encoder_global(edge_length=edge_length, edge_type=edge_type)
        node_attr_global = self.encoder_global(
            z=atom_type, edge_index=edge_index, edge_length=edge_length, edge_attr=edge_attr_global
        )
        node_attr_global = node_attr_global + node_time_emb

        edge_attr_local = self.edge_encoder_local(
            edge_length=edge_length[local_edge_mask], edge_type=edge_type[local_edge_mask]
        )
        node_attr_local = self.encoder_local(
            z=atom_type, edge_index=edge_index[:, local_edge_mask], edge_attr=edge_attr_local
        )
        node_attr_local = node_attr_local + node_time_emb

        # --- 光谱-几何融合 (仅在微调阶段) ---
        if self.training_phase == "finetune":
            spectral_concepts, kl_loss = self.spectrum_encoder(batch)
            self.kl_loss = kl_loss

            # 【核心修改】分别对 global 和 local 特征进行门控融合
            node_attr_global = self.fusion_global(node_attr_global, spectral_concepts, batch_idx)
            node_attr_local = self.fusion_local(node_attr_local, spectral_concepts, batch_idx)

        # --- 输出 MLP 计算 ---
        h_pair_global = assemble_atom_pair_feature(
            node_attr=node_attr_global, edge_index=edge_index, edge_attr=edge_attr_global
        )
        edge_inv_global = self.grad_global_dist_mlp(h_pair_global)

        h_pair_local = assemble_atom_pair_feature(
            node_attr=node_attr_local, edge_index=edge_index[:, local_edge_mask], edge_attr=edge_attr_local
        )
        edge_inv_local = self.grad_local_dist_mlp(h_pair_local)

        return edge_inv_global, edge_inv_local, edge_index, edge_length, local_edge_mask

    # ... [get_loss 和其他采样函数保持不变] ...
    def get_loss(self, batch, anneal_power=2.0):
        pos_true, node2graph, num_graphs = batch.pos, batch.batch, batch.num_graphs
        time_step = torch.randint(0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=pos_true.device)
        time_step = torch.cat([time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        a = self.alphas.index_select(0, time_step)
        a_pos = a.index_select(0, node2graph).unsqueeze(-1)
        pos_noise = torch.randn_like(pos_true)
        pos_perturbed = pos_true + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()
        edge_inv_global, edge_inv_local, edge_index, edge_length, local_edge_mask = self.forward(
            batch=batch, pos_perturbed=pos_perturbed, time_step=time_step
        )
        edge2graph = node2graph.index_select(0, edge_index[0])
        a_edge = a.index_select(0, edge2graph).unsqueeze(-1)
        d_gt = get_distance(pos_true, edge_index).unsqueeze(-1)
        d_target = (d_gt - edge_length) / (1.0 - a_edge).sqrt() * a_edge.sqrt()
        global_mask = torch.logical_and(edge_length <= self.config.model.cutoff, ~local_edge_mask.unsqueeze(-1))
        target_d_global = torch.where(global_mask, d_target, torch.zeros_like(d_target))
        edge_inv_global = torch.where(global_mask, edge_inv_global, torch.zeros_like(edge_inv_global))
        target_pos_global = eq_transform(target_d_global, pos_perturbed, edge_index, edge_length)
        node_eq_global = eq_transform(edge_inv_global, pos_perturbed, edge_index, edge_length)
        loss_global = (node_eq_global - target_pos_global) ** 2
        loss_global = 2 * torch.sum(loss_global, dim=-1, keepdim=True)
        target_pos_local = eq_transform(
            d_target[local_edge_mask], pos_perturbed, edge_index[:, local_edge_mask], edge_length[local_edge_mask]
        )
        node_eq_local = eq_transform(
            edge_inv_local, pos_perturbed, edge_index[:, local_edge_mask], edge_length[local_edge_mask]
        )
        loss_local = (node_eq_local - target_pos_local) ** 2
        loss_local = 2 * torch.sum(loss_local, dim=-1, keepdim=True)
        geom_loss = (loss_global + loss_local).mean()
        total_loss = geom_loss
        if self.training_phase == "finetune" and hasattr(self, "kl_loss") and self.kl_loss is not None:
            kl_weight = self.config.train.get("kl_weight", 1e-4)
            total_loss = total_loss + self.kl_loss * kl_weight
            self.kl_loss = None
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
        global_start_sigma=float("inf"),
        clip=10.0,
        clip_local=5.0,
        temperature=1.0,  # 控制初始噪声和每步噪声的温度
        debug=False,
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
            edge_inv_global, edge_inv_local, edge_index, edge_length, local_edge_mask = self.forward(
                batch=batch, pos_perturbed=pos, time_step=t_tensor
            )

            # --- 将距离空间的预测转换为坐标空间的分数 ---
            # 局部（化学键）分数
            if local_edge_mask.sum() > 0:
                node_eq_local = eq_transform(
                    edge_inv_local, pos, edge_index[:, local_edge_mask], edge_length[local_edge_mask]
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
                        edge_inv_global[non_local_mask], pos, edge_index[:, non_local_mask], edge_length[non_local_mask]
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
            alpha_t = 1.0 - beta_t

            # 1. 计算均值 (Drift Term)
            # 这个公式直接使用了分数 score_pred，是修复的关键
            pos_mean = (1.0 / alpha_t.sqrt()) * (pos + beta_t * score_pred)

            # 2. 添加噪声 (Diffusion Term)
            if i < len(timesteps) - 1:  # 除了最后一步，都添加噪声
                # 计算后验方差
                alpha_cumprod_prev = (
                    alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0, device=device)
                )
                posterior_variance = (1.0 - alpha_cumprod_prev) / (1.0 - alphas_cumprod[t]) * beta_t

                noise = torch.randn_like(pos) * temperature
                pos = pos_mean + torch.sqrt(posterior_variance) * noise
            else:
                # 最后一步不加噪声
                pos = pos_mean

            # --- 后处理，增强稳定性 ---
            pos = center_pos(pos, batch.batch)
            if torch.isnan(pos).any():
                print(f"NaN detected at step {i}, t={t}. Stopping.")
                break  # 如果出现NaN，提前终止

            pos_traj.append(pos.clone().cpu())

            if debug and i % (len(timesteps) // 10) == 0:
                print(f"Step {i}/{len(timesteps)}, t={t}, pos_std={pos.std():.4f}, score_norm={score_pred.norm():.4f}")

        return pos, pos_traj

    def langevin_dynamics_sample_ode(
        self,
        batch,
        n_steps=100,  # ODE求解器可以使用较少的步数
        w_global=0.5,
        global_start_sigma=float("inf"),
        clip=10.0,
        clip_local=5.0,
    ):
        """
        基于概率流常微分方程(Probability Flow ODE)的快速、确定性采样器。
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
            t_next = timesteps[i + 1]

            t_tensor = t_current.expand(batch_size)

            score_pred = self._get_score_pred(batch, pos, t_tensor, w_global, global_start_sigma, clip, clip_local)

            sigma_current_sq = sigmas[t_current] ** 2
            sigma_next_sq = sigmas[t_next] ** 2

            drift = 0.5 * (sigma_current_sq - sigma_next_sq) * score_pred

            pos = pos + drift

            pos = center_pos(pos, batch.batch)
            pos_traj.append(pos.clone().cpu())

        return pos, pos_traj

    def _get_score_pred(self, batch, pos, t_tensor, w_global, global_start_sigma, clip, clip_local):
        """辅助函数，用于计算组合后的分数预测。"""
        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()

        edge_inv_global, edge_inv_local, edge_index, edge_length, local_edge_mask = self.forward(
            batch=batch, pos_perturbed=pos, time_step=t_tensor
        )

        if local_edge_mask.sum() > 0:
            node_eq_local = eq_transform(
                edge_inv_local, pos, edge_index[:, local_edge_mask], edge_length[local_edge_mask]
            )
            if clip_local is not None:
                node_eq_local = clip_norm(node_eq_local, limit=clip_local)
        else:
            node_eq_local = torch.zeros_like(pos)

        current_sigma = sigmas[t_tensor[0].item()]
        if current_sigma < global_start_sigma:
            non_local_mask = ~local_edge_mask
            if non_local_mask.sum() > 0:
                node_eq_global = eq_transform(
                    edge_inv_global[non_local_mask], pos, edge_index[:, non_local_mask], edge_length[non_local_mask]
                )
                node_eq_global = clip_norm(node_eq_global, limit=clip)
            else:
                node_eq_global = torch.zeros_like(pos)
        else:
            node_eq_global = torch.zeros_like(pos)

        score_pred = node_eq_local + w_global * node_eq_global
        return score_pred


def clip_norm(vec, limit, p=2):
    """改进的梯度裁剪"""
    if limit is None or limit <= 0:
        return vec

    norm = torch.norm(vec, dim=-1, p=p, keepdim=True)
    norm = torch.clamp(norm, min=1e-12)
    scale = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * scale


def center_pos(pos, batch):
    """改进的位置中心化"""
    try:
        from torch_scatter import scatter_mean

        pos_center = scatter_mean(pos, batch, dim=0)[batch]
        result = pos - pos_center

        if torch.isnan(result).any() or torch.isinf(result).any():
            print("Warning: NaN/Inf in center_pos, returning original")
            return pos

        return result
    except Exception as e:
        print(f"Error in center_pos: {e}, returning original pos")
        return pos

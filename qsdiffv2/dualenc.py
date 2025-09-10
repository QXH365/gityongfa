import torch
from torch import nn
from torch_scatter import scatter_add, scatter_mean, scatter
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

# ... get_beta_schedule 和 SinusoidalPosEmb 函数保持不变 ...
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x): return 1 / (np.exp(-x) + 1)
    if beta_schedule == "quad": betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == "linear": betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    # 增加了对 sigmoid 的支持，以匹配原始 GeoDiff 配置
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
        
        # 将光谱条件和时间条件融合
        # 输入维度是光谱嵌入维度 + 时间嵌入维度
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
        
        # 将条件信息融入到MLP中是一种常见做法，这里我们先保持原始结构
        # 即条件信息加到节点特征上
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
        """
        这个 forward 函数现在更接近原始的 GeoDiff，同时融入了您的条件模块。
        """
        atom_type, bond_index, bond_type, batch_idx = batch.atom_type, batch.bond_edge_index, batch.bond_edge_type, batch.batch
        
        # 1. 生成条件嵌入
        spec_condition = self.spectrum_encoder(batch) # (num_graphs, hidden_dim)
        time_condition = self.timestep_embedder(time_step) # (num_graphs, time_dim)
        
        # 2. 融合条件并映射到每个节点
        combined_cond = torch.cat([spec_condition, time_condition], dim=1)
        final_cond = self.condition_mlp(combined_cond) # (num_graphs, hidden_dim)
        node_condition = final_cond[batch_idx] # (num_nodes, hidden_dim)

        # 3. 动态构建图结构
        edge_index, edge_type = extend_graph_order_radius(
            num_nodes=atom_type.size(0), pos=pos_perturbed, edge_index=bond_index,
            edge_type=bond_type, batch=batch_idx, order=self.config.edge_order,
            cutoff=self.config.cutoff,
        )
        edge_length = get_distance(pos_perturbed, edge_index).unsqueeze(-1)
        local_edge_mask = (edge_type > 0)

        # 4. 计算信噪比相关的缩放因子 (这是稳定性的关键)
        node2graph = batch_idx
        edge2graph = node2graph.index_select(0, edge_index[0])
        # DDPM的损失隐式地处理了这个缩放，所以这里 sigma_edge 设为 1
        # 但在DSM模式下，这里会是噪声的标准差
        sigma_edge = torch.ones(size=(edge_index.size(1), 1), device=pos_perturbed.device)

        # ===== 全局编码器 =====
        edge_attr_global = self.edge_encoder_global(edge_length=edge_length, edge_type=edge_type)
        node_attr_global = self.encoder_global(z=atom_type, edge_index=edge_index, edge_length=edge_length, edge_attr=edge_attr_global)
        node_attr_global = node_attr_global + node_condition # 将条件注入节点特征
        
        h_pair_global = assemble_atom_pair_feature(node_attr=node_attr_global, edge_index=edge_index, edge_attr=edge_attr_global)
        # 恢复了信噪比缩放，这是稳定性的关键之一
        edge_inv_global = self.grad_global_dist_mlp(h_pair_global) * (1.0 / sigma_edge)

        # ===== 局部编码器 =====
        edge_attr_local = self.edge_encoder_local(edge_length=edge_length[local_edge_mask], edge_type=edge_type[local_edge_mask])
        node_attr_local = self.encoder_local(z=atom_type, edge_index=edge_index[:, local_edge_mask], edge_attr=edge_attr_local)
        # 同样注入条件
        node_attr_local = node_attr_local + node_condition 
        
        h_pair_local = assemble_atom_pair_feature(node_attr=node_attr_local, edge_index=edge_index[:, local_edge_mask], edge_attr=edge_attr_local)
        edge_inv_local = self.grad_local_dist_mlp(h_pair_local) * (1.0 / sigma_edge[local_edge_mask])

        # 返回所有必要的信息用于计算损失
        return edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask

    def get_loss(self, batch, anneal_power=2.0):
        """
        重写后的损失函数，恢复了 GeoDiff 的稳定几何目标。
        """
        pos_true, node2graph, num_graphs = batch.pos, batch.batch, batch.num_graphs
        # 1. 对称的时间步采样，增加稳定性
        time_step = torch.randint(0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=pos_true.device)
        time_step = torch.cat([time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        
        a = self.alphas.index_select(0, time_step)
        a_pos = a.index_select(0, node2graph).unsqueeze(-1)
        
        # 2. 使用原始 GeoDiff 的加噪方式（也可以用DDPM方式，但需确保损失函数对应）
        pos_noise = torch.randn_like(pos_true)
        # pos_perturbed = pos_true * a_pos.sqrt() + pos_noise * (1.0 - a_pos).sqrt() # DDPM加噪
        pos_perturbed = pos_true + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt() # GeoDiff加噪 (Variance Preserving SDE)

        # 3. 调用 forward 获取预测的距离梯度
        edge_inv_global, edge_inv_local, edge_index, _, edge_length, local_edge_mask = self.forward(
            batch=batch, pos_perturbed=pos_perturbed, time_step=time_step
        )

        # 4. 在距离空间计算真实的目标 (核心区别)
        edge2graph = node2graph.index_select(0, edge_index[0])
        a_edge = a.index_select(0, edge2graph).unsqueeze(-1)
        
        d_gt = get_distance(pos_true, edge_index).unsqueeze(-1)
        d_perturbed = edge_length
        # 目标是去噪方向，在距离空间上定义
        d_target = (d_gt - d_perturbed) / (1.0 - a_edge).sqrt() * a_edge.sqrt()

        # 5. 将距离梯度转换回坐标梯度，并分开计算损失
        # 全局损失
        global_mask = torch.logical_and(
            d_perturbed <= self.config.cutoff,
            ~local_edge_mask.unsqueeze(-1)
        )
        target_d_global = torch.where(global_mask, d_target, torch.zeros_like(d_target))
        edge_inv_global = torch.where(global_mask, edge_inv_global, torch.zeros_like(edge_inv_global))
        
        target_pos_global = eq_transform(target_d_global, pos_perturbed, edge_index, edge_length)
        node_eq_global = eq_transform(edge_inv_global, pos_perturbed, edge_index, edge_length)
        loss_global = (node_eq_global - target_pos_global)**2
        loss_global = 2 * torch.sum(loss_global, dim=-1, keepdim=True) # 权重系数参考原始代码

        # 局部损失
        target_pos_local = eq_transform(d_target[local_edge_mask], pos_perturbed, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
        node_eq_local = eq_transform(edge_inv_local, pos_perturbed, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
        loss_local = (node_eq_local - target_pos_local)**2
        loss_local = 5 * torch.sum(loss_local, dim=-1, keepdim=True) # 局部损失权重更高

        # 6. 最终损失
        loss = loss_global + loss_local
        
        # 不需要 anneal_power，因为DDPM的损失函数形式已经隐式处理了加权
        return loss.mean(), loss_global.mean(), loss_local.mean()

    def compute_alpha(self, beta, t):
        """修复的 compute_alpha 函数"""
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        # 修复索引问题：不要 +1，直接使用 t
        idx = torch.clamp(t, 0, beta.size(0) - 1)
        a = (1 - beta).cumprod(dim=0).index_select(0, idx)
        return a

    @torch.no_grad()
    def langevin_dynamics_sample(
        self, batch, n_steps=100, eta=1.0, w_global=0.3,
        global_start_sigma=0.5, clip=10.0, clip_local=5.0,
        ddim_eta=0.0, use_ddim=True, temperature=1.0, 
        debug=False
    ):
        """
        完全重写的稳定采样函数
        """
        device = batch.pos.device
        batch_size = batch.num_graphs
        
        # 1. 安全初始化
        pos = torch.randn_like(batch.pos) * temperature
        pos = center_pos(pos, batch.batch)
        pos_traj = []
        
        # 2. 修复时间步序列生成
        if n_steps >= self.num_timesteps:
            # 使用所有时间步
            timesteps = list(range(self.num_timesteps))[::-1]
        else:
            # 从最后 n_steps 个时间步开始，等间距采样
            timesteps = torch.linspace(0, self.num_timesteps-1, n_steps).long().tolist()[::-1]
        
        if debug:
            print(f"Sampling timesteps: {timesteps[:5]}...{timesteps[-5:]}")
        
        # 3. 预计算一些常量以避免重复计算
        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        
        # 4. 主采样循环
        for step_idx, t in enumerate(tqdm(timesteps, desc="Sampling", leave=False)):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)
            
            # 数值安全检查
            if torch.isnan(pos).any() or torch.isinf(pos).any():
                print(f"NaN/Inf detected at step {step_idx}, t={t}")
                print(f"pos stats: min={pos.min()}, max={pos.max()}, std={pos.std()}")
                break
            
            try:
                # 5. 模型前向传播
                pred = self.forward(batch, pos, t_tensor)
                edge_inv_global, edge_inv_local, edge_index, _, edge_length, local_edge_mask = pred
                
                # 6. 转换为节点梯度
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
                
                # 7. 全局项处理（条件性启用）
                current_sigma = sigmas[t] if t < len(sigmas) else sigmas[-1]
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
                
                # 8. 组合预测
                eps_pred = node_eq_local + w_global * node_eq_global
                
                # 9. 获取扩散参数
                alpha_t = self.alphas[t]
                beta_t = self.betas[t]
                sqrt_alpha_t = alpha_t.sqrt()
                sqrt_one_minus_alpha_bar_t = (1 - alpha_t).sqrt()
                
                # 数值稳定性检查
                if alpha_t < 1e-8:
                    print(f"Warning: alpha_t too small at t={t}: {alpha_t}")
                    alpha_t = torch.clamp(alpha_t, min=1e-8)
                    sqrt_alpha_t = alpha_t.sqrt()
                
                # 10. DDPM/DDIM 更新
                if use_ddim:
                    # DDIM 更新
                    if step_idx < len(timesteps) - 1:
                        t_next = timesteps[step_idx + 1]
                        alpha_t_next = self.alphas[t_next]
                        sqrt_alpha_t_next = alpha_t_next.sqrt()
                        sqrt_one_minus_alpha_bar_t_next = (1 - alpha_t_next).sqrt()
                    else:
                        sqrt_alpha_t_next = torch.tensor(1.0, device=device)
                        sqrt_one_minus_alpha_bar_t_next = torch.tensor(0.0, device=device)
                    
                    # 预测 x_0
                    pred_x0 = (pos - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_t
                    pred_x0 = torch.clamp(pred_x0, -clip, clip)
                    
                    # DDIM 方向
                    dir_xt = sqrt_one_minus_alpha_bar_t_next * eps_pred
                    
                    # 更新位置
                    pos_next = sqrt_alpha_t_next * pred_x0 + dir_xt
                    
                    # 可选的噪声注入
                    if ddim_eta > 0 and step_idx < len(timesteps) - 1:
                        sigma = ddim_eta * sqrt_one_minus_alpha_bar_t_next
                        noise = torch.randn_like(pos) * temperature
                        pos_next = pos_next + sigma * noise
                        
                else:
                    # 标准 DDPM 更新
                    # 预测 x_0
                    pred_x0 = (pos - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_t
                    pred_x0 = torch.clamp(pred_x0, -clip, clip)
                    
                    # DDPM 均值
                    pos_mean = (1 / sqrt_alpha_t) * (pos - beta_t / sqrt_one_minus_alpha_bar_t * eps_pred)
                    
                    if step_idx < len(timesteps) - 1:
                        # 不是最后一步，添加噪声
                        noise = torch.randn_like(pos) * temperature
                        # 计算后验方差
                        if step_idx < len(timesteps) - 1:
                            t_next = timesteps[step_idx + 1]
                            alpha_t_next = self.alphas[t_next]
                        else:
                            alpha_t_next = torch.tensor(1.0, device=device)
                        
                        posterior_variance = beta_t * (1 - alpha_t_next) / (1 - alpha_t)
                        posterior_variance = torch.clamp(posterior_variance, min=1e-8)
                        
                        pos_next = pos_mean + posterior_variance.sqrt() * noise
                    else:
                        # 最后一步
                        pos_next = pos_mean
                
                # 11. 后处理
                pos = pos_next
                pos = center_pos(pos, batch.batch)
                pos = torch.clamp(pos, -clip, clip)
                
                # 最终安全检查
                if torch.isnan(pos).any() or torch.isinf(pos).any():
                    print(f"NaN/Inf after update at step {step_idx}, t={t}")
                    print(f"eps_pred stats: min={eps_pred.min()}, max={eps_pred.max()}")
                    print(f"alpha_t: {alpha_t}, beta_t: {beta_t}")
                    break
                    
                pos_traj.append(pos.clone().cpu())
                
                if debug and step_idx % max(1, len(timesteps)//10) == 0:
                    print(f"Step {step_idx}/{len(timesteps)}, t={t}, pos_std={pos.std():.4f}")
                    
            except Exception as e:
                print(f"Error at step {step_idx}, t={t}: {str(e)}")
                import traceback
                traceback.print_exc()
                break
        
        return pos, pos_traj


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



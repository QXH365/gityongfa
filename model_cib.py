import os
import time
import logging
import pickle
import math
import json
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Batch
import numpy as np
from tqdm import tqdm


class DiscreteNoiseScheduler:
    # 离散噪声调度器
    def __init__(self, noise_schedule: str = "cosine", timesteps: int = 1000):
        self.timesteps = timesteps

        if noise_schedule == "cosine":
            betas = self._cosine_beta_schedule_discrete(timesteps)
        else:
            raise NotImplementedError(noise_schedule)

        self.register_buffer("betas", torch.from_numpy(betas).float())
        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)

        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar)

    def register_buffer(self, name: str, tensor: torch.Tensor):
        setattr(self, name, tensor)

    def _cosine_beta_schedule_discrete(self, timesteps: int) -> np.ndarray:
        s = 0.008
        steps = timesteps + 1
        x = np.linspace(0, timesteps, steps)
        alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, a_min=0, a_max=0.999)

    def get_alpha_bar(self, t_normalized: torch.Tensor = None, t_int: torch.Tensor = None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        t_int = torch.clamp(t_int, 0, self.timesteps - 1)
        alpha_bar = self.alphas_bar.to(t_int.device)[t_int.long()]
        alpha_bar = torch.clamp(alpha_bar, min=1e-8, max=1.0 - 1e-8)
        return alpha_bar


class DiscreteTransitionModel:
    # 离散状态转换模型

    def __init__(self, bond_classes: int):
        self.bond_classes = bond_classes
        self.u_e = (
            torch.ones(1, bond_classes, bond_classes) / bond_classes
            if bond_classes > 0
            else torch.zeros(1, bond_classes, bond_classes)
        )

    def get_Qt(self, beta_t: torch.Tensor, device: torch.device):
        beta_t = beta_t.unsqueeze(1).unsqueeze(1).to(device)
        self.u_e = self.u_e.to(device)
        batch_size = beta_t.size(0)
        eye_matrix = torch.eye(self.bond_classes, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        u_e_expanded = self.u_e.expand(batch_size, -1, -1)
        q_e = beta_t * u_e_expanded + (1 - beta_t) * eye_matrix
        return q_e

    def get_Qt_bar(self, alpha_bar_t: torch.Tensor, device: torch.device):
        alpha_bar_t = alpha_bar_t.unsqueeze(1).unsqueeze(1).to(device)
        self.u_e = self.u_e.to(device)
        batch_size = alpha_bar_t.size(0)
        eye_matrix = torch.eye(self.bond_classes, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        u_e_expanded = self.u_e.expand(batch_size, -1, -1)
        q_e = alpha_bar_t * eye_matrix + (1 - alpha_bar_t) * u_e_expanded
        return q_e


class ConditionalInformationBottleneck(nn.Module):
    """条件信息瓶颈模块，基于骨架信息压缩光谱"""

    def __init__(self, spectrum_dim: int, skeleton_dim: int, bottleneck_dim: int = 128, beta: float = 1e-3):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.beta = beta

        # 骨架信息编码器
        self.skeleton_encoder = nn.Sequential(
            nn.Linear(skeleton_dim, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 128)
        )

        # 条件编码器 - 基于骨架信息对光谱进行编码
        self.conditional_encoder = nn.Sequential(
            nn.Linear(spectrum_dim + 128, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # 变分参数预测
        self.mu_head = nn.Linear(256, bottleneck_dim)
        self.logvar_head = nn.Linear(256, bottleneck_dim)

        # 重构解码器
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim + 128, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, spectrum_dim),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, spectrum: torch.Tensor, skeleton_features: torch.Tensor, training: bool = True):
        batch_size = spectrum.shape[0]

        # 编码骨架信息
        skeleton_h = self.skeleton_encoder(skeleton_features)  # [B, 128]

        # 条件编码 - 将光谱和骨架信息结合
        combined_input = torch.cat([spectrum, skeleton_h], dim=-1)  # [B, spectrum_dim + 128]
        encoded = self.conditional_encoder(combined_input)  # [B, 256]

        # 变分参数
        mu = self.mu_head(encoded)  # [B, bottleneck_dim]
        logvar = self.logvar_head(encoded)  # [B, bottleneck_dim]

        if training:
            # 训练时使用重参数化
            z = self.reparameterize(mu, logvar)
        else:
            # 推理时直接使用均值
            z = mu

        # 重构光谱
        decoder_input = torch.cat([z, skeleton_h], dim=-1)
        reconstructed = self.decoder(decoder_input)

        # 计算KL散度损失
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        # 重构损失
        recon_loss = F.mse_loss(reconstructed, spectrum, reduction="mean")

        # 总的信息瓶颈损失
        ib_loss = recon_loss + self.beta * kl_loss

        return z, ib_loss, {"kl_loss": kl_loss, "recon_loss": recon_loss}


class ImprovedSpectrumEncoder(nn.Module):
    # 光谱编码器

    def __init__(
        self,
        spectrum_dim: int = 600,
        hidden_dim: int = 512,
        output_dim: int = 512,
        use_information_bottleneck: bool = True,
        skeleton_dim: int = None,
    ):
        super().__init__()
        self.spectrum_dim = spectrum_dim
        self.hidden_dim = hidden_dim
        self.use_information_bottleneck = use_information_bottleneck

        # 信息瓶颈模块
        if use_information_bottleneck and skeleton_dim is not None:
            self.info_bottleneck = ConditionalInformationBottleneck(
                spectrum_dim=spectrum_dim, skeleton_dim=skeleton_dim, bottleneck_dim=128
            )
            # 调整后续网络输入维度
            spectrum_input_dim = 128  # 使用压缩后的特征
        else:
            self.info_bottleneck = None
            spectrum_input_dim = spectrum_dim

        self.conv_blocks = nn.ModuleList(
            [
                self._make_conv_block(1, 64, kernel_size=15, padding=7),
                self._make_conv_block(64, 128, kernel_size=11, padding=5),
                self._make_conv_block(128, 256, kernel_size=7, padding=3),
                self._make_conv_block(256, 512, kernel_size=5, padding=2),
            ]
        )
        self.pos_encoding = self._create_positional_encoding(spectrum_input_dim, hidden_dim)
        self.input_projection = nn.Linear(1, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.attention_pool = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=0.1, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def _make_conv_block(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def _create_positional_encoding(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(
        self, spectrum: torch.Tensor, skeleton_features: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        batch_size = spectrum.shape[0]

        # 应用信息瓶颈
        ib_loss = None
        ib_metrics = None
        if self.use_information_bottleneck and self.info_bottleneck is not None and skeleton_features is not None:
            compressed_spectrum, ib_loss, ib_metrics = self.info_bottleneck(
                spectrum, skeleton_features, training=self.training
            )
            # 使用压缩后的光谱特征
            spectrum_to_use = compressed_spectrum
            # 为transformer调整维度
            trans_input = spectrum_to_use.unsqueeze(-1)  # [B, 128, 1]
        else:
            spectrum_to_use = spectrum
            trans_input = spectrum.unsqueeze(-1)  # [B, 600, 1]

        # CNN分支 - 原始光谱
        conv_input = spectrum.unsqueeze(1)  # [B, 1, 600]
        conv_features = conv_input
        for conv_block in self.conv_blocks:
            conv_features = conv_block(conv_features)
        conv_features = F.adaptive_avg_pool1d(conv_features, 1).squeeze(-1)  # [B, 512]

        # Transformer分支 - 可能是压缩后的光谱
        trans_input = self.input_projection(trans_input)  # [B, seq_len, hidden_dim]
        trans_input = trans_input + self.pos_encoding.to(trans_input.device)[: trans_input.size(1)].unsqueeze(0)
        trans_features = self.transformer(trans_input)  # [B, seq_len, hidden_dim]
        pool_query = self.pool_query.expand(batch_size, -1, -1)
        pooled_features, _ = self.attention_pool(pool_query, trans_features, trans_features)
        pooled_features = pooled_features.squeeze(1)  # [B, hidden_dim]

        combined_features = torch.cat([conv_features, pooled_features], dim=-1)
        output = self.fusion_mlp(combined_features)

        return output, ib_loss, ib_metrics


class ImprovedGraphAttentionLayer(nn.Module):
    # 改进的图注意力层，借鉴NodeEdgeBlock的设计"""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, n_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        assert hidden_dim % n_heads == 0, f"hidden_dim {hidden_dim} must be divisible by n_heads {n_heads}"

        # 节点特征处理
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # 边特征处理
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.edge_gate = nn.Linear(hidden_dim, hidden_dim)

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_output_proj = nn.Linear(hidden_dim, edge_dim)

        # 归一化和dropout
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        adj_mask: torch.Tensor,
        condition_h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, _ = node_features.shape

        # 节点特征投影
        h = self.node_proj(node_features)  # [B, N, H]
        if condition_h is not None:
            h = h + condition_h.unsqueeze(1)
        residual = h

        # 计算注意力
        Q = self.q_proj(h).view(batch_size, num_nodes, self.n_heads, self.head_dim)
        K = self.k_proj(h).view(batch_size, num_nodes, self.n_heads, self.head_dim)
        V = self.v_proj(h).view(batch_size, num_nodes, self.n_heads, self.head_dim)

        # 重排维度用于注意力计算
        Q = Q.transpose(1, 2)  # [B, H, N, D]
        K = K.transpose(1, 2)  # [B, H, N, D]
        V = V.transpose(1, 2)  # [B, H, N, D]

        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 边特征调制注意力
        edge_h = self.edge_proj(edge_features)  # [B, N, N, H]
        edge_h = edge_h.view(batch_size, num_nodes, num_nodes, self.n_heads, self.head_dim)
        edge_h = edge_h.permute(0, 3, 1, 2, 4)  # [B, H, N, N, D]

        # 应用边特征门控
        edge_gate = torch.sigmoid(self.edge_gate(edge_features))  # [B, N, N, H]
        edge_gate = edge_gate.view(batch_size, num_nodes, num_nodes, self.n_heads, self.head_dim)
        edge_gate = edge_gate.permute(0, 3, 1, 2, 4)  # [B, H, N, N, D]

        # 边特征增强注意力
        edge_attn = torch.sum(edge_h * edge_gate, dim=-1)  # [B, H, N, N]
        attn_scores = attn_scores + edge_attn

        # 应用掩码
        adj_mask_expanded = adj_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        attn_scores = attn_scores.masked_fill(adj_mask_expanded == 0, -1e9)

        # Softmax和dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力
        attn_output = torch.matmul(attn_weights, V)  # [B, H, N, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.hidden_dim)

        # 残差连接和归一化
        h = self.norm1(residual + self.dropout(self.output_proj(attn_output)))

        # 更新边特征
        edge_update = self._update_edge_features(h, edge_features, attn_weights)
        edge_features = self.norm2(edge_features + self.dropout(self.edge_output_proj(edge_update)))

        return h, edge_features

    def _update_edge_features(
        self, node_features: torch.Tensor, edge_features: torch.Tensor, attn_weights: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_nodes, hidden_dim = node_features.shape

        # 使用注意力权重来更新边特征
        # 聚合邻居节点信息
        attn_weights_mean = attn_weights.mean(dim=1)  # [B, N, N]

        # 计算边的源节点和目标节点特征
        src_features = node_features.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # [B, N, N, H]
        tgt_features = node_features.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # [B, N, N, H]

        # 结合注意力权重
        attn_weighted = attn_weights_mean.unsqueeze(-1)  # [B, N, N, 1]
        edge_update = attn_weighted * (src_features + tgt_features) / 2

        return edge_update


# ============================================================================
# 5. 改进的图Transformer (借鉴 transformer_model.py)
# ============================================================================


class ImprovedGraphTransformer(nn.Module):
    """改进的图Transformer，借鉴GraphTransformer的设计"""

    def __init__(
        self,
        node_vocab_size: int,
        edge_vocab_size: int,
        spectrum_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # 节点和边嵌入
        self.node_embedding = nn.Embedding(node_vocab_size, hidden_dim)
        self.edge_embedding = nn.Embedding(edge_vocab_size, hidden_dim)

        # 光谱编码器 - 启用信息瓶颈
        self.spectrum_encoder = ImprovedSpectrumEncoder(
            spectrum_dim=spectrum_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            use_information_bottleneck=True,
            skeleton_dim=hidden_dim,  # 骨架特征维度
        )

        # 时间嵌入
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # 图注意力层
        self.graph_layers = nn.ModuleList(
            [
                ImprovedGraphAttentionLayer(
                    node_dim=hidden_dim if i > 0 else hidden_dim,
                    edge_dim=hidden_dim if i > 0 else hidden_dim,
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                )
                for i in range(n_layers)
            ]
        )
        self.fusion_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        # 输出层
        self.edge_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, edge_vocab_size),
        )

    def _extract_skeleton_features(self, node_features: torch.Tensor, scaffold_mask: torch.Tensor) -> torch.Tensor:
        """从节点特征和骨架掩码中提取骨架特征"""
        # 嵌入节点特征
        node_h = self.node_embedding(node_features)  # [B, N, H]

        # 使用骨架掩码提取骨架节点的平均特征
        # scaffold_mask: [B, N, N] -> 提取对角线表示节点是否为骨架
        skeleton_node_mask = torch.diagonal(scaffold_mask, dim1=-2, dim2=-1)  # [B, N]
        skeleton_node_mask = skeleton_node_mask.unsqueeze(-1)  # [B, N, 1]

        # 加权平均
        masked_features = node_h * skeleton_node_mask
        skeleton_features = masked_features.sum(dim=1) / (skeleton_node_mask.sum(dim=1) + 1e-8)  # [B, H]

        return skeleton_features

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        adj_mask: torch.Tensor,
        spectrum: torch.Tensor,
        timestep: torch.Tensor,
        scaffold_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        batch_size, num_nodes = node_features.shape

        # 嵌入特征
        h = self.node_embedding(node_features)  # [B, N, H]
        edge_h = self.edge_embedding(edge_features)  # [B, N, N, H]

        # 提取骨架特征
        skeleton_features = None
        if scaffold_mask is not None:
            skeleton_features = self._extract_skeleton_features(node_features, scaffold_mask)

        # 编码光谱（带信息瓶颈）
        spectrum_h, ib_loss, ib_metrics = self.spectrum_encoder(spectrum, skeleton_features)  # [B, H]

        # 时间嵌入
        time_h = self.time_embedding(timestep.float().unsqueeze(-1))  # [B, H]
        # 将时间和光谱信息融合：拼接后投影，避免尺度问题
        combined_h = torch.cat([spectrum_h, time_h], dim=-1)  # [B, H*2]
        condition_h = self.fusion_projection(combined_h)
        condition_h = self.fusion_norm(condition_h)  # [B, H]
        # 多层图处理
        for layer in self.graph_layers:
            h_residual = h
            edge_h_residual = edge_h

            # 图注意力
            h, edge_h = layer(h, edge_h, adj_mask, condition_h)

        # 边预测
        # 构建边特征：源节点+目标节点+原始边特征+条件信息
        src_h = h.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # [B, N, N, H]
        tgt_h = h.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # [B, N, N, H]

        edge_input = src_h + tgt_h + edge_h
        edge_logits = self.edge_output(edge_input)  # [B, N, N, E]

        return edge_logits, ib_loss, ib_metrics


# ============================================================================
# 6. 主扩散模型 (借鉴 diffusion_model_spec2mol.py)
# ============================================================================


class ImprovedMolecularDiffusionModel(pl.LightningModule):
    """改进的分子扩散模型"""

    def __init__(
        self,
        node_vocab_size: int,
        edge_vocab_size: int,
        spectrum_dim: int = 600,
        hidden_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        timesteps: int = 1000,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        ib_weight: float = 0.1,  # 信息瓶颈损失权重
    ):
        super().__init__()
        self.save_hyperparameters()

        self.timesteps = timesteps
        self.edge_vocab_size = edge_vocab_size
        self.ib_weight = ib_weight

        # 核心模型
        self.model = ImprovedGraphTransformer(
            node_vocab_size=node_vocab_size,
            edge_vocab_size=edge_vocab_size,
            spectrum_dim=spectrum_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
        )

        # 噪声调度器
        self.noise_scheduler = DiscreteNoiseScheduler("cosine", timesteps)

        # 转换模型
        self.transition_model = DiscreteTransitionModel(edge_vocab_size)

        # 损失权重
        self.loss_weights = nn.Parameter(torch.ones(edge_vocab_size), requires_grad=False)

    def add_noise(self, edges: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """添加离散噪声"""
        batch_size = edges.shape[0]
        device = edges.device

        # 获取转换矩阵
        alpha_bar_t = self.noise_scheduler.get_alpha_bar(t_int=timesteps)
        Qt_bar = self.transition_model.get_Qt_bar(alpha_bar_t, device)

        # 转换为one-hot
        edges_onehot = F.one_hot(edges, num_classes=self.edge_vocab_size).float()

        # 计算转换概率
        batch_size, num_nodes = edges.shape[:2]
        edges_flat = edges_onehot.view(batch_size, -1, self.edge_vocab_size)

        # 应用转换矩阵
        Qt_bar_expanded = Qt_bar.expand(batch_size, -1, -1)
        probs = torch.bmm(edges_flat, Qt_bar_expanded)
        probs = probs.view(batch_size, num_nodes, num_nodes, self.edge_vocab_size)

        # 修复概率分布
        probs = torch.clamp(probs, min=1e-8)  # 避免负数和零概率
        probs = probs / probs.sum(dim=-1, keepdim=True)  # 确保概率和为1

        # 检查是否有NaN或Inf
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print("Warning: Invalid probabilities detected, using uniform distribution")
            probs = torch.ones_like(probs) / self.edge_vocab_size

        # 采样 - 使用更安全的方式
        probs_flat = probs.view(-1, self.edge_vocab_size)

        # 进一步确保概率分布有效
        probs_flat = torch.where(
            probs_flat.sum(dim=-1, keepdim=True) > 0, probs_flat, torch.ones_like(probs_flat) / self.edge_vocab_size
        )

        try:
            noisy_edges = torch.multinomial(probs_flat, 1, replacement=True)
        except RuntimeError as e:
            print(f"Multinomial sampling failed: {e}")
            # 降级到argmax采样
            noisy_edges = torch.argmax(probs_flat, dim=-1, keepdim=True)

        noisy_edges = noisy_edges.view(batch_size, num_nodes, num_nodes)

        return noisy_edges

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        node_features = batch["node_features"]
        target_edges = batch["adjacency_matrix"]
        scaffold_mask = batch["scaffold_mask"]
        spectrum = batch["spectrum"]

        batch_size = node_features.shape[0]
        device = node_features.device

        # 随机采样时间步
        timesteps = torch.randint(1, self.timesteps + 1, (batch_size,), device=device)

        # 添加噪声
        noisy_edges = self.add_noise(target_edges, timesteps)

        # 应用骨架掩码 (保持骨架部分不变)
        noisy_edges = torch.where(scaffold_mask.bool(), target_edges, noisy_edges)

        # 创建邻接掩码 (用于图注意力)
        adj_mask = (noisy_edges > 0).float()

        # 模型预测（包含信息瓶颈）
        predicted_logits, ib_loss, ib_metrics = self.model(
            node_features=node_features,
            edge_features=noisy_edges,
            adj_mask=adj_mask,
            spectrum=spectrum,
            timestep=timesteps,
            scaffold_mask=scaffold_mask,
        )

        # 计算损失 (只在非骨架位置)
        loss_mask = (1 - scaffold_mask).float()

        # 使用标签平滑的交叉熵损失
        prediction_loss = F.cross_entropy(
            predicted_logits.view(-1, self.edge_vocab_size),
            target_edges.view(-1),
            reduction="none",
            label_smoothing=0.1,
            weight=self.loss_weights,
        )

        prediction_loss = prediction_loss.view(target_edges.shape)
        masked_loss = prediction_loss * loss_mask
        final_prediction_loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)

        # 总损失：预测损失 + 信息瓶颈损失
        total_loss = final_prediction_loss
        if ib_loss is not None:
            total_loss = total_loss + self.ib_weight * ib_loss

        # 记录指标
        with torch.no_grad():
            pred_edges = predicted_logits.argmax(dim=-1)
            accuracy = ((pred_edges == target_edges).float() * loss_mask).sum() / (loss_mask.sum() + 1e-8)

        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_pred_loss", final_prediction_loss, prog_bar=True)
        if ib_loss is not None:
            self.log("train_ib_loss", ib_loss, prog_bar=True)
            if ib_metrics:
                self.log("train_kl_loss", ib_metrics["kl_loss"], prog_bar=False)
                self.log("train_recon_loss", ib_metrics["recon_loss"], prog_bar=False)
        self.log("train_acc", accuracy, prog_bar=True)

        return total_loss

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        node_features = batch["node_features"]
        target_edges = batch["adjacency_matrix"]
        scaffold_mask = batch["scaffold_mask"]
        spectrum = batch["spectrum"]

        batch_size = node_features.shape[0]
        device = node_features.device

        # 使用固定时间步进行验证
        timesteps = torch.full((batch_size,), self.timesteps // 2, device=device)

        # 添加噪声
        noisy_edges = self.add_noise(target_edges, timesteps)
        noisy_edges = torch.where(scaffold_mask.bool(), target_edges, noisy_edges)

        adj_mask = (noisy_edges > 0).float()

        # 模型预测
        predicted_logits, ib_loss, ib_metrics = self.model(
            node_features=node_features,
            edge_features=noisy_edges,
            adj_mask=adj_mask,
            spectrum=spectrum,
            timestep=timesteps,
            scaffold_mask=scaffold_mask,
        )

        # 计算损失
        loss_mask = (1 - scaffold_mask).float()
        prediction_loss = F.cross_entropy(
            predicted_logits.view(-1, self.edge_vocab_size), target_edges.view(-1), reduction="none"
        )

        prediction_loss = prediction_loss.view(target_edges.shape)
        masked_loss = prediction_loss * loss_mask
        final_prediction_loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)

        # 总损失
        total_loss = final_prediction_loss
        if ib_loss is not None:
            total_loss = total_loss + self.ib_weight * ib_loss

        # 计算准确率
        pred_edges = predicted_logits.argmax(dim=-1)
        accuracy = ((pred_edges == target_edges).float() * loss_mask).sum() / (loss_mask.sum() + 1e-8)

        # 新增：计算整个分子的预测准确率
        # 只考虑非骨架部分，若所有非骨架边都预测正确，则该分子预测正确
        correct_matrix = ((pred_edges == target_edges) | scaffold_mask.bool()).float()
        # 对每个分子，所有元素都为1则为1，否则为0
        molecule_correct = correct_matrix.view(batch_size, -1).all(dim=1).float()
        molecule_accuracy = molecule_correct.mean()

        self.log("val_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("val_pred_loss", final_prediction_loss, prog_bar=True, sync_dist=True)
        if ib_loss is not None:
            self.log("val_ib_loss", ib_loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", accuracy, prog_bar=True, sync_dist=True)
        self.log("val_molecule_acc", molecule_accuracy, prog_bar=True, sync_dist=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, betas=(0.9, 0.999), eps=1e-8
        )

        # 使用余弦退火学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-7)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1}}

    @torch.no_grad()
    def sample(
        self,
        node_features: torch.Tensor,
        scaffold_edges: torch.Tensor,
        scaffold_mask: torch.Tensor,
        spectrum: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """从噪声采样分子"""
        if num_steps is None:
            num_steps = self.timesteps

        batch_size, num_nodes = node_features.shape[:2]
        device = node_features.device

        # 从噪声开始
        edges = torch.randint(0, self.edge_vocab_size, (batch_size, num_nodes, num_nodes), device=device)
        edges = torch.where(scaffold_mask.bool(), scaffold_edges, edges)

        for t in tqdm(reversed(range(1, num_steps + 1)), desc="Sampling"):
            timesteps = torch.full((batch_size,), t, device=device)

            adj_mask = (edges > 0).float()

            # 模型预测
            predicted_logits, _, _ = self.model(
                node_features=node_features,
                edge_features=edges,
                adj_mask=adj_mask,
                spectrum=spectrum,
                timestep=timesteps,
                scaffold_mask=scaffold_mask,
            )

            if t > 1:
                # DDPM逆向步骤
                # 这里简化为直接使用预测的最大概率类别
                # 在实际实现中，可能需要更复杂的逆向采样过程
                edges = predicted_logits.argmax(dim=-1)
            else:
                # 最后一步
                edges = predicted_logits.argmax(dim=-1)

            # 重新应用骨架掩码
            edges = torch.where(scaffold_mask.bool(), scaffold_edges, edges)

        return edges


# ============================================================================
# 7. 数据处理和训练脚本
# ============================================================================


def create_improved_dataloader(processed_data: List[Dict], batch_size: int = 16):
    """创建改进的数据加载器"""
    from torch.utils.data import Dataset, DataLoader

    class ImprovedMoleculeDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            return {
                "node_features": torch.tensor(item["node_features"], dtype=torch.long),
                "adjacency_matrix": torch.tensor(item["adjacency_matrix"], dtype=torch.long),
                "scaffold_mask": torch.tensor(item["scaffold_mask"], dtype=torch.float),
                "spectrum": torch.tensor(item["spectrum"], dtype=torch.float),
            }

    def collate_fn(batch):
        max_nodes = max([item["node_features"].size(0) for item in batch])

        padded_batch = {"node_features": [], "adjacency_matrix": [], "scaffold_mask": [], "spectrum": []}

        for item in batch:
            num_nodes = item["node_features"].size(0)
            pad_size = max_nodes - num_nodes

            # 填充
            padded_nodes = F.pad(item["node_features"], (0, pad_size), value=0)
            padded_adj = F.pad(item["adjacency_matrix"], (0, pad_size, 0, pad_size), value=0)
            padded_mask = F.pad(item["scaffold_mask"], (0, pad_size, 0, pad_size), value=1)

            padded_batch["node_features"].append(padded_nodes)
            padded_batch["adjacency_matrix"].append(padded_adj)
            padded_batch["scaffold_mask"].append(padded_mask)
            padded_batch["spectrum"].append(item["spectrum"])

        return {
            "node_features": torch.stack(padded_batch["node_features"]),
            "adjacency_matrix": torch.stack(padded_batch["adjacency_matrix"]),
            "scaffold_mask": torch.stack(padded_batch["scaffold_mask"]),
            "spectrum": torch.stack(padded_batch["spectrum"]),
        }

    dataset = ImprovedMoleculeDataset(processed_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)


def main():
    """主训练函数"""
    # 设置固定随机种子以确保可重现性
    seed = 17
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random

    batch_size = 32
    data_dir = "./processed_massspecgymR/"
    logl_dir = "version1_v4_e/"
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 设置日志记录
    log_dir = data_dir + logl_dir + "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 创建日志记录器
    logger = logging.getLogger("molecular_diffusion")
    logger.setLevel(logging.INFO)

    # 控制台处理器
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(stream_formatter)

    # 文件处理器
    file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # 添加处理器到日志记录器
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.info("开始程序执行")

    # 加载数据
    logger.info("加载数据...")
    processed_data = torch.load(data_dir + "processed_data.pt")

    with open(data_dir + "atom_dict.json", "r") as f:
        atom_dict = json.load(f)

    with open(data_dir + "bond_dict.json", "r") as f:
        bond_dict = json.load(f)

    logger.info(f"数据集大小: {len(processed_data)}")
    logger.info(f"原子类型数: {len(atom_dict)}")
    logger.info(f"键类型数: {len(bond_dict)}")

    # 创建数据加载器并保存拆分的数据集
    train_size = int(0.9 * len(processed_data))
    val_size = len(processed_data) - train_size

    # 使用固定随机种子拆分数据集
    train_data, val_data = torch.utils.data.random_split(
        processed_data, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )

    # 保存拆分后的数据集，方便复现
    split_dir = data_dir + logl_dir + "data_split"
    os.makedirs(split_dir, exist_ok=True)

    with open(os.path.join(split_dir, "train_indices.json"), "w") as f:
        json.dump(train_data.indices, f)

    with open(os.path.join(split_dir, "val_indices.json"), "w") as f:
        json.dump(val_data.indices, f)

    logger.info(f"数据集拆分完成，训练集大小: {len(train_data)}，验证集大小: {len(val_data)}")
    logger.info(f"数据集拆分索引已保存至 {split_dir} 目录")

    train_dataloader = create_improved_dataloader(list(train_data), batch_size=batch_size)
    val_dataloader = create_improved_dataloader(list(val_data), batch_size=batch_size)

    # 创建模型
    model = ImprovedMolecularDiffusionModel(
        node_vocab_size=len(atom_dict),
        edge_vocab_size=len(bond_dict),
        spectrum_dim=600,
        hidden_dim=512,
        n_layers=6,
        n_heads=8,
        timesteps=1000,
        lr=1e-4,
        weight_decay=0.0001,
        ib_weight=0.6,  # 信息瓶颈损失权重
    )

    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练设置
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger

    # 自定义回调函数，用于记录每个epoch的效果
    class LoggingCallback(pl.Callback):
        def __init__(self, logger):
            super().__init__()
            self.logger = logger

        def on_train_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            epoch = trainer.current_epoch

            log_message = f"Epoch {epoch} 完成:"
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                log_message += f" {k}={v:.4f}"

            self.logger.info(log_message)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", filename="improved-diffusion-{epoch:02d}-{val_loss:.2f}", save_top_k=3, mode="min"
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=100, mode="min")

    lr_monitor = LearningRateMonitor(logging_interval="step")
    logging_callback = LoggingCallback(logger)

    tb_logger = TensorBoardLogger("tb_logs", name="improved_diffusion")

    # 训练器 - 支持多GPU训练
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",  # 使用所有可用的GPU
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",  # 使用DDP进行多GPU训练
        callbacks=[checkpoint_callback, early_stopping, lr_monitor, logging_callback],
        logger=tb_logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        precision=32,
        log_every_n_steps=10,
    )

    # 开始训练
    logger.info("开始训练...")
    trainer.fit(model, train_dataloader, val_dataloader)

    logger.info("训练完成！")


if __name__ == "__main__":
    main()

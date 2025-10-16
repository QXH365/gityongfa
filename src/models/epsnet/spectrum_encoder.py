# models/epsnet/spectrum_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# SpectralCIB 模块与之前版本保持一致，因为它负责的是后端的信息蒸馏，与前端特征提取解耦
class SpectralCIB(nn.Module):
    """
    条件信息瓶颈 (Conditional Information Bottleneck) 模块。
    使用 Transformer 架构，将特征序列压缩为一组服从高斯分布的低维潜变量（“光谱概念”）。
    """

    def __init__(self, embed_dim=128, num_heads=4, num_layers=2, num_concepts=8):
        super().__init__()
        self.num_concepts = num_concepts

        # Transformer Encoder 用于捕捉 patch 序列之间的内部相关性
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            dropout=0.1,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 可学习的 "概念" token，作为信息压缩的“容器”
        self.concept_tokens = nn.Parameter(torch.randn(1, num_concepts, embed_dim))

        # 交叉注意力模块，将 patch 序列的信息“注入”到 concept token 中
        self.compress_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # 线性层，将交叉注意力的输出映射为高斯分布的均值 (mu) 和对数方差 (log_var)
        self.fc_mu = nn.Linear(embed_dim, embed_dim)
        self.fc_log_var = nn.Linear(embed_dim, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """重参数化技巧，使得从分布中采样的过程可导。"""
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, feature_sequence: torch.Tensor, condition_vector=None):
        """
        返回:
            - z (torch.Tensor): 从潜变量分布中采样得到的光谱概念。(B, NumConcepts, EmbedDim)
            - kl_loss (torch.Tensor): 潜变量分布与标准正态分布之间的KL散度损失。
        """
        batch_size = feature_sequence.size(0)

        # 1. 特征序列内部的自注意力
        processed_sequence = self.transformer_encoder(feature_sequence)

        # 2. 交叉注意力：用 concept_tokens 去查询 processed_sequence
        compressed_output, _ = self.compress_attention(
            query=self.concept_tokens.expand(batch_size, -1, -1), key=processed_sequence, value=processed_sequence
        )
        compressed_output = self.norm(compressed_output)

        # 3. 将压缩后的输出映射为高斯分布的参数
        mu = self.fc_mu(compressed_output)
        log_var = self.fc_log_var(compressed_output)

        # 4. 使用重参数化技巧进行采样
        z = self.reparameterize(mu, log_var)

        # 5. 计算 KL 散度损失
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss / batch_size

        return z, kl_loss


class CNNBranch(nn.Module):
    """
    用于处理单一光谱（IR或Raman）的CNN分支，实现高保真局部特征提取。
    """

    def __init__(self, cnn_out_channels=64, kernel_size=11, stride=2):
        super().__init__()

        # 【修正】: 手动计算 padding 来替代 'same'
        # 当 stride=2 时, padding = (kernel_size - 1) // 2 可以很好地模拟'same'并使长度减半
        # 对于 kernel_size=11, padding = (11-1)//2 = 5
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(1, 32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, cnn_out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(cnn_out_channels)

    def forward(self, x):
        # x: [B, 1, SeqLen]
        # 经过两层 stride=2 的卷积，序列长度变为 SeqLen / 4
        features = F.relu(self.bn1(self.conv1(x)))
        features = F.relu(self.bn2(self.conv2(features)))
        # 输出: [B, cnn_out_channels, SeqLen / 4]
        return features


class Spectroformer(nn.Module):
    """
    全新的光谱编码器，实现了分层的、高分辨率的特征提取与抽象。
    流程: (CNN -> Concat) -> Patchify -> Project -> Transformer -> CIB
    """

    def __init__(self, config):
        super().__init__()

        # --- 参数配置 ---
        ir_len = config.model.get("target_ir_len", 3500)
        raman_len = config.model.get("target_raman_len", 3500)

        cnn_out_channels = config.model.get("spec_cnn_out_channels", 64)
        cnn_kernel_size = config.model.get("spec_cnn_kernel_size", 11)
        cnn_stride = 2
        num_cnn_layers_with_stride = 2

        # 重新计算拼接长度: (3500/4) + (3500/4) = 875 + 875 = 1750
        concat_len = (ir_len // (cnn_stride**num_cnn_layers_with_stride)) + (
            raman_len // (cnn_stride**num_cnn_layers_with_stride)
        )

        # 重新计算分片大小: 1750 / 70 = 25. 最终序列长度为 70.
        patch_size = config.model.get("spec_patch_size", 25)
        if concat_len % patch_size != 0:
            raise ValueError(f"拼接后的CNN特征长度({concat_len})无法被patch_size({patch_size})整除。")

        num_patches = concat_len // patch_size
        patch_dim = cnn_out_channels * patch_size

        embed_dim = config.model.get("spec_embed_dim", 128)

        # --- 模型层定义 ---
        self.cnn_ir = CNNBranch(cnn_out_channels, cnn_kernel_size, cnn_stride)
        self.cnn_raman = CNNBranch(cnn_out_channels, cnn_kernel_size, cnn_stride)

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_dim = patch_dim

        # 线性层，负责将每个展平的光谱分片映射到高维嵌入空间
        self.patch_projection = nn.Linear(patch_dim, embed_dim)

        # 可学习的位置编码
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        # 条件信息瓶颈模块
        self.cib = SpectralCIB(
            embed_dim=embed_dim,
            num_heads=config.model.spec_num_heads,
            num_layers=config.model.spec_num_layers,
            num_concepts=config.model.spec_num_concepts,
        )

    def forward(self, batch, condition_vector=None):
        # 1. 数据准备
        ir_spec = batch.ir_spectrum.view(batch.num_graphs, 1, -1)
        raman_spec = batch.raman_spectrum.view(batch.num_graphs, 1, -1)

        # 2. 双分支CNN高保真局部特征提取
        ir_features = self.cnn_ir(ir_spec)  # [B, C, 875]
        raman_features = self.cnn_raman(raman_spec)  # [B, C, 875]

        # 3. 特征序列拼接
        # [B, C, 1750]
        concat_features = torch.cat([ir_features, raman_features], dim=2)

        # 4. 分片 (Patching)
        # 调整维度以进行分片: [B, 1750, C]
        concat_features = concat_features.permute(0, 2, 1)

        B, L, C = concat_features.shape
        # [B, 70, 25, C]
        patches = concat_features.view(B, self.num_patches, self.patch_size, C)
        # 展平每个patch: [B, 70, 25 * C]
        patches_flattened = patches.flatten(2)

        # 5. 分片投影
        # [B, 70, embed_dim]
        patch_embeddings = self.patch_projection(patches_flattened)

        # 6. 添加位置编码
        patch_embeddings = patch_embeddings + self.position_embedding

        # 7. 信息瓶颈与概念提取
        spectral_concepts, kl_loss = self.cib(patch_embeddings, condition_vector)

        return spectral_concepts, kl_loss

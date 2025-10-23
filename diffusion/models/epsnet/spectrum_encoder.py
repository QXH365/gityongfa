# diffusion/models/epsnet/spectrum_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 辅助模块 ---

class PatchEmbedding(nn.Module):
    """将光谱序列分割成 patch 并进行线性投影"""
    def __init__(self, seq_len=3500, patch_size=25, embed_dim=128):
        super().__init__()
        if seq_len % patch_size != 0:
            raise ValueError(f"光谱长度 ({seq_len}) 必须能被 patch_size ({patch_size}) 整除。")
        self.num_patches = seq_len // patch_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # 线性投影层，将每个 patch 映射到 embed_dim
        self.projection = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        # x: [B, 1, SeqLen]
        B, C, L = x.shape
        # Reshape to patches: [B, NumPatches, PatchSize]
        patches = x.view(B, self.num_patches, self.patch_size)
        # Project patches: [B, NumPatches, EmbedDim]
        embeddings = self.projection(patches)
        return embeddings

class DisentanglementHead(nn.Module):
    """
    信息瓶颈头，用于从特征序列中提取一组解耦的潜变量 (z)。
    使用交叉注意力将信息压缩到 concept tokens，然后映射到高斯分布。
    """
    def __init__(self, embed_dim=128, num_heads=4, num_concepts=8):
        super().__init__()
        self.num_concepts = num_concepts
        self.embed_dim = embed_dim

        # 可学习的 "概念" token
        self.concept_tokens = nn.Parameter(torch.randn(1, num_concepts, embed_dim))

        # 交叉注意力模块
        self.compress_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(embed_dim)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # 线性层映射到高斯分布参数
        self.fc_mu = nn.Linear(embed_dim, embed_dim)
        self.fc_log_var = nn.Linear(embed_dim, embed_dim)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, feature_sequence: torch.Tensor):
        """
        Args:
            feature_sequence (torch.Tensor): 输入的特征序列 (B, SeqLen, EmbedDim)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: mu, log_var, z
                - mu (torch.Tensor): 潜变量均值 (B, NumConcepts, EmbedDim)
                - log_var (torch.Tensor): 潜变量对数方差 (B, NumConcepts, EmbedDim)
                - z (torch.Tensor): 采样得到的潜变量 (B, NumConcepts, EmbedDim)
        """
        batch_size = feature_sequence.size(0)

        # 1. 交叉注意力：用 concept_tokens 去查询 feature_sequence
        attn_output, _ = self.compress_attention(
            query=self.concept_tokens.expand(batch_size, -1, -1),
            key=feature_sequence,
            value=feature_sequence
        )
        # Add & Norm
        compressed_output = self.norm1(self.concept_tokens.expand(batch_size, -1, -1) + attn_output)

        # 2. 前馈网络
        ffn_output = self.ffn(compressed_output)
        # Add & Norm
        processed_output = self.norm2(compressed_output + ffn_output)

        # 3. 映射为高斯分布参数
        mu = self.fc_mu(processed_output)
        log_var = self.fc_log_var(processed_output)

        # 4. 重参数化采样
        z = self.reparameterize(mu, log_var)

        return mu, log_var, z

# --- KL 散度计算函数 ---

def kl_divergence_gaussian(mu1, log_var1, mu2, log_var2, reduce='mean'):
    """
    计算两组对角多元高斯分布之间的 KL 散度 KL(N(mu1, var1) || N(mu2, var2))。
    Args:
        mu1, log_var1: 第一组分布的参数 (B, NumConcepts, EmbedDim)
        mu2, log_var2: 第二组分布的参数 (B, NumConcepts, EmbedDim)
        reduce: 'mean' 或 'sum'，如何在 batch 维度上聚合 KL 散度。
    Returns:
        torch.Tensor: KL 散度值 (标量)。
    """
    var1 = torch.exp(log_var1)
    var2 = torch.exp(log_var2)
    kl_components = 0.5 * (
        log_var2 - log_var1 + (var1 + (mu1 - mu2)**2) / (var2 + 1e-8) - 1
    )
    # 在 concept 和 embed_dim 维度上求和
    kl_per_sample = torch.sum(kl_components, dim=[1, 2])

    if reduce == 'mean':
        return torch.mean(kl_per_sample)
    elif reduce == 'sum':
        return torch.sum(kl_per_sample)
    else:
        raise ValueError("reduce 参数必须是 'mean' 或 'sum'")

# --- 新的光谱编码器 ---

class SpectrumEncoderDisentangled(nn.Module):
    """
    使用 Patch Attention 和 KL 散度损失实现信息解耦的光谱编码器。
    流程: Patchify -> Embed -> Transformer -> Concat -> Disentanglement Heads -> KL Loss
    """
    def __init__(self, config):
        super().__init__()

        # --- 参数配置 ---
        # 从 config.model (EasyDict) 中获取参数，提供默认值
        seq_len = config.model.get('spec_seq_len', 3500)
        patch_size = config.model.get('spec_patch_size', 25) # 3500 / 25 = 140 patches
        embed_dim = config.model.get('spec_embed_dim', 128)
        num_heads_tf = config.model.get('spec_num_heads_tf', 8) # Transformer Encoder 头数
        num_layers_tf = config.model.get('spec_num_layers_tf', 4) # Transformer Encoder 层数
        num_heads_dis = config.model.get('spec_num_heads_dis', 4) # Disentanglement Head 头数
        num_concepts1 = config.model.get('spec_num_concepts1', 8) # z1 的概念数
        num_concepts2 = config.model.get('spec_num_concepts2', 8) # z2 的概念数
        dropout = config.model.get('spec_dropout', 0.1)

        # 检查光谱长度是否能被 patch_size 整除
        if seq_len % patch_size != 0:
            raise ValueError(f"光谱长度 ({seq_len}) 必须能被 patch_size ({patch_size}) 整除。")
        num_patches = seq_len // patch_size

        # --- 模型层定义 ---

        # 1. Patch Embedding (共享 IR 和 Raman)
        self.patch_embed = PatchEmbedding(seq_len, patch_size, embed_dim)

        # 2. 位置编码 (IR 和 Raman 分开)
        self.pos_embed_ir = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.pos_embed_raman = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.embed_dropout = nn.Dropout(dropout)

        # 3. Transformer Encoder (共享 IR 和 Raman)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads_tf, batch_first=True,
            dim_feedforward=embed_dim * 4, activation='gelu', dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers_tf)

        # 4. 信息解耦头 (两个独立实例)
        self.disentanglement_head1 = DisentanglementHead(embed_dim, num_heads_dis, num_concepts1)
        self.disentanglement_head2 = DisentanglementHead(embed_dim, num_heads_dis, num_concepts2)

    def forward(self, batch):
        # 1. 数据准备
        # 确保输入光谱形状为 [B, 1, SeqLen]
        ir_spec = batch.ir_spectrum.view(batch.num_graphs, 1, -1)
        raman_spec = batch.raman_spectrum.view(batch.num_graphs, 1, -1)

        # 2. Patching 和 Embedding
        ir_embeddings = self.patch_embed(ir_spec)   # [B, NumPatches, EmbedDim]
        raman_embeddings = self.patch_embed(raman_spec) # [B, NumPatches, EmbedDim]

        # 3. 添加位置编码
        ir_embeddings = self.embed_dropout(ir_embeddings + self.pos_embed_ir)
        raman_embeddings = self.embed_dropout(raman_embeddings + self.pos_embed_raman)

        # 4. 通过共享 Transformer Encoder
        processed_ir = self.transformer_encoder(ir_embeddings)    # [B, NumPatches, EmbedDim]
        processed_raman = self.transformer_encoder(raman_embeddings) # [B, NumPatches, EmbedDim]

        # 5. 特征序列拼接
        combined_features = torch.cat([processed_ir, processed_raman], dim=1) # [B, 2 * NumPatches, EmbedDim]

        # 6. 通过两个解耦头
        mu1, log_var1, z1 = self.disentanglement_head1(combined_features)
        mu2, log_var2, z2 = self.disentanglement_head2(combined_features)

        # 7. 计算 KL 散度损失 (最大化分离度)
        kl_12 = kl_divergence_gaussian(mu1, log_var1, mu2, log_var2, reduce='mean')
        kl_21 = kl_divergence_gaussian(mu2, log_var2, mu1, log_var1, reduce='mean')
        # 我们想最大化 KL，所以在总损失中添加 - (kl_12 + kl_21)
        # 注意：这里只计算损失值，不直接添加到模型的总损失中，由训练脚本处理
        kl_separation_loss = -(kl_12 + kl_21)

        # 8. (可选) 计算与标准正态分布的 KL，用于正则化每个 z (类似 VAE)
        # kl_prior1 = kl_divergence_gaussian(mu1, log_var1, torch.zeros_like(mu1), torch.zeros_like(log_var1))
        # kl_prior2 = kl_divergence_gaussian(mu2, log_var2, torch.zeros_like(mu2), torch.zeros_like(log_var2))
        # total_kl_prior = kl_prior1 + kl_prior2

        # 返回解耦的潜变量和分离损失
        # 训练脚本需要将 kl_separation_loss (乘以一个权重) 添加到总损失中
        return z1, z2, kl_separation_loss #, total_kl_prior # 如果需要正则化
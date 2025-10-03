# models/epsnet/spectroformer_encoder.py

import torch
import torch.nn as nn

class SpectrumPatcher(nn.Module):
    """
    将一维光谱数据(如IR+Raman)转换为一系列 patch embeddings。
    这个模块将长序列分解为多个可处理的局部特征块，类似于Vision Transformer处理图像的方式。
    """
    def __init__(self, seq_len=1200, patch_size=20, embed_dim=128):
        super().__init__()
        if seq_len % patch_size != 0:
            raise ValueError("序列长度必须能够被 patch 大小整除")
            
        self.num_patches = seq_len // patch_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 线性层，负责将每个展平的光谱片段映射到高维嵌入空间
        self.projection = nn.Linear(patch_size, embed_dim)
        # 可学习的位置编码参数，为每个 patch 嵌入添加其在原始光谱中的位置信息
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入 x 的形状: (BatchSize, 1, SeqLen)
        输出的形状: (BatchSize, NumPatches, EmbedDim)
        """
        # 1. 将连续的光谱序列分解为多个 patches
        # (B, 1, NumPatches * PatchSize) -> (B, NumPatches, PatchSize)
        patches = x.squeeze(1).view(
            -1, self.num_patches, self.patch_size
        )
        
        # 2. 将 patches 投影到嵌入空间并添加位置编码
        embedded_patches = self.projection(patches)
        embedded_patches = embedded_patches + self.position_embedding
        
        return embedded_patches

class SpectralCIB(nn.Module):
    """
    完整的条件信息瓶颈 (Conditional Information Bottleneck) 模块。
    使用 Transformer 架构，将光谱 patch 序列依据条件（可选）压缩为一组服从高斯分布的低维潜变量（“光谱概念”）。
    """
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2, num_concepts=8):
        super().__init__()
        self.num_concepts = num_concepts

        # Transformer Encoder 用于捕捉光谱 patch 之间的内部相关性
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True,
            dim_feedforward=embed_dim * 4, activation='gelu', dropout=0.1
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

    def forward(self, patch_embeddings: torch.Tensor, condition_vector=None):
        """
        返回:
            - z (torch.Tensor): 从潜变量分布中采样得到的光谱概念。(B, NumConcepts, EmbedDim)
            - kl_loss (torch.Tensor): 潜变量分布与标准正态分布之间的KL散度损失。
        """
        batch_size = patch_embeddings.size(0)
        
        # 1. 光谱 patch 序列内部的自注意力
        processed_patches = self.transformer_encoder(patch_embeddings)
        
        # 2. 交叉注意力：用 concept_tokens 去查询 processed_patches
        compressed_output, _ = self.compress_attention(
            query=self.concept_tokens.expand(batch_size, -1, -1),
            key=processed_patches,
            value=processed_patches
        )
        compressed_output = self.norm(compressed_output)
        
        # 3. 将压缩后的输出映射为高斯分布的参数
        mu = self.fc_mu(compressed_output)
        log_var = self.fc_log_var(compressed_output)
        
        # 4. 使用重参数化技巧进行采样
        z = self.reparameterize(mu, log_var)
        
        # 5. 计算 KL 散度损失 (信息瓶颈的核心)
        # 这个损失项会惩罚潜变量分布偏离标准正态分布太远，从而起到压缩和正则化的作用。
        # KL(q(z|x) || p(z)) where p(z) is N(0, I)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss / batch_size # 对批次大小进行平均
        
        return z, kl_loss

class Spectroformer(nn.Module):
    """
    完整的光谱编码器，整合了 Patcher 和 CIB 模块。
    """
    def __init__(self, config):
        super().__init__()
        # 从配置文件中读取参数，并提供默认值
        total_spec_len = config.get('total_spec_len', 1200)
        patch_size = config.get('spec_patch_size', 20)
        embed_dim = config.get('spec_embed_dim', 128)
        
        self.patcher = SpectrumPatcher(
            seq_len=total_spec_len, 
            patch_size=patch_size, 
            embed_dim=embed_dim
        )
        self.cib = SpectralCIB(
            embed_dim=embed_dim,
            num_heads=config.spec_num_heads,
            num_layers=config.spec_num_layers,
            num_concepts=config.spec_num_concepts
        )

    def forward(self, batch, condition_vector=None):
        """
        返回:
            - spectral_concepts (torch.Tensor): 最终的光谱概念表示。
            - kl_loss (torch.Tensor): 从 CIB 模块计算得到的 KL 散度损失。
        """
        # 假设 batch 对象中包含 ir_spectrum 和 raman_spectrum
        ir_spec = batch.ir_spectrum.view(batch.num_graphs, -1)
        raman_spec = batch.raman_spectrum.view(batch.num_graphs, -1)
        
        # 拼接成一个长序列并增加通道维度以匹配 Patcher 的输入
        combined_spec = torch.cat([ir_spec, raman_spec], dim=1).unsqueeze(1)

        patch_embeddings = self.patcher(combined_spec)
        spectral_concepts, kl_loss = self.cib(patch_embeddings, condition_vector)
        
        return spectral_concepts, kl_loss
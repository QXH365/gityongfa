import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
import math


class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GumbelSoftmaxSampler(nn.Module):
    def __init__(self, temperature: float = 1.0, hard: bool = False):
        super().__init__()
        self.temperature = temperature
        self.hard = hard
        
    def forward(self, logits: torch.Tensor, training: bool = True) -> torch.Tensor:
        if training:
            return F.gumbel_softmax(logits, tau=self.temperature, hard=self.hard, dim=-1)
        else:
            return F.softmax(logits / self.temperature, dim=-1)


class SpectralCompressionLayer(nn.Module):
    def __init__(self, spectrum_dim: int, compressed_dim: int, num_components: int = 32):
        super().__init__()
        self.num_components = num_components
        self.compressed_dim = compressed_dim
        
        self.frequency_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(num_components)
        )
        
        self.spectral_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=3
        )
        
        self.compression_head = nn.Sequential(
            nn.Linear(128 * num_components, compressed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(compressed_dim * 2, compressed_dim)
        )
        
    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        batch_size = spectrum.size(0)
        
        x = spectrum.unsqueeze(1)
        x = self.frequency_encoder(x)
        x = x.transpose(1, 2)
        
        x = self.spectral_transformer(x)
        x = x.reshape(batch_size, -1)
        
        compressed = self.compression_head(x)
        return compressed


class StructureAwareAttention(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0
        
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, skeleton_features: torch.Tensor, spectrum_features: torch.Tensor) -> torch.Tensor:
        batch_size = skeleton_features.size(0)
        
        q = self.q_proj(skeleton_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(spectrum_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(spectrum_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.feature_dim)
        
        output = self.out_proj(attn_output)
        return self.layer_norm(output + skeleton_features)


class ConditionalInformationBottleneckCore(nn.Module):
    def __init__(
        self,
        spectrum_dim: int,
        skeleton_dim: int,
        bottleneck_dim: int = 128,
        compressed_spectrum_dim: int = 64,
        beta1: float = 1e-3,
        beta2: float = 1e-4,
        gumbel_temperature: float = 1.0
    ):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.beta1 = beta1
        self.beta2 = beta2
        
        self.spectral_compressor = SpectralCompressionLayer(
            spectrum_dim, compressed_spectrum_dim
        )
        
        self.skeleton_encoder = nn.Sequential(
            nn.Linear(skeleton_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        self.structure_attention = StructureAwareAttention(128)
        
        self.gating_network = GatingNetwork(compressed_spectrum_dim + 128)
        
        self.conditional_encoder = nn.Sequential(
            nn.Linear(compressed_spectrum_dim + 128, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.mu_head = nn.Linear(256, bottleneck_dim)
        self.logvar_head = nn.Linear(256, bottleneck_dim)
        
        self.importance_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, compressed_spectrum_dim),
            nn.Sigmoid()
        )
        
        self.gumbel_sampler = GumbelSoftmaxSampler(gumbel_temperature)
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim + 128, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, spectrum_dim)
        )
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def compute_kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
    def forward(
        self, 
        spectrum: torch.Tensor, 
        skeleton_features: torch.Tensor, 
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        batch_size = spectrum.size(0)
        
        compressed_spectrum = self.spectral_compressor(spectrum)
        skeleton_h = self.skeleton_encoder(skeleton_features)
        
        spectrum_h_expanded = compressed_spectrum.unsqueeze(1)
        skeleton_h_expanded = skeleton_h.unsqueeze(1)
        attended_skeleton = self.structure_attention(skeleton_h_expanded, spectrum_h_expanded)
        attended_skeleton = attended_skeleton.squeeze(1)
        
        combined_features = torch.cat([compressed_spectrum, attended_skeleton], dim=-1)
        
        gate_scores = self.gating_network(combined_features)
        gated_features = combined_features * gate_scores
        
        encoded = self.conditional_encoder(gated_features)
        
        mu = self.mu_head(encoded)
        logvar = self.logvar_head(encoded)
        
        importance_scores = self.importance_predictor(encoded)
        
        if training:
            z = self.reparameterize(mu, logvar)
            importance_logits = torch.log(importance_scores + 1e-8) - torch.log(1 - importance_scores + 1e-8)
            discrete_mask = self.gumbel_sampler(importance_logits.unsqueeze(-1).expand(-1, -1, 2), training)
            discrete_mask = discrete_mask[:, :, 1]
        else:
            z = mu
            discrete_mask = (importance_scores > 0.5).float()
            
        masked_compressed = compressed_spectrum * discrete_mask
        
        decoder_input = torch.cat([z, attended_skeleton], dim=-1)
        reconstructed = self.decoder(decoder_input)
        
        kl_loss1 = self.compute_kl_divergence(mu, logvar).mean()
        
        prior_importance = torch.ones_like(importance_scores) * 0.5
        kl_loss2 = F.kl_div(
            torch.log(importance_scores + 1e-8),
            prior_importance,
            reduction='batchmean'
        )
        
        recon_loss = F.mse_loss(reconstructed, spectrum, reduction='mean')
        
        sparsity_loss = torch.mean(importance_scores)
        
        total_loss = recon_loss + self.beta1 * kl_loss1 + self.beta2 * kl_loss2 + 0.01 * sparsity_loss
        
        metrics = {
            'kl_loss1': kl_loss1,
            'kl_loss2': kl_loss2,
            'recon_loss': recon_loss,
            'sparsity_loss': sparsity_loss,
            'gate_mean': gate_scores.mean(),
            'importance_mean': importance_scores.mean()
        }
        
        return z, total_loss, metrics


class AdaptiveInformationBottleneck(nn.Module):
    def __init__(
        self,
        spectrum_dim: int,
        skeleton_dim: int,
        bottleneck_dim: int = 128,
        num_scales: int = 3
    ):
        super().__init__()
        self.num_scales = num_scales
        
        self.multi_scale_cibs = nn.ModuleList([
            ConditionalInformationBottleneckCore(
                spectrum_dim=spectrum_dim,
                skeleton_dim=skeleton_dim,
                bottleneck_dim=bottleneck_dim // num_scales,
                compressed_spectrum_dim=64 // num_scales,
                beta1=1e-3 * (i + 1),
                beta2=1e-4 * (i + 1),
                gumbel_temperature=1.0 / (i + 1)
            ) for i in range(num_scales)
        ])
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim)
        )
        
    def forward(
        self, 
        spectrum: torch.Tensor, 
        skeleton_features: torch.Tensor, 
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        
        all_z = []
        total_loss = 0
        combined_metrics = {}
        
        for i, cib in enumerate(self.multi_scale_cibs):
            z_i, loss_i, metrics_i = cib(spectrum, skeleton_features, training)
            all_z.append(z_i)
            total_loss += loss_i
            
            for key, value in metrics_i.items():
                if key not in combined_metrics:
                    combined_metrics[key] = []
                combined_metrics[key].append(value)
        
        fused_z = torch.cat(all_z, dim=-1)
        final_z = self.fusion_layer(fused_z)
        
        avg_metrics = {key: torch.stack(values).mean() for key, values in combined_metrics.items()}
        
        return final_z, total_loss / self.num_scales, avg_metrics
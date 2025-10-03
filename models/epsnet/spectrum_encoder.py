# coding: utf-8
# -----------------------------------------------------------------------------------------
# New Spectrum Encoder for Conditional GeoDiff Model (Corrected)
# Author: Gemini
# Date: 2025-09-08
#
# Description:
# This file contains the corrected SpectrumEncoder.
#
# Key Correction:
# - In the __init__ method of SpectrumEncoder, the way configuration parameters are
#   accessed has been fixed. It now correctly accesses parameters directly from the
#   `config` object (e.g., `config.spec_cnn_out_dim`) instead of the incorrect
#   `config.model.spec_cnn_out_dim`. This resolves the AttributeError.
# -----------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class _ResidualBlock(nn.Module):
    # ... (This class is correct and remains unchanged)
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 11):
        super().__init__()
        padding = kernel_size // 2
        self.main_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.main_path(x) + self.shortcut(x))

class _LocalEncoderCNN(nn.Module):
    # ... (This class is correct and remains unchanged)
    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.resnet = nn.Sequential(
            _ResidualBlock(1, 32),
            nn.MaxPool1d(2),
            _ResidualBlock(32, 64),
            nn.MaxPool1d(2),
            _ResidualBlock(64, out_channels)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

class _GlobalEncoderTransformer(nn.Module):
    # ... (This class is correct and remains unchanged)
    def __init__(self, seq_len: int, patch_size: int, embed_dim: int, num_heads: int, num_layers: int = 3):
        super().__init__()
        assert seq_len % patch_size == 0, "Sequence length must be divisible by patch size."
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.patch_embedding = nn.Linear(patch_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True,
            dim_feedforward=embed_dim * 4, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.squeeze(1).reshape(-1, self.num_patches, self.patch_size)
        embedded_patches = self.patch_embedding(patches)
        embedded_patches_with_pos = embedded_patches + self.pos_embedding
        return self.transformer_encoder(embedded_patches_with_pos)


class _GatedFusion(nn.Module):
    # ... (This class is correct and remains unchanged)
    def __init__(self, size: int):
        super().__init__()
        self.gate_layer = nn.Linear(size * 2, size)
        self.transform_layer = nn.Linear(size, size)
        self.norm = nn.LayerNorm(size)
    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        target_length = local_feat.size(1)
        if global_feat.size(1) != target_length:
            global_feat_t = global_feat.transpose(1, 2)
            global_feat_aligned_t = F.interpolate(global_feat_t, size=target_length, mode='linear', align_corners=False)
            global_feat_aligned = global_feat_aligned_t.transpose(1, 2)
        else:
            global_feat_aligned = global_feat
        gate_input = torch.cat([local_feat, global_feat_aligned], dim=-1)
        gate = torch.sigmoid(self.gate_layer(gate_input))
        fused = (1 - gate) * local_feat + gate * self.transform_layer(global_feat_aligned)
        return self.norm(fused)

class SpectrumEncoder(nn.Module):
    """
    The main encoder module that orchestrates the entire spectrum encoding process.
    """
    def __init__(self, config):
        super().__init__()
        
        # ------------------------------------------------------------------
        # --- BUG FIX: Access parameters directly from `config` object ---
        # The `config` object passed here is already `config.model` from the YAML file.
        cnn_out_dim = config.spec_cnn_out_dim
        transformer_embed_dim = config.spec_transformer_embed_dim
        num_heads = config.spec_num_heads
        num_layers = config.spec_num_layers
        final_out_dim = config.hidden_dim # Output dim must match GeoDiff's hidden_dim
        # ------------------------------------------------------------------

        # Encoders for UV spectrum (length 600)
        self.local_encoder_uv = _LocalEncoderCNN(out_channels=cnn_out_dim)
        self.global_encoder_uv = _GlobalEncoderTransformer(
            seq_len=600, patch_size=10, embed_dim=transformer_embed_dim, 
            num_heads=num_heads, num_layers=num_layers
        )
        
        # Encoders for auxiliary spectra (IR + Raman, combined length 1200)
        self.local_encoder_aux = _LocalEncoderCNN(out_channels=cnn_out_dim)
        self.global_encoder_aux = _GlobalEncoderTransformer(
            seq_len=1200, patch_size=20, embed_dim=transformer_embed_dim, 
            num_heads=num_heads, num_layers=num_layers
        )
        
        self.transformer_projector = nn.Linear(transformer_embed_dim, cnn_out_dim)
        self.fusion = _GatedFusion(size=cnn_out_dim)
        
        final_seq_length = 150 + 300 
        self.output_head = nn.Sequential(
            nn.Linear(final_seq_length * cnn_out_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, final_out_dim)
        )

    def forward(self, batch) -> torch.Tensor:
        batch_size = batch.num_graphs
        uv_spec = batch.uv_spectrum.reshape(batch_size, -1)
        ir_spec = batch.ir_spectrum.reshape(batch_size, -1)
        raman_spec = batch.raman_spectrum.reshape(batch_size, -1)
        x_uv = uv_spec.unsqueeze(1)
        aux_spec_cat = torch.cat([ir_spec, raman_spec], dim=1)
        x_aux = aux_spec_cat.unsqueeze(1) # Final shape: (B, 1, 1200)
        local_feat_uv = self.local_encoder_uv(x_uv).transpose(1, 2)
        local_feat_aux = self.local_encoder_aux(x_aux).transpose(1, 2)
        global_feat_uv = self.global_encoder_uv(x_uv)
        global_feat_aux = self.global_encoder_aux(x_aux)
        local_feat_seq = torch.cat([local_feat_uv, local_feat_aux], dim=1)
        global_feat_seq = torch.cat([global_feat_uv, global_feat_aux], dim=1)
        global_feat_projected = self.transformer_projector(global_feat_seq)
        fused_seq = self.fusion(local_feat_seq, global_feat_projected)
        fused_flat = fused_seq.reshape(fused_seq.size(0), -1)
        final_embedding = self.output_head(fused_flat)
        return final_embedding
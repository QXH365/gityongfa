# spectrum_encoder.py (Corrected Version)

import torch
import torch.nn as nn
import torch.nn.functional as F

class _ResidualBlock(nn.Module):
    # ... (This class is correct and remains unchanged)
    def __init__(self, in_channels, out_channels, kernel_size=11):
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
    def forward(self, x):
        return F.relu(self.main_path(x) + self.shortcut(x))

class _LocalEncoderCNN(nn.Module):
    # ... (This class is correct and remains unchanged)
    def __init__(self, out_channels=64):
        super().__init__()
        self.resnet = nn.Sequential(
            _ResidualBlock(1, 32),
            nn.MaxPool1d(2),
            _ResidualBlock(32, 64),
            nn.MaxPool1d(2),
            _ResidualBlock(64, out_channels)
        )
    def forward(self, x):
        return self.resnet(x)
    
class _GlobalEncoderTransformer(nn.Module):
    """
    Uses Transformer to extract global features.
    (UPDATED with a more robust patching method)
    """
    def __init__(self, seq_len, patch_size, embed_dim, num_heads, num_layers=3):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        
        self.patch_embedding = nn.Linear(patch_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True, 
            dim_feedforward=embed_dim*4
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # Input x shape: [B, 1, Length]
        
        # --- FIX: Replace `unfold` with a more standard `reshape` for patching ---
        # `x.squeeze(1)` results in shape [B, Length]
        # `.reshape()` groups the sequence into `num_patches` chunks of `patch_size`.
        patches = x.squeeze(1).reshape(-1, self.num_patches, self.patch_size)
        # The resulting `patches` shape is correctly [B, num_patches, patch_size]
        # --- End of FIX ---

        # Patch embedding and positional encoding addition
        # Shapes are now compatible: [B, num_patches, embed_dim] + [1, num_patches, embed_dim]
        embedded_patches = self.patch_embedding(patches) + self.pos_embedding
        
        # Transformer encoding
        return self.transformer_encoder(embedded_patches)

# -----------------------------------------------------------------------------
class SpectrumEncoder(nn.Module):
    # ... (This class is correct and remains unchanged)
    def __init__(self, cnn_out_dim=64, transformer_embed_dim=64, num_heads=4, num_layers=3, final_out_dim=256):
        super().__init__()
        self.local_encoder_uv = _LocalEncoderCNN(out_channels=cnn_out_dim)
        self.global_encoder_uv = _GlobalEncoderTransformer(seq_len=600, patch_size=10, embed_dim=transformer_embed_dim, num_heads=num_heads, num_layers=num_layers)
        self.local_encoder_aux = _LocalEncoderCNN(out_channels=cnn_out_dim)
        self.global_encoder_aux = _GlobalEncoderTransformer(seq_len=1200, patch_size=20, embed_dim=transformer_embed_dim, num_heads=num_heads, num_layers=num_layers)
        self.transformer_projector = nn.Linear(transformer_embed_dim, cnn_out_dim)
        self.fusion = _GatedFusion(size=cnn_out_dim)
        self.output_head = nn.Sequential(
            nn.Linear(450 * cnn_out_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, final_out_dim)
        )

    def forward(self, uv_spec, ir_spec, raman_spec):
        if uv_spec.dim() == 1: uv_spec = uv_spec.unsqueeze(0)
        if ir_spec.dim() == 1: ir_spec = ir_spec.unsqueeze(0)
        if raman_spec.dim() == 1: raman_spec = raman_spec.unsqueeze(0)
        x_main = uv_spec.unsqueeze(1)
        x_aux = torch.cat([ir_spec, raman_spec], dim=1).unsqueeze(1)
        local_feat_main = self.local_encoder_uv(x_main).transpose(1, 2)
        local_feat_aux = self.local_encoder_aux(x_aux).transpose(1, 2)
        global_feat_main = self.global_encoder_uv(x_main)
        global_feat_aux = self.global_encoder_aux(x_aux)
        local_feat_seq = torch.cat([local_feat_main, local_feat_aux], dim=1)
        global_feat_seq = torch.cat([global_feat_main, global_feat_aux], dim=1)
        global_feat_projected = self.transformer_projector(global_feat_seq)
        fused_seq = self.fusion(local_feat_seq, global_feat_projected)
        fused_flat = fused_seq.reshape(fused_seq.size(0), -1)
        final_embedding = self.output_head(fused_flat)
        return final_embedding
class _GatedFusion(nn.Module):
    # ... (This class is correct and remains unchanged)
    def __init__(self, size):
        super().__init__()
        self.gate_layer = nn.Linear(size * 2, size)
        self.transform_layer = nn.Linear(size, size)
        self.norm = nn.LayerNorm(size)
    def forward(self, local_feat, global_feat):
        target_length = local_feat.size(1)
        if global_feat.size(1) != target_length:
            global_feat_aligned = F.interpolate(global_feat.transpose(1, 2), size=target_length, mode='linear', align_corners=False).transpose(1, 2)
        else:
            global_feat_aligned = global_feat
        gate = torch.sigmoid(self.gate_layer(torch.cat([local_feat, global_feat_aligned], dim=-1)))
        fused = (1 - gate) * local_feat + gate * self.transform_layer(global_feat_aligned)
        return self.norm(fused)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List
import numpy as np


class SpectralGraphTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 6,
        spectral_dim: int = 128
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.spectral_encoder = nn.Sequential(
            nn.Linear(spectral_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ) for _ in range(num_layers)
        ])
        
        self.spectral_cross_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
    def forward(
        self,
        node_features: torch.Tensor,
        spectral_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        x = self.input_projection(node_features)
        spectral_h = self.spectral_encoder(spectral_features).unsqueeze(1)
        
        for i in range(self.num_layers):
            residual = x
            
            x = self.transformer_layers[i](x, src_key_padding_mask=attention_mask)
            
            cross_attended, _ = self.spectral_cross_attention[i](
                x, spectral_h, spectral_h
            )
            
            x = self.layer_norms[i](residual + cross_attended)
        
        return x


class GraphSpectralFusionStrategy(nn.Module):
    def __init__(
        self,
        graph_dim: int,
        spectral_dim: int,
        output_dim: int,
        num_fusion_layers: int = 4
    ):
        super().__init__()
        
        self.graph_projection = nn.Sequential(
            nn.Linear(graph_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.spectral_projection = nn.Sequential(
            nn.Linear(spectral_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim * 2, output_dim * 4),
                nn.LayerNorm(output_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim * 4, output_dim),
                nn.LayerNorm(output_dim)
            ) for _ in range(num_fusion_layers)
        ])
        
        self.attention_fusion = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.gating_mechanism = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        graph_features: torch.Tensor,
        spectral_features: torch.Tensor
    ) -> torch.Tensor:
        
        graph_h = self.graph_projection(graph_features)
        spectral_h = self.spectral_projection(spectral_features)
        
        if spectral_h.dim() == 2:
            spectral_h = spectral_h.unsqueeze(1).expand(-1, graph_h.size(1), -1)
        
        combined = torch.cat([graph_h, spectral_h], dim=-1)
        
        for fusion_layer in self.fusion_layers:
            residual = combined[:, :, :graph_h.size(-1)] + combined[:, :, graph_h.size(-1):]
            fused = fusion_layer(combined)
            combined = torch.cat([fused, residual], dim=-1)
        
        final_graph = combined[:, :, :graph_h.size(-1)]
        final_spectral = combined[:, :, graph_h.size(-1):]
        
        attended_graph, _ = self.attention_fusion(
            final_graph, final_spectral, final_spectral
        )
        
        gate = self.gating_mechanism(torch.cat([final_graph, final_spectral], dim=-1))
        
        output = gate * attended_graph + (1 - gate) * final_graph
        
        return output


class DeepMultiLayerArchitecture(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()
        self.use_residual = use_residual
        
        dims = [input_dim] + hidden_dims
        
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.layer_norms.append(nn.LayerNorm(dims[i + 1]))
            self.dropouts.append(nn.Dropout(dropout))
        
        self.output_layer = nn.Linear(dims[-1], output_dim)
        
        self.residual_projections = nn.ModuleList()
        if use_residual:
            for i in range(len(dims) - 1):
                if dims[i] != dims[i + 1]:
                    self.residual_projections.append(nn.Linear(dims[i], dims[i + 1]))
                else:
                    self.residual_projections.append(nn.Identity())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (layer, norm, dropout) in enumerate(zip(self.layers, self.layer_norms, self.dropouts)):
            residual = x
            
            x = layer(x)
            x = norm(x)
            x = F.gelu(x)
            x = dropout(x)
            
            if self.use_residual:
                projected_residual = self.residual_projections[i](residual)
                x = x + projected_residual
        
        return self.output_layer(x)


class MultiscaleSidechainProcessor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_scales: int = 4,
        hidden_dim: int = 512
    ):
        super().__init__()
        self.num_scales = num_scales
        
        self.scale_processors = nn.ModuleList([
            DeepMultiLayerArchitecture(
                input_dim=input_dim,
                hidden_dims=[hidden_dim // (2 ** i) for _ in range(3 + i)],
                output_dim=output_dim,
                dropout=0.1 + 0.05 * i
            ) for i in range(num_scales)
        ])
        
        self.scale_attention = nn.Sequential(
            nn.Linear(output_dim * num_scales, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_scales),
            nn.Softmax(dim=-1)
        )
        
        self.final_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_outputs = []
        
        for processor in self.scale_processors:
            output = processor(x)
            scale_outputs.append(output)
        
        concatenated = torch.cat(scale_outputs, dim=-1)
        attention_weights = self.scale_attention(concatenated)
        
        weighted_sum = sum(
            weight.unsqueeze(-1) * output 
            for weight, output in zip(attention_weights.unbind(dim=-1), scale_outputs)
        )
        
        final_output = self.final_projection(weighted_sum)
        
        return final_output


class AdvancedSideChainPredictor(nn.Module):
    def __init__(
        self,
        node_vocab_size: int,
        edge_vocab_size: int,
        spectrum_dim: int = 600,
        hidden_dim: int = 512,
        output_vocab_size: int = None,
        num_transformer_layers: int = 8,
        num_fusion_layers: int = 6,
        num_prediction_layers: int = 10
    ):
        super().__init__()
        
        if output_vocab_size is None:
            output_vocab_size = node_vocab_size
            
        self.output_vocab_size = output_vocab_size
        
        self.node_embedding = nn.Embedding(node_vocab_size, hidden_dim)
        self.edge_embedding = nn.Embedding(edge_vocab_size, hidden_dim // 2)
        
        self.spectral_processor = nn.Sequential(
            nn.Linear(spectrum_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.graph_spectral_transformer = SpectralGraphTransformer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=8,
            num_layers=num_transformer_layers,
            spectral_dim=hidden_dim
        )
        
        self.graph_spectral_fusion = GraphSpectralFusionStrategy(
            graph_dim=hidden_dim,
            spectral_dim=hidden_dim,
            output_dim=hidden_dim,
            num_fusion_layers=num_fusion_layers
        )
        
        self.multiscale_processor = MultiscaleSidechainProcessor(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            num_scales=4,
            hidden_dim=hidden_dim
        )
        
        self.deep_predictor = DeepMultiLayerArchitecture(
            input_dim=hidden_dim,
            hidden_dims=[hidden_dim * 2, hidden_dim * 4, hidden_dim * 2, hidden_dim],
            output_dim=hidden_dim,
            dropout=0.15
        )
        
        self.context_aware_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_vocab_size)
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        spectrum: torch.Tensor,
        attachment_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size, num_nodes = node_features.shape
        
        node_h = self.node_embedding(node_features)
        
        spectral_h = self.spectral_processor(spectrum)
        
        graph_output = self.graph_spectral_transformer(
            node_features=node_h,
            spectral_features=spectral_h,
            attention_mask=attachment_mask
        )
        
        fused_features = self.graph_spectral_fusion(
            graph_features=graph_output,
            spectral_features=spectral_h
        )
        
        multiscale_output = self.multiscale_processor(fused_features)
        
        deep_output = self.deep_predictor(multiscale_output)
        
        context_attended, _ = self.context_aware_attention(
            deep_output, deep_output, deep_output,
            key_padding_mask=attachment_mask
        )
        
        final_features = context_attended + deep_output
        
        predictions = self.output_head(final_features)
        uncertainty = self.uncertainty_head(final_features)
        
        return predictions, uncertainty


class HierarchicalSideChainPredictor(nn.Module):
    def __init__(
        self,
        node_vocab_size: int,
        edge_vocab_size: int,
        spectrum_dim: int = 600,
        hidden_dim: int = 512,
        output_vocab_size: int = None,
        num_hierarchy_levels: int = 3
    ):
        super().__init__()
        
        self.num_levels = num_hierarchy_levels
        
        self.level_predictors = nn.ModuleList([
            AdvancedSideChainPredictor(
                node_vocab_size=node_vocab_size,
                edge_vocab_size=edge_vocab_size,
                spectrum_dim=spectrum_dim,
                hidden_dim=hidden_dim // (i + 1),
                output_vocab_size=output_vocab_size,
                num_transformer_layers=4 + i * 2,
                num_fusion_layers=3 + i * 2,
                num_prediction_layers=6 + i * 2
            ) for i in range(num_hierarchy_levels)
        ])
        
        self.hierarchical_fusion = nn.Sequential(
            nn.Linear(output_vocab_size * num_hierarchy_levels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_vocab_size),
            nn.LayerNorm(output_vocab_size)
        )
        
        self.uncertainty_fusion = nn.Sequential(
            nn.Linear(num_hierarchy_levels, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.level_weights = nn.Parameter(torch.ones(num_hierarchy_levels) / num_hierarchy_levels)
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        spectrum: torch.Tensor,
        attachment_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        level_predictions = []
        level_uncertainties = []
        
        for predictor in self.level_predictors:
            pred, unc = predictor(
                node_features, edge_features, edge_index, spectrum, attachment_mask
            )
            level_predictions.append(pred)
            level_uncertainties.append(unc)
        
        concatenated_predictions = torch.cat(level_predictions, dim=-1)
        fused_predictions = self.hierarchical_fusion(concatenated_predictions)
        
        stacked_uncertainties = torch.stack(level_uncertainties, dim=-1)
        fused_uncertainty = self.uncertainty_fusion(stacked_uncertainties.squeeze(-2))
        
        weights = F.softmax(self.level_weights, dim=0)
        weighted_predictions = sum(
            w * pred for w, pred in zip(weights, level_predictions)
        )
        
        final_predictions = 0.7 * fused_predictions + 0.3 * weighted_predictions
        
        return final_predictions, fused_uncertainty


class AdaptiveSideChainClassifier(nn.Module):
    def __init__(
        self,
        node_vocab_size: int,
        edge_vocab_size: int,
        spectrum_dim: int = 600,
        hidden_dim: int = 512,
        output_vocab_size: int = None,
        temperature: float = 1.0,
        use_hierarchical: bool = True
    ):
        super().__init__()
        
        self.temperature = temperature
        
        if use_hierarchical:
            self.predictor = HierarchicalSideChainPredictor(
                node_vocab_size=node_vocab_size,
                edge_vocab_size=edge_vocab_size,
                spectrum_dim=spectrum_dim,
                hidden_dim=hidden_dim,
                output_vocab_size=output_vocab_size
            )
        else:
            self.predictor = AdvancedSideChainPredictor(
                node_vocab_size=node_vocab_size,
                edge_vocab_size=edge_vocab_size,
                spectrum_dim=spectrum_dim,
                hidden_dim=hidden_dim,
                output_vocab_size=output_vocab_size
            )
        
        self.calibration_layer = nn.Sequential(
            nn.Linear(output_vocab_size or node_vocab_size, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_vocab_size or node_vocab_size)
        )
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        spectrum: torch.Tensor,
        attachment_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        raw_predictions, uncertainty = self.predictor(
            node_features, edge_features, edge_index, spectrum, attachment_mask
        )
        
        calibrated_predictions = self.calibration_layer(raw_predictions)
        
        final_predictions = calibrated_predictions / self.temperature
        
        return final_predictions, uncertainty
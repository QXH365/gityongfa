import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List
import numpy as np


class MultiScaleGraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        self.convs = nn.ModuleList([
            nn.Linear(in_features, out_features // num_scales)
            for _ in range(num_scales)
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(out_features // num_scales)
            for _ in range(num_scales)
        ])
        self.fusion = nn.Linear(out_features, out_features)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, num_nodes, in_dim = x.shape
        
        outputs = []
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            hop_distance = i + 1
            
            x_transformed = conv(x)
            
            adj_power = torch.eye(num_nodes, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
            edge_adj = torch.zeros(batch_size, num_nodes, num_nodes, device=x.device)
            
            for b in range(batch_size):
                if edge_index.size(1) > 0:
                    edge_adj[b, edge_index[0], edge_index[1]] = 1.0 if edge_weight is None else edge_weight
            
            current_adj = edge_adj.clone()
            for _ in range(hop_distance - 1):
                current_adj = torch.bmm(current_adj, edge_adj)
                current_adj = (current_adj > 0).float()
            
            aggregated = torch.bmm(current_adj, x_transformed)
            
            aggregated_flat = aggregated.view(-1, x_transformed.size(-1))
            normalized = bn(aggregated_flat)
            normalized = normalized.view(batch_size, num_nodes, -1)
            
            outputs.append(normalized)
        
        concatenated = torch.cat(outputs, dim=-1)
        fused = self.fusion(concatenated)
        return F.gelu(fused)


class HierarchicalAttentionMechanism(nn.Module):
    def __init__(self, feature_dim: int, num_levels: int = 4, num_heads: int = 8):
        super().__init__()
        self.num_levels = num_levels
        self.feature_dim = feature_dim
        
        self.level_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_levels)
        ])
        
        self.level_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(num_levels)
        ])
        
        self.hierarchical_fusion = nn.Sequential(
            nn.Linear(feature_dim * num_levels, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        self.position_encoding = self._create_position_encoding(feature_dim)
        
    def _create_position_encoding(self, d_model: int, max_len: int = 100):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe, requires_grad=False)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        pos_encoding = self.position_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_encoding.to(x.device)
        
        level_outputs = []
        
        for level, (attention, projection) in enumerate(zip(self.level_attentions, self.level_projections)):
            
            query = x
            key = value = x
            
            if level > 0:
                stride = 2 ** level
                query = query[:, ::stride, :]
                key = value = key[:, ::stride, :]
                if mask is not None:
                    level_mask = mask[:, ::stride]
                else:
                    level_mask = None
            else:
                level_mask = mask
            
            attn_output, _ = attention(query, key, value, key_padding_mask=level_mask)
            
            if level > 0:
                expanded_output = torch.zeros_like(x)
                expanded_output[:, ::stride, :] = attn_output
                
                for i in range(1, stride):
                    if (i + seq_len) % stride == 0:
                        continue
                    prev_idx = ((torch.arange(seq_len) - i) // stride) * stride + i
                    prev_idx = torch.clamp(prev_idx, 0, seq_len - 1)
                    expanded_output[:, i::stride, :] = expanded_output[:, prev_idx[i::stride], :]
                    
                attn_output = expanded_output
            
            projected = projection(attn_output)
            level_outputs.append(projected)
        
        concatenated = torch.cat(level_outputs, dim=-1)
        fused = self.hierarchical_fusion(concatenated)
        
        return fused + x


class AdvancedGraphNeuralNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_layers = num_layers
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.graph_conv_layers = nn.ModuleList([
            MultiScaleGraphConvolution(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.attention_layers = nn.ModuleList([
            HierarchicalAttentionMechanism(hidden_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        
        self.residual_connections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_weight: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        x = self.input_projection(x)
        
        for i in range(self.num_layers):
            residual = x
            
            x = self.graph_conv_layers[i](x, edge_index, edge_weight)
            x = self.layer_norms[i](x + residual)
            
            residual = x
            x = self.attention_layers[i](x, mask)
            x = x + residual
            
            residual = x
            x = self.residual_connections[i](x)
            x = self.layer_norms[i](x + residual)
        
        return self.output_projection(x)


class ScaffoldStructureExtractor(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, output_dim: int):
        super().__init__()
        
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.structure_analyzer = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        
        self.topology_encoder = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(
        self, 
        node_features: torch.Tensor, 
        edge_features: torch.Tensor,
        scaffold_mask: torch.Tensor
    ) -> torch.Tensor:
        
        batch_size, num_nodes, _ = node_features.shape
        
        node_h = self.node_encoder(node_features)
        
        edge_h = self.edge_encoder(edge_features)
        edge_h = edge_h.mean(dim=2)
        
        combined = torch.cat([node_h, edge_h], dim=-1)
        structure_features = self.structure_analyzer(combined)
        
        scaffold_features = structure_features * scaffold_mask.unsqueeze(-1)
        
        lstm_out, (hidden, _) = self.topology_encoder(scaffold_features)
        
        final_hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)
        
        output = self.output_projection(final_hidden)
        
        return output


class EnhancedScaffoldEncoder(nn.Module):
    def __init__(
        self,
        node_vocab_size: int,
        edge_vocab_size: int,
        hidden_dim: int = 512,
        output_dim: int = 512,
        num_gnn_layers: int = 6,
        num_attention_heads: int = 8
    ):
        super().__init__()
        
        self.node_embedding = nn.Embedding(node_vocab_size, hidden_dim // 2)
        self.edge_embedding = nn.Embedding(edge_vocab_size, hidden_dim // 4)
        
        self.scaffold_extractor = ScaffoldStructureExtractor(
            node_dim=hidden_dim // 2,
            edge_dim=hidden_dim // 4,
            output_dim=hidden_dim
        )
        
        self.advanced_gnn = AdvancedGraphNeuralNetwork(
            input_dim=hidden_dim // 2,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_gnn_layers,
            num_heads=num_attention_heads
        )
        
        self.global_attention_pool = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self.structure_aware_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.scaffold_importance_predictor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        scaffold_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size, num_nodes = node_features.shape
        
        node_h = self.node_embedding(node_features)
        edge_h = self.edge_embedding(edge_features)
        
        scaffold_structure = self.scaffold_extractor(node_h, edge_h, scaffold_mask)
        
        gnn_output = self.advanced_gnn(node_h, edge_index, mask=scaffold_mask)
        
        pool_query = self.pool_query.expand(batch_size, -1, -1)
        pooled_features, attention_weights = self.global_attention_pool(
            pool_query, gnn_output, gnn_output,
            key_padding_mask=(scaffold_mask == 0).all(dim=-1)
        )
        pooled_features = pooled_features.squeeze(1)
        
        fused_features = torch.cat([scaffold_structure, pooled_features], dim=-1)
        final_representation = self.structure_aware_fusion(fused_features)
        
        importance_score = self.scaffold_importance_predictor(final_representation)
        
        return final_representation, importance_score


class MultiResolutionScaffoldEncoder(nn.Module):
    def __init__(
        self,
        node_vocab_size: int,
        edge_vocab_size: int,
        hidden_dim: int = 512,
        output_dim: int = 512,
        num_resolutions: int = 3
    ):
        super().__init__()
        self.num_resolutions = num_resolutions
        
        self.resolution_encoders = nn.ModuleList([
            EnhancedScaffoldEncoder(
                node_vocab_size=node_vocab_size,
                edge_vocab_size=edge_vocab_size,
                hidden_dim=hidden_dim // (i + 1),
                output_dim=output_dim // num_resolutions,
                num_gnn_layers=4 + i * 2,
                num_attention_heads=max(4, 8 // (i + 1))
            ) for i in range(num_resolutions)
        ])
        
        self.resolution_fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.adaptive_weighting = nn.Sequential(
            nn.Linear(output_dim, num_resolutions),
            nn.Softmax(dim=-1)
        )
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        scaffold_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        resolution_outputs = []
        resolution_importance = []
        
        for encoder in self.resolution_encoders:
            output, importance = encoder(node_features, edge_features, edge_index, scaffold_mask)
            resolution_outputs.append(output)
            resolution_importance.append(importance)
        
        concatenated_output = torch.cat(resolution_outputs, dim=-1)
        fused_output = self.resolution_fusion(concatenated_output)
        
        weights = self.adaptive_weighting(fused_output).unsqueeze(-1)
        
        importance_stack = torch.stack(resolution_importance, dim=-1)
        weighted_importance = torch.sum(importance_stack * weights, dim=-1)
        
        return fused_output, weighted_importance
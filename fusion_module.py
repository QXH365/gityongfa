import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any
import math

from cib_core import ConditionalInformationBottleneckCore, AdaptiveInformationBottleneck
from scaffold_encoder import EnhancedScaffoldEncoder, MultiResolutionScaffoldEncoder
from sidechain_predictor import AdaptiveSideChainClassifier


class SeamlessDiffusionIntegrator(nn.Module):
    def __init__(
        self,
        original_model_dim: int,
        enhanced_feature_dim: int,
        integration_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.feature_aligner = nn.Sequential(
            nn.Linear(enhanced_feature_dim, original_model_dim),
            nn.LayerNorm(original_model_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.integration_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=original_model_dim,
                nhead=8,
                dim_feedforward=original_model_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ),
            num_layers=integration_layers
        )
        
        self.adaptive_gate = nn.Sequential(
            nn.Linear(original_model_dim * 2, original_model_dim),
            nn.Sigmoid()
        )
        
        self.residual_projection = nn.Linear(original_model_dim, original_model_dim)
        
    def forward(
        self,
        original_features: torch.Tensor,
        enhanced_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        aligned_features = self.feature_aligner(enhanced_features)
        
        combined_input = original_features + aligned_features
        
        integrated = self.integration_transformer(
            combined_input,
            src_key_padding_mask=attention_mask
        )
        
        gate = self.adaptive_gate(torch.cat([original_features, integrated], dim=-1))
        
        output = gate * integrated + (1 - gate) * original_features
        output = output + self.residual_projection(original_features)
        
        return output


class ModularDesignFramework(nn.Module):
    def __init__(
        self,
        node_vocab_size: int,
        edge_vocab_size: int,
        spectrum_dim: int = 600,
        hidden_dim: int = 512,
        use_cib: bool = True,
        use_enhanced_scaffold: bool = True,
        use_advanced_sidechain: bool = True
    ):
        super().__init__()
        
        self.use_cib = use_cib
        self.use_enhanced_scaffold = use_enhanced_scaffold
        self.use_advanced_sidechain = use_advanced_sidechain
        
        if use_cib:
            self.cib_module = AdaptiveInformationBottleneck(
                spectrum_dim=spectrum_dim,
                skeleton_dim=hidden_dim,
                bottleneck_dim=128,
                num_scales=3
            )
            
        if use_enhanced_scaffold:
            self.scaffold_encoder = MultiResolutionScaffoldEncoder(
                node_vocab_size=node_vocab_size,
                edge_vocab_size=edge_vocab_size,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_resolutions=3
            )
            
        if use_advanced_sidechain:
            self.sidechain_predictor = AdaptiveSideChainClassifier(
                node_vocab_size=node_vocab_size,
                edge_vocab_size=edge_vocab_size,
                spectrum_dim=spectrum_dim,
                hidden_dim=hidden_dim,
                output_vocab_size=node_vocab_size,
                use_hierarchical=True
            )
        
        self.diffusion_integrator = SeamlessDiffusionIntegrator(
            original_model_dim=hidden_dim,
            enhanced_feature_dim=hidden_dim,
            integration_layers=4
        )
        
        self.feature_harmonizer = nn.Sequential(
            nn.Linear(hidden_dim * 3 if all([use_cib, use_enhanced_scaffold, use_advanced_sidechain]) else hidden_dim,
                     hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(
        self,
        original_features: torch.Tensor,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        spectrum: torch.Tensor,
        scaffold_mask: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        
        enhanced_features = []
        module_outputs = {}
        
        if self.use_cib:
            skeleton_features = original_features.mean(dim=1)
            cib_output, cib_loss, cib_metrics = self.cib_module(
                spectrum, skeleton_features, training=self.training
            )
            enhanced_features.append(cib_output.unsqueeze(1).expand(-1, original_features.size(1), -1))
            module_outputs['cib'] = {'loss': cib_loss, 'metrics': cib_metrics, 'features': cib_output}
            
        if self.use_enhanced_scaffold:
            scaffold_features, scaffold_importance = self.scaffold_encoder(
                node_features, edge_features, edge_index, scaffold_mask
            )
            enhanced_features.append(scaffold_features.unsqueeze(1).expand(-1, original_features.size(1), -1))
            module_outputs['scaffold'] = {
                'features': scaffold_features, 
                'importance': scaffold_importance
            }
            
        if self.use_advanced_sidechain:
            sidechain_predictions, sidechain_uncertainty = self.sidechain_predictor(
                node_features, edge_features, edge_index, spectrum
            )
            enhanced_features.append(sidechain_predictions)
            module_outputs['sidechain'] = {
                'predictions': sidechain_predictions,
                'uncertainty': sidechain_uncertainty
            }
        
        if enhanced_features:
            if len(enhanced_features) > 1:
                harmonized_features = self.feature_harmonizer(torch.cat(enhanced_features, dim=-1))
            else:
                harmonized_features = enhanced_features[0]
                
            integrated_output = self.diffusion_integrator(
                original_features, harmonized_features
            )
        else:
            integrated_output = original_features
        
        return integrated_output, module_outputs


class ConditionalGenerationPipeline(nn.Module):
    def __init__(
        self,
        base_diffusion_model: nn.Module,
        fusion_framework: ModularDesignFramework,
        conditioning_strength: float = 1.0
    ):
        super().__init__()
        
        self.base_model = base_diffusion_model
        self.fusion_framework = fusion_framework
        self.conditioning_strength = conditioning_strength
        
        self.condition_projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 512)
        )
        
        self.adaptive_conditioning = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        spectrum: torch.Tensor,
        scaffold_mask: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        
        with torch.no_grad():
            base_output = self.base_model(
                node_features=node_features,
                edge_features=edge_features,
                adj_mask=(edge_features > 0).float(),
                spectrum=spectrum,
                timestep=timestep,
                scaffold_mask=scaffold_mask
            )
            
            if isinstance(base_output, tuple):
                base_features = base_output[0]
                base_loss = base_output[1] if len(base_output) > 1 else None
                base_metrics = base_output[2] if len(base_output) > 2 else {}
            else:
                base_features = base_output
                base_loss = None
                base_metrics = {}
        
        node_embeddings = self.base_model.model.node_embedding(node_features)
        
        enhanced_output, module_outputs = self.fusion_framework(
            original_features=node_embeddings,
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index,
            spectrum=spectrum,
            scaffold_mask=scaffold_mask
        )
        
        conditioning_signal = self.condition_projector(enhanced_output.mean(dim=1))
        adaptive_weight = self.adaptive_conditioning(conditioning_signal)
        
        conditioned_features = (
            self.conditioning_strength * adaptive_weight.unsqueeze(1) * enhanced_output +
            (1 - self.conditioning_strength * adaptive_weight.unsqueeze(1)) * node_embeddings
        )
        
        final_output = self.base_model.model.edge_output(conditioned_features)
        
        total_loss = base_loss if base_loss is not None else 0
        if 'cib' in module_outputs and 'loss' in module_outputs['cib']:
            total_loss = total_loss + 0.1 * module_outputs['cib']['loss']
        
        combined_metrics = {**base_metrics}
        for module_name, outputs in module_outputs.items():
            if 'metrics' in outputs:
                for key, value in outputs['metrics'].items():
                    combined_metrics[f"{module_name}_{key}"] = value
        
        return final_output, total_loss, combined_metrics


class ExtensibleArchitecture(nn.Module):
    def __init__(
        self,
        base_config: Dict[str, Any],
        enhancement_config: Dict[str, Any] = None
    ):
        super().__init__()
        
        self.base_config = base_config
        self.enhancement_config = enhancement_config or {}
        
        self.module_registry = nn.ModuleDict()
        
        self._register_base_modules()
        self._register_enhancement_modules()
        
    def _register_base_modules(self):
        pass
        
    def _register_enhancement_modules(self):
        if self.enhancement_config.get('use_cib', True):
            self.module_registry['cib'] = AdaptiveInformationBottleneck(
                spectrum_dim=self.enhancement_config.get('spectrum_dim', 600),
                skeleton_dim=self.enhancement_config.get('hidden_dim', 512),
                bottleneck_dim=self.enhancement_config.get('bottleneck_dim', 128)
            )
            
        if self.enhancement_config.get('use_enhanced_scaffold', True):
            self.module_registry['scaffold'] = MultiResolutionScaffoldEncoder(
                node_vocab_size=self.base_config['node_vocab_size'],
                edge_vocab_size=self.base_config['edge_vocab_size'],
                hidden_dim=self.enhancement_config.get('hidden_dim', 512),
                output_dim=self.enhancement_config.get('hidden_dim', 512)
            )
            
        if self.enhancement_config.get('use_advanced_sidechain', True):
            self.module_registry['sidechain'] = AdaptiveSideChainClassifier(
                node_vocab_size=self.base_config['node_vocab_size'],
                edge_vocab_size=self.base_config['edge_vocab_size'],
                spectrum_dim=self.enhancement_config.get('spectrum_dim', 600),
                hidden_dim=self.enhancement_config.get('hidden_dim', 512)
            )
    
    def add_module_extension(self, name: str, module: nn.Module):
        self.module_registry[name] = module
        
    def remove_module_extension(self, name: str):
        if name in self.module_registry:
            del self.module_registry[name]
            
    def forward(self, **kwargs):
        outputs = {}
        
        for name, module in self.module_registry.items():
            try:
                if name == 'cib':
                    output = module(
                        kwargs['spectrum'],
                        kwargs.get('skeleton_features', kwargs['node_features'].mean(dim=1)),
                        training=self.training
                    )
                elif name == 'scaffold':
                    output = module(
                        kwargs['node_features'],
                        kwargs['edge_features'],
                        kwargs['edge_index'],
                        kwargs['scaffold_mask']
                    )
                elif name == 'sidechain':
                    output = module(
                        kwargs['node_features'],
                        kwargs['edge_features'],
                        kwargs['edge_index'],
                        kwargs['spectrum']
                    )
                else:
                    output = module(**kwargs)
                    
                outputs[name] = output
                
            except Exception as e:
                print(f"Module {name} failed: {e}")
                continue
        
        return outputs


class AdvancedFusionModule(nn.Module):
    def __init__(
        self,
        node_vocab_size: int,
        edge_vocab_size: int,
        spectrum_dim: int = 600,
        hidden_dim: int = 512,
        integration_strategy: str = 'adaptive',
        conditioning_strength: float = 1.0
    ):
        super().__init__()
        
        self.integration_strategy = integration_strategy
        
        self.modular_framework = ModularDesignFramework(
            node_vocab_size=node_vocab_size,
            edge_vocab_size=edge_vocab_size,
            spectrum_dim=spectrum_dim,
            hidden_dim=hidden_dim,
            use_cib=True,
            use_enhanced_scaffold=True,
            use_advanced_sidechain=True
        )
        
        if integration_strategy == 'adaptive':
            self.integration_layer = SeamlessDiffusionIntegrator(
                original_model_dim=hidden_dim,
                enhanced_feature_dim=hidden_dim,
                integration_layers=4
            )
        elif integration_strategy == 'attention':
            self.integration_layer = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
        else:
            self.integration_layer = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
        
        self.output_harmonizer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def integrate_with_base_model(
        self,
        base_model: nn.Module,
        **inputs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        
        base_embeddings = base_model.model.node_embedding(inputs['node_features'])
        
        enhanced_output, module_outputs = self.modular_framework(
            original_features=base_embeddings,
            **inputs
        )
        
        if self.integration_strategy == 'adaptive':
            integrated = self.integration_layer(base_embeddings, enhanced_output)
        elif self.integration_strategy == 'attention':
            integrated, _ = self.integration_layer(
                enhanced_output, base_embeddings, base_embeddings
            )
        else:
            integrated = self.integration_layer(
                torch.cat([base_embeddings, enhanced_output], dim=-1)
            )
        
        harmonized = self.output_harmonizer(integrated)
        
        final_output = base_model.model.edge_output(harmonized)
        
        return final_output, module_outputs
        
    def forward(
        self,
        base_model: nn.Module,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        spectrum: torch.Tensor,
        scaffold_mask: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        
        inputs = {
            'node_features': node_features,
            'edge_features': edge_features,
            'edge_index': edge_index,
            'spectrum': spectrum,
            'scaffold_mask': scaffold_mask,
            **kwargs
        }
        
        return self.integrate_with_base_model(base_model, **inputs)

# Example: Integrating Advanced Framework with Existing Model

import torch
import torch.nn as nn
from model_cib import ImprovedMolecularDiffusionModel
from fusion_module import AdvancedFusionModule
from advanced_trainer import AdvancedMolecularDiffusionTrainer, AdvancedTrainingPipeline

def integrate_advanced_framework():
    # Load existing model
    base_model = ImprovedMolecularDiffusionModel(
        node_vocab_size=100,
        edge_vocab_size=10,
        spectrum_dim=600,
        hidden_dim=512,
        n_layers=6,
        n_heads=8,
        timesteps=1000
    )
    
    # Create advanced fusion module (non-invasive extension)
    fusion_module = AdvancedFusionModule(
        node_vocab_size=100,
        edge_vocab_size=10,
        spectrum_dim=600,
        hidden_dim=512,
        integration_strategy='adaptive'
    )
    
    # Enhanced model with all advanced features
    advanced_model = AdvancedMolecularDiffusionTrainer(
        base_diffusion_model=base_model,
        node_vocab_size=100,
        edge_vocab_size=10,
        spectrum_dim=600,
        hidden_dim=512,
        use_multistage=True
    )
    
    return advanced_model, fusion_module

def run_advanced_training():
    # Configuration for advanced training
    config = {
        'node_vocab_size': 100,
        'edge_vocab_size': 10,
        'spectrum_dim': 600,
        'hidden_dim': 512,
        'lr': 1e-4,
        'weight_decay': 0.01
    }
    
    # Load base model
    base_model = ImprovedMolecularDiffusionModel(**config)
    
    # Create training pipeline
    pipeline = AdvancedTrainingPipeline(
        base_model=base_model,
        config=config,
        save_dir='./advanced_results'
    )
    
    # Run training with advanced features
    # pipeline.run_training(train_loader, val_loader, max_epochs=100)
    
    return pipeline

# Example usage patterns:

# 1. Direct integration with existing model
advanced_model, fusion = integrate_advanced_framework()

# 2. Training with advanced framework
training_pipeline = run_advanced_training()

# 3. Using individual components
from cib_core import AdaptiveInformationBottleneck
from scaffold_encoder import MultiResolutionScaffoldEncoder
from sidechain_predictor import AdaptiveSideChainClassifier

# Initialize individual components
cib = AdaptiveInformationBottleneck(spectrum_dim=600, skeleton_dim=512)
scaffold_encoder = MultiResolutionScaffoldEncoder(node_vocab_size=100, edge_vocab_size=10)
sidechain_predictor = AdaptiveSideChainClassifier(node_vocab_size=100, edge_vocab_size=10)

print("Advanced Spectral-Guided Molecular Diffusion Framework ready!")

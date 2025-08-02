import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_module_imports():
    try:
        from cib_core import ConditionalInformationBottleneckCore, AdaptiveInformationBottleneck
        from scaffold_encoder import EnhancedScaffoldEncoder, MultiResolutionScaffoldEncoder
        from sidechain_predictor import AdaptiveSideChainClassifier
        from fusion_module import AdvancedFusionModule
        from advanced_trainer import AdvancedMolecularDiffusionTrainer, CombinedLossFunction
        print("✓ All module imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_cib_module():
    try:
        from cib_core import ConditionalInformationBottleneckCore
        
        cib = ConditionalInformationBottleneckCore(
            spectrum_dim=600,
            skeleton_dim=512,
            bottleneck_dim=128
        )
        
        spectrum = torch.randn(2, 600)
        skeleton_features = torch.randn(2, 512)
        
        z, loss, metrics = cib(spectrum, skeleton_features, training=True)
        
        assert z.shape == (2, 128), f"Expected z shape (2, 128), got {z.shape}"
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert isinstance(metrics, dict), "Metrics should be a dictionary"
        
        print("✓ CIB module test passed")
        return True
    except Exception as e:
        print(f"✗ CIB module test failed: {e}")
        return False

def test_scaffold_encoder():
    try:
        from scaffold_encoder import EnhancedScaffoldEncoder
        
        encoder = EnhancedScaffoldEncoder(
            node_vocab_size=100,
            edge_vocab_size=10,
            hidden_dim=512,
            output_dim=512
        )
        
        node_features = torch.randint(0, 100, (2, 10))
        edge_features = torch.randint(0, 10, (2, 10, 10))
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        scaffold_mask = torch.ones(2, 10, 10)
        
        output, importance = encoder(node_features, edge_features, edge_index, scaffold_mask)
        
        assert output.shape == (2, 512), f"Expected output shape (2, 512), got {output.shape}"
        assert importance.shape == (2, 1), f"Expected importance shape (2, 1), got {importance.shape}"
        
        print("✓ Scaffold encoder test passed")
        return True
    except Exception as e:
        print(f"✗ Scaffold encoder test failed: {e}")
        return False

def test_sidechain_predictor():
    try:
        from sidechain_predictor import AdaptiveSideChainClassifier
        
        predictor = AdaptiveSideChainClassifier(
            node_vocab_size=100,
            edge_vocab_size=10,
            spectrum_dim=600,
            hidden_dim=512,
            output_vocab_size=100
        )
        
        node_features = torch.randint(0, 100, (2, 10))
        edge_features = torch.randint(0, 10, (2, 10, 10))
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        spectrum = torch.randn(2, 600)
        
        predictions, uncertainty = predictor(node_features, edge_features, edge_index, spectrum)
        
        assert predictions.shape == (2, 10, 100), f"Expected predictions shape (2, 10, 100), got {predictions.shape}"
        assert uncertainty.shape == (2, 10, 1), f"Expected uncertainty shape (2, 10, 1), got {uncertainty.shape}"
        
        print("✓ Sidechain predictor test passed")
        return True
    except Exception as e:
        print(f"✗ Sidechain predictor test failed: {e}")
        return False

def test_fusion_module():
    try:
        from fusion_module import AdvancedFusionModule
        
        fusion = AdvancedFusionModule(
            node_vocab_size=100,
            edge_vocab_size=10,
            spectrum_dim=600,
            hidden_dim=512
        )
        
        # Create a simple mock base model
        class MockBaseModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.node_embedding = nn.Embedding(100, 512)
                self.model.edge_output = nn.Linear(512, 10)
                
        base_model = MockBaseModel()
        
        node_features = torch.randint(0, 100, (2, 10))
        edge_features = torch.randint(0, 10, (2, 10, 10))
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        spectrum = torch.randn(2, 600)
        scaffold_mask = torch.ones(2, 10, 10)
        
        output, module_outputs = fusion(
            base_model, node_features, edge_features, edge_index, spectrum, scaffold_mask
        )
        
        assert output.shape == (2, 10, 10, 10), f"Expected output shape (2, 10, 10, 10), got {output.shape}"
        assert isinstance(module_outputs, dict), "Module outputs should be a dictionary"
        
        print("✓ Fusion module test passed")
        return True
    except Exception as e:
        print(f"✗ Fusion module test failed: {e}")
        return False

def test_combined_loss():
    try:
        from advanced_trainer import CombinedLossFunction
        
        loss_fn = CombinedLossFunction()
        
        predicted_logits = torch.randn(2, 10, 10, 5)
        target_edges = torch.randint(0, 5, (2, 10, 10))
        scaffold_mask = torch.zeros(2, 10, 10)
        module_outputs = {
            'cib': {'loss': torch.tensor(0.1)},
            'scaffold': {'importance': torch.randn(2, 1)},
            'sidechain': {'predictions': torch.randn(2, 10, 5)}
        }
        
        total_loss, loss_components = loss_fn(
            predicted_logits, target_edges, scaffold_mask, module_outputs
        )
        
        assert isinstance(total_loss, torch.Tensor), "Total loss should be a tensor"
        assert isinstance(loss_components, dict), "Loss components should be a dictionary"
        assert len(loss_components) == 6, f"Expected 6 loss components, got {len(loss_components)}"
        
        print("✓ Combined loss function test passed")
        return True
    except Exception as e:
        print(f"✗ Combined loss function test failed: {e}")
        return False

def main():
    print("Running Advanced Spectral-Guided Molecular Diffusion Framework Tests...")
    print("=" * 70)
    
    tests = [
        test_module_imports,
        test_cib_module,
        test_scaffold_encoder,
        test_sidechain_predictor,
        test_fusion_module,
        test_combined_loss
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 70)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The advanced framework is ready for use.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
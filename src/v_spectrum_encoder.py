# v_spectrum_encoder.py

import torch
import pickle
import sys
from easydict import EasyDict
from torch_geometric.loader import DataLoader

# 确保项目根目录在 Python 搜索路径中，以便能找到 models 和 utils 文件夹
sys.path.append('.')

from tt import Spectroformer

def main():
    """
    用于验证 Spectroformer 编码器完整实现的脚本。
    """
    print("--- 开始验证 Spectroformer 编码器 (完整版) ---")

    # 1. 定义一个与 Spectroformer 模块完全匹配的配置文件
    config = EasyDict({
        'total_spec_len': 1200,    # 假设 IR(600) + Raman(600) 的总长度
        'spec_patch_size': 20,     # 每个 patch 的长度
        'spec_embed_dim': 128,     # 嵌入向量的维度
        'spec_num_heads': 4,       # Transformer 注意力头的数量
        'spec_num_layers': 2,      # Transformer 编码器的层数
        'spec_num_concepts': 8     # 期望最终压缩成的“光谱概念”数量
    })
    print(f"使用的模型配置: \n{config}")

    # 2. 加载真实的数据集文件
    data_path = 'qme14s_all/test_data.pkl'
    try:
        with open(data_path, 'rb') as f:
            # 注意：原始的 ConformationDataset 包装可能不需要，直接加载 pkl 里的 list 即可
            test_data_list = pickle.load(f)
        if not isinstance(test_data_list, list):
            raise TypeError("数据集文件应包含一个 Python 列表 (list)。")
        print(f"✅ 成功从 '{data_path}' 加载 {len(test_data_list)} 个样本。")
    except FileNotFoundError:
        print(f"❌ 错误: 未找到测试数据 '{data_path}'。请确保文件路径正确且位于项目根目录下。")
        return
    except Exception as e:
        print(f"❌ 加载数据时发生错误: {e}")
        return

    # 3. 创建 DataLoader 和一个批次的数据
    batch_size = 4
    # 过滤掉可能不包含光谱数据的样本（如果存在）
    test_data_with_spectra = [d for d in test_data_list if hasattr(d, 'ir_spectrum') and hasattr(d, 'raman_spectrum')]
    if len(test_data_with_spectra) < batch_size:
        print(f"❌ 错误: 数据集中包含光谱的样本不足 {batch_size} 个，无法创建批次。")
        return
        
    test_loader = DataLoader(test_data_with_spectra, batch_size=batch_size, shuffle=False)
    batch = next(iter(test_loader))
    print(f"✅ 成功创建批次数据，批次大小: {batch.num_graphs}")

    # 4. 实例化 Spectroformer 模型
    try:
        model = Spectroformer(config)
        model.eval() # 设置为评估模式
        print("✅ 模型实例化成功。")
    except Exception as e:
        print(f"❌ 模型实例化失败: {e}")
        return

    # 5. 执行前向传播并获取输出
    print("\n--- 执行前向传播 ---")
    try:
        with torch.no_grad():
            spectral_concepts, kl_loss = model(batch)
        print("✅ 前向传播成功。")
    except Exception as e:
        print(f"❌ 前向传播时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. 验证输出形状
    print(f"模型输出 'spectral_concepts' 的形状: {spectral_concepts.shape}")
    expected_shape = (batch_size, config.spec_num_concepts, config.spec_embed_dim)
    print(f"期望输出形状: {expected_shape}")

    assert spectral_concepts.shape == expected_shape, "❌ 'spectral_concepts' 的形状与期望不符！"
    print("✅ 'spectral_concepts' 形状验证通过。")

    # 7. 验证KL损失
    print(f"模型输出 'kl_loss' 的值: {kl_loss.item():.4f}")
    assert kl_loss.dim() == 0, "❌ 'kl_loss' 应该是一个标量张量！"
    print("✅ 'kl_loss' 类型验证通过。")

    print("\n🎉 全部验证成功！新版光谱编码器工作正常。")
    print("--- 验证结束 ---")

if __name__ == '__main__':
    main()
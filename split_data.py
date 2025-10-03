# split_dataset.py
import os
import pickle
import argparse
import random
from tqdm.auto import tqdm

def main():
    """
    主函数，用于加载、分割和保存数据集。
    """
    parser = argparse.ArgumentParser(
        description="将数据集文件(.pkl)按照指定比例分割为训练集和测试集。"
    )
    parser.add_argument(
        '--input_path', 
        type=str, 
        required=True, 
        help="输入的完整数据集.pkl文件路径"
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True, 
        help="保存分割后文件的输出目录"
    )
    parser.add_argument(
        '--train_name', 
        type=str, 
        default='train_data.pkl', 
        help="训练集输出文件名 (默认: train_data.pkl)"
    )
    parser.add_argument(
        '--test_name', 
        type=str, 
        default='test_data.pkl', 
        help="测试集输出文件名 (默认: test_data.pkl)"
    )
    parser.add_argument(
        '--split_ratio', 
        type=float, 
        default=0.9, 
        help="训练集所占的比例 (默认: 0.9)"
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help="用于随机打乱的种子，确保结果可复现 (默认: 42)"
    )
    args = parser.parse_args()

    # --- 1. 设置随机种子以保证结果可复现 ---
    print(f"🌱 使用随机种子: {args.seed}")
    random.seed(args.seed)

    # --- 2. 加载原始数据集 ---
    print(f"🔄 正在从 '{args.input_path}' 加载数据...")
    try:
        with open(args.input_path, 'rb') as f:
            full_dataset = pickle.load(f)
        if not isinstance(full_dataset, list):
            raise TypeError("输入文件应包含一个Python列表 (list)。")
        print(f"✅ 数据加载成功，共包含 {len(full_dataset)} 个样本。")
    except FileNotFoundError:
        print(f"❌ 错误: 输入文件未找到 '{args.input_path}'")
        return
    except Exception as e:
        print(f"❌ 加载数据时发生错误: {e}")
        return

    # --- 3. 随机打乱数据集 ---
    print("🔀 正在随机打乱数据集...")
    random.shuffle(full_dataset)
    print("✅ 数据集已打乱。")

    # --- 4. 计算分割点并分割数据 ---
    num_total = len(full_dataset)
    num_train = int(num_total * args.split_ratio)
    
    train_set = full_dataset[:num_train]
    test_set = full_dataset[num_train:]

    print(f"✂️ 数据集分割完成:")
    print(f"   - 训练集样本数: {len(train_set)}")
    print(f"   - 测试集样本数: {len(test_set)}")

    # --- 5. 创建输出目录并保存文件 ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_train_path = os.path.join(args.output_dir, args.train_name)
    output_test_path = os.path.join(args.output_dir, args.test_name)

    print(f"\n💾 正在保存训练集至 '{output_train_path}'...")
    with open(output_train_path, 'wb') as f:
        pickle.dump(train_set, f)
    print("✅ 训练集保存成功。")

    print(f"💾 正在保存测试集至 '{output_test_path}'...")
    with open(output_test_path, 'wb') as f:
        pickle.dump(test_set, f)
    print("✅ 测试集保存成功。")

    print("\n🎉 所有操作完成！")


if __name__ == '__main__':
    main()
import os
import argparse
import yaml
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from tqdm.auto import tqdm
from easydict import EasyDict
from typing import Callable, Union, List
import numpy as np
import shutil

# --- 1. 导入模型定义 ---
# 确保 mask_generator_layers.py 在 Python 路径中
try:
    # 假设 mask_generator_layers 在 src 目录下
    from src.mask_generator_layers import MaskGeneratorNet
except ImportError:
    try:
        # 或者在当前目录
        from mask_generator_layers import MaskGeneratorNet
    except ImportError:
        print("错误: 无法导入 'MaskGeneratorNet'。")
        print("请确保 'mask_generator_layers.py' 文件在 'src' 目录或当前目录中。")
        exit()


# --- 2. 导入必要的 utils ---
# 假设 src 目录在 Python 路径中
try:
    # 导入您代码库中的日志工具
    from src.utils.misc import get_logger, get_new_log_dir
except ImportError:
    print("警告: 无法从 'src.utils.misc' 导入日志工具。将使用基本的 print。")
    # 定义一个简单的回退（fallback）
    import datetime
    class SimpleLogger:
        def info(self, msg): print(msg)
        def warning(self, msg): print(f"警告: {msg}")
        def error(self, msg): print(f"错误: {msg}")

    def get_logger(name, log_dir):
        # 简单的回退：只打印到控制台，不写入文件
        print(f"(注意: 日志未写入文件。无法导入 get_logger)")
        return SimpleLogger()

    def get_new_log_dir(base_dir, prefix='run'):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
        # 如果目录已存在，添加后缀
        count = 1
        while os.path.exists(log_dir):
             log_dir = os.path.join(base_dir, f"{prefix}_{timestamp}_{count}")
             count += 1
        os.makedirs(log_dir, exist_ok=True) # 确保创建目录
        return log_dir

# --- 3. 复制必要的辅助函数 ---

def get_edge_length(pos, edge_index):
    """计算边长"""
    row, col = edge_index
    dist = torch.norm(pos[row] - pos[col], p=2, dim=-1).view(-1, 1)
    return dist

def load_data(data_path, dataset_name="", logger=None):
    """加载 pickle 数据并计算真实边长"""
    log_func = logger.info if logger else print
    log_func_err = logger.error if logger else print

    log_func(f"正在从 '{data_path}' 加载 {dataset_name} 数据...")
    try:
        with open(data_path, "rb") as f:
            data_list = pickle.load(f)
        if not data_list:
            raise ValueError("加载的数据列表为空。")
        log_func(f"成功加载 {len(data_list)} 个样本。")
    except FileNotFoundError:
        log_func_err(f"错误：找不到数据文件 '{data_path}'。请确保路径正确。")
        raise
    except Exception as e:
        log_func_err(f"加载数据时出错: {e}")
        raise

    log_func("根据 data.pos 计算真实边长...")
    num_missing_pos = 0
    # 使用 tqdm 时确保 logger 存在
    pbar_desc = "计算边长"
    data_iterator = tqdm(data_list, desc=pbar_desc) if logger else data_list

    for i, data in enumerate(data_iterator):
        if not hasattr(data, 'pos') or data.pos is None:
            log_func_err(f"错误：样本 {i} (SMILES: {getattr(data, 'smiles', 'N/A')}) 缺少 'pos' 属性。")
            log_func_err("请确保使用了包含 'pos' 的预处理脚本重新生成数据。")
            num_missing_pos += 1
            if hasattr(data, 'edge_index'):
                data.edge_length = torch.empty((data.edge_index.shape[1], 1), dtype=torch.float32)
            else:
                 data.edge_length = torch.empty((0, 1), dtype=torch.float32)
            continue

        if hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.numel() > 0:
            data.edge_length = get_edge_length(data.pos, data.edge_index)
        else:
            data.edge_length = torch.empty((0, 1), dtype=torch.float32)

    if num_missing_pos > 0:
        log_func_err(f"严重错误: {num_missing_pos} 个样本缺少 'pos' 属性。测试将失败。")
        exit()
    log_func("真实边长计算完成。")

    return data_list


# --- 4. 主测试函数 ---

def main():
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description="测试掩码生成器模型 (V3 - 保存结果)")
    parser.add_argument("--config", type=str, required=True, help="模型配置文件的路径 (e.g., /path/to/config_mask_gen.yml)")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点文件的路径 (e.g., /path/to/best.pt)")
    parser.add_argument("--test_data_path", type=str, required=True, help="测试集 .pkl 文件的路径 (e.g., /path/to/test_mask_gen.pkl)")
    parser.add_argument("--log_base_dir", type=str, default="./logs_mask_test", help="存放所有评估结果的 *基础* 目录")
    parser.add_argument("--device", type=str, default="cuda", help="使用的设备 (cuda 或 cpu)")
    parser.add_argument("--batch_size", type=int, default=128, help="评估时的批次大小")
    args = parser.parse_args()

    # --- 创建新的日志目录 ---
    config_name = os.path.basename(args.config).split('.')[0]
    eval_log_dir = get_new_log_dir(args.log_base_dir, prefix=f"eval_{config_name}")

    # --- 设置日志记录 ---
    logger = get_logger("test_mask_gen_no_viz", eval_log_dir)
    logger.info(f"评估结果将保存至: {eval_log_dir}")
    logger.info(f"评估参数 (无可视化): {args}")

    # --- 备份配置文件 ---
    try:
        shutil.copyfile(args.config, os.path.join(eval_log_dir, "config_mask_gen.yml"))
        logger.info(f"配置文件已备份至: {eval_log_dir}")
    except Exception as e:
        logger.warning(f"无法备份配置文件: {e}")

    # --- 加载配置 ---
    try:
        with open(args.config, "r") as f:
            config = EasyDict(yaml.safe_load(f))
    except FileNotFoundError:
        logger.error(f"错误: 配置文件未找到: {args.config}")
        exit()

    # --- 设备设置 ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # --- 加载测试数据 ---
    test_data = load_data(args.test_data_path, dataset_name="测试集", logger=logger)

    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.train.get('num_workers', 2),
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=False
    )
    logger.info(f"测试集样本数: {len(test_data)}")

    # --- 加载模型 ---
    logger.info("正在初始化模型...")
    model = MaskGeneratorNet(
        embed_dim=config.model.embed_dim,
        max_atomic_number=config.model.max_atomic_number,
        num_bond_types=config.model.num_bond_types,
        num_convs=config.model.num_convs,
        activation=config.model.activation,
        short_cut=config.model.short_cut,
        concat_hidden=config.model.concat_hidden,
        output_mlp_hidden_dims=config.model.output_mlp_hidden_dims
    ).to(device)

    if not os.path.exists(args.checkpoint):
        logger.error(f"错误: 检查点文件未找到: {args.checkpoint}")
        exit()

    try:
        logger.info(f"正在加载检查点: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])
        logger.info("模型加载成功。")
    except Exception as e:
        logger.error(f"加载模型状态字典失败: {e}")
        exit()

    # --- 评估指标计算 与 结果收集 ---
    logger.info("开始在测试集上评估并收集结果...")
    model.eval()
    
    all_results = [] # 用于存储每个样本的详细结果
    all_preds_tensors = [] # 用于计算总体指标
    all_targets_tensors = [] # 用于计算总体指标

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="推理和收集结果"):
            batch = batch.to(device)
            pred_scores = model(batch)
            target_scores = batch.atom_flexibility_score
            
            # 收集用于总体指标计算的张量
            all_preds_tensors.append(pred_scores.cpu())
            all_targets_tensors.append(target_scores.cpu())

            # --- *** 新增：拆分批次并保存详细结果 *** ---
            ptr = batch.ptr.cpu().tolist()
            smiles_list = batch.smiles
            atom_type_tensor = batch.atom_type.cpu()
            
            for i in range(batch.num_graphs):
                start, end = ptr[i], ptr[i+1]
                pred = pred_scores[start:end].cpu()
                target = target_scores[start:end].cpu()
                atom_type = atom_type_tensor[start:end]
                smiles = smiles_list[i]
                
                all_results.append({
                    "smiles": smiles,
                    "pred_scores": pred,    # (N_atoms, 1)
                    "true_scores": target,  # (N_atoms, 1)
                    "atom_type": atom_type    # (N_atoms,)
                })
            # --- *** 收集结束 *** ---

    # --- 计算总体指标 ---
    all_preds = torch.cat(all_preds_tensors, dim=0)
    all_targets = torch.cat(all_targets_tensors, dim=0)

    mse = F.mse_loss(all_preds, all_targets).item()
    mae = F.l1_loss(all_preds, all_targets).item()

    target_mean = all_targets.mean()
    ss_tot = torch.sum((all_targets - target_mean)**2)
    ss_res = torch.sum((all_targets - all_preds)**2)
    if ss_tot.item() < 1e-6:
        r2 = float('nan')
        logger.warning("目标值方差接近零，R-squared 未定义。")
    else:
        r2 = (1 - ss_res / ss_tot).item()

    metrics = {
        "MSE": mse,
        "MAE": mae,
        "R-squared": r2
    }

    logger.info("\n--- 评估结果 ---")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.6f}")

    # --- 保存指标到文件 ---
    results_file = os.path.join(eval_log_dir, "test_metrics.txt")
    with open(results_file, 'w') as f:
        f.write(f"评估检查点: {args.checkpoint}\n")
        f.write(f"评估配置文件: {args.config}\n")
        f.write(f"测试数据集: {args.test_data_path}\n")
        f.write("--- 评估指标 ---\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")
    logger.info(f"评估指标已保存至: {results_file}")

    # --- *** 新增：保存详细推理结果 *** ---
    results_pkl_path = os.path.join(eval_log_dir, "inference_results.pkl")
    try:
        with open(results_pkl_path, 'wb') as f:
            pickle.dump(all_results, f)
        logger.info(f"详细推理结果 (含SMILES、分数、原子类型) 已保存至: {results_pkl_path}")
    except Exception as e:
        logger.error(f"保存推理结果 .pkl 文件失败: {e}")
    # --- *** 保存结束 *** ---

    logger.info("--- 测试完成 ---")


if __name__ == "__main__":
    main()
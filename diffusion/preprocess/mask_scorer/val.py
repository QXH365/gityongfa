# analyze_results.py

import os
import argparse
import yaml
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm.auto import tqdm
from easydict import EasyDict
from typing import List
import numpy as np
import shutil
from scipy.stats import spearmanr # 用于计算秩相关系数

# --- 尝试导入可视化库 ---
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D, MolsToGridImage
    from PIL import Image
    import io
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    VIZ_ENABLED = True
except ImportError:
    print("警告: 无法导入 RDKit, Pillow 或 Matplotlib。")
    print("可视化功能将被禁用。 Spearman 相关系数计算仍会执行 (需要 Scipy)。")
    VIZ_ENABLED = False

# --- 1. 导入必要的 utils ---
# 假设 src 目录在 Python 路径中
try:
    from src.utils.misc import get_logger, get_new_log_dir
except ImportError:
    print("警告: 无法从 'src.utils.misc' 导入日志工具。将使用基本的 print。")
    import datetime
    class SimpleLogger:
        def info(self, msg): print(msg)
        def warning(self, msg): print(f"警告: {msg}")
        def error(self, msg): print(f"错误: {msg}")

    def get_logger(name, log_dir):
        print(f"(注意: 日志未写入文件。无法导入 get_logger)")
        return SimpleLogger()

    def get_new_log_dir(base_dir, prefix='run'):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
        count = 1
        while os.path.exists(log_dir):
             log_dir = os.path.join(base_dir, f"{prefix}_{timestamp}_{count}")
             count += 1
        os.makedirs(log_dir, exist_ok=True) # 确保创建目录
        return log_dir

# --- 2. 辅助函数 ---

if VIZ_ENABLED:
    def get_color_gradient(score, cmap_name='coolwarm'):
        """将 0-1 的分数映射到 RGBA 颜色"""
        cmap = cm.get_cmap(cmap_name)
        norm = mcolors.Normalize(vmin=0, vmax=1)
        
        # --- 修复：处理 NaN 值并确保输出是元组 ---
        if np.isnan(score):
            rgba = cmap(np.nan) # 获取 matplotlib 的 "bad" 颜色
        else:
            rgba = cmap(norm(score))
        
        # 将 matplotlib 的输出 (可能是 ndarray) 转换为标准 float 元组
        # RDKit 的 C++ 绑定需要标准 float
        rgba_tuple = tuple(float(c) for c in rgba)
        
        # 返回 (r, g, b, 0.8)
        return rgba_tuple[:3] + (0.8,)
        # --- 修复结束 ---

    def prepare_mol_for_drawing(smiles, atom_types, scores_tensor, cmap_name, norm_heavy_only, logger):
        """准备 RDKit Mol 对象和绘图所需的高亮/颜色信息"""
        log_func_warn = logger.warning if logger else print
        
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            log_func_warn(f"RDKit 无法解析 SMILES: {smiles}。")
            return None, None, None

        try:
            mol = Chem.AddHs(mol)
            AllChem.Compute2DCoords(mol)
        except Exception as e:
            log_func_warn(f"RDKit 处理 {smiles} 出错: {e}。")
            return None, None, None

        num_atoms = mol.GetNumAtoms()
        # --- 修复: 确保 scores_tensor 被正确 squeeze ---
        if num_atoms != scores_tensor.numel():
            log_func_warn(f"警告: {smiles} 原子数 ({num_atoms}) 与分数 ({scores_tensor.numel()}) 不匹配。")
            return None, None, None

        # 确保 scores_tensor 是一维的 (N,)
        scores_tensor_squeezed = scores_tensor.squeeze()

        scores_normalized = scores_tensor_squeezed.clone().cpu()
        if norm_heavy_only:
            heavy_atom_indices = [i for i, atype in enumerate(atom_types) if atype > 1]
            if heavy_atom_indices:
                heavy_scores = scores_normalized[heavy_atom_indices]
                min_heavy_score, max_heavy_score = heavy_scores.min(), heavy_scores.max()
                if max_heavy_score - min_heavy_score > 1e-6:
                    scores_normalized = (scores_normalized - min_heavy_score) / (max_heavy_score - min_heavy_score)
                    scores_normalized = torch.clamp(scores_normalized, 0.0, 1.0)
        
        # --- 修复: .tolist() 应该在一个 1D 张量上调用 ---
        # 如果 scores_normalized 是 0D (单个原子), tolist() 返回 float
        # 如果 scores_normalized 是 1D (多个原子), tolist() 返回 list[float]
        if scores_normalized.dim() == 0:
            scores_list = [scores_normalized.item()] # 转换为列表
        else:
            scores_list = scores_normalized.tolist()
        # --- 修复结束 ---

        atom_colors = {}
        for i, score in enumerate(scores_list):
            atom_colors[i] = get_color_gradient(score, cmap_name=cmap_name)

        highlight_atoms = list(range(num_atoms))
        
        return mol, highlight_atoms, atom_colors

# --- 3. 主分析函数 ---

def main():
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description="分析掩码生成器模型预测结果")
    parser.add_argument("--results_pkl", type=str, required=True, help="指向 inference_results.pkl 文件的路径")
    parser.add_argument("--log_base_dir", type=str, default="./logs_mask_analysis", help="存放分析结果的基础目录")
    parser.add_argument("--num_viz", type=int, default=20, help="生成对比图的样本数量 (如果启用可视化)")
    parser.add_argument("--cmap", type=str, default="coolwarm", help="可视化使用的颜色映射")
    parser.add_argument("--norm_heavy_only", action='store_true', help="可视化时基于重原子归一化")
    args = parser.parse_args()

    # --- 创建新的日志目录 ---
    results_name = os.path.basename(os.path.dirname(args.results_pkl)) # 基于父目录名
    if not results_name: results_name = "analysis"
    analysis_log_dir = get_new_log_dir(args.log_base_dir, prefix=f"analyze_{results_name}")
    viz_dir = os.path.join(analysis_log_dir, "visualizations")
    if VIZ_ENABLED:
        os.makedirs(viz_dir, exist_ok=True)

    # --- 设置日志记录 ---
    logger = get_logger("analyze_mask_gen", analysis_log_dir)
    logger.info(f"分析结果将保存至: {analysis_log_dir}")
    logger.info(f"分析参数: {args}")

    # --- 加载推理结果 ---
    logger.info(f"正在加载推理结果: {args.results_pkl}")
    try:
        with open(args.results_pkl, "rb") as f:
            all_results = pickle.load(f)
        if not all_results:
            raise ValueError("加载的推理结果列表为空。")
        logger.info(f"成功加载 {len(all_results)} 个分子的推理结果。")
    except FileNotFoundError:
        logger.error(f"错误：找不到推理结果文件 '{args.results_pkl}'。")
        exit()
    except Exception as e:
        logger.error(f"加载推理结果时出错: {e}")
        exit()

    # --- 分析任务 1: 可视化对比 (如果启用) ---
    if VIZ_ENABLED and args.num_viz > 0:
        logger.info(f"\n开始生成 {min(args.num_viz, len(all_results))} 个可视化对比图...")
        mols_per_row = 2 # True | Pred
        img_size_single = (350, 300) # Size for one molecule image

        num_to_viz = min(args.num_viz, len(all_results))
        for i in tqdm(range(num_to_viz), desc="生成对比图像"):
            result_data = all_results[i]
            
            true_scores = result_data["true_scores"]
            pred_scores = result_data["pred_scores"]
            smiles = result_data["smiles"]
            atom_types = result_data["atom_type"]
            
            mols = []
            legends = []
            
            # 准备真实图
            mol_true, highlights_true, colors_true = prepare_mol_for_drawing(
                smiles, atom_types, true_scores, args.cmap, args.norm_heavy_only, logger
            )
            if mol_true:
                mols.append(mol_true)
                legends.append(f"Sample {i}: True")
            
            # 准备预测图
            mol_pred, highlights_pred, colors_pred = prepare_mol_for_drawing(
                smiles, atom_types, pred_scores, args.cmap, args.norm_heavy_only, logger
            )
            if mol_pred:
                mols.append(mol_pred)
                legends.append(f"Sample {i}: Predicted")

            # 绘制网格图
            if len(mols) == 2:
                 highlightAtomLists = [highlights_true, highlights_pred]
                 highlightAtomColors = [colors_true, colors_pred]
                 
                 try:
                      grid_image = MolsToGridImage(
                           mols,
                           molsPerRow=mols_per_row,
                           subImgSize=img_size_single,
                           legends=legends,
                           highlightAtomLists=highlightAtomLists,
                           highlightAtomColors=highlightAtomColors,
                           useSVG=False # 使用 PNG
                      )
                      filename = os.path.join(viz_dir, f"comparison_{i:03d}.png")
                      grid_image.save(filename)
                 except Exception as draw_err:
                      logger.error(f"绘制网格图像失败 for sample {i} (SMILES: {smiles}): {draw_err}")
            else:
                 logger.warning(f"无法为样本 {i} (SMILES: {smiles}) 生成对比图 (缺少一个或两个分子图)")

        logger.info(f"可视化对比图已保存至: {viz_dir}")
    elif not VIZ_ENABLED:
         logger.info("\n可视化功能因缺少依赖库而被禁用。跳过图像生成。")
    else:
         logger.info("\nnum_viz 设置为 0，跳过可视化。")


    # --- 分析任务 2: 分布趋势一致性 (Spearman Rank Correlation) ---
    logger.info("\n开始计算 Spearman 秩相关系数...")
    spearman_correlations = []
    skipped_mols = 0

    for i, result_data in enumerate(tqdm(all_results, desc="计算 Spearman 相关系数")):
        true_scores = result_data["true_scores"].squeeze().numpy()
        pred_scores = result_data["pred_scores"].squeeze().numpy()

        # --- 修复：增加 NaN 检查 ---
        if np.isnan(true_scores).any() or np.isnan(pred_scores).any():
             logger.warning(f"Skipping Spearman for sample {i} (SMILES: {result_data['smiles']}): Scores contain NaN.")
             skipped_mols += 1
             spearman_correlations.append(np.nan)
             continue
        
        # 修复：检查方差时，允许一个数组有方差即可
        if len(true_scores) < 2 or (np.all(true_scores == true_scores[0]) and np.all(pred_scores == pred_scores[0])):
             logger.warning(f"Skipping Spearman for sample {i} (SMILES: {result_data['smiles']}): Insufficient data points or no variance in both arrays.")
             skipped_mols += 1
             spearman_correlations.append(np.nan)
             continue

        try:
            correlation, p_value = spearmanr(true_scores, pred_scores)
            spearman_correlations.append(correlation)
        except (ValueError, TypeError) as e: # 捕获 TypeError
            logger.error(f"Spearman calculation failed for sample {i} (SMILES: {result_data['smiles']}): {e}")
            spearman_correlations.append(np.nan)

    # 计算平均相关系数 (忽略 NaNs)
    valid_correlations = [c for c in spearman_correlations if not np.isnan(c)]
    if valid_correlations:
        average_spearman = np.mean(valid_correlations)
        std_spearman = np.std(valid_correlations)
        median_spearman = np.median(valid_correlations)
        logger.info("\n--- Spearman 秩相关系数结果 ---")
        logger.info(f"计算的分子数: {len(valid_correlations)} (跳过 {skipped_mols} 个)")
        logger.info(f"平均 Spearman 相关系数: {average_spearman:.4f}")
        logger.info(f"Spearman 相关系数标准差: {std_spearman:.4f}")
        logger.info(f"Spearman 相关系数中位数: {median_spearman:.4f}")

        # 保存 Spearman 结果
        spearman_file = os.path.join(analysis_log_dir, "spearman_analysis.txt")
        with open(spearman_file, 'w') as f:
            f.write(f"分析的推理结果文件: {args.results_pkl}\n")
            f.write("--- Spearman Rank Correlation ---\n")
            f.write(f"计算的分子数: {len(valid_correlations)} (跳过 {skipped_mols} 个)\n")
            f.write(f"平均值: {average_spearman:.6f}\n")
            f.write(f"标准差: {std_spearman:.6f}\n")
            f.write(f"中位数: {median_spearman:.6f}\n\n")
            f.write("每个分子的相关系数 (不含 NaN):\n")
            for corr in valid_correlations:
                f.write(f"{corr:.6f}\n")
        logger.info(f"Spearman 相关系数结果已保存至: {spearman_file}")

        # 可选：绘制分布直方图
        if VIZ_ENABLED:
            try:
                plt.figure(figsize=(10, 6))
                plt.hist(valid_correlations, bins=30, edgecolor='black', alpha=0.7)
                plt.title('Spearman Correlation Distribution per Molecule')
                plt.xlabel('Spearman Correlation Coefficient')
                plt.ylabel('Frequency (Number of Molecules)')
                plt.axvline(average_spearman, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {average_spearman:.3f}')
                plt.axvline(median_spearman, color='g', linestyle='dotted', linewidth=2, label=f'Median: {median_spearman:.3f}')
                plt.legend()
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                hist_filename = os.path.join(analysis_log_dir, "spearman_distribution.png")
                plt.savefig(hist_filename)
                plt.close()
                logger.info(f"Spearman 相关系数分布直方图已保存至: {hist_filename}")
            except Exception as plot_err:
                logger.error(f"绘制 Spearman 分布图失败: {plot_err}")

    else:
        logger.warning("未能计算任何有效的 Spearman 相关系数。")

    logger.info("--- 分析完成 ---")


if __name__ == "__main__":
    if not VIZ_ENABLED:
        print("\n---")
        print("警告: 缺少 RDKit, Pillow, 或 Matplotlib 库。")
        print("将无法运行可视化分析。Spearman 相关系数计算仍将尝试运行 (需要 Scipy)。")
        print("请安装: pip install rdkit-pypi pillow matplotlib scipy")
        print("---")
    main()
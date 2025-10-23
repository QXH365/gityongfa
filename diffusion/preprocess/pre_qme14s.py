import os
import pickle
import copy
import glob
import numpy as np
import random
import argparse
import yaml
from easydict import EasyDict
from tqdm import tqdm
import sys
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, BondType
from rdkit import RDLogger
# 使用 scipy 进行线性插值
import scipy.interpolate

# --- 修正导入路径 ---
# 确保项目根目录在 PYTHONPATH 或从根目录运行
try:
    from mask_scorer.src.mask_generator_layers import MaskGeneratorNet
    from mask_scorer.src.misc import get_logger, seed_all
    # from mask_scorer.src.common import ... # 如有需要
except ImportError as e:
    print(f"错误: 无法导入 Mask Scorer 模块。")
    print(f"  请确保 'mask_scorer' 和 'mask_scorer/src' 目录下存在 '__init__.py' 文件。")
    print(f"  并且从项目根目录 (gityongfa-5) 运行此脚本，或者该目录已添加到 PYTHONPATH。")
    print(f"原始错误: {e}")
    exit()

try:
    # 假设 transforms.py 在 diffusion/preprocess 目录下
    # 需要能够被 Python 找到，通常在运行脚本的目录下或 PYTHONPATH 中
    # 如果你的 transforms.py 在 diffusion/preprocess 下，运行脚本时 Python 可能找不到它
    # 更好的方式是将 transforms.py 移到 diffusion/utils 下，并确保 utils 有 __init__.py
    # 然后使用 from diffusion.utils.transforms import ...
    # 这里暂时假设它能被找到
    from transforms import AddHigherOrderEdges, CountNodesPerGraph
except ImportError:
    # 尝试从 utils 导入 (推荐结构)
    try:
        from diffusion.utils.transforms import AddHigherOrderEdges, CountNodesPerGraph
    except ImportError:
        print("错误: 无法导入 diffusion transforms。")
        print("  请确保 'transforms.py' 在 PYTHONPATH 中可访问的路径下")
        print("  (例如，在 'diffusion/utils' 目录下，并确保 'diffusion' 和 'diffusion/utils' 有 '__init__.py')")
        exit()
# --- 结束导入 ---


RDLogger.DisableLog('rdApp.*')
BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}

def get_edge_length(pos, edge_index):
    """计算边长"""
    if edge_index is None or edge_index.numel() == 0:
        return torch.empty((0, 1), dtype=pos.dtype, device=pos.device)
    row, col = edge_index
    dist = torch.norm(pos[row] - pos[col], p=2, dim=-1).view(-1, 1)
    return dist

# --- 更新后的光谱处理函数 ---
def process_spectrum(spectrum, target_len, normalize):
    """根据参数选择性地归一化并使用线性插值进行重采样。"""
    spectrum_np = np.array(spectrum, dtype=np.float32)
    if spectrum_np.size == 0:
         return torch.zeros(target_len, dtype=torch.float32) if target_len > 0 else torch.empty(0, dtype=torch.float32)

    if normalize:
        min_val, max_val = spectrum_np.min(), spectrum_np.max()
        range_val = max_val - min_val
        spectrum_np = (spectrum_np - min_val) / range_val if range_val > 1e-8 else np.zeros_like(spectrum_np)

    current_len = len(spectrum_np)
    if target_len > 0 and current_len != target_len:
        if current_len < 2:
            resampled = np.full(target_len, spectrum_np[0] if current_len == 1 else 0.0, dtype=np.float32)
        else:
            x_original = np.linspace(0, 1, current_len)
            x_target = np.linspace(0, 1, target_len)
            try:
                # 使用线性插值
                interp_func = scipy.interpolate.interp1d(x_original, spectrum_np, kind='linear', fill_value="extrapolate")
                resampled = interp_func(x_target).astype(np.float32)
            except ValueError: # 如果插值失败（例如所有值相同）
                resampled = np.full(target_len, np.mean(spectrum_np), dtype=np.float32) # 使用平均值填充

        spectrum_processed = torch.tensor(resampled, dtype=torch.float32)
    else: # 不需要重采样或 target_len <= 0
        spectrum_processed = torch.tensor(spectrum_np, dtype=torch.float32)

    return spectrum_processed

# --- 其他辅助函数 (保持不变) ---
def canonicalize_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None

def parse_spectrum_csv_with_smiles(file_path: str):
    with open(file_path, 'r') as f: lines = [l.strip() for l in f if l.strip()]
    if not lines: raise ValueError(f"{file_path} empty")
    smiles = lines[0]; coord_lines, spectrum_parts = [], []; is_coord = True
    for line in lines[1:]:
        parts = line.split(',')
        try:
            if len(parts) == 4 and is_coord: _=[float(p) for p in parts]; coord_lines.append(parts)
            else: is_coord = False; spectrum_parts.extend([p for p in parts if p.strip()])
        except ValueError: is_coord = False; spectrum_parts.extend([p for p in parts if p.strip()])
    if not coord_lines: raise ValueError(f"{file_path} no coords")
    if not spectrum_parts: raise ValueError(f"{file_path} no spectrum")
    coords = np.array([list(map(float, line)) for line in coord_lines])
    z = torch.tensor(coords[:, 0], dtype=torch.long)
    pos = torch.tensor(coords[:, 1:], dtype=torch.float32)
    try: spectrum = [float(val) for val in spectrum_parts]
    except ValueError: raise ValueError(f"{file_path} non-numeric spectrum")
    return smiles, z, pos, spectrum

def create_mol_with_coords_from_smiles(smiles: str, true_coords: np.ndarray, true_elements: list) -> Mol:
    try:
        template_mol = Chem.MolFromSmiles(smiles);
        if not template_mol: return None
        template_mol = Chem.AddHs(template_mol)
        if template_mol.GetNumAtoms() != len(true_elements): return None
        query_mol = Chem.RWMol()
        for elem in true_elements:
            try: query_mol.AddAtom(Chem.Atom(Chem.GetPeriodicTable().GetAtomicNumber(elem)))
            except: return None
        if sorted([a.GetAtomicNum() for a in template_mol.GetAtoms()]) != sorted([a.GetAtomicNum() for a in query_mol.GetAtoms()]): return None
        match = template_mol.GetSubstructMatch(query_mol.GetMol())
        if not match or len(match) != template_mol.GetNumAtoms(): return None
        conf = Chem.Conformer(template_mol.GetNumAtoms())
        if len(match) != len(true_coords): return None
        for i in range(template_mol.GetNumAtoms()):
            try: conf.SetAtomPosition(i, true_coords[match[i]].tolist())
            except: return None
        final_mol = copy.deepcopy(template_mol); final_mol.RemoveAllConformers(); final_mol.AddConformer(conf)
        try: Chem.SanitizeMol(final_mol)
        except: return None
        return final_mol
    except: return None

def rdmol_to_data(mol: Mol) -> Data:
    if not mol or mol.GetNumConformers() < 1: return None
    try:
        N = mol.GetNumAtoms()
        pos = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float32)
        z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
        row, col, edge_type = [], [], []
        for b in mol.GetBonds():
            start, end = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if start < N and end < N:
                row.extend([start, end]); col.extend([end, start])
                edge_type.extend([BOND_TYPES.get(b.GetBondType(), 0)] * 2)
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)
        if edge_index.numel() > 0:
            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index, edge_type_tensor = edge_index[:, perm], edge_type_tensor[perm]
        smiles = Chem.MolToSmiles(mol, canonical=True)
        return Data(atom_type=z, pos=pos, edge_index=edge_index, edge_type=edge_type_tensor, rdmol=copy.deepcopy(mol), smiles=smiles)
    except: return None

# --- 主函数 ---
def main():
    parser = argparse.ArgumentParser(description="Preprocess diffusion dataset with flexibility scores")
    # 参数定义（保持你的默认值和类型）
    parser.add_argument('--ir_dir', type=str, default='../../../dataset/qme14s/IR_broaden', help="Path to IR spectra CSV directory")
    parser.add_argument('--raman_dir', type=str, default='../../../dataset/qme14s/Raman_broaden', help="Path to Raman spectra CSV directory")
    parser.add_argument('--output_dir', type=str, default='./qme14s_100', help="Directory to save processed files")
    parser.add_argument('--max_molecules', type=int, default=100, help="Max molecules to process")
    parser.add_argument('--edge_order', type=int, default=3, help="Order for higher order edges")
    parser.add_argument('--normalize_spectra', action='store_true', help="Normalize spectra to [0, 1]") # 改回 store_true
    parser.add_argument('--ir_length', type=int, default=-1, help="Target length for IR spectrum (-1 for no change)") # 默认-1
    parser.add_argument('--raman_length', type=int, default=-1, help="Target length for Raman spectrum (-1 for no change)") # 默认-1
    parser.add_argument('--scorer_config', type=str,default='./mask_scorer/configs/config.yml', help="Path to mask_scorer config yaml")
    parser.add_argument('--scorer_ckpt', type=str, default='./mask_scorer/best.pt', help="Path to mask_scorer checkpoint (.pt)")
    parser.add_argument('--train_ratio', type=float, default=0.9, help="Ratio of data for training set")
    parser.add_argument('--seed', type=int, default=2024, help="Random seed for train/test split")
    parser.add_argument('--device', type=str, default='cuda', help='Device for scorer inference')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for scorer inference')
    args = parser.parse_args()

    # --- 初始化 ---
    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger('preprocess_with_flex', args.output_dir)
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    logger.info(f"运行参数: {args}")

    # --- 加载 Mask Scorer 模型 ---
    logger.info("加载 Mask Scorer 模型...")
    # ... (加载逻辑保持不变) ...
    try:
        with open(args.scorer_config, "r") as f: scorer_config = EasyDict(yaml.safe_load(f))
    except FileNotFoundError: logger.error(f"Scorer config 未找到: {args.scorer_config}"); exit()
    try:
        scorer_model = MaskGeneratorNet(
            embed_dim=scorer_config.model.embed_dim, max_atomic_number=scorer_config.model.max_atomic_number,
            num_bond_types=scorer_config.model.num_bond_types, num_convs=scorer_config.model.num_convs,
            activation=scorer_config.model.activation, short_cut=scorer_config.model.short_cut,
            concat_hidden=scorer_config.model.concat_hidden, output_mlp_hidden_dims=scorer_config.model.output_mlp_hidden_dims
        ).to(device)
    except AttributeError as e: logger.error(f"初始化 MaskGeneratorNet 出错，请检查 scorer config: {e}"); exit()
    try:
        ckpt = torch.load(args.scorer_ckpt, map_location=device); scorer_model.load_state_dict(ckpt['model']); scorer_model.eval()
        logger.info("Mask Scorer 模型加载成功。")
    except FileNotFoundError: logger.error(f"Scorer checkpoint 未找到: {args.scorer_ckpt}"); exit()
    except Exception as e: logger.error(f"加载 scorer checkpoint 失败: {e}"); exit()

    # --- 定义图变换 (用于 diffusion 模型) ---
    graph_transforms = Compose([CountNodesPerGraph(), AddHigherOrderEdges(order=args.edge_order)])

    # --- 主处理循环 ---
    logger.info("处理分子并进行灵活性分数推理...")
    data_for_scoring_list = [] # 存储用于 scorer 推理的基础 Data 对象
    original_spectra_list = [] # 存储原始光谱，以便后续处理
    processed_smiles = set()

    ir_files = sorted(glob.glob(os.path.join(args.ir_dir, 'IR_*.csv')))
    num_files_to_process = len(ir_files)
    if args.max_molecules is not None and args.max_molecules > 0:
        num_files_to_process = min(len(ir_files), args.max_molecules)
        logger.info(f"基于 --max_molecules 限制，最多处理 {num_files_to_process} 个分子。")
    ir_files_subset = ir_files[:num_files_to_process]

    # --- 第一阶段：解析文件，创建基础 Data 对象 ---
    logger.info("阶段 1: 解析文件并创建基础图结构...")
    for ir_path in tqdm(ir_files_subset, desc="解析与构建基础图"):
        file_id = os.path.basename(ir_path).replace('IR_', '').replace('.csv', '')
        raman_path = os.path.join(args.raman_dir, f'Raman_{file_id}.csv')
        if not os.path.exists(raman_path): continue

        try:
            smiles, z, pos, raw_ir_spec = parse_spectrum_csv_with_smiles(ir_path)
            _, _, _, raw_raman_spec = parse_spectrum_csv_with_smiles(raman_path)
            canonical_smiles = canonicalize_smiles(smiles)
            if not canonical_smiles or canonical_smiles in processed_smiles: continue

            elements = [Chem.GetPeriodicTable().GetElementSymbol(int(atom_num)) for atom_num in z]
            mol = create_mol_with_coords_from_smiles(smiles, pos.numpy(), elements)
            if not mol: continue

            base_data = rdmol_to_data(mol)
            if base_data is None: continue

            # *** 计算 scorer 需要的 edge_length ***
            base_data.edge_length = get_edge_length(base_data.pos, base_data.edge_index)

            data_for_scoring_list.append(base_data)
            original_spectra_list.append({'ir': raw_ir_spec, 'raman': raw_raman_spec, 'id': file_id}) # 存储原始光谱
            processed_smiles.add(canonical_smiles)

        except ValueError as ve: logger.warning(f"跳过 {file_id}: 解析错误 - {ve}"); continue
        except FileNotFoundError as fnf: logger.warning(f"跳过 {file_id}: 文件未找到 - {fnf}"); continue
        except Exception as e: logger.error(f"处理 {file_id} 时发生意外错误: {e}", exc_info=False); continue

    logger.info(f"成功构建 {len(data_for_scoring_list)} 个用于评分的基础分子图。")

    # --- 第二阶段：批量推理 Mask Scorer ---
    all_processed_data = [] # 最终包含所有信息的数据列表
    if data_for_scoring_list:
        logger.info("阶段 2: 进行灵活性分数批量推理...")
        scorer_loader = DataLoader(data_for_scoring_list, batch_size=args.batch_size, shuffle=False, num_workers=4)
        all_scores_list = [] # 存储每个 batch 的分数 tensor
        with torch.no_grad():
            for batch in tqdm(scorer_loader, desc="推理灵活性分数"):
                try:
                    batch = batch.to(device)
                    # 再次检查 edge_length 是否存在且维度匹配 (DataLoader 可能丢失属性或批处理错误)
                    if not hasattr(batch, 'edge_length') or batch.edge_length is None or \
                       (batch.edge_index.numel() > 0 and batch.edge_length.shape[0] != batch.edge_index.shape[1]):
                        logger.warning(f"批次中 edge_length 存在问题 (smiles: {batch.smiles[:2]}...). 跳过批次。"); continue
                    scores = scorer_model(batch)
                    all_scores_list.append(scores.cpu())
                except Exception as e: logger.error(f"评分批次时出错 (smiles: {batch.smiles[:2]}...): {e}. 跳过批次。", exc_info=False); continue

        # --- 第三阶段：整合分数、处理光谱、应用图变换 ---
        if all_scores_list:
            logger.info("阶段 3: 整合分数、处理光谱并应用图变换...")
            try:
                all_scores_tensor = torch.cat(all_scores_list, dim=0)
                logger.info(f"推理得到的分数张量形状: {all_scores_tensor.shape}")
                current_score_idx = 0
                processed_count = 0

                # 使用原始的 data_for_scoring_list 顺序来添加分数和处理
                for i, data in enumerate(tqdm(data_for_scoring_list, desc="整合与变换")):
                    num_nodes = data.num_nodes
                    if current_score_idx + num_nodes <= all_scores_tensor.shape[0]:
                        # 1. 添加灵活性分数
                        data.atom_flexibility_score = all_scores_tensor[current_score_idx : current_score_idx + num_nodes]
                        current_score_idx += num_nodes

                        # 2. 处理并添加光谱 (从 original_spectra_list 获取)
                        # 注意：需要确保 original_spectra_list 和 data_for_scoring_list 顺序一致
                        spectra = original_spectra_list[i]
                        try:
                            data.ir_spectrum = process_spectrum(spectra['ir'], args.ir_length, args.normalize_spectra)
                            data.raman_spectrum = process_spectrum(spectra['raman'], args.raman_length, args.normalize_spectra)
                        except Exception as spec_e:
                            logger.warning(f"处理光谱时出错 (ID: {spectra['id']}): {spec_e}. 跳过此样本。"); continue

                        # 3. 应用 diffusion 特定的图变换
                        try:
                            transformed_data = graph_transforms(data)
                            # (可选) 验证变换后的数据
                            # ...

                            all_processed_data.append(transformed_data)
                            processed_count += 1
                        except Exception as trans_e:
                            logger.warning(f"应用图变换时出错 (SMILES: {data.smiles}): {trans_e}. 跳过此样本。"); continue

                    else:
                        logger.warning("分数张量索引越界，停止处理剩余数据。")
                        break
                logger.info(f"成功处理并变换 {processed_count} 个分子。")

                # (可选) 验证总节点数和分数总数是否匹配
                total_nodes_final = sum(d.num_nodes for d in all_processed_data)
                if all_scores_tensor.shape[0] != total_nodes_final:
                     logger.warning(f"最终数据总节点数 ({total_nodes_final}) 与分数总数 ({all_scores_tensor.shape[0]}) 不匹配！")

            except Exception as e: logger.error(f"整合分数/光谱/变换时出错: {e}"); all_processed_data = []
        else: logger.warning("未推理出任何分数。"); all_processed_data = []
    else: logger.warning("初始数据列表为空。"); all_processed_data = []

    # --- 第四阶段：划分与保存 ---
    if all_processed_data:
        logger.info("阶段 4: 划分并保存数据集...")
        random.shuffle(all_processed_data)
        split_idx = int(len(all_processed_data) * args.train_ratio)
        train_data = all_processed_data[:split_idx]
        test_data = all_processed_data[split_idx:]
        logger.info(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")

        # 使用固定输出文件名
        train_path = os.path.join(args.output_dir, 'train.pkl')
        test_path = os.path.join(args.output_dir, 'test.pkl')

        logger.info(f"保存训练数据到: {train_path}")
        try:
            with open(train_path, 'wb') as f: pickle.dump(train_data, f)
        except Exception as e: logger.error(f"保存训练数据失败: {e}")
        logger.info(f"保存测试数据到: {test_path}")
        try:
            with open(test_path, 'wb') as f: pickle.dump(test_data, f)
        except Exception as e: logger.error(f"保存测试数据失败: {e}")
    else:
        logger.warning("无数据可供划分或保存。")

    logger.info("预处理完成！")

if __name__ == '__main__':
    main()
    RDLogger.EnableLog('rdApp.*') # 重新启用 RDKit 日志
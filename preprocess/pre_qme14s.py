import os
import pickle
import copy
import glob
import random
import numpy as np
from scipy.signal import resample
from tqdm import tqdm

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import Compose

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, BondType
from rdkit import RDLogger
from transforms import AddHigherOrderEdges, CountNodesPerGraph

# 禁用 RDKit 的冗余日志
RDLogger.DisableLog("rdApp.*")

# --- 全局常量定义 ---
BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}

# --- 辅助函数 ---


# 【优化】如果维度匹配，此函数现在会直接复制而不是重采样
def resample_spectrum(spectrum, target_len):
    """
    对光谱进行重采样至目标长度。
    如果原始长度与目标长度相同，则直接转换类型而不进行重采样。
    """
    spectrum_np = np.array(spectrum, dtype=np.float32)

    if spectrum_np.shape[0] == target_len:
        # 维度匹配，直接返回张量副本
        return torch.from_numpy(spectrum_np)
    else:
        # 维度不匹配，执行重采样
        resampled = resample(spectrum_np, target_len)
        return torch.tensor(resampled, dtype=torch.float32)


def canonicalize_smiles(smiles: str):
    """将SMILES字符串转换为唯一的规范形式。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol, canonical=True)
    return None


def parse_spectrum_csv_with_smiles(file_path: str):
    """解析包含SMILES、原子坐标和光谱数据的特殊CSV文件。"""
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    smiles = lines[0]
    coord_lines, spectrum_parts = [], []
    is_coord_section = True

    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) == 4 and is_coord_section:
            try:
                _ = [float(p) for p in parts]
                coord_lines.append(parts)
            except ValueError:
                is_coord_section = False
                spectrum_parts.extend(p for p in parts if p)
        else:
            is_coord_section = False
            spectrum_parts.extend(p for p in parts if p)

    if not coord_lines or not spectrum_parts:
        raise ValueError(f"File {file_path} format error: missing coords or spectrum.")

    coords = np.array([list(map(float, line)) for line in coord_lines])
    z = torch.tensor(coords[:, 0], dtype=torch.long)
    pos = torch.tensor(coords[:, 1:], dtype=torch.float32)
    spectrum = [float(val) for val in spectrum_parts]

    return smiles, z, pos, spectrum


def create_mol_with_coords_from_smiles(smiles: str, true_coords: np.ndarray, true_elements: list) -> Mol:
    """通过子结构匹配，从SMILES和原子坐标/类型创建可靠的RDKit分子对象。"""
    try:
        template_mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if template_mol.GetNumAtoms() != len(true_elements):
            return None

        query_mol = Chem.RWMol()
        for elem in true_elements:
            query_mol.AddAtom(Chem.Atom(elem))

        if sorted([a.GetSymbol() for a in template_mol.GetAtoms()]) != sorted(true_elements):
            return None

        match_indices = template_mol.GetSubstructMatch(query_mol.GetMol())
        if not match_indices:
            return None

        final_mol = copy.deepcopy(template_mol)
        conformer = Chem.Conformer(template_mol.GetNumAtoms())
        for query_idx, template_idx in enumerate(match_indices):
            pos = true_coords[query_idx].tolist()
            conformer.SetAtomPosition(template_idx, pos)
        final_mol.RemoveAllConformers()
        final_mol.AddConformer(conformer)
        return final_mol
    except Exception:
        return None


def rdmol_to_data(mol: Mol) -> Data:
    """将RDKit分子对象转换为PyG Data对象。"""
    assert mol.GetNumConformers() == 1
    N = mol.GetNumAtoms()
    pos = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float32)
    atomic_number = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row.extend([start, end])
        col.extend([end, start])
        edge_type.extend([BOND_TYPES.get(bond.GetBondType(), 0)] * 2)

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    if edge_index.numel() > 0:
        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index, edge_type = edge_index[:, perm], edge_type[perm]

    return Data(atom_type=z, pos=pos, edge_index=edge_index, edge_type=edge_type, rdmol=copy.deepcopy(mol))


def main():
    """主执行函数，包含数据处理和数据集分割。"""
    config = {
        "spectra_dirs": {
            "ir": "../../../dataset/qme14s/IR_broaden",
            "raman": "../../../dataset/qme14s/Raman_broaden",
        },
        "target_ir_len": 3500,
        "target_raman_len": 3500,
        "max_molecules": 10000000000,
        "model_edge_order": 3,
        # 【新增】数据集分割配置
        "output_dir": "qme14s_all",  # 保存分割后文件的输出目录
        "split_ratios": {"train": 0.9, "val": 0, "test": 0.1},  # 将此值设为 0 可仅划分训练/测试集
        "seed": 42,  # 用于随机打乱的种子，确保结果可复现
    }

    # 验证比例设置是否合理
    ratios = config["split_ratios"]
    assert (
        abs(ratios["train"] + ratios["val"] + ratios["test"] - 1.0) < 1e-8
    ), "错误：train, val, test 的比例总和必须为 1。"
    assert ratios["train"] > 0 and ratios["test"] > 0, "错误：train 和 test 的比例必须大于 0。"

    print("--- 启动数据集预处理与分割流程 ---")

    # --- 步骤 1: 定义图变换流程 ---
    print(f"\n[步骤 1/3] 定义图变换流程...")
    graph_transforms = Compose([CountNodesPerGraph(), AddHigherOrderEdges(order=config["model_edge_order"])])
    print(" > 流程创建成功。")

    # --- 步骤 2: 处理分子、光谱并应用变换 ---
    print("\n[步骤 2/3] 正在处理分子和光谱数据...")
    final_data_list = []
    ir_files = sorted(glob.glob(os.path.join(config["spectra_dirs"]["ir"], "IR_*.csv")))
    if config["max_molecules"] and len(ir_files) > config["max_molecules"]:
        ir_files = ir_files[: config["max_molecules"]]

    for ir_path in tqdm(ir_files, desc=" > 正在处理分子"):
        file_id = os.path.basename(ir_path).replace("IR_", "").replace(".csv", "")
        raman_path = os.path.join(config["spectra_dirs"]["raman"], f"Raman_{file_id}.csv")
        if not os.path.exists(raman_path):
            continue

        try:
            smiles, z, pos, raw_ir_spec = parse_spectrum_csv_with_smiles(ir_path)
            _, _, _, raw_raman_spec = parse_spectrum_csv_with_smiles(raman_path)

            ir_spec = resample_spectrum(raw_ir_spec, config["target_ir_len"])
            raman_spec = resample_spectrum(raw_raman_spec, config["target_raman_len"])

            elements = [Chem.GetPeriodicTable().GetElementSymbol(int(atom_num)) for atom_num in z]
            mol = create_mol_with_coords_from_smiles(smiles, pos.numpy(), elements)
            if not mol:
                continue

            base_data = rdmol_to_data(mol)
            transformed_data = graph_transforms(base_data)

            transformed_data.smiles = canonicalize_smiles(smiles)
            transformed_data.ir_spectrum = ir_spec
            transformed_data.raman_spectrum = raman_spec
            transformed_data.combined_spectrum = torch.cat([ir_spec, raman_spec], dim=0)
            transformed_data.sample_id = torch.tensor([int(file_id)], dtype=torch.long)

            final_data_list.append(transformed_data)
        except Exception as e:
            # print(f"警告: 处理 {os.path.basename(ir_path)} 失败. 错误: {e}")
            continue

    print(f"✅ 数据处理完成，共获得 {len(final_data_list)} 个有效样本。")

    # --- 步骤 3: 分割数据集并保存 ---
    print(f"\n[步骤 3/3] 分割并保存数据集...")

    # 设置随机种子并打乱数据
    print(f"🌱 使用随机种子: {config['seed']}")
    random.seed(config["seed"])
    random.shuffle(final_data_list)
    print("🔀 数据集已随机打乱。")

    # 计算分割点
    num_total = len(final_data_list)
    num_train = int(num_total * ratios["train"])

    # 根据验证集比例决定分割策略
    if ratios["val"] > 0:
        num_val = int(num_total * ratios["val"])

        train_set = final_data_list[:num_train]
        val_set = final_data_list[num_train : num_train + num_val]
        test_set = final_data_list[num_train + num_val :]

        print(f"✂️ 数据集已分割为三部分:")
        print(f"   - 训练集: {len(train_set)} 个样本")
        print(f"   - 验证集: {len(val_set)} 个样本")
        print(f"   - 测试集: {len(test_set)} 个样本")

        sets_to_save = {"train.pkl": train_set, "val.pkl": val_set, "test.pkl": test_set}
    else:
        # 仅分割为训练集和测试集
        train_set = final_data_list[:num_train]
        test_set = final_data_list[num_train:]

        print(f"✂️ 数据集已分割为两部分 (无验证集):")
        print(f"   - 训练集: {len(train_set)} 个样本")
        print(f"   - 测试集: {len(test_set)} 个样本")

        sets_to_save = {"train.pkl": train_set, "test.pkl": test_set}

    # 创建输出目录并保存文件
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    for filename, dataset in sets_to_save.items():
        output_path = os.path.join(output_dir, filename)
        print(f"💾 正在保存 {filename} 至 '{output_path}'...")
        with open(output_path, "wb") as f:
            pickle.dump(dataset, f)

    print("\n🎉 所有操作完成！")


if __name__ == "__main__":
    main()
    RDLogger.EnableLog("rdApp.*")

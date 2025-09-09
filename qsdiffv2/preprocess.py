import os
import pickle
import copy
import numpy as np
import pandas as pd
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
RDLogger.DisableLog('rdApp.*')

# --- 全局常量定义 ---
BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}

# --- 核心函数 (与版本 1 类似，但现在将与变换流程结合) ---

def rdmol_to_data(mol: Mol, smiles: str = None) -> Data:
    """
    将一个 RDKit 分子对象转换为 PyTorch Geometric Data 对象。
    这个函数是 GeoDiff/ConfGF 数据流水线的基础。

    Args:
        mol (Mol): 包含单个构象的 RDKit 分子对象。
        smiles (str, optional): 分子的SMILES字符串。如果为None，将从mol对象生成。

    Returns:
        Data: 一个 PyTorch Geometric Data 对象，包含原子和键的信息。
    """
    assert mol.GetNumConformers() == 1, "RDKit Mol object must have exactly one conformer."
    N = mol.GetNumAtoms()

    # 1. 获取原子坐标
    pos = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float32)

    # 2. 提取原子特征 (原子序数)
    atomic_number = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    z = torch.tensor(atomic_number, dtype=torch.long)

    # 3. 提取键的特征 (索引和类型)
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # 添加双向边
        row.extend([start, end])
        col.extend([end, start])
        # 添加两次键类型
        edge_type.extend([BOND_TYPES[bond.GetBondType()]] * 2)

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    # 4. 对边进行排序，确保一致性
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    if smiles is None:
        smiles = Chem.MolToSmiles(mol)

    # 5. 创建 Data 对象
    data = Data(
        atom_type=z,
        pos=pos,
        edge_index=edge_index,
        edge_type=edge_type,
        rdmol=copy.deepcopy(mol),
        smiles=smiles
    )
    return data

def process_spectrum_file(file_path: str, target_len: int, max_mols: int = None):
    """
    加载、归一化并重采样光谱数据。
    """
    # ... (此函数内容与上一版完全相同，为简洁起见此处省略) ...
    print(f"  > Processing spectrum file: {os.path.basename(file_path)}")
    nrows = max_mols + 1 if max_mols is not None else None
    df = pd.read_csv(file_path, header=None, nrows=nrows)
    sample_ids = df.iloc[1:, 0].values.astype(int)
    spectra_data = df.iloc[1:, 1:].values.astype(np.float32)
    min_vals = spectra_data.min(axis=1, keepdims=True)
    max_vals = spectra_data.max(axis=1, keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    normalized_spectra = (spectra_data - min_vals) / range_vals
    resampled_spectra = resample(normalized_spectra, target_len, axis=1)
    print(f"  > Finished. Shape after resampling: {resampled_spectra.shape}")
    return sample_ids, resampled_spectra

def create_mol_with_coords_from_smiles(smiles: str, true_coords: np.ndarray, true_elements: list) -> Mol:
    """
    通过子结构匹配，从SMILES和原子坐标/类型创建可靠的RDKit分子对象。
    """
    # ... (此函数内容与上一版完全相同，为简洁起见此处省略) ...
    try:
        template_mol = Chem.MolFromSmiles(smiles)
        if template_mol is None: return None
        template_mol = Chem.AddHs(template_mol)
        if template_mol.GetNumAtoms() != len(true_elements): return None
        query_mol = Chem.RWMol()
        for elem in true_elements:
            query_mol.AddAtom(Chem.Atom(elem))
        template_atoms = sorted([a.GetSymbol() for a in template_mol.GetAtoms()])
        query_atoms = sorted(true_elements)
        if template_atoms != query_atoms: return None
        match_indices = template_mol.GetSubstructMatch(query_mol.GetMol())
        if not match_indices: return None
        conformer = Chem.Conformer(template_mol.GetNumAtoms())
        for query_idx, template_idx in enumerate(match_indices):
            pos = true_coords[query_idx].tolist()
            conformer.SetAtomPosition(template_idx, pos)
        final_mol = copy.deepcopy(template_mol)
        final_mol.RemoveAllConformers()
        final_mol.AddConformer(conformer)
        return final_mol
    except Exception:
        return None

def main():
    """
    主执行函数，完成整个数据预处理流程。
    """
    # --- 配置区 ---
    config = {
        'spectra_files': {
            'uv': '../../dataset/qm9spectra/uv_boraden.csv',
            'ir': '../../dataset/qm9spectra/ir_boraden.csv',
            'raman': '../../dataset/qm9spectra/raman_boraden.csv'
        },
        'molecule_info_dir': '../../dataset/qm9spectra/qm9s_csv',
        'output_pickle_file': 'qm9_spectra_all.pkl', # 新文件名以示区别
        'target_spectra_length': 600,
        'max_molecules': 100000000,
        'model_edge_order': 3, # **重要**: 这个参数必须和 GeoDiff 模型配置文件中的 `edge_order` 保持一致
    }

    print("--- Starting Data Preprocessing for GeoDiff with Spectra (Version 2) ---")

    # --- 关键步骤：定义图变换流程 ---
    # 这个变换流程必须与 GeoDiff 训练和测试时使用的流程相匹配。
    # - CountNodesPerGraph: 添加每个图的节点数字段，用于批处理。
    # - AddHigherOrderEdges: 添加高阶边，这是 GeoDiff 模型的核心要求。
    print(f"\n[Step 1/5] Defining graph transformation pipeline with edge_order={config['model_edge_order']}...")
    graph_transforms = Compose([
        CountNodesPerGraph(),
        AddHigherOrderEdges(order=config['model_edge_order'])
    ])
    print("  > Pipeline created successfully.")

    # 步骤 2: 加载并处理所有光谱数据
    print("\n[Step 2/5] Loading and processing spectra...")
    all_spectra, all_ids = {}, {}
    for spec_type, path in config['spectra_files'].items():
        ids, spectra_data = process_spectrum_file(path, config['target_spectra_length'], config['max_molecules'])
        all_spectra[spec_type] = {int(id_): spec for id_, spec in zip(ids, spectra_data)}
        all_ids[spec_type] = set(ids)

    # 步骤 3: 找到所有光谱类型共有的样本ID
    print("\n[Step 3/5] Finding common sample IDs...")
    common_ids = set.intersection(*all_ids.values())
    sorted_common_ids = sorted(list(common_ids))
    print(f"  > Found {len(sorted_common_ids)} common molecules.")

    # 步骤 4: 迭代处理分子，创建并变换 Data 对象
    print("\n[Step 4/5] Processing molecules, applying transforms, and adding spectra...")
    final_data_list = []
    for sample_id in tqdm(sorted_common_ids, desc="  > Processing molecules"):
        mol_csv_path = os.path.join(config['molecule_info_dir'], f"{(sample_id+1):06d}.csv")
        if not os.path.exists(mol_csv_path): continue

        try:
            mol_df = pd.read_csv(mol_csv_path, header=None)
            smiles = mol_df.iloc[1, 3]
            num_atoms = int(mol_df.iloc[1, 4])
            atom_lines = mol_df.iloc[11:11 + num_atoms].values
            true_elements = [Chem.GetPeriodicTable().GetElementSymbol(int(round(mass))) for mass in atom_lines[:, 1]]
            true_coords = atom_lines[:, 2:5].astype(np.float32)
        except Exception:
            print('error')
            continue
        mol_with_coords = create_mol_with_coords_from_smiles(smiles, true_coords, true_elements)

        if mol_with_coords:
            # a. 创建基础 Data 对象
            base_data = rdmol_to_data(mol_with_coords, smiles)
            
            # b. **应用图变换**
            transformed_data = graph_transforms(base_data)
            
            # c. 添加光谱数据到 *变换后* 的对象
            transformed_data.uv_spectrum = torch.tensor(all_spectra['uv'][sample_id], dtype=torch.float32)
            transformed_data.ir_spectrum = torch.tensor(all_spectra['ir'][sample_id], dtype=torch.float32)
            transformed_data.raman_spectrum = torch.tensor(all_spectra['raman'][sample_id], dtype=torch.float32)
            transformed_data.combined_spectrum = torch.cat([
                transformed_data.uv_spectrum, transformed_data.ir_spectrum, transformed_data.raman_spectrum
            ], dim=0)
            transformed_data.sample_id = torch.tensor([sample_id], dtype=torch.long)
            
            final_data_list.append(transformed_data)

    # 步骤 5: 保存最终的数据集
    print(f"\n[Step 5/5] Saving final dataset...")
    with open(config['output_pickle_file'], 'wb') as f:
        pickle.dump(final_data_list, f)

    print(f"\n✅ Preprocessing complete!")
    print(f"  > Successfully processed and saved {len(final_data_list)} molecules.")
    print(f"  > Fully processed dataset saved to: {config['output_pickle_file']}")
    
    if final_data_list:
        print("\n--- Sample of Final Processed Data Object ---")
        sample_data = final_data_list[0]
        print(sample_data)
        print("\nKey attributes added by transforms:")
        print(f" - Edge index shape: {sample_data.edge_index.shape}")
        print(f" - Has 'edge_order': {'edge_order' in sample_data}")
        print(f" - Has 'num_nodes_per_graph': {'num_nodes_per_graph' in sample_data}")
        print("---------------------------------------------")

if __name__ == '__main__':
    main()
    RDLogger.EnableLog('rdApp.*')
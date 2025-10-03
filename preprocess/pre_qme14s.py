import os
import pickle
import copy
import glob
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
RDLogger.DisableLog('rdApp.*')

# --- 全局常量定义 ---
BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}

# --- 辅助函数 ---

def normalize_and_resample(spectrum, target_len):
    """对光谱进行归一化和重采样。"""
    spectrum_np = np.array(spectrum, dtype=np.float32)
    min_val = spectrum_np.min()
    max_val = spectrum_np.max()
    range_val = max_val - min_val
    if range_val > 1e-8:
        spectrum_np = (spectrum_np - min_val) / range_val
    else:
        spectrum_np = np.zeros_like(spectrum_np)
        
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
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    smiles = lines[0]
    
    coord_lines = []
    spectrum_parts = []
    
    is_coord_section = True
    for line in lines[1:]:
        parts = line.split(',')
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
    """
    通过子结构匹配，从SMILES和原子坐标/类型创建可靠的RDKit分子对象。
    """
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
        row.extend([start, end]); col.extend([end, start])
        edge_type.extend([BOND_TYPES.get(bond.GetBondType(), 0)] * 2)

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    
    if edge_index.numel() > 0:
        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index, edge_type = edge_index[:, perm], edge_type[perm]
    
    # 【修改】将 `z` 重命名为 `atom_type` 以匹配 script_2 的 `rdmol_to_data` 函数输出
    data = Data(atom_type=z, pos=pos, edge_index=edge_index, edge_type=edge_type, rdmol=copy.deepcopy(mol))
    return data

def main():
    """主执行函数。"""
    config = {
        'spectra_dirs': {
            'ir': '../../../dataset/qme14s/IR_broaden',
            'raman': '../../../dataset/qme14s/Raman_broaden',
        },
        'output_pickle_file': 'qme14s_all.pkl', 
        'target_spectra_length': 600,
        'max_molecules': 10000000,
        'model_edge_order': 3,
    }

    print("--- Starting Dataset Preprocessing (Compatible with Model 2) ---")

    # --- 步骤 1: 定义图变换流程 ---
    print(f"\n[Step 1/3] Defining graph transformation pipeline...")
    graph_transforms = Compose([CountNodesPerGraph(), AddHigherOrderEdges(order=config['model_edge_order'])])
    print(" > Pipeline created successfully.")

    # --- 步骤 2: 迭代主光谱（IR），匹配并处理数据 ---
    print("\n[Step 2/3] Processing molecules, spectra, and applying transforms...")
    final_data_list = []
    ir_files = sorted(glob.glob(os.path.join(config['spectra_dirs']['ir'], 'IR_*.csv')))
    if config['max_molecules'] and len(ir_files) > config['max_molecules']:
        ir_files = ir_files[:config['max_molecules']]

    for ir_path in tqdm(ir_files, desc=" > Processing molecules"):
        file_id = os.path.basename(ir_path).replace('IR_', '').replace('.csv', '')
        raman_path = os.path.join(config['spectra_dirs']['raman'], f'Raman_{file_id}.csv')
        if not os.path.exists(raman_path):
            print(f'Warning: Raman file for ID {file_id} not found. Skipping.')
            continue

        try:
            # a. 解析文件
            smiles, z, pos, raw_ir_spec = parse_spectrum_csv_with_smiles(ir_path)
            _, _, _, raw_raman_spec = parse_spectrum_csv_with_smiles(raman_path)

            # b. 【修改】处理光谱，以匹配 script_2 的结构
            # 分别处理 IR 和 Raman 光谱至目标长度 600
            ir_spec = normalize_and_resample(raw_ir_spec, config['target_spectra_length'])
            raman_spec = normalize_and_resample(raw_raman_spec, config['target_spectra_length'])
            
            # 因为没有 UV 光谱数据，我们创建一个全零的占位符张量
            uv_spec = torch.zeros(config['target_spectra_length'], dtype=torch.float32)

            # c. 创建 RDKit Mol 和 PyG Data 对象
            elements = [Chem.GetPeriodicTable().GetElementSymbol(int(atom_num)) for atom_num in z]
            mol = create_mol_with_coords_from_smiles(smiles, pos.numpy(), elements)
            if not mol: 
                print(f'Warning: Failed to create RDKit molecule for ID {file_id}. Skipping.')
                continue
            
            base_data = rdmol_to_data(mol)
            
            # d. 应用图变换
            transformed_data = graph_transforms(base_data)

            # e. 【修改】将光谱数据以 script_2 的键名添加到 Data 对象
            transformed_data.smiles = canonicalize_smiles(smiles)
            
            # 添加三个独立的光谱
            transformed_data.uv_spectrum = uv_spec
            transformed_data.ir_spectrum = ir_spec
            transformed_data.raman_spectrum = raman_spec

            # 按照 script_2 的顺序合并光谱
            transformed_data.combined_spectrum = torch.cat([
                transformed_data.uv_spectrum, 
                transformed_data.ir_spectrum, 
                transformed_data.raman_spectrum
            ], dim=0)
            
            # 添加 sample_id
            sample_id = int(file_id)
            transformed_data.sample_id = torch.tensor([sample_id], dtype=torch.long)
            
            final_data_list.append(transformed_data)
        except Exception as e:
            print(f"Warning: Failed to process {os.path.basename(ir_path)}. Error: {e}")
            continue

    # --- 步骤 3: 保存最终的数据集 ---
    print(f"\n[Step 3/3] Saving final dataset...")
    with open(config['output_pickle_file'], 'wb') as f:
        pickle.dump(final_data_list, f)

    print(f"\n✅ Preprocessing complete!")
    print(f" > Successfully processed and saved {len(final_data_list)} molecules.")
    print(f" > Dataset saved to: {config['output_pickle_file']}")
    
    if final_data_list:
        print("\n--- Sample of Final Processed Data Object (Compatible Format) ---")
        sample_data = final_data_list[0]
        print(sample_data)
        print("\nSpectrum shapes:")
        # 【修改】更新打印信息以反映新的键名
        print(f" - uv_spectrum: {sample_data.uv_spectrum.shape}")
        print(f" - ir_spectrum: {sample_data.ir_spectrum.shape}")
        print(f" - raman_spectrum: {sample_data.raman_spectrum.shape}")
        print(f" - combined_spectrum: {sample_data.combined_spectrum.shape}")
        print(f" - sample_id: {sample_data.sample_id}")
        print("---------------------------------------------------------------")

if __name__ == '__main__':
    main()
    RDLogger.EnableLog('rdApp.*')
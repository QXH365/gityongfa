import os
import numpy as np
import pandas as pd
from scipy.signal import resample
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm
import torch
from torch_geometric.transforms import Compose
import pickle
import copy

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# 导入ConfGF的工具函数
try:
    from confgf import utils
    from dataset import rdmol_to_data, smiles_to_data
    CONFGF_AVAILABLE = True
    print("ConfGF modules loaded successfully.")
except ImportError:
    print("Error: ConfGF not found. Please ensure ConfGF is installed and in Python path.")
    exit(1)

# --- 配置参数 ---
SPECTRA_FILES = {
    'uv': '../dataset/qm9spectra/uv_boraden.csv',
    'ir': '../dataset/qm9spectra/ir_boraden.csv', 
    'raman': '../dataset/qm9spectra/raman_boraden.csv'
}

MOLECULE_INFO_DIR = '../dataset/qm9spectra/qm9s_csv'
OUTPUT_PICKLE_FILE = 'qm9s_for_confgf_generation_all.pkl'
TARGET_SPECTRA_LENGTH = 600
MAX_MOLECULES_TO_PROCESS = 10000000

# 原子质量到元素符号的精确映射
ATOMIC_MASSES = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl',
    18: 'Ar', 19: 'K', 20: 'Ca', 35: 'Br', 53: 'I',
}

def mass_to_element(mass, tolerance=0.5):
    """根据原子质量确定元素符号"""
    for ref_mass, element in ATOMIC_MASSES.items():
        if abs(mass - ref_mass) < tolerance:
            return element
    print(f"Warning: Unknown atomic mass {mass}")
    return None

def process_spectrum(file_path, target_len, max_rows=None):
    """加载光谱数据,进行归一化和插值处理。"""
    print(f"  > Loading and processing {os.path.basename(file_path)}...")
    df = pd.read_csv(file_path, header=None, nrows=max_rows)
    sample_ids = df.iloc[1:, 0].values.astype(int)
    spectra_data = df.iloc[1:, 1:].values
    min_vals = spectra_data.min(axis=1, keepdims=True)
    max_vals = spectra_data.max(axis=1, keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    normalized_spectra = (spectra_data - min_vals) / range_vals
    resampled_spectra = resample(normalized_spectra, target_len, axis=1)
    print(f"  > Finished. Shape after processing: {resampled_spectra.shape}")
    return sample_ids, resampled_spectra

def parse_molecular_csv(file_path):
    """
    解析QM9S分子CSV文件,提取SMILES和以原子索引为键的原子信息字典。
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        line0_parts = lines[1].strip().split(',')
        smiles = line0_parts[3]
        num_atoms = int(float(line0_parts[4]))
        
        pos_lines = lines[11:11+num_atoms]
        
        # 使用字典存储原子信息,键为CSV中的原子索引
        atoms_data = {}
        
        for line in pos_lines:
            parts = line.strip().split(',')
            if len(parts) < 5: continue
            
            atom_index = int(parts[0])
            atom_mass = float(parts[1])
            element = mass_to_element(atom_mass)
            if element is None: return None, None
            
            position = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float32)
            atoms_data[atom_index] = {'element': element, 'pos': position}
            
        return smiles, atoms_data
        
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None, None

def create_molecule_from_atoms_and_coords(smiles, atoms_data):
    """
    【已二次修正】根据SMILES和从CSV解析的原子数据创建RDKit分子对象。
    该方法能够正确处理原子顺序不匹配的问题。
    """
    try:
        # 1. 从SMILES创建模板分子,包含正确的化学键
        template_mol = Chem.MolFromSmiles(smiles)
        if template_mol is None:
            print(f"Error: RDKit could not parse SMILES: {smiles}")
            return None, None
        template_mol = Chem.AddHs(template_mol)
        
        # 2. 从CSV数据动态创建一个"查询"分子,它没有键,但有原子类型
        query_mol = Chem.RWMol()
        csv_indices = sorted(atoms_data.keys())
        for idx in csv_indices:
            atom = Chem.Atom(atoms_data[idx]['element'])
            query_mol.AddAtom(atom)

        # 3. 检查原子构成是否一致
        template_atoms = sorted([a.GetSymbol() for a in template_mol.GetAtoms()])
        query_atoms = sorted([atoms_data[idx]['element'] for idx in csv_indices])
        if template_atoms != query_atoms:
            print("Error: Atom composition mismatch between SMILES and CSV data.")
            print(f"  From SMILES: {template_atoms}")
            print(f"  From CSV:    {query_atoms}")
            return None, None

        # 4. 关键步骤: 找到模板分子和查询分子之间的原子映射关系
        match = template_mol.GetSubstructMatch(query_mol.GetMol())
        
        if not match:
            print("Error: Could not find atom mapping (substructure match) between SMILES and CSV atoms.")
            return None, None
            
        # 5. 创建构象并根据映射关系设置正确的坐标
        conf = Chem.Conformer(template_mol.GetNumAtoms())
        for query_idx, template_idx in enumerate(match):
            original_csv_index = csv_indices[query_idx]
            pos = atoms_data[original_csv_index]['pos']
            
            # --- 修正点 ---
            # 将numpy数组转换为python列表,以满足RDKit函数的要求
            conf.SetAtomPosition(template_idx, pos.tolist())
            # --- 修正结束 ---

        # 6. 将构象添加到模板分子中,完成最终分子的构建
        final_mol = copy.deepcopy(template_mol)
        final_mol.RemoveAllConformers()
        final_mol.AddConformer(conf)
        
        # 7. 转换为ConfGF数据格式并应用变换
        data_with_real_coords = rdmol_to_data(final_mol, smiles=smiles)
        data_for_generation = smiles_to_data(smiles)
        
        if data_with_real_coords is None or data_for_generation is None:
            print("Error: Failed to convert to ConfGF data format")
            return None, None
        
        transform = Compose([
            utils.AddHigherOrderEdges(order=3),
            utils.AddEdgeLength(),
            utils.AddPlaceHolder(),
        ])
        
        data_with_real_coords = transform(data_with_real_coords)
        data_for_generation = transform(data_for_generation)
        
        print("Successfully created molecule data using robust mapping.")
        return data_with_real_coords, data_for_generation
        
    except Exception as e:
        print(f"Error creating molecule: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def get_molecular_data(info_dir, sample_ids):
    """从文件夹中读取分子信息，创建用于ConfGF生成任务的数据。"""
    print(f"\nProcessing molecular information from '{info_dir}'...")
    data_list = []
    valid_indices = []
    RDLogger.DisableLog('rdApp.*')
    
    for idx, sample_id in enumerate(tqdm(sample_ids, desc="  > Processing molecules")):
        print(f"\n--- Processing sample {sample_id} (index {idx}) ---")
        file_name = f"{(sample_id+1):06d}.csv"
        file_path = os.path.join(info_dir, file_name)
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        smiles, atoms_data = parse_molecular_csv(file_path)
        
        if smiles is None or atoms_data is None:
            print(f"Failed to parse molecule data from {file_path}")
            continue
        
        data_real, data_gen = create_molecule_from_atoms_and_coords(smiles, atoms_data)
        
        if data_real is not None and data_gen is not None:
            data_gen.pos_ref = data_real.pos.clone()
            data_gen.num_pos_ref = torch.tensor([1], dtype=torch.long)
            data_gen.smiles = smiles
            
            # For debugging, we can store original parsed data
            original_coords = torch.tensor([atoms_data[k]['pos'] for k in sorted(atoms_data.keys())], dtype=torch.float32)
            data_gen.coordinates_original = original_coords

            data_list.append(data_gen)
            valid_indices.append(idx)
            print(f"Successfully processed molecule {sample_id}")
        else:
            print(f"Failed to create molecule data for sample {sample_id}")
    
    RDLogger.EnableLog('rdApp.*')
    print(f"\nSuccessfully processed {len(data_list)} molecules out of {len(sample_ids)}")
    return data_list, valid_indices

def main():
    """主执行函数"""
    print("--- Starting QM9S Data Preprocessing for Spectrum-to-Conformation Generation (Corrected) ---")
    if MAX_MOLECULES_TO_PROCESS is not None:
        print(f"\n--- Processing limit: {MAX_MOLECULES_TO_PROCESS} molecules ---")

    # 1. 处理光谱数据
    print("\n[Step 1/4] Processing spectral data files...")
    all_spectra_data, all_sample_ids = {}, {}
    for name, path in SPECTRA_FILES.items():
        if not os.path.exists(path):
            print(f"Error: Spectrum file not found at '{path}'. Aborting.")
            return
        sample_ids, spectra = process_spectrum(path, TARGET_SPECTRA_LENGTH, max_rows=MAX_MOLECULES_TO_PROCESS + 1 if MAX_MOLECULES_TO_PROCESS else None)
        all_spectra_data[name] = spectra
        all_sample_ids[name] = sample_ids

    # 2. 验证样本序号一致性
    print("\n[Step 2/4] Verifying sample IDs consistency...")
    base_ids = next(iter(all_sample_ids.values()))
    for name, ids in all_sample_ids.items():
        if not np.array_equal(base_ids, ids):
            print(f"Error: Sample IDs in '{SPECTRA_FILES[name]}' do not match. Aborting.")
            return
    print("  > Sample IDs are consistent across all files.")

    # 3. 处理分子结构和真实构象
    print("\n[Step 3/4] Processing molecular structures with reference conformations...")
    if not os.path.isdir(MOLECULE_INFO_DIR):
        print(f"Error: Molecule info directory not found at '{MOLECULE_INFO_DIR}'. Aborting.")
        return
    mol_data_list, valid_indices = get_molecular_data(MOLECULE_INFO_DIR, base_ids)

    if not mol_data_list:
        print("Error: No valid molecules processed. Check data format and paths.")
        return

    # 4. 组合数据
    print(f"\n[Step 4/4] Preparing final dataset...")
    final_data_list = []
    for i, mol_data in enumerate(mol_data_list):
        original_idx = valid_indices[i]
        mol_data.uv_spectrum = torch.tensor(all_spectra_data['uv'][original_idx], dtype=torch.float32)
        mol_data.ir_spectrum = torch.tensor(all_spectra_data['ir'][original_idx], dtype=torch.float32)
        mol_data.raman_spectrum = torch.tensor(all_spectra_data['raman'][original_idx], dtype=torch.float32)
        mol_data.combined_spectrum = torch.cat([mol_data.uv_spectrum, mol_data.ir_spectrum, mol_data.raman_spectrum], dim=0)
        mol_data.sample_id = torch.tensor([base_ids[original_idx]], dtype=torch.long)
        final_data_list.append(mol_data)
    
    # 保存
    print(f"\nSaving {len(final_data_list)} processed molecules to '{OUTPUT_PICKLE_FILE}'...")
    with open(OUTPUT_PICKLE_FILE, 'wb') as f:
        pickle.dump(final_data_list, f)

    print(f"\n✅ Successfully processed {len(final_data_list)} molecules!")
    print(f"Data saved to: {OUTPUT_PICKLE_FILE}")
    
    if final_data_list:
        sample_data = final_data_list[0]
        print("\nDataset Statistics (sample):")
        print(f"- Atoms per molecule: {sample_data.num_nodes}")
        print(f"- Edges per molecule (including high-order): {sample_data.num_edges}")

if __name__ == '__main__':
    main()
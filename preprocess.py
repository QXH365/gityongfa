# preprocess_mask_generator_data.py

import os
import pickle
import copy
import json
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import argparse

import torch
from torch_geometric.data import Data, Batch

# from torch_geometric.transforms import Compose # Compose might not be needed now

import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Chem.rdchem import Mol, BondType, Conformer  # Import Conformer
from rdkit import RDLogger

# 禁用 RDKit 的冗余日志
RDLogger.DisableLog("rdApp.*")

# --- 全局常量定义 ---
BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}

# --- 核心函数 ---


def get_graph_structure_from_mol(mol: Mol) -> dict:
    """从 RDKit Mol 对象中提取图结构信息 (不含坐标)。"""
    N = mol.GetNumAtoms()
    atomic_number = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row.extend([start, end])
        col.extend([end, start])
        edge_type.extend([BOND_TYPES.get(bond.GetBondType(), 0)] * 2)

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)

    if edge_index.numel() > 0:
        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index, edge_type_tensor = edge_index[:, perm], edge_type_tensor[perm]

    return {"atom_type": z, "edge_index": edge_index, "edge_type": edge_type_tensor, "num_nodes": N}


def canonicalize_smiles(smiles: str):
    """将SMILES字符串转换为唯一的规范形式。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # AddHs might be needed before canonicalization if Hs are important
        # mol = Chem.AddHs(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    return None


# --- 修改后的 align_conformers 函数 ---
def align_conformers(mols):
    """
    Aligns conformers from a list of RDKit Mol objects.
    Assumes all mols in the list represent the same molecule with one conformer each.
    Returns a list of aligned positions (numpy arrays).
    """
    if not mols:
        return []
    if len(mols) < 2:
        return [m.GetConformer(0).GetPositions() for m in mols]

    # 创建一个新的分子对象，并将所有构象添加到其中
    mol_with_all_confs = copy.deepcopy(mols[0])  # 从第一个分子复制基础结构
    mol_with_all_confs.RemoveAllConformers()  # 移除自带的构象

    conf_ids = []
    for i, m in enumerate(mols):
        if m.GetNumConformers() > 0:
            conf = m.GetConformer(0)
            conf.SetId(i)  # 设置唯一的构象ID
            conf_id = mol_with_all_confs.AddConformer(conf, assignId=True)  # 添加构象
            conf_ids.append(conf_id)
        else:
            # print(f"Warning: Mol at index {i} has no conformers.") # 可选：添加警告
            pass  # 跳过没有构象的分子

    if len(conf_ids) < 2:  # 如果添加后有效构象少于2个，则无法对齐
        # print("Warning: Less than 2 valid conformers found for alignment.") # 可选
        # 返回未对齐的坐标作为后备
        return [mol_with_all_confs.GetConformer(cid).GetPositions() for cid in conf_ids]

    # 使用包含所有构象的单个分子对象进行对齐
    rmsd_list = []
    try:
        # 使用 confIds 参数指定要对齐哪些构象 (虽然默认可能就是全部)
        rdMolAlign.AlignMolConformers(mol_with_all_confs, confIds=conf_ids, RMSlist=rmsd_list)
        # print(f"Alignment RMSD for {len(conf_ids)} confs: {rmsd_list}") # Optional
    except RuntimeError as e:
        # 捕获可能的对齐错误 (例如原子数不匹配等极端情况)
        print(f"RuntimeError during AlignMolConformers: {e}. Returning unaligned positions.")
        # 返回未对齐的坐标
        return [mol_with_all_confs.GetConformer(cid).GetPositions() for cid in conf_ids]

    # 从对齐后的分子对象中提取坐标
    aligned_positions = [mol_with_all_confs.GetConformer(cid).GetPositions() for cid in conf_ids]
    return aligned_positions


# --- align_conformers 函数修改结束 ---


def main(config):
    """主执行函数。"""
    print("--- Starting GEOM Dataset Preprocessing for Mask Generator ---")

    # 设置随机种子
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    # 创建输出目录
    os.makedirs(config["output_dir"], exist_ok=True)
    print(f" > Output directory: {config['output_dir']}")

    # --- 步骤 1: 从 GEOM summary 文件中筛选分子 ---
    print("\n[Step 1/4] Filtering molecules from GEOM summary file...")
    summary_path = os.path.join(config["base_path"], f"summary_{config['dataset_name']}.json")
    try:
        with open(summary_path, "r") as f:
            summ = json.load(f)
    except FileNotFoundError:
        print(f"Error: Summary file not found at {summary_path}")
        return

    pickle_path_list = []
    smiles_map = {}  # Store canonical smiles to original path for reference
    num_skipped_low_confs = 0
    for smiles_orig, meta_mol in tqdm(summ.items(), desc=" > Scanning summary"):
        # We need at least 2 conformers to calculate variance
        if meta_mol.get("uniqueconfs", 0) >= 2:
            pickle_path = meta_mol.get("pickle_path")
            if pickle_path:
                canonical_smi = canonicalize_smiles(smiles_orig)
                if canonical_smi:  # Ensure SMILES is valid
                    # Avoid duplicate canonical SMILES if summary contains variations
                    if canonical_smi not in smiles_map:
                        pickle_path_list.append(pickle_path)
                        smiles_map[canonical_smi] = pickle_path
        else:
            num_skipped_low_confs += 1

    print(f" > Found {len(pickle_path_list)} unique molecules with >= 2 conformers.")
    print(f" > Skipped {num_skipped_low_confs} molecules with < 2 conformers.")

    random.shuffle(pickle_path_list)
    if config["tot_mol_size"] <= 0 or len(pickle_path_list) < config["tot_mol_size"]:
        if config["tot_mol_size"] > 0:  # Only print warning if a positive size was requested
            print(
                f"Warning: Only {len(pickle_path_list)} valid unique molecules found, less than requested {config['tot_mol_size']}. Using all available molecules."
            )
        else:
            print(f" > Using all {len(pickle_path_list)} available unique molecules.")
        config["tot_mol_size"] = len(pickle_path_list)

    pickle_path_list = pickle_path_list[: config["tot_mol_size"]]
    print(f" > Selected {len(pickle_path_list)} unique molecules for processing based on config.")

    # --- 步骤 2: 处理分子，计算灵活性标签 ---
    print("\n[Step 2/4] Processing molecules and calculating flexibility scores...")
    all_processed_data = []
    bad_case_count = 0
    skipped_alignment_fail = 0

    for pkl_path in tqdm(pickle_path_list, desc=" > Processing molecules"):
        try:
            full_pkl_path = os.path.join(config["base_path"], pkl_path)
            with open(full_pkl_path, "rb") as f:
                mol_info = pickle.load(f)

            conformers_info = mol_info.get("conformers", [])
            if len(conformers_info) < 2:  # Should have been filtered, but double check
                bad_case_count += 1
                continue

            # Select conformers for variance calculation
            confs_to_use_info = conformers_info[: config["max_confs_for_variance"]]
            # --- 修复：确保传入的是 RDKit Mol 对象 ---
            rdkit_mols = []
            for c_info in confs_to_use_info:
                mol = c_info.get("rd_mol")
                # 检查 mol 是否有效并且包含至少一个构象
                if mol and isinstance(mol, Mol) and mol.GetNumConformers() > 0:
                    rdkit_mols.append(mol)
                # else: # 可选：添加警告或调试信息
                #     print(f"Skipping invalid conformer in {pkl_path}")

            if len(rdkit_mols) < 2:  # Need at least two valid RDKit mols with conformers
                # print(f"Warning: Less than 2 valid RDKit mols with conformers found in {pkl_path}")
                bad_case_count += 1
                continue
            # --- 修复结束 ---

            # Align conformers - 现在传入的是 rdkit_mols 列表
            try:
                # align_conformers 现在处理列表，并返回对齐后的 numpy 坐标列表
                aligned_positions_np = align_conformers(rdkit_mols)
                if len(aligned_positions_np) < 2:  # Check if alignment returned less than 2 valid results
                    skipped_alignment_fail += 1
                    continue
                aligned_positions = [torch.tensor(p, dtype=torch.float32) for p in aligned_positions_np]
            except Exception as align_error:
                # print(f"Warning: Alignment failed for {pkl_path}. Error: {align_error}")
                skipped_alignment_fail += 1
                continue

            # Calculate standard deviation
            pos_stack = torch.stack(aligned_positions, dim=0)  # (NumConfs, N, 3)
            # --- 修复：检查原子数量是否一致 ---
            if pos_stack.shape[1] != rdkit_mols[0].GetNumAtoms():
                print(f"Warning: Atom count mismatch after alignment for {pkl_path}. Skipping.")
                bad_case_count += 1
                continue
            # --- 修复结束 ---
            pos_std = torch.std(pos_stack, dim=0)  # (N, 3)
            atom_std = torch.norm(pos_std, dim=1)  # (N,)

            # Normalize standard deviation to get flexibility score (0=stable, 1=flexible)
            min_std, max_std = atom_std.min(), atom_std.max()
            if max_std - min_std < 1e-6:  # Avoid division by zero if all atoms have same std dev
                atom_flexibility_score = torch.zeros_like(atom_std)
            else:
                atom_flexibility_score = (atom_std - min_std) / (max_std - min_std)

            # Get graph structure from the first conformer
            graph_structure = get_graph_structure_from_mol(rdkit_mols[0])
            canonical_smi = canonicalize_smiles(mol_info.get("smiles"))

            data = Data(
                atom_type=graph_structure["atom_type"],
                edge_index=graph_structure["edge_index"],
                edge_type=graph_structure["edge_type"],
                pos=aligned_positions[0],  # <-- 添加这一行
                atom_flexibility_score=atom_flexibility_score.unsqueeze(-1),  # Shape (N, 1)
                smiles=canonical_smi,
                num_nodes=graph_structure["num_nodes"],
            )

            all_processed_data.append(data)

        except FileNotFoundError:
            print(f"Warning: Pickle file not found: {full_pkl_path}. Skipping.")
            bad_case_count += 1
        except Exception as e:
            print(f"Warning: Failed to process {pkl_path}. Error: {e}")
            bad_case_count += 1

    print(f" > Finished processing. Successfully processed {len(all_processed_data)} molecules.")
    print(f" > Skipped {bad_case_count} molecules due to loading/processing errors.")
    print(f" > Skipped {skipped_alignment_fail} molecules due to alignment failures.")

    # --- 步骤 3: 划分数据集 ---
    print("\n[Step 3/4] Splitting dataset into training and testing sets (90:10)...")
    random.shuffle(all_processed_data)  # Shuffle molecules
    num_total = len(all_processed_data)
    num_train = int(num_total * 0.9)

    train_data = all_processed_data[:num_train]
    test_data = all_processed_data[num_train:]

    print(f" > Training set size: {len(train_data)} molecules")
    print(f" > Test set size:     {len(test_data)} molecules")

    # --- 步骤 4: 保存最终的数据集和示例文件 ---
    print("\n[Step 4/4] Saving final datasets and example file...")

    train_output_path = os.path.join(config["output_dir"], "train_mask_gen.pkl")
    test_output_path = os.path.join(config["output_dir"], "test_mask_gen.pkl")
    example_output_path = os.path.join(config["output_dir"], "preprocessing_example.txt")

    try:
        with open(train_output_path, "wb") as f:
            pickle.dump(train_data, f)
        print(f" > Training data saved to: {train_output_path}")

        with open(test_output_path, "wb") as f:
            pickle.dump(test_data, f)
        print(f" > Test data saved to:   {test_output_path}")

        # Generate example output file
        with open(example_output_path, "w") as f:
            f.write("--- Example Processed Data for Mask Generator ---\n\n")
            num_examples = min(2, len(train_data) + len(test_data))
            examples_to_show = (train_data + test_data)[:num_examples]

            for i, data_example in enumerate(examples_to_show):
                f.write(f"--- Sample {i+1} ---\n")
                f.write(f"SMILES: {data_example.smiles}\n")
                f.write(f"Number of atoms: {data_example.num_nodes}\n")
                f.write(f"Atom Types (Tensor): {data_example.atom_type.tolist()}\n")
                # Limit printing edge_index if it's too large
                if data_example.edge_index.shape[1] < 50:
                    f.write(
                        f"Edge Index (Tensor shape {list(data_example.edge_index.shape)}):\n{data_example.edge_index.tolist()}\n"
                    )
                else:
                    f.write(f"Edge Index (Tensor shape {list(data_example.edge_index.shape)}): [Too large to print]\n")
                if data_example.edge_type.shape[0] < 50:
                    f.write(
                        f"Edge Type (Tensor shape {list(data_example.edge_type.shape)}):\n{data_example.edge_type.tolist()}\n"
                    )
                else:
                    f.write(f"Edge Type (Tensor shape {list(data_example.edge_type.shape)}): [Too large to print]\n")

                f.write(f"Atom Flexibility Score (Tensor shape {list(data_example.atom_flexibility_score.shape)}):\n")
                # Print scores with atom index
                for atom_idx, score in enumerate(data_example.atom_flexibility_score.squeeze().tolist()):
                    f.write(f"  Atom {atom_idx}: {score:.4f}\n")
                f.write("\n")
        print(f" > Example output saved to: {example_output_path}")

    except Exception as save_e:
        print(f"Error during saving datasets: {save_e}")

    print("\n✅ Preprocessing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess GEOM dataset for Mask Generator Training")
    parser.add_argument(
        "--base_path",
        type=str,
        default="../../../ConfGF/GEOM/rdkit_folder",
        help="Path to the GEOM dataset root directory containing summary JSON and pickle files.",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="drugs", choices=["drugs", "qm9"], help="Name of the GEOM dataset subset."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./geom_mask_data_100000", help="Directory to save the processed data."
    )
    parser.add_argument(
        "--tot_mol_size",
        type=int,
        default=100000,
        help="Total number of unique molecules to process (<=0 to use all available).",
    )
    parser.add_argument(
        "--max_confs_for_variance",
        type=int,
        default=50,
        help="Maximum number of conformers per molecule to use for calculating variance.",
    )
    parser.add_argument("--seed", type=int, default=2021, help="Random seed for shuffling and splitting.")
    # Add other arguments if needed, like train/test split ratio if you want it configurable

    args = parser.parse_args()

    config_dict = {
        "base_path": args.base_path,
        "dataset_name": args.dataset_name,
        "output_dir": args.output_dir,
        "max_confs_for_variance": args.max_confs_for_variance,
        "tot_mol_size": args.tot_mol_size,
        "seed": args.seed,
        # 'model_edge_order': 3, # This is not needed for mask generator input data
        # 'train_size': 0.9, # Hardcoded 9:1 split as requested
    }

    main(config_dict)
    RDLogger.EnableLog("rdApp.*")  # Re-enable logging if needed elsewhere

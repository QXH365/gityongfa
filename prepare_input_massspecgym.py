import os
import json
import click
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from scipy.interpolate import CubicSpline
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path


class DataPreparationV2:
    """
    为图扩散模型准备质谱和分子数据（支持虚拟原子*的骨架定义）。

    该类负责将SMILES字符串转换为图结构，将质谱数据处理为固定长度的向量，
    并生成基于虚拟原子*的精确骨架掩码。
    """

    def __init__(self, save_dir):
        """
        初始化数据准备类。

        Args:
            save_dir (str or Path): 保存处理后数据的文件夹路径。
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.atom_dict = {}
        self.bond_dict = {0: 0}  # 0 代表无键

        self.atom_counter = 0
        self.bond_counter = 1  # 从1开始，因为0已经表示无键

    def make_msms_spectrum(self, spectrum_peaks):
        """将MSMS光谱转换为固定长度向量（应用高斯扩散）。"""
        msms_spectrum = np.zeros(6000)
        sigma = 1.0
        for mz, intensity in spectrum_peaks:
            peak_pos = mz * 3
            x_min = max(0, int(np.floor(peak_pos - 3 * sigma)))
            x_max = min(5999, int(np.ceil(peak_pos + 3 * sigma)))
            if x_min >= x_max:
                continue

            x_values = np.arange(x_min, x_max + 1)
            distances = x_values - peak_pos
            weights = np.exp(-(distances**2) / (2 * sigma**2))

            # 累加到光谱数组
            msms_spectrum[x_values] += intensity * weights

        return msms_spectrum

    def interpolate_msms_to_600(self, msms_spectrum):
        """将6000点的MSMS光谱插值到600点（使用三次样条插值）。"""
        old_x = np.arange(len(msms_spectrum))

        if len(old_x) < 4:
            return np.zeros(600)

        cs = CubicSpline(old_x, msms_spectrum)
        new_x = np.linspace(old_x.min(), old_x.max(), 600)
        interpolated = cs(new_x)

        interpolated = np.clip(interpolated, 0, None)
        norm = np.linalg.norm(interpolated)
        return interpolated / norm if norm > 0 else interpolated

    def smiles_to_graph(self, smiles, update_dicts=True):
        """
        将SMILES字符串转换为节点特征和邻接矩阵。

        Args:
            smiles (str): 分子的SMILES表示。
            update_dicts (bool): 是否更新原子和键的字典。

        Returns:
            Tuple[np.array, np.array] or Tuple[None, None]: 节点特征和邻接矩阵。
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None

        num_atoms = mol.GetNumAtoms()
        node_features = np.zeros(num_atoms, dtype=np.int32)
        adjacency_matrix = np.zeros((num_atoms, num_atoms), dtype=np.int32)

        for i, atom in enumerate(mol.GetAtoms()):
            atom_symbol = atom.GetSymbol()
            if update_dicts and atom_symbol not in self.atom_dict:
                self.atom_dict[atom_symbol] = self.atom_counter
                self.atom_counter += 1

            if atom_symbol not in self.atom_dict:
                return None, None
            node_features[i] = self.atom_dict[atom_symbol]

        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type_str = str(bond.GetBondType())

            if update_dicts and bond_type_str not in self.bond_dict:
                self.bond_dict[bond_type_str] = self.bond_counter
                self.bond_counter += 1

            if bond_type_str not in self.bond_dict:
                return None, None

            bond_val = self.bond_dict[bond_type_str]
            adjacency_matrix[i, j] = bond_val
            adjacency_matrix[j, i] = bond_val

        return node_features, adjacency_matrix

    def parse_scaffold_with_attachment_points(self, scaffold_smiles):
        """
        解析包含虚拟原子*的骨架SMILES，返回实际骨架和连接点信息。
        
        Args:
            scaffold_smiles (str): 包含*的骨架SMILES
            
        Returns:
            Tuple[Chem.Mol, set]: 实际骨架分子对象和连接点原子索引集合
        """
        mol = Chem.MolFromSmiles(scaffold_smiles)
        if mol is None:
            return None, set()
        
        # 找到所有虚拟原子*的索引
        attachment_points = set()
        atoms_to_remove = []
        
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == '*':
                # 找到与*相连的原子
                for neighbor in atom.GetNeighbors():
                    attachment_points.add(neighbor.GetIdx())
                atoms_to_remove.append(atom.GetIdx())
        
        # 创建一个不包含*的新分子
        if atoms_to_remove:
            # 使用RDKit的编辑功能移除虚拟原子
            edit_mol = Chem.EditableMol(mol)
            # 按逆序删除原子，避免索引变化
            for atom_idx in sorted(atoms_to_remove, reverse=True):
                edit_mol.RemoveAtom(atom_idx)
            
            clean_mol = edit_mol.GetMol()
            if clean_mol is None:
                return None, set()
            
            # 重新计算连接点索引（因为删除了原子）
            adjusted_attachment_points = set()
            original_to_new_idx = {}
            new_idx = 0
            
            for old_idx in range(mol.GetNumAtoms()):
                if old_idx not in atoms_to_remove:
                    original_to_new_idx[old_idx] = new_idx
                    new_idx += 1
            
            for old_idx in attachment_points:
                if old_idx in original_to_new_idx:
                    adjusted_attachment_points.add(original_to_new_idx[old_idx])
            
            return clean_mol, adjusted_attachment_points
        else:
            # 没有虚拟原子，返回原分子和空的连接点集合
            return mol, set()

    def generate_enhanced_scaffold_mask(self, scaffold_smiles, full_smiles, full_adjacency_matrix):
        """
        生成增强的骨架掩码矩阵，基于虚拟原子*的连接点信息。
        
        Args:
            scaffold_smiles (str): 包含*的骨架SMILES
            full_smiles (str): 完整分子SMILES
            full_adjacency_matrix (np.array): 完整分子的邻接矩阵
            
        Returns:
            Tuple[np.array, np.array]: (骨架掩码, 可变区域掩码)
        """
        scaffold_mol, attachment_points = self.parse_scaffold_with_attachment_points(scaffold_smiles)
        full_mol = Chem.MolFromSmiles(full_smiles)

        if scaffold_mol is None or full_mol is None:
            return np.zeros_like(full_adjacency_matrix), np.ones_like(full_adjacency_matrix)

        # 找到骨架在完整分子中的匹配
        matches = full_mol.GetSubstructMatches(scaffold_mol)
        if not matches:
            return np.zeros_like(full_adjacency_matrix), np.ones_like(full_adjacency_matrix)

        match = matches[0]  # 使用第一个匹配
        scaffold_mask = np.zeros_like(full_adjacency_matrix)
        variable_mask = np.zeros_like(full_adjacency_matrix)

        # 标记骨架中的键（这些键是固定的，不会被mask）
        for bond in scaffold_mol.GetBonds():
            begin_idx = match[bond.GetBeginAtomIdx()]
            end_idx = match[bond.GetEndAtomIdx()]
            scaffold_mask[begin_idx, end_idx] = 1
            scaffold_mask[end_idx, begin_idx] = 1

        # 识别可变区域：只有连接到attachment points的区域才是可变的
        attachment_atoms_in_full = set()
        for scaffold_idx in attachment_points:
            if scaffold_idx < len(match):
                full_idx = match[scaffold_idx]
                attachment_atoms_in_full.add(full_idx)

        # 找到所有从attachment points出发的支链
        visited = set()
        
        def dfs_mark_variable_region(atom_idx, came_from=None):
            """深度优先搜索标记可变区域"""
            if atom_idx in visited:
                return
            visited.add(atom_idx)
            
            # 检查这个原子的所有邻居
            for neighbor_idx in range(len(full_adjacency_matrix)):
                if full_adjacency_matrix[atom_idx, neighbor_idx] > 0:  # 有键连接
                    if neighbor_idx != came_from:  # 避免回到来源
                        # 如果邻居不在骨架中，标记这条键为可变
                        if neighbor_idx not in match:
                            variable_mask[atom_idx, neighbor_idx] = 1
                            variable_mask[neighbor_idx, atom_idx] = 1
                            # 继续搜索支链
                            dfs_mark_variable_region(neighbor_idx, atom_idx)
                        elif atom_idx not in match:  # 当前原子不在骨架中，但邻居在
                            variable_mask[atom_idx, neighbor_idx] = 1
                            variable_mask[neighbor_idx, atom_idx] = 1

        # 从每个attachment point开始标记可变区域
        for attachment_atom in attachment_atoms_in_full:
            dfs_mark_variable_region(attachment_atom)

        return scaffold_mask, variable_mask

    def create_training_mask(self, scaffold_mask, variable_mask, mask_ratio=0.2):
        """
        创建训练用的掩码，主要mask可变区域。
        
        Args:
            scaffold_mask (np.array): 骨架掩码
            variable_mask (np.array): 可变区域掩码
            mask_ratio (float): 可变区域的掩码比例
            
        Returns:
            np.array: 训练掩码（1表示需要预测，0表示已知）
        """
        training_mask = np.zeros_like(scaffold_mask)
        
        # 骨架部分始终已知（不需要预测）
        # 可变区域按比例进行掩码
        variable_positions = np.where(variable_mask == 1)
        n_variable = len(variable_positions[0])
        
        if n_variable > 0:
            n_mask = int(n_variable * mask_ratio)
            mask_indices = np.random.choice(n_variable, n_mask, replace=False)
            
            for idx in mask_indices:
                i, j = variable_positions[0][idx], variable_positions[1][idx]
                training_mask[i, j] = 1

        return training_mask

    def process_dataframe(self, df, set_name, update_dicts=True):
        """处理DataFrame中的所有数据。"""
        all_data = []
        skipped_count = 0
        
        for _, row in tqdm(df.iterrows(), desc=f"Processing {set_name} set", total=len(df)):
            smiles = row["smiles"]
            backbone = row["backboneR"]

            spectrum_peaks = list(zip(row["mzs"], row["intensities"]))
            spectrum_features = self.interpolate_msms_to_600(self.make_msms_spectrum(spectrum_peaks))

            node_features, adjacency_matrix = self.smiles_to_graph(smiles, update_dicts=update_dicts)

            if node_features is None:
                skipped_count += 1
                continue

            # 生成增强的掩码
            scaffold_mask, variable_mask = self.generate_enhanced_scaffold_mask(
                backbone, smiles, adjacency_matrix
            )
            
            # 生成训练掩码
            training_mask = self.create_training_mask(scaffold_mask, variable_mask)

            all_data.append(
                {
                    "smiles": smiles,
                    "backbone": backbone,
                    "node_features": node_features,
                    "adjacency_matrix": adjacency_matrix,
                    "scaffold_mask": scaffold_mask,
                    "variable_mask": variable_mask,
                    "training_mask": training_mask,
                    "spectrum": spectrum_features,
                }
            )

        if skipped_count > 0:
            print(f"Skipped {skipped_count} molecules in {set_name} set due to unknown atoms/bonds.")

        return all_data

    def save_dictionaries(self):
        """将原子和键的字典保存到JSON文件。"""
        with open(self.save_dir / "atom_dict.json", "w") as f:
            json.dump(self.atom_dict, f, indent=4)
        with open(self.save_dir / "bond_dict.json", "w") as f:
            json.dump(self.bond_dict, f, indent=4)
        print(f"Dictionaries saved to {self.save_dir}")


@click.command()
@click.option(
    "--ms_data",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    default="../../massspecgym/processed_massspecgymR.parquet",
    required=True,
    help="Path to the MS data parquet file.",
)
@click.option(
    "--out_path",
    "-o",
    type=click.Path(path_type=Path),
    default="./processed_massspecgymR",
    help="Output directory to save processed data.",
)
@click.option("--seed", type=int, default=3245, help="Random seed for reproducibility.")
def main(ms_data: Path, out_path: Path, seed: int):
    """
    处理质谱数据，为图扩散模型准备训练和测试数据集（支持虚拟原子*的骨架定义）。
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"Loading MS data from {ms_data}...")
    data = pd.read_parquet(ms_data)

    # 过滤无效数据
    data.dropna(subset=["smiles", "backboneR", "mzs", "intensities"], inplace=True)
    data = data[data["backboneR"] != ""]
    print(f"Loaded {len(data)} valid entries.")

    # 初始化数据处理器
    data_prep = DataPreparationV2(save_dir=out_path)

    # 处理并保存数据
    processed_data = data_prep.process_dataframe(data, "all", update_dicts=True)
    torch.save(processed_data, out_path / "processed_data_20.pt")

    # 保存字典
    data_prep.save_dictionaries()
    
    # 打印统计信息
    print(f"Atom dictionary: {data_prep.atom_dict}")
    print(f"Bond dictionary: {data_prep.bond_dict}")
    
    # 分析处理结果
    if processed_data:
        total_bonds = sum(np.sum(item["adjacency_matrix"] > 0) // 2 for item in processed_data)
        scaffold_bonds = sum(np.sum(item["scaffold_mask"] > 0) // 2 for item in processed_data)
        variable_bonds = sum(np.sum(item["variable_mask"] > 0) // 2 for item in processed_data)
        
        print(f"\nStatistics:")
        print(f"Total processed molecules: {len(processed_data)}")
        print(f"Average bonds per molecule: {total_bonds / len(processed_data):.2f}")
        print(f"Average scaffold bonds per molecule: {scaffold_bonds / len(processed_data):.2f}")
        print(f"Average variable bonds per molecule: {variable_bonds / len(processed_data):.2f}")
        print(f"Variable region ratio: {variable_bonds / total_bonds * 100:.2f}%")

    print("\nData preparation complete!")


if __name__ == "__main__":
    main()
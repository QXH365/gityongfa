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


class SideChainDataPreparation:
    """
    为支链预测模型准备数据。
    
    基于骨架和完整分子，提取挂载点信息和支链原子类型信息。
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
            if atom.GetSymbol() == "*":
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

    def extract_side_chain_info(self, scaffold_smiles, full_smiles):
        """
        提取支链信息，包括挂载点位置和每个挂载点支链中的原子类型。

        Args:
            scaffold_smiles (str): 包含*的骨架SMILES
            full_smiles (str): 完整分子SMILES

        Returns:
            Tuple[np.array, np.array]: (挂载点标签, 原子类型标签)
        """
        scaffold_mol, attachment_points = self.parse_scaffold_with_attachment_points(scaffold_smiles)
        full_mol = Chem.MolFromSmiles(full_smiles)

        if scaffold_mol is None or full_mol is None:
            return None, None

        # 找到骨架在完整分子中的匹配
        matches = full_mol.GetSubstructMatches(scaffold_mol)
        if not matches:
            return None, None

        match = matches[0]  # 使用第一个匹配
        scaffold_atoms = set(match)
        num_atoms = full_mol.GetNumAtoms()

        # 初始化标签
        attachment_labels = np.zeros(num_atoms, dtype=np.int32)
        # **关键修复**: 使用固定的字典长度，而不是动态变化的长度
        atom_labels = np.zeros((num_atoms, len(self.atom_dict)), dtype=np.float32)

        # 找到挂载点在完整分子中的对应位置
        attachment_atoms_in_full = []
        for scaffold_idx in attachment_points:
            if scaffold_idx < len(match):
                full_idx = match[scaffold_idx]
                attachment_atoms_in_full.append(full_idx)
                attachment_labels[full_idx] = 1

        # 为每个挂载点找到其支链并提取原子类型
        for attachment_atom in attachment_atoms_in_full:
            side_chain_atoms = set()
            
            # DFS搜索支链
            visited = set()
            def dfs_collect_side_chain(atom_idx, came_from=None):
                if atom_idx in visited or atom_idx in scaffold_atoms:
                    return
                visited.add(atom_idx)
                side_chain_atoms.add(atom_idx)
                
                atom = full_mol.GetAtomWithIdx(atom_idx)
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx != came_from:
                        dfs_collect_side_chain(neighbor_idx, atom_idx)
            
            # 从挂载点开始搜索支链
            atom = full_mol.GetAtomWithIdx(attachment_atom)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx not in scaffold_atoms:
                    dfs_collect_side_chain(neighbor_idx, attachment_atom)

            # 统计支链中的原子类型
            atom_types_in_side_chain = set()
            for atom_idx in side_chain_atoms:
                atom = full_mol.GetAtomWithIdx(atom_idx)
                atom_symbol = atom.GetSymbol()
                if atom_symbol in self.atom_dict:
                    atom_types_in_side_chain.add(self.atom_dict[atom_symbol])
            
            # 设置原子类型标签（只在挂载点位置设置）
            for atom_type_idx in atom_types_in_side_chain:
                atom_labels[attachment_atom, atom_type_idx] = 1.0

        return attachment_labels, atom_labels

    def process_dataframe_for_side_chain(self, df, set_name, update_dicts=False):
        """
        处理DataFrame中的所有数据，为支链预测准备数据。
        **注意**: update_dicts 默认应为 False，因为字典应该已经预先构建好。
        """
        all_data = []
        skipped_count = 0

        for _, row in tqdm(df.iterrows(), desc=f"Processing {set_name} set for side chain prediction", total=len(df)):
            smiles = row["smiles"]
            backbone = row["backboneR"]

            # 处理光谱数据
            spectrum_peaks = list(zip(row["mzs"], row["intensities"]))
            spectrum_features = self.interpolate_msms_to_600(self.make_msms_spectrum(spectrum_peaks))

            # 转换分子为图
            node_features, adjacency_matrix = self.smiles_to_graph(smiles, update_dicts=update_dicts)

            if node_features is None:
                skipped_count += 1
                continue

            # 提取支链信息
            attachment_labels, atom_labels = self.extract_side_chain_info(backbone, smiles)

            if attachment_labels is None or atom_labels is None:
                skipped_count += 1
                continue

            # 检查维度一致性
            if len(attachment_labels) != len(node_features) or len(atom_labels) != len(node_features):
                skipped_count += 1
                continue

            all_data.append({
                "smiles": smiles,
                "backbone": backbone,
                "node_features": node_features,
                "adjacency_matrix": adjacency_matrix,
                "attachment_labels": attachment_labels,
                "atom_labels": atom_labels,
                "spectrum": spectrum_features,
            })

        if skipped_count > 0:
            print(f"Skipped {skipped_count} molecules in {set_name} set due to processing errors.")

        return all_data

    def build_dictionaries(self, df):
        """
        遍历整个数据集一次，以构建完整的原子和键字典。
        """
        print("Building dictionaries by scanning the entire dataset...")
        for _, row in tqdm(df.iterrows(), desc="Building dictionaries", total=len(df)):
            # 只需处理SMILES即可构建字典
            self.smiles_to_graph(row["smiles"], update_dicts=True)
        print("Dictionaries built successfully.")

    def save_dictionaries(self):
        """将原子和键的字典保存到JSON文件。"""
        with open(self.save_dir / "atom_dict.json", "w") as f:
            json.dump(self.atom_dict, f, indent=4)
        with open(self.save_dir / "bond_dict.json", "w") as f:
            json.dump(self.bond_dict, f, indent=4)
        print(f"Dictionaries saved to {self.save_dir}")

    def analyze_side_chain_data(self, processed_data):
        """分析处理后的支链数据统计信息。"""
        if not processed_data:
            print("No data to analyze.")
            return

        total_molecules = len(processed_data)
        total_attachment_points = 0
        atom_type_counts = {}
        
        for item in processed_data:
            # 统计挂载点数量
            num_attachments = np.sum(item["attachment_labels"])
            total_attachment_points += num_attachments
            
            # 统计原子类型出现次数
            attachment_indices = np.where(item["attachment_labels"] == 1)[0]
            for idx in attachment_indices:
                atom_types = np.where(item["atom_labels"][idx] == 1)[0]
                for atom_type in atom_types:
                    if atom_type not in atom_type_counts:
                        atom_type_counts[atom_type] = 0
                    atom_type_counts[atom_type] += 1

        print(f"\n=== 支链数据统计 ===")
        print(f"总分子数: {total_molecules}")
        print(f"总挂载点数: {total_attachment_points}")
        print(f"平均每分子挂载点数: {total_attachment_points / total_molecules:.2f}")
        
        print(f"\n原子类型在支链中的出现频率:")
        # 转换atom_dict用于反向查找
        reverse_atom_dict = {v: k for k, v in self.atom_dict.items()}
        for atom_type, count in sorted(atom_type_counts.items(), key=lambda x: x[1], reverse=True):
            atom_symbol = reverse_atom_dict.get(atom_type, f"Unknown({atom_type})")
            print(f"  {atom_symbol}: {count} 次")


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
    default="./side_chain_processed_data",
    help="Output directory to save processed side chain data.",
)
@click.option("--seed", type=int, default=3245, help="Random seed for reproducibility.")
def main(ms_data: Path, out_path: Path, seed: int):
    """
    处理质谱数据，为支链预测模型准备训练数据。
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Loading MS data from {ms_data}...")
    data = pd.read_parquet(ms_data)

    # 过滤无效数据
    print("Filtering invalid data...")
    original_count = len(data)
    data.dropna(subset=["smiles", "backboneR", "mzs", "intensities"], inplace=True)
    data = data[data["backboneR"] != ""]
    print(f"Filtered from {original_count} to {len(data)} valid entries.")

    # 初始化数据处理器
    data_prep = SideChainDataPreparation(save_dir=out_path)

    # **步骤 1: 预先构建完整的字典**
    data_prep.build_dictionaries(data)
    
    # **步骤 2: 使用固定的字典处理数据**
    # 注意：这里的 update_dicts=False
    print("Processing data for side chain prediction...")
    processed_data = data_prep.process_dataframe_for_side_chain(data, "all", update_dicts=False)
    
    # 保存处理后的数据
    torch.save(processed_data, out_path / "side_chain_data.pt")
    print(f"Saved {len(processed_data)} processed molecules to {out_path / 'side_chain_data.pt'}")

    # 保存字典
    data_prep.save_dictionaries()

    # 打印字典信息
    print(f"\nAtom dictionary: {data_prep.atom_dict}")
    print(f"Bond dictionary: {data_prep.bond_dict}")

    # 分析数据
    data_prep.analyze_side_chain_data(processed_data)

    print(f"\n支链预测数据预处理完成！")
    print(f"数据保存在: {out_path}")
    print(f"- side_chain_data.pt: 处理后的训练数据")
    print(f"- atom_dict.json: 原子类型字典")
    print(f"- bond_dict.json: 键类型字典")


if __name__ == "__main__":
    main()
import torch
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import numpy as np
from pathlib import Path
import json

class SideChainDataset(Dataset):
    """
    用于支链预测任务的PyTorch数据集。
    将预处理的数据转换为PyTorch Geometric的Data对象。
    """
    def __init__(self, data_path, atom_dict_path):
        """
        Args:
            data_path (str or Path): 'side_chain_data.pt'文件的路径。
            atom_dict_path (str or Path): 'atom_dict.json'文件的路径。
        """
        print(f"Loading data from {data_path}...")
        self.data = torch.load(data_path)
        
        with open(atom_dict_path, 'r') as f:
            self.atom_dict = json.load(f)
        self.atom_vocab_size = len(self.atom_dict)
        print(f"Data loaded. Total samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 节点特征
        node_features = torch.tensor(item['node_features'], dtype=torch.long)
        
        # 边索引和边属性
        adj = item['adjacency_matrix']
        edge_indices = np.stack(np.where(adj > 0))
        edge_attr = torch.tensor(adj[edge_indices[0], edge_indices[1]], dtype=torch.long)
        edge_index = torch.tensor(edge_indices, dtype=torch.long)

        # 全局特征 (质谱)
        spectrum = torch.tensor(item['spectrum'], dtype=torch.float32).unsqueeze(0) # Shape: [1, 600]

        # 标签
        attachment_labels = torch.tensor(item['attachment_labels'], dtype=torch.bool) # [N]
        atom_labels = torch.tensor(item['atom_labels'], dtype=torch.float32) # [N, atom_vocab_size]

        # 创建PyG Data对象
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            spectrum=spectrum,
            attachment_mask=attachment_labels,
            y=atom_labels
        )
        
        return graph_data

def create_dataloaders(data_dir, batch_size, seed=42):
    """
    创建训练、验证和测试的DataLoader。
    """
    data_path = Path(data_dir) / "side_chain_data.pt"
    atom_dict_path = Path(data_dir) / "atom_dict.json"
    
    dataset = SideChainDataset(data_path, atom_dict_path)
    
    # 划分数据集 (80% 训练, 10% 验证, 10% 测试)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # 创建DataLoader
    # PyTorch Geometric的DataLoader会自动处理图的批处理
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, dataset.atom_vocab_size
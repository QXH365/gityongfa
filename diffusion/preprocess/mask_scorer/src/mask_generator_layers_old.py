# mask_generator_layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn,tensor,float32
# 假设你的项目中有一个类似 detanet/modules/acts.py 的文件提供激活函数
# 或者我们直接在这里定义需要的激活函数
def get_activation(act_name, num_features=None):
    if act_name == 'relu':
        return nn.ReLU()
    elif act_name == 'silu' or act_name == 'swish':
        return nn.SiLU()
    elif act_name == 'gelu':
        return nn.GELU()
    # 可以添加更多激活函数
    else:
        raise NotImplementedError(f"激活函数 {act_name} 未实现")

def get_elec_feature(max_atomic_number):
    '''
    Electronic feature of the first 4 periodic elements.
     The number of single and paired electrons in an orbital.
     Also indicates the filled and half-filled state of the orbitals.
      '''
    elec_feature = tensor(
        # P:Pair electron
        # S:Single electron
        #|  1s |  2s |  2p |  3s |  3p |  4s |  3d |  4p |
        # P  S  P  S  P  S  P  S  P  S  P  S  P  S  P  S
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 0 None
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 1 H
         [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 2 He
         [2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 3 Li
         [2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 4 Be
         [2, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 5 B
         [2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 6 C
         [2, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 7 N
         [2, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 8 O
         [2, 0, 2, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 9 F
         [2, 0, 2, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 10 Ne
         [2, 0, 2, 0, 6, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 11 Na
         [2, 0, 2, 0, 6, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 12 Me
         [2, 0, 2, 0, 6, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, ],  # 13 Al
         [2, 0, 2, 0, 6, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, ],  # 14 Si
         [2, 0, 2, 0, 6, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, ],  # 15 P
         [2, 0, 2, 0, 6, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, ],  # 16 S
         [2, 0, 2, 0, 6, 0, 2, 0, 4, 1, 0, 0, 0, 0, 0, 0, ],  # 17 Cl
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 0, 0, 0, 0, 0, 0, ],  # 18 Ar
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 0, 1, 0, 0, 0, 0, ],  # 19 K
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0, 0, 0, 0, 0, ],  # 20 Ca
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0, 0, 1, 0, 0, ],  # 21 Sc
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0, 0, 2, 0, 0, ],  # 22 Ti
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0, 0, 3, 0, 0, ],  # 23 V
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 0, 1, 0, 5, 0, 0, ],  # 24 Cr
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0, 0, 5, 0, 0, ],  # 25 Mn
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0, 2, 4, 0, 0, ],  # 26 Fe
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0, 4, 3, 0, 0, ],  # 27 Co
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0, 6, 2, 0, 0, ],  # 28 Ni
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 0, 1,10, 0, 0, 0, ],  # 29 Cu
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0,10, 0, 0, 0, ],  # 30 Zn
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0,10, 0, 0, 1, ],  # 31 Ga
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0,10, 0, 0, 2, ],  # 32 Ge
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0,10, 0, 0, 3, ],  # 33 As
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0,10, 0, 2, 2, ],  # 34 Se
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0,10, 0, 4, 1, ],  # 35 Br
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0,10, 0, 6, 0, ],  # 36 Kr
         ], dtype=float32)
    elec=elec_feature[0:max_atomic_number+1]
    return (elec/elec.max())

class AtomEmbeddingComplex(nn.Module):
    """
    更复杂的原子嵌入层，结合了原子核嵌入（基于原子序数）和电子特征。
    灵感来自 detanet/modules/embedding.py。

    Args:
        embed_dim (int): 输出嵌入向量的维度。
        max_atomic_number (int): 数据集中可能出现的最大原子序数。
        activation (str): 嵌入后使用的激活函数名称。
    """
    def __init__(self, embed_dim: int, max_atomic_number: int = 100, activation: str = 'silu'):
        super().__init__()
        self.max_atomic_number = max_atomic_number
        # 加载或定义电子特征
        self.register_buffer("elec_features", get_elec_feature(max_atomic_number))
        elec_dim = self.elec_features.shape[1]

        # 原子核嵌入层 (基于原子序数)
        self.nuclear_embedding = nn.Embedding(num_embeddings=max_atomic_number + 1, embedding_dim=embed_dim)

        # 电子特征嵌入层
        self.electronic_embedding = nn.Linear(elec_dim, embed_dim, bias=False)

        # 最终混合层和激活
        self.final_linear = nn.Linear(embed_dim, embed_dim)
        self.activation = get_activation(activation, embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.nuclear_embedding.reset_parameters()
        nn.init.xavier_uniform_(self.electronic_embedding.weight)
        nn.init.xavier_uniform_(self.final_linear.weight)
        self.final_linear.bias.data.fill_(0)

    def forward(self, atom_types: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atom_types (torch.Tensor): 原子类型张量，形状 (num_atoms,)，数据类型 Long。

        Returns:
            torch.Tensor: 原子嵌入特征，形状 (num_atoms, embed_dim)。
        """
        # 确保索引不越界
        if atom_types.max() > self.max_atomic_number:
            raise IndexError(f"原子类型 {atom_types.max()} 超出最大原子序数 {self.max_atomic_number}")

        # 获取原子核嵌入和电子特征嵌入
        nuclear_feat = self.nuclear_embedding(atom_types)
        electronic_feat = self.electronic_embedding(self.elec_features[atom_types])

        # 混合特征并通过最终层和激活
        combined_feat = nuclear_feat + electronic_feat # 简单相加，也可以是其他组合方式
        output_feat = self.activation(self.final_linear(combined_feat))

        return output_feat

class MultiLayerPerceptron(nn.Module):
    """
    简单的多层感知机，用于边特征编码器。
    参考 src/models/common.py 和 detanet/modules/multilayer_perceptron.py。
    """
    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0):
        super().__init__()
        dims = [input_dim] + hidden_dims
        if isinstance(activation, str):
            self.activation = get_activation(activation)
        else: # Assume it's an nn.Module
            self.activation = activation

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2: # No activation/dropout after the last layer
                if self.activation:
                    layers.append(self.activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class EdgeFeatureEncoder(nn.Module):
    """
    边特征编码器，结合了键类型嵌入和边长信息。
    灵感来自 src/models/encoder/edge.py 中的 MLPEdgeEncoder。

    Args:
        embed_dim (int): 嵌入维度。
        num_bond_types (int): 数据集中不同键类型的数量。
        activation (str): MLP 中使用的激活函数名称。
    """
    def __init__(self, embed_dim: int, num_bond_types: int = 10, activation: str = 'relu'):
        super().__init__()
        self.embed_dim = embed_dim
        # 键类型嵌入
        self.bond_embedding = nn.Embedding(num_embeddings=num_bond_types, embedding_dim=embed_dim)
        # 用于处理边长的 MLP
        self.length_mlp = MultiLayerPerceptron(1, [embed_dim, embed_dim], activation=activation)
        # 可以添加一个最终的激活层或 LayerNorm
        # self.final_act = get_activation(activation, embed_dim)

    @property
    def out_channels(self):
        return self.embed_dim

    def forward(self, edge_types: torch.Tensor, edge_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_types (torch.Tensor): 边类型张量，形状 (num_edges,)，数据类型 Long。
            edge_lengths (torch.Tensor): 边长张量，形状 (num_edges, 1)，数据类型 Float。

        Returns:
            torch.Tensor: 边特征，形状 (num_edges, embed_dim)。
        """
        # 获取键类型嵌入
        bond_feat = self.bond_embedding(edge_types)
        # 处理边长
        length_feat = self.length_mlp(edge_lengths)
        # 结合特征 (例如，逐元素相乘)
        edge_features = bond_feat * length_feat
        # 可选: 应用最终激活或归一化
        # edge_features = self.final_act(edge_features)
        return edge_features
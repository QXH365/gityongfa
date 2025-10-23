# mask_generator_layers.py (更新版)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch import nn,tensor,float32
from torch_geometric.data import Data # 导入 Data 用于类型提示

def get_activation(act_name, num_features=None):
    if act_name == 'relu':
        return nn.ReLU()
    elif act_name == 'silu' or act_name == 'swish':
        return nn.SiLU()
    elif act_name == 'gelu':
        return nn.GELU()
    elif act_name == 'softplus': # GINEConv 默认可能使用 softplus
        return nn.Softplus()
    else:
        raise NotImplementedError(f"激活函数 {act_name} 未实现")

def get_elec_feature(max_atomic_number):
    """
    Electronic feature based on detanet/modules/embedding.py,
    extended with zero padding for higher atomic numbers.
    """
    # Features defined up to Kr (atomic number 36)
    elec_feature_base = torch.tensor(
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
         [2, 0, 2, 0, 6, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 12 Mg, corrected typo Me->Mg
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
         ], dtype=torch.float32)

    # Get the number of features (columns)
    num_features = elec_feature_base.shape[1]
    # Create the full tensor with zeros up to max_atomic_number
    elec_feature_full = torch.zeros(max_atomic_number + 1, num_features, dtype=torch.float32)
    # Copy the known features
    num_known = elec_feature_base.shape[0]
    elec_feature_full[:num_known, :] = elec_feature_base

    # Normalize (optional, as in original code)
    if elec_feature_full.max() > 0:
         elec = elec_feature_full / elec_feature_full.max() # Normalize by the overall max value
    else:
         elec = elec_feature_full

    return elec

class AtomEmbeddingComplex(nn.Module):
    # ... (代码同上一个回复) ...
    def __init__(self, embed_dim: int, max_atomic_number: int = 100, activation: str = 'silu'):
        super().__init__()
        self.max_atomic_number = max_atomic_number
        self.register_buffer("elec_features", get_elec_feature(max_atomic_number))
        elec_dim = self.elec_features.shape[1]
        self.nuclear_embedding = nn.Embedding(num_embeddings=max_atomic_number + 1, embedding_dim=embed_dim)
        self.electronic_embedding = nn.Linear(elec_dim, embed_dim, bias=False)
        self.final_linear = nn.Linear(embed_dim, embed_dim)
        self.activation = get_activation(activation, embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.nuclear_embedding.reset_parameters()
        nn.init.xavier_uniform_(self.electronic_embedding.weight)
        nn.init.xavier_uniform_(self.final_linear.weight)
        self.final_linear.bias.data.fill_(0)

    def forward(self, atom_types: torch.Tensor) -> torch.Tensor:
        if atom_types.max() > self.max_atomic_number:
            raise IndexError(f"原子类型 {atom_types.max()} 超出最大原子序数 {self.max_atomic_number}")
        nuclear_feat = self.nuclear_embedding(atom_types)
        electronic_feat = self.electronic_embedding(self.elec_features[atom_types])
        combined_feat = nuclear_feat + electronic_feat
        output_feat = self.activation(self.final_linear(combined_feat))
        return output_feat


class MultiLayerPerceptron(nn.Module):
    # ... (代码同上一个回复) ...
    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0):
        super().__init__()
        dims = [input_dim] + hidden_dims
        # --- 修正：确保 get_activation 能处理 Module ---
        if isinstance(activation, str):
            self.activation = get_activation(activation)
        elif isinstance(activation, nn.Module):
            self.activation = activation
        else:
             self.activation = None # Or raise error

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                if self.activation:
                    layers.append(self.activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class EdgeFeatureEncoder(nn.Module):
    # ... (代码同上一个回复) ...
    def __init__(self, embed_dim: int, num_bond_types: int = 10, activation: str = 'relu'):
        super().__init__()
        self.embed_dim = embed_dim
        self.bond_embedding = nn.Embedding(num_embeddings=num_bond_types, embedding_dim=embed_dim)
        self.length_mlp = MultiLayerPerceptron(1, [embed_dim, embed_dim], activation=activation)

    @property
    def out_channels(self):
        return self.embed_dim

    def forward(self, edge_types: torch.Tensor, edge_lengths: torch.Tensor) -> torch.Tensor:
        bond_feat = self.bond_embedding(edge_types)
        length_feat = self.length_mlp(edge_lengths)
        edge_features = bond_feat * length_feat
        return edge_features


# --- 新增 GNN 组件 ---

class GINEConv(MessagePassing):
    """
    Graph Isomorphism Network with Edge features (GINE) Convolution Layer.
    从 src/models/encoder/gin.py 借鉴。
    """
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 activation="softplus", **kwargs):
        super().__init__(aggr='add', **kwargs) # 使用 'add' 聚合邻居信息
        self.nn = nn # 用于更新节点特征的 MLP
        self.initial_eps = eps

        if isinstance(activation, str):
            # 使用我们定义的 get_activation
            self.activation = get_activation(activation)
        elif isinstance(activation, nn.Module):
            self.activation = activation
        else:
            self.activation = None

        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            # 对于标准 GNN，输入 x 通常是 (num_nodes, node_feat_dim)
            # MessagePassing 基类期望 OptPairTensor (x_source, x_target)
            # 如果只提供一个 Tensor，则假定源节点和目标节点特征相同
            x: OptPairTensor = (x, x)

        # 检查节点和边特征维度是否匹配（在 message 函数中需要它们交互）
        if isinstance(edge_index, Tensor):
            if edge_attr is None:
                 raise ValueError("edge_attr must be provided for GINEConv when edge_index is a Tensor")
            # 假设 message 函数中 x_j 和 edge_attr 需要相加或类似操作
            # assert x[0].size(-1) == edge_attr.size(-1) # 维度不一定需要完全相等，取决于 message 实现
        # elif isinstance(edge_index, SparseTensor): # SparseTensor 的处理方式不同
        #     assert x[0].size(-1) == edge_index.size(-1)

        # 调用基类的 propagate 方法，它会依次调用 message, aggregate, update
        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        # GIN 更新公式: h_i^{(l+1)} = MLP^{(l+1)}( (1 + eps^{(l)}) * h_i^{(l)} + sum_{j in N(i)} relu(h_j^{(l)} + e_{ij}) )
        x_r = x[1] # 目标节点特征 (中心节点自身)
        if x_r is not None:
            out += (1 + self.eps) * x_r # 加上中心节点自身的特征 (带 epsilon 权重)

        return self.nn(out) # 通过 MLP 更新最终输出

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        """
        定义如何从邻居节点 j 和边 (j,i) 的特征构造消息。
        x_j: 源节点 (邻居 j) 的特征，形状 (num_edges, node_feat_dim)
        edge_attr: 边 (j,i) 的特征，形状 (num_edges, edge_feat_dim)
        """
        # GINEConv 的标准 message: 对邻居节点特征和边特征求和（或拼接后处理），然后应用激活
        # 这里假设边特征维度与节点特征维度相同以便相加
        msg = x_j + edge_attr
        if self.activation:
            return self.activation(msg)
        else:
            return msg

    # update 函数默认是直接返回聚合后的消息，这里不需要重写

    def __repr__(self):
        return f'{self.__class__.__name__}(nn={self.nn})'


class MaskGNNBackbone(nn.Module):
    """
    掩码生成器的 GNN 骨干网络，使用多层 GINEConv。
    改编自 src/models/encoder/gin.py 中的 GINEncoder。

    Args:
        input_dim (int): 输入节点特征的维度 (来自 AtomEmbedding)。
        hidden_dim (int): GNN 层的隐藏维度。
        edge_dim (int): 输入边特征的维度 (来自 EdgeFeatureEncoder)。
        num_convs (int): GINEConv 层的数量。
        activation (str): GNN 层之间使用的激活函数名称。
        short_cut (bool): 是否使用残差连接。
        concat_hidden (bool): 是否将所有 GNN 层的输出拼接作为最终节点表示。
    """
    def __init__(self, input_dim: int, hidden_dim: int, edge_dim: int, num_convs: int = 3,
                 activation: str = 'relu', short_cut: bool = True, concat_hidden: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim # 记录边特征维度
        self.num_convs = num_convs
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        # 如果输入维度与隐藏维度不同，添加一个初始线性层
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = None

        if isinstance(activation, str):
            self.activation = get_activation(activation, hidden_dim)
        elif isinstance(activation, nn.Module):
            self.activation = activation
        else:
            self.activation = None

        self.convs = nn.ModuleList()
        for _ in range(self.num_convs):
            # GINEConv 内部的 MLP：输入是聚合后的节点特征 (hidden_dim)，输出也是 hidden_dim
            # GINEConv 的 message 函数需要节点特征和边特征交互，这里我们假设它们维度相同
            if edge_dim != hidden_dim:
                 # 如果维度不同，需要适配，例如在 message 中先将 edge_attr 投影
                 # 或者修改 GINEConv 的 MLP 输入维度 (但这不标准)
                 # 一个简单的适配是在 GINEConv 前面对 edge_attr 做一次投影
                 # 但更标准的 GINE 是期望 edge_dim == hidden_dim
                 # 这里我们先假设它们相同，如果报错再调整
                 # 【注意】如果 hidden_dim != edge_dim, GINEConv 中的 x_j + edge_attr 会失败
                 pass # 保持维度一致是 GINE 的常见做法

            mlp = MultiLayerPerceptron(hidden_dim, [hidden_dim, hidden_dim], activation=activation)
            self.convs.append(GINEConv(nn=mlp, activation=activation)) # GINEConv 消息计算时的激活

        # 如果 concat_hidden，计算最终输出维度
        if self.concat_hidden:
            self.output_dim = hidden_dim * num_convs
        else:
            self.output_dim = hidden_dim

    def forward(self, node_attr: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """
        Args:
            node_attr (Tensor): 节点特征 (来自 AtomEmbedding)，形状 (num_atoms, input_dim)。
            edge_index (Tensor): 边索引，形状 (2, num_edges)。
            edge_attr (Tensor): 边特征 (来自 EdgeFeatureEncoder)，形状 (num_edges, edge_dim)。

        Returns:
            Tensor: GNN 处理后的节点特征，形状 (num_atoms, output_dim)。
        """
        # 输入投影（如果需要）
        if self.input_proj:
            conv_input = self.input_proj(node_attr)
        else:
            conv_input = node_attr # (num_atoms, hidden_dim)

        hiddens = []
        for conv_idx, conv in enumerate(self.convs):
            hidden = conv(conv_input, edge_index, edge_attr)
            # 应用层间的激活函数
            if conv_idx < len(self.convs) - 1 and self.activation is not None:
                hidden = self.activation(hidden)

            # 残差连接
            if self.short_cut and hidden.shape == conv_input.shape:
                # 使用原地加法可能导致梯度问题，显式赋值更好
                hidden = hidden + conv_input

            hiddens.append(hidden)
            conv_input = hidden # 更新下一层的输入

        # 处理输出
        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1] # 只取最后一层的输出

        return node_feature
    
    
class MaskGeneratorNet(nn.Module):
    """
    完整的掩码生成器模型。
    包含原子嵌入、边特征编码、GNN 骨干和输出 MLP。

    Args:
        embed_dim (int): 原子和边的嵌入维度，以及 GNN 的隐藏维度。
        max_atomic_number (int): 最大原子序数。
        num_bond_types (int): 最大键类型。
        num_convs (int): GNN 层数。
        activation (str): 模型中主要的激活函数。
        short_cut (bool): GNN 是否使用残差连接。
        concat_hidden (bool): GNN 是否拼接隐藏层输出。
        output_mlp_hidden_dims (list[int]): 输出 MLP 的隐藏层维度列表。
    """
    def __init__(self, embed_dim: int = 128, max_atomic_number: int = 100, num_bond_types: int = 10,
                 num_convs: int = 4, activation: str = 'silu', short_cut: bool = True,
                 concat_hidden: bool = False, output_mlp_hidden_dims: list = [64]):
        super().__init__()

        # 1. 原子嵌入层
        self.atom_embed = AtomEmbeddingComplex(
            embed_dim=embed_dim,
            max_atomic_number=max_atomic_number,
            activation=activation
        )

        # 2. 边特征编码器
        self.edge_encoder = EdgeFeatureEncoder(
            embed_dim=embed_dim, # 输出维度与 GNN 隐藏维度匹配
            num_bond_types=num_bond_types,
            activation=activation
        )

        # 3. GNN 骨干网络
        self.gnn_backbone = MaskGNNBackbone(
            input_dim=embed_dim,    # 来自 AtomEmbedding
            hidden_dim=embed_dim,   # GNN 内部维度
            edge_dim=embed_dim,     # 来自 EdgeEncoder
            num_convs=num_convs,
            activation=activation,
            short_cut=short_cut,
            concat_hidden=concat_hidden
        )

        # 4. 输出 MLP
        gnn_output_dim = self.gnn_backbone.output_dim
        # 输出 MLP： GNN 输出维度 -> 隐藏层 -> 1 (分数)
        self.output_mlp = MultiLayerPerceptron(
            input_dim=gnn_output_dim,
            hidden_dims=output_mlp_hidden_dims + [1], # 添加最终输出维度 1
            activation=activation
        )

        # 5. 最终 Sigmoid 激活
        self.final_activation = nn.Sigmoid()

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data (torch_geometric.data.Data): 输入的图数据批次，
                需要包含 data.atom_type, data.edge_index, data.edge_type。
                如果使用 EdgeFeatureEncoder，还需要 data.edge_length (可以是模拟的)。

        Returns:
            torch.Tensor: 预测的原子灵活性分数，形状 (num_atoms, 1)，值在 [0, 1] 之间。
        """
        # 获取原子嵌入
        atom_features = self.atom_embed(data.atom_type) # (N, embed_dim)

        # 获取边特征
        # 处理无边的情况
        if data.edge_index.numel() > 0:
             # 确保 data 对象中有 edge_length
            if not hasattr(data, 'edge_length'):
                raise AttributeError("数据对象缺少 'edge_length' 属性，EdgeFeatureEncoder 需要它。")
            edge_features = self.edge_encoder(data.edge_type, data.edge_length) # (E, embed_dim)
        else:
            # 创建一个正确形状和类型的空张量
             edge_features = torch.empty((0, self.edge_encoder.out_channels),
                                       dtype=atom_features.dtype,
                                       device=atom_features.device)

        # 通过 GNN 骨干
        node_features_final = self.gnn_backbone(atom_features, data.edge_index, edge_features) # (N, gnn_output_dim)

        # 通过输出 MLP
        scores_raw = self.output_mlp(node_features_final) # (N, 1)

        # 应用最终 Sigmoid 激活
        scores_pred = self.final_activation(scores_raw) # (N, 1)

        return scores_pred
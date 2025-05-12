from typing import List, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.nn.init import xavier_normal_


class MLPLayers(nn.Module):
    """多层感知机 (MLP) 模块。

    Args:
        layers (List[int]): 一个整数列表，表示 MLP 中每一层的维度。
            例如，[64, 32, 16] 表示一个三层 MLP，输入层维度为 64，
            隐藏层维度为 32，输出层维度为 16。
        dropout (float, optional): Dropout 比率。默认为 0.0。
        activation (Union[str, Type[nn.Module]], optional): 激活函数的名称或类型。
            支持 'sigmoid', 'tanh', 'relu', 'leakyrelu', 'none' 或 nn.Module 的子类。
            默认为 "relu"。
        bn (bool, optional): 是否在除最后一层外的每一层之后使用 BatchNorm1d。
            默认为 False。
    """

    def __init__(
        self,
        layers: List[int],
        dropout: float = 0.0,
        activation: Union[str, Type[nn.Module]] = "relu",
        bn: bool = False,
    ):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))

            # 在除最后一层外的每一层之后添加 BatchNorm
            if self.use_bn and idx < (len(self.layers) - 2):
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))

            # 在除最后一层外的每一层之后添加激活函数
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None and idx < (len(self.layers) - 2):
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        """初始化线性层的权重。

        使用 Xavier 正态分布初始化权重，并将偏置初始化为 0。

        Args:
            module (nn.Module): 需要初始化的模块。
        """
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        """MLP 的前向传播。

        Args:
            input_feature (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: MLP 的输出张量。
        """
        return self.mlp_layers(input_feature)


def activation_layer(
    activation_name: Optional[Union[str, Type[nn.Module]]] = "relu",
    emb_dim: Optional[int] = None,
) -> Optional[nn.Module]:
    """根据名称或类型创建激活层。

    Args:
        activation_name (Optional[Union[str, Type[nn.Module]]], optional): 激活函数的名称或类型。
            支持 'sigmoid', 'tanh', 'relu', 'leakyrelu', 'none' 或 nn.Module 的子类。
            如果为 None 或 'none'，则不返回任何激活层。
            默认为 "relu"。
        emb_dim (Optional[int], optional): 嵌入维度，当前未使用。
            默认为 None。

    Returns:
        Optional[nn.Module]: 对应的 PyTorch 激活层模块，如果 activation_name 为 None 或 'none' 则返回 None。

    Raises:
        NotImplementedError: 如果提供的 activation_name 不被支持。
    """
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        name = activation_name.lower()
        if name == "sigmoid":
            activation = nn.Sigmoid()
        elif name == "tanh":
            activation = nn.Tanh()
        elif name == "relu":
            activation = nn.ReLU()
        elif name == "leakyrelu":
            activation = nn.LeakyReLU()
        elif name == "none":
            activation = None
        else:
            raise NotImplementedError(
                f"activation function {activation_name} is not implemented"
            )
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            f"activation function {activation_name} is not implemented"
        )

    return activation


def kmeans(
    samples: torch.Tensor,
    num_clusters: int,
    num_iters: int = 10,
) -> torch.Tensor:
    """使用 scikit-learn 的 KMeans 对样本进行聚类。

    Args:
        samples (torch.Tensor): 需要聚类的样本张量，形状为 (B, dim)。
        num_clusters (int): 聚类的数量。
        num_iters (int, optional): KMeans 算法的最大迭代次数。默认为 10。

    Returns:
        torch.Tensor: 聚类中心的张量，形状为 (num_clusters, dim)。
    """
    device = samples.device
    x = samples.cpu().detach().numpy()

    cluster = KMeans(n_clusters=num_clusters, max_iter=num_iters, n_init=1).fit(
        x
    )  # Set n_init explicitly

    centers = cluster.cluster_centers_
    tensor_centers = torch.from_numpy(centers.astype(np.float32)).to(
        device
    )  # Ensure float32 for consistency

    return tensor_centers


@torch.no_grad()
def sinkhorn_algorithm(
    distances: torch.Tensor, epsilon: float, sinkhorn_iterations: int
) -> torch.Tensor:
    """应用 Sinkhorn-Knopp 算法来获得最优传输分配矩阵。

    Args:
        distances (torch.Tensor): 距离矩阵，形状为 (B, K)，其中 B 是样本数，K 是质心数。
        epsilon (float): Sinkhorn 算法中的正则化强度。
        sinkhorn_iterations (int): Sinkhorn 算法的迭代次数。

    Returns:
        torch.Tensor: 分配矩阵 Q，形状为 (B, K)。每一列的和为 1。
    """
    Q = torch.exp(-distances / epsilon)

    B = Q.shape[0]  # number of samples to assign
    K = Q.shape[1]  # how many centroids per block

    # 确保矩阵的和为 1
    sum_Q = Q.sum()
    if sum_Q != 0:
        Q /= sum_Q
    else:
        # Handle case where sum_Q is zero, e.g., assign uniform probability
        Q.fill_(1.0 / (B * K))

    for _ in range(sinkhorn_iterations):
        # 归一化每一列：每个样本的总权重必须为 1/B
        # Avoid division by zero
        sum_col = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_col + 1e-8  # Add small epsilon for numerical stability
        Q /= B

        # 归一化每一行：每个原型的总权重必须为 1/K
        sum_row = torch.sum(Q, dim=0, keepdim=True)
        Q /= sum_row + 1e-8  # Add small epsilon for numerical stability
        Q /= K

    # 列必须和为 1，这样 Q 才是一个分配矩阵
    # Ensure columns sum to 1/B before multiplying by B
    # Re-normalize columns to sum to 1/B
    sum_col_final = torch.sum(Q, dim=1, keepdim=True)
    Q /= sum_col_final + 1e-8
    Q /= B

    # Now multiply by B to make columns sum to 1
    Q *= B
    return Q

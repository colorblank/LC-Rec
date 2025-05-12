# -*- coding: utf-8 -*-
"""RQ-VAE 模型处理服务。

该模块包含用于处理 RQ-VAE 模型相关操作的函数，
例如从嵌入向量中获取量化索引。
"""

import torch
from fastapi import HTTPException
from models.rqvae import RQVAE


async def get_rqvae_indices_from_embeddings(
    embeddings_tensor: torch.Tensor, current_rqvae_model: RQVAE
) -> list[list[int]]:
    """从嵌入向量中获取 RQ-VAE 量化索引的辅助函数。

    Args:
        embeddings_tensor: 包含一个或多个嵌入向量的 PyTorch 张量。
                           如果是一维张量，代表单个嵌入；
                           如果是二维张量，代表一批嵌入 (batch_size, embedding_dim)。
        current_rqvae_model: 当前使用的 RQ-VAE 模型实例。

    Returns:
        一个整数列表的列表，其中每个内部列表包含对应输入嵌入的 RQ-VAE 量化索引。

    Raises:
        HTTPException: 如果输入嵌入的维度与 RQ-VAE 模型期望的输入维度不匹配。
    """
    if embeddings_tensor.dim() == 1:  # 单个嵌入向量
        embeddings_tensor = embeddings_tensor.unsqueeze(0)  # 增加批处理维度

    # 确保嵌入维度与 RQ-VAE 模型的输入维度匹配
    if embeddings_tensor.shape[1] != current_rqvae_model.in_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Input embedding dimension ({embeddings_tensor.shape[1]}) "
            f"does not match RQVAE model's expected input dimension ({current_rqvae_model.in_dim}).",
        )

    # 使用 RQ-VAE 模型获取量化索引
    # use_sk=False 表示不使用 scikit-learn 的 KMeans 进行量化，而是使用模型内部的码本
    indices = current_rqvae_model.get_indices(embeddings_tensor, use_sk=False)
    # 将索引张量调整形状为 (batch_size, num_quantizers)，然后转移到 CPU 并转换为 NumPy 数组
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    return indices.tolist()  # 转换为 Python 列表

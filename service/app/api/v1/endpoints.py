# -*- coding: utf-8 -*-
"""API 端点模块。

该模块定义了服务的所有 API 端点，
用于处理与 RQ-VAE 模型和文本嵌入相关的请求。
"""

import torch
from app.models.schemas import BatchTextItems, BatchVectorInput, SingleTextItem
from app.services.rqvae_processing import get_rqvae_indices_from_embeddings
from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.post("/predict")
async def predict_from_vectors(input_data: BatchVectorInput, request: Request):
    """从向量批量预测 RQ-VAE 索引。

    Args:
        input_data: 包含批量向量数据的请求体。
        request: FastAPI 请求对象，用于访问应用状态中的模型实例。

    Returns:
        一个包含每个输入向量对应 RQ-VAE 索引列表的字典。

    Raises:
        HTTPException: 如果输入数据格式无效、向量维度不一致或与模型期望不符，
                       或者在处理过程中发生其他错误。
    """
    try:
        rqvae_model_instance = request.app.state.rqvae_model  # 获取 RQ-VAE 模型实例

        # 校验输入数据格式
        if (
            not input_data.data
            or not isinstance(input_data.data, list)
            or not all(isinstance(vec, list) for vec in input_data.data)
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid input data format. "
                    "Expected a list of vectors (list of lists of floats)."
                ),
            )

        if input_data.data:  # 确保输入数据非空
            # 校验向量维度一致性及与模型输入维度匹配性
            first_vec_dim = len(input_data.data[0])
            if not all(len(vec) == first_vec_dim for vec in input_data.data):
                raise HTTPException(
                    status_code=400,
                    detail="All vectors in the batch must have the same dimension.",
                )
            if first_vec_dim != rqvae_model_instance.in_dim:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Input vector dimension ({first_vec_dim}) does not match "
                        f"RQVAE model's expected input dimension ({rqvae_model_instance.in_dim})."
                    ),
                )

        data_tensor = torch.tensor(
            input_data.data, dtype=torch.float32
        )  # 将输入数据转换为 PyTorch 张量
        # 调用服务函数获取 RQ-VAE 索引
        indices_list = await get_rqvae_indices_from_embeddings(
            data_tensor, rqvae_model_instance
        )
        return {"indices": indices_list}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing vectors: {str(e)}"
        )


@router.post("/embed_text_single")
async def embed_single_text(item: SingleTextItem, request: Request):
    """嵌入单个文本并获取其 RQ-VAE 索引。

    Args:
        item: 包含单个文本的请求体。
        request: FastAPI 请求对象，用于访问应用状态中的模型实例。

    Returns:
        一个包含单个文本对应 RQ-VAE 索引列表的字典。

    Raises:
        HTTPException: 如果在处理过程中发生错误。
    """
    try:
        text_embedding_model_instance = (
            request.app.state.text_embedding_model
        )  # 获取文本嵌入模型实例
        rqvae_model_instance = request.app.state.rqvae_model  # 获取 RQ-VAE 模型实例

        embedding_np = text_embedding_model_instance.encode(
            item.text
        )  # 对文本进行编码得到嵌入向量
        embedding_tensor = torch.tensor(
            embedding_np, dtype=torch.float32
        )  # 将嵌入向量转换为 PyTorch 张量

        # 调用服务函数获取 RQ-VAE 索引
        indices_list = await get_rqvae_indices_from_embeddings(
            embedding_tensor, rqvae_model_instance
        )
        return {"indices": indices_list[0] if indices_list else []}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing single text item: {str(e)}"
        )


@router.post("/embed_texts_batch")
async def embed_batch_texts(items: BatchTextItems, request: Request):
    """批量嵌入文本并获取它们的 RQ-VAE 索引。

    Args:
        items: 包含批量文本的请求体。
        request: FastAPI 请求对象，用于访问应用状态中的模型实例。

    Returns:
        一个包含每个文本对应 RQ-VAE 索引列表的字典。

    Raises:
        HTTPException: 如果在处理过程中发生错误。
    """
    try:
        text_embedding_model_instance = (
            request.app.state.text_embedding_model
        )  # 获取文本嵌入模型实例
        rqvae_model_instance = request.app.state.rqvae_model  # 获取 RQ-VAE 模型实例

        if not items.texts:  # 如果文本列表为空，直接返回空结果
            return {"indices": []}

        embeddings_np = text_embedding_model_instance.encode(
            items.texts
        )  # 对批量文本进行编码得到嵌入向量
        embeddings_tensor = torch.tensor(
            embeddings_np, dtype=torch.float32
        )  # 将嵌入向量转换为 PyTorch 张量

        # 调用服务函数获取 RQ-VAE 索引
        indices_list = await get_rqvae_indices_from_embeddings(
            embeddings_tensor, rqvae_model_instance
        )
        return {"indices": indices_list}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing batch text items: {str(e)}"
        )

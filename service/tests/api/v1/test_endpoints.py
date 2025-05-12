# -*- coding: utf-8 -*-
"""API 端点测试模块。"""

import pytest
from httpx import AsyncClient
from fastapi import FastAPI
from unittest.mock import AsyncMock, MagicMock

# 假设 FastAPI 应用实例在 app.main 中定义为 app
# from app.main import app  # 取消注释并根据您的项目结构进行调整
# 假设 schemas 在 app.models.schemas 中定义
# from app.models.schemas import BatchVectorInput, SingleTextItem, BatchTextItems # 取消注释并根据您的项目结构进行调整

# 创建一个 FastAPI app 实例用于测试
# 这通常是你主应用的一个简化版本，或者直接导入主应用
# 为简单起见，我们在这里创建一个新的 FastAPI 实例，并手动挂载路由
# 在实际项目中，您可能会导入并使用您的主 FastAPI 应用实例

# 模拟模型实例，因为它们是在应用启动时加载的
mock_rqvae_model = MagicMock()
mock_rqvae_model.in_dim = 128  # 假设模型的输入维度是128
mock_rqvae_model.encode_np.return_value = [[1,2,3]] # 模拟返回值

mock_text_embedding_model = MagicMock()
mock_text_embedding_model.encode.return_value = [[0.1, 0.2, 0.3]] # 模拟返回值

# 模拟 get_rqvae_indices_from_embeddings 函数
async def mock_get_rqvae_indices_from_embeddings(embeddings, model_instance):
    if embeddings.ndim == 1: # single embedding
        return [[1, 2, 3]]
    return [[i, i+1, i+2] for i in range(embeddings.shape[0])]


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"

@pytest.fixture(scope="module")
async def test_app() -> FastAPI:
    from app.api.v1.endpoints import router as api_router # 导入您的路由
    from app.services import rqvae_processing # 导入包含 get_rqvae_indices_from_embeddings 的模块

    app = FastAPI()
    app.include_router(api_router, prefix="/api/v1")

    # 替换真实的模型实例和处理函数为模拟对象
    app.state.rqvae_model = mock_rqvae_model
    app.state.text_embedding_model = mock_text_embedding_model
    rqvae_processing.get_rqvae_indices_from_embeddings = AsyncMock(side_effect=mock_get_rqvae_indices_from_embeddings)
    return app

@pytest.fixture(scope="module")
async def client(test_app: FastAPI) -> AsyncClient:
    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        yield ac

# 测试 /api/v1/predict 端点
@pytest.mark.anyio
async def test_predict_from_vectors_valid_input(client: AsyncClient):
    """测试 /predict 端点 - 有效输入"""
    payload = {"data": [[0.1] * 128, [0.2] * 128]} # 假设维度为128
    response = await client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    response_data = response.json()
    assert "indices" in response_data
    assert isinstance(response_data["indices"], list)
    assert len(response_data["indices"]) == 2
    assert response_data["indices"][0] == [0, 1, 2]
    assert response_data["indices"][1] == [1, 2, 3]

@pytest.mark.anyio
async def test_predict_from_vectors_invalid_format(client: AsyncClient):
    """测试 /predict 端点 - 无效输入格式"""
    payload = {"data": "not_a_list"}
    response = await client.post("/api/v1/predict", json=payload)
    assert response.status_code == 400
    assert "Invalid input data format" in response.json()["detail"]

@pytest.mark.anyio
async def test_predict_from_vectors_empty_data(client: AsyncClient):
    """测试 /predict 端点 - 空数据列表"""
    payload = {"data": []}
    response = await client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200 # 行为是返回空列表
    assert response.json() == {"indices": []}

@pytest.mark.anyio
async def test_predict_from_vectors_mismatched_dimensions(client: AsyncClient):
    """测试 /predict 端点 - 向量维度不一致"""
    payload = {"data": [[0.1] * 128, [0.2] * 64]}
    response = await client.post("/api/v1/predict", json=payload)
    assert response.status_code == 400
    assert "All vectors in the batch must have the same dimension" in response.json()["detail"]

@pytest.mark.anyio
async def test_predict_from_vectors_incorrect_dimension(client: AsyncClient):
    """测试 /predict 端点 - 向量维度与模型不匹配"""
    payload = {"data": [[0.1] * 64, [0.2] * 64]} # 假设模型期望128
    response = await client.post("/api/v1/predict", json=payload)
    assert response.status_code == 400
    assert "Input vector dimension (64) does not match RQVAE model's expected input dimension (128)" in response.json()["detail"]

# 测试 /api/v1/embed_text_single 端点
@pytest.mark.anyio
async def test_embed_single_text_valid_input(client: AsyncClient):
    """测试 /embed_text_single 端点 - 有效输入"""
    payload = {"text": "这是一个测试文本"}
    response = await client.post("/api/v1/embed_text_single", json=payload)
    assert response.status_code == 200
    response_data = response.json()
    assert "indices" in response_data
    assert isinstance(response_data["indices"], list)
    assert response_data["indices"] == [1, 2, 3] # 基于模拟的 get_rqvae_indices_from_embeddings

@pytest.mark.anyio
async def test_embed_single_text_empty_input(client: AsyncClient):
    """测试 /embed_text_single 端点 - 空文本输入 (预期行为依赖于模型)
       这里假设模型会处理空字符串或 FastAPI 会因 pydantic 模型验证失败
    """
    payload = {"text": ""} # Pydantic模型通常会要求非空字符串，除非特别配置
    # 实际行为取决于 SingleTextItem schema 定义
    # 如果 text 字段不能为空，FastAPI 会返回 422 Unprocessable Entity
    # 此处我们假设它能接受空字符串并由模型处理
    response = await client.post("/api/v1/embed_text_single", json=payload)
    assert response.status_code == 200 # 或 422，取决于模型定义
    # 如果是200，检查返回的索引是否符合预期（例如空列表或特定错误指示）
    # assert response.json() == {"indices": []} # 示例，取决于模型如何处理空文本

# 测试 /api/v1/embed_texts_batch 端点
@pytest.mark.anyio
async def test_embed_texts_batch_valid_input(client: AsyncClient):
    """测试 /embed_texts_batch 端点 - 有效输入"""
    payload = {"texts": ["文本一", "文本二"]}
    response = await client.post("/api/v1/embed_texts_batch", json=payload)
    assert response.status_code == 200
    response_data = response.json()
    assert "indices" in response_data
    assert isinstance(response_data["indices"], list)
    assert len(response_data["indices"]) == 2
    assert response_data["indices"][0] == [0, 1, 2]
    assert response_data["indices"][1] == [1, 2, 3]

@pytest.mark.anyio
async def test_embed_texts_batch_empty_list(client: AsyncClient):
    """测试 /embed_texts_batch 端点 - 空文本列表"""
    payload = {"texts": []}
    response = await client.post("/api/v1/embed_texts_batch", json=payload)
    assert response.status_code == 200
    assert response.json() == {"indices": []}

@pytest.mark.anyio
async def test_embed_texts_batch_list_with_empty_string(client: AsyncClient):
    """测试 /embed_texts_batch 端点 - 列表中包含空字符串"""
    payload = {"texts": ["有效文本", ""]}
    # 行为取决于模型如何处理列表中的空字符串
    response = await client.post("/api/v1/embed_texts_batch", json=payload)
    assert response.status_code == 200 # 或 422，取决于模型定义
    # 示例断言，具体取决于模型行为
    # response_data = response.json()
    # assert len(response_data["indices"]) == 2 
    # assert response_data["indices"][0] == [0,1,2]
    # assert response_data["indices"][1] == [] # 或者其他错误指示

# 注意：
# 1. 上述测试假设了 `app.main.app` 是您的 FastAPI 应用实例。
#    如果不是，请相应地调整 `test_app` fixture 中的应用导入和创建。
# 2. `BatchVectorInput`, `SingleTextItem`, `BatchTextItems` 的具体结构（例如字段是否允许为空）
#    会影响某些错误情况的测试（例如空文本输入）。测试应根据实际的 schema 定义进行调整。
# 3. 模型模拟 (`mock_rqvae_model`, `mock_text_embedding_model`) 和
#    `mock_get_rqvae_indices_from_embeddings` 的返回值应尽可能接近真实模型的行为，
#    或者至少覆盖您想要测试的逻辑路径。
# 4. 您需要安装 `pytest`, `httpx`, `pytest-asyncio` (如果使用 pytest < 7.0) 和 `anyio`。
#    `pip install pytest httpx "pytest-asyncio>=0.17.0" anyio`
# 5. 要运行测试，请在 `service` 目录下执行 `pytest` 命令。
#    确保 `PYTHONPATH` 设置正确，以便 Python 可以找到您的应用模块，例如：
#    `PYTHONPATH=. pytest tests/api/v1/test_endpoints.py` 或在项目根目录运行 `pytest`。
# 6. 对于 `/embed_text_single` 和 `/embed_texts_batch` 中空字符串的处理，
#    如果 Pydantic 模型不允许空字符串 (e.g. `text: str = Field(..., min_length=1)`),
#    FastAPI 会在请求到达端点逻辑之前返回 422 Unprocessable Entity。
#    测试用例需要反映这种行为。
#    如果允许空字符串，则测试应验证模型或服务层如何处理它们。
#    当前测试假设空字符串可以传递给模型层。
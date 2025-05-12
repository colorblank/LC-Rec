**API 文档：**

FastAPI 会自动根据您的代码（包括 Pydantic 模型和路径操作函数的文档字符串）生成 OpenAPI 规范的 API 文档。

1.  **启动您的 FastAPI 应用**：
    通常，您会有一个 `main.py` 文件来启动 Uvicorn 服务器。例如，如果您在 `service` 目录下，并且您的主应用实例在 `app/main.py` 中定义为 `app`，您可能会这样启动它：

    ```bash
    uvicorn app.main:app --reload
    ```

    请根据您的项目结构调整命令。

2.  **访问自动生成的文档**：
    应用启动后，您可以在浏览器中访问以下 URL 来查看 API 文档：

    - **Swagger UI**: `http://127.0.0.1:8000/docs` (假设您的应用运行在 `http://120.0.1:8000`)
    - **ReDoc**: `http://127.0.0.1:8000/redoc`

    这些页面提供了交互式的 API 文档，您可以在其中查看所有端点、它们的参数、请求体、响应模型，并直接从浏览器发送测试请求。

    <mcfile name="endpoints.py" path="～/LC-Rec/service/app/api/v1/endpoints.py"></mcfile> 文件中的文档字符串（docstrings）已经比较详细，这将有助于生成清晰的 API 文档。

请检查新创建的 <mcfile name="test_endpoints.py" path="～/LC-Rec/service/tests/api/v1/test_endpoints.py"></mcfile> 文件，并根据您的具体模型行为和 Pydantic schema 定义调整模拟对象的返回值和某些测试用例的断言。例如，对于空文本输入，实际行为（返回 200 还是 422）取决于 `SingleTextItem` 和 `BatchTextItems` schema 中 `text` 和 `texts` 字段的约束。

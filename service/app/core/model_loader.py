import torch
from sentence_transformers import SentenceTransformer

# 假设 models.rqvae 可以从这个路径访问
from models.rqvae import RQVAE
from .config import settings, ConfigurationError


def load_text_embedding_model():
    """根据配置加载文本嵌入模型。

    Raises:
        ConfigurationError: 如果加载文本嵌入模型失败。
        RuntimeError: 如果无法确定文本嵌入模型的输出维度。

    Returns:
        tuple: 包含加载的文本嵌入模型和其输出维度的元组。
    """
    try:
        # 尝试加载在配置中指定的文本嵌入模型
        text_embedding_model = SentenceTransformer(settings.TEXT_EMBEDDING_MODEL_NAME)
    except Exception as e:
        raise ConfigurationError(
            f"加载文本嵌入模型失败 '{settings.TEXT_EMBEDDING_MODEL_NAME}': {str(e)}"
        )
    print(f"文本嵌入模型 '{settings.TEXT_EMBEDDING_MODEL_NAME}' 已加载。")

    # 获取模型的输出维度
    text_model_output_dim = text_embedding_model.get_sentence_embedding_dimension()
    if text_model_output_dim is None:
        try:
            # 如果无法直接获取维度，尝试通过编码一个虚拟句子来获取
            dummy_embedding = text_embedding_model.encode("test")
            text_model_output_dim = dummy_embedding.shape[0]
        except Exception as e:
            raise RuntimeError(
                f"无法确定文本嵌入模型的输出维度 "
                f"'{settings.TEXT_EMBEDDING_MODEL_NAME}': {str(e)}"
            )

    print(f"文本嵌入模型输出维度: {text_model_output_dim}")
    return text_embedding_model, text_model_output_dim


def load_rqvae_model(in_dimension: int):
    """根据配置和给定的输入维度加载 RQVAE 模型。

    Args:
        in_dimension: RQVAE 模型的输入维度。

    Raises:
        ConfigurationError: 如果 RQVAE 检查点文件未找到。
        RuntimeError: 如果加载 RQVAE 检查点时发生错误。

    Returns:
        RQVAE: 加载的 RQVAE 模型。
    """
    try:
        # 加载 RQVAE 模型检查点
        ckpt = torch.load(settings.RQVAE_CKPT_PATH, map_location=torch.device("cpu"))
    except FileNotFoundError:
        raise ConfigurationError(
            f"在 {settings.RQVAE_CKPT_PATH} 未找到 RQVAE 检查点文件。请确保路径正确。"
        )
    except Exception as e:
        raise RuntimeError(
            f"从 {settings.RQVAE_CKPT_PATH} 加载 RQVAE 检查点时出错: {str(e)}"
        )

    args = ckpt["args"]  # 模型参数
    state_dict = ckpt["state_dict"]  # 模型状态字典

    # 初始化 RQVAE 模型
    rqvae_model = RQVAE(
        in_dim=in_dimension,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )
    # 加载模型状态
    rqvae_model.load_state_dict(state_dict)
    # 设置为评估模式
    rqvae_model.eval()
    print(f"RQVAE 模型已加载。预期输入维度: {rqvae_model.in_dim}")

    return rqvae_model

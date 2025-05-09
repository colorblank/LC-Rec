import os
import torch
from sentence_transformers import SentenceTransformer
from models.rqvae import RQVAE  # Assuming models.rqvae is accessible from this path
from .config import settings, ConfigurationError


def load_text_embedding_model():
    """Loads the Text Embedding model based on configuration."""
    try:
        text_embedding_model = SentenceTransformer(settings.TEXT_EMBEDDING_MODEL_NAME)
    except Exception as e:
        raise ConfigurationError(
            f"Failed to load text embedding model '{settings.TEXT_EMBEDDING_MODEL_NAME}': {str(e)}"
        )
    print(f"Text embedding model '{settings.TEXT_EMBEDDING_MODEL_NAME}' loaded.")

    text_model_output_dim = text_embedding_model.get_sentence_embedding_dimension()
    if text_model_output_dim is None:
        try:
            # Attempt to get dimension by encoding a dummy sentence if not directly available
            dummy_embedding = text_embedding_model.encode("test")
            text_model_output_dim = dummy_embedding.shape[0]
        except Exception as e:
            raise RuntimeError(
                f"Could not determine output dimension of text embedding model '{settings.TEXT_EMBEDDING_MODEL_NAME}': {str(e)}"
            )

    print(f"Text embedding model output dimension: {text_model_output_dim}")
    return text_embedding_model, text_model_output_dim


def load_rqvae_model(in_dimension: int):
    """Loads the RQVAE model based on configuration and a given input dimension."""
    try:
        ckpt = torch.load(settings.RQVAE_CKPT_PATH, map_location=torch.device("cpu"))
    except FileNotFoundError:
        raise ConfigurationError(
            f"RQVAE Checkpoint file not found at {settings.RQVAE_CKPT_PATH}. Please ensure the path is correct."
        )
    except Exception as e:
        raise RuntimeError(
            f"Error loading RQVAE checkpoint from {settings.RQVAE_CKPT_PATH}: {str(e)}"
        )

    args = ckpt["args"]
    state_dict = ckpt["state_dict"]

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
    rqvae_model.load_state_dict(state_dict)
    rqvae_model.eval()
    print(f"RQVAE model loaded. Expected input dimension: {rqvae_model.in_dim}")

    return rqvae_model

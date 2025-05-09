import torch
from fastapi import HTTPException
from models.rqvae import RQVAE # Assuming models.rqvae is accessible

async def get_rqvae_indices_from_embeddings(
    embeddings_tensor: torch.Tensor, current_rqvae_model: RQVAE
) -> list[list[int]]:
    """Helper function to get RQVAE indices from embeddings."""
    if embeddings_tensor.dim() == 1:  # Single embedding
        embeddings_tensor = embeddings_tensor.unsqueeze(0)  # Add batch dimension

    # Ensure embedding dimension matches RQVAE model's input dimension
    if embeddings_tensor.shape[1] != current_rqvae_model.in_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Input embedding dimension ({embeddings_tensor.shape[1]}) "
                   f"does not match RQVAE model's expected input dimension ({current_rqvae_model.in_dim}).",
        )

    indices = current_rqvae_model.get_indices(embeddings_tensor, use_sk=False)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    return indices.tolist()
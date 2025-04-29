from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import kmeans, sinkhorn_algorithm


class VectorQuantizer(nn.Module):
    """Vector Quantizer module.

    This module implements the Vector Quantization (VQ) technique.
    It maintains a codebook of embedding vectors and maps input tensors
    to the closest embedding vector in the codebook.

    Args:
        n_e (int): Number of embeddings in the codebook.
        e_dim (int): Dimension of the embedding vectors.
        beta (float, optional): Commitment loss weight. Defaults to 0.25.
        kmeans_init (bool, optional): Whether to initialize embeddings using K-Means.
            Defaults to False.
        kmeans_iters (int, optional): Number of iterations for K-Means initialization.
            Defaults to 10.
        sk_epsilon (float, optional): Epsilon value for Sinkhorn-Knopp algorithm.
            If <= 0, Sinkhorn-Knopp is not used. Defaults to 0.003.
        sk_iters (int, optional): Number of iterations for Sinkhorn-Knopp algorithm.
            Defaults to 100.
    """

    def __init__(
        self,
        n_e: int,
        e_dim: int,
        beta: float = 0.25,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        sk_epsilon: float = 0.003,
        sk_iters: int = 100,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self) -> torch.Tensor:
        """Returns the codebook embeddings.

        Returns:
            torch.Tensor: The embedding weights tensor.
        """
        return self.embedding.weight

    def get_codebook_entry(
        self, indices: torch.Tensor, shape: Optional[Tuple] = None
    ) -> torch.Tensor:
        """Gets the quantized latent vectors corresponding to the indices.

        Args:
            indices (torch.Tensor): Indices of the codebook entries.
            shape (Optional[Tuple], optional): Desired shape for the output tensor.
                Defaults to None.

        Returns:
            torch.Tensor: The quantized latent vectors.
        """
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data: torch.Tensor):
        """Initializes the codebook embeddings using K-Means.

        Args:
            data (torch.Tensor): Input data to cluster for K-Means.
        """
        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )

        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances: torch.Tensor) -> torch.Tensor:
        """Centers the distances for Sinkhorn-Knopp constraint.

        Args:
            distances (torch.Tensor): Input distances (B, K).

        Returns:
            torch.Tensor: Centered distances.
        """
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(
        self, x: torch.Tensor, use_sk: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the Vector Quantizer.

        Args:
            x (torch.Tensor): Input tensor.
            use_sk (bool, optional): Whether to use Sinkhorn-Knopp algorithm for assignment.
                Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Quantized output tensor (x_q).
                - VQ loss.
                - Indices of the selected codebook entries.
        """
        # Flatten input
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the L2 Norm between latent and Embedded weights
        d = (
            torch.sum(latent**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
            - 2 * torch.matmul(latent, self.embedding.weight.t())
        )
        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d_centered = self.center_distance_for_constraint(d)
            d_centered = d_centered.double()
            Q = sinkhorn_algorithm(d_centered, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print("Sinkhorn Algorithm returns nan/inf values.")
                # Fallback to argmin if Sinkhorn fails
                indices = torch.argmin(d, dim=-1)
            else:
                indices = torch.argmax(Q, dim=-1)

        # indices = torch.argmin(d, dim=-1) # Original argmin kept commented

        x_q = self.embedding(indices).view(x.shape)

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices

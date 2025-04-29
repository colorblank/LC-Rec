from typing import List, Tuple

import torch
import torch.nn as nn

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer (RVQ).

    Implements the residual vector quantization technique proposed in the
    SoundStream paper. It uses multiple quantizers sequentially, where each
    quantizer processes the residual from the previous one.

    References:
        SoundStream: An End-to-End Neural Audio Codec
        https://arxiv.org/pdf/2107.03312.pdf

    Args:
        n_e_list (List[int]): List containing the number of embeddings (codebook size)
            for each quantizer.
        e_dim (int): The dimensionality of the embeddings.
        sk_epsilons (List[float]): List of epsilon values for Sinkhorn-Knopp
            algorithm for each quantizer.
        beta (float, optional): Commitment loss weight. Defaults to 0.25.
        kmeans_init (bool, optional): Whether to initialize embeddings using K-Means.
            Defaults to False.
        kmeans_iters (int, optional): Number of K-Means iterations if `kmeans_init`
            is True. Defaults to 100.
        sk_iters (int, optional): Number of Sinkhorn-Knopp iterations.
            Defaults to 100.
    """

    def __init__(
        self,
        n_e_list: List[int],
        e_dim: int,
        sk_epsilons: List[float],
        beta: float = 0.25,
        kmeans_init: bool = False,
        kmeans_iters: int = 100,
        sk_iters: int = 100,
    ):
        super().__init__()
        self.n_e_list: List[int] = n_e_list
        self.e_dim: int = e_dim
        self.num_quantizers: int = len(n_e_list)
        self.beta: float = beta
        self.kmeans_init: bool = kmeans_init
        self.kmeans_iters: int = kmeans_iters
        self.sk_epsilons: List[float] = sk_epsilons
        self.sk_iters: int = sk_iters
        self.vq_layers: nn.ModuleList = nn.ModuleList(
            [
                VectorQuantizer(
                    n_e,
                    e_dim,
                    beta=self.beta,
                    kmeans_init=self.kmeans_init,
                    kmeans_iters=self.kmeans_iters,
                    sk_epsilon=sk_epsilon,
                    sk_iters=sk_iters,
                )
                for n_e, sk_epsilon in zip(n_e_list, sk_epsilons)
            ]
        )

    def get_codebook(self) -> torch.Tensor:
        """Retrieves the codebooks from all quantizer layers.

        Returns:
            torch.Tensor: A tensor containing the stacked codebooks from each
                quantizer layer. Shape: (num_quantizers, n_e, e_dim).
        """
        all_codebook: List[torch.Tensor] = []
        for quantizer in self.vq_layers:
            codebook: torch.Tensor = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(
        self, x: torch.Tensor, use_sk: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the Residual Vector Quantizer.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch_size, ..., e_dim).
            use_sk (bool, optional): Whether to use Sinkhorn-Knopp algorithm for
                assignment. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - x_q (torch.Tensor): The quantized output tensor, which is the sum
                  of quantized vectors from all layers.
                  Shape: (batch_size, ..., e_dim).
                - mean_losses (torch.Tensor): The mean quantization loss across all
                  layers.
                - all_indices (torch.Tensor): The indices of the chosen codebook vectors
                  for each quantizer layer. Shape: (batch_size, ..., num_quantizers).
        """
        all_losses: List[torch.Tensor] = []
        all_indices: List[torch.Tensor] = []

        x_q: torch.Tensor = torch.zeros_like(x)
        residual: torch.Tensor = x
        for quantizer in self.vq_layers:
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses: torch.Tensor = torch.stack(all_losses).mean()
        # Ensure indices are stacked correctly, assuming indices shape is (batch_size, ...)
        # Stack along a new last dimension
        all_indices_tensor: torch.Tensor = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices_tensor

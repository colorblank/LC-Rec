from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


class RQVAE(nn.Module):
    """Residual Quantization Variational Autoencoder (RQ-VAE).

    Combines a standard VAE architecture with a Residual Vector Quantizer (RVQ)
    in the latent space.

    Args:
        in_dim (int): Dimensionality of the input features.
        num_emb_list (Optional[List[int]]): List containing the number of embeddings
            (codebook size) for each residual quantizer. Defaults to None.
        e_dim (int): Dimensionality of the latent embeddings.
        layers (Optional[List[int]]): List defining the hidden layer sizes for the
            encoder and decoder MLPs. Defaults to None.
        dropout_prob (float): Dropout probability for MLP layers. Defaults to 0.0.
        bn (bool): Whether to use Batch Normalization in MLP layers. Defaults to False.
        loss_type (str): Type of reconstruction loss ('mse' or 'l1').
            Defaults to "mse".
        quant_loss_weight (float): Weight for the quantization loss component.
            Defaults to 1.0.
        beta (float): Commitment loss weight for the VQ layers. Defaults to 0.25.
        kmeans_init (bool): Whether to initialize VQ embeddings using K-Means.
            Defaults to False.
        kmeans_iters (int): Number of K-Means iterations if kmeans_init is True.
            Defaults to 100.
        sk_epsilons (Optional[List[float]]): List of epsilon values for Sinkhorn-Knopp
            algorithm for each quantizer. Defaults to None.
        sk_iters (int): Number of Sinkhorn-Knopp iterations. Defaults to 100.
    """

    def __init__(
        self,
        in_dim: int = 768,
        num_emb_list: Optional[List[int]] = None,
        e_dim: int = 64,
        layers: Optional[List[int]] = None,
        dropout_prob: float = 0.0,
        bn: bool = False,
        loss_type: str = "mse",
        quant_loss_weight: float = 1.0,
        beta: float = 0.25,
        kmeans_init: bool = False,
        kmeans_iters: int = 100,
        sk_epsilons: Optional[List[float]] = None,
        sk_iters: int = 100,
    ):
        super(RQVAE, self).__init__()

        # Default values if None
        if num_emb_list is None:
            num_emb_list = [256, 256, 256, 256]
        if layers is None:
            layers = [512, 256, 128]
        if sk_epsilons is None:
            # Default sk_epsilons based on the length of num_emb_list
            sk_epsilons = [0.0] * len(num_emb_list)
            if len(num_emb_list) > 2:
                sk_epsilons[-2] = 0.003
            if len(num_emb_list) > 3:
                sk_epsilons[-1] = 0.01

        self.in_dim: int = in_dim
        self.num_emb_list: List[int] = num_emb_list
        self.e_dim: int = e_dim
        self.layers: List[int] = layers
        self.dropout_prob: float = dropout_prob
        self.bn: bool = bn
        self.loss_type: str = loss_type
        self.quant_loss_weight: float = quant_loss_weight
        self.beta: float = beta
        self.kmeans_init: bool = kmeans_init
        self.kmeans_iters: int = kmeans_iters
        self.sk_epsilons: List[float] = sk_epsilons
        self.sk_iters: int = sk_iters

        self.encode_layer_dims: List[int] = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder: MLPLayers = MLPLayers(
            layers=self.encode_layer_dims, dropout=self.dropout_prob, bn=self.bn
        )

        self.rq: ResidualVectorQuantizer = ResidualVectorQuantizer(
            self.num_emb_list,
            self.e_dim,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
        )

        self.decode_layer_dims: List[int] = self.encode_layer_dims[::-1]
        self.decoder: MLPLayers = MLPLayers(
            layers=self.decode_layer_dims, dropout=self.dropout_prob, bn=self.bn
        )

    def forward(
        self, x: torch.Tensor, use_sk: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the RQ-VAE.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch_size, ..., in_dim).
            use_sk (bool): Whether to use Sinkhorn-Knopp in the RVQ layer.
                Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - out (torch.Tensor): Reconstructed output tensor.
                  Shape: (batch_size, ..., in_dim).
                - rq_loss (torch.Tensor): The quantization loss from the RVQ layer.
                - indices (torch.Tensor): The indices of the chosen codebook vectors
                  from the RVQ layer. Shape: (batch_size, ..., num_quantizers).
        """
        x_encoded: torch.Tensor = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x_encoded, use_sk=use_sk)
        out: torch.Tensor = self.decoder(x_q)

        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs: torch.Tensor, use_sk: bool = False) -> torch.Tensor:
        """Encodes input and returns the quantization indices without gradients.

        Args:
            xs (torch.Tensor): Input tensor. Shape: (batch_size, ..., in_dim).
            use_sk (bool): Whether to use Sinkhorn-Knopp in the RVQ layer.
                Defaults to False.

        Returns:
            torch.Tensor: The indices of the chosen codebook vectors from the RVQ
                layer. Shape: (batch_size, ..., num_quantizers).
        """
        x_e: torch.Tensor = self.encoder(xs)
        _, _, indices = self.rq(x_e, use_sk=use_sk)
        return indices

    def compute_loss(
        self, out: torch.Tensor, quant_loss: torch.Tensor, xs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the total loss for the RQ-VAE.

        Args:
            out (torch.Tensor): The reconstructed output from the decoder.
                Shape: (batch_size, ..., in_dim).
            quant_loss (torch.Tensor): The quantization loss from the RVQ layer.
            xs (torch.Tensor): The original input tensor.
                Shape: (batch_size, ..., in_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - loss_total (torch.Tensor): The total combined loss (reconstruction +
                  weighted quantization loss).
                - loss_recon (torch.Tensor): The reconstruction loss component.

        Raises:
            ValueError: If an unsupported loss_type is specified.
        """
        if self.loss_type == "mse":
            loss_recon: torch.Tensor = F.mse_loss(out, xs, reduction="mean")
        elif self.loss_type == "l1":
            loss_recon: torch.Tensor = F.l1_loss(out, xs, reduction="mean")
        else:
            raise ValueError(f"Incompatible loss type: {self.loss_type}")

        loss_total: torch.Tensor = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon

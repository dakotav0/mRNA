"""
CBSAE — Concept Bottleneck Sparse Autoencoder
Router Node (promoted from sandbox-scripts/sae_routing.py)

Separates polysemantic LLM activations into a sparse feature space where
specific dimensions are supervision-aligned to user-defined concepts (adapters).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBSAE(nn.Module):
    """
    Concept Bottleneck Sparse Autoencoder.

    Encodes dense residual-stream activations (batch, [seq], d_model) into a
    wide sparse feature vector (d_sae = d_model * expansion_factor).  The first
    `bottleneck_features` dimensions are forced — via CB loss during training —
    to correspond to specific user-defined concepts (one per .mrna adapter).

    At inference, only the encoder is used:
        sparse = ReLU(encoder(activations))
        concept_idx = sparse[:, :n_concepts].argmax(dim=-1)
    """

    def __init__(
        self,
        d_model: int = 2048,
        expansion_factor: int = 8,
        bottleneck_features: int = 10,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_model * expansion_factor
        self.bottleneck_features = bottleneck_features

        self.encoder = nn.Linear(self.d_model, self.d_sae, bias=True)
        self.decoder = nn.Linear(self.d_sae, self.d_model, bias=True)

        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def forward(self, residual_activations: torch.Tensor):
        sparse_features = F.relu(self.encoder(residual_activations))
        reconstructed = self.decoder(sparse_features)
        return reconstructed, sparse_features

    def compute_loss(self, original_activations: torch.Tensor, labels=None):
        """
        Three-part loss:
          1. MSE reconstruction — preserves the LLM's thought process
          2. L1 sparsity penalty — forces most features to zero
          3. Concept bottleneck — aligns first `bottleneck_features` dims to labels
        """
        reconstructed, sparse_features = self(original_activations)

        mse_loss = F.mse_loss(reconstructed, original_activations)
        l1_loss = torch.sum(torch.abs(sparse_features))

        cb_loss = 0.0
        if labels is not None:
            cb_loss = F.binary_cross_entropy_with_logits(
                sparse_features[:, :, : self.bottleneck_features],
                labels,
            )

        return mse_loss + (0.01 * l1_loss) + (1.0 * cb_loss), sparse_features

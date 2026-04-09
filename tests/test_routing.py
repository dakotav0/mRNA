import pytest
import torch

from mrna.router.sae import CBSAE


def test_cbsae_dimensions():
    """Verifies that CBSAE initializes with correct internal dimensions."""
    d_model = 128
    expansion = 4
    n_concepts = 3

    sae = CBSAE(
        d_model=d_model, expansion_factor=expansion, bottleneck_features=n_concepts
    )

    # Encoder should map d_model to d_model * expansion
    assert sae.encoder.weight.shape == (d_model * expansion, d_model)
    # Decoder should map d_model * expansion back to d_model
    assert sae.decoder.weight.shape == (d_model, d_model * expansion)


def test_cbsae_forward_pass():
    """Verifies forward pass logic with synthetic tensors."""
    d_model = 32
    expansion = 2
    n_concepts = 2
    batch_size = 4

    sae = CBSAE(
        d_model=d_model, expansion_factor=expansion, bottleneck_features=n_concepts
    )
    x = torch.randn(batch_size, d_model)

    # We test the individual components like use in training
    pre_relu = sae.encoder(x)
    assert pre_relu.shape == (batch_size, d_model * expansion)

    # Botleneck corresponds to the first n_concepts features
    bottleneck = pre_relu[:, :n_concepts]
    assert bottleneck.shape == (batch_size, n_concepts)

    # Probabilities for routing
    probs = torch.softmax(bottleneck, dim=-1)
    assert probs.shape == (batch_size, n_concepts)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-6)


def test_sae_reconstruction_shape():
    """Verifies that reconstruction preserves the input shape."""
    d_model = 64
    sae = CBSAE(d_model=d_model)
    x = torch.randn(8, d_model)

    # Complete reconstruction path
    recon, sparse = sae(x)
    assert recon.shape == x.shape

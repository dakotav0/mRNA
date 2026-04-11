from unittest.mock import MagicMock, patch

import pytest
import torch

from mrna.factory.adapter import train_adapter
from mrna.factory.sae import train_sae_weights

# Import functions to test
from mrna.factory.sampler import harvest_activations


@patch("unsloth.FastLanguageModel")
@patch("datasets.load_dataset")
def test_sampler_orchestration(mock_load_ds, mock_flm):
    """Verifies that harvest_activations correctly sets up the model and dataset."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)

    mock_ds = MagicMock()
    mock_ds.__iter__.return_value = [{"text": "test"}]
    mock_ds.take.return_value = [{"text": "test"}]
    mock_load_ds.return_value = mock_ds

    # Bypass the deep model probing logic
    with patch("mrna.factory.sampler.ActivationInterceptor") as mock_interceptor:
        harvest_activations(
            concept="biology",
            dataset_id="test/ds",
            model_id="gemma-4-e2b",
            max_examples=1,
            skip_if_exists=False,
        )
        mock_flm.from_pretrained.assert_called()


@patch("trl.SFTConfig")
@patch("trl.SFTTrainer")
@patch("unsloth.is_bfloat16_supported")
@patch("unsloth.FastVisionModel")
@patch("unsloth.FastLanguageModel")
def test_adapter_training_orchestration(
    mock_flm, mock_fvm, mock_bf16, mock_trainer, mock_sft_config
):
    """Verifies that train_adapter initializes the SFTTrainer."""
    mock_bf16.return_value = False
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token = "</s>"
    mock_tokenizer.pad_token = None
    # Set eos_token on the inner tokenizer that _tok unwraps to
    mock_tokenizer.tokenizer.eos_token = "</s>"
    mock_tokenizer.tokenizer.pad_token = None
    # gemma-4-e2b uses FastVisionModel; both loaders return the same mock pair
    mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
    mock_fvm.from_pretrained.return_value = (mock_model, mock_tokenizer)

    # Mock dataset that supports the two-step map pipeline (format+truncate, then tokenize)
    mock_ds = MagicMock()
    mock_ds.__len__ = MagicMock(return_value=10)
    mock_ds.column_names = ["instruction", "input", "output"]
    mock_ds.map.return_value = mock_ds
    mock_ds.select.return_value = mock_ds

    # Both load_smart_dataset and get_dataset_formatter are imported inside the function
    # body, so patch them at the source module rather than at mrna.factory.adapter.*
    with patch("mrna.data.dataset_utils.load_smart_dataset") as mock_load:
        with patch("mrna.data.dataset_utils.get_dataset_formatter") as mock_fmt:
            mock_load.return_value = (mock_ds, False, "test/ds")
            mock_fmt.return_value = MagicMock(return_value={"text": ["test text"]})
            train_adapter(
                concept="biology",
                dataset_id="test/ds",
                model_id="gemma-4-e2b",
                output_dir="data/test_adapter",
            )
            mock_fvm.from_pretrained.assert_called()  # gemma-4 routes through FastVisionModel
            mock_trainer.assert_called()


def test_sae_training_stub():
    """Verifies that train_sae_weights processes tensors and saves results."""
    with patch("torch.load") as mock_load:
        with patch("torch.save") as mock_save:
            mock_load.return_value = torch.randn(10, 128)
            train_sae_weights(
                concepts=["test"], activation_files=["test.pt"], epochs=1, d_model=128
            )
            mock_save.assert_called()

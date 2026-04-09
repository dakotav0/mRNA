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


@patch("unsloth.is_bfloat16_supported")
@patch("mrna.factory.adapter.TrainingArguments")
@patch("unsloth.FastLanguageModel")
@patch("trl.SFTTrainer")
@patch("datasets.load_dataset")
@patch("mrna.factory.adapter.HFDataset")
def test_adapter_training_orchestration(
    mock_hf_ds, mock_load_ds, mock_trainer, mock_flm, mock_args, mock_bf16
):
    """Verifies that train_adapter initializes the SFTTrainer."""
    mock_bf16.return_value = False
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token = "</s>"
    mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)

    # Mocking dataset loading entirely
    mock_ds = MagicMock()
    mock_ds.take.return_value = [{"instruction": "hi", "input": "", "output": "hey"}]
    mock_load_ds.return_value = mock_ds

    # Bypass formatter probing which hits the network/pyarrow
    with patch("mrna.factory.adapter.get_dataset_formatter") as mock_formatter:
        mock_formatter.return_value = lambda x: x
        train_adapter(
            concept="biology",
            dataset_id="test/ds",
            model_id="gemma-4-e2b",
            output_dir="data/test_adapter",
        )
        mock_flm.from_pretrained.assert_called()
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

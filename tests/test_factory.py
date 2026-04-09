import pytest
import torch
from unittest.mock import MagicMock, patch
from mrna.factory.sampler import harvest_activations
from mrna.factory.adapter import train_adapter
from mrna.factory.sae import train_sae_weights

@patch("mrna.factory.sampler.FastLanguageModel")
@patch("mrna.factory.sampler.load_dataset")
def test_sampler_orchestration(mock_load_ds, mock_flm):
    """Verifies that harvest_activations correctly sets up model and dataset."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
    
    # Mock dataset to return a slice
    mock_ds = MagicMock()
    mock_ds.take.return_value = [{"text": "hello"}]
    mock_load_ds.return_value = mock_ds
    
    # We don't want to actually run the loop, so we'll mock the internal forward calls if needed
    # but here we just check if it initializes
    with patch("mrna.factory.sampler.ActivationInterceptor") as mock_interceptor:
        # We limit examples to something tiny
        harvest_activations(
            dataset_id="test/ds",
            model_id="unsloth/gemma-4-E2B-it",
            max_examples=1
        )
        
        mock_flm.from_pretrained.assert_called()
        mock_load_ds.assert_called_with("test/ds", split="train", streaming=True)

@patch("mrna.factory.adapter.FastLanguageModel")
@patch("mrna.factory.adapter.SFTTrainer")
def test_adapter_training_orchestration(mock_trainer, mock_flm):
    """Verifies that train_adapter initializes the SFTTrainer with LoRA."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
    mock_flm.get_peft_model.return_value = mock_model
    
    train_adapter(
        dataset_id="test/ds",
        model_id="unsloth/gemma-4-E2B-it",
        output_dir="data/test_adapter"
    )
    
    mock_flm.get_peft_model.assert_called()
    mock_trainer.assert_called()

@patch("torch.load")
@patch("torch.save")
def test_sae_training_stub(mock_save, mock_load):
    """Verifies that train_sae_weights processes tensors and saves results."""
    # Synthetic activations
    mock_load.return_value = torch.randn(10, 128)
    
    with patch("mrna.factory.sae.CBSAE") as mock_sae_class:
        mock_sae_instance = MagicMock()
        mock_sae_class.return_value = mock_sae_instance
        
        train_sae_weights(
            concepts=["test"],
            activation_files=["test.pt"],
            epochs=1,
            d_model=128
        )
        
        mock_save.assert_called()

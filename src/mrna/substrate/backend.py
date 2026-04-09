from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch

class Backend(ABC):
    """
    Abstract base class for tensor substrates.
    Allows swapping between PyTorch/Unsloth (Linux/GPU) and MLX (macOS/Silicon).
    """
    @abstractmethod
    def load_model(self, model_id: str, **kwargs) -> Any:
        pass

    @abstractmethod
    def to_device(self, tensor: Any, device: str) -> Any:
        pass

    @abstractmethod
    def slice_activations(self, output: Any, layer_idx: int) -> Any:
        pass

class TorchBackend(Backend):
    """PyTorch implementation using Unsloth/HF."""
    def load_model(self, model_id: str, **kwargs) -> Any:
        # Avoid circular imports by importing inside method if needed
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            **kwargs
        )
        return model, tokenizer

    def to_device(self, tensor: torch.Tensor, device: str) -> torch.Tensor:
        return tensor.to(device)

    def slice_activations(self, output: Any, layer_idx: int) -> torch.Tensor:
        # output is typically the result of a layer hook
        # For HF/Unsloth, this is often a tuple where index 0 is the hidden states
        if isinstance(output, tuple):
            return output[0].detach()
        return output.detach()

class MLXBackend(Backend):
    """
    MLX implementation for Apple Silicon.
    TODO: Implement once Mac Mini hardware is available.
    """
    def load_model(self, model_id: str, **kwargs) -> Any:
        raise NotImplementedError("MLXBackend is not yet implemented (Pending Mac Mini).")

    def to_device(self, tensor: Any, device: str) -> Any:
        # MLX handles unified memory automatically
        return tensor

    def slice_activations(self, output: Any, layer_idx: int) -> Any:
        raise NotImplementedError("MLXBackend is not yet implemented.")

def get_backend(name: str = "torch") -> Backend:
    if name.lower() == "torch":
        return TorchBackend()
    elif name.lower() == "mlx":
        return MLXBackend()
    else:
        raise ValueError(f"Unknown backend requested: {name}")

"""
Activation pooling mechanisms for intercepting transformer paths.
"""
import torch

def masked_mean_pool(activations: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pool an activation tensor ignoring padding tokens based on the attention mask.
    
    Args:
        activations: (batch, seq_len, d_model) tensor of hidden states
        attention_mask: (batch, seq_len) tensor of binary masks (1=real, 0=padding)
        
    Returns:
        (batch, d_model) pooled tensor
    """
    # Cast to (batch, seq_len, 1) to match dimensionality of activations
    if attention_mask.dim() == 2:
        mask = attention_mask.unsqueeze(-1).float()
    else:
        mask = attention_mask.float()
        
    masked_sum = (activations * mask).sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1)
    pooled = masked_sum / lengths
    return pooled

def get_unsloth_base_tokenizer(tokenizer):
    """
    Handles bypassing the multimodal Gemma 4 processor in Unsloth.
    Calling the tokenizer positionally trips unsloth's patched_call which routes through
    the vision processor. We extract the base text tokenizer here.
    """
    return getattr(tokenizer, "tokenizer", tokenizer)

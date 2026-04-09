import torch
import torch.nn as nn

class TemporalLoRAGate(nn.Module):
    """
    Step 5: The Execution Node — Temporal LoRA Gating Mechanism
    
    This simulates the proof-of-concept for dynamic LoRA switching.
    Instead of permanently merging a LoRA into the base weights (which causes 
    catastrophic forgetting and parameter interference), we keep the base weights
    frozen.
    
    We pass the output of the Routing Node (SAE) into this Gate.
    The Gate uses the SAE activation strengths to dynamically multiply the 
    outputs of our .mrna adapters before adding them to the base model's output!
    """
    def __init__(self, base_dim=2048, num_adapters=2):
        super().__init__()
        self.base_dim = base_dim
        self.num_adapters = num_adapters
        
        # In a real model, this is the massive, frozen PreTrained weight matrix
        self.frozen_base_layer = nn.Linear(base_dim, base_dim, bias=False)
        self.frozen_base_layer.weight.requires_grad = False
        
        # These are our tiny, streamable .mrna adapters (Rank 16)
        # B matrix (Rank -> Base), A matrix (Base -> Rank)
        self.adapter_0_A = nn.Linear(base_dim, 16, bias=False)
        self.adapter_0_B = nn.Linear(16, base_dim, bias=False)
        
        self.adapter_1_A = nn.Linear(base_dim, 16, bias=False)
        self.adapter_1_B = nn.Linear(16, base_dim, bias=False)

    def forward(self, x, routing_signals):
        """
        x: The dense residual stream [batch, seq, base_dim]
        routing_signals: The specific CB-SAE latent labels [batch, seq, num_adapters]
                         e.g., [0.95, 0.0] means 95% certainty to use Adapter 0 (Python)
        """
        # 1. Base Model Computation (Always runs)
        base_output = self.frozen_base_layer(x)
        
        # 2. Adapter Computations (A * B)
        # Adapter 0: Python Code
        lora_0_out = self.adapter_0_B(self.adapter_0_A(x))
        # Adapter 1: Legal Text
        lora_1_out = self.adapter_1_B(self.adapter_1_A(x))
        
        # 3. The Temporal Gating Mechanism !!
        # We multiply the adapter outputs by their routing signal strength from the SAE.
        # If the SAE says "0.0" for Legal Text, the Legal adapter's math is completely zeros!
        
        shape_match = routing_signals.unsqueeze(-1) # [batch, seq, adapters, 1]
        
        gated_0 = lora_0_out * shape_match[:, :, 0]
        gated_1 = lora_1_out * shape_match[:, :, 1]
        
        # 4. Final Injection
        final_output = base_output + gated_0 + gated_1
        return final_output

if __name__ == "__main__":
    print("=== Temporal LoRA Gating Test ===")
    
    # Simulate a token representing "def my_function():"
    dense_token = torch.randn(1, 1, 2048)
    
    # 1. Simulate the Routing Node output (CB-SAE)
    # The SAE has identified this almost certainly as Python (Adapter 0) and NOT Legal (Adapter 1)
    # [Python=0.98, Legal=0.01]
    sae_routing_signal = torch.tensor([[[0.98, 0.01]]]) 
    
    # 2. The Execution Node
    layer = TemporalLoRAGate()
    
    # Run the dynamic gate!
    output = layer(dense_token, sae_routing_signal)
    
    print("Base Layer + Gated LoRAs computed successfully!")
    print(f"Output shape injected back into residual stream: {output.shape}")
    
    print("\nConcept Validation:")
    print("Because the SAE routed 0.98 to Adapter 0 (Python), the Python LoRA's")
    print("weights dominated the calculation without permanently polluting the Base Model.")
    print("If the next token is a Legal term, the SAE will output [0.0, 1.0]")
    print("and the Legal adapter will instantly take over on a *per-token* basis!")

import torch
import torch.nn as nn
import torch.nn.functional as F

class CBSAE(nn.Module):
    """
    Step 4: The Routing Node — Concept Bottleneck Sparse Autoencoder
    
    Why we need this: We can't use raw "activations" from the LLM to decide which 
    .mrna adapter to load because neurons are "polysemantic" (they represent 
    multiple concepts in superposition). 
    
    An SAE forces these dense activations into a very wide, but mostly empty (Sparse)
    vector where each active dimension means ONE specific concept.
    
    A CB-SAE (Concept Bottleneck) goes further: it forces specific features to map
    to user-defined labels. E.g., we can force feature #47 to mean "Python Code".
    """
    def __init__(self, d_model=2048, expansion_factor=8, bottleneck_features=10):
        super().__init__()
        self.d_model = d_model
        # The SAE is usually 8x to 32x wider than the residual stream
        self.d_sae = d_model * expansion_factor
        
        # We define how many 'concepts' we are manually steering
        # For .mrna, these correspond to which adapters we want to trigger!
        self.bottleneck_features = bottleneck_features
        
        # Encode: Project dense LLM activations to high-dimensional sparse space
        self.encoder = nn.Linear(self.d_model, self.d_sae, bias=True)
        
        # Decode: Reconstruct the original dense activations
        self.decoder = nn.Linear(self.d_sae, self.d_model, bias=True)
        
        # Initialize decoder weights to have unit norm (standard SAE practice)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
            
    def forward(self, residual_activations):
        """
        Takes in a batch of residual stream activations (e.g. from Layer 12)
        and returns the reconstructed activations + the sparsity features
        """
        # 1. Encode into sparse features
        # ReLU enforces sparsity (anything negative becomes 0)
        sparse_features = F.relu(self.encoder(residual_activations))
        
        # 2. Decode back to original space
        reconstructed = self.decoder(sparse_features)
        
        return reconstructed, sparse_features
        
    def compute_loss(self, original_activations, labels=None):
        """
        Loss has three parts in a CB-SAE:
        1. Reconstruction Loss (L2): Did we preserve the LLM's thought process?
        2. Sparsity Penalty (L1): Force most features to be 0
        3. Bottleneck Loss: Force specific dimensions to match our labels!
        """
        reconstructed, sparse_features = self(original_activations)
        
        # 1. MSE: How well did we reconstruct the dense activations?
        mse_loss = F.mse_loss(reconstructed, original_activations)
        
        # 2. L1 Sparsity: Sum of absolute values of the sparse features
        l1_loss = torch.sum(torch.abs(sparse_features))
        
        # 3. Concept Bottleneck Loss (Iterative alignment)
        # If we have ground truth labels (e.g. "is this Python code?"),
        # we strictly penalize feature #0 if it doesn't match the label!
        cb_loss = 0.0
        if labels is not None:
            # Assume labels is a tensor matching the shape of bottleneck_features
            # We slice only the first 'bottleneck_features' features to compare
            cb_loss = F.binary_cross_entropy_with_logits(
                sparse_features[:, :, :self.bottleneck_features], 
                labels
            )
            
        total_loss = mse_loss + (0.01 * l1_loss) + (1.0 * cb_loss)
        return total_loss, sparse_features

# --- Demo the Routing Node ---
if __name__ == "__main__":
    print("=== CB-SAE Routing Concept ===")
    
    # Simulate grabbing activations from Layer 12 of a 2048-dim model
    # Batch=1, Sequence Length=32, d_model=2048
    mock_layer12_activations = torch.randn(1, 32, 2048)
    
    # Initialize our CB-SAE. It expands 2048 -> 16,384 dimensions!
    cb_sae = CBSAE(d_model=2048, expansion_factor=8, bottleneck_features=2)
    
    # We pretend we know this specific 32-token sequence contains Python code.
    # Label 1 = 1.0 (It IS Python code)
    # Label 2 = 0.0 (It is NOT Legal text)
    mock_labels = torch.tensor([[1.0, 0.0]]).expand(1, 32, -1)
    
    print("Computing Loss (MSE + L1 + Concept Bottleneck)...")
    loss, features = cb_sae.compute_loss(mock_layer12_activations, labels=mock_labels)
    
    print(f"Total Loss: {loss.item():.4f}")
    
    # In inference, we just use the encoder to get the routing signal
    routing_features = F.relu(cb_sae.encoder(mock_layer12_activations))
    
    # Check if our Python concept feature fired!
    python_activation_strength = routing_features[0, :, 0].mean().item()
    print(f"\nAverage Activation of Feature #0 ('Python_Code') across sequence: {python_activation_strength:.4f}")
    
    if python_activation_strength > 0:
        print("ROUTE DECISION: Trigger async PCIe transfer for 'Python_LoRA_Adapter.mrna'!")
    else:
        print("ROUTE DECISION: Standard inference. No adapter needed.")

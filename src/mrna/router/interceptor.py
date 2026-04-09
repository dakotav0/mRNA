import os
import torch
from unsloth import FastLanguageModel

class ActivationInterceptor:
    """
    Priority 3: The Routing Node — Real Model Forward Hooks
    
    Instead of `torch.randn` mock shapes, this class attaches a hook to a 
    frozen base model (like Llama-3.2-1B). 
    
    Whenever real text is processed, this interceptor intercepts the residual 
    stream activation tensor out of a specific intermediate layer (e.g., Layer 12).
    
    We collect these massive dense vectors, save them to disk, and use them 
    offline to train our CB-SAE decoder to recognize specific concepts!
    """
    def __init__(self, target_layer=12):
        self.target_layer = target_layer
        self.intercepted_activations = []
        self.hook_handle = None

    def _hook_fn(self, module, input_args, output):
        """
        The callback PyTorch invokes during the forward pass.
        output[0] contains the residual stream embedding for that layer.
        Shape: [batch, sequence_length, hidden_dim]
        """
        # We detach it from the graph and move to CPU to avoid blowing up VRAM
        # during sequential data collection.
        hidden_states = output[0].detach().cpu()
        self.intercepted_activations.append(hidden_states)

    def attach_to_model(self, model):
        """Dynamically latches the hook onto the target transformer layer."""
        
        # Find the module that contains the transformer layers/blocks.
        container = None
        for name, module in model.named_modules():
            if any(k in name.lower() for k in ["vision", "encoder", "siglip", "aligner"]):
                continue
            # Check for common naming patterns for layer lists
            if name.endswith(".layers") or name.endswith(".blocks") or name in ["layers", "blocks"]:
                if hasattr(module, "__getitem__") and len(module) > self.target_layer:
                    container = module
                    break
        
        if not container:
            # Fallback for very nested or custom Unsloth/HF structures
            # (e.g., model.model.language_model.model.layers)
            for name, module in model.named_modules():
                if "layers" in name or "blocks" in name:
                    if hasattr(module, "__getitem__") and len(module) > self.target_layer:
                        container = module
                        break
        
        if not container:
            available_names = [n for n, _ in model.named_modules()]
            print(f"[Interceptor] [ERROR] Failed to find layers/blocks. Available modules (first 10): {available_names[:10]}")
            if any("layers" in n for n in available_names):
                print(f"[Interceptor] [DEBUG] Found potential 'layers' matches: {[n for n in available_names if 'layers' in n][:5]}")
            
            raise AttributeError(
                f"Could not find transformer layer container in model type {type(model)}. "
                "Ensure target_layer is within bounds."
            )
            
        layer_module = container[self.target_layer]
        
        # Register the hook so it triggers *after* the layer finishes its math
        self.hook_handle = layer_module.register_forward_hook(self._hook_fn)
        print(f"[Interceptor] Latched forward hook onto {type(layer_module).__name__} (Layer {self.target_layer}) successfully.")

    def detach(self):
        """Crucial to remove the hook when not training the SAE to save memory/compute."""
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None
            print("[Interceptor] Hook detached successfully.")

    def save_harvested_dataset(self, filename="activations_dataset.pt"):
        """Saves the harvested real-world thoughts of the LLM to disk."""
        if not self.intercepted_activations:
            print("No activations intercepted!")
            return
            
        dataset = torch.cat(self.intercepted_activations, dim=0) # [total_batches, seq, hidden_dim]
        
        # Save to disk for offline SAE training
        torch.save(dataset, filename)
        print(f"Saved {dataset.shape[0]} sequences of {dataset.shape[2]}-dim activations to {filename}")
        
        # Clear RAM
        self.intercepted_activations.clear()

if __name__ == "__main__":
    print("=== Testing Priority 3: SAE Activation Interceptor ===")
    
    # Let's boot up the exact model we proved in test_llama.py!
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct",
        max_seq_length=2048,
        dtype=None, 
        load_in_4bit=True
    )
    
    # 1. Prepare for inference
    FastLanguageModel.for_inference(model)
    
    # 2. Attach our interceptor to Layer 6 
    # (1B model is 16 layers shallow. Layer 6 is early-middle features)
    harvester = ActivationInterceptor(target_layer=6)
    harvester.attach_to_model(model)
    
    # 3. Create real data! We'll prompt it with Python code knowing we want  
    # to train our SAE identifying "Python Code"
    inputs = tokenizer(
        [
            "def calculate_fibonacci(n):\n    if n <= 1:\n        return n\n    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)",
            "The Supreme Court of the United States ruled today that the injunction...",
        ], 
        return_tensors="pt",
        padding=True,
    ).to("cuda")
    
    print("\n[Base Model] Computing forward pass for 2 distinct semantic domains...")
    # Generate doesn't need gradients. Our hook fires silently under the hood.
    _ = model.generate(**inputs, max_new_tokens=10, use_cache=True)
    
    # 4. Offload the math!
    harvester.save_harvested_dataset("mock_python_legal_layer6.pt")
    
    harvester.detach()
    print("\nWe now have authentic residual stream 'thought' vectors representing Python and Legal text!")
    print("These are perfectly formulated to be the input to the CB-SAE training script.")

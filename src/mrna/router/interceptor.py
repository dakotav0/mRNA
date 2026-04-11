import os

import torch


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

    def __init__(self, target_layers=None):
        if target_layers is None:
            target_layers = [12]
        if isinstance(target_layers, int):
            target_layers = [target_layers]

        self.target_layers = target_layers
        self.intercepted_activations = {layer: [] for layer in target_layers}
        self.hook_handles = []

    def _make_hook(self, layer_idx):
        """Factory for generating per-layer hooks."""

        def _hook_fn(module, input_args, output):
            # output[0] = residual stream hidden states
            hidden_states = output[0].detach().cpu()
            self.intercepted_activations[layer_idx].append(hidden_states)

        return _hook_fn

    def attach_to_model(self, model):
        """Dynamically latches hooks onto all target transformer layers."""
        # Standard backbone detection logic remains the same...
        backbone = model
        if hasattr(model, "language_model"):
            backbone = model.language_model
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            backbone = model.model
        elif hasattr(model, "text_model"):
            backbone = model.text_model

        # 2. Find the module list (layers/blocks) within the selected backbone
        container = None
        for child_name, child_module in backbone.named_children():
            if child_name in ["layers", "blocks"] or child_name.endswith(".layers"):
                # Use max of target_layers to verify bounds
                if hasattr(child_module, "__getitem__") and len(child_module) > max(
                    self.target_layers
                ):
                    container = child_module
                    break

        if not container:
            for name, module in backbone.named_modules():
                if any(
                    k in name.lower() for k in ["vision", "encoder", "siglip", "visual"]
                ):
                    continue
                if (
                    name.endswith(".layers")
                    or name.endswith(".blocks")
                    or name in ["layers", "blocks"]
                ):
                    if hasattr(module, "__getitem__") and len(module) > max(
                        self.target_layers
                    ):
                        container = module
                        break

        if not container:
            raise AttributeError(
                f"Could not find transformer layer container in {type(model)}."
            )

        # 3. Register a hook for every requested layer
        for layer_idx in self.target_layers:
            layer_module = container[layer_idx]
            handle = layer_module.register_forward_hook(self._make_hook(layer_idx))
            self.hook_handles.append(handle)
            print(
                f"[Interceptor] Latched hook onto Layer {layer_idx} ({type(layer_module).__name__})"
            )

    def detach(self):
        """Removes all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        print(f"[Interceptor] All {len(self.target_layers)} hooks detached.")

    def save_harvested_dataset(
        self, activations_dir: str, concept: str, is_test: bool = False
    ):
        """Saves harvested activations for ALL layers into separate subfolders."""
        suffix = "test" if is_test else "train"

        for layer_idx, tensors in self.intercepted_activations.items():
            if not tensors:
                continue

            layer_dir = os.path.join(activations_dir, f"layer_{layer_idx}")
            os.makedirs(layer_dir, exist_ok=True)

            combined = torch.cat(tensors, dim=0)
            target_path = os.path.join(layer_dir, f"{concept}_{suffix}.pt")

            torch.save(combined, target_path)
            print(f"[Interceptor] Saved {combined.shape[0]} sequences to {target_path}")

            # Clear list to save RAM
            tensors.clear()


if __name__ == "__main__":
    print("=== Testing Priority 3: SAE Activation Interceptor ===")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
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
    print(
        "\nWe now have authentic residual stream 'thought' vectors representing Python and Legal text!"
    )
    print(
        "These are perfectly formulated to be the input to the CB-SAE training script."
    )

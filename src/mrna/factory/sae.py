import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from mrna.core.config import MRNAPaths, config
from mrna.router.sae import CBSAE


def train_sae_weights(
    concepts: List[str],
    activation_files: Optional[List[str]] = None,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    expansion_factor: Optional[int] = None,
    batch_size: Optional[int] = None,
    val_split: float = 0.1,
    l1_coeff: Optional[float] = None,
    cb_coeff: Optional[float] = None,
    max_examples: Optional[int] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_path: Optional[str] = None,
    **kwargs,
):
    """
    Trains the Concept Bottleneck SAE. Supporting multi-layer looping.
    """
    mid = kwargs.get("model_id") or config.current_model_id
    m_cfg = config.get_model_config(mid)

    # Resolve target layers
    target_layers = (
        [kwargs.get("layer")]
        if kwargs.get("layer") is not None
        else config.get_harvest_layers(mid)
    )

    # Resolve pipeline arguments
    epochs = int(config.get_pipeline_arg("sae", "epochs", mid, epochs) or 50)
    lr = float(config.get_pipeline_arg("sae", "lr", mid, lr) or 1e-4)
    expansion_factor = int(
        config.get_pipeline_arg("sae", "expansion_factor", mid, expansion_factor) or 8
    )
    batch_size = int(
        config.get_pipeline_arg("sae", "batch_size", mid, batch_size) or 32
    )
    l1_coeff = float(config.get_pipeline_arg("sae", "l1_coeff", mid, l1_coeff) or 0.01)
    cb_coeff = float(config.get_pipeline_arg("sae", "cb_coeff", mid, cb_coeff) or 1.0)

    # Priority for max_examples: 1. CLI flag 2. Config pipeline arg 3. All (None)
    max_examples = config.get_pipeline_arg("sae", "max_examples", mid, max_examples)
    if max_examples is not None:
        max_examples = int(max_examples)
        print(f"[SAE] Limiting training data to {max_examples} per concept.")

    d_model = kwargs.get("d_model") or m_cfg.get("d_model")

    last_save_path = None

    for layer in target_layers:
        ldir = f"layer_{layer}"
        print(f"\n[SAE] [Layer {layer}] Beginning training suite...")

        # 1. Resolve Data Files for THIS layer
        layer_activation_files = []
        if not activation_files:
            act_dir = MRNAPaths.get_activations_dir(mid, ldir)
            for concept in concepts:
                p = act_dir / f"{concept}_train.pt"
                if not p.exists():
                    print(
                        f"Skipping Layer {layer}: '{concept}' activations missing ({p})"
                    )
                    continue
                layer_activation_files.append(str(p))
        else:
            # If explicit files passed, we assume they match the layer or caller knows what they are doing
            layer_activation_files = activation_files

        if not layer_activation_files:
            continue

        n_concepts = len(concepts)
        all_acts = []
        class_indices = []
        concept_counts = []

        # 2. Load and Pool
        for idx, path in enumerate(layer_activation_files):
            print(f"[*] Loading {os.path.basename(path)}...")
            acts = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(acts, list):
                acts = torch.cat(acts, dim=0)
            if acts.dim() == 3:
                acts = acts.mean(dim=1)  # Mean pool

            # Optional capping
            if max_examples is not None and len(acts) > max_examples:
                indices = torch.randperm(len(acts))[:max_examples]
                acts = acts[indices]

            n = len(acts)
            concept_counts.append(n)
            all_acts.append(acts)
            class_indices.append(torch.full((n,), idx, dtype=torch.long))

        X = torch.cat(all_acts, dim=0).float().to(device)
        Y = torch.cat(class_indices, dim=0).to(device)

        # Weights for imbalance
        total = sum(concept_counts)
        class_weights = torch.tensor(
            [total / (n_concepts * c) for c in concept_counts],
            dtype=torch.float32,
            device=device,
        )

        # 3. Split
        dataset = TensorDataset(X, Y)
        n_val = max(1, int(len(dataset) * val_split))
        train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # 4. Model
        model = CBSAE(
            d_model=d_model,
            expansion_factor=expansion_factor,
            bottleneck_features=n_concepts,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        ce_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        # 5. Loop
        for epoch in range(1, epochs + 1):
            model.train()
            batch_loss = 0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pre_relu = model.encoder(x_batch)
                sparse = F.relu(pre_relu)
                recon = model.decoder(sparse)
                loss = (
                    F.mse_loss(recon, x_batch)
                    + l1_coeff * sparse.abs().mean()
                    + cb_coeff * ce_loss_fn(pre_relu[:, :n_concepts], y_batch)
                )
                loss.backward()
                with torch.no_grad():
                    model.decoder.weight.data = F.normalize(
                        model.decoder.weight.data, dim=0
                    )
                optimizer.step()
                batch_loss += loss.item()

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"[*] Layer {layer} | Epoch {epoch}/{epochs} | Loss: {batch_loss / len(train_loader):.4f}"
                )

        # 6. Save
        save_path = output_path or str(MRNAPaths.get_sae_weights_path(mid, layer))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"[SAE] Saved: {save_path}")
        last_save_path = save_path

    return last_save_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="mRNA SAE Weight Trainer")
    parser.add_argument("--concepts", required=True, help="Comma-separated concepts")
    parser.add_argument("--files", help="Optional: Comma-separated activation files")
    parser.add_argument("--model", help="Model ID override")
    parser.add_argument("--layer", type=int, help="Target layer override")
    parser.add_argument("--epochs", type=int, help="Training epochs override")
    parser.add_argument("--max-examples", type=int, help="Max activations per concept")
    args = parser.parse_args()

    train_sae_weights(
        concepts=args.concepts.split(","),
        activation_files=args.files.split(",") if args.files else None,
        model_id=args.model,
        layer=args.layer,
        epochs=args.epochs,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()

import os
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from typing import List, Optional, Dict

from mrna.core.config import config, MRNAPaths
from mrna.router.sae import CBSAE

def train_sae_weights(
    concepts: List[str],
    activation_files: List[str],
    epochs: int = 50,
    lr: float = 1e-4,
    expansion_factor: int = 8,
    batch_size: int = 64,
    val_split: float = 0.1,
    l1_coeff: float = 0.01,
    cb_coeff: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_path: Optional[str] = None,
    **kwargs
):
    """
    Trains the Concept Bottleneck SAE.
    """
    n_concepts = len(concepts)
    all_acts = []
    class_indices = []
    concept_counts = []
    
    # 1. Load and Pool
    # We use config to help resolve d_model if needed
    d_model = kwargs.get("d_model") or config.get_model_config().get("d_model")

    for idx, path in enumerate(activation_files):
        acts = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(acts, list): acts = torch.cat(acts, dim=0)
        
        if acts.dim() == 3:
            acts = acts.mean(dim=1)
        
        n = len(acts)
        concept_counts.append(n)
        all_acts.append(acts)
        class_indices.append(torch.full((n,), idx, dtype=torch.long))

    X = torch.cat(all_acts, dim=0).float()
    Y = torch.cat(class_indices, dim=0)
    
    # Weights for imbalance
    total = sum(concept_counts)
    class_weights = torch.tensor(
        [total / (n_concepts * c) for c in concept_counts],
        dtype=torch.float32, device=device
    )

    # 2. Split
    dataset = TensorDataset(X, Y)
    n_val = max(1, int(len(dataset) * val_split))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # 3. Model
    model = CBSAE(
        d_model=d_model,
        expansion_factor=expansion_factor,
        bottleneck_features=n_concepts
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # 4. Loop
    print(f"Starting SAE training for {n_concepts} concepts...")
    for epoch in range(1, epochs + 1):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            pre_relu = model.encoder(x_batch)
            sparse = F.relu(pre_relu)
            recon = model.decoder(sparse)
            
            loss = F.mse_loss(recon, x_batch) + l1_coeff * sparse.abs().mean() + cb_coeff * ce_loss_fn(pre_relu[:, :n_concepts], y_batch)
            loss.backward()
            
            with torch.no_grad():
                model.decoder.weight.data = F.normalize(model.decoder.weight.data, dim=0)
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} complete.")

    # 5. Save
    save_path = output_path or str(MRNAPaths.DATA / config.current_model_id / "sae_weights.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"SAE weights saved to {save_path}")
    return save_path

def main():
    import argparse
    parser = argparse.ArgumentParser(description="mRNA SAE Weight Trainer")
    parser.add_argument("--concepts", required=True, help="Comma-separated concepts")
    parser.add_argument("--files", required=True, help="Comma-separated activation files")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    train_sae_weights(
        concepts=args.concepts.split(","),
        activation_files=args.files.split(","),
        epochs=args.epochs
    )

if __name__ == "__main__":
    main()

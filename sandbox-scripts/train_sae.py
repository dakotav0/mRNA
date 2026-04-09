"""
CBSAE Training Script — mRNA Routing Node

Trains the Concept Bottleneck SAE on activation files harvested by
ActivationInterceptor.save_harvested_dataset(). The trained weights can be
passed directly to mRNAPipeline(sae_weights_path=...).

Usage
-----
    python sandbox-scripts/train_sae.py \
        --activations python:data/activations/gemma-3-4b/layer-17/python.pt \
        --epochs 50 \
        --output data/sae_weights.pt

Activation file format
----------------------
torch.Tensor saved with torch.save(), shape either:
  (N, seq_len, d_model)  — raw interceptor output  [will be pooled]
  (N, d_model)           — pre-pooled

Train / inference consistency
------------------------------
mRNAPipeline.route() pools (batch, seq_len, d_model) → (batch, d_model) BEFORE
encoding. This script does the same pooling so the SAE encoder sees the same
distribution during training that it will see at inference time.

Loss design (v2)
----------------
The original BCE-on-post-ReLU formulation had a floor problem: after ReLU all
values are ≥ 0, so sigmoid(v) ≥ 0.5 always — the model could never express
"NOT this concept" with > 50% confidence. CB loss was stuck near log(2) ≈ 0.693.

Fix: split the encoder output before ReLU.
  - pre_relu  = encoder(x)                  ← used for routing classification
  - sparse    = ReLU(pre_relu)              ← used for reconstruction + L1
  - cb_loss   = CrossEntropy(pre_relu[:, :n_concepts], class_idx, weight=w)

CrossEntropyLoss (softmax + NLL) is correct here: routing is one-of-N, not N
independent binary decisions. It also handles the negative logit space that BCE
couldn't access, and naturally suppresses all non-winning concepts.

Class weights correct for dataset size imbalance (e.g. poetry ~573 vs 5000 others).
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mrna.router.sae import CBSAE

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    concepts: list[str],
    activation_files: list[str],
    d_model: int,
    expansion_factor: int,
    epochs: int,
    lr: float,
    batch_size: int,
    val_split: float,
    l1_coeff: float,
    cb_coeff: float,
    output_path: str,
    device: str,
):
    n_concepts = len(concepts)

    # ------------------------------------------------------------------
    # 1. Load and pool activations — track per-concept counts for class weights
    # ------------------------------------------------------------------
    print("Loading activation files...")
    all_acts: list[torch.Tensor] = []
    class_indices: list[torch.Tensor] = []
    concept_counts: list[int] = []

    for concept_idx, (concept_name, path) in enumerate(zip(concepts, activation_files)):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Activation file not found: {path}\n"
                f"Run harvest_hf.py to generate it first."
            )

        acts = torch.load(path, map_location="cpu", weights_only=True)

        # harvest_hf.py and discover_concepts.py both save lists of (1, d_model) tensors
        if isinstance(acts, list):
            acts = torch.cat(acts, dim=0)

        if acts.dim() == 3:
            acts = acts.mean(dim=1)
        elif acts.dim() != 2:
            raise ValueError(f"Expected 2D or 3D tensor in {path}, got {acts.dim()}D")

        if d_model is None:
            d_model = acts.shape[-1]
            print(f"  Detected d_model: {d_model}")
        elif acts.shape[-1] != d_model:
            raise ValueError(
                f"d_model mismatch in {path}: expected {d_model}, got {acts.shape[-1]}."
            )

        n = len(acts)
        concept_counts.append(n)
        print(f"  '{concept_name}': {n} examples from {path}")

        all_acts.append(acts)
        class_indices.append(torch.full((n,), concept_idx, dtype=torch.long))

    X = torch.cat(all_acts, dim=0).float()  # (total_N, d_model)
    Y = torch.cat(class_indices, dim=0)  # (total_N,) — integer class labels

    # Inverse-frequency class weights to compensate for imbalance (e.g. poetry 573 vs 5000)
    total = sum(concept_counts)
    class_weights = torch.tensor(
        [total / (n_concepts * c) for c in concept_counts],
        dtype=torch.float32,
        device=device,
    )
    max_w = class_weights.max().item()
    print("\nClass weights (inverse freq, normalised to max=1):")
    for name, w in zip(concepts, class_weights.tolist()):
        print(
            f"  {name:>14}: {w / max_w:.3f}  ({concept_counts[concepts.index(name)]} examples)"
        )

    # Shuffle before split
    perm = torch.randperm(len(X))
    X, Y = X[perm], Y[perm]

    # ------------------------------------------------------------------
    # 2. Train / val split
    # ------------------------------------------------------------------
    dataset = TensorDataset(X, Y)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print(
        f"\nDataset: {n_train} train / {n_val} val  |  {n_concepts} concepts: {concepts}"
    )

    # ------------------------------------------------------------------
    # 3. Model + optimizer + LR scheduler
    # ------------------------------------------------------------------
    model = CBSAE(
        d_model=d_model,
        expansion_factor=expansion_factor,
        bottleneck_features=n_concepts,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Cosine decay — gently reduces LR over training to help convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.1
    )

    ce_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    print(
        f"\nTraining CBSAE  (d_model={d_model}, "
        f"d_sae={d_model * expansion_factor}, "
        f"bottleneck={n_concepts})\n"
    )

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = train_mse = train_l1 = train_cb = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)  # (B,) integer class indices

            optimizer.zero_grad()

            # Split before/after ReLU so CE loss can access negative logit space
            pre_relu = model.encoder(x_batch)  # (B, d_sae) — full range
            sparse = F.relu(pre_relu)  # (B, d_sae) — reconstruction path
            recon = model.decoder(sparse)

            mse_loss = F.mse_loss(recon, x_batch)
            l1_loss = sparse.abs().mean()

            # CrossEntropy over the bottleneck slice of the pre-ReLU logits
            # This is one-of-N routing — softmax + NLL, not per-dim BCE
            cb_loss = ce_loss_fn(pre_relu[:, :n_concepts], y_batch)

            loss = mse_loss + l1_coeff * l1_loss + cb_coeff * cb_loss
            loss.backward()

            # Re-normalise decoder columns (standard SAE practice)
            with torch.no_grad():
                model.decoder.weight.data = F.normalize(
                    model.decoder.weight.data, dim=0
                )

            optimizer.step()

            train_loss += loss.item()
            train_mse += mse_loss.item()
            train_l1 += l1_loss.item()
            train_cb += cb_loss.item()

        scheduler.step()

        n = len(train_loader)
        train_loss /= n
        train_mse /= n
        train_l1 /= n
        train_cb /= n

        # --- Val ---
        model.eval()
        val_loss = val_acc = 0.0
        per_concept_correct = torch.zeros(n_concepts)
        per_concept_total = torch.zeros(n_concepts)

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                pre_relu = model.encoder(x_batch)
                sparse = F.relu(pre_relu)
                recon = model.decoder(sparse)

                mse_loss = F.mse_loss(recon, x_batch)
                l1_loss = sparse.abs().mean()
                cb_loss = ce_loss_fn(pre_relu[:, :n_concepts], y_batch)
                val_loss += (mse_loss + l1_coeff * l1_loss + cb_coeff * cb_loss).item()

                # Routing accuracy
                pred_idx = pre_relu[:, :n_concepts].argmax(dim=-1)  # argmax on pre-ReLU
                correct = pred_idx == y_batch
                val_acc += correct.float().mean().item()

                for c in range(n_concepts):
                    mask = y_batch == c
                    per_concept_correct[c] += correct[mask].sum().item()
                    per_concept_total[c] += mask.sum().item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        print(
            f"Epoch {epoch:>3}/{epochs}  "
            f"train={train_loss:.4f} "
            f"[mse={train_mse:.4f} l1={train_l1:.4f} cb={train_cb:.4f}]  "
            f"val={val_loss:.4f}  routing_acc={val_acc:.1%}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Per-concept breakdown every 10 epochs
        if epoch % 10 == 0:
            print("  Per-concept val accuracy:")
            for c, name in enumerate(concepts):
                n_total = int(per_concept_total[c].item())
                if n_total > 0:
                    acc = per_concept_correct[c].item() / n_total
                    print(f"    {name:>14}: {acc:.1%}  ({n_total} examples)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Saved SAE weights → {output_path}")
    print(
        f"\nUse in pipeline:\n"
        f"  pipeline = mRNAPipeline(\n"
        f"      adapter_registry={{{', '.join(repr(c) + ': ...' for c in concepts)}}},\n"
        f"      d_model={d_model},\n"
        f"      sae_weights_path={repr(output_path)},\n"
        f"  )"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train CBSAE on harvested activations from ActivationInterceptor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--activations",
        nargs="+",
        metavar="NAME:FILE",
        required=True,
        help="concept_name:path pairs  e.g.  python:data/python.pt legal:data/legal.pt",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=None,
        help="Activation dimension (e.g. 2560 for Gemma 3). Auto-detected if None.",
    )
    parser.add_argument("--expansion-factor", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument(
        "--l1-coeff", type=float, default=0.01, help="Weight on L1 sparsity penalty"
    )
    parser.add_argument(
        "--cb-coeff",
        type=float,
        default=1.0,
        help="Weight on concept bottleneck CrossEntropy loss",
    )
    parser.add_argument("--output", type=str, default="data/sae_weights.pt")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    concepts, files = [], []
    for entry in args.activations:
        if ":" not in entry:
            parser.error(f"Expected NAME:FILE format, got: {entry!r}")
        name, path = entry.split(":", 1)
        concepts.append(name)
        files.append(path)

    os.makedirs(
        os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
        exist_ok=True,
    )

    train(
        concepts=concepts,
        activation_files=files,
        d_model=args.d_model,
        expansion_factor=args.expansion_factor,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_split=args.val_split,
        l1_coeff=args.l1_coeff,
        cb_coeff=args.cb_coeff,
        output_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()

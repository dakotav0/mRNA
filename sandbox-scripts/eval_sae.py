"""
Automated SAE Evaluator

Iterates through holdout datasets (*_test.pt) and processes them
through a trained SAE to determine routing accuracy and generate a
confusion matrix mapping (Bleed Detection).

Usage
-----
# Evaluate all _test.pt files in the data/ directory for Gemma 4 E2B Layer 25
python sandbox-scripts/eval_sae.py \
    --model-id gemma-4-e2b-it \
    --layer 25 \
    --concepts biology,chemistry,physics
"""

import argparse
import glob
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mrna.data.paths import MRNAPaths
from mrna.router.sae import CBSAE


def evaluate_holdout(
    file_path: str, sae: CBSAE, concept_idx: int, n_concepts: int
) -> dict:
    """Evaluates a single holdout tensor and returns metrics."""
    # Load test activations
    activations = torch.load(file_path, map_location="cpu", weights_only=True)
    if isinstance(activations, list):
        activations = torch.cat(activations, dim=0)
    if not isinstance(activations, torch.Tensor):
        print(f"Skipping {file_path}: Not a valid tensor.")
        return None

    # Forward pass
    with torch.no_grad():
        pre_relu = sae.encoder(activations.float())
        bottleneck = pre_relu[:, :n_concepts]
        predictions = bottleneck.argmax(dim=-1)

    total = len(predictions)
    correct = (predictions == concept_idx).sum().item()

    # Generate confusion distributions
    confusion_counts = [0] * n_concepts
    for p in predictions:
        confusion_counts[p.item()] += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "confusion_counts": confusion_counts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SAE routing matrix on holdout datasets."
    )
    parser.add_argument(
        "--model-id",
        default="unsloth/gemma-4-E2B-it",
        help="Model ID or name to evaluate",
    )
    parser.add_argument("--layer", type=int, default=25, help="Layer index")
    parser.add_argument(
        "--concepts",
        required=True,
        help="Comma-separated concepts in the trained SAE to map indices (e.g. biology,chemistry,physics)",
    )
    parser.add_argument("--d-model", type=int, default=1536)
    args = parser.parse_args()

    paths = MRNAPaths(model_id=args.model_id)
    sae_path = paths.get_sae_weights(args.layer)
    holdout_dir = paths.get_activations_dir(args.layer)

    concept_list = [c.strip() for c in args.concepts.split(",")]
    n_concepts = len(concept_list)

    # 1. Initialize SAE
    print(f"Loading SAE Weights from: {sae_path}")
    sae = CBSAE(
        d_model=args.d_model, expansion_factor=8, bottleneck_features=n_concepts
    )
    if not os.path.exists(sae_path):
        print(f"[FATAL] Could not find {sae_path}")
        sys.exit(1)

    sae.load_state_dict(torch.load(sae_path, map_location="cpu", weights_only=True))
    sae.eval()

    # 2. Gather Holdout Files
    holdout_files = glob.glob(
        os.path.join(holdout_dir, "**", "*_test.pt"), recursive=True
    )

    if not holdout_files:
        print(
            f"No valid _test.pt holdout files found in {holdout_dir}. Ensure you have harvested data with --holdout-ratio > 0."
        )
        sys.exit(0)

    # 3. Benchmark
    print("\n" + "=" * 70)
    print("SAE BENCHMARK RUNNER")
    print("=" * 70)

    overall_correct = 0
    overall_total = 0
    results_map = {}

    for file_path in holdout_files:
        filename = os.path.basename(file_path).lower()

        # Determine the ground truth index by checking if the concept name is in the filename
        mapped_concept = None
        mapped_idx = -1
        for i, concept in enumerate(concept_list):
            if concept.lower() in filename:
                mapped_concept = concept
                mapped_idx = i
                break

        if mapped_concept is None:
            print(
                f"[WARNING] Skipping {filename} - could not infer concept from {concept_list}"
            )
            continue

        print(f"Evaluating {mapped_concept.capitalize()} ({filename}) ...")
        metrics = evaluate_holdout(file_path, sae, mapped_idx, n_concepts)
        if metrics is None:
            continue

        results_map[mapped_concept] = metrics
        overall_correct += metrics["correct"]
        overall_total += metrics["total"]

    # 4. Summary Table Output
    if not results_map:
        print("No evaluations succeeded.")
        sys.exit(0)

    print("\n" + "=" * 70)
    print("ROUTING ACCURACY RESULTS")
    print("=" * 70)

    print(f"{'Concept':<15} | {'Acc':<8} | {'Total':<6} | Confusion (Bleed)")
    print("-" * 70)
    for concept in concept_list:
        if concept in results_map:
            metrics = results_map[concept]
            acc_str = f"{metrics['accuracy'] * 100:.1f}%"

            # Format confusion string showing only misclassifications
            bleeds = []
            for i, count in enumerate(metrics["confusion_counts"]):
                if i != concept_list.index(concept) and count > 0:
                    bleeds.append(f"{concept_list[i]}:{count}")

            bleed_str = ", ".join(bleeds) if bleeds else "Clean"

            print(
                f"{concept.capitalize():<15} | {acc_str:<8} | {metrics['total']:<6} | {bleed_str}"
            )
        else:
            print(
                f"{concept.capitalize():<15} | {'N/A':<8} | {'-':<6} | No holdout data evaluated"
            )

    final_acc = (overall_correct / overall_total) * 100 if overall_total > 0 else 0
    print("-" * 70)
    print(
        f"OVERALL PERFORMANCE: {final_acc:.2f}% ({overall_correct}/{overall_total} successful routes)"
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

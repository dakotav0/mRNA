import argparse
import sys
from typing import List, Optional

from mrna.analysis.eval import evaluate_sae_holdouts
from mrna.core.config import MRNAPaths, config
from mrna.factory.adapter import train_adapter
from mrna.factory.sae import train_sae_weights
from mrna.factory.sampler import harvest_activations


def run_e2e_pipeline(
    concepts: List[str],
    model_id: Optional[str] = None,
    max_examples: Optional[int] = None,
    sae_epochs: Optional[int] = None,
    adapter_steps: Optional[int] = None,
    run_eval: bool = True,
    dry_run: bool = False,
    download_datasets: bool = False,
):
    """
    Automatic end-to-end pipeline: Harvest -> SAE -> Adapter -> Eval
    """
    mid = model_id or config.current_model_id
    print(f"🚀 Starting E2E pipeline for model: {mid}")
    print(f"Target concepts: {concepts}")

    if dry_run:
        config.validate_setup(concepts, mid)
        print("[Dry Run] Workflow validated. Terminating before intensive load.")
        return

    # 1. Harvesting
    for concept in concepts:
        print(f"\n--- Harvesting: {concept} ---")
        harvest_activations(
            concept=concept,
            model_id=mid,
            max_examples=max_examples,
            streaming=not download_datasets,
        )

    # 2. SAE Training
    print("\n--- Training SAE weights ---")
    train_sae_weights(concepts=concepts, model_id=mid, epochs=sae_epochs)

    # 3. Adapter Training
    for concept in concepts:
        print(f"\n--- Training Adapter: {concept} ---")
        train_adapter(concept=concept, model_id=mid, max_steps=adapter_steps)

    # 4. Evaluation
    if run_eval:
        print("\n--- Running Cumulative Evaluation ---")
        results = evaluate_sae_holdouts(concepts=concepts, model_id=mid)
        print(
            f"Pipeline complete. Global Accuracy: {results.get('overall_accuracy', 0):.2%}"
        )


def main():
    parser = argparse.ArgumentParser(description=".mrna Auto-Train Pipeline")
    parser.add_argument(
        "--concepts", help="Comma-separated concepts (defaults to all approved)"
    )
    parser.add_argument("--model", help="Model ID override")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup only")
    parser.add_argument(
        "--download", action="store_true", help="Download datasets instead of streaming"
    )
    parser.add_argument("--examples", type=int, help="Examples to harvest per concept")
    parser.add_argument("--epochs", type=int, help="SAE training epochs")
    parser.add_argument("--steps", type=int, help="Adapter training steps")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation phase")

    args = parser.parse_args()

    concept_list = (
        args.concepts.split(",")
        if args.concepts
        else list(config.science_triad_datasets.keys())
    )

    run_e2e_pipeline(
        concepts=concept_list,
        model_id=args.model,
        max_examples=args.examples,
        sae_epochs=args.epochs,
        adapter_steps=args.steps,
        run_eval=not args.no_eval,
        dry_run=args.dry_run,
        download_datasets=args.download,
    )


if __name__ == "__main__":
    main()

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
    max_examples: int = 5000,
    sae_epochs: int = 50,
    adapter_steps: int = 200,
    run_eval: bool = True,
):
    """
    Automatic end-to-end pipeline: Harvest -> SAE -> Adapter -> Eval
    """
    mid = model_id or config.current_model_id
    print(f"🚀 Starting E2E pipeline for model: {mid}")
    print(f"Target concepts: {concepts}")

    # 1. Harvesting
    activation_files = []
    for concept in concepts:
        print(f"\n--- Harvesting: {concept} ---")
        harvest_activations(concept=concept, model_id=mid, max_examples=max_examples)

        # Resolve the resulting file path (assumes standard naming from sampler.py)
        m_cfg = config.get_model_config(mid)
        layer = m_cfg.get("harvest_layer")
        path = (
            MRNAPaths.DATA
            / mid
            / "activations"
            / f"layer_{layer}"
            / f"{concept}_train.pt"
        )
        activation_files.append(str(path))

    # 2. SAE Training
    print("\n--- Training SAE weights ---")
    train_sae_weights(
        concepts=concepts, activation_files=activation_files, epochs=sae_epochs
    )

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
        "--concepts", help="Comma-separated concepts (e.g. biology,math)"
    )
    parser.add_argument("--model", help="Model ID override")
    parser.add_argument(
        "--examples", type=int, default=5000, help="Examples to harvest per concept"
    )
    parser.add_argument("--epochs", type=int, default=50, help="SAE training epochs")
    parser.add_argument("--steps", type=int, default=200, help="Adapter training steps")
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
    )


if __name__ == "__main__":
    main()

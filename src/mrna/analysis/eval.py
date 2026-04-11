import glob
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from mrna.core.config import MRNAPaths, config
from mrna.router.sae import CBSAE


def calculate_metrics(
    confusion: Dict[str, int], concept_name: str, concepts: List[str]
) -> Dict[str, float]:
    """Calculates Precision, Recall, and F1 for a given concept."""
    tp = confusion.get(concept_name, 0)
    # False Positives: Instances of OTHER concepts classified as THIS concept
    # This requires looking at the confusion of ALL OTHER concepts (not available here yet)
    # We will compute this in the main loop instead.
    return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def print_report(layers_results: Dict[str, Any], concepts: List[str]):
    """Prints a professional tabular report of evaluation results."""
    print("\n" + "=" * 90)
    print(f"{'mRNA SAE EVALUATION REPORT':^90}")
    print("=" * 90)

    for layer_id, results in layers_results.items():
        layer_num = layer_id.split("_")[-1]
        print(
            f"\n[Layer {layer_num}] - {'Logic Gate' if int(layer_num) < 20 else 'Voice Pass'}"
        )
        print(
            f"{'Concept':<15} | {'Acc':<8} | {'Conf':<6} | {'P':<6} | {'R':<6} | {'F1':<6} | {'Bleed (Confusion)':<20}"
        )
        print("-" * 90)

        # Calculate P/R/F1 across all concepts for this layer
        all_tp = {c: 0 for c in concepts}
        all_fp = {c: 0 for c in concepts}
        all_fn = {c: 0 for c in concepts}

        for res in results["concepts"]:
            c_name = res["name"]
            all_tp[c_name] = res["confusion"][c_name]
            for other_c, count in res["confusion"].items():
                if other_c != c_name:
                    all_fn[c_name] += count
                    all_fp[other_c] += count

        for res in results["concepts"]:
            c_name = res["name"]
            tp = all_tp[c_name]
            fp = all_fp[c_name]
            fn = all_fn[c_name]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            acc_str = f"{res['accuracy'] * 100:.1f}%"
            conf_str = f"{res.get('avg_confidence', 0):.2f}"
            gap_str = f"{res.get('avg_conf_gap', 0):.2f}"
            p_str = f"{precision:.2f}"
            r_str = f"{recall:.2f}"
            f1_str = f"{f1:.2f}"

            bleeds = [
                f"{k}:{v}" for k, v in res["confusion"].items() if k != c_name and v > 0
            ]
            bleed_str = ", ".join(bleeds) if bleeds else "Clean"

            print(
                f"{c_name.capitalize():<15} | {acc_str:<8} | {conf_str:<6} | {gap_str:<6} | {p_str:<6} | {r_str:<6} | {f1_str:<6} | {bleed_str}"
            )

        print("-" * 105)
        overall_acc = results["overall_accuracy"] * 100
        print(
            f"OVERALL ACCURACY: {overall_acc:.2f}% ({results['correct_samples']}/{results['total_samples']} samples)"
        )
    print("=" * 105 + "\n")


def evaluate_sae_holdouts(
    concepts: List[str],
    model_id: Optional[str] = None,
    layer: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Evaluates routing accuracy across holdout datasets.
    """
    mid = model_id or config.current_model_id
    m_cfg = config.get_model_config(mid)

    # If no layer passed, we check ALL configured harvest layers
    layers = [layer] if layer is not None else config.get_harvest_layers(mid)

    overall_report = {}
    d_model = m_cfg.get("d_model")

    for target_layer in layers:
        # Load SAE
        sae_path = MRNAPaths.get_sae_weights_path(mid, target_layer)
        if not sae_path.exists():
            continue

        state_dict = torch.load(sae_path, map_location="cpu", weights_only=True)
        expansion = state_dict["encoder.weight"].shape[0] // d_model
        sae = CBSAE(
            d_model=d_model,
            expansion_factor=expansion,
            bottleneck_features=len(concepts),
        )
        sae.load_state_dict(state_dict)
        sae.eval()

        # Find Holdout Files
        holdout_dir = MRNAPaths.get_activations_dir(mid, f"layer_{target_layer}")
        holdout_files = glob.glob(str(holdout_dir / "**" / "*_test.pt"), recursive=True)

        overall_results = {
            "concepts": [],
            "overall_accuracy": 0.0,
            "total_samples": 0,
            "correct_samples": 0,
        }

        # Process
        for file_path in holdout_files:
            filename = os.path.basename(file_path).lower()
            mapped_idx = -1
            for i, concept in enumerate(concepts):
                if concept.lower() in filename:
                    mapped_idx = i
                    break

            if mapped_idx == -1:
                continue

            acts = torch.load(file_path, map_location="cpu", weights_only=True)
            if isinstance(acts, list):
                acts = torch.cat(acts, dim=0)

            with torch.no_grad():
                pre_relu = sae.encoder(acts.float())
                bottleneck = pre_relu[:, : len(concepts)]

                # Softmax confidence for the winning concept
                probs = F.softmax(bottleneck, dim=-1)
                confidences, preds = probs.max(dim=-1)
                avg_conf = confidences.mean().item()

                # Calculate Confidence Gap (Top-1 vs Top-2)
                top2_vals, _ = torch.topk(probs, k=2, dim=-1)
                conf_gap = (top2_vals[:, 0] - top2_vals[:, 1]).mean().item()

            correct = (preds == mapped_idx).sum().item()
            total = len(preds)

            confusion = [0] * len(concepts)
            for p in preds:
                confusion[p.item()] += 1

            res = {
                "name": concepts[mapped_idx],
                "accuracy": correct / total,
                "avg_confidence": avg_conf,
                "avg_conf_gap": conf_gap,
                "samples": total,
                "confusion": {concepts[i]: confusion[i] for i in range(len(concepts))},
            }
            overall_results["concepts"].append(res)
            overall_results["correct_samples"] += correct
            overall_results["total_samples"] += total

        if overall_results["total_samples"] > 0:
            overall_results["overall_accuracy"] = (
                overall_results["correct_samples"] / overall_results["total_samples"]
            )

        overall_report[f"layer_{target_layer}"] = overall_results

    return overall_report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="mRNA SAE Holdout Evaluator")
    parser.add_argument("--concepts", required=True, help="Comma-sep concepts")
    parser.add_argument("--model", help="Model ID override")
    parser.add_argument("--layer", type=int, help="Layer override")
    parser.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of table"
    )
    args = parser.parse_args()

    concepts = [c.strip() for c in args.concepts.split(",")]
    results = evaluate_sae_holdouts(
        concepts=concepts, model_id=args.model, layer=args.layer
    )

    if args.json:
        import json

        print(json.dumps(results, indent=2))
    else:
        print_report(results, concepts)


if __name__ == "__main__":
    main()

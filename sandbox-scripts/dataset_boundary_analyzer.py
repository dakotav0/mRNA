"""
Dataset Boundary Analyzer

Streams a random subset of a candidate HF dataset, formats it, runs a
prefill pass to extract residual stream activations, and passes them
through the current SAE.

Outputs a diagnostic report detailing whether the candidate dataset
overlaps strongly with existing concepts (danger) or sits in an
orthogonal subspace (safe for a new adapter).

Usage
-----
python sandbox-scripts/dataset_boundary_analyzer.py \
    --dataset camel-ai/biology \
    --max-examples 200
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datasets import load_dataset
from unsloth import FastLanguageModel

from mrna.data.dataset_utils import extract_text, extract_text2
from mrna.data.paths import MRNAPaths
from mrna.router.interceptor import ActivationInterceptor
from mrna.router.pooling import get_unsloth_base_tokenizer, masked_mean_pool
from mrna.router.sae import CBSAE


def main():
    parser = argparse.ArgumentParser(
        description="Analyze boundary overlap of a candidate dataset using the current SAE."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="HF dataset ID to analyze (e.g., camel-ai/math)",
    )
    parser.add_argument(
        "--dataset-config", default=None, help="HF dataset config/subset name"
    )
    parser.add_argument("--split", default="train", help="Dataset split to evaluate")
    parser.add_argument(
        "--text-column",
        default="instruction",
        help="Fallback primary text column (instruction, question, etc.)",
    )
    parser.add_argument(
        "--text-column2",
        default="output",
        help="Fallback secondary text column (e.g. output, answer, message_2)",
    )
    parser.add_argument(
        "--max-examples", type=int, default=200, help="Number of examples to profile"
    )

    # Gemma 4 E2B Defaults based on synthesis log discussions
    parser.add_argument("--model-id", default="unsloth/gemma-4-E2B-it")
    parser.add_argument(
        "--model-revision", default="37ea165b3fba25b7d851f8ce4ccff9a4f0751cee"
    )
    parser.add_argument(
        "--layer", type=int, default=25, help="Layer to extract activations from"
    )
    parser.add_argument("--d-model", type=int, default=1536)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument(
        "--concepts",
        default="biology,chemistry,physics",
        help="Comma-separated concepts in the trained SAE",
    )

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    paths = MRNAPaths(model_id=args.model_id)
    sae_weights = paths.get_sae_weights(args.layer)

    print("\n--- mRNA Dataset Boundary Analyzer ---")
    print(f"Candidate Dataset: {args.dataset}")
    print(f"Target Layer: {args.layer}")

    # 1. Format Validator (Loading and subsetting dataset)
    print("Loading dataset and applying format verification...")
    # Use streaming=True so HF never downloads more than max_examples rows.
    # split-slice notation only helps for Parquet-backed datasets; script-generated
    # datasets (like camel-ai/math) still iterate the full split without streaming.
    load_kwargs = dict(split=args.split, streaming=True)
    if args.dataset_config:
        load_kwargs["name"] = args.dataset_config

    ds = load_dataset(args.dataset, **load_kwargs)
    # take() hard-stops the generator after max_examples rows — works for both
    # Parquet-backed and script-generated (camel-ai/math-style) HF datasets
    max_examples = args.max_examples
    ds_subset = ds.take(max_examples)

    fallback_cols = [
        "text",
        "instruction",
        "input",
        "question",
        "content",
        "prompt",
        "message_1",
    ]
    fallback_cols2 = ["output", "answer", "response", "solution", "message_2"]

    batch_texts = []
    format_success = 0
    for example in ds_subset:
        txt = extract_text2(
            example, args.text_column, args.text_column2, fallback_cols + fallback_cols2
        )
        if txt.strip():
            batch_texts.append(txt)
            format_success += 1

    if format_success == 0:
        print(
            "[ERROR] Could not extract any valid text from the dataset. Check column names."
        )
        sys.exit(1)

    print(
        f"Format Match: ✅ successfully extracted text from {format_success}/{max_examples} examples."
    )

    # 2. Forward Pass Profiling
    print(f"\nLoading base model {args.model_id} (4-bit)...")
    load_kwargs_model = dict(
        model_name=args.model_id,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    if args.model_revision:
        load_kwargs_model["revision"] = args.model_revision
    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs_model)
    FastLanguageModel.for_inference(model)
    model.eval()

    interceptor = ActivationInterceptor(target_layer=args.layer)
    interceptor.attach_to_model(model)

    # Handle multimodal tokenizer
    _tok = get_unsloth_base_tokenizer(tokenizer)

    batch_size = 8
    pooled_activations = []

    print("\nRunning prefill passes to extract latent representations...")
    with torch.no_grad():
        for i in range(0, len(batch_texts), batch_size):
            batch = batch_texts[i : i + batch_size]
            enc = _tok(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_seq_len,
            ).to(device)

            model(**enc)

            # act is on CPU (interceptor calls .detach().cpu()); move mask to match
            mask_cpu = enc["attention_mask"].cpu()
            for act in interceptor.intercepted_activations:
                pooled = masked_mean_pool(act, mask_cpu)
                pooled_activations.append(pooled)

            interceptor.intercepted_activations.clear()

    interceptor.detach()
    all_pooled = torch.cat(pooled_activations, dim=0).float()

    # 3. The Overlap Metric
    concept_names = args.concepts.split(",")
    n_concepts = len(concept_names)

    print("\nEvaluating conceptual bleed against SAE weights...")
    if not os.path.exists(sae_weights):
        print(
            f"[ERROR] Trained SAE weights not found at {sae_weights}. Cannot analyze bleed without a trained SAE."
        )
        sys.exit(1)

    state_dict = torch.load(sae_weights, map_location="cpu", weights_only=True)
    # Derive expansion_factor from the saved encoder weight shape so the CBSAE
    # always matches the checkpoint regardless of CLI args.
    # encoder.weight: (d_sae, d_model) → expansion_factor = d_sae / d_model
    saved_expansion = state_dict["encoder.weight"].shape[0] // args.d_model
    sae = CBSAE(
        d_model=args.d_model,
        expansion_factor=saved_expansion,
        bottleneck_features=n_concepts,
    )
    sae.load_state_dict(state_dict)

    sae.eval()

    with torch.no_grad():
        pre_relu = sae.encoder(all_pooled)  # (N, d_sae)
        bottleneck = pre_relu[:, :n_concepts]  # (N, n_concepts)
        # Convert logits into soft probabilities for reporting confidence
        probabilities = torch.softmax(bottleneck, dim=-1)

    avg_confidences = probabilities.mean(dim=0)
    max_confidences, _ = probabilities.max(dim=0)

    # 4. Diagnostic Report
    highest_bleed_idx = avg_confidences.argmax().item()
    highest_bleed_concept = concept_names[highest_bleed_idx]
    highest_bleed_score = avg_confidences[highest_bleed_idx].item()

    print("\n" + "=" * 50)
    print("DIAGNOSTIC REPORT")
    print("=" * 50)
    print(f"Candidate: {args.dataset}")
    print(f"Layer Evaluated: {args.layer}")
    print(f"Samples profiled: {len(batch_texts)} (requested {args.max_examples})")
    print("\n[Average Concept Confidences]")
    for i, name in enumerate(concept_names):
        print(
            f"  {name.capitalize()}: {avg_confidences[i].item():.4f} (Max Peak: {max_confidences[i].item():.4f})"
        )

    print("\n[Analysis]")
    print(
        f"Highest Average Bleed: {highest_bleed_concept.capitalize()} ({highest_bleed_score:.4f})"
    )

    if highest_bleed_score > 0.7:
        print("\n❌ VERDICT: HIGH OVERLAP DANGER")
        print("This dataset frequently triggers an existing concept's routing pathway.")
        print("It acts too similarly to an existing domain (Le Chatelier warning).")
        print(
            "Recommendation: Filter or select an alternative dataset to avoid catastrophic interference."
        )
    elif highest_bleed_score < 0.3:
        print("\n✅ VERDICT: ORTHOGONAL / CLEAN")
        print("This dataset occupies a distinct subspace from existing concepts.")
        print("Recommendation: Safe to harvest and add as a new adapter.")
    else:
        print("\n⚠️ VERDICT: MODERATE OVERLAP")
        print(
            "This dataset has noticeable similarity to existing concepts but is not entirely dominant."
        )
        print(
            "Recommendation: Proceed with caution or increase sample size for better certainty."
        )

    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()

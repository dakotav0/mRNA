"""
HuggingFace Activation Harvester

Streams examples from an HF dataset, runs prefill-only forward passes through a
frozen unsloth model with ActivationInterceptor attached, and saves pooled
activations to disk for CBSAE training.

Designed to be run once per concept, then feed into train_sae.py.

Usage
-----
python sandbox-scripts/harvest_hf.py \
    --dataset camel-ai/biology \
    --text-column message_1 \
    --concept biology \
    --max-examples 5000 \
    --output data/biology_layer17.pt

Output format
-------------
torch.Tensor of shape (N, d_model), FP32.
Activations are mean-pooled over the sequence dimension before saving so they
match mRNAPipeline.route() and train_sae.py without further processing.

Notes
-----
- Uses a single forward pass per example (no generate) — prefill only.
  This is exactly the activation distribution the router will see at inference.
- Saves a checkpoint every --save-every examples so long jobs can be resumed.
  Resume detection: if --output already exists, loads it and continues from N.
- Truncates all inputs to --max-seq-len to cap memory per example.
"""

import argparse
import os
import sys

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import random

from mrna.data.dataset_utils import extract_text, extract_text2
from mrna.data.paths import MRNAPaths
from mrna.router.interceptor import ActivationInterceptor
from mrna.router.pooling import get_unsloth_base_tokenizer, masked_mean_pool


def harvest(
    dataset_id: str,
    dataset_config: str | None,
    split: str,
    text_column: str,
    text_column2: str | None,
    concept: str,
    model_id: str,
    model_revision: str | None,
    layer: int,
    max_seq_len: int,
    max_examples: int,
    batch_size: int,
    save_every: int,
    device: str,
    skip_if_exists: bool = True,
    holdout_ratio: float = 0.0,
):
    paths = MRNAPaths(model_id=model_id)
    train_path = paths.get_activation_file(
        layer, concept, "train" if holdout_ratio > 0 else None
    )
    test_path = (
        paths.get_activation_file(layer, concept, "test") if holdout_ratio > 0 else None
    )

    # ------------------------------------------------------------------
    # Cache / Skip check
    # ------------------------------------------------------------------
    if skip_if_exists and os.path.exists(train_path):
        prev = torch.load(train_path, map_location="cpu", weights_only=True)
        if len(prev) >= max_examples:
            print(
                f"Skipping: {train_path} already contains {len(prev)} examples (requested {max_examples})."
            )
            return

    # ------------------------------------------------------------------
    # Resume detection
    # ------------------------------------------------------------------
    existing_train: list[torch.Tensor] = []
    existing_test: list[torch.Tensor] = []
    already_done = 0

    if os.path.exists(train_path):
        prev = torch.load(train_path, map_location="cpu", weights_only=True)
        existing_train = [prev]
        already_done += len(prev)
    if test_path and os.path.exists(test_path):
        prev_test = torch.load(test_path, map_location="cpu", weights_only=True)
        existing_test = [prev_test]
        already_done += len(prev_test)

    if already_done > 0:
        print(f"Resuming: found {already_done} existing activations for {concept}")

    remaining = max_examples - already_done
    if remaining <= 0:
        print(f"Already have {already_done} examples, nothing to do.")
        return

    # ------------------------------------------------------------------
    # Load dataset (cached locally via Parquet/Arrow to avoid streaming IO bottlenecks)
    # ------------------------------------------------------------------
    print(
        f"Loading dataset '{dataset_id}'{f' (config: {dataset_config})' if dataset_config else ''} in streaming mode..."
    )
    load_kwargs = dict(split=split, streaming=True)
    if dataset_config:
        load_kwargs["name"] = dataset_config
    ds = load_dataset(dataset_id, **load_kwargs)

    # For script-generated datasets (e.g. camel-ai/math), streaming=True is the only way
    # to avoid running the full generator. skip() + take() handle resume correctly.
    if already_done > 0:
        ds = ds.skip(already_done)
    ds = ds.take(remaining)

    # ------------------------------------------------------------------
    # Load frozen base model
    # ------------------------------------------------------------------
    rev_label = f" @ {model_revision[:8]}" if model_revision else ""
    print(f"Loading base model '{model_id}'{rev_label} (4-bit)...")
    load_kwargs = dict(
        model_name=model_id, max_seq_length=max_seq_len, dtype=None, load_in_4bit=True
    )
    if model_revision:
        load_kwargs["revision"] = model_revision
    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    FastLanguageModel.for_inference(model)
    model.eval()

    # ------------------------------------------------------------------
    # Attach interceptor
    # ------------------------------------------------------------------
    interceptor = ActivationInterceptor(target_layer=layer)
    interceptor.attach_to_model(model)

    # ------------------------------------------------------------------
    # Stream and harvest
    # ------------------------------------------------------------------
    fallback_cols = ["text", "instruction", "input", "question", "content", "prompt"]
    collected_train: list[torch.Tensor] = []
    collected_test: list[torch.Tensor] = []
    n_processed = 0
    n_saved_total = already_done

    print(f"\nHarvesting {remaining} examples for concept '{concept}' ...")
    print(
        f"  Layer: {layer}  |  max_seq_len: {max_seq_len}  |  batch_size: {batch_size}\n"
    )
    if holdout_ratio > 0:
        print(
            f"  Holdout Enabled: {holdout_ratio * 100}% of activations going to holdout test set."
        )

    batch_texts: list[str] = []

    def flush_batch():
        nonlocal n_processed, n_saved_total

        if not batch_texts:
            return

        _tok = get_unsloth_base_tokenizer(tokenizer)
        enc = _tok(
            text=batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to(device)

        with torch.no_grad():
            model(**enc)

        for act in interceptor.intercepted_activations:
            pooled = masked_mean_pool(act, enc["attention_mask"].cpu())
            for i in range(len(pooled)):
                if holdout_ratio > 0 and random.random() < holdout_ratio:
                    collected_test.append(pooled[i].unsqueeze(0))
                else:
                    collected_train.append(pooled[i].unsqueeze(0))

        interceptor.intercepted_activations.clear()
        batch_texts.clear()

        n_processed += 1
        n_saved_total = already_done + len(collected_train) + len(collected_test)

        if n_saved_total % save_every < batch_size:
            _save(train_path, existing_train, collected_train)
            if test_path:
                _save(test_path, existing_test, collected_test)
            print(f"  Checkpoint: {n_saved_total} / {already_done + remaining} total")

    for example in ds:
        if len(collected_train) + len(collected_test) >= remaining:
            break

        text = extract_text2(example, text_column, text_column2, fallback_cols)
        if not text.strip():
            continue

        batch_texts.append(text)

        if len(batch_texts) >= batch_size:
            flush_batch()
            print(
                f"  {already_done + len(collected_train) + len(collected_test):>5} / {already_done + remaining}"
            )

    # Final partial batch
    flush_batch()

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    interceptor.detach()
    _save(train_path, existing_train, collected_train)
    if test_path:
        _save(test_path, existing_test, collected_test)

    total = already_done + len(collected_train) + len(collected_test)
    print(f"\nDone. Saved {total} pooled activations ({model_id}, layer {layer})")
    print(
        f"  Train: {len(collected_train) + (len(existing_train[-1]) if existing_train else 0)} elements -> {train_path}"
    )
    if test_path:
        print(
            f"  Test: {len(collected_test) + (len(existing_test[-1]) if existing_test else 0)} elements -> {test_path}"
        )
    print(
        f"\nNext step:\n  python sandbox-scripts/train_sae.py "
        f"--activations {concept}:{train_path} ..."
    )


def _save(path: str, existing: list, new: list):
    all_acts = existing + new
    combined = torch.cat(all_acts, dim=0)
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(combined, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Harvest pooled activations from an HF dataset using prefill-only forward passes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="HF dataset ID")
    parser.add_argument(
        "--dataset-config", default=None, help="HF dataset config/subset name"
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--text-column", default="text", help="Column name to use as prompt text"
    )
    parser.add_argument(
        "--text-column2",
        default=None,
        help="Optional second column to concatenate (e.g. message_2 for Q+A). "
        "Provides richer domain signal for SAE boundary disambiguation.",
    )
    parser.add_argument(
        "--concept",
        required=True,
        help="Concept name for this dataset (e.g. python, legal)",
    )
    parser.add_argument("--model-id", default="unsloth/gemma-4-E2B-it")
    parser.add_argument(
        "--model-revision",
        default="37ea165b3fba25b7d851f8ce4ccff9a4f0751cee",
        help="HF commit hash to pin. Prevents silent weight updates from breaking "
        "harvest/SAE/GGUF consistency. Matches the revision GGUF was built from.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=25,
        help="Transformer layer to intercept (25 = 70% depth for 35-layer Gemma 4 E2B; "
        "layer 17 midpoint gives chem/physics boundary ambiguity)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Truncate inputs to this length (controls VRAM per batch)",
    )
    parser.add_argument("--max-examples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Checkpoint to disk every N examples",
    )
    parser.add_argument(
        "--no-skip",
        action="store_false",
        dest="skip_if_exists",
        help="Disable automatic skipping if the output file already exists",
    )
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.2,
        help="Fraction of examples to save into a distinct '_test.pt' file",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    paths = MRNAPaths(model_id=args.model_id)
    print("\n--- mRNA Activation Harvesting Tool ---")
    print(f"Concept: {args.concept}")
    print(f"Output Directory: {paths.get_activations_dir(args.layer)}")

    harvest(
        dataset_id=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        text_column=args.text_column,
        text_column2=args.text_column2,
        concept=args.concept,
        model_revision=args.model_revision,
        model_id=args.model_id,
        layer=args.layer,
        max_seq_len=args.max_seq_len,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        save_every=args.save_every,
        device=args.device,
        skip_if_exists=args.skip_if_exists,
        holdout_ratio=args.holdout_ratio,
    )


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception as e:
        print("\n[FATAL] Script crashed at startup!")
        traceback.print_exc()
        sys.exit(1)

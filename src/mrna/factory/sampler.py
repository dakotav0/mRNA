import os

os.environ["UNSLOTH_FORCE_SDPA"] = "1"
import random
import warnings
from pathlib import Path
from typing import List, Optional

import datasets
import torch
from tqdm import tqdm

from mrna.core.config import MRNAPaths, config
from mrna.data.dataset_utils import extract_text
from mrna.router.interceptor import ActivationInterceptor
from mrna.router.pooling import get_unsloth_base_tokenizer, masked_mean_pool
from mrna.substrate.backend import get_backend

# Silence the deprecated attention mask API warning from transformers/unsloth
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.modeling_attn_mask_utils"
)


def harvest_activations(
    concept: str,
    dataset_id: Optional[str] = None,
    model_id: Optional[str] = None,
    layer: Optional[int] = None,
    max_examples: Optional[int] = None,
    holdout_ratio: float = 0.2,
    batch_size: Optional[int] = None,
    save_every: int = 500,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    skip_if_exists: bool = True,
    **kwargs,
):
    """
    Generalized activation harvester supporting multi-layer collection.
    """
    backend = get_backend("torch")
    mid = model_id or config.current_model_id
    m_cfg = config.get_model_config(mid)

    # 1. Resolve pipeline arguments
    max_examples = config.get_pipeline_arg("harvest", "max_examples", mid, max_examples)
    batch_size = config.get_pipeline_arg("harvest", "batch_size", mid, batch_size)
    max_seq_len = config.get_pipeline_arg(
        "harvest", "max_seq_len", mid, kwargs.get("max_seq_len")
    )

    ds_info = config.approved_datasets.get(concept, {})
    ds_id = dataset_id or ds_info.get("id")

    # NEW: Multilayer support
    target_layers = [layer] if layer is not None else config.get_harvest_layers(mid)
    if not target_layers:
        raise ValueError(f"No harvest layers configured for model {mid}")

    # 2. Setup paths (Root activations dir, not layer-specific yet)
    activations_root = MRNAPaths.DATA / mid / "activations"
    activations_root.mkdir(parents=True, exist_ok=True)

    # 3. Quick check for existing data (only if single layer specified)
    if skip_if_exists and len(target_layers) == 1:
        ldir = f"layer_{target_layers[0]}"
        check_path = MRNAPaths.get_activations_dir(mid, ldir) / f"{concept}_train.pt"
        if check_path.exists():
            print(f"Skipping: {check_path} already exists.")
            return

    # 4. Load Model
    model_path = m_cfg["path"]
    print(
        f"Loading model {mid} for multi-layer harvesting (Layers: {target_layers})..."
    )
    use_unsloth = kwargs.get("use_unsloth", True)
    model, tokenizer = backend.load_model(
        model_path,
        max_seq_length=max_seq_len,
        load_in_4bit=True,
        use_unsloth=use_unsloth,
    )
    if use_unsloth:
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(model)

    model.eval()

    # 5. Attach Interceptor (Now handles multiple layers)
    interceptor = ActivationInterceptor(target_layers=target_layers)
    interceptor.attach_to_model(model)

    # 6. Load Dataset
    from mrna.data.dataset_utils import load_smart_dataset

    ds, streaming, _ = load_smart_dataset(
        ds_id,
        root_path=config.datasets_path,
        download=kwargs.get("download", False),
        split="train",
    )

    # 7. Harvest Loop
    collected_train = {layer: [] for layer in target_layers}
    collected_test = {layer: [] for layer in target_layers}
    batch_texts = []
    fallback_cols = ["text", "instruction", "input", "question", "content", "prompt"]

    # Deterministic seed for reproducible holdouts across runs/layers
    rng = random.Random(42)

    def _flush():
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

        # Interceptor now has a dict of {layer: [tensors]}
        # We process each layer's results
        for layer_idx in target_layers:
            acts = interceptor.intercepted_activations[layer_idx]
            if not acts:
                continue

            # Combine batch segments into one tensor
            act_batch = torch.cat(acts, dim=0) if len(acts) > 1 else acts[0]
            pooled = masked_mean_pool(
                act_batch, enc["attention_mask"].to(act_batch.device)
            )

            # Deterministically split indices for this batch
            indices = list(range(len(pooled)))
            for i in indices:
                is_holdout = holdout_ratio > 0 and rng.random() < holdout_ratio
                if is_holdout:
                    collected_test[layer_idx].append(pooled[i].unsqueeze(0))
                else:
                    collected_train[layer_idx].append(pooled[i].unsqueeze(0))

            # Clear interceptor's temporary buffer for this layer
            acts.clear()

        nonlocal total_harvested
        total_harvested += len(batch_texts)
        batch_texts.clear()

        # Periodic save logic (if we have enough in RAM)
        total_in_ram = len(next(iter(collected_train.values()))) + len(
            next(iter(collected_test.values()))
        )
        if total_in_ram >= save_every:
            _save_checkpoint()

    def _save_checkpoint():
        interceptor.intercepted_activations = (
            collected_train  # Temporary swap for naming helper
        )
        interceptor.save_harvested_dataset(
            str(activations_root), concept, is_test=False
        )

        interceptor.intercepted_activations = collected_test
        interceptor.save_harvested_dataset(str(activations_root), concept, is_test=True)

        # Restore empty references (save_harvested_dataset clears them)
        for layer_idx in target_layers:
            collected_train[layer_idx] = []
            collected_test[layer_idx] = []

    print(f"Harvesting {max_examples} examples...")
    pbar = tqdm(total=max_examples, desc=f"[{concept}]")

    iterator = ds if hasattr(ds, "__iter__") else ds.to_iterable_dataset()

    # Concatenation logic: prefer user flag, then config column
    col1 = kwargs.get("text_column") or ds_info.get("column", "message_1")
    col2 = kwargs.get("text_column2")
    pref_cols = [col1, col2] if col2 else col1

    total_harvested = 0
    for example in iterator:
        text = extract_text(example, pref_cols, fallback_cols)
        if not text or not text.strip():
            continue

        batch_texts.append(text)

        if len(batch_texts) >= batch_size:
            _flush()
            pbar.n = min(total_harvested, max_examples)
            pbar.refresh()

        if total_harvested >= max_examples:
            break

    _flush()
    # One last update for any remaining in the final flush (before save clears it)
    final_batch = len(next(iter(collected_train.values()))) + len(
        next(iter(collected_test.values()))
    )
    total_harvested += final_batch

    _save_checkpoint()
    pbar.n = min(total_harvested, max_examples)
    pbar.refresh()
    pbar.close()

    interceptor.detach()
    print(f"Done. Outputs in {activations_root}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="mRNA Activation Harvester")
    parser.add_argument("--concept", required=True, help="Concept pass (biology, etc.)")
    parser.add_argument("--dataset", help="HF Dataset ID or local path override")
    parser.add_argument("--model", help="Model ID from config")
    parser.add_argument("--max-examples", type=int, help="Limit total examples")
    parser.add_argument("--batch-size", type=int, help="GPU batch size")
    parser.add_argument("--layer", type=int, help="Layer index (0-indexed)")
    parser.add_argument("--layer-name", help="Custom layer directory name")
    parser.add_argument("--text-column", help="Primary text column")
    parser.add_argument(
        "--text-column2", help="Secondary text column for concatenation"
    )
    parser.add_argument("--max-seq-len", type=int, help="Context length")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download dataset locally first (no streaming)",
    )
    parser.add_argument("--no-skip", action="store_false", dest="skip")
    parser.add_argument(
        "--no-unsloth",
        action="store_false",
        dest="use_unsloth",
        help="Bypass Unsloth (disables Triton kernels)",
    )
    parser.set_defaults(skip=True, use_unsloth=True)

    args = parser.parse_args()

    harvest_activations(
        concept=args.concept,
        dataset_id=args.dataset,
        model_id=args.model,
        layer=args.layer,
        layer_name=args.layer_name,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        text_column=args.text_column,
        text_column2=args.text_column2,
        skip_if_exists=args.skip,
        use_unsloth=args.use_unsloth,
        streaming=not args.download,
    )


if __name__ == "__main__":
    main()

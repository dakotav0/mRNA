import os

os.environ["UNSLOTH_FORCE_SDPA"] = "1"
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets
import torch
from datasets import Dataset as HFDataset
from tqdm import tqdm
from transformers import TrainingArguments

from mrna.core.config import MRNAPaths, config
from mrna.execution.adapter_convert import convert_adapter

# Formatting logic moved to mrna.data.dataset_utils


def _unwrap_clippable_gemma(model, target_modules: List[str]):
    """PEFT compatibility for Gemma 4."""
    import torch.nn as nn

    candidates = [
        (n, m)
        for n, m in model.named_modules()
        if any(tgt in n for tgt in target_modules)
        and hasattr(m, "linear")
        and isinstance(m.linear, nn.Linear)
    ]
    for name, mod in candidates:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], mod.linear)
    if candidates:
        print(f"[Factory] Unwrapped {len(candidates)} Gemma4Clippable layers.")


def _truncate_batch(batch, tokenizer, formatter, max_seq_length):
    """Clean truncation function that avoids capturing environment closures."""
    texts = formatter(batch)["text"]
    out = []
    for t in texts:
        ids = tokenizer(text=t, truncation=True, max_length=max_seq_length - 1)[
            "input_ids"
        ]
        out.append(tokenizer.decode(ids, skip_special_tokens=False))
    return {"text": out}


def train_adapter(
    concept: str,
    model_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    max_steps: Optional[int] = None,
    rank: Optional[int] = None,
    lr: Optional[float] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    convert: bool = False,
    **kwargs,
):
    """
    Core LoRA training orchestration.
    """
    from trl import SFTTrainer
    from unsloth import FastLanguageModel, is_bfloat16_supported

    try:
        from unsloth import FastVisionModel
    except ImportError:
        FastVisionModel = None

    mid = model_id or config.current_model_id
    m_cfg = config.get_model_config(mid)

    # Detect multimodal Gemma 4 (Conditional Generation)
    is_gemma4 = "gemma-4" in mid.lower() or "gemma4" in mid.lower()

    # 1. Resolve pipeline arguments with sensible fallbacks
    max_steps = int(config.get_pipeline_arg("adapter", "steps", mid, max_steps) or 200)
    rank = int(config.get_pipeline_arg("adapter", "rank", mid, rank) or 16)
    lr = float(config.get_pipeline_arg("adapter", "lr", mid, lr) or 2e-4)
    batch_size = int(
        config.get_pipeline_arg("adapter", "batch_size", mid, kwargs.get("batch_size"))
        or 1
    )
    grad_accum = int(
        config.get_pipeline_arg("adapter", "grad_accum", mid, kwargs.get("grad_accum"))
        or 8
    )
    max_seq_length = int(
        config.get_pipeline_arg(
            "harvest", "max_seq_len", mid, kwargs.get("max_seq_len")
        )
        or 512
    )
    max_examples = int(
        config.get_pipeline_arg(
            "adapter", "max_examples", mid, kwargs.get("max_examples")
        )
        or 2000
    )

    ds_info = config.approved_datasets.get(concept, {})
    ds_id = dataset_id or ds_info.get("id") or "camel-ai/biology"
    output_dir = kwargs.get("output_dir") or str(
        MRNAPaths.get_adapter_dir(concept, mid)
    )

    # 1. Load Model
    model_name = m_cfg.get("path", mid)
    print(f"Loading base model {mid} (from {model_name}) for adapter training...")

    loader = FastVisionModel if (is_gemma4 and FastVisionModel) else FastLanguageModel

    model, tokenizer = loader.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    # PEFT Setup
    peft_kwargs = {
        "model": model,
        "r": 16,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "lora_alpha": 16,
        "lora_dropout": 0,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": 3407,
    }

    # Critical fix for Gemma 4 Multimodal:
    # 1. Disable vision layers in PEFT
    # 2. Unwrap clippable layers to satisfy PEFT type checks
    if is_gemma4:
        peft_kwargs["finetune_vision_layers"] = False
        _target_list = peft_kwargs["target_modules"]
        _unwrap_clippable_gemma(model, _target_list)

    model = loader.get_peft_model(**peft_kwargs)

    # 2. Tokenizer Setup
    _tok = tokenizer
    if hasattr(tokenizer, "tokenizer"):
        _tok = tokenizer.tokenizer

    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token

    # 3. Process Dataset
    from mrna.data.dataset_utils import get_dataset_formatter, load_smart_dataset

    dataset, streaming, res_id = load_smart_dataset(
        ds_id,
        root_path=config.datasets_path,
        download=kwargs.get("download", False),
        split=kwargs.get("split", "train"),
    )

    formatter = get_dataset_formatter(res_id, _tok.eos_token, kwargs.get("text_column"))

    if streaming:
        print(f"Sampling {max_examples} examples from stream...")
        rows = list(
            tqdm(
                dataset.take(max_examples),
                desc="Sampling (Streaming)",
                total=max_examples,
            )
        )
        dataset = HFDataset.from_list(rows)
    else:
        print(f"Subsetting {max_examples} examples from local dataset...")
        if len(dataset) > max_examples:
            dataset = dataset.select(range(max_examples))

    original_columns = dataset.column_names

    # Use partial to avoid closure capture of 'config' or other locals
    map_func = partial(
        _truncate_batch,
        tokenizer=_tok,
        formatter=formatter,
        max_seq_length=max_seq_length,
    )

    dataset = dataset.map(
        map_func,
        batched=True,
        remove_columns=original_columns,
        desc="Formatting & Truncating",
    )

    # Pre-tokenize in the main process (no num_proc → in-process, no subprocess, no pickling).
    # SFTTrainer checks `"input_ids" in column_names` to decide if the dataset is already processed.
    # When True it skips its own multiprocessing _prepare_dataset step entirely, which is what
    # caused the ConfigModuleInstance pickle error via the tokenize_fn closure capturing `self`.
    dataset = dataset.map(
        lambda batch: _tok(
            batch["text"], truncation=True, max_length=max_seq_length, padding=False
        ),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    print(f"Dataset prepared: {len(dataset)} rows.")

    # 4. Trainer Setup
    # BUGFIX: Newer TRL versions require SFTConfig for SFT-specific params.
    try:
        from trl import SFTConfig

        trainer_args = SFTConfig(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=10,
            max_steps=max_steps,
            learning_rate=lr,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            output_dir=os.path.join(output_dir, "checkpoints"),
            report_to="none",
            max_seq_length=max_seq_length,
            # dataset_text_field omitted — dataset is already tokenized (has input_ids)
        )
    except ImportError:
        # Fallback for slightly older versions that use TrainingArguments + direct args
        # (Though we'll simplify it to help it pass)
        trainer_args = TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=10,
            max_steps=max_steps,
            learning_rate=lr,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            output_dir=os.path.join(output_dir, "checkpoints"),
            report_to="none",
        )

    # Must be set before SFTTrainer.__init__: Unsloth reads it during init to disable
    # padding-free mode, and reads it again in the model forward pass to return real
    # logit tensors. TRL's compute_loss calls entropy_from_logits(outputs.logits) and
    # crashes if Unsloth returns a lazy callable instead of a tensor.
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=trainer_args,
        processing_class=_tok,
    )

    print(f"Starting training for {concept}...")
    trainer.train()

    # 5. Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter saved to {output_dir}")

    if convert:
        print(f"[Factory] Triggering GGUF conversion for {concept}...")
        convert_adapter(concept=concept, model_id=mid)

    return output_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description="mRNA Adapter Trainer")
    parser.add_argument("--concept", required=True, help="Concept pass (biology, etc.)")
    parser.add_argument(
        "--model", help="Model ID override (resolves via config by default)"
    )
    parser.add_argument("--dataset", help="HF Dataset ID override")
    parser.add_argument("--steps", type=int, help="Training steps override")
    parser.add_argument("--batch-size", type=int, help="Per-device batch size override")
    parser.add_argument("--grad-accum", type=int, help="Gradient accumulation override")
    parser.add_argument("--lr", type=float, help="Learning rate override")
    parser.add_argument("--rank", type=int, help="LoRA rank override")
    parser.add_argument("--max-examples", type=int, help="Sampling limit override")
    parser.add_argument("--max-seq-len", type=int, help="Context window override")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download dataset to local cache before training",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Automatically convert to GGUF after training",
    )

    args = parser.parse_args()

    train_adapter(
        concept=args.concept,
        model_id=args.model,
        dataset_id=args.dataset,
        max_steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        rank=args.rank,
        max_examples=args.max_examples,
        max_seq_len=args.max_seq_len,
        download=args.download,
        convert=args.convert,
    )


if __name__ == "__main__":
    main()

"""
.mrna Adapter Training — Biology Concept

Trains a rank-16 LoRA adapter on camel-ai/biology using Unsloth + SFTTrainer.
Saves to adapters/biology_lora in the HuggingFace PEFT format that vLLM's
LoRARequest can load directly.

Usage
-----
    python sandbox-scripts/train_mrna_adapter.py

    # Different concept (any dataset with message_1/message_2 or alpaca columns):
    python sandbox-scripts/train_mrna_adapter.py \
        --dataset camel-ai/biology \
        --concept biology \
        --max-steps 200

Output
------
    adapters/biology_lora/
        adapter_config.json
        adapter_model.safetensors   ← loaded by vLLM LoRARequest

Then wire into the pipeline:
    LoRARequest("biology", concept_idx+1, "adapters/biology_lora")

.mrna constraints honoured
--------------------------
- rank=16           small enough to stream over PCIe (rank × d_model × 2 layers ≈ 4MB)
- target_modules    q_proj + v_proj only (early-to-middle layer steering per SKILL.md)
- 4-bit base model  preserves 4.2GB VRAM headroom for Engram + adapter hot-swap
- packing=True      Unsloth strict causal masking — no cross-document attention bleed
- output saved in PEFT format (not merged) — vLLM loads delta weights, not merged model
"""

import argparse
import os
import sys

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

# Liger-Kernel: fused Triton kernels for RMSNorm, SwiGLU, and FusedLinearCrossEntropy.
# FusedLinearCE is the critical one — never materializes (batch × seq × 262144) logits,
# computes chunk-by-chunk. Replaces the unsloth_num_chunks workaround.
# rope=False: Unsloth owns the RoPE path; let it handle that.
try:
    from liger_kernel.transformers import apply_liger_kernel_to_gemma

    _LIGER_AVAILABLE = True
except ImportError:
    _LIGER_AVAILABLE = False
    print("[WARNING] liger-kernel not installed. Run: pip install liger-kernel")

try:
    from muon import MuonWithAuxAdam

    _MUON_AVAILABLE = True
except ImportError:
    _MUON_AVAILABLE = False
    print(
        "[WARNING] Muon not installed. Run: pip install git+https://github.com/KellerJordan/Muon"
    )

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

CHAT_PROMPT = """Below is a question about {topic}. Write a thorough and accurate response.

### Question:
{question}

### Response:
{answer}"""

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


def make_formatter(
    dataset_id: str, tokenizer, text_column: str | None = None, split: str = "train"
):
    """Return a batched formatting function appropriate for the dataset columns.

    text_column: if set, train on that single column (e.g. 'message_1' for
    questions-only). Useful for dense datasets (physics) where Q+A concatenation
    OOMs at constrained seq lengths.
    """

    def _single_column_format(examples):
        return {"text": [t + tokenizer.eos_token for t in examples[text_column]]}

    def _chat_format(examples):
        topics = examples.get("topic", [""] * len(examples["message_1"]))
        texts = []
        for topic, q, a in zip(topics, examples["message_1"], examples["message_2"]):
            text = (
                CHAT_PROMPT.format(
                    topic=topic or "biology",
                    question=q,
                    answer=a,
                )
                + tokenizer.eos_token
            )
            texts.append(text)
        return {"text": texts}

    def _alpaca_format(examples):
        texts = []
        for instr, inp, out in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            text = ALPACA_PROMPT.format(instruction=instr, input=inp, output=out)
            text += tokenizer.eos_token
            texts.append(text)
        return {"text": texts}

    # Probe which format the dataset uses
    sample = load_dataset(dataset_id, split=split, streaming=True).take(1)
    sample = list(sample)
    cols = list(sample[0].keys())

    if text_column is not None:
        if text_column not in cols:
            raise ValueError(
                f"--text-column {text_column!r} not found. Available: {cols}"
            )
        print(f"[Formatter] Single-column mode: '{text_column}'")
        return _single_column_format
    elif "message_1" in cols and "message_2" in cols:
        print("[Formatter] Detected chat format (message_1 / message_2)")
        return _chat_format
    elif "instruction" in cols:
        print("[Formatter] Detected alpaca format (instruction / input / output)")
        return _alpaca_format
    else:
        raise ValueError(
            f"Unknown dataset format. Columns: {cols}\n"
            "Expected 'message_1'/'message_2' or 'instruction'/'input'/'output'."
        )


# ---------------------------------------------------------------------------
# Gemma 4 LoRA compatibility
# ---------------------------------------------------------------------------


def _unwrap_clippable(model, target_module_names: list[str]) -> None:
    """Replace Gemma4ClippableLinear wrappers with their inner nn.Linear.

    PEFT's module allowlist only accepts torch.nn.Linear (and a few other
    standard types).  Gemma 4 wraps q_proj / v_proj in Gemma4ClippableLinear
    which fails that check.  Swapping in the inner .linear is safe: the
    forward() of Gemma4ClippableLinear is `return self.linear(x)` and the
    clipping path is only exercised during quantisation, not fine-tuning.

    This is a no-op on models that don't use Gemma4ClippableLinear.
    """
    import torch.nn as nn

    # Snapshot list — modifying model while iterating named_modules() is safe
    # with list() but we snapshot to be explicit.
    candidates = [
        (full_name, module)
        for full_name, module in model.named_modules()
        if any(
            full_name == tgt or full_name.endswith(f".{tgt}")
            for tgt in target_module_names
        )
        and hasattr(module, "linear")
        and isinstance(module.linear, nn.Linear)
    ]

    if not candidates:
        return

    for full_name, module in candidates:
        parts = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module.linear)

    print(
        f"[LoRA compat] Unwrapped {len(candidates)} Gemma4ClippableLinear "
        f"module(s) → nn.Linear for PEFT compatibility."
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    dataset_id: str,
    concept: str,
    model_id: str,
    model_revision: str | None,
    max_seq_length: int,
    rank: int,
    max_steps: int,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    output_dir: str,
    packing: bool = True,
    text_column: str | None = None,
    max_examples: int | None = None,
    split: str = "train",
):
    adapter_path = os.path.join(output_dir, f"{concept}_lora")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load frozen base model (4-bit, preserves VRAM headroom)
    # ------------------------------------------------------------------
    if _LIGER_AVAILABLE:
        apply_liger_kernel_to_gemma(
            rope=False,  # Unsloth owns RoPE — no double-patch
            rms_norm=True,
            cross_entropy=False,  # replaced by fused_linear_cross_entropy
            fused_linear_cross_entropy=True,  # never materializes (batch×seq×262144) tensor
        )
        print(
            "[Substrate] Liger-Kernel patches applied (FusedLinearCE + RMSNorm + SwiGLU)"
        )

    rev_label = f" @ {model_revision[:8]}" if model_revision else ""
    print(f"\nLoading base model {model_id!r}{rev_label}...")
    load_kwargs = dict(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    if model_revision:
        load_kwargs["revision"] = model_revision
    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    # Gemma 4's tokenizer is a Gemma4Processor (multimodal). SFTTrainer and
    # _truncate need the inner text tokenizer; formatters/save still use processor.
    _tok = getattr(tokenizer, "tokenizer", tokenizer)

    # ------------------------------------------------------------------
    # 2. Inject .mrna LoRA adapter
    #    rank=16, q_proj+v_proj only — minimal payload for streaming
    #    (SKILL.md: early-to-middle layer steering is sufficient)
    # ------------------------------------------------------------------

    # Gemma 4: q_proj / v_proj are wrapped in Gemma4ClippableLinear, a custom
    # nn.Module that PEFT's isinstance check doesn't recognize as a valid LoRA
    # target (only torch.nn.Linear and a fixed allowlist pass).
    # The wrapper's forward() is just `return self.linear(x)` — clipping is
    # only needed at quantisation time — so swapping in the inner nn.Linear
    # before get_peft_model() is safe for fine-tuning.
    _TARGET_MODULES = ["q_proj", "v_proj"]
    _unwrap_clippable(model, _TARGET_MODULES)

    print(f"Injecting .mrna LoRA (rank={rank}, target: q_proj + v_proj)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        target_modules=["q_proj", "v_proj"],  # minimal target set for streaming
        lora_alpha=rank,  # scale = alpha/rank = 1.0
        lora_dropout=0,  # required for Unsloth Triton kernels
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # ------------------------------------------------------------------
    # 3. Load and format dataset
    # ------------------------------------------------------------------
    print(f"\nLoading dataset {dataset_id!r}...")
    formatter = make_formatter(
        dataset_id, tokenizer, text_column=text_column, split=split
    )
    from datasets import Dataset as HFDataset
    from tqdm import tqdm

    ds_stream = load_dataset(dataset_id, split=split, streaming=True)
    cap = max_examples or 50_000
    rows = list(
        tqdm(ds_stream.take(cap), total=cap, desc="Streaming dataset", unit="ex")
    )
    dataset = HFDataset.from_list(rows)
    print(f"Dataset: {len(dataset)} examples loaded")

    # Pre-map to 'text' then tokenize-truncate-decode so unsloth's fused CE loss
    # never sees inputs longer than max_seq_length. This avoids the input/label
    # batch-size mismatch that fires when unsloth silently truncates inputs but
    # not labels (happens with both packing=True and packing=False on long Q+A).
    def _truncate(batch):
        texts = formatter(batch)["text"]
        out = []
        for t in texts:
            ids = _tok(text=t, truncation=True, max_length=max_seq_length - 1)[
                "input_ids"
            ]
            out.append(_tok.decode(ids, skip_special_tokens=False))
        return {"text": out}

    dataset = dataset.map(_truncate, batched=True)
    sft_kwargs = dict(train_dataset=dataset, dataset_text_field="text")

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    effective_batch = batch_size * grad_accum
    print(f"\nTraining .mrna adapter for concept '{concept}'")
    print(
        f"  steps={max_steps}  batch={batch_size}×{grad_accum}={effective_batch}  lr={learning_rate}"
    )
    print(f"  packing={packing}\n")

    # Muon optimizer: LoRA A/B matrices are ndim=2 non-embedding weights — exactly the
    # case where orthogonality pressure helps most. ~30% fewer steps to same val loss.
    # Falls back to adamw_8bit if Muon isn't installed.
    if _MUON_AVAILABLE:
        hidden = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad
            and p.ndim >= 2
            and "embed" not in n
            and "lm_head" not in n
        ]
        other = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad
            and not (p.ndim >= 2 and "embed" not in n and "lm_head" not in n)
        ]
        muon_optimizer = MuonWithAuxAdam(
            [
                dict(params=hidden, use_muon=True, lr=0.02, weight_decay=0.01),
                dict(
                    params=other,
                    use_muon=False,
                    lr=learning_rate,
                    betas=(0.9, 0.95),
                    weight_decay=0.01,
                ),
            ]
        )
        print(
            f"[Substrate] Muon optimizer: {len(hidden)} hidden matrices, {len(other)} other params"
        )
        optimizers = (muon_optimizer, None)
        optim_arg = "adamw_8bit"  # ignored when optimizers= is set, but required by TrainingArguments
    else:
        optimizers = (None, None)
        optim_arg = "adamw_8bit"

    trainer = SFTTrainer(
        model=model,
        processing_class=_tok,
        **sft_kwargs,
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=packing,
        optimizers=optimizers,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=10,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim=optim_arg,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=os.path.join(output_dir, "checkpoints"),
            report_to="none",
        ),
    )

    trainer_stats = trainer.train()
    print("\nTraining complete.")
    print(f"  Runtime : {trainer_stats.metrics['train_runtime']:.1f}s")
    print(f"  Loss    : {trainer_stats.metrics['train_loss']:.4f}")

    # ------------------------------------------------------------------
    # 5. Save in PEFT format — vLLM LoRARequest loads delta weights directly
    #    Do NOT save merged model (that defeats streaming architecture)
    # ------------------------------------------------------------------
    print(f"\nSaving .mrna adapter (PEFT delta format) → {adapter_path}")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    adapter_size_mb = (
        sum(
            os.path.getsize(os.path.join(adapter_path, f))
            for f in os.listdir(adapter_path)
        )
        / 1e6
    )
    print(f"Adapter size: {adapter_size_mb:.1f} MB  (streaming viable at PCIe Gen4)")

    print(f"""
.mrna adapter ready. Wire into pipeline:

  from vllm.lora.request import LoRARequest
  lora_request = LoRARequest("{concept}", concept_idx+1, "{adapter_path}")

  # Or via mRNAWorker / MergedStreamingExecutionNode:
  ADAPTER_REGISTRY["{concept}"] = "{adapter_path}"
""")

    return adapter_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train a .mrna LoRA adapter on a domain-specific dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default="camel-ai/biology")
    parser.add_argument(
        "--concept",
        default="biology",
        help="Concept name — sets output dir: adapters/<concept>_lora",
    )
    parser.add_argument("--model-id", default="unsloth/gemma-4-E2B-it")
    parser.add_argument(
        "--model-revision",
        default="37ea165b3fba25b7d851f8ce4ccff9a4f0751cee",
        help="HF commit hash to pin (prevents silent weight updates).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="128 required for Gemma 4 E2B (vocab=262,144 causes fused CE OOM at seq>128 on 12GB VRAM).",
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="200 steps ≈ 5-10 min on 4070 Super. "
        "Use 500+ for a more polished adapter.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Reduced to 1 from 2 to preserve VRAM during training",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Increased to 8 to maintain effective batch size of 8",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--output-dir",
        default="data/gemma-4-e2b/adapters",
    )
    parser.add_argument(
        "--packing",
        action="store_true",
        help="Enable sequence packing (disabled by default — pre-truncation handles length).",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help="Train on a single column (e.g. 'message_1') instead of full Q+A. "
        "Halves sequence length for OOM-prone datasets.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Cap dataset rows via streaming take(). Essential for script-generated "
        "HF datasets like camel-ai/math that would otherwise download all 50k rows.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use. Some datasets (e.g. HuggingFaceH4/MATH-500) only have 'test'.",
    )

    args = parser.parse_args()

    train(
        dataset_id=args.dataset,
        concept=args.concept,
        model_id=args.model_id,
        model_revision=args.model_revision,
        max_seq_length=args.max_seq_len,
        rank=args.rank,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        packing=args.packing,
        text_column=args.text_column,
        max_examples=args.max_examples,
        split=args.split,
    )


if __name__ == "__main__":
    main()

import os
import torch
from datasets import load_dataset, Dataset as HFDataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from typing import Optional, Dict, Any, List
from tqdm import tqdm

from mrna.core.config import config, MRNAPaths
from mrna.substrate.backend import get_backend

# ---------------------------------------------------------------------------
# Formatting Templates
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

def get_dataset_formatter(dataset_id: str, tokenizer, text_column: Optional[str] = None):
    """Probes dataset and returns appropriate formatting function."""
    sample = list(load_dataset(dataset_id, split="train", streaming=True).take(1))
    cols = list(sample[0].keys())

    if text_column:
        return lambda ex: {"text": [t + tokenizer.eos_token for t in ex[text_column]]}
    elif "message_1" in cols and "message_2" in cols:
        return lambda ex: {"text": [CHAT_PROMPT.format(topic=ex.get("topic", [""])[i] or "science", 
                                                      question=q, answer=a) + tokenizer.eos_token 
                                   for i, (q, a) in enumerate(zip(ex["message_1"], ex["message_2"]))]}
    elif "instruction" in cols:
        return lambda ex: {"text": [ALPACA_PROMPT.format(instruction=i, input=inp, output=o) + tokenizer.eos_token 
                                   for i, inp, o in zip(ex["instruction"], ex["input"], ex["output"])]}
    else:
        raise ValueError(f"Unknown dataset format for {dataset_id}. Columns: {cols}")

def _unwrap_clippable_gemma(model, target_modules: List[str]):
    """PEFT compatibility for Gemma 4."""
    import torch.nn as nn
    candidates = [
        (n, m) for n, m in model.named_modules()
        if any(tgt in n for tgt in target_modules) and hasattr(m, "linear") and isinstance(m.linear, nn.Linear)
    ]
    for name, mod in candidates:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]: parent = getattr(parent, part)
        setattr(parent, parts[-1], mod.linear)
    if candidates:
        print(f"[Factory] Unwrapped {len(candidates)} Gemma4Clippable layers.")

def train_adapter(
    concept: str,
    model_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    max_steps: int = 200,
    rank: int = 16,
    lr: float = 2e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs
):
    """
    Core LoRA training orchestration.
    """
    mid = model_id or config.current_model_id
    m_cfg = config.get_model_config(mid)
    ds_id = dataset_id or config.science_triad_datasets.get(concept)
    
    max_seq_length = kwargs.get("max_seq_len", 128)
    output_dir = kwargs.get("output_dir", str(MRNAPaths.ROOT / "adapters" / f"{concept}_lora"))
    
    # 1. Load Model
    backend = get_backend("torch")
    model, tokenizer = backend.load_model(mid, max_seq_length=max_seq_length, load_in_4bit=True)
    _tok = getattr(tokenizer, "tokenizer", tokenizer)

    # 2. Prepare PEFT
    _TARGET_MODULES = ["q_proj", "v_proj"]
    _unwrap_clippable_gemma(model, _TARGET_MODULES)
    
    model = FastLanguageModel.get_peft_model(
        model, r=rank, target_modules=_TARGET_MODULES,
        lora_alpha=rank, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=3407
    )

    # 3. Process Dataset
    print(f"Loading and formatting {ds_id}...")
    formatter = get_dataset_formatter(ds_id, tokenizer, kwargs.get("text_column"))
    ds_stream = load_dataset(ds_id, split=kwargs.get("split", "train"), streaming=True)
    rows = list(tqdm(ds_stream.take(kwargs.get("max_examples", 5000)), desc="Streaming"))
    dataset = HFDataset.from_list(rows)

    def _truncate(batch):
        texts = formatter(batch)["text"]
        out = []
        for t in texts:
            ids = _tok(text=t, truncation=True, max_length=max_seq_length - 1)["input_ids"]
            out.append(_tok.decode(ids, skip_special_tokens=False))
        return {"text": out}

    dataset = dataset.map(_truncate, batched=True)

    # 4. Trainer Setup
    trainer = SFTTrainer(
        model=model, processing_class=_tok,
        train_dataset=dataset, dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=TrainingArguments(
            per_device_train_batch_size=kwargs.get("batch_size", 1),
            gradient_accumulation_steps=kwargs.get("grad_accum", 8),
            warmup_steps=10, max_steps=max_steps, learning_rate=lr,
            fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
            logging_steps=10, optim="adamw_8bit", weight_decay=0.01,
            output_dir=os.path.join(output_dir, "checkpoints"),
            report_to="none"
        ),
    )

    print(f"Starting training for {concept}...")
    trainer.train()

    # 5. Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter saved to {output_dir}")
    return output_dir

if __name__ == "__main__":
    # Test stub
    # train_adapter(concept="biology", max_steps=5)
    pass

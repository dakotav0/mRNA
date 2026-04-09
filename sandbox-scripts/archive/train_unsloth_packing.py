import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

# Step 1 of the Trajectory: Unsloth fine-tune to understand packing/masking.
# Unsloth packing concatenates short sequences into one context window (e.g. 2048)
# to ensure the GPU isn't underutilized with mostly padding tokens.

max_seq_length = 2048
print("Loading Base Model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

print("Injecting .mrna target LoRA Adapter...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Small rank for streaming
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Dropout=0 strictly required for optimal unsloth training
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }


print("Loading toy dataset (Alpaca Cleaned) for packing test...")
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)

print("\n--- Initializing SFTTrainer with packing=True ---")
# The crucial mechanism to study here is packing=True.
# Watch the console output closely. Unsloth's custom Triton kernels enforce strict
# causal masking at the CUDA level, meaning even though multiple distinct documents
# are squashed into a single 2048 block, attention won't bleed across them!
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch size 2 * 4 = 8
        warmup_steps=5,
        max_steps=60,  # We only need to run a few steps to observe the logs
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # Disable wandb for local tests
    ),
)

print("\nStarting Training. Watch for Unsloth's packing and cross-entropy logs!\n")
trainer_stats = trainer.train()

print("\nTest training completed. Observe how the documents were packed.")

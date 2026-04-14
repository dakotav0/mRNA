# Setup & LoRA Readiness Guide

This document provides a deep-dive into the environmental requirements for training and deploying `.mrna` adapters.

## 1. Environmental Readiness Checklist

To ensure your system can handle the high-throughput activation harvesting and training, verify the following:

- [ ] **NVIDIA Drivers**: 535+ (Required for CUDA 12.x and Flash Attention).
- [ ] **WSL2 Memory**: At least 75% of system RAM should be allocated to WSL if running on Windows.
- [ ] **HF Auth**: Run `huggingface-cli login` and ensure you have access to `unsloth/gemma-4-E2B-it`.
- [ ] **Llama.cpp**: `llama-server` must be in your `PATH` or at `~/llama.cpp/build/bin/llama-server`.

## 2. Hardware Tuning

### VRAM Budgeting (12GB Limit)
When training **Gemma 4 E2B** or **Qwen 3.5 2B**, the VRAM usage is tight. 
- **4-bit Loading**: Mandatory. Use `load_in_4bit=True`.
- **Rank=16**: High enough for steering, low enough to preserve KV cache headroom.
- **Seq Length**: Keep at 512 for harvesting, 128 for training if OOM occurs.

## 3. The "Unwrapping" Manual Fix
If you encounter `Target module Gemma4ClippableLinear(...) is not supported`, our factory now handles this automatically. If you are writing custom scripts, ensure you call:
```python
from mrna.factory.adapter import _unwrap_clippable_gemma
_unwrap_clippable_gemma(model, ["q_proj", "v_proj"])
```
This is required because PEFT does not natively support the Gemma 4 clippable linear layers.

## 4. Inference Setup
To run the dual-routing API (`api.py`), you must have a GGUF version of the base model.
1. Download or convert the base model to GGUF (e.g., `q8_0` format).
2. Point to the GGUF path in `model_config.yaml`.
3. Ensure all adapters you wish to hot-swap are also converted via `mrna-convert`.

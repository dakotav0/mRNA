# mRNA (Modular Representation Network Adapter)

[![Status: Validated](https://img.shields.io/badge/Status-Validated-success?style=flat-square)](LATEST_LOG.md)
[![Target: Multimodal](https://img.shields.io/badge/Target-Multimodal-blue?style=flat-square)](model_config.yaml)

**mRNA** is a high-performance framework for training and deploying **Modular Representation Adapters**. It utilizes **Concept Bottleneck Sparse Autoencoders (CBSAE)** to execute the **Split-Brain Architecture**, decoupling a model's structural logic from its tonal persona.

---

## 🏗️ Split-Brain Architecture

mRNA splits inference into two distinct conceptual passes to ensure identity stability independent of domain knowledge:

1.  **Abstract Pass (Logic Gate)**: Mid-layers (e.g., Layer 6 on Qwen, 14 on Gemma) are intercepted to detect domain activation. Structural SAE concepts (Biology, physics, etc.) trigger domain-specific LoRA adapters.
2.  **Voice Pass (Persona Gate)**: Late-layers (e.g., Layer 12 on Qwen, 25 on Gemma) are intercepted to detect higher-level cadence and slang. This triggers the PIDX (Personality Index) identity profile.

### NOTE:
- Layer 6 on Qwen 3.5 0.8B had higher routing accuracy than Layer 12.
- Layer 25 on Gemma 4 E2B had higher routing accuracy than Layer 14.
- Need to run more experiments to determine the optimal layer depth for each model.

---

## 🛠️ Prerequisites & LoRA Readiness

Before starting training, ensure your environment meets the `.mrna` substrate requirements:

### 1. Hardware Requirements
- **Compute**: NVIDIA GPU with 12GB+ VRAM (RTX 4070 SUPER or higher). 
- **Memory**: 32GB+ System RAM (needed for activation shuffling and dataset streaming).
- **Storage**: 50GB+ free space (HuggingFace cache, GGUF models, and activation buffers).

### 2. Software Substrate
- **OS**: Linux (WSL2 Ubuntu 22.04+ supported).
- **Python**: 3.12+ (managed via `uv` or `venv`).
- **Inference Backend**: `llama.cpp` installed and compiled with CUDA support.
- **Environment**: 
  - `HF_TOKEN`: Required for restricted model access (Llama 3, Gemma 2).
  - `CUDA_HOME`: Pointing to your CUDA toolkit (12.x recommended).

---

## ⚙️ Configuration (`model_config.yaml`)

The entire pipeline is *model-agnostic and relies on the configuration registry:

- **`current_target`**: The model ID to use across all scripts.
- **`harvest_layers`**: Crucial mapping of **Logic** (Index 0) and **Voice** (Index 1) layer depths.
- **`expansion_factor`**: SAE bottleneck width (usually 8x `d_model`).

---

## 🚀 The Pipeline (CLI Reference)

mRNA provides a suite of standardized commands for the end-to-end lifecycle.

### 1. `mrna-harvest`
Samples neural activations from the residual stream of the base model.
```bash
mrna-harvest --concept biology --max-examples 2000 --download
```
- **Prereq**: HF Base weights must be present.
- **Output**: `~/layer_<N>/<concept>_train.pt`, `~/layer_<N>/<concept>_test.pt`.

### 2. `mrna-train-sae`
Trains the logic gate (CBSAE) to recognize a concept in the neural stream.
```bash
mrna-train-sae --concepts biology,chemistry,physics --epochs 50 --layer 14
mrna-train-sae --concepts formal,concise,creative --epochs 50 --layer 25
```
- **Prereq**: Harvested `.pt` files for all target concepts.
- **Output**: `sae_weights_L<X>.pt` in the model data dir.

### 3. `mrna-train-adapter`
Fine-tunes the base model for a specific behavioral or domain-specific logic.
```bash
mrna-train-adapter --concept biology --dataset camel-ai/biology --steps 200 --max-examples 800
```
- **Prereq**: Domain dataset (Camel-AI or custom).
- **Compatibility**: Automatically "unwraps" `Gemma4ClippableLinear` layers for PEFT support.

### 4. `mrna-eval`
Produces a precision confusion matrix for the logic gateways.
```bash
mrna-eval --concepts biology,chemistry,physics --layer 14
mrna-eval --concepts formal,concise,creative --layer 25
```
- **Target**: Validates if Layer X can reliably distinguish between overlapping domains.

### 5. `mrna-convert`
Transforms PEFT adapters into GGUF format for llama.cpp hot-swapping.
```bash
mrna-convert --concept biology
```
- **Output**: GGUF adapter bin ready for the `/lora-adapters` endpoint.

---

## 🧪 Troubleshooting

### Harvest and Train Adapter
Steps 1 and 3 are compute heavy so be prepared to wait for data to warm up for a slow and steady process.

### Gemma 4 `ClippableLinear` Error
Gemma 4 models (e.g., E2B) use custom vision/audio layers that standard PEFT cannot wrap. Our factory automatically "unwraps" these into `nn.Linear` modules during training to avoid the `ValueError`.

### Qwen 3.5
Qwen 3.5 0.8B had issues with thinking and it's output latency may not be worth the time. Yet to test Qwen 3.5 2B.

### OOM (Out of Memory)
- **Training**: Decrease `max_seq_len` to 128 or enable `liger-kernel` for fused attention.
- **Harvesting**: Decrease `batch_size` in `model_config.yaml`.

---

## 📜 Research & Synthesis
- [LATEST_LOG.md](LATEST_LOG.md) — Real-time architecture changes and benchmark results.
- [docs/synthesis-log.md](docs/synthesis-log.md) — Historical context and design rationale.

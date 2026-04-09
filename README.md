# mRNA (Neural Routing & Adapter Infrastructure)

mRNA is a production-ready framework for training, evaluating, and deploying specialized LoRA adapters using **Concept Bottleneck Sparse Autoencoders (CBSAE)**. It provides the neural "routing" logic required for dynamic, multi-adapter inference pipelines like the MIIN NPC system.

## Features
- 🧬 **Neural Routing**: Transition from keyword-based triggers to latent-space conceptual classification.
- 🧪 **Factory Pipelines**: Automated activation harvesting, adapter training, and SAE weight optimization.
- 🔬 **Bleed Detection**: Diagnostic tools to measure conceptual overlap between datasets and prevent catastrophic interference.
- 🍎 **Multi-Backend**: Native support for CUDA (Torch) with a planned substrate for Apple Silicon (MLX).

## Installation

```bash
git clone https://github.com/dakotav0/mRNA.git
cd mRNA
pip install -e .
```

## Quick Start

### 1. Harvest Activations
Identify the residual stream representations for a specific concept.
```bash
mrna-harvest --dataset camel-ai/biology --max-examples 500
```

### 2. Train SAE Weights
Map the latent space and supervise the concept bottleneck.
```bash
mrna-train-sae --concepts biology,chemistry,physics --activation-files data/.../biology.pt ...
```

### 3. Evaluate Routing
Verify the F1-Score and confusion matrix for your neural router.
```bash
mrna-eval --concepts biology,chemistry,physics
```

## Architecture
mRNA is organized into four primary pillars:
- `mrna.core`: Configuration management and model-centric path resolution.
- `mrna.substrate`: Platform-agnostic tensor operations (Torch/MLX).
- `mrna.factory`: Orchestration for data sampling and training.
- `mrna.analysis`: Interpretability and diagnostic metrics.

## Roadmap
- [ ] Full MLX Backend implementation.
- [ ] Dynamic adapter hot-swapping via llama.cpp.
- [ ] Web-based visualization for SAE activations.

## License
[MIT](LICENSE)

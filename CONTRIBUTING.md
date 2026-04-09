# Contributing to mRNA

Thank you for your interest in the mRNA project! We are building the foundational infrastructure for neural routing and concept-aligned adapters.

## Research Philosophy
mRNA is a **research-first** repository. We prioritize:
1. **Interpretability**: Using SAEs to map the latent space.
2. **Modularity**: Adapters should be swappable and orthogonal.
3. **Cross-Platform**: Support for both CUDA (Torch) and Apple Silicon (MLX).

## How to Contribute

### 1. New Concept Adapters
If you want to contribute a new specialized adapter (e.g., for the "Science Triad"), follow this workflow:
- **Harvest**: Use `mrna-harvest` to collect activations from a diverse HF dataset.
- **Bleed Check**: Use `mrna-bleed-check` to ensure the new data doesn't collapse into an existing concept's subspace.
- **Train**: Use `mrna-train-adapter`.

### 2. Core Architecture
If modifying the library core:
- Ensure all logic is abstracted behind the `Backend` substrate.
- Add unit tests in `tests/` and verify with `pytest`.

## Code of Conduct
Please be respectful and focus on rigorous neural interpretability.

## Data Structure
Always adhere to the model-centric hierarchy:
`data/<model_id>/activations/layer_<N>/<concept>.pt`
`data/<model_id>/sae_weights.pt`

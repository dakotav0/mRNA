"""
Prefill-Only Routing Prototype

Architecture under test:

    ┌─────────────────────────────────────────────┐
    │  unsloth base model (frozen, 4-bit, CPU→GPU) │
    │  ActivationInterceptor on layer N            │
    │  Single forward pass — NO generation         │
    └──────────────┬──────────────────────────────┘
                   │ activations (batch, seq_len, d_model)
                   ▼
    ┌─────────────────────────────────────────────┐
    │  CBSAE encoder                              │
    │  pool → sparse → argmax bottleneck slice    │
    └──────────────┬──────────────────────────────┘
                   │ (concept_idx, adapter_path, confidence)
                   ▼
    ┌─────────────────────────────────────────────┐
    │  vLLM StreamingExecutionNode                │
    │  run_inference_with_adapter(prompt, path, id)│
    └─────────────────────────────────────────────┘

Why this first
--------------
Least invasive approach: unsloth and vLLM don't share a process or model state.
Unsloth reads the prompt and exits gracefully before vLLM starts. The only cost
is one extra prefill pass per request — acceptable for the prototype, measurable
for deciding whether a faster integration path is worth the complexity.

Tradeoff vs alternatives
-------------------------
  Sidecar model  — faster (tiny classification model, ms-latency routing),
                   but requires training a separate model and breaks the
                   "SAE on the same residual stream" property.
  vLLM hook      — zero extra prefill, but vLLM internals are moving fast
                   and hooking CUDA graphs is fragile.
  Prefill-only   — ~same latency as the generation prefill phase anyway
                   (prompt is processed once either way). Clean separation.
                   Start here.

Usage
-----
# With random-init SAE (tests structural wiring, not real routing quality)
python sandbox-scripts/prefill_router.py

# With trained SAE weights
python sandbox-scripts/prefill_router.py --sae-weights data/gemma-4-e2b/sae/w_25.pt

# Full argument list
python sandbox-scripts/prefill_router.py \
    --model-id unsloth/gemma-4-E2B-it \
    --layer 17 \
    --sae-weights data/sae_weights.pt \
    --d-model 2560
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mrna.router.interceptor import ActivationInterceptor
from mrna.router.sae import CBSAE
from mrna.execution.llama_cpp_node import LlamaCppExecutionNode


# ---------------------------------------------------------------------------
# Router — unsloth prefill-only pass
# ---------------------------------------------------------------------------

class PrefillRouter:
    """
    Wraps the unsloth model + ActivationInterceptor + CBSAE into a single
    routing call that produces an adapter decision without generating any tokens.
    """

    def __init__(
        self,
        model_id: str,
        layer: int,
        sae: CBSAE,
        concept_names: list[str],
        adapter_registry: dict[str, str],
        model_revision: str | None = None,
        max_seq_len: int = 512,
    ):
        from unsloth import FastLanguageModel

        rev_label = f" @ {model_revision[:8]}" if model_revision else ""
        print(f"[PrefillRouter] Loading {model_id!r}{rev_label} for prefill-only routing ...")
        load_kwargs = dict(model_name=model_id, max_seq_length=max_seq_len, dtype=None, load_in_4bit=True)
        if model_revision:
            load_kwargs["revision"] = model_revision
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
        FastLanguageModel.for_inference(self.model)
        self.model.eval()

        self.interceptor = ActivationInterceptor(target_layer=layer)
        self.interceptor.attach_to_model(self.model)

        self.sae = sae
        self.sae.eval()
        self.concept_names = concept_names
        self.adapter_registry = adapter_registry
        self.max_seq_len = max_seq_len

    def route(self, prompt: str) -> tuple:
        """
        Run a single prefill-only forward pass and return a routing decision.

        Returns
        -------
        (concept_idx, concept_name, adapter_path, confidence, latency_ms)
        """
        t0 = time.perf_counter()

        _tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        enc = _tok(
            text=[prompt],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len,
        ).to(next(self.model.parameters()).device)

        with torch.no_grad():
            # Prefill only — hook fires, no tokens are generated
            self.model(**enc)

        # Pull activations deposited by the hook
        acts = self.interceptor.intercepted_activations[-1]  # (1, seq_len, d_model)
        self.interceptor.intercepted_activations.clear()

        # Masked mean pool — matches harvest_hf.py's pooling exactly.
        # With batch_size=1 there's no padding, but using the mask is consistent
        # and guards against future batched use.
        mask     = enc["attention_mask"].cpu().unsqueeze(-1).float()  # (1, seq_len, 1)
        pooled   = ((acts * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).float()
        pre_relu = self.sae.encoder(pooled)                          # (1, d_sae)
        bottleneck = pre_relu[:, :len(self.concept_names)]   # (1, n_concepts)
        strengths = bottleneck[0]

        concept_idx = int(strengths.argmax().item())
        confidence = float(strengths[concept_idx].item())

        # Full score breakdown — useful for diagnosing boundary cases
        score_str = "  ".join(
            f"{name}={strengths[i].item():.3f}"
            for i, name in enumerate(self.concept_names)
        )
        print(f"  [SAE scores] {score_str}")

        latency_ms = (time.perf_counter() - t0) * 1000
        concept_name = self.concept_names[concept_idx]
        adapter_path = self.adapter_registry[concept_name]

        return concept_idx, concept_name, adapter_path, confidence, latency_ms

    def detach(self):
        self.interceptor.detach()


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

DEMO_CASES = [
    {
        "concept": "biology",
        "prompt": "What is the role of mitochondria in eukaryotic cells?",
    },
    {
        "concept": "biology",
        "prompt": "Explain the process of DNA replication and the enzymes involved.",
    },
    {
        "concept": "chemistry",
        "prompt": "What is the difference between ionic and covalent bonding?",
    },
    {
        "concept": "chemistry",
        "prompt": "Explain Le Chatelier's principle with an example.",
    },
    {
        "concept": "physics",
        "prompt": "What is the relationship between electric field and electric potential?",
    },
    {
        "concept": "physics",
        "prompt": "Explain how a Carnot engine achieves maximum thermodynamic efficiency.",
    },
]

# PEFT adapter dirs (used by PrefillRouter.adapter_registry for display only)
ADAPTER_REGISTRY = {
    "biology":   "data/gemma-4-e2b/adapters/biology_lora",
    "chemistry": "data/gemma-4-e2b/adapters/chemistry_lora",
    "physics":   "data/gemma-4-e2b/adapters/physics_lora",
}

# GGUF LoRA paths for llama-server (convert with convert_adapters_to_gguf.py first)
GGUF_ADAPTER_REGISTRY = {
    "biology":   "data/gemma-4-e2b/adapters/biology_lora/biology.gguf",
    "chemistry": "data/gemma-4-e2b/adapters/chemistry_lora/chemistry.gguf",
    "physics":   "data/gemma-4-e2b/adapters/physics_lora/physics.gguf",
}


def run_demo(model_id: str, layer: int, sae_weights: str | None, d_model: int,
             model_path: str = "models/gemma-4-e2b-q8.gguf",
             model_revision: str | None = None):
    concept_names = list(ADAPTER_REGISTRY.keys())
    n_concepts = len(concept_names)

    # SAE — load trained weights if available, else random init
    sae = CBSAE(d_model=d_model, expansion_factor=8, bottleneck_features=n_concepts)
    if sae_weights and os.path.exists(sae_weights):
        sae.load_state_dict(torch.load(sae_weights, map_location="cpu", weights_only=True))
        print(f"[SAE] Loaded weights from {sae_weights}")
    else:
        if sae_weights:
            print(f"[SAE] Weights not found at {sae_weights!r} — using random init.")
            print("      Routing decisions will be random until the SAE is trained.")
            print("      Run harvest_hf.py + train_sae.py to fix this.\n")
        else:
            print("[SAE] No weights provided — using random init (structural test only).\n")

    # Build prefill router
    router = PrefillRouter(
        model_id=model_id,
        layer=layer,
        sae=sae,
        concept_names=concept_names,
        adapter_registry=ADAPTER_REGISTRY,
        model_revision=model_revision,
    )

    print("\n[Execution] Initializing llama-server node ...")
    execution = LlamaCppExecutionNode(
        model_path=model_path,
        adapter_registry=GGUF_ADAPTER_REGISTRY,
        tokenizer=router.tokenizer,   # for chat template formatting
    )

    print("\n" + "=" * 60)
    print("Prefill-Only Routing Prototype — Demo")
    print("=" * 60)

    total_route_ms = 0.0
    correct = 0

    for case in DEMO_CASES:
        prompt = case["prompt"]
        expected = case["concept"]

        print(f"\nPrompt: {prompt[:80]}...")
        print(f"Expected concept: {expected}")

        idx, name, path, conf, latency = router.route(prompt)
        total_route_ms += latency

        match = "✓" if name == expected else "✗"
        print(f"  {match} Routed → '{name}'  (confidence={conf:.4f}, latency={latency:.1f}ms)")
        gguf_path = GGUF_ADAPTER_REGISTRY.get(name, path)
        print(f"  Adapter GGUF: {gguf_path}")
        print(f"  Handing off to llama-server ...")

        # llama-server takes over with the selected adapter
        execution.run_inference_with_adapter(prompt, gguf_path, idx + 1)

        if name == expected:
            correct += 1

    router.detach()
    execution.cleanup()

    print("\n" + "=" * 60)
    print(f"Routing accuracy : {correct}/{len(DEMO_CASES)}  "
          f"({'random-init SAE — meaningless' if not sae_weights else 'trained SAE'})")
    print(f"Avg route latency: {total_route_ms / len(DEMO_CASES):.1f} ms  "
          f"(prefill-only pass, no generation)")
    print("=" * 60)

    if not sae_weights or not os.path.exists(sae_weights or ""):
        print("\nNext steps:")
        print("  1. python sandbox-scripts/harvest_hf.py --dataset iamtarun/python_code_instructions_18k_alpaca \\")
        print("         --text-column instruction --concept python --output data/python_layer6.pt")
        print("  2. (repeat for each concept)")
        print("  3. python sandbox-scripts/train_sae.py \\")
        print("         --activations python:data/python_layer6.pt legal:data/legal_layer6.pt ... \\")
        print("         --output data/sae_weights.pt")
        print("  4. python sandbox-scripts/convert_adapters_to_gguf.py --base-model models/gemma-4-e2b-q8.gguf --adapters biology chemistry physics")
        print("  5. python sandbox-scripts/prefill_router.py --sae-weights data/sae_weights.pt --model-path models/gemma-4-e2b-q8.gguf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prefill-only unsloth routing prototype → vLLM generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-id",       default="unsloth/gemma-4-E2B-it")
    parser.add_argument("--model-revision", default="37ea165b3fba25b7d851f8ce4ccff9a4f0751cee",
                        help="HF commit hash. Pinned to match GGUF + adapter training snapshot.")
    parser.add_argument("--model-path",  default="models/gemma-4-e2b-q8.gguf",
                        help="Path to the GGUF base model for llama-server.")
    parser.add_argument("--layer",       type=int, default=25)
    parser.add_argument("--sae-weights", default="data/gemma-4-e2b/sae/w_25.pt",
                        help="Path to trained CBSAE state_dict (.pt). "
                             "Omit or set to None to run structural test with random-init SAE.")
    parser.add_argument("--d-model",     type=int, default=1536)
    args = parser.parse_args()

    run_demo(
        model_id=args.model_id,
        layer=args.layer,
        sae_weights=args.sae_weights,
        d_model=args.d_model,
        model_path=args.model_path,
        model_revision=args.model_revision,
    )


if __name__ == "__main__":
    main()

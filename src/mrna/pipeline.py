"""
mRNAPipeline — end-to-end orchestrator

Data flow per request:

  ActivationInterceptor (forward hook on base model layer N)
      ↓  activations: (batch, seq_len, d_model)
  CBSAE encoder → sparse features → argmax over bottleneck slice
      ↓  concept_idx, adapter_path
  StreamingExecutionNode.run_inference_with_adapter(prompt, path, id)
      ↓  [if hot-swap detected AND lora_weights provided]
  lora_merge Triton kernel fires on background thread
      → pre-merged weight tensor cached in _merged_weights[concept_idx]
         (available for zero-copy reuse when this adapter is requested again)
"""

import threading
from typing import Optional

import torch
import torch.nn.functional as F

from mrna.execution.streaming_lora import StreamingExecutionNode
from mrna.router.sae import CBSAE
from mrna.substrate.lora_merge import lora_merge


class mRNAPipeline:
    """
    Wires the four mRNA subsystems into a single call interface.

    Parameters
    ----------
    adapter_registry : dict[str, str]
        Maps concept name → adapter path on disk.
        Order determines concept index (argmax maps to list position).
        Example: {"python": "/adapters/python_lora", "legal": "/adapters/legal_lora"}
    model_id : str
        HuggingFace model ID passed to vLLM.
    d_model : int
        Hidden dimension of the base model (must match interceptor layer output).
    expansion_factor : int
        SAE width multiplier — d_sae = d_model * expansion_factor.
    max_loras : int
        Max number of adapters resident in VRAM simultaneously (vLLM constraint).
    max_vram : float
        Fraction of VRAM reserved for the vLLM engine.
    sae_weights_path : str | None
        Path to a saved CBSAE state_dict. If None, uses random initialization
        (useful for structural testing before a trained SAE is available).
    """

    def __init__(
        self,
        adapter_registry: dict,
        model_id: str = "unsloth/Llama-3.2-1B-Instruct",
        d_model: int = 2048,
        expansion_factor: int = 8,
        max_loras: int = 2,
        max_vram: float = 0.6,
        sae_weights_path: Optional[str] = None,
    ):
        self.adapter_registry = adapter_registry
        self.concept_names = list(adapter_registry.keys())
        self.n_concepts = len(self.concept_names)

        # SAE router — inference only
        self.sae = CBSAE(
            d_model=d_model,
            expansion_factor=expansion_factor,
            bottleneck_features=self.n_concepts,
        )
        self.sae.eval()
        if sae_weights_path:
            self.sae.load_state_dict(torch.load(sae_weights_path, map_location="cpu"))

        # vLLM execution node
        self.execution = StreamingExecutionNode(
            model_id=model_id,
            max_vram=max_vram,
            max_loras=max_loras,
        )

        self._current_adapter_id: Optional[int] = None
        self._merge_thread: Optional[threading.Thread] = None

        # Pre-merged weight buffer: concept_idx → merged W tensor (GPU)
        # Populated lazily by _pre_warm when adapter AB weights are provided.
        self._merged_weights: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def route(self, activations: torch.Tensor) -> tuple:
        """
        Map residual-stream activations to an adapter routing decision.

        Parameters
        ----------
        activations : (batch, seq_len, d_model)
            Output of ActivationInterceptor._hook_fn, moved to CPU.

        Returns
        -------
        (concept_idx, concept_name, adapter_path, confidence)
        """
        with torch.no_grad():
            # Pool over sequence → (batch, d_model), then encode
            pooled = activations.mean(dim=1).float()
            pre_relu = self.sae.encoder(pooled)  # (batch, d_sae) — full logit range

            # Argmax over pre-ReLU bottleneck logits (matches CrossEntropy training objective)
            bottleneck = pre_relu[:, : self.n_concepts]  # (batch, n_concepts)
            concept_strengths = bottleneck.mean(dim=0)  # (n_concepts,)

            concept_idx = int(concept_strengths.argmax().item())
            confidence = float(concept_strengths[concept_idx].item())

        concept_name = self.concept_names[concept_idx]
        adapter_path = self.adapter_registry[concept_name]
        return concept_idx, concept_name, adapter_path, confidence

    def run(
        self,
        prompt: str,
        activations: torch.Tensor,
        lora_weights: Optional[dict] = None,
    ) -> None:
        """
        Full pipeline call: route → execute → (async) merge pre-warm.

        Parameters
        ----------
        prompt : str
            Raw text prompt sent to vLLM.
        activations : (batch, seq_len, d_model)
            Residual stream tensor from ActivationInterceptor.
        lora_weights : dict | None
            Optional pre-warm weights, keyed by concept_idx:
                { concept_idx: {"W": tensor, "A": tensor, "B": tensor, "scale": float} }
            When provided and a hot-swap is detected, lora_merge fires in the
            background so the merged tensor is ready for the next request.
        """
        concept_idx, concept_name, adapter_path, confidence = self.route(activations)

        hot_swap = (
            self._current_adapter_id is not None
            and concept_idx != self._current_adapter_id
        )

        if hot_swap:
            print(
                f"[Pipeline] Hot-swap: adapter {self._current_adapter_id} "
                f"→ {concept_idx} ('{concept_name}', confidence={confidence:.3f})"
            )
            if lora_weights and concept_idx in lora_weights:
                self._pre_warm(concept_idx, lora_weights[concept_idx])
        else:
            print(
                f"[Pipeline] SAE route → '{concept_name}' "
                f"(id={concept_idx}, confidence={confidence:.3f})"
            )

        self._current_adapter_id = concept_idx
        # vLLM adapter_id is 1-indexed (0 is reserved for base model)
        self.execution.run_inference_with_adapter(prompt, adapter_path, concept_idx + 1)

    def cleanup(self):
        if self._merge_thread and self._merge_thread.is_alive():
            self._merge_thread.join(timeout=5.0)
        self.execution.cleanup()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _pre_warm(self, concept_idx: int, weights: dict) -> None:
        """
        Fire lora_merge in a background thread, caching the result so the
        next hot-swap to this adapter pays zero merge cost.

        weights must contain keys: "W", "A", "B", "scale"
        """
        # Capture tensors now; .contiguous().half().cuda() is cheap and avoids
        # a race if the caller mutates the originals after returning.
        W = weights["W"].contiguous().half().cuda()
        A = weights["A"].contiguous().half().cuda()
        B = weights["B"].contiguous().half().cuda()
        scale = float(weights["scale"])

        def _merge():
            merged = lora_merge(W, A, B, scale)
            self._merged_weights[concept_idx] = merged
            print(
                f"[Pipeline] Pre-warm complete: adapter {concept_idx} "
                f"merged weight cached ({merged.shape}, "
                f"{merged.element_size() * merged.numel() / 1e6:.1f} MB)"
            )

        self._merge_thread = threading.Thread(target=_merge, daemon=True)
        self._merge_thread.start()

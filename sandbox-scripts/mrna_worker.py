"""
mRNA Custom vLLM Worker — Merged-Weight Serving Path
vLLM 0.18.1 / v1 engine

Execution Node — closes the `_merged_weights → vLLM` gap.

Architecture
------------
Based on Gemini's pointer-swap concept, adapted to vLLM 0.18.1's actual APIs:

  class mRNAWorker(Worker):
      def execute_model(self, scheduler_output):
          concept_idx = _get_concept_idx(scheduler_output)
          if concept_idx in merged_cache:
              _apply_merged(concept_idx)   # copy_() into base weight tensors
              out = super().execute_model(scheduler_output)
              _restore_base()
              return out
          return super().execute_model(scheduler_output)

Verified API paths (vLLM 0.18.1)
---------------------------------
  Worker base class:       vllm.v1.worker.gpu_worker.Worker
  execute_model sig:       (self, scheduler_output: SchedulerOutput)
  LoRA request in output:  scheduler_output.scheduled_new_reqs[i].lora_request
  LoRA weight manager:     self.model_runner.lora_manager._adapter_manager   (LRUCacheLoRAModelManager)
  Get loaded model:        _adapter_manager.get_adapter(lora_int_id) → LoRAModel
  Get layer weights:       lora_model.get_lora(module_name) → LoRALayerWeights
  Weight fields:           .lora_a (Tensor), .lora_b (Tensor), .scaling (float)

In-place copy vs pointer swap
------------------------------
weight.data.copy_() writes into the same tensor allocation — same CUDA address.
CUDA-graph-safe: graphs record addresses, not values. No graph recapture needed
even if enforce_eager is ever dropped.

Lazy merge / IPC
----------------
merged_cache lives in the worker subprocess. Populated on first request for each
concept_idx by calling lora_merge() against weights vLLM has already loaded.
No inter-process communication needed.

Base restore cost
-----------------
One clone per LoRA-targeted Linear at init_device time (not on hot path).
Restore is copy_() — ~0.06ms per layer at 504 GB/s for a 4096-dim FP16 weight.
"""

import os
import sys

# Both sandbox-scripts/ (so vLLM can `import mrna_worker` in the subprocess)
# and src/ (so mrna_worker can `from mrna.substrate.lora_merge import lora_merge`)
# must be on PYTHONPATH before vLLM spawns workers — os.environ propagates to subprocesses,
# sys.path.insert does not.
_here = os.path.dirname(os.path.abspath(__file__))
_src  = os.path.join(_here, "..", "src")
for _p in [_here, _src]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
    _env_path = os.environ.get("PYTHONPATH", "")
    if _p not in _env_path:
        os.environ["PYTHONPATH"] = _p + (":" + _env_path if _env_path else "")

import gc
from typing import Optional

import torch
from vllm.v1.worker.gpu_worker import Worker
from vllm.lora.request import LoRARequest

from mrna.substrate.lora_merge import lora_merge


class mRNAWorker(Worker):
    """
    vLLM v1 Worker subclass that applies pre-merged LoRA weights in-place
    before each forward pass, then restores base weights after.
    """

    def init_device(self) -> None:
        super().init_device()

        # concept_idx → {layer_name: merged_W tensor}
        self._merged_cache: dict[int, dict[str, torch.Tensor]] = {}

        # layer_name → base weight clone  (allocated once, reused every restore)
        self._base_weights: dict[str, torch.Tensor] = {}

        # layer_name → base_layer module (for weight.data access)
        self._linear_layers: dict[str, torch.nn.Module] = {}

        # layer_name → BaseLayerWithLoRA wrapper (for output_slices on packed layers)
        self._lora_wrappers: dict[str, torch.nn.Module] = {}

        # Deferred: model weights are not loaded until after init_device.
        # _build_layer_map() is called lazily on first execute_model() call.
        self._layer_map_built = False

        # Track the currently active concept across prefill + decode steps.
        # Set when a new LoRA request is seen; cleared when no running reqs.
        self._active_concept_idx: Optional[int] = None

    # ------------------------------------------------------------------
    # Override
    # ------------------------------------------------------------------

    def execute_model(self, scheduler_output=None, **kwargs):
        if not self._layer_map_built:
            self._build_layer_map()

        # Update active concept tracking.
        # _get_concept_idx reads scheduled_new_reqs (prefill only).
        # Decode steps carry no new_reqs, so we reuse _active_concept_idx.
        new_concept_idx = self._get_concept_idx(scheduler_output)
        if new_concept_idx is not None:
            self._active_concept_idx = new_concept_idx
        # (decode steps leave _active_concept_idx unchanged)

        concept_idx = self._active_concept_idx

        if concept_idx is not None and self._linear_layers:
            self._ensure_merged(concept_idx)
            if concept_idx in self._merged_cache:
                self._apply_merged(concept_idx)
                output = super().execute_model(scheduler_output, **kwargs)
                self._restore_base()
                return output
            else:
                # Adapter not yet loaded — super() will load it via
                # set_active_adapters. Populate cache immediately after so
                # all decode steps for this request use merged weights.
                output = super().execute_model(scheduler_output, **kwargs)
                self._ensure_merged(concept_idx)  # adapter is now in manager
                return output

        return super().execute_model(scheduler_output, **kwargs)

    # ------------------------------------------------------------------
    # Layer map
    # ------------------------------------------------------------------

    def _build_layer_map(self) -> None:
        """
        Walk the loaded model, record every LoRA-capable layer, clone base weights.
        Called lazily on first execute_model() — model is guaranteed loaded by then.
        Uses get_model() which safely unwraps any CUDAGraphWrapper.

        vLLM wraps base linear layers in BaseLayerWithLoRA subclasses.
        The actual weight parameter lives on module.base_layer.weight (registered
        via layer.register_parameter("weight", ...) in the quant method).
        """
        from vllm.lora.layers.base import BaseLayerWithLoRA

        self._layer_map_built = True  # mark before any early return to avoid retry loops

        if not hasattr(self, "model_runner") or self.model_runner is None:
            return

        try:
            model = self.model_runner.get_model()
        except (AttributeError, ValueError):
            return

        n = 0
        for name, module in model.named_modules():
            if not isinstance(module, BaseLayerWithLoRA):
                continue
            base = getattr(module, "base_layer", None)
            if base is None:
                continue
            weight = getattr(base, "weight", None)
            if weight is None or not isinstance(weight, torch.Tensor):
                continue
            self._linear_layers[name] = base   # store base_layer for weight.data
            self._lora_wrappers[name] = module  # store wrapper for output_slices
            self._base_weights[name] = weight.data.clone()
            n += 1

        print(f"[mRNAWorker] Layer map: {n} LoRA-capable layers tracked, base weights cloned.")

    # ------------------------------------------------------------------
    # Extract concept_idx from SchedulerOutput
    # ------------------------------------------------------------------

    def _get_concept_idx(self, scheduler_output) -> Optional[int]:
        """
        Read lora_request from the first new request in SchedulerOutput.
        StreamingExecutionNode sets lora_int_id = concept_idx + 1.
        Returns None when no LoRA is active.

        vLLM 0.18.1 path:
            scheduler_output.scheduled_new_reqs  →  list[NewRequestData]
            new_req.lora_request                 →  LoRARequest | None
            lora_request.lora_int_id             →  int
        """
        if scheduler_output is None:
            return None
        try:
            for req in scheduler_output.scheduled_new_reqs:
                lr = req.lora_request
                if lr is not None:
                    return lr.lora_int_id - 1   # undo +1 offset from StreamingExecutionNode
        except (AttributeError, TypeError):
            pass
        return None

    # ------------------------------------------------------------------
    # Lazy merge
    # ------------------------------------------------------------------

    def _ensure_merged(self, concept_idx: int) -> None:
        """
        On first request for concept_idx: extract lora_a/lora_b from the
        vLLM LoRAModel, run lora_merge, cache per layer.
        Subsequent calls for the same concept_idx are no-ops.

        vLLM 0.18.1 path:
            self.model_runner.lora_manager._adapter_manager  →  LRUCacheLoRAModelManager
            .get_adapter(lora_int_id)           →  LoRAModel
            .get_lora(module_name)              →  LoRALayerWeights | None
            LoRALayerWeights: .lora_a, .lora_b, .scaling (float)
            PackedLoRALayerWeights: .scaling is list[float] — skipped (packed qkv_proj etc.)
        """
        if concept_idx in self._merged_cache:
            return

        lora_int_id = concept_idx + 1

        try:
            adapter_manager = self.model_runner.lora_manager._adapter_manager
        except AttributeError:
            print(f"[mRNAWorker] model_runner.lora_manager._adapter_manager not found — skipping merge.")
            return

        lora_model = adapter_manager.get_adapter(lora_int_id)
        if lora_model is None:
            # Adapter not yet loaded by vLLM — will be available on the first
            # request after add_lora() runs. Try again next call.
            return

        from vllm.lora.lora_weights import PackedLoRALayerWeights

        merged = {}
        for layer_name, base_module in self._linear_layers.items():
            layer_weights = lora_model.get_lora(layer_name)
            if layer_weights is None:
                continue  # this layer is not a LoRA target for this adapter

            W_full = base_module.weight.data  # full weight tensor (out_total, in)

            if isinstance(layer_weights, PackedLoRALayerWeights):
                # Packed layer (qkv_proj, gate_up_proj, etc.).
                # output_slices on the wrapper gives the output-dim size of each sub-slot.
                # lora_a/lora_b are lists; scaling is already baked into lora_b (optimize()).
                wrapper = self._lora_wrappers[layer_name]
                output_slices = getattr(wrapper, "output_slices", None)
                if output_slices is None:
                    continue  # can't determine offsets — skip

                merged_W = W_full.clone()
                offset = 0
                for i, (A_i, B_i) in enumerate(
                    zip(layer_weights.lora_a, layer_weights.lora_b)
                ):
                    slice_size = output_slices[i] if i < len(output_slices) else 0
                    if A_i is not None and B_i is not None and slice_size > 0:
                        # scaling is 1.0 after optimize() — delta = B @ A
                        scale_i = 1.0
                        s = layer_weights.scaling
                        if s is not None and i < len(s) and s[i] is not None:
                            scale_i = float(s[i])
                        try:
                            merged_W[offset:offset + slice_size, :] = lora_merge(
                                W_full[offset:offset + slice_size, :].contiguous().half().cuda(),
                                A_i.contiguous().half().cuda(),
                                B_i.contiguous().half().cuda(),
                                scale_i,
                            )
                        except Exception as e:
                            print(f"[mRNAWorker] lora_merge failed on {layer_name}[{i}]: {e}")
                    offset += slice_size

                merged[layer_name] = merged_W
            else:
                # Unpacked layer — simple W + scale * B @ A
                A = layer_weights.lora_a                    # (rank, in)
                B = layer_weights.lora_b                    # (out, rank)
                scale = float(layer_weights.scaling)

                try:
                    merged[layer_name] = lora_merge(
                        W_full.contiguous().half().cuda(),
                        A.contiguous().half().cuda(),
                        B.contiguous().half().cuda(),
                        scale,
                    )
                except Exception as e:
                    print(f"[mRNAWorker] lora_merge failed on {layer_name}: {e}")
                    continue

        if merged:
            self._merged_cache[concept_idx] = merged
            print(f"[mRNAWorker] Merged {len(merged)} layers for concept_idx={concept_idx} "
                  f"(lora_int_id={lora_int_id})")

    # ------------------------------------------------------------------
    # Apply / restore
    # ------------------------------------------------------------------

    def _apply_merged(self, concept_idx: int) -> None:
        """Write merged weights into base tensors in-place before forward pass."""
        for layer_name, merged_W in self._merged_cache[concept_idx].items():
            if layer_name in self._linear_layers:
                self._linear_layers[layer_name].weight.data.copy_(merged_W)

    def _restore_base(self) -> None:
        """Restore original base weights in-place after forward pass."""
        for layer_name, base_W in self._base_weights.items():
            if layer_name in self._linear_layers:
                self._linear_layers[layer_name].weight.data.copy_(base_W)


# ---------------------------------------------------------------------------
# Drop-in for StreamingExecutionNode
# ---------------------------------------------------------------------------

class MergedStreamingExecutionNode:
    """
    Drop-in replacement for StreamingExecutionNode using mRNAWorker.

    When concept_idx has a cached merged weight, the forward pass runs on the
    pre-merged W rather than the base W + LoRA delta — removing the delta
    decomposition overhead entirely.

    Falls back to standard LoRARequest delta path silently if:
      - lora_merge extraction fails (API mismatch)
      - adapter not yet loaded when _ensure_merged runs
    """

    def __init__(
        self,
        model_id: str = "unsloth/Llama-3.2-1B-Instruct",
        max_vram: float = 0.6,
        max_loras: int = 2,
    ):
        from vllm import LLM
        print(f"[MergedExecutionNode] Initializing with mRNAWorker on {model_id} ...")
        # vLLM 0.18.1 requires worker_cls as a dotted import string —
        # it is pickled and re-imported in the worker subprocess.
        # The module must be importable from the subprocess's sys.path.
        self.engine = LLM(
            model=model_id,
            enable_lora=True,
            max_lora_rank=16,
            max_loras=max_loras,
            gpu_memory_utilization=max_vram,
            enforce_eager=True,
            worker_cls="mrna_worker.mRNAWorker",
        )
        print("[MergedExecutionNode] Engine ready.")

    def run_inference_with_adapter(
        self, prompt: str, adapter_path: str, adapter_id: int
    ) -> None:
        from vllm import SamplingParams
        sampling_params = SamplingParams(temperature=0.7, min_tokens=8, max_tokens=128)

        print(f"\n[MergedExecution] concept_idx={adapter_id - 1}  adapter={adapter_path}")

        if not os.path.exists(adapter_path):
            print("[MergedExecution] Adapter not on disk — base model fallback.")
            lora_request = None
        else:
            lora_request = LoRARequest(str(adapter_id), adapter_id, adapter_path)

        outputs = self.engine.generate(
            prompt, sampling_params, lora_request=lora_request, use_tqdm=False
        )
        for output in outputs:
            print(f"Prompt:    {output.prompt!r}")
            print(f"Generated: {output.outputs[0].text!r}\n")

    def cleanup(self) -> None:
        print("\n[MergedExecution] Shutting down...")
        for _fn in [
            lambda: self.engine.llm_engine.shutdown(),
            lambda: self.engine.llm_engine.worker_executor.shutdown(await_termination=True),
        ]:
            try:
                _fn(); break
            except AttributeError:
                continue
        del self.engine
        gc.collect()
        torch.cuda.empty_cache()
        print("Shutdown complete.")


# ---------------------------------------------------------------------------
# Smoke test — biology adapter must exist on disk
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Science triad — adapter_id = concept_idx + 1 (1-indexed for vLLM LoRARequest)
    SCIENCE_CASES = [
        {
            "prompt":       "What is the role of mitochondria in eukaryotic cells?",
            "adapter_path": "/home/kotaraine/Coder/mRNA/adapters/biology_lora",
            "adapter_id":   1,  # biology
        },
        {
            "prompt":       "Explain the process of DNA replication and the enzymes involved.",
            "adapter_path": "/home/kotaraine/Coder/mRNA/adapters/biology_lora",
            "adapter_id":   1,
        },
        {
            "prompt":       "What is the difference between ionic and covalent bonding?",
            "adapter_path": "/home/kotaraine/Coder/mRNA/adapters/chemistry_lora",
            "adapter_id":   2,  # chemistry
        },
        {
            "prompt":       "Explain Le Chatelier's principle with an example.",
            "adapter_path": "/home/kotaraine/Coder/mRNA/adapters/chemistry_lora",
            "adapter_id":   2,
        },
        {
            "prompt":       "What is the relationship between electric field and electric potential?",
            "adapter_path": "/home/kotaraine/Coder/mRNA/adapters/physics_lora",
            "adapter_id":   3,  # physics
        },
        {
            "prompt":       "Explain how a Carnot engine achieves maximum thermodynamic efficiency.",
            "adapter_path": "/home/kotaraine/Coder/mRNA/adapters/physics_lora",
            "adapter_id":   3,
        },
    ]

    print("=== mRNAWorker Science Triad (vLLM 0.18.1) ===\n")

    node = MergedStreamingExecutionNode(
        model_id="unsloth/Llama-3.2-1B-Instruct",
        max_vram=0.6,
    )

    for case in SCIENCE_CASES:
        node.run_inference_with_adapter(
            prompt=case["prompt"],
            adapter_path=case["adapter_path"],
            adapter_id=case["adapter_id"],
        )

    node.cleanup()

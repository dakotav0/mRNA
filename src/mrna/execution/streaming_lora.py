import os

import torch

# Suppress heavy warnings from vLLM and specific frameworks for clean testing logs
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


class StreamingExecutionNode:
    """
    Priority 4: The Execution Node — vLLM Dynamic Loading Testing Suite

    This fulfills the final conceptual piece of the orchestration!
    Previously, we built a PyTorch module that manually multiplied the base layer
    and TWO LoRA adapters at the same time. While clean math, keeping 50 different
    .mrna adapters in VRAM is impossible.

    Instead, vLLM (the industry standard serving framework) allows us to keep
    adapters on the NVMe / System RAM, and dynamically request them per prompt
    via the `LoRARequest` abstraction.
    """

    def __init__(
        self, model_id="unsloth/Llama-3.2-1B-Instruct", max_vram=0.6, max_loras=2
    ):
        self._tmp_model_dir = None
        print(f"Initializing vLLM Engine on {model_id}...")

        # We explicitly cap the VRAM usage to leave headroom for our Engram table
        # and background streaming tasks!
        # Gemma 4 is multimodal (Gemma4ForConditionalGeneration). vLLM routes to
        # TransformersMultiModalForCausalLM when hf_config != hf_text_config (i.e.
        # when the config is nested/multimodal) — regardless of the architectures field.
        # Fix: write a flat config.json using text_config values + model_type=gemma4_text
        # and architecture=Gemma4ForCausalLM, so vLLM sees a non-nested config and routes
        # to TransformersForCausalLM (text-only backend).
        import json
        import tempfile

        from huggingface_hub import snapshot_download
        from transformers import AutoConfig

        is_gemma4 = "gemma-4" in model_id.lower() or "gemma4" in model_id.lower()

        if is_gemma4:
            src = snapshot_download(model_id)
            # Use a stable cache dir (not tempfile) so we don't re-extract on every run.
            cache_dir = os.path.join(os.path.dirname(src), "_mrna_gemma4_text_vllm")
            if not os.path.isdir(cache_dir):
                self._build_gemma4_text_cache(src, cache_dir)
            self._tmp_model_dir = None  # stable cache — don't delete on cleanup
            load_model_id = cache_dir
        else:
            load_model_id = model_id

        self.engine = LLM(
            model=load_model_id,
            enable_lora=True,
            max_lora_rank=16,
            max_loras=max_loras,
            gpu_memory_utilization=max_vram,
            enforce_eager=True,
            max_model_len=512,
            limit_mm_per_prompt={"image": 0, "audio": 0},
        )
        print("Engine initialized successfully.")

    @staticmethod
    def _build_gemma4_text_cache(src: str, cache_dir: str) -> None:
        """
        Build a text-only vLLM-compatible model directory from the full Gemma 4 snapshot.

        vLLM reads every tensor in the safetensors shards; there is no index-level
        filtering.  We must produce actual safetensors files that contain only the
        text backbone weights (no audio/vision tower).  This runs once and is reused.
        """
        import json
        import shutil

        from safetensors import safe_open
        from safetensors.torch import save_file
        from transformers import AutoConfig

        # Keep only text backbone weights — everything else is multimodal.
        # In Gemma 4 the full weight map has 5 top-level prefixes:
        #   model.language_model.*  (600) — text backbone, keep these
        #   model.audio_tower.*     (751) — drop
        #   model.vision_tower.*    (658) — drop
        #   model.embed_audio       (1)   — drop
        #   model.embed_vision      (1)   — drop
        _TEXT_PREFIX = "model.language_model."

        print(
            "[StreamingExecutionNode] Building Gemma4 text-only vLLM cache "
            f"(runs once) → {cache_dir}"
        )
        os.makedirs(cache_dir, exist_ok=True)

        # 1. Symlink non-weight files (tokenizer, processor, generation config…)
        skip = {"model.safetensors.index.json", "config.json"}
        for fname in os.listdir(src):
            if fname.endswith(".safetensors") or fname in skip:
                continue
            dst = os.path.join(cache_dir, fname)
            if not os.path.exists(dst):
                os.symlink(os.path.join(src, fname), dst)

        # 2. Flat text-only config.json
        full_cfg = AutoConfig.from_pretrained(src)
        text_cfg = full_cfg.text_config.to_dict()
        text_cfg["model_type"] = "gemma4_text"
        text_cfg["architectures"] = ["Gemma4ForCausalLM"]
        with open(os.path.join(cache_dir, "config.json"), "w") as f:
            json.dump(text_cfg, f, indent=2)

        # 3. Collect text-only tensors from all shards and write a single shard.
        idx_path = os.path.join(src, "model.safetensors.index.json")
        with open(idx_path) as f:
            idx = json.load(f)
        weight_map = idx["weight_map"]
        shards = sorted(
            {v for k, v in weight_map.items() if k.startswith(_TEXT_PREFIX)}
        )

        # Strip "language_model." from keys: shard has "model.language_model.layers.0..."
        # but Gemma4ForCausalLM (wrapped by TransformersForCausalLM) expects "model.layers.0..."
        _STRIP = "model.language_model."
        _KEEP = "model."
        # layer_scalar is in the safetensors but is not a registered parameter or buffer
        # in Gemma4TextDecoderLayer — vLLM raises ValueError if it sees it.
        _SKIP_SUFFIXES = (".layer_scalar",)

        text_tensors: dict[str, "torch.Tensor"] = {}
        for shard_file in shards:
            shard_path = os.path.join(src, shard_file)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith(_TEXT_PREFIX):
                        if any(key.endswith(s) for s in _SKIP_SUFFIXES):
                            continue
                        new_key = _KEEP + key[len(_STRIP) :]
                        text_tensors[new_key] = f.get_tensor(key)
            print(
                f"  Loaded {len(text_tensors)} text tensors so far (shard: {shard_file})"
            )

        out_shard = os.path.join(cache_dir, "model.safetensors")
        save_file(text_tensors, out_shard)
        print(f"  Saved {len(text_tensors)} tensors → {out_shard}")

        # 4. Write a trivial index pointing all text keys to the single shard.
        new_index = {
            "metadata": {"total_size": str(os.path.getsize(out_shard))},
            "weight_map": {k: "model.safetensors" for k in text_tensors},
        }
        with open(os.path.join(cache_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(new_index, f)

        print("[StreamingExecutionNode] Gemma4 text-only cache ready.")

    def run_inference_with_adapter(
        self, prompt: str, adapter_path: str, adapter_id: int
    ):
        """
        Takes a prompt and a specific adapter path (Simulating our SAE decision!),
        and dynamically streams the weights from disk directly into the vLLM batch.
        """
        sampling_params = SamplingParams(
            temperature=0.7,
            min_tokens=8,  # prevent immediate EOS when wrong adapter is merged
            max_tokens=128,
        )

        # Simulate SAE routing decision: We chose an adapter!
        print(
            f"\n[Execution Node] SAE Router triggered adapter ID {adapter_id}: {adapter_path}"
        )
        if not os.path.exists(adapter_path):
            print(
                f"[Execution Node] NOTE: '{adapter_path}' does not exist on disk yet."
            )
            print("Simulating base model inference as fallback...")
            lora_request = None
        else:
            print("Streaming .mrna dynamically via PCIe...")
            # This is the magic! vLLM handles the async weight swapping
            # and CUDA kernel injection seamlessly.
            lora_request = LoRARequest(str(adapter_id), adapter_id, adapter_path)

        print("Executing batch...")
        outputs = self.engine.generate(
            prompt, sampling_params, lora_request=lora_request, use_tqdm=False
        )

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}")
            print(f"Generated: {generated_text!r}\n")

    def cleanup(self):
        """
        Gracefully tears down the vLLM engine and frees GPU memory.

        vLLM spawns worker subprocesses that own their own NCCL process groups.
        Calling dist.destroy_process_group() from the parent process does not
        reach those groups and causes interference. Instead, trigger vLLM's own
        shutdown path, then release the reference so Python GC handles the rest.
        """
        import gc

        print("\nInitiating graceful shutdown of Execution Node...")

        # Try vLLM's internal shutdown hooks (API varies by version)
        for _shutdown in [
            lambda: self.engine.llm_engine.shutdown(),
            lambda: self.engine.llm_engine.worker_executor.shutdown(
                await_termination=True
            ),
        ]:
            try:
                _shutdown()
                break
            except AttributeError:
                continue

        del self.engine
        gc.collect()
        torch.cuda.empty_cache()

        if self._tmp_model_dir and os.path.isdir(self._tmp_model_dir):
            import shutil

            shutil.rmtree(self._tmp_model_dir, ignore_errors=True)
            self._tmp_model_dir = None

        print("Shutdown complete.")


if __name__ == "__main__":
    print("=== Testing Priority 4: vLLM Adapter Streaming Suite ===")

    # 1. Initialize the Execution node (capping VRAM at 60% so we have 40% free for
    # Engram tables and soft-collision sockets)
    execution_node = StreamingExecutionNode(
        model_id="unsloth/Llama-3.2-1B-Instruct", max_vram=0.6
    )

    # 2. Assume our Interceptor/SAE Router has identified these intents!
    test_cases = [
        {
            "intent": "Python",
            "prompt": "Write a python function to compute factorial.",
            "mrna_payload": "/home/kotaraine/Coder/mRNA/adapters/python_lora",
            "id": 1,
        },
        {
            "intent": "Legal",
            "prompt": "What is the legal definition of tort?",
            "mrna_payload": "/home/kotaraine/Coder/mRNA/adapters/legal_lora",
            "id": 2,
        },
    ]

    # 3. Stream and Generate! (It will fall back smoothly if the adapters don't exist yet)
    for case in test_cases:
        execution_node.run_inference_with_adapter(
            prompt=case["prompt"],
            adapter_path=case["mrna_payload"],
            adapter_id=case["id"],
        )

    # 4. Clean up resources
    execution_node.cleanup()

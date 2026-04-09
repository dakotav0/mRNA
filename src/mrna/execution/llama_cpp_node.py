"""
LlamaCppExecutionNode — llama.cpp execution backend for .mrna adapters.

Drop-in replacement for StreamingExecutionNode.  Spawns llama-server as a
subprocess, pre-loads all registered adapters at scale=0, and activates the
routed adapter per request via the /lora-adapters HTTP API.

Why llama.cpp instead of vLLM
------------------------------
Gemma 4 (Gemma4ForConditionalGeneration) uses a Gemma 3n per-layer embedding
reversal that broadcasts inputs_embeds against the full vocabulary during
TransformersForCausalLM's forward pass, allocating ~3072 GiB.  No config
workaround resolves this.  llama.cpp has native Gemma 4 GGUF support and
handles LoRA hot-swap natively.

Requirements
------------
- ~/llama.cpp built (CPU-only is fine for conversion + testing;
  CUDA build requires nvcc from system package manager)
- Base model converted: models/gemma-4-e2b-q8.gguf
- Per-adapter GGUF: adapters/{concept}_lora/{concept}.gguf
  (run sandbox-scripts/convert_adapters_to_gguf.py first)

Interface (preserved from StreamingExecutionNode)
-------------------------------------------------
    run_inference_with_adapter(prompt, adapter_path, adapter_id) -> None
    cleanup() -> None
"""

import os
import signal
import subprocess
import time
from typing import Optional

import requests

LLAMA_CPP_DIR = os.path.expanduser("~/llama.cpp")
LLAMA_SERVER_BIN = os.path.join(LLAMA_CPP_DIR, "build", "bin", "llama-server")


class LlamaCppExecutionNode:
    """
    Execution node backed by llama-server (llama.cpp HTTP API).

    All adapters are pre-loaded at scale=0 on startup.  Each call to
    run_inference_with_adapter activates one adapter (scale=1) and disables
    the rest before posting to /completion.
    """

    def __init__(
        self,
        model_path: str,
        adapter_registry: dict,           # {concept_name: path/to/adapter.gguf}
        tokenizer=None,                   # Transformers tokenizer for chat template
        host: str = "127.0.0.1",
        port: int = 8080,
        n_gpu_layers: int = 99,           # 99 = full CUDA offload; 0 for CPU only
        ctx_size: int = 2048,
        max_tokens: int = 128,
        startup_timeout: int = 120,
    ):
        self.model_path = model_path
        self.adapter_registry = adapter_registry  # {name: gguf_path}
        self.tokenizer = tokenizer
        self.host = host
        self.port = port
        self.max_tokens = max_tokens
        self.base_url = f"http://{host}:{port}"

        # Map concept name → llama-server adapter index (0-based)
        # Ordering matches the --lora-scaled flags passed to the server.
        self._adapter_index: dict[str, int] = {}
        self._proc: Optional[subprocess.Popen] = None

        self._start_server(
            n_gpu_layers=n_gpu_layers,
            ctx_size=ctx_size,
            startup_timeout=startup_timeout,
        )

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _start_server(self, n_gpu_layers: int, ctx_size: int, startup_timeout: int) -> None:
        cmd = [
            LLAMA_SERVER_BIN,
            "--model", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--n-gpu-layers", str(n_gpu_layers),
            "--ctx-size", str(ctx_size),
            "--log-disable",          # suppress llama.cpp verbose to keep output clean
        ]

        # Pre-load every adapter at scale=0.  llama-server assigns IDs in
        # the order adapters appear on the command line (0-indexed).
        # Format: --lora-scaled FNAME:SCALE  (colon-separated, not space)
        valid_adapters = {}
        for i, (name, gguf_path) in enumerate(self.adapter_registry.items()):
            if os.path.isfile(gguf_path):
                cmd += ["--lora-scaled", f"{gguf_path}:0"]
                self._adapter_index[name] = i
                valid_adapters[name] = gguf_path
            else:
                print(f"[LlamaCppNode] Adapter not found, skipping: {gguf_path}")

        if not valid_adapters:
            print("[LlamaCppNode] No adapter GGUFs found — base model only.")
        else:
            print(f"[LlamaCppNode] Pre-loading {len(valid_adapters)} adapter(s): "
                  f"{list(valid_adapters.keys())}")

        print(f"[LlamaCppNode] Starting llama-server: {self.model_path}")
        print(f"[LlamaCppNode] n_gpu_layers={n_gpu_layers}  ctx={ctx_size}")

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,   # capture for diagnostics on early exit
        )

        # Wait until /health returns 200
        deadline = time.time() + startup_timeout
        while time.time() < deadline:
            try:
                r = requests.get(f"{self.base_url}/health", timeout=2)
                if r.status_code == 200:
                    print(f"[LlamaCppNode] Server ready at {self.base_url}")
                    return
            except requests.exceptions.ConnectionError:
                pass

            if self._proc.poll() is not None:
                stderr_out = self._proc.stderr.read().decode(errors="replace").strip()
                raise RuntimeError(
                    f"llama-server exited early (code {self._proc.returncode}).\n"
                    + (f"  stderr: {stderr_out}" if stderr_out else
                       f"  Check that {self.model_path} is valid.")
                )
            time.sleep(1)

        self._proc.terminate()
        raise TimeoutError(
            f"llama-server did not become healthy within {startup_timeout}s."
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_inference_with_adapter(
        self,
        prompt: str,
        adapter_path: str,
        adapter_id: int,
    ) -> None:
        """
        Activate adapter at adapter_path, then call /completion.

        adapter_path should be the GGUF path (e.g. adapters/biology_lora/biology.gguf).
        adapter_id is 1-indexed (matching legacy interface from StreamingExecutionNode).
        If adapter_path isn't in the registry the call falls back to the base model.
        """
        print(f"\n[LlamaCppNode] SAE Router triggered adapter ID {adapter_id}: {adapter_path}")

        # Determine which concept name maps to this gguf_path
        matched_name = None
        for name, path in self.adapter_registry.items():
            if os.path.abspath(path) == os.path.abspath(adapter_path):
                matched_name = name
                break

        if matched_name and matched_name in self._adapter_index:
            self._set_adapter_scales(active_name=matched_name)
            print(f"[LlamaCppNode] Streaming .mrna via /lora-adapters (adapter: {matched_name})")
        else:
            if not os.path.isfile(adapter_path):
                print(f"[LlamaCppNode] NOTE: '{adapter_path}' not found on disk.")
            else:
                print(f"[LlamaCppNode] NOTE: Adapter not pre-loaded (wrong base model?).")
            print("[LlamaCppNode] Falling back to base model inference.")
            self._set_adapter_scales(active_name=None)

        # Apply chat template if tokenizer is available
        formatted_prompt = self._apply_chat_template(prompt)

        print("[LlamaCppNode] Executing...")
        payload = {
            "prompt": formatted_prompt,
            "n_predict": self.max_tokens,
            "temperature": 0.7,
            "min_p": 0.05,
            "stream": False,
            "cache_prompt": False,
        }

        try:
            r = requests.post(f"{self.base_url}/completion", json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            generated = data.get("content", "").strip()
            print(f"Prompt   : {prompt!r}")
            print(f"Generated: {generated!r}\n")
        except requests.exceptions.RequestException as e:
            print(f"[LlamaCppNode] /completion request failed: {e}\n")

    def _set_adapter_scales(self, active_name: Optional[str]) -> None:
        """POST /lora-adapters to activate one adapter and disable the rest."""
        if not self._adapter_index:
            return  # No adapters loaded

        scales = []
        for name, idx in self._adapter_index.items():
            scale = 1.0 if name == active_name else 0.0
            scales.append({"id": idx, "scale": scale})

        try:
            r = requests.post(
                f"{self.base_url}/lora-adapters",
                json=scales,
                timeout=10,
            )
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"[LlamaCppNode] /lora-adapters update failed: {e}")

    def _apply_chat_template(self, prompt: str) -> str:
        """Apply instruct chat template if a tokenizer is available."""
        if self.tokenizer is None:
            return prompt

        # Unwrap multimodal processor (Gemma 4)
        tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        if not hasattr(tok, "apply_chat_template"):
            return prompt

        try:
            messages = [{"role": "user", "content": prompt}]
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Terminate llama-server and free resources."""
        print("\n[LlamaCppNode] Shutting down llama-server...")
        if self._proc and self._proc.poll() is None:
            self._proc.send_signal(signal.SIGTERM)
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        print("[LlamaCppNode] Shutdown complete.")

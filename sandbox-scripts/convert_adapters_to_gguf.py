"""
Convert .mrna PEFT LoRA adapters to GGUF LoRA format for llama.cpp.

Wraps llama.cpp's convert_lora_to_gguf.py.  Run this once after training
adapters with train_mrna_adapter.py and before using LlamaCppExecutionNode.

IMPORTANT: --base-model must be the HuggingFace model directory (the local
snapshot with config.json), NOT the .gguf file.  convert_lora_to_gguf.py
reads the architecture config from config.json to write the GGUF header.

Usage
-----
    # Recommended: pass HF model ID — auto-resolved to local snapshot cache
    python sandbox-scripts/convert_adapters_to_gguf.py \
        --base-model unsloth/gemma-4-E2B-it \
        --adapters-root data/gemma-4-e2b/adapters \
        --adapters biology chemistry physics

    # Or pass the snapshot dir explicitly:
    python sandbox-scripts/convert_adapters_to_gguf.py \
        --base-model ~/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it/snapshots/<hash> \
        --adapters biology chemistry physics

    # Single adapter:
    python sandbox-scripts/convert_adapters_to_gguf.py \
        --base-model unsloth/gemma-4-E2B-it \
        --adapter-dir adapters/biology_lora \
        --out adapters/biology_lora/biology.gguf

Output
------
    adapters/{concept}_lora/{concept}.gguf
"""

import argparse
import os
import subprocess
import sys

LLAMA_CPP_DIR = os.path.expanduser("~/llama.cpp")
CONVERT_SCRIPT = os.path.join(LLAMA_CPP_DIR, "convert_lora_to_gguf.py")

VENV_PYTHON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".venv", "bin", "python",
)


def resolve_base_model(base_model: str) -> str:
    """Resolve an HF model ID or path to a local directory with config.json.

    convert_lora_to_gguf.py needs a directory containing config.json, not a
    .gguf file.  If base_model looks like an HF model ID (no slash at start,
    not an existing directory), download/resolve via huggingface_hub.
    """
    if os.path.isdir(base_model):
        return base_model

    if os.path.isfile(base_model):
        print(f"[ERROR] --base-model is a file, not a directory: {base_model}")
        print("  convert_lora_to_gguf.py needs the HF model directory with config.json,")
        print("  not the .gguf file.  Pass a model ID (e.g. unsloth/gemma-4-E2B-it)")
        print("  or the local snapshot directory.")
        sys.exit(1)

    # Treat as HF model ID — resolve to local cache
    print(f"[Resolve] Locating local snapshot for '{base_model}' ...")
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(base_model, local_files_only=True)
        print(f"[Resolve] Found: {path}")
        return path
    except Exception as e:
        print(f"[ERROR] Could not resolve '{base_model}': {e}")
        print("  Make sure the model is cached locally (run harvest or training first).")
        sys.exit(1)


def convert_one(base_model: str, adapter_dir: str, outfile: str) -> bool:
    if not os.path.isfile(CONVERT_SCRIPT):
        print(f"[ERROR] convert_lora_to_gguf.py not found at {CONVERT_SCRIPT}")
        print("  Clone llama.cpp to ~/llama.cpp first:")
        print("  git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp --depth=1")
        return False

    if not os.path.isdir(adapter_dir):
        print(f"[SKIP] Adapter dir not found: {adapter_dir}")
        return False

    if not os.path.isfile(os.path.join(adapter_dir, "adapter_model.safetensors")):
        print(f"[SKIP] No adapter_model.safetensors in {adapter_dir}")
        return False

    print(f"\n[Convert] {adapter_dir} → {outfile}")
    cmd = [
        VENV_PYTHON,
        CONVERT_SCRIPT,
        "--base", base_model,
        adapter_dir,
        "--outfile", outfile,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = LLAMA_CPP_DIR

    result = subprocess.run(cmd, env=env, capture_output=False)
    if result.returncode != 0:
        print(f"[ERROR] Conversion failed for {adapter_dir}")
        return False

    size_mb = os.path.getsize(outfile) / 1e6
    print(f"[OK] {outfile}  ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert PEFT LoRA adapters to GGUF format for llama.cpp.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-model",
        default="unsloth/gemma-4-E2B-it",
        help="HF model ID or local snapshot directory with config.json. "
             "NOT the .gguf file — convert_lora_to_gguf.py reads config.json.",
    )
    parser.add_argument(
        "--adapters",
        nargs="+",
        default=["biology", "chemistry", "physics"],
        help="Concept names to convert. Expects adapters/{concept}_lora/ to exist.",
    )
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="Convert a single adapter directory (overrides --adapters).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output GGUF path when using --adapter-dir.",
    )
    parser.add_argument(
        "--adapters-root",
        default="adapters",
        help="Root directory containing {concept}_lora/ subdirectories.",
    )

    args = parser.parse_args()

    base_model = resolve_base_model(args.base_model)

    if args.adapter_dir:
        # Single adapter mode
        outfile = args.out or os.path.join(
            args.adapter_dir,
            os.path.basename(args.adapter_dir.rstrip("/")).replace("_lora", "") + ".gguf",
        )
        ok = convert_one(base_model, args.adapter_dir, outfile)
        sys.exit(0 if ok else 1)

    # Batch mode
    results = {}
    for concept in args.adapters:
        adapter_dir = os.path.join(args.adapters_root, f"{concept}_lora")
        outfile = os.path.join(adapter_dir, f"{concept}.gguf")
        results[concept] = convert_one(base_model, adapter_dir, outfile)

    print("\n" + "=" * 50)
    print("Conversion summary:")
    for concept, ok in results.items():
        status = "OK" if ok else "FAILED / SKIPPED"
        print(f"  {status:15s}  {concept}")

    failed = [c for c, ok in results.items() if not ok]
    if failed:
        print(f"\n[WARN] {len(failed)} adapter(s) not converted: {failed}")
        print("  Train them first with train_mrna_adapter.py")
    else:
        print("\nAll adapters ready for LlamaCppExecutionNode.")


if __name__ == "__main__":
    main()

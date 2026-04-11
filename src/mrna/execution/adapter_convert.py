"""
Standardized GGUF LoRA conversion utility for mRNA.
Wraps llama.cpp/convert_lora_to_gguf.py with mRNA path resolution.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from mrna.core.config import MRNAPaths, config


def resolve_hf_snapshot(model_id: str) -> str:
    """Resolve an HF model ID to a local directory containing config.json."""
    try:
        from huggingface_hub import snapshot_download

        print(f"[Convert] Locating local snapshot for '{model_id}'...")
        path = snapshot_download(model_id, local_files_only=True)
        return path
    except Exception as e:
        print(f"[ERROR] Could not resolve '{model_id}': {e}")
        print("  Ensure the model is cached (run harvest or training first).")
        sys.exit(1)


def convert_adapter(
    concept: str,
    model_id: Optional[str] = None,
    llama_cpp_dir: str = "~/llama.cpp",
    force: bool = False,
) -> bool:
    """Converts a concept adapter from .safetensors to .gguf."""
    mid = model_id or config.current_model_id
    m_cfg = config.get_model_config(mid)
    base_model_id = m_cfg.get("path", mid)

    llama_path = Path(os.path.expanduser(llama_cpp_dir))
    convert_script = llama_path / "convert_lora_to_gguf.py"

    if not convert_script.exists():
        print(f"[ERROR] llama.cpp conversion script not found at {convert_script}")
        print("  Clone llama.cpp to ~/llama.cpp or specify --llama-cpp-dir")
        return False

    # Standard mRNA paths
    adapter_dir = MRNAPaths.get_adapter_dir(concept, mid)
    input_file = adapter_dir / "adapter_model.safetensors"
    output_file = adapter_dir / f"{concept}.gguf"

    if not input_file.exists():
        print(f"[SKIP] No safety tensors found at {input_file}")
        return False

    if output_file.exists() and not force:
        print(f"[SKIP] GGUF already exists: {output_file} (use --force to overwrite)")
        return True

    # Resolve HF snapshot directory (required for architecture config)
    base_dir = resolve_hf_snapshot(base_model_id)

    print(f"[Convert] {concept} ({mid})")
    print(f"  Input:  {input_file}")
    print(f"  Output: {output_file}")

    # We use the venv python to ensure dependencies are matched
    venv_python = MRNAPaths.ROOT / ".venv" / "bin" / "python"

    cmd = [
        str(venv_python),
        str(convert_script),
        "--base",
        str(base_dir),
        str(adapter_dir),
        "--outfile",
        str(output_file),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(llama_path)

    try:
        result = subprocess.run(cmd, env=env, check=True)
        if result.returncode == 0:
            print(f"[OK] Conversion complete: {output_file}")
            return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Conversion failed for {concept}: {e}")

    return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert mRNA adapters to GGUF format."
    )
    parser.add_argument(
        "--concept", required=True, help="Concept to convert (e.g. biology)"
    )
    parser.add_argument("--model", help="Model ID override")
    parser.add_argument(
        "--llama-cpp-dir", default="~/llama.cpp", help="Path to llama.cpp repo"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing GGUF")

    args = parser.parse_args()

    success = convert_adapter(
        concept=args.concept,
        model_id=args.model,
        llama_cpp_dir=args.llama_cpp_dir,
        force=args.force,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

"""
miin_router.py — The Split-Brain Routing Bridge.

Coordinates:
1. Abstract Pass: SAE routing via prompt activation harvesting.
2. Voice Pass: NPC Identity arbitration via PIDX profile lookups.
"""

import json
import os
import subprocess
from typing import Dict, Optional


class MiinRouter:
    def __init__(self, pidx_path: str = "pidx"):
        # We assume the 'pidx' binary is built and in the path,
        # or we could point to target/debug/pidx
        self.pidx_path = pidx_path

    def get_voice_adapter(self, npc_id: str) -> str:
        """
        Calls PIDX to determine the authorized voice adapter for this NPC.
        """
        try:
            # We call the PIDX CLI in JSON mode to get the 'active_adapter' field
            # which is authorized by the Rust Quality Gates.
            result = subprocess.run(
                [self.pidx_path, "--format", "json", "show", npc_id],
                capture_output=True,
                text=True,
                check=True,
            )
            data = json.loads(result.stdout)
            return data.get("active_adapter", "merchant")
        except Exception as e:
            # If PIDX isn't found or NPC doesn't exist, we fall back to a safe default.
            print(f"[miin_router] PIDX bridge failed for {npc_id}: {e}")
            return "merchant"

    def resolve_stack(
        self, sae_route: dict, npc_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Combines SAE abstract logic with PIDX identity to form a LoRA stack.

        Stack weights:
        - Logic LoRA (SAE): 1.0 (Logic Layer) -> Structural integrity of the task.
        - Voice LoRA (PIDX): 0.8 (Persona Layer) -> Persona cadence and slang.
        """
        stack = {}

        # 1. Add Logic LoRA from SAE (The "What")
        logic_adapter = sae_route.get("concept")
        if logic_adapter:
            stack[logic_adapter] = 1.0

        # 2. Add Voice LoRA from PIDX (The "Who")
        if npc_id:
            voice_adapter = self.get_voice_adapter(npc_id)
            # Both adapters are loaded simultaneously via llama-server's /lora-adapters
            stack[voice_adapter] = 0.8

        return stack

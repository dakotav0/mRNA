"""
.mrna FastAPI sidecar — Garden integration endpoint.

Runs as a persistent background service at localhost:7437.
PrefillRouter + LlamaCppExecutionNode are initialised once on startup;
each request pays only the SAE forward pass (~500ms) + llama-server generation.

Endpoints
---------
GET  /health    — readiness probe (Garden polls before sending requests)
POST /route     — SAE routing decision only, no generation
POST /generate  — route + generate; streams Garden-format SSE events

Start
-----
    source .venv/bin/activate
    python src/mrna/api.py
    # or via start-mrna.sh

Garden talks to this at MRNA_URL (default http://localhost:7437).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Resolve project root so src/ modules import cleanly regardless of cwd
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mrna.core.config import MRNAPaths, config
from src.mrna.execution.llama_cpp_node import LlamaCppExecutionNode
from src.mrna.router.interceptor import ActivationInterceptor
from src.mrna.router.sae import CBSAE

# ---------------------------------------------------------------------------
# Configuration (Dynamic from model_config.yaml)
# ---------------------------------------------------------------------------

m_id = config.current_model_id
m_cfg = config.get_model_config(m_id)

MODEL_ID = m_cfg["path"]
MODEL_REVISION = m_cfg.get("revision")
MODEL_PATH = str(MRNAPaths.ROOT / m_cfg.get("gguf_path", f"models/{m_id}.gguf"))
LAYER = m_cfg.get("harvest_layer", 0)
D_MODEL = m_cfg.get("d_model", 2048)
SAE_WEIGHTS = str(MRNAPaths.DATA / m_id / "sae_weights.pt")

PORT = int(os.getenv("MRNA_PORT", "7437"))

CONCEPT_NAMES = list(config.science_triad_datasets.keys())

# Derive GGUF adapters from data hierarchy
GGUF_ADAPTER_REGISTRY = {
    c: str(MRNAPaths.DATA / m_id / "adapters" / f"{c}_lora" / f"{c}.gguf")
    for c in CONCEPT_NAMES
}

# ---------------------------------------------------------------------------
# Global state — initialised once at startup
# ---------------------------------------------------------------------------

_router: Optional["_PrefillRouter"] = None
_execution: Optional[LlamaCppExecutionNode] = None
_ready = False


class _PrefillRouter:
    """Thin wrapper around unsloth + CBSAE, mirroring prefill_router.py."""

    def __init__(self):
        import torch
        from unsloth import FastLanguageModel

        print(
            f"[mRNA API] Loading {MODEL_ID!r} @ {MODEL_REVISION[:8]} for prefill routing …"
        )
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_ID,
            revision=MODEL_REVISION,
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        self.model.eval()

        self.interceptor = ActivationInterceptor(target_layer=LAYER)
        self.interceptor.attach_to_model(self.model)

        self.sae = CBSAE(
            d_model=D_MODEL, expansion_factor=8, bottleneck_features=len(CONCEPT_NAMES)
        )
        import torch as _torch

        if os.path.exists(SAE_WEIGHTS):
            self.sae.load_state_dict(
                _torch.load(SAE_WEIGHTS, map_location="cpu", weights_only=True)
            )
            print(f"[mRNA API] SAE weights loaded from {SAE_WEIGHTS}")
        else:
            print(
                f"[mRNA API] WARNING: SAE weights not found at {SAE_WEIGHTS} — routing will be random"
            )
        self.sae.eval()

    def route(self, prompt: str) -> dict:
        import torch

        _tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        enc = _tok(
            text=[prompt],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(next(self.model.parameters()).device)

        t0 = time.perf_counter()
        with torch.no_grad():
            self.model(**enc)

        acts = self.interceptor.intercepted_activations[-1]  # (1, seq, d_model)
        self.interceptor.intercepted_activations.clear()

        mask = enc["attention_mask"].cpu().unsqueeze(-1).float()
        pooled = ((acts * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).float()
        pre_relu = self.sae.encoder(pooled)
        bottleneck = pre_relu[:, : len(CONCEPT_NAMES)]
        strengths = bottleneck[0]

        concept_idx = int(strengths.argmax().item())
        scores = {n: float(strengths[i].item()) for i, n in enumerate(CONCEPT_NAMES)}
        return {
            "concept": CONCEPT_NAMES[concept_idx],
            "concept_idx": concept_idx,
            "confidence": scores[CONCEPT_NAMES[concept_idx]],
            "scores": scores,
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title=".mrna API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup():
    global _router, _execution, _ready

    # Run the heavy model load in a thread so the event loop stays responsive
    loop = asyncio.get_event_loop()

    def _init():
        global _router, _execution
        _router = _PrefillRouter()
        _execution = LlamaCppExecutionNode(
            model_path=MODEL_PATH,
            adapter_registry=GGUF_ADAPTER_REGISTRY,
            tokenizer=_router.tokenizer,
        )

    await loop.run_in_executor(None, _init)
    _ready = True
    print("[mRNA API] Ready — listening on port", PORT)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RouteRequest(BaseModel):
    prompt: str


class GenerateRequest(BaseModel):
    prompt: str
    force_adapter: Optional[str] = None  # "biology" | "chemistry" | "physics"
    max_tokens: int = 2048


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ready" if _ready else "loading"}


@app.post("/route")
def route(req: RouteRequest):
    if not _ready:
        raise HTTPException(503, "Service initialising — retry in a moment")
    return _router.route(req.prompt)


@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    Route prompt through SAE, activate the correct adapter, stream generation.

    Emits Garden-format SSE events:
        event: token\\ndata: "<text>"\\n\\n
        event: mrna_route\\ndata: {concept, confidence, scores, latency_ms}\\n\\n
        event: done\\ndata: {concept, generated_by, total_ms}\\n\\n
        event: error\\ndata: "<message>"\\n\\n
    """
    if not _ready:
        raise HTTPException(503, "Service initialising — retry in a moment")

    async def _stream() -> AsyncGenerator[str, None]:
        t_start = time.perf_counter()
        loop = asyncio.get_event_loop()

        # ── 1. Routing decision ─────────────────────────────────────────────
        if req.force_adapter and req.force_adapter in CONCEPT_NAMES:
            concept = req.force_adapter
            adapter_path = GGUF_ADAPTER_REGISTRY[concept]
            route_info = {
                "concept": concept,
                "concept_idx": CONCEPT_NAMES.index(concept),
                "confidence": 1.0,
                "scores": {c: (1.0 if c == concept else 0.0) for c in CONCEPT_NAMES},
                "latency_ms": 0.0,
                "forced": True,
            }
        else:
            route_info = await loop.run_in_executor(None, _router.route, req.prompt)
            concept = route_info["concept"]
            adapter_path = GGUF_ADAPTER_REGISTRY[concept]

        yield f"event: mrna_route\ndata: {json.dumps(route_info)}\n\n"

        # ── 2. Activate adapter & generate ─────────────────────────────────
        def _generate_sync() -> dict:
            """Runs in thread — returns generated text + llama-server timing."""
            # Activate adapter via llama-server /lora-adapters
            _execution._set_adapter_scales(active_name=concept)

            # Apply chat template
            formatted = _execution._apply_chat_template(req.prompt)

            import requests as _req

            payload = {
                "prompt": formatted,
                "n_predict": req.max_tokens,
                "temperature": 0.7,
                "min_p": 0.05,
                "stream": True,
                "cache_prompt": False,
            }
            full_text = ""
            tokens_predicted = 0
            tokens_evaluated = 0
            tok_per_sec = 0.0
            with _req.post(
                f"{_execution.base_url}/completion",
                json=payload,
                stream=True,
                timeout=120,
            ) as r:
                r.raise_for_status()
                for raw_line in r.iter_lines():
                    if not raw_line:
                        continue
                    line = (
                        raw_line.decode("utf-8")
                        if isinstance(raw_line, bytes)
                        else raw_line
                    )
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    full_text += chunk.get("content", "")
                    if chunk.get("stop"):
                        tokens_predicted = chunk.get("tokens_predicted", 0)
                        tokens_evaluated = chunk.get("tokens_evaluated", 0)
                        timings = chunk.get("timings", {})
                        tok_per_sec = round(timings.get("predicted_per_second", 0.0), 1)
            return {
                "text": full_text,
                "tokens_predicted": tokens_predicted,
                "tokens_evaluated": tokens_evaluated,
                "tok_per_sec": tok_per_sec,
            }

        # Stream tokens: we collect in a thread and emit word-by-word for now
        # (llama-server streaming over subprocess is synchronous; full async
        #  streaming requires aiohttpx — acceptable for v1)
        try:
            generated = await loop.run_in_executor(None, _generate_sync)
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps(str(exc))}\n\n"
            return

        # Emit the full text as tokens split on spaces (preserve streaming feel)
        words = generated["text"].split(" ")
        for i, word in enumerate(words):
            token = word if i == len(words) - 1 else word + " "
            yield f"event: token\ndata: {json.dumps(token)}\n\n"

        total_ms = (time.perf_counter() - t_start) * 1000
        yield f"event: done\ndata: {
            json.dumps(
                {
                    'concept': concept,
                    'generated_by': f'{m_id}-mrna',
                    'total_ms': round(total_ms),
                    'tokens_generated': generated['tokens_predicted'],
                    'tokens_prompt': generated['tokens_evaluated'],
                    'tok_per_sec': generated['tok_per_sec'],
                }
            )
        }\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "src.mrna.api:app", host="0.0.0.0", port=PORT, reload=False, log_level="info"
    )

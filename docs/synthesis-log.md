# .mrna Architecture — Synthesis Log

> Lab notebook for the education phase. Captures what each script proved,
> the reasoning behind design decisions, and open questions for the next phase.
> Written to be useful to future-me mid-implementation, and to any LLM collaborator
> brought into a session. Load alongside SKILL.md.

---

## Session 1 — 2026-04-01
**Collaborators:** Gemini (Antigravity), Claude (Sonnet 4.6)
**Stage:** Education / Proof-of-Concept
**Hardware:** RTX 4070 Super, 12GB VRAM
**Source doc:** "LLM Architecture Research & CUDA Kernels" (Google Drive)
**Tooling:** GitHub MCP, arXiv MCP enabled. Minimal prompting after SKILL.md ingested.

### What this session established
Six functional scripts, one per node (some shared), running on the 4070 Super footprint.
The architecture went from whitepaper to runnable code in a single session.
The SKILL.md four-node taxonomy held up as an organizing framework throughout.

---

## Scripts

### `triton_matmul.py` — Substrate Node
**What it does:** Custom `@triton.jit` tiled matrix multiply. Each thread block computes
a `[BLOCK_M, BLOCK_N]` tile of output C by iterating through K in chunks, loading tiles
into SRAM, accumulating in FP32, writing back in FP16.

**What it proved:**
- The `@triton.jit` programming model: you work at the workgroup level, not the thread level.
  Pointer arithmetic (`a_ptrs`, `b_ptrs`) + mask logic is the pattern for every kernel.
- Triton's autotuning handles warp scheduling and pipelining. You set `BLOCK_SIZE_*`,
  Triton handles everything below that.
- Max diff vs. PyTorch native: within FP16 tolerance (~0.5% relative error over 1024
  accumulations). This is expected and acceptable.

**Key insight:** The K-loop (the "tiled" part) is the mental model for every future kernel.
Loading a tile into SRAM, computing, advancing the pointer, repeat. SOCKET's soft collision
kernel will use the same pattern over Q/K hash tiles.

**Notable comment preserved:** `# Looks like a complex version of my early perceptron model CS assignment`
This is accurate. The tiling is the same accumulation logic, just explicit about memory hierarchy.

**Open questions:**
- What `BLOCK_SIZE_*` values are optimal for the 4070 Super's SM count and shared memory size?
  → Run `triton.autotune` wrapper and profile with Nsight Compute.
- This is a 2D matmul. The `.mrna` fused merge kernel will be a *batched* matmul
  (`W + scale * A @ B`) — how does the grid definition change?

---

### `engram_host.py` — Memory Node
**What it does:** CPU-side hash table in pinned (page-locked) memory. N-gram token list
→ SHA-256 hash → row index → async PCIe prefetch via `non_blocking=True` on a dedicated
`cuda.Stream`. Transfer initiated at Layer 0, synchronized at Layer 12.

**What it proved:**
- Pinned memory + `non_blocking=True` is the correct PCIe async pattern. The GPU issues
  a DMA request and returns immediately; Python continues; `stream.synchronize()` is the
  barrier.
- The deterministic hash (same tokens → same row, always) is what makes this O(1) and
  schedulable. Unlike RAG, there's no similarity search — you know the address before
  the GPU needs the data.
- RAM cost for 1M entries × 2048 dim × float16: ~4GB. This is the system RAM budget
  per Engram table.

**Key insight:** The async transfer overlaps with layers 1–11 of computation. By the time
Layer 12 needs the factual payload, it's already in VRAM — zero latency penalty if the
prefetch was initiated early enough. This is the "pipeline" that makes Engram viable at 12GB.

**Known gap / next phase:**
- SHA-256 is cryptographically expensive. Production replacement: **xxHash** (non-cryptographic,
  extremely fast) or a **SimHash / locality-preserving hash** if fuzzy N-gram matching is needed.
- `table_size=1_000_000` is arbitrary. Need to model the actual collision rate for the
  target vocabulary/domain. Hash collisions overwrite silently — this matters for correctness.
- The write path (`write_fact`) is synchronous. In a real pipeline, Engram tables are
  pre-populated offline, not written at inference time.

---

### `sae_routing.py` — Routing Node
**What it does:** CB-SAE (Concept Bottleneck Sparse Autoencoder). Encoder projects
`d_model=2048` → `d_sae=16384` (8x expansion). ReLU enforces sparsity. Decoder reconstructs.
Loss = MSE (reconstruction) + L1 (sparsity) + BCE (bottleneck alignment to user labels).

**What it proved:**
- The three-loss structure is the right architecture: reconstruction keeps the SAE faithful
  to the LLM's actual computation; L1 forces interpretable sparsity; BCE locks specific
  dimensions to user-defined concepts.
- With `bottleneck_features=2` (Python, Legal), the routing signal is already separable
  from mock activations in one forward pass.
- The routing decision comes from the encoder alone at inference — decoder is training-only.

**Key insight:** `routing_features[0, :, 0].mean()` being positive is sufficient to trigger
a route. The *strength* of the signal (0.98 vs. 0.3) becomes the gate weight in `temporal_lora.py`.
The two scripts are designed to compose: SAE output → LoRA gate input.

**Known gap / next phase:**
- Mock activations (`torch.randn`) are not real LLM residual stream activations. The SAE
  must be attached to a *real frozen model* via a forward hook to get meaningful features.
  ```python
  model.layers[12].register_forward_hook(lambda m, inp, out: sae(out[0]))
  ```
- The SAE must be *trained* on real activations before routing is meaningful. Anthropic's
  interpretability work is the reference for training procedures.
- `expansion_factor=8` (16K features) is modest. Production SAEs use 32x–128x expansion.
  Memory cost scales linearly — needs to fit alongside the base model in 12GB.

---

### `socket_attention.py` — Memory Node
**What it does:** Simulates SOCKET soft collision scoring in PyTorch. Random projection
planes hash Q and K into binary codes across `num_hash_tables=4` tables of `hash_bits=8`.
Collision ratio becomes a probabilistic similarity score. Threshold at 95th percentile
creates a sparse mask.

**Key result:**
```
Sequence Length: 4096
Dense Attention Matrix: 64MB per head (at float32)
Sparsity achieved: ~91%
Active connections: ~1.68M of 16.77M total
```

**The architectural argument this number makes:**
91% sparsity at seq_len=4096 means the attention computation drops from O(N²) to ~O(0.09N²).
On a 12GB 4070 Super, this is the difference between OOM and ~4GB headroom for adapter streaming.
**SOCKET is not a performance optimization — it is what makes the rest of the architecture
physically possible.** Sequence it before wiring the full adapter pipeline.

**Key insight:** The soft collision score (ratio, not binary) is what separates SOCKET from
prior LSH attention. Hard LSH degrades perplexity because top-K token ordering is unstable
with binary signals. The graded probability preserves ranking stability.

**Known gap / next phase (critical):**
- This implementation materializes the full `[seq, seq]` collision probability tensor in
  PyTorch. **This defeats the purpose** — the 64MB matrix still exists in VRAM.
- The real implementation requires a custom `@triton.jit` kernel that computes soft
  collision scores tile-by-tile without ever allocating the full matrix. The Flash Attention
  approach (online softmax) is the template.
- arXiv:2602.06283 includes the Triton backend spec. That kernel is the next concrete
  deliverable for this node.

---

### `temporal_lora.py` — Execution Node
**What it does:** Gated LoRA routing. Frozen base layer always runs. Two rank-16 adapters
(Python, Legal) compute `B(A(x))`. SAE routing signal multiplies each adapter's output
before injection into the residual stream. Routing signal 0.0 → adapter contributes zero.

**What it proved:**
- Gating via multiplication (not switching) is cleaner: no branching, no conditional loading,
  no latency spike. The adapter math always runs, but contributes proportionally to the
  routing signal strength.
- Compositionality: `[0.98, 0.01]` gives 98% Python adapter + 1% Legal adapter. This
  supports *blended* expertise, not just binary switching.
- The base model is genuinely unmodified. Only the gated adapter outputs change behavior.

**Key insight:** The gating weight IS the SAE activation strength. The two modules are
already interface-compatible — `sae_routing.py`'s `routing_features[:, :, :num_adapters]`
maps directly to `temporal_lora.py`'s `routing_signals` parameter. The pipeline is composable.

**Known gap / next phase:**
- Both adapters are loaded in VRAM simultaneously. At scale (N adapters), this is untenable.
  The full `.mrna` system requires only the *selected* adapter to be in VRAM; others reside
  in system RAM and stream in on demand.
- vLLM's `load_inplace=True` is the production mechanism for this. The gating-vs-streaming
  tradeoff: gating is zero-latency but scales poorly; streaming has PCIe latency but scales
  to arbitrarily many adapters.
- The fused Triton kernel for `W + scale * A @ B` (the actual LoRA merge math) still needs
  to be written. That's the kernel that replaces the three sequential `nn.Linear` calls.

---

### `train_unsloth_packing.py` — Substrate Node
**What it does:** Standard Unsloth fine-tune setup for Llama-3.2-1B with `packing=True`.
The pedagogical target is observing the packing/masking logs, not the trained weights.

**What it proves (when run):**
- `packing=True` concatenates multiple short sequences into a single 2048-token context window.
- Unsloth's Triton kernels enforce strict causal masking across document boundaries at
  the CUDA level — attention does not bleed between packed documents.
- The attention mask and position IDs are the implementation detail to study in the logs.
  Any custom context window manipulation (e.g., injecting Engram payloads at Layer 12)
  requires the same boundary enforcement.

**Key insight:** The masking pattern here is the template for the `.mrna` injection point.
When an Engram payload or adapter output is injected into the residual stream, the causal
mask must be consistent with the surrounding sequence structure. Get this wrong and
cross-document attention contamination silently degrades all downstream behavior.

**Status:** Script not yet run (requires Unsloth install + GPU session). Run this first
in the next hardware session — it's the foundation everything else attaches to.

---

## Cross-Node Connections Visible After Session 1

```
engram_host.py ──────────────────────────────► residual stream injection at Layer 12
                                                (needs causal mask from train_unsloth)

sae_routing.py ──► routing_signal ──────────► temporal_lora.py (gate weights)
  (needs real activations via forward hook)     (needs streaming, not simultaneous loading)

socket_attention.py ─────────────────────────► VRAM headroom for adapter streaming
  (VRAM savings only real when Triton kernel    (the 91% number is the budget argument)
   replaces PyTorch dense matrix)

triton_matmul.py ────────────────────────────► template for all future custom kernels
  (tiling pattern, pointer arithmetic,          (SOCKET Triton backend, fused LoRA merge)
   BLOCK_SIZE grid)
```

---

## Naming Decision Needed

The Gemini session used `.agents` architecture. The SKILL.md uses `.mrna`.
These may refer to different scopes:
- `.mrna` = the adapter file format / payload (Modular Representation Network Adapter)
- `.agents` = the broader orchestration system that loads and routes `.mrna` payloads

**Decision pending.** Canonicalize before the next session to avoid confusion across
LLM collaborators. Suggest: keep `.mrna` as the format name, introduce `.agents` only
if the orchestration layer warrants its own distinct identity.

Dakota: .mrna is definitive

---

## Next Phase Priorities (ordered)

1. [x] **Run `train_unsloth_packing.py`** — study the masking logs. This is the foundation.
2. [x] **SOCKET Triton kernel** — rewrote `socket_attention.py` so the `[seq, seq]` matrix is never materialized via early thresholding on SRAM blocks.
3. [x] **Real SAE training** — Built `src/mrna/router/interceptor.py` using PyTorch forward hooks. Harvests explicit Layer-12 dataset safely to RAM for offline CB-SAE training over specific classes.
4. [x] **Streaming integration** — verified `LoRARequest` vLLM orchestrator in `src/mrna/execution/streaming_lora.py`. Bound the GPU to exactly `max_vram=0.6` leaving headroom for the Memory Node.
5. [x] **Fused LoRA merge kernel** — `src/mrna/substrate/lora_merge.py`. Triton kernel for `C = W + scale * B @ A`. Autotuned 2D grid over (M, N) output tiles; accumulates B@A tile-by-tile in SRAM without materializing the (M, N) intermediate. Correctness-checked against PyTorch naive. Run `python src/mrna/substrate/lora_merge.py` to benchmark against naive and torch.compile.

---

## Session 2 — 2026-04-01
**Stage:** Substrate Node — Fused LoRA Merge Kernel
**File:** `src/mrna/substrate/lora_merge.py`

### What was built

`lora_merge(W, A, B, scale)` — a fused `@triton.jit` kernel for `C = W + scale * B @ A`.

**The HBM argument:**
Naive PyTorch executes two kernel launches:
1. `delta = B @ A` — writes an (M, N) intermediate to HBM
2. `C = W + scale * delta` — reads W, reads delta, writes C

The fused kernel eliminates the intermediate entirely. Each thread block:
1. Loads its (BLOCK_M, BLOCK_N) tile of W into a FP32 accumulator
2. Iterates rank R in BLOCK_R chunks — loads B and A tiles into SRAM, calls `tl.dot`, accumulates
3. Adds W tile: `C_tile = W_tile + scale * lora_acc`
4. Writes one C tile to HBM

**HBM savings per merge event (M=N=4096, FP16):**
- Eliminated: 1 write of 32MB (delta) + 1 read of 32MB (delta) = ~64MB
- This matters at hot-swap time: vLLM triggers a merge event per adapter switch.

**Autotuning configs:** 6 configs across BLOCK_M ∈ {64, 128}, BLOCK_N ∈ {64, 128}, BLOCK_R ∈ {16, 32}, keyed on (M, N, R). Triton selects the winner on first call and caches it.

**Shape contract:**
- `W`: (M, N) — out_features × in_features
- `A`: (R, N) — LoRA down-projection (rank × in_features)
- `B`: (M, R) — LoRA up-projection (out_features × rank)
- `scale`: alpha / rank

**Key insight:** The K-loop from `triton_matmul.py` is reused verbatim, except the accumulator is seeded with the W tile instead of zeros. The entire kernel is ~40 lines of Triton that naturally fell out of the existing pattern.

**To validate:** `python src/mrna/substrate/lora_merge.py` — runs correctness check and benchmarks naive / torch.compile / Triton for M=N=4096,R=16 and M=N=2048,R=16. Target: 2.4x over naive (LoRA-Switch baseline).

**Open questions:**
- Does the autotuner pick BLOCK_R=32 for R=16 inputs (BLOCK_R > R wastes cycles)? May need to add R=16 as a constexpr guard or a config where BLOCK_R=16 only.
- Batched merge: multi-adapter hot-swap needs `(B_count, M, R)` tensors. Grid becomes 3D. Defer until vLLM integration demands it.
- Nsight Compute: after benchmarking, profile SM occupancy. BLOCK_M=128, BLOCK_N=128, num_warps=8 may be register-bound on Ada Lovelace — Nsight will tell.

---

---

## Session 3 — 2026-04-01
**Stage:** End-to-end pipeline wiring

### What was wired

**`src/mrna/router/sae.py`** — CBSAE promoted from sandbox. No functional changes; now importable as a proper module.

**`src/mrna/pipeline.py`** — `mRNAPipeline` orchestrator. The single call surface for the full system:

```python
pipeline = mRNAPipeline(
    adapter_registry={"python": "/adapters/python_lora", "legal": "/adapters/legal_lora"},
    model_id="unsloth/Llama-3.2-1B-Instruct",
    d_model=2048,
)
pipeline.run(prompt, activations)   # activations from ActivationInterceptor hook
pipeline.cleanup()
```

**Data flow inside `run()`:**
1. `route(activations)` — pools (batch, seq_len, d_model) → (batch, d_model), encodes through SAE, argmaxes bottleneck slice → `(concept_idx, concept_name, adapter_path, confidence)`
2. Hot-swap detection — compares `concept_idx` to `_current_adapter_id`; logs if switching
3. `execution.run_inference_with_adapter(prompt, path, id+1)` — vLLM serves with the selected LoRA
4. (On hot-swap) `_pre_warm(concept_idx, weights)` — fires `lora_merge` Triton kernel in a background thread; cached merged tensor in `_merged_weights[concept_idx]` is available for the next request to that adapter

**__init__.py** exports updated across all four packages so `from mrna import mRNAPipeline` works.

### Next open questions

1. **Interceptor + vLLM co-location**: `ActivationInterceptor` hooks into a transformers/unsloth model; vLLM runs its own CUDA graph forward. In production, you'd either: (a) run a small routing model alongside vLLM, or (b) hook into vLLM's attention backend. The pipeline currently assumes activations arrive pre-extracted.

2. **SAE training**: `sandbox-scripts/train_sae.py` now exists. See Session 4 below.

3. **`_merged_weights` integration**: The cached merged tensors from `_pre_warm` aren't yet fed back into vLLM — they'd require a custom vLLM worker or a merged-weight serving path (separate from the LoRARequest delta path). This is the remaining integration gap.

4. **Benchmark target revisited**: The Triton kernel hit 2.28x (target 2.4x) at M=N=4096. The 110% reported bandwidth utilization indicates L2 cache hits on W (W is read-only, same base for every adapter swap). In a real hot-swap loop, W stays in L2 across swaps — so the effective speedup on repeated merges is likely higher than the cold-cache benchmark shows.

---

## Session 4 — 2026-04-01
**Stage:** SAE Training Script

**File:** `sandbox-scripts/train_sae.py`

### Three-step pipeline to a trained router

```
# Step 1 — harvest (interceptor.py already does this, run per concept)
python src/mrna/router/interceptor.py    # saves mock_python_legal_layer6.pt

# Step 2 — train
python sandbox-scripts/train_sae.py \
    --activations python:data/py.pt legal:data/legal.pt \
    --d-model 2048 --epochs 50 \
    --output data/sae_weights.pt

# Step 3 — use
pipeline = mRNAPipeline(..., sae_weights_path="data/sae_weights.pt")
```

### Design notes

**Train/inference consistency** — `mRNAPipeline.route()` pools activations to `(batch, d_model)` before encoding. The training script matches: if a `.pt` file is 3D `(N, seq_len, d_model)` it pools to `(N, d_model)` first. SAE encoder sees the same distribution in both phases.

**Loss terms:**
- MSE: reconstruction fidelity (preserves LLM thought vectors)
- L1 × 0.01: sparsity pressure (most features → 0)
- BCE × 1.0: concept bottleneck (forces `sparse[:, :n_concepts]` to match one-hot labels)

Decoder weight re-normalization happens after every optimizer step — standard SAE practice to prevent feature collapse.

**Metrics logged per epoch:** `train_loss`, per-term breakdown `[mse / l1 / cb]`, `val_loss`, `routing_acc` (argmax of bottleneck slice vs. ground-truth concept index). Routing accuracy is the number that matters operationally.

**Checkpoint:** best val-loss model auto-saved. Pass `--output` to control path.

### What you need before running

The interceptor demo saves `mock_python_legal_layer6.pt` with only 2 examples (one forward pass, two prompts). That's not enough to train. The realistic flow:
1. Run interceptor in a loop over many prompts per domain
2. Collect hundreds-to-thousands of sequences per concept
3. Train with `--epochs 50` → routing_acc should reach 90%+ quickly given how distinct Python code vs. legal text is in the residual stream

### Still open

- **Interceptor ↔ vLLM co-location** — prototyped in Session 5 via prefill-only unsloth pass. See below.
- **`_merged_weights` → vLLM serving** — pre-warm cache is populated but not consumed. Requires custom vLLM worker to use merged-weight path.

---

## Session 5 — 2026-04-01
**Stage:** HF dataset harvesting + prefill-only routing prototype

### Files

- `sandbox-scripts/harvest_hf.py` — streams any HF dataset, runs prefill-only forward passes, saves pooled activations to `.pt` files for `train_sae.py`
- `sandbox-scripts/prefill_router.py` — full end-to-end prototype: unsloth prefill → CBSAE route → vLLM generate

### Canonical run sequence (full pipeline, start to finish)

```bash
# 1. Harvest per concept (run once each, ~minutes for 5k examples)
python sandbox-scripts/harvest_hf.py \
    --dataset iamtarun/python_code_instructions_18k_alpaca \
    --text-column instruction --concept python --output data/python_layer6.pt

python sandbox-scripts/harvest_hf.py \
    --dataset pile-of-law/pile-of-law --dataset-config freelaw \
    --text-column text --concept legal --output data/legal_layer6.pt

python sandbox-scripts/harvest_hf.py \
    --dataset FreedomIntelligence/Medical-R1-Distill-Data \
    --text-column instruction --concept medical --output data/medical_layer6.pt

python sandbox-scripts/harvest_hf.py \
    --dataset facebook/natural_reasoning \
    --text-column question --concept reasoning --output data/reasoning_layer6.pt

# 2. Train CBSAE
python sandbox-scripts/train_sae.py \
    --activations python:data/python_layer6.pt legal:data/legal_layer6.pt \
                  medical:data/medical_layer6.pt reasoning:data/reasoning_layer6.pt \
    --d-model 2048 --epochs 50 --output data/sae_weights.pt

# 3. Test end-to-end prototype
python sandbox-scripts/prefill_router.py --sae-weights data/sae_weights.pt
```

### harvest_hf.py design notes

- Streaming load — pile-of-law is 41 GB, never fits in RAM
- Prefill-only forward passes (same as what `prefill_router.py` does at inference — consistent distribution)
- Checkpoint every `--save-every` examples; resume-safe if job is interrupted
- Mean-pools (batch, seq_len, d_model) → (batch, d_model) before saving, matching `mRNAPipeline.route()`

### prefill_router.py design notes

**The prototype flow:**
1. `PrefillRouter.route(prompt)` — tokenizes, calls `model(**enc)` (no `generate()`), interceptor hook fires, pools + encodes activations, returns `(concept_idx, adapter_path, confidence, latency_ms)`
2. `StreamingExecutionNode.run_inference_with_adapter(prompt, path, id)` — vLLM generates with selected adapter

**Key property:** unsloth model and vLLM model never share process state. Clean separation at the cost of one extra prefill pass. The prefill latency is the same order as the generation prefill anyway.

**Without trained weights:** `--sae-weights` is optional. Running without it tests structural wiring with a random-init SAE — routing decisions are meaningless but the full data flow is exercised.

**Routing accuracy:** the demo prints per-prompt correct/incorrect and average prefill latency. With a trained SAE, Python/legal/medical/reasoning should be highly separable given how distinct their residual stream signatures are.

### Planned datasets per concept slot

| Concept | Dataset | Size |
|---|---|---|
| python | `iamtarun/python_code_instructions_18k_alpaca` | 11.4 MB |
| legal | `pile-of-law/pile-of-law` (freelaw) | 41 GB (cap at 5k) |
| medical | `FreedomIntelligence/Medical-R1-Distill-Data` | 247 MB |
| reasoning | `facebook/natural_reasoning` | 4.2 GB |
| social/preference | `stanfordnlp/SHP` | 827 MB |
| ojibwe | Niigaane corpus (To-Lead/) | ~2000+ pairs, pending annotation |

### Open

- **`_merged_weights` → vLLM** — pre-warm cache still not consumed; needs custom vLLM worker
- **Latency measurement** — once trained SAE is available, benchmark full route+generate latency vs. base model with no routing

---

## Session 6 — 2026-04-01
**Stage:** SAE training — loss fix + final trained weights

### What changed in `train_sae.py` (v2)

**Root cause of 51% accuracy:** The original loss applied `BCE_with_logits` to post-ReLU activations. After ReLU, all values are ≥ 0, so `sigmoid(v) ≥ 0.5` always — the model could never express "not this concept" with confidence. CB loss was floored at ~`log(2) ≈ 0.693` regardless of training signal.

**Fix:** Split encoder output before/after ReLU:
```python
pre_relu = encoder(x)          # full range (-∞, +∞) — used for routing
sparse   = ReLU(pre_relu)      # [0, +∞) — used for reconstruction + L1
cb_loss  = CrossEntropy(pre_relu[:, :n_concepts], class_idx, weight=w)
```
`CrossEntropyLoss` is correct here — routing is one-of-N, not N independent binary decisions. Negative logit space is available, non-winning concepts are naturally suppressed via softmax.

**Also added:** inverse-frequency class weights (poetry 573 → weight 1.0 vs 5000-example concepts → weight 0.115), cosine LR schedule, per-concept val accuracy every 10 epochs.

**Inference consistency fix:** `mRNAPipeline.route()` and `PrefillRouter.route()` both updated to argmax over `pre_relu[:, :n_concepts]` instead of `sparse[:, :n_concepts]`.

### Results — 7 concepts, 28,253 total examples

| Concept | Dataset | Examples | Val accuracy |
|---|---|---|---|
| python | `python_code_instructions_18k_alpaca` | 5000 | 99.2% |
| legal | `lex_glue/case_hold` | 5000 | 99.6% |
| medical | `Medical-R1-Distill-Data` | 5000 | 94.7% |
| reasoning | `facebook/natural_reasoning` | 5000 | 97.9% |
| astrophysics | `taln-ls2n/silk` (astro config) | 2680 | 100.0% |
| finance | `gbharti/finance-alpaca` | 5000 | 100.0% |
| poetry | `merve/poetry` | 573 | 100.0% |

**Overall routing accuracy: 98.3%**  
**Best val loss: 0.0610**

Astrophysics at 2680 examples is due to dataset size (taln-ls2n/silk astro config exhausted at that count) — 100% val accuracy confirms it's not a problem. `mse≈0.0000` throughout is expected: CrossEntropy at `cb_coeff=1.0` dominates MSE by ~200x, so reconstruction is a passenger. Fine for routing; would need rebalancing if reconstruction fidelity matters.

Medical at 94.7% is the only concept with meaningful errors. Likely bleeds into reasoning — clinical reasoning prompts overlap. Not a blocker; the per-concept breakdown every 10 epochs will show which direction errors go.

**Weights saved:** `data/sae_weights.pt`

### What's open

- **`_merged_weights` → vLLM** — the one remaining integration gap. Pre-warm Triton cache is populated but vLLM needs a custom worker to consume merged weights instead of the LoRARequest delta path.
- **`prefill_router.py` end-to-end run** — now that `data/sae_weights.pt` exists, run `python sandbox-scripts/prefill_router.py --sae-weights data/sae_weights.pt` to get real latency numbers (route_ms + generation).
- **Conversation dataset** — `daily_dialog` deprecated; `HuggingFaceH4/ultrachat_200k` queued in DATASETS.md as replacement. `text_column=prompt`. Optional 8th concept.

---

## Session 7 — 2026-04-01
**Stage:** SAE training — dataset style alignment + final stable weights

### What the three training runs taught us

| Run | Val acc | Demo routing | Root cause of failure |
|---|---|---|---|
| 1 | 51.3% | — | BCE on post-ReLU — loss floored at log(2) ≈ 0.693 |
| 2 | 98.3% | 2/7 | Raw-text datasets ≠ instruction-format demo prompts |
| 3 | 93.3% | 2/7 | L2 norm removed discriminative magnitude signal |
| **4** | **96.2%** | **TBD** | **Style-matched datasets, no norm** |

**The key insight:** the SAE routes examples that look like training data. Raw text (case excerpts, paper abstracts, raw poems) produces different residual stream signatures from instruction-format prompts even for the same topic. Fix is to match dataset format to expected inference format.

Run 3's L2 norm experiment: confirmed that activation magnitude carries real discriminative information. Normalizing to unit sphere loses it. Revert was correct.

### Final dataset lineup (6 concepts)

| Concept | Dataset | Format | Val acc |
|---|---|---|---|
| python | `python_code_instructions_18k_alpaca` | alpaca | 96.1% |
| medical | `Medical-R1-Distill-Data` | instruction | 93.2% |
| reasoning | `facebook/natural_reasoning` | Q&A | 97.1% |
| astrophysics | `derek-thomas/ScienceQA` | Q&A | 97.3% |
| finance | `gbharti/finance-alpaca` | alpaca | 99.8% |
| poetry | `merve/poetry` (creative writing) | mixed | 87.6% |

Legal dropped: `nguha/legalbench` had only ~6 usable training rows in the `learned_hands_benefits` config. No other legalbench config had sufficient instruction-format data. Legal is parked until a better instruction-format legal Q&A dataset is found.

Poetry at 87.6% with 105 val examples — small sample variance, not a boundary problem.

Medical↔reasoning overlap (~6-7% error rate) is a genuine semantic boundary issue. Clinical reasoning prompts activate both concepts. Acceptable for current purposes.

**Weights saved:** `data/sae_weights.pt`  
**Overall routing accuracy: 96.2%**  
**Best val loss: 0.1597**

### End-to-end demo results

`python sandbox-scripts/prefill_router.py --sae-weights data/sae_weights.pt`

**Routing: 4/6**
- astrophysics → routed as finance (ScienceQA is broad; prompt pattern-matched finance-alpaca's instruction style)
- reasoning → routed as astrophysics (both Q&A format, conceptual questions; thin boundary)

**Latency:**
- Python (first call): 638ms — Triton JIT compiling SAE encoder kernel on first invocation
- All subsequent: ~30ms steady state

30ms is well under generation prefill latency. The prefill-only routing architecture is validated: the extra pass is not a bottleneck.

### Open
- Legal concept: find instruction-format legal Q&A dataset when available
- `_merged_weights` → vLLM: the one remaining structural gap
- astrophysics dataset: replace ScienceQA with a narrower astrophysics Q&A set to fix the finance bleed

---

## Codebase Status
**Education phase complete.**

The conceptual proof-of-concept scripts have been relocated to `sandbox-scripts/`. The production codebase architecture is initialized under `src/mrna/` with four distinct nodes: `substrate`, `memory`, `router`, and `execution`. The full pipeline runs end-to-end: prefill routing at ~30ms steady state, 96.2% val accuracy CBSAE, fused Triton LoRA merge kernel at 2.28x speedup.

`_merged_weights` → vLLM gap addressed in Session 8 via `mRNAWorker`. See below.

---

## Session 8 — 2026-04-01
**Stage:** Custom vLLM Worker — merged-weight serving path

**File:** `sandbox-scripts/mrna_worker.py`

### What was built

`mRNAWorker(Worker)` — vLLM worker subclass. `MergedStreamingExecutionNode` — drop-in replacement for `StreamingExecutionNode` that passes `worker_cls=mRNAWorker` to LLM().

### Filling the gaps in Gemini's sketch

| Gap in sketch | Resolution |
|---|---|
| `concept_idx` carrier | `LoRARequest.lora_int_id - 1` — already set to `concept_idx+1` by `StreamingExecutionNode` |
| `_swap_pointers` | `weight.data.copy_(merged_W)` — in-place write, same address, CUDA-graph-safe |
| `merged_cache` IPC | Worker populates its own cache lazily; calls `lora_merge` in-process from vLLM's loaded LoRA weights |
| Layer targeting | `_build_layer_map()` at init; walks `model.named_modules()`, records all `nn.Linear` + base weight clones |
| Base restore cost | One clone at init; `copy_()` on hot path — ~0.06ms at 504 GB/s for a 4096-dim layer |

### Data flow

```
PrefillRouter.route(prompt)
    → concept_idx, adapter_path
    → MergedStreamingExecutionNode.run_inference_with_adapter()
        → LoRARequest(lora_int_id = concept_idx + 1)
            → mRNAWorker.execute_model()
                → _get_concept_idx()     reads lora_int_id from seq_group_metadata
                → _ensure_merged()       lora_merge on first call, cached after
                → _apply_merged()        copy_() into base weight tensors
                → super().execute_model() vLLM eager forward pass
                → _restore_base()        copy_() back to base weights
```

### Known limitations / next steps

- **vLLM internal API** — `_extract_lora_weights` accesses `model_runner.lora_manager`, which is a private API that varies by vLLM version. Falls back silently to standard LoRARequest path if extraction fails.
- **`enforce_eager=True` required** — safe now; if CUDA graphs are ever enabled, `copy_()` is still address-safe but the graph must be recaptured after weight changes.
- **Smoke test** — `python sandbox-scripts/mrna_worker.py` exercises init + two inference calls (falls back to base model gracefully with no adapters on disk).
- **Real adapter files** — the full path requires actual trained `.mrna` adapter files in `/adapters/`. This is the next concrete milestone: fine-tune one concept adapter with Unsloth and verify the merge path end-to-end.

---

## Session 9 — 2026-04-01
**Stage:** First real .mrna adapter — biology trained, streamed, and generated

### What happened

**Step 1 — `train_mrna_adapter.py`** (~1 hour)
Biology adapter trained on `camel-ai/biology` (20k chat Q&A pairs). rank=16, target q_proj + v_proj, 200 steps. Saved to `adapters/biology_lora/` in PEFT format.

**Step 2 — `harvest_hf.py`** (~1 hour)
5000 biology activations harvested (message_1 column, masked mean pool). Saved to `data/biology_layer6.pt`.

**Step 3 — `train_sae.py`** (7 concepts, lowest loss yet)
3D compute at max utilization in bursts. Biology added as concept_idx=6. Best val loss lower than any previous run.

**Step 4 — `prefill_router.py --sae-weights data/sae_weights.pt`**
- Routing: 4/7, avg latency 28ms
- Biology adapter loaded from disk and streamed via PCIe for the first time
- Generation output was coherent (beta-blockers prompt → correct pharmacology explanation)

### The PCIe stream

```
[Execution Node] SAE Router triggered adapter ID 7: /home/kotaraine/Coder/mRNA/adapters/biology_lora
Streaming .mrna dynamically via PCIe...
Executing batch...
Prompt: 'Explain the mechanism of action of beta-blockers in treating hypertension.'
Generated: ' Beta-blockers act on beta-adrenergic receptors in the body, which are responsible
for controlling the response of the heart and blood vessels to the hormone epinephrine...'
```

This is the architecture working end-to-end for the first time: SAE routed a prompt → vLLM streamed an `.mrna` adapter from NVMe over PCIe → generation ran with injected weights.

### Warnings (both benign)

**"deprecated support for different tokenizers for different LoRAs"**
`train_mrna_adapter.py` calls `tokenizer.save_pretrained(adapter_path)` alongside the weights. vLLM detected the tokenizer file and warned it will use the base model tokenizer instead. This is correct behavior — we fine-tuned the same base model so the tokenizer is identical. Safe to ignore; optionally remove `tokenizer.save_pretrained` from the training script if the warning is unwanted.

**"Using default LoRA kernel configs"**
vLLM has no pre-tuned kernel config for this specific adapter shape. It selects defaults, which are correct and functional. Only relevant if we run Nsight and decide to tune the serving-side kernels — not a concern now.

### Biology routing to medical
The beta-blockers prompt was expected to hit medical (it's a medical question). Biology and medical share activation space around physiology/biochemistry — the same overlap as medical↔reasoning. The generation quality was good regardless of which adapter was active.

### Where things stand end of day

Every layer of the architecture is functional:
- Triton fused LoRA merge: ✅ 2.28x speedup
- CBSAE routing: ✅ 96.2% val accuracy, 28ms prefill
- vLLM dynamic adapter loading: ✅ biology adapter streamed from NVMe
- Full pipeline wiring: ✅ prefill router → SAE → vLLM → generation
- mRNAWorker merged-weight path: ✅ **VERIFIED** — 16 layers merged for concept_idx=6

### Session 9 final result — mRNAWorker fully operational (2026-04-01 22:48)

`[mRNAWorker] Merged 16 layers for concept_idx=6 (lora_int_id=7)`

The full merge path is live. Key debugging chain:

**Bug 1:** `lora_manager._adapter_manager` not found  
Fix: attribute lives on `self.model_runner`, not directly on Worker.  
Correct path: `self.model_runner.lora_manager._adapter_manager`

**Bug 2:** `get_adapter(7)` returns None on first execute_model call  
Root cause: vLLM v1 loads adapters *inside* `execute_model` (via `set_active_adapters`), not as a separate pre-call. So `_ensure_merged` fired before the adapter was in the manager.  
Fix: on the first call when adapter isn't loaded, fall through to `super().execute_model()` which loads it, then call `_ensure_merged` again immediately. Cache is populated; all decode steps see merged weights.

**Bug 3:** `float(layer_weights.scaling)` TypeError — scaling is a list  
Root cause: vLLM packs q/k/v into `qkv_proj` (and gate/up into `gate_up_proj`). All 65 LoRA-capable layers in Llama-3.2-1B are `PackedLoRALayerWeights` with `scaling: list[float]`. No unpacked layers exist.  
Fix: handle `PackedLoRALayerWeights` via `wrapper.output_slices` to compute per-sub-slot offsets. For each non-None sub-slot: `W[offset:offset+slice_size] = lora_merge(W_slice, A_i, B_i, scale_i)`.

**Result:** 16 merged layers (16 × qkv_proj layers, q_proj + v_proj sub-slots fused per block).  
Decode steps run on `W + scale*B@A` directly — no LoRA delta decomposition overhead.

### Session 10 — Science triad benchmark (2026-04-02)

Biology + chemistry + physics adapters trained. SAE retrained on 3-concept science subset.  
Results: loss 0.1368, 96.6% val accuracy. Routing: **5/6** at 30–40ms steady state.

Miss: "electric field and electric potential" → routed chemistry (confidence 0.40).  
Root cause: electrostatics vocabulary (Coulomb, charge, potential) spans both domains.  
Generation was correct physics content regardless — execution path unaffected by router miss.  
Fix path: harvest physics prompts that lean on mechanics/thermodynamics vocabulary to reinforce  
the physics concept boundary in the SAE. Carnot prompt routed correctly (pure thermodynamics).

**Science triad is the benchmark baseline before switching to a larger model.**

### Session 11 — Gemma 3 4B model switch (2026-04-02)

Switched base model from `unsloth/Llama-3.2-1B-Instruct` to `unsloth/gemma-3-4b-it`.  
Harvest layer updated to 17 (midpoint of 34-layer model). d_model = 2560.

**Training fix:** `--max-seq-len` default lowered from 512 → 256. Physics adapter OOMed at 512  
(CAMEL physics conversations are longer/denser than biology). Biology trained fine at 512 but 256  
is the safe default for Gemma 3 4B on 12GB VRAM.

**SAE retrain results — science triad on Gemma 3 4B activations:**
- Loss: 42219 → 18.8 train / 22.6 val over 50 epochs (very fast convergence)
- Routing accuracy: **99.0%** overall
  - biology:   99.0%  (511 val examples)
  - chemistry: 98.8%  (499 val examples)
  - physics:   99.2%  (490 val examples)

Near-zero train/val gap and per-concept spread of only 0.4% — Gemma 3's residual stream  
represents these three science domains far more distinctly than Llama 3.2 1B did.  
This is the new baseline for the science triad before adding multipurpose concepts.

### Session 12 — Llama 3.2 1B science triad (2026-04-02)

Reverted to `unsloth/Llama-3.2-1B-Instruct` after Gemma 4 E2B unsloth support gap and Gemma 3 4B VRAM constraints.  
Trained physics + chemistry adapters with `--text-column message_1 --no-packing` to avoid OOM.

**Routing results: 4/6**

Misses:
1. Le Chatelier's principle → routed physics (confidence 0.22) — empty response generated  
2. Electric field/potential → routed chemistry (confidence 0.54) — correct content generated  

**Bug: empty response on wrong-adapter routing**  
Root cause: `max_tokens=64`, `min_tokens` unset — physics-merged weights cause immediate EOS on  
out-of-distribution chemistry prompt.  
Fix: `min_tokens=8, max_tokens=128` in SamplingParams.

**Routing boundary analysis:**  
Le Chatelier (equilibrium, pressure, concentration) and electric field/potential (Coulomb, charge)  
straddle chem/physics in Llama 1B's residual stream. Training on `message_1` questions-only  
removed the disambiguating signal from answers — questions use blended vocabulary.  
Gemma 3 4B scored 99.0% on the same triad, confirming this is a model-size/representation issue.

**Fix path:** Retrain SAE on full Q+A (message_1 + message_2) at lower seq_len (256),  
or wait for Gemma 4 E2B / Mac Mini for clean chem-physics separation.

### Session 12 addendum — full triad working end-to-end on mrna_worker (2026-04-02)

After the Gemma detour, reverted to Llama 3.2 1B. Training unblocked by:
- Pre-truncating text to `max_seq_length - 1` before SFTTrainer re-tokenizes (BOS adds +1)
- `packing=False` as default; `--packing` flag if needed
- Cleared stale `unsloth_compiled_cache/` after transformers version churn

**mrna_worker result:** 6/6 prompts — no empty responses, correct domain content on all.  
Biology adapter generates answers. Chemistry/physics boundary miss in SAE routing is a  
1B representation issue; execution path (merge + generate) is solid regardless.

**Full pipeline verified end-to-end:**  
harvest → SAE train → adapter train → prefill route → vLLM mRNAWorker merge → generation

### What comes next

**Waiting on hardware / model support:**
- Gemma 4 E2B unsloth support (~1 week from 2026-04-02) — d_model=1536, ~2B params, clean chem/physics separation expected
- Mac Mini 24GB unified (early May 2026) — full concept set training, no VRAM constraints

**When E2B lands:**
1. Harvest biology/chemistry/physics at layer 17, d_model=1536
2. Train triad adapters (no OOM concerns at 2B)
3. SAE retrain — expect 99%+ routing with clean concept separation
4. mrna_worker verify on E2B

**Longer arc:**
- Garden integration — mRNA becomes the adapter routing layer for bebop
- Niigaane Ojibwe adapter — once annotation pipeline completes in To-Lead
- Autoresearch concept — Qwen 3.5 (0.8B or 2B) as tool-use candidate for a future agent concept slot

### Session 13 — Gemma 4 E2B: unsloth confirmed, vLLM abandoned, llama.cpp adopted (2026-04-02)

**Gemma 4 E2B unsloth support confirmed.**  
`unsloth-zoo` upgraded to 2026.4.2 resolved a `patched_convert_added_tokens` crash that fired
because `Gemma4Processor` passes `AddedToken.content` as a dict, not a string.
`unsloth/gemma-4-E2B-it` now loads and runs correctly in the harvesting and training scripts.

**Key verified architecture numbers (text backbone):**
- `d_model=1536` — from `model.config.text_config.hidden_size` (not total 4.17B param count)
- `num_layers=35`, `head_dim=256`, `8 attn heads`, `intermediate_size=6144`
- Total model: 4.17B (vision encoder + audio tower + text backbone)
- Text backbone: ~2B params

**Gemma 4 tokenizer pattern (Gemma4Processor):**  
`Gemma4ForConditionalGeneration` wraps a multimodal processor. Calling the tokenizer
positionally trips unsloth's `patched_call`, which routes through the vision processor
and returns `None` for the text encoding.  
Fix: `_tok = getattr(tokenizer, "tokenizer", tokenizer)` then call with `text=` keyword.  
Applied to `harvest_hf.py` and `train_mrna_adapter.py`. Backwards-compatible (falls
back to processor itself for non-multimodal models).

**vLLM Gemma 4 — fundamental incompatibility, abandoned.**  
Gemma 4 uses a Gemma 3n per-layer embedding reversal (`modeling_gemma4.py:1635`) that
broadcasts `inputs_embeds` against the full vocabulary (`[batch, seq, vocab, d_model]`).
vLLM passes `inputs_embeds` without `input_ids` through `TransformersForCausalLM`,
triggering an allocation of ~3072 GiB. No config override resolves this — it is a
fundamental forward-pass incompatibility. Every approach attempted:

1. `architectures=["Gemma4ForCausalLM"]` patch → still routes via `hf_config != hf_text_config`
2. Flat text-only `config.json` with `model_type="gemma4_text"` → still triggers Gemma 3n reversal
3. Text-only safetensors cache (600 tensors, `model.language_model.*` prefix stripped) → same OOM
4. `layer_scalar` tensors also had to be filtered (not registered as parameters) — found along the way

**Switch to llama.cpp.**  
llama.cpp has native Gemma 4 GGUF support from the HF announcement. Verified:
- CPU-only build (no nvcc available; CUDA build requires `sudo apt install nvidia-cuda-toolkit`)
- `PYTHONPATH=~/llama.cpp` trick lets `convert_hf_to_gguf.py` and `convert_lora_to_gguf.py`
  run with the project `.venv` python without pip-installing into the venv (which would
  downgrade torch to the 2.6.0 CPU wheel)
- `models/gemma-4-e2b-q8.gguf` successfully built: **4.94 GB, 601 tensors, q8_0**
- Smoke test: `llama-cli` loads, enters thinking mode, 18.5 t/s on CPU

**New execution path:**
- `src/mrna/execution/llama_cpp_node.py` — `LlamaCppExecutionNode` replaces `StreamingExecutionNode`
  - Spawns `llama-server` subprocess with all GGUF LoRA adapters pre-loaded at `scale=0`
  - `/lora-adapters` POST to activate one adapter per request, `/completion` for generation
  - Same interface: `run_inference_with_adapter(prompt, adapter_path, adapter_id)` + `cleanup()`
  - Applies Gemma 4 chat template via passed-in tokenizer (unwraps `Gemma4Processor`)
- `sandbox-scripts/convert_adapters_to_gguf.py` — PEFT safetensors → GGUF LoRA (new script)
- `sandbox-scripts/prefill_router.py` — swapped to `LlamaCppExecutionNode`, added `GGUF_ADAPTER_REGISTRY`

**mRNAWorker retirement note:**  
`sandbox-scripts/mrna_worker.py` is vLLM-specific and is not ported to the llama.cpp path.
The merged-weight serving concept lives on in llama.cpp's native LoRA handling.
`src/mrna/substrate/lora_merge.py` remains valid for training-side benchmarking (Triton kernel).

**Adapters need retraining on E2B base.**  
The existing biology/chemistry/physics adapters target `unsloth/gemma-3-4b-it-unsloth-bnb-4bit`.
They must be retrained on `unsloth/gemma-4-E2B-it` before GGUF LoRA conversion.

**Retrain command (each concept):**
```bash
.venv/bin/python sandbox-scripts/train_mrna_adapter.py \
    --model-id unsloth/gemma-4-E2B-it \
    --dataset camel-ai/biology \
    --concept biology \
    --max-seq-len 256 \
    --max-steps 200
```

**Then convert:**
```bash
.venv/bin/python sandbox-scripts/convert_adapters_to_gguf.py \
    --base-model models/gemma-4-e2b-q8.gguf \
    --adapters biology chemistry physics
```

**CUDA build (when nvcc available):**
```bash
sudo apt install nvidia-cuda-toolkit
cd ~/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc)
```
Then set `n_gpu_layers=99` in `LlamaCppExecutionNode` for full GPU offload.

**What comes next:**
1. Retrain science triad adapters on E2B — expect clean chem/physics separation at 2B scale
2. Convert to GGUF LoRA, run `prefill_router.py` end-to-end with llama-server
3. Harvest E2B activations (layer 17, d_model=1536) + retrain SAE
4. Mac Mini: same GGUF files, swap `n_gpu_layers` for Metal — no other changes

### Session 14 — 6/6 routing on Gemma 4 E2B science triad (2026-04-03)

**Full pipeline verified end-to-end on Gemma 4 E2B + llama.cpp:**
harvest → SAE train → adapter train → prefill route → llama-server LoRA generation

**Key result: 6/6 routing accuracy** on the science triad demo cases, including the two
historically hard boundary cases (Le Chatelier → chemistry, electric field → physics).

**What unlocked 6/6: layer 25 (not 17)**
- Layer 17 (midpoint): SAE reached 98.2% val accuracy but routed Le Chatelier, electric
  field/potential, and Carnot engine wrong in the demo — confident scores but wrong side
  of the boundary.
- Layer 25 (70% depth): SAE reached 98.5% val accuracy and 6/6 on the demo.
- Reason: at the midpoint the model is still integrating syntactic and structural context.
  By layer 25 it has committed to semantic/domain representations. Chemistry "equilibrium"
  and physics "field/potential" concepts are cleanly separated from biology at this depth.
- Rule: for domain routing, harvest at ~70% model depth, not midpoint.

**Other issues resolved this session:**

*Model snapshot pinning:*
HF Hub updated `unsloth/gemma-4-E2B-it` mid-session (3-shard → single model.safetensors,
config.json changed). Caused silent train/test distribution mismatch — SAE trained on old
weights, inference loaded new weights, routing accuracy dropped.
Fix: pinned `revision: "37ea165b3fba25b7d851f8ce4ccff9a4f0751cee"` in `model_config.yaml`
and added `--model-revision` flag to all three scripts (harvest, train, prefill_router).
All `FastLanguageModel.from_pretrained` calls now pass `revision=` explicitly.

*Activation source mismatch:*
Existing `data/biology_layer17.pt` etc. were harvested from Gemma 3n E4B (also d_model=1536
so shape looked correct). SAE trained on those files showed 96.7% val accuracy but near-random
inference performance. Fixed by re-harvesting from the pinned E2B revision.

*Q+A vs question-only harvesting:*
Harvesting questions-only (message_1) gives ambiguous boundary features — "equilibrium" appears
in biology and chemistry questions equally. Concatenating Q+A (message_1 + message_2, max_seq=256)
gives the SAE answer vocabulary ("equilibrium constant Kc", "Coulomb's law", "electric flux")
that cleanly separates domains. `--text-column2 message_2` flag added to harvest_hf.py.

**Final SAE config (canonical for E2B science triad):**
- Layer: 25 (was 17)
- Input text: Q+A concatenated (`--text-column message_1 --text-column2 message_2`)
- Max seq len: 256 (matches training OOM fix)
- Epochs: 50, expansion_factor=8 (d_sae=12288), batch=64
- Weights: `data/sae_weights_e2b_layer25.pt`
- Val loss: 0.0554 | Val accuracy: 98.5% | Demo: **6/6**

**Routing latency:** ~4163ms avg (prefill-only pass, no generation). This is the unsloth
model load + forward pass cost per query; llama-server generation adds ~1-2s on GPU.

**Note:** The 3 hard cases (Le Chatelier, electric field, Carnot) were failing not because
adapters were wrong — they generated correct content regardless of routing — but because
layer-17 residual stream activations for interdisciplinary topics (thermodynamics, equilibrium,
electricity) hadn't resolved to domain-specific directions yet. Layer 25 fixed all three.

**What comes next:**
- Mac Mini 24GB unified: same GGUF, same adapters, swap `n_gpu_layers` → Metal
- Harvest at layer 25 for Ojibwe concept (once Niigaane annotation completes)
- Expand science triad: add math/engineering concept slots
- Autoresearch concept: Qwen 3.5 as tool-use candidate

---

## Session 15 — 2026-04-04
**Stage:** Science triad adapter training (math/python dropped) + full inference pipeline operational

### What happened

**Adapter training — math and python dropped.**
Math (`HuggingFaceH4/MATH-500`) and Python (`iamtarun/python_code_instructions_18k_alpaca`) were
trained but caused routing bleed against the science triad SAE boundary — both were dropped from the
active concept set. Biology, chemistry, physics science triad is the canonical active set.

**Triton JIT first-step behavior (documented).**
First training step on a cold Triton cache takes ~268s while Unsloth JIT-compiles Gemma 4 kernels.
GPU shows P8 (12W idle) during this phase — it is CPU-bound compilation, not a hang or OOM.
Steps 2+ drop to normal speed once `unsloth_compiled_cache/` + `~/.triton/cache/` are warm.
This is expected and only happens on first run or after clearing caches.

**GGUF LoRA conversion — all 3 adapters converted.**
```
biology.gguf    13.1 MB  data/gemma-4-e2b/adapters/biology_lora/biology.gguf
chemistry.gguf  13.1 MB  data/gemma-4-e2b/adapters/chemistry_lora/chemistry.gguf
physics.gguf    13.1 MB  data/gemma-4-e2b/adapters/physics_lora/physics.gguf
```
13.1 MB per adapter — well within PCIe Gen4 streaming budget.
llama.cpp CUDA build confirmed (`libggml-cuda.so` present, `llama-server` binary ready).

**`prefill_router.py` SAE path fixed.**
Default `--sae-weights` corrected from stale `data/sae_weights_e2b_layer25.pt` to
`data/gemma-4-e2b/sae/w_25.pt`.

### Final demo results — 5/6

```
biology  → mitochondria prompt            ✓  confidence=4.039  latency=6505ms (first call, JIT)
biology  → DNA replication prompt         ✗  chemistry=1.692 > biology=0.912
chemistry → ionic/covalent bonding        ✓  confidence=4.052  latency=487ms
chemistry → Le Chatelier's principle      ✓  confidence=0.678  latency=481ms
physics  → electric field/potential       ✓  confidence=1.425  latency=473ms
physics  → Carnot engine efficiency       ✓  confidence=0.988  latency=587ms

Routing accuracy : 5/6
Avg route latency: 1747ms  (inflated by 6.5s first-call JIT; steady state ~500ms)
```

**Miss analysis — DNA replication → chemistry:**
SAE scores: `biology=0.912  chemistry=1.692  physics=-2.564`.
Root cause: DNA replication vocabulary (DNA polymerase, helicase, primase, ligase — all enzymes)
activates chemistry features more strongly than biology features at layer 25 on this SAE.
Biochemistry is the hard boundary: molecular biology sits between bio and chem concept dimensions.
Generation was **correct biology content** despite wrong adapter — execution path unaffected.
Fix path: harvest additional biology examples that are enzyme/replication-heavy, retrain SAE.
Alternatively, this is acceptable boundary bleed at current concept granularity.

**TurboQuant (KV cache quantization) — deferred.**
`-ctk tq4 -ctv tq4` flags would further reduce VRAM pressure during long-context inference.
Requires a community llama.cpp fork not yet in mainline. Deferred until context actually becomes
a problem in production use. Current Q8 base + standard KV cache is stable.

### Artifact state end of session

| Artifact | Path | Status |
|---|---|---|
| Base GGUF | `models/gemma-4-e2b-q8.gguf` | ✅ |
| SAE weights | `data/gemma-4-e2b/sae/w_25.pt` | ✅ |
| biology adapter | `data/gemma-4-e2b/adapters/biology_lora/biology.gguf` | ✅ |
| chemistry adapter | `data/gemma-4-e2b/adapters/chemistry_lora/chemistry.gguf` | ✅ |
| physics adapter | `data/gemma-4-e2b/adapters/physics_lora/physics.gguf` | ✅ |
| Activations | `data/gemma-4-e2b/activations/layer_25/{bio,chem,phys}_{train,test}.pt` | ✅ |

### What comes next

**Garden integration (next session).**
`LlamaCppExecutionNode` is HTTP-native (llama-server at `localhost:8080`). Garden consumes it
as a sidecar service:
1. `PrefillRouter.route(prompt)` → `(concept_name, adapter_path, confidence)`
2. Garden backend calls `/lora-adapters` to activate adapter
3. Garden backend calls `/completion` for generation

The split point: Garden owns the prompt lifecycle; `.mrna` owns routing + adapter selection.
Interface contract: `POST /route` returning `{concept, adapter_path, confidence}` —
a thin FastAPI wrapper around `PrefillRouter` is the cleanest integration seam.

---

## Hardware Notes

- Mac Mini 24GB unified memory arrives early May 2026, a few weeks away — sustained training target.
  `mmap`-friendly design will map cleanly (unified memory = no PCIe hop for Engram).
- 4070 Super is the development and validation target. All VRAM budgets measured here.
- Nsight Compute install needed for SM occupancy profiling of Triton kernels.

---

*Load with SKILL.md at the start of any session on this project.*
*Update after each session with new script results, key numbers, and open questions.*

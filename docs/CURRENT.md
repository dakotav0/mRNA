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

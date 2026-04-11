## Session 16 — 2026-04-10
**Stage:** Dual-Routing Strategy Architecture Finalized & Validated (Qwen 0.8B target)

### What happened

**Gap Recovery: Resolving Catastrophic Routing Failures.**
The transition from single-layer harvesting to dual-layer was initially met with "catastrophic" indexing failures. This was traced to a singular `harvest_layer` config attribute being used sequentially rather than a plural-aware registry. Standardized `ConfigManager` to use `get_logic_layer()` and `get_voice_layer()` to decouple the pipeline.

**The "Double-Prefill" Architecture.**
Identified a critical constraint in `llama.cpp` integration: the `/lora-adapters` endpoint applies swaps at the *next* token, not retroactively on the current prefill. 
- Implemented a **Double-Pass Routing** logic in `api.py`.
- **Pass 1**: Detects domain logic (Biology/Physics/Chemistry) via early-layer SAE (Layer 6 on Qwen, Layer 14 on Gemma 4).
- **Pass 2 (Optional)**: If a voice persona is required, a second deliberate prefill is executed with the Logic adapter active to capture the high-level voice features (Layer 12 on Qwen, Layer 25 on Gemma 4).

**Qwen 3.5 0.8B Logic Validation.**
Successfully trained and evaluated the science triad on the ultra-light Qwen backbone. 
- **Layer 6 (Logic)**: 92.1% accuracy.
- **Layer 12 (Voice)**: 89.2% accuracy.
- Physics remains the hardest boundary to define, showing the most "bleed" into chemistry.

### Final Accuracy Results (Qwen 3.5 0.8B)

```
layer_6 (Logic):
  biology   → 95.5% accuracy
  chemistry → 95.1% accuracy
  physics   → 86.1% accuracy (bleed into chemistry)
  OVERALL   : 92.1%

layer_12 (Voice/Refined):
  biology   → 97.8% accuracy
  chemistry → 89.1% accuracy
  physics   → 81.1% accuracy
  OVERALL   : 89.2%
```

### Artifact state end of session

| Artifact | Path | Status |
|---|---|---|
| Qwen 0.8B Logic SAE | `data/qwen3.5-0.8b/sae_weights_L6.pt` | ✅ |
| Qwen 0.8B Voice SAE | `data/qwen3.5-0.8b/sae_weights_L12.pt` | ✅ |
| Qwen 0.8B Physics LoRA| `data/qwen3.5-0.8b/adapters/physics_lora/` | ✅ |
| Config Manager | `src/mrna/core/config.py` (pluralized) | ✅ |
| API Interceptor | `src/mrna/router/interceptor.py` (multi-layer) | ✅ |

### What comes next

**Gemma 4 E2B Validation.**
Now that the architecture is stable, the next priority is validating the science triad on the larger Gemma 4 backbone (Layers 14 & 25).
- Retrain SAEs on the E2B base under the new pluralized config.
- Benchmark the "Double-Prefill" latency on the RTX 4070 SUPER (<500ms target).

**S&box Project Handover.**
Ensure the MIIN router can consume the new `mrna_route` and `lora_stack` events for real-time NPC personality adjustment.

---

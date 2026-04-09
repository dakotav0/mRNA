"""
Fused LoRA merge kernel: C = W + scale * B @ A

Execution Node — Priority 5 from synthesis-log.md

The math
--------
Given:
  W : (M, N)  — frozen base weight (FP16)
  A : (R, N)  — LoRA A matrix, down-projection (FP16)
  B : (M, R)  — LoRA B matrix, up-projection (FP16)
  scale       — float = alpha / rank

Naive PyTorch:
  delta = B @ A            # kernel 1: write (M, N) intermediate to HBM
  C     = W + scale*delta  # kernel 2: read W, read delta, write C

Fused Triton:
  Single kernel pass. Each thread block loads a (BLOCK_M, BLOCK_N) tile of W
  into FP32 accumulator, then iterates the rank dimension R in BLOCK_R chunks,
  loading B and A tiles into SRAM and accumulating. Writes only C — the (M, N)
  intermediate is never allocated.

HBM trip savings vs naive baseline
  Reads:   W + A + B          (same in both)
  Writes:  C only             (naive also writes delta, then reads it back)
  Saved:   1 full (M,N) write + 1 full (M,N) read
  For M=N=4096 FP16: ~64MB of HBM traffic eliminated per merge event.

Target: match LoRA-Switch's 2.4x vs three-nn.Linear baseline (arXiv:2405.17741).
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_R": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_R": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_R": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_R": 16}, num_warps=8),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_R": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_R": 32}, num_warps=4),
    ],
    key=["M", "N", "R"],
)
@triton.jit
def lora_merge_kernel(
    w_ptr, a_ptr, b_ptr, c_ptr,
    M, N, R,
    scale,
    stride_wm, stride_wn,
    stride_ar, stride_an,
    stride_bm, stride_br,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """
    Each thread block owns a (BLOCK_M, BLOCK_N) output tile.
    Grid: (ceil(M/BLOCK_M), ceil(N/BLOCK_N)) — 2D.

    Memory layout (same K-loop pattern as triton_matmul.py, but accumulator
    is seeded with W tile values rather than zeros):

        for r_chunk in range(0, R, BLOCK_R):
            load B tile : (BLOCK_M, BLOCK_R)  <- SRAM
            load A tile : (BLOCK_R, BLOCK_N)  <- SRAM
            accumulator += tl.dot(B_tile, A_tile)
            advance pointers

        C tile = (accumulator * scale) + W_tile
        store C tile -> HBM
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row / col indices for this tile
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rr = tl.arange(0, BLOCK_R)

    # Bounds masks
    m_mask = rm < M
    n_mask = rn < N

    # ------------------------------------------------------------------
    # 1. Load W tile into FP32 working buffer
    # ------------------------------------------------------------------
    w_ptrs = w_ptr + rm[:, None] * stride_wm + rn[None, :] * stride_wn
    w_mask = m_mask[:, None] & n_mask[None, :]
    w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

    # ------------------------------------------------------------------
    # 2. Accumulate (scale * B @ A) tile-by-tile across rank dimension
    # ------------------------------------------------------------------
    # Initialise B and A pointers at rank-chunk 0 for this output tile
    b_ptrs = b_ptr + rm[:, None] * stride_bm + rr[None, :] * stride_br
    a_ptrs = a_ptr + rr[:, None] * stride_ar + rn[None, :] * stride_an

    lora_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for r in range(0, tl.cdiv(R, BLOCK_R)):
        r_offset = r * BLOCK_R + rr
        b_mask = m_mask[:, None] & (r_offset[None, :] < R)
        a_mask = (r_offset[:, None] < R) & n_mask[None, :]

        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)  # (BLOCK_M, BLOCK_R)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)  # (BLOCK_R, BLOCK_N)

        # tl.dot requires both inputs in the same type; FP16 inputs → FP32 accumulation
        lora_acc += tl.dot(b_tile, a_tile)

        b_ptrs += BLOCK_R * stride_br
        a_ptrs += BLOCK_R * stride_ar

    # ------------------------------------------------------------------
    # 3. Fuse the add: C = W + scale * (B @ A)
    # ------------------------------------------------------------------
    c_tile = (w_tile + scale * lora_acc).to(tl.float16)

    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, c_tile, mask=w_mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def lora_merge(W: torch.Tensor, A: torch.Tensor, B: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Fused weight merge: C = W + scale * B @ A

    Args
    ----
    W     : (M, N) FP16, base weight matrix (frozen)
    A     : (R, N) FP16, LoRA down-projection
    B     : (M, R) FP16, LoRA up-projection
    scale : float, typically alpha / rank

    Returns
    -------
    C : (M, N) FP16 — merged weight; W is not modified in place.

    Shape contract
    --------------
    W.shape[0] == B.shape[0]  (M — out_features)
    W.shape[1] == A.shape[1]  (N — in_features)
    A.shape[0] == B.shape[1]  (R — LoRA rank)
    """
    assert W.is_contiguous() and A.is_contiguous() and B.is_contiguous(), \
        "All input tensors must be contiguous. Call .contiguous() if needed."
    assert W.dtype == torch.float16 and A.dtype == torch.float16 and B.dtype == torch.float16, \
        "Kernel expects FP16 inputs."

    M, N = W.shape
    R, _ = A.shape

    assert B.shape == (M, R), f"B shape mismatch: expected ({M}, {R}), got {B.shape}"
    assert A.shape[1] == N,   f"A shape mismatch: expected (R, {N}), got {A.shape}"

    C = torch.empty_like(W)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    lora_merge_kernel[grid](
        W, A, B, C,
        M, N, R,
        scale,
        W.stride(0), W.stride(1),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _baseline_naive(W, A, B, scale):
    """Three-operation naive: two separate kernel launches, one intermediate."""
    delta = B @ A
    return W + scale * delta


def _baseline_fused_torch(W, A, B, scale):
    """torch.compile baseline — PyTorch's own fusion pass."""
    delta = B @ A
    return W + scale * delta


def run_benchmark(M=4096, N=4096, R=16, scale=0.5):
    """
    Benchmark the fused Triton kernel against PyTorch naive and torch.compile.

    Typical .mrna target: M=N=4096 (hidden dim of a 7B-class model), R=16.
    A merge event of this size moves ~32MB of FP16 data in the naive path.
    """
    W = torch.randn(M, N, device="cuda", dtype=torch.float16).contiguous()
    A = torch.randn(R, N, device="cuda", dtype=torch.float16).contiguous()
    B = torch.randn(M, R, device="cuda", dtype=torch.float16).contiguous()

    # --- Correctness check first ---
    ref = _baseline_naive(W, A, B, scale)
    out = lora_merge(W, A, B, scale)
    max_diff = (ref.float() - out.float()).abs().max().item()
    rel_err  = max_diff / ref.float().abs().max().item()
    print(f"\nCorrectness check (M={M}, N={N}, R={R})")
    print(f"  Max abs diff : {max_diff:.6f}")
    print(f"  Relative err : {rel_err:.4%}")
    assert rel_err < 0.01, f"Relative error {rel_err:.4%} exceeds 1% tolerance"
    print("  PASS")

    compiled_baseline = torch.compile(_baseline_fused_torch)
    # Warm up the compile
    _ = compiled_baseline(W, A, B, scale)

    # --- Timing ---
    t_naive   = triton.testing.do_bench(lambda: _baseline_naive(W, A, B, scale))
    t_compile = triton.testing.do_bench(lambda: compiled_baseline(W, A, B, scale))
    t_triton  = triton.testing.do_bench(lambda: lora_merge(W, A, B, scale))

    print(f"\nBenchmark (M={M}, N={N}, R={R})")
    print(f"  PyTorch naive    : {t_naive:.4f} ms")
    print(f"  torch.compile    : {t_compile:.4f} ms  ({t_naive/t_compile:.2f}x vs naive)")
    print(f"  Triton fused     : {t_triton:.4f} ms  ({t_naive/t_triton:.2f}x vs naive)")

    target = 2.4
    achieved = t_naive / t_triton
    status = "✅ MEETS TARGET" if achieved >= target else f"⚠️  below {target}x target"
    print(f"\n  LoRA-Switch target (2.4x): {status} (achieved {achieved:.2f}x)")

    # Bandwidth analysis
    # Reads: W (M*N) + A (R*N) + B (M*R)  |  Write: C (M*N)
    # Naive also writes delta (M*N) and reads it back — +2 * M*N transfers
    bytes_fused = (3 * M * N + M * R + R * N) * 2  # FP16 = 2 bytes, approx
    bytes_naive = bytes_fused + 2 * M * N * 2
    bw_peak_gb  = 504.2  # RTX 4070 Super HBM bandwidth (GB/s)
    bw_fused_gb = (bytes_fused / 1e9) / (t_triton / 1e3)
    print(f"\n  Effective bandwidth (Triton): {bw_fused_gb:.1f} GB/s  "
          f"(peak: {bw_peak_gb} GB/s, {bw_fused_gb/bw_peak_gb:.1%} utilization)")
    print(f"  HBM bytes saved vs naive   : {(bytes_naive - bytes_fused)/1e6:.1f} MB "
          f"per merge event")


if __name__ == "__main__":
    print("=== .mrna Fused LoRA Merge Kernel ===")
    print("Substrate Node — Priority 5\n")

    # Standard .mrna LoRA dimensions (rank-16, 4096-dim hidden)
    run_benchmark(M=4096, N=4096, R=16, scale=0.5)

    # Smaller model (1B-class): 2048-dim
    print()
    run_benchmark(M=2048, N=2048, R=16, scale=0.5)

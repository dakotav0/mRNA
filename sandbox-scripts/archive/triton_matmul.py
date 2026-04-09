import torch
import triton
import triton.language as tl

# Step 2: Triton Tiled Matrix Multiply
# This is the foundational kernel for understaning how GPUs perform compute.
# We map a grid of blocks over our output matrix C, and each thread block
# computes a tile of C by iteratively loading tiles from A and B into SRAM.


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # Stride variables (how much the pointer moves to get to the next row)
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters for block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):  # ^ Looks like a complex version of my early perceptron model CS assignment
    # 1. Map to Thread Block
    # Each thread block computes a [BLOCK_SIZE_M, BLOCK_SIZE_N] tile of the output C
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Grid is 1D, we calculate the 2D block coordinates (pid_m, pid_n)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    # 2. Compute Offset Arrays
    # Arrays (vectors) of row indices for A and col indices for B
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # 3. Memory Pointers setup
    # A tile points to row `rm` of A, and advances through K dimension
    # B tile points to col `rn` of B, and advances through K dimension
    a_ptrs = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # 4. Initialize Accumulator SRAM
    # accumulator stores our FP32 sum for this tile of C
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 5. The K-Loop (The "Tiled" part of Tiled Matmul)
    # We iterate across the K dimension of A and B, loading a block into SRAM,
    # doing dense dot products, and accumulating.
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Calculate mask to not out-of-bounds read
        a_mask = (rm[:, None] < M) & ((k * BLOCK_SIZE_K + rk)[None, :] < K)
        b_mask = ((k * BLOCK_SIZE_K + rk)[:, None] < K) & (rn[None, :] < N)

        # Load blocks into SRAM
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Do the dot product and accumulate
        accumulator += tl.dot(a, b)

        # Advance pointers to next block in K dimension
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 6. Write to Global Memory (C)
    # Cast accumulator back to the same dtype as A and B
    c = accumulator.to(tl.float16)

    # Offset array for C matrix
    c_ptrs = c_ptr + stride_cm * rm[:, None] + stride_cn * rn[None, :]
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    # Setup constraints
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"

    M, K = a.shape
    K, N = b.shape

    # Allocate output C on GPU
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # Grid definition: How many blocks we are launching
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    # Launch Kernel
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
    )
    return c


# --- Test against PyTorch ---
torch.manual_seed(0)
A = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
B = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)

print("Running PyTorch native matmul...")
trch_out = torch.matmul(A, B)

print("Running Custom Triton kernel matmul...")
trtn_out = matmul(A, B)

# Check precision diff
max_diff = torch.max(torch.abs(trch_out - trtn_out))
print(f"Maximum difference between Torch and Triton: {max_diff}")

# 1024 accumulates of fp16 random normal numbers will have some rounding variance.
# Since the sum is over 1024 elements with roughly standard stddev, the actual values in the matrix
# are around ~32.0. An absolute error of ~0.125 is actually less than 0.5% relative error, which is great for FP16!
if torch.allclose(trch_out, trtn_out, atol=0.5, rtol=1e-2):
    print("✅ Triton Matmul is Correct! (within fp16 tolerance)")
else:
    print("❌ Output differs too much.")

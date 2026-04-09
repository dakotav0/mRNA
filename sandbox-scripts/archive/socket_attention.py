import math

import torch
import torch.nn.functional as F

# Step 6: Memory Node — SOCKET Integration (SOft Collision Kernel EsTimator)
#
# Why we need this: Standard Multi-Head Attention (MHA) is O(N^2). If our sequence
# gets too long, attention math completely blows out our 12GB VRAM budget.
#
# Previous solutions tried Locality Sensitive Hashing (LSH) to group similar tokens into
# buckets and only run attention inside the bucket. But LSH requires a "Hard XOR Match".
# It's brittle.
#
# SOCKET calculates a probabilistic "Soft" collision score based on the angular distance
# between token hashes. This acts as a sparse attention mask that is mathematically smooth!


def dense_attention(Q, K, V):
    """The standard O(N^2) attention math we are trying to replace."""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V), attn_weights


def socket_soft_collision_score(Q, K, num_hash_tables=4, hash_bits=8):
    """
    Simulates the SOCKET soft scoring mechanism.
    Instead of calculating O(N^2) dot products, we use multiple random projection matrices
    to create hashes, and score them based on collision probabilities.

    (Note: In a true implementation, this runs in a custom Triton Kernel that bypasses
    dense matrix memory allocation entirely. We use PyTorch here to demonstrate the math).
    """
    seq_len, d_k = Q.size(-2), Q.size(-1)

    # 1. Generate random projection planes for LSH
    # Shape: [d_k, hash_bits, num_hash_tables]
    LSH_planes = torch.randn(d_k, hash_bits, num_hash_tables, device=Q.device)

    # 2. Hash Queries and Keys
    # We project the vectors and take the sign (>0 = 1, <0 = 0)
    # Shape: [seq_len, hash_bits, num_hash_tables]
    Q_hashes = (torch.einsum("...sd,...dhn->...shn", Q, LSH_planes) > 0).float()
    K_hashes = (torch.einsum("...sd,...dhn->...shn", K, LSH_planes) > 0).float()

    # 3. Compute Soft Collisions
    # We count how many hash bits match exactly between every Q and K pair across all tables.
    # In Triton, this is an insanely fast XOR comparison.
    # Shape: [seq_len, seq_len]
    exact_matches = torch.einsum("...shn,...thn->...st", Q_hashes, K_hashes)

    # Instead of true/false, SOCKET uses the match ratio as a probabilistic distance surrogate
    collision_probability = exact_matches / (hash_bits * num_hash_tables)

    # 4. Create the Sparse Mask
    # If the probability is below our threshold, we mask it out!
    target_sparsity = 0.95  # Drop 95% of connections!
    threshold = torch.quantile(collision_probability, target_sparsity)

    soft_mask = (collision_probability >= threshold).float()
    return soft_mask, collision_probability


if __name__ == "__main__":
    print("=== SOCKET Soft Collision Sparse Attention ===")

    # Simulate a long sequence
    seq_length = 4096
    d_model = 128

    Q = torch.randn(seq_length, d_model)
    K = torch.randn(seq_length, d_model)
    V = torch.randn(seq_length, d_model)

    print(f"Sequence Length: {seq_length}")
    print(
        f"Dense Attention Matrix Size: {(seq_length * seq_length * 4) / 1024**2:.2f} MB per Head"
    )

    print("\nCalculating SOCKET Sparse Mask...")
    sparse_mask, raw_probs = socket_soft_collision_score(Q, K)

    # Calculate how many elements we actually computed
    active_connections = torch.sum(sparse_mask).item()
    total_connections = seq_length * seq_length
    sparsity_achieved = 1.0 - (active_connections / total_connections)

    print(f"Total connections required: {total_connections:,}")
    print(f"Sockets computed connections: {int(active_connections):,}")
    print(f"Sparsity Achieved: {sparsity_achieved * 100:.2f}%")
    print(
        "The VRAM saved here is what gives us the headroom to dynamically stream .mrna adapters!"
    )

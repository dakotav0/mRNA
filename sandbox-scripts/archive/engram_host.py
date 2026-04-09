import torch
import hashlib

class EngramHostTable:
    """
    Step 3: Memory Node - CPU-side Engram Hash Table
    
    This class decouples factual knowledge from the Transformer's weights.
    Instead of burning GPU VRAM on memorizing exact facts (like "Paris is the capital of France"),
    we store factual embeddings in cheap CPU System RAM.
    
    When an n-gram (e.g., "capital of France") is encountered, we hash it mathematically.
    Because the hash gives us an exact, deterministic memory address, we don't need to do 
    any attention or neural net math to find it. We just grab it from RAM and send it to the GPU
    asynchronously so it arrives exactly when the later transformer layers need it!
    """
    def __init__(self, table_size=1000000, embedding_dim=2048):
        self.table_size = table_size
        self.embedding_dim = embedding_dim
        
        print(f"Allocating {table_size} Engram entries of size {embedding_dim} in CPU RAM...")
        
        # 1. PINNED MEMORY
        # We allocate this massive tensor on the CPU, but we 'pin' it.
        # Pinned memory (page-locked) allows the GPU to use PCIe Direct Memory Access (DMA)
        # to pull the data directly without the CPU getting in the way.
        self.memory_bank = torch.zeros(
            (table_size, embedding_dim), 
            dtype=torch.float16
        ).pin_memory()
        
        # In a real system, this is populated with trained factual embeddings.
        # We'll seed a few exact addresses with random "facts" for our test.
        print("Engram table initialized in Pinned CPU Memory!")
        print(f"RAM Cost: {self.memory_bank.element_size() * self.memory_bank.nelement() / (1024**3):.2f} GB")

    def hash_ngram(self, ngram_tokens: list[int]) -> int:
        """
        Deterministic O(1) mathematical lookup.
        We take the tokens, hash them, and modulo by table size to get the exact row index.
        """
        # Convert list of tokens into safely encoded byte string (handles ints > 255)
        byte_rep = str(ngram_tokens).encode('utf-8')
        
        # Use an extremely fast, deterministic hash (SHA-256 for demonstration)
        hash_digest = hashlib.sha256(byte_rep).digest()
        
        # Convert first 8 bytes of hash into an integer, module table_size
        row_id = int.from_bytes(hash_digest[:8], 'little') % self.table_size
        return row_id

    def write_fact(self, ngram_tokens: list[int], factual_embedding: torch.Tensor):
        """Saves a factual embedding to the CPU row determined by the n-gram hash."""
        row_id = self.hash_ngram(ngram_tokens)
        self.memory_bank[row_id] = factual_embedding.cpu().to(torch.float16)
        print(f"   [Host] Fact written to row {row_id}")

    def async_prefetch_to_gpu(self, ngram_tokens: list[int], gpu_stream: torch.cuda.Stream) -> torch.Tensor:
        """
        The Magic Mechanism: PCIe Asynchronous Transfer!
        We trigger this lookup during layer 0. The GPU will continue computing layers 1, 2, 3...
        while this factual payload travels over the PCIe bus in the background.
        By Layer 12, the payload has arrived on the GPU with zero time penalty.
        """
        row_id = self.hash_ngram(ngram_tokens)
        cpu_tensor_slice = self.memory_bank[row_id]
        
        # Run the transfer on a dedicated background CUDA stream
        with torch.cuda.stream(gpu_stream):
            # non_blocking=True is the key! It tells PyTorch: 
            # "Initiate DMA transfer from Pinned RAM -> VRAM, but don't pause Python."
            gpu_tensor = cpu_tensor_slice.to('cuda', non_blocking=True)
            
        return gpu_tensor

# --- Test the Engram Architecture ---
if __name__ == "__main__":
    print("=== Testing Engram PCIe Offload ===")
    
    # 1. Initialize our CPU RAM factual store (takes ~4GB of system RAM)
    engram = EngramHostTable(table_size=1000000, embedding_dim=2048)
    
    # 2. Simulate encountering a sequence of tokens representing a factual query
    # e.g., tokens for "capital of France" = [842, 13, 2940]
    query_tokens = [842, 13, 2940]
    
    # Let's write a simulated factual embedding (representing "Paris") into the Engram
    mock_paris_embedding = torch.randn(2048, dtype=torch.float16)
    engram.write_fact(query_tokens, mock_paris_embedding)
    
    # 3. PCIe ASYNC PREFETCH DEMONSTRATION
    print("\n[Transformer Layer 0] Encountered query tokens. Initiating background PCIe prefetch...")
    
    # Create a background stream for the PCIe highway
    pcie_stream = torch.cuda.Stream()
    
    # Initiate the transfer. This takes theoretical time, but Python moves on instantly!
    gpu_factual_payload = engram.async_prefetch_to_gpu(query_tokens, pcie_stream)
    
    print("[Transformer Layer 1-11] GPU is busy doing heavy dense math...")
    # Simulate the GPU doing heavy work while the data travels
    dummy_work = torch.matmul(torch.randn(4096, 4096, device='cuda'), torch.randn(4096, 4096, device='cuda'))
    
    print("[Transformer Layer 12] We need the fact now! Synchronizing stream...")
    # Wait for the background transfer to guarantee it has arrived
    pcie_stream.synchronize()
    
    # It's here! We can inject it into the residual stream over the GPU.
    print(f"Success! Payload arrived on GPU. Shape: {gpu_factual_payload.shape}, Device: {gpu_factual_payload.device}")

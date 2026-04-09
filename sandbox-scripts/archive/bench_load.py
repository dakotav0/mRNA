import torch
import torch.cuda as cuda
from unsloth import FastLanguageModel
import time

def get_vram_usage():
    # Returns used VRAM in GB
    return cuda.memory_allocated() / (1024**3)

def report_metrics(step_name):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    allocated = cuda.memory_allocated() / (1024**3)
    reserved = cuda.memory_reserved() / (1024**3)
    free, total = cuda.mem_get_info()
    used_total = (total - free) / (1024**3)
    
    print(f"--- {step_name} ---")
    print(f"PyTorch Allocated: {allocated:.2f} GB")
    print(f"PyTorch Reserved:  {reserved:.2f} GB")
    print(f"System Reported Used: {used_total:.2f} GB / {total / (1024**3):.2f} GB")
    print("-" * 30)

def main():
    print("=== Gemma 3 4B VRAM Benchmark (4-bit Unsloth) ===")
    report_metrics("Baseline")

    model_name = "unsloth/gemma-3-4b-it"
    max_seq_length = 4096
    
    print(f"Loading {model_name} in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    report_metrics("After Load (4-bit)")

    context_lengths = [1024, 2048, 4096]
    
    for clen in context_lengths:
        print(f"Testing forward pass with context length: {clen}...")
        # Create dummy input
        input_ids = torch.randint(0, 30000, (1, clen), device="cuda")
        
        # Clear cache before forward pass
        torch.cuda.empty_cache()
        start_vram = get_vram_usage()
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        torch.cuda.synchronize()
        end_vram = get_vram_usage()
        peak_vram = cuda.max_memory_allocated() / (1024**3)
        
        print(f"Context {clen} | Footprint: {end_vram:.2f} GB | Peak: {peak_vram:.2f} GB")
        # Reset peak for next test
        cuda.reset_peak_memory_stats()
        
    print("\nBenchmark complete.")

if __name__ == "__main__":
    main()

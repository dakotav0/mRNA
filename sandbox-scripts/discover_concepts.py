"""
Unsupervised Concept Discovery Pipeline

Extracts activations from an unlabeled dataset, clusters them locally via HDBSCAN 
to isolate dense domains, utilizes the base model to auto-generate labels, 
and interactively routes pristine adapters subsets.

Usage
-----
python sandbox-scripts/discover_concepts.py \
    --layer 25 \
    --dataset camel-ai/biology \
    --max-examples 5000
"""

import argparse
import os
import sys
import random
import queue
import threading
import torch
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mrna.router.interceptor import ActivationInterceptor
from mrna.router.pooling import masked_mean_pool, get_unsloth_base_tokenizer
from mrna.data.dataset_utils import extract_text, extract_text2
from mrna.data.paths import MRNAPaths

from datasets import load_dataset
from unsloth import FastLanguageModel
from sklearn.cluster import HDBSCAN

def local_llm_labeler(model, tokenizer, texts, device, max_tokens=15):
    """
    Submits 5 texts to the local base model to generate a cluster label.
    Designed modularly to allow simple swapping to Gemini/Claude APIs later if needed.
    """
    prompt = "Identify the highly specific academic sub-field these prompts share. Reply within 1 to 3 words, underscore-separated (example outputs: fluid_dynamics, calculus, electromagnetism).\n\n"
    for i, t in enumerate(texts[:5]):
        # Snip overly long prompts to save context
        snip = t[:300].replace('\n', ' ') + "..." if len(t) > 300 else t.replace('\n', ' ')
        prompt += f"Prompt {i+1}: {snip}\n"
    prompt += "\nSub-field label (snake_case):"

    _tok = get_unsloth_base_tokenizer(tokenizer)
    inputs = _tok(prompt, return_tensors="pt").to(device)

    # Use strict generation constraints for naming
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3, # Low temp for deterministic naming
            pad_token_id=_tok.eos_token_id
        )
    
    response = _tok.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Simple sanitization
    clean = response.strip().lower()
    clean = ''.join(c if c.isalnum() else '_' for c in clean)
    import re
    clean = re.sub(r'_+', '_', clean).strip('_')
    if not clean:
        return "unknown_cluster"
    return clean

def main():
    parser = argparse.ArgumentParser(description="Unsupervised Concept Discovery via HDBSCAN and LLM auto-labeling.")
    parser.add_argument("--dataset", required=True, help="HF dataset ID to analyze (e.g., camel-ai/math)")
    parser.add_argument("--dataset-config", default=None, help="HF dataset config/subset name")
    parser.add_argument("--split", default="train", help="Dataset split to evaluate")
    parser.add_argument("--text-column", default="instruction", help="Fallback primary text column")
    parser.add_argument("--text-column2", default="output", help="Fallback secondary text column")
    
    parser.add_argument("--model-id", default="unsloth/gemma-4-E2B-it")
    parser.add_argument("--model-revision", default="37ea165b3fba25b7d851f8ce4ccff9a4f0751cee")
    parser.add_argument("--layer", type=int, default=25, help="Transformer layer index to intercept")
    
    parser.add_argument("--max-examples", type=int, default=5000, help="Number of examples to profile broadly")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prefetch-buffer", type=int, default=128,
                        help="RAM prefetch queue depth (examples). Generator runs in background thread, "
                             "overlapping data loading with GPU forward passes.")
    parser.add_argument("--holdout-ratio", type=float, default=0.2, help="Holdout split for approved clusters")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = MRNAPaths(model_id=args.model_id)

    print(f"\n{'='*60}")
    print(f"mRNA Unsupervised Concept Discovery")
    print(f"{'='*60}")
    print(f"Scanning Dataset : {args.dataset}")
    print(f"Sample Limit     : {args.max_examples}")

    # 1. Broad Sweep (Activation Extraction)
    print("\n[Phase 1/4] Preparing Base Model and Streaming Dataset...")
    load_kwargs_model = dict(model_name=args.model_id, max_seq_length=args.max_seq_len, dtype=None, load_in_4bit=True)
    if args.model_revision:
        load_kwargs_model["revision"] = args.model_revision
    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs_model)
    FastLanguageModel.for_inference(model)

    interceptor = ActivationInterceptor(target_layer=args.layer)
    interceptor.attach_to_model(model)
    _tok = get_unsloth_base_tokenizer(tokenizer)

    load_kwargs = dict(split=args.split, streaming=True)
    if args.dataset_config:
        load_kwargs["name"] = args.dataset_config
    ds = load_dataset(args.dataset, **load_kwargs)

    fallback_cols = ["text", "instruction", "input", "question", "content", "prompt", "message_1"]
    fallback_cols2 = ["output", "answer", "response", "solution", "message_2"]

    collected_activations = []
    collected_texts = []
    
    batch_texts = []
    
    def flush_batch():
        if not batch_texts:
            return
        enc = _tok(
            text=batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
        ).to(device)

        with torch.no_grad():
            model(**enc)

        for act in interceptor.intercepted_activations:
            pooled = masked_mean_pool(act, enc["attention_mask"].cpu())
            for i in range(len(pooled)):
                collected_activations.append(pooled[i].unsqueeze(0))
                collected_texts.append(batch_texts[i])
        
        interceptor.intercepted_activations.clear()
        batch_texts.clear()

    # Prefetch generator runs in a background thread so GPU is never idle waiting on data.
    # Queue maxsize applies backpressure — generator stays prefetch_buffer examples ahead,
    # which is trivially small in system RAM while overlapping IO with GPU compute.
    prefetch_q = queue.Queue(maxsize=args.prefetch_buffer)

    def _generator_worker():
        try:
            count = 0
            for example in ds:
                if count >= args.max_examples:
                    break
                txt = extract_text2(example, args.text_column, args.text_column2, fallback_cols + fallback_cols2)
                if txt.strip():
                    prefetch_q.put(txt)  # blocks when buffer full (backpressure)
                    count += 1
        finally:
            prefetch_q.put(None)  # sentinel

    gen_thread = threading.Thread(target=_generator_worker, daemon=True)
    gen_thread.start()

    print(f"Extracting up to {args.max_examples} latent vectors (prefetch buffer: {args.prefetch_buffer})...")
    while True:
        txt = prefetch_q.get()
        if txt is None:
            break

        batch_texts.append(txt)
        if len(batch_texts) >= args.batch_size:
            flush_batch()
            if len(collected_activations) % 500 == 0 and len(collected_activations) > 0:
                print(f"  Processed: {len(collected_activations)}")

    flush_batch()
    interceptor.detach() # Explicitly free the hook so we can generate labels safely later
    all_vectors = torch.cat(collected_activations, dim=0).float() # (N, d_model)
    print(f"Extracted {len(all_vectors)} representations.")

    # 2. HDBSCAN Density Clustering
    print("\n[Phase 2/4] Clustering vectors via PCA -> HDBSCAN...")

    # torch.pca_lowrank runs on GPU (randomized SVD) — orders of magnitude faster than
    # sklearn PCA on CPU for (5000, 1536) matrices. Drop to CPU only after reduction.
    n_components = min(50, all_vectors.shape[1], all_vectors.shape[0])
    print(f"  Reducing {all_vectors.shape[1]}D space to {n_components}D PCA (GPU)...")
    U, S, V = torch.pca_lowrank(all_vectors, q=n_components, niter=4)
    reduced_gpu = all_vectors @ V  # (N, n_components)
    reduced_vectors = reduced_gpu.cpu().numpy()

    print("  Running HDBSCAN density clustering...")
    # min_cluster_size helps prevent microscopic fragmentation
    clusterer = HDBSCAN(min_cluster_size=15, min_samples=3, n_jobs=-1)
    labels = clusterer.fit_predict(reduced_vectors)

    cluster_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_to_indices[label].append(idx)
        
    noise_count = len(cluster_to_indices.get(-1, []))
    n_clusters = len(cluster_to_indices) - (1 if -1 in cluster_to_indices else 0)
    print(f"  Found {n_clusters} clusters (Discarding {noise_count} noise layers).")

    if n_clusters == 0:
        print("[WARNING] HDBSCAN failed to find distinct clusters. The dataset might be too uniform or too small.")
        sys.exit(0)

    # 3. LLM Auto-Labeler
    print("\n[Phase 3/4] Repurposing Base Model for Cluster Auto-Labeling...")
    cluster_metadata = {}
    
    for label_id, indices in cluster_to_indices.items():
        if label_id == -1:
            continue # Skip noise
            
        # Sample 5 texts for the LLM
        sample_indices = random.sample(indices, min(5, len(indices)))
        sample_texts = [collected_texts[i] for i in sample_indices]
        
        suggested_name = local_llm_labeler(model, tokenizer, sample_texts, device)
        cluster_metadata[label_id] = {
            "name": suggested_name,
            "size": len(indices),
            "indices": indices,
            "samples": sample_texts
        }
        print(f"  Cluster {label_id}: -> {suggested_name} ({len(indices)} rows)")

    # Disable VRAM occupation for interactive phase safely
    del model
    torch.cuda.empty_cache()

    # 4. Interactive Triage & Export
    print("\n[Phase 4/4] Interactive Triage...")
    print("Review discovered concepts. You may [A]ccept, [R]ename, or [D]iscard them.\n")
    
    approved_clusters = []

    for label_id, meta in cluster_metadata.items():
        print("-" * 50)
        print(f"Cluster {label_id} | Size: {meta['size']} rows")
        print(f"Suggested Auto-Name: \033[92m{meta['name']}\033[0m")
        print("Sample Data:")
        for i, s in enumerate(meta['samples'][:2]): # Just show 2 samples to keep terminal clean
            snip = s[:150].replace('\n', ' ') + "..." if len(s) > 150 else s.replace('\n', ' ')
            print(f"  > {snip}")
            
        while True:
            choice = input(f"\nAction for '{meta['name']}' ([A]ccept / [R]ename / [D]iscard): ").strip().lower()
            if choice in ['a', 'accept']:
                approved_clusters.append((meta['name'], meta['indices']))
                break
            elif choice in ['r', 'rename']:
                new_name = input("Enter new snake_case name: ").strip()
                if new_name:
                    approved_clusters.append((new_name, meta['indices']))
                    break
                else:
                    print("Invalid name. Try again.")
            elif choice in ['d', 'discard']:
                print(f"Discarded cluster.")
                break
            else:
                print("Invalid choice.")

    # Export Loop
    if not approved_clusters:
        print("\nNo clusters approved. Exiting.")
        sys.exit(0)

    print("\nSlicing tensors and exporting to MRNAPaths...")
    activations_dir = paths.get_activations_dir(args.layer)
    
    for concept_name, indices in approved_clusters:
        # Pull correct vectors from the master list
        cluster_tensors = [all_vectors[i].unsqueeze(0) for i in indices]
        
        # Shuffle for holdout split
        random.shuffle(cluster_tensors)
        holdout_count = int(len(cluster_tensors) * args.holdout_ratio)
        train_count = len(cluster_tensors) - holdout_count
        
        test_tensors = cluster_tensors[:holdout_count]
        train_tensors = cluster_tensors[holdout_count:]
        
        train_path = paths.get_activation_file(args.layer, concept_name, "train")
        torch.save(train_tensors, train_path)
        
        if holdout_count > 0:
            test_path = paths.get_activation_file(args.layer, concept_name, "test")
            torch.save(test_tensors, test_path)
            
        print(f"  Saved '{concept_name}' -> {train_count} train / {holdout_count} test")
        
    print(f"\nDiscovery Process Complete! Tensors saved in: {activations_dir}")

if __name__ == "__main__":
    main()

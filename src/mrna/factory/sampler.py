import os
import random
import torch
from datasets import load_dataset
from typing import Optional, List

from mrna.core.config import MRNAPaths, config
from mrna.substrate.backend import get_backend
from mrna.router.interceptor import ActivationInterceptor
from mrna.router.pooling import masked_mean_pool, get_unsloth_base_tokenizer
from mrna.data.dataset_utils import extract_text2

def harvest_activations(
    concept: str,
    dataset_id: Optional[str] = None,
    model_id: Optional[str] = None,
    layer: Optional[int] = None,
    max_examples: int = 5000,
    holdout_ratio: float = 0.2,
    batch_size: int = 8,
    save_every: int = 500,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    skip_if_exists: bool = True,
    **kwargs
):
    """
    Generalized activation harvester.
    Uses mrna.core.config to resolve defaults and mrna.substrate.backend for tensor operations.
    """
    backend = get_backend("torch") # Default to torch for now
    
    # 1. Resolve parameters from config if not provided
    mid = model_id or config.current_model_id
    m_cfg = config.get_model_config(mid)
    
    ds_id = dataset_id or config.science_triad_datasets.get(concept)
    if not ds_id:
        raise ValueError(f"No dataset_id provided and concept '{concept}' not found in science triad.")
        
    target_layer = layer if layer is not None else m_cfg.get("harvest_layer")
    max_seq_len = kwargs.get("max_seq_len", 512)
    model_revision = m_cfg.get("revision")
    
    # 2. Setup paths
    # Note: Using the new MRNAPaths logic
    activations_dir = MRNAPaths.DATA / mid / "activations" / f"layer_{target_layer}"
    activations_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = activations_dir / f"{concept}_train.pt"
    test_path = activations_dir / f"{concept}_test.pt"
    
    # 3. Check for existing data
    if skip_if_exists and train_path.exists():
        prev = torch.load(train_path, map_location="cpu", weights_only=True)
        if len(prev) >= max_examples:
            print(f"Skipping: {train_path} already complete.")
            return

    # 4. Load Model (via Backend)
    print(f"Loading model {mid}...")
    model, tokenizer = backend.load_model(
        mid, 
        max_seq_length=max_seq_len, 
        load_in_4bit=True,
        revision=model_revision
    )
    # FastLanguageModel specific optimizations (TODO: move to backend for true abstraction)
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    model.eval()

    # 5. Attach Interceptor
    interceptor = ActivationInterceptor(target_layer=target_layer)
    interceptor.attach_to_model(model)

    # 6. Load Dataset
    print(f"Loading dataset {ds_id}...")
    ds = load_dataset(ds_id, split=kwargs.get("split", "train"), streaming=True)
    
    # 7. Harvest Loop (Simplified version of sandbox-scripts/harvest_hf.py)
    collected_train = []
    collected_test = []
    batch_texts = []
    fallback_cols = ["text", "instruction", "input", "question", "content", "prompt"]

    def _flush():
        if not batch_texts: return
        _tok = get_unsloth_base_tokenizer(tokenizer)
        enc = _tok(
            text=batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to(device)
        
        with torch.no_grad():
            model(**enc)
            
        for act in interceptor.intercepted_activations:
            pooled = masked_mean_pool(act, enc["attention_mask"].cpu())
            for i in range(len(pooled)):
                if holdout_ratio > 0 and random.random() < holdout_ratio:
                    collected_test.append(pooled[i].unsqueeze(0))
                else:
                    collected_train.append(pooled[i].unsqueeze(0))
        
        interceptor.intercepted_activations.clear()
        batch_texts.clear()
        
        # Incremental save
        if (len(collected_train) + len(collected_test)) % save_every < batch_size:
            _save_checkpoint()

    def _save_checkpoint():
        if collected_train:
            combined = torch.cat(collected_train, dim=0)
            torch.save(combined, train_path)
        if collected_test:
            combined = torch.cat(collected_test, dim=0)
            torch.save(combined, test_path)

    print(f"Harvesting {max_examples} examples...")
    for example in ds:
        if len(collected_train) + len(collected_test) >= max_examples:
            break
            
        text = extract_text2(example, kwargs.get("text_column", "message_1"), None, fallback_cols)
        if not text.strip(): continue
        
        batch_texts.append(text)
        if len(batch_texts) >= batch_size:
            _flush()

    _flush()
    _save_checkpoint()
    interceptor.detach()
    print(f"Done. Saved to {activations_dir}")

if __name__ == "__main__":
    # Internal test/CLI entry point
    harvest_activations(concept="biology", max_examples=10)

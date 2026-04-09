import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from typing import Optional, List, Dict, Any

from mrna.core.config import config, MRNAPaths
from mrna.substrate.backend import get_backend
from mrna.router.interceptor import ActivationInterceptor
from mrna.router.sae import CBSAE
from mrna.router.pooling import masked_mean_pool, get_unsloth_base_tokenizer
from mrna.data.dataset_utils import extract_text2

def analyze_dataset_boundary(
    dataset_id: str,
    concepts: List[str],
    model_id: Optional[str] = None,
    layer: Optional[int] = None,
    max_examples: int = 200,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs
) -> Dict[str, Any]:
    """
    Analyzes whether a new dataset overlaps with existing concepts.
    Returns a diagnostic report dictionary.
    """
    mid = model_id or config.current_model_id
    m_cfg = config.get_model_config(mid)
    target_layer = layer if layer is not None else m_cfg.get("harvest_layer")
    
    # 1. Load Model & Interceptor
    backend = get_backend("torch")
    model, tokenizer = backend.load_model(mid, load_in_4bit=True)
    _tok = get_unsloth_base_tokenizer(tokenizer)
    
    interceptor = ActivationInterceptor(target_layer=target_layer)
    interceptor.attach_to_model(model)
    
    # 2. Extract Activations
    ds = load_dataset(dataset_id, split=kwargs.get("split", "train"), streaming=True).take(max_examples)
    batch_texts = []
    fallback_cols = ["text", "instruction", "input", "question", "content", "prompt", "message_1"]
    
    for ex in ds:
        txt = extract_text2(ex, kwargs.get("text_column", "instruction"), None, fallback_cols)
        if txt.strip(): batch_texts.append(txt)

    pooled_activations = []
    batch_size = 8
    with torch.no_grad():
        for i in range(0, len(batch_texts), batch_size):
            batch = batch_texts[i:i+batch_size]
            enc = _tok(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            model(**enc)
            for act in interceptor.intercepted_activations:
                pooled_activations.append(masked_mean_pool(act, enc["attention_mask"].cpu()))
            interceptor.intercepted_activations.clear()

    interceptor.detach()
    all_activations = torch.cat(pooled_activations, dim=0).float()

    # 3. Predict with SAE
    sae_weights = MRNAPaths.DATA / mid / "sae_weights.pt"
    if not sae_weights.exists():
        raise FileNotFoundError(f"Trained SAE weights not found at {sae_weights}")

    state_dict = torch.load(sae_weights, map_location="cpu", weights_only=True)
    d_model = m_cfg.get("d_model")
    expansion = state_dict["encoder.weight"].shape[0] // d_model
    sae = CBSAE(d_model=d_model, expansion_factor=expansion, bottleneck_features=len(concepts))
    sae.load_state_dict(state_dict)
    sae.eval()

    with torch.no_grad():
        pre_relu = sae.encoder(all_activations)
        probs = torch.softmax(pre_relu[:, :len(concepts)], dim=-1)
    
    avg_probs = probs.mean(dim=0)
    report = {
        "dataset": dataset_id,
        "avg_confidences": {name: avg_probs[i].item() for i, name in enumerate(concepts)},
        "highest_bleed": concepts[avg_probs.argmax().item()],
        "highest_score": avg_probs.max().item(),
    }
    
    # Verdict logic
    score = report["highest_score"]
    if score > 0.7: report["verdict"] = "DANGER"
    elif score < 0.3: report["verdict"] = "CLEAN"
    else: report["verdict"] = "MODERATE"
    
    return report

if __name__ == "__main__":
    # Example usage
    # print(analyze_dataset_boundary("camel-ai/math", ["biology", "chemistry", "physics"]))
    pass

import os
import glob
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

from mrna.core.config import config, MRNAPaths
from mrna.router.sae import CBSAE

def evaluate_sae_holdouts(
    concepts: List[str],
    model_id: Optional[str] = None,
    layer: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluates routing accuracy and conceptual bleed across all relevant holdout datasets.
    """
    mid = model_id or config.current_model_id
    m_cfg = config.get_model_config(mid)
    target_layer = layer if layer is not None else m_cfg.get("harvest_layer")
    d_model = m_cfg.get("d_model")
    
    # 1. Load SAE
    sae_path = MRNAPaths.DATA / mid / "sae_weights.pt"
    if not sae_path.exists():
        raise FileNotFoundError(f"SAE weights not found at {sae_path}")
        
    state_dict = torch.load(sae_path, map_location="cpu", weights_only=True)
    expansion = state_dict["encoder.weight"].shape[0] // d_model
    sae = CBSAE(d_model=d_model, expansion_factor=expansion, bottleneck_features=len(concepts))
    sae.load_state_dict(state_dict)
    sae.eval()

    # 2. Find Holdout Files
    holdout_dir = MRNAPaths.DATA / mid / "activations" / f"layer_{target_layer}"
    holdout_files = glob.glob(str(holdout_dir / "**" / "*_test.pt"), recursive=True)
    
    overall_results = {
        "concepts": [],
        "overall_accuracy": 0.0,
        "total_samples": 0,
        "correct_samples": 0
    }

    # 3. Process
    for file_path in holdout_files:
        filename = os.path.basename(file_path).lower()
        mapped_idx = -1
        for i, concept in enumerate(concepts):
            if concept.lower() in filename:
                mapped_idx = i
                break
        
        if mapped_idx == -1: continue
        
        acts = torch.load(file_path, map_location="cpu", weights_only=True)
        if isinstance(acts, list): acts = torch.cat(acts, dim=0)
        
        with torch.no_grad():
            pre_relu = sae.encoder(acts.float())
            preds = pre_relu[:, :len(concepts)].argmax(dim=-1)
            
        correct = (preds == mapped_idx).sum().item()
        total = len(preds)
        
        confusion = [0] * len(concepts)
        for p in preds: confusion[p.item()] += 1
        
        res = {
            "name": concepts[mapped_idx],
            "accuracy": correct / total,
            "samples": total,
            "confusion": {concepts[i]: confusion[i] for i in range(len(concepts))}
        }
        overall_results["concepts"].append(res)
        overall_results["correct_samples"] += correct
        overall_results["total_samples"] += total

    if overall_results["total_samples"] > 0:
        overall_results["overall_accuracy"] = overall_results["correct_samples"] / overall_results["total_samples"]
        
    return overall_results

if __name__ == "__main__":
    # Example usage
    # print(evaluate_sae_holdouts(["biology", "chemistry", "physics"]))
    pass

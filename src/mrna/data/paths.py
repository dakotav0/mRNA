from mrna.core.config import config, MRNAPaths as NewPaths
import os

class MRNAPaths:
    def __init__(self, data_root: str = "data", model_id: str = None):
        self.config = config
        self.model_id = model_id or config.current_model_id
        self.data_root = data_root

    @property
    def model_dir(self) -> str:
        return str(NewPaths.DATA / self.model_id)

    def get_activations_dir(self, layer: int) -> str:
        p = NewPaths.DATA / self.model_id / "activations" / f"layer_{layer}"
        p.mkdir(parents=True, exist_ok=True)
        return str(p)

    def get_activation_file(self, layer: int, concept: str, split: str = None) -> str:
        d = self.get_activations_dir(layer)
        if split:
            return os.path.join(d, f"{concept}_{split}.pt")
        return os.path.join(d, f"{concept}.pt")

    def get_sae_dir(self) -> str:
        p = NewPaths.DATA / self.model_id / "sae"
        p.mkdir(parents=True, exist_ok=True)
        return str(p)

    def get_sae_weights(self, layer: int) -> str:
        # Note: we are moving to concept-based sae_weights.pt in root of data/model/
        return os.path.join(self.get_sae_dir(), f"weights_layer{layer}.pt")

    def get_adapter_dir(self, concept: str) -> str:
        p = NewPaths.DATA / self.model_id / "adapters" / f"{concept}_lora"
        p.mkdir(parents=True, exist_ok=True)
        return str(p)

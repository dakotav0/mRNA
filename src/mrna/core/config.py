import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class MRNAPaths:
    """
    Centralized path manager for the mRNA infrastructure.
    Ensures relative paths resolve correctly regardless of the CWD.
    """

    @staticmethod
    def _find_root():
        # Search upwards for a root marker
        current = Path(__file__).resolve().parent
        # Look for model_config.yaml or pyproject.toml up to 5 levels
        for _ in range(5):
            if (current / "model_config.yaml").exists() or (
                current / "pyproject.toml"
            ).exists():
                return current
            current = current.parent
        # Fallback to CWD to ensure it works in localized CI runners
        return Path.cwd()

    ROOT = _find_root()
    SRC = ROOT / "src"
    DATA = ROOT / "data"
    MODELS = ROOT / "models"
    OUTPUTS = ROOT / "outputs"
    CONFIG_FILE = ROOT / "model_config.yaml"

    @classmethod
    def resolve(cls, path_str: str) -> Path:
        """Resolves a path string relative to the project root."""
        return cls.ROOT / path_str

    @classmethod
    def get_activations_dir(cls, model_id: str, layer_name: str) -> Path:
        """Standard path for hidden state activations."""
        return cls.DATA / model_id / "activations" / layer_name

    @classmethod
    def get_sae_weights_path(cls, model_id: str, layer: int) -> Path:
        """Standard path for trained CBSAE weights at a specific layer."""
        return cls.DATA / model_id / f"sae_weights_L{layer}.pt"

    @classmethod
    def get_adapter_dir(cls, concept: str, model_id: Optional[str] = None) -> Path:
        """Standard path for LoRA adapter checkpoints."""
        # Note: Adapters are often shared across models if the base is similar,
        # but structured under models/ if preferred.
        mid = model_id or config.current_model_id
        return cls.DATA / mid / "adapters" / f"{concept}_lora"


class ConfigManager:
    """
    Handles loading and accessing model/dataset configurations.
    Shared by mRNA pipeline and MIIN router.
    """

    def __init__(self):
        self.config_data = self._load_yaml()

    def _load_yaml(self) -> Dict[str, Any]:
        if not MRNAPaths.CONFIG_FILE.exists():
            return {"models": {}, "current_target": None}

        with open(MRNAPaths.CONFIG_FILE, "r") as f:
            return yaml.safe_load(f)

    @property
    def current_model_id(self) -> Optional[str]:
        return self.config_data.get("current_target")

    @property
    def datasets_path(self) -> Path:
        """Root directory for local datasets."""
        p = self.config_data.get("datasets_path", "data/datasets")
        return MRNAPaths.resolve(p)

    def get_harvest_layers(self, model_id: Optional[str] = None) -> List[int]:
        """Returns the list of layers configured for activation harvesting."""
        m_cfg = self.get_model_config(model_id)
        # Handle both list (new) and single int (legacy)
        layers = m_cfg.get("harvest_layers", m_cfg.get("harvest_layer"))
        if layers is None:
            return []
        return [layers] if isinstance(layers, int) else layers

    def get_logic_layer(self, model_id: Optional[str] = None) -> int:
        """Returns the first layer in the list (the abstract logic pass)."""
        layers = self.get_harvest_layers(model_id)
        return layers[0] if layers else 0

    def get_voice_layer(self, model_id: Optional[str] = None) -> Optional[int]:
        """Returns the second layer in the list (the persona voice pass)."""
        layers = self.get_harvest_layers(model_id)
        return layers[1] if len(layers) > 1 else None

    def get_model_config(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        mid = model_id or self.current_model_id
        if not mid:
            raise ValueError(
                "No model_id provided and no current_target set in config."
            )

        models = self.config_data.get("models", {})
        if mid not in models:
            raise KeyError(f"Model '{mid}' not found in configuration.")

        return models[mid]

    def get_pipeline_arg(
        self, stage: str, key: str, model_id: Optional[str] = None, override: Any = None
    ) -> Any:
        """
        Resolves a hyperparameter value based on hierarchy:
        1. CLI Override (if not None)
        2. Model-specific pipeline setting
        3. Global pipeline_defaults
        """
        if override is not None:
            return override

        # 2. Check model-specific
        try:
            m_cfg = self.get_model_config(model_id)
            m_pipe = m_cfg.get("pipeline", {}).get(stage, {})
            if key in m_pipe:
                return m_pipe[key]
        except (ValueError, KeyError):
            pass

        # 3. Check global defaults
        global_defaults = self.config_data.get("pipeline_defaults", {}).get(stage, {})
        return global_defaults.get(key)

    @property
    def approved_datasets(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns the library of approved datasets from the config.
        """
        return self.config_data.get("datasets", {})

    @property
    def science_triad_datasets(self) -> Dict[str, str]:
        """
        Maintains backward compatibility by returning a mapping of name -> HF ID.
        """
        return {name: ds.get("id") for name, ds in self.approved_datasets.items()}

    def validate_setup(self, concepts: List[str], model_id: Optional[str] = None):
        """
        Performs a 'Dry Run' validation of the environment.
        """
        mid = model_id or self.current_model_id
        m_cfg = self.get_model_config(mid)
        print(f"[Dry Run] Validating setup for {mid}...")

        # 1. Model path check (if local)
        m_path = m_cfg.get("path")
        if os.path.isdir(m_path) or os.path.isfile(m_path):
            if not os.path.exists(m_path):
                print(f"  [!] Warning: Model path {m_path} not found on disk.")

        # 2. Dataset check
        for concept in concepts:
            ds_info = self.approved_datasets.get(concept)
            if not ds_info:
                print(
                    f"  [!] Warning: Concept '{concept}' not in approved_datasets library."
                )
                continue

            ds_id = ds_info.get("id")
            if ds_id:
                local_p = Path(ds_id)
                if not (local_p.is_absolute() or ds_id.startswith(".")):
                    local_p = self.datasets_path / ds_id

                if local_p.exists():
                    print(f"  [✓] Local dataset for {concept} found at {local_p}")
                else:
                    print(
                        f"  [~] {concept} will be fetched from Hugging Face Hub (ID: {ds_id})"
                    )

        print("[Dry Run] Validation complete.")


# Global singleton for easy import
config = ConfigManager()

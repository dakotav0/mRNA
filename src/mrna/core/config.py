import os
from pathlib import Path
from typing import Any, Dict, Optional

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

    @property
    def science_triad_datasets(self) -> Dict[str, str]:
        """
        Global pointers for the science triad datasets.
        Used by both the factory (training) and MIIN (context/routing).
        """
        return {
            "biology": "camel-ai/biology",
            "physics": "camel-ai/physics",
            "chemistry": "camel-ai/chemistry",
        }


# Global singleton for easy import
config = ConfigManager()

import os
from pathlib import Path

import pytest

from mrna.core.config import MRNAPaths, config


def test_root_path():
    """Verifies that MRNAPaths resolves to the project root (containing config)."""
    # ROOT should exist and be a directory
    assert MRNAPaths.ROOT.exists()
    assert MRNAPaths.ROOT.is_dir()
    # In CI, we definitely expect the config file to be reachable from ROOT
    assert MRNAPaths.CONFIG_FILE.exists()


def test_config_loading():
    """Verifies that the config manager loads the expected models."""
    # Check if we have at least one of our expected models
    m_id = config.current_model_id
    assert m_id is not None

    cfg = config.get_model_config()
    assert "path" in cfg
    assert "d_model" in cfg
    assert isinstance(cfg["d_model"], int)


def test_dataset_config():
    """Verifies retrieval of science triad dataset identifiers."""
    datasets = config.science_triad_datasets
    assert "biology" in datasets
    assert datasets["biology"] == "camel-ai/biology"


def test_path_resolution():
    """Verifies that MRNAPaths.resolve works as expected."""
    resolved = MRNAPaths.resolve("non_existent_test_file.txt")
    assert resolved == MRNAPaths.ROOT / "non_existent_test_file.txt"

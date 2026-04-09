import sys
from unittest.mock import MagicMock

import pytest


# Global mock to intercept hardware-dependent libraries before they trigger GPU checks
@pytest.fixture(autouse=True, scope="session")
def mock_hardware_deps():
    """
    Ensures unsloth and trl are mocked globally for CPU-only CI environments.
    This prevents NotImplementedError at import time.
    """
    # Mock unsloth
    if "unsloth" not in sys.modules:
        mock_unsloth = MagicMock()
        mock_flm = MagicMock()
        mock_flm.from_pretrained.return_value = (MagicMock(), MagicMock())
        mock_unsloth.FastLanguageModel = mock_flm
        sys.modules["unsloth"] = mock_unsloth

    # Mock trl (SFTTrainer)
    if "trl" not in sys.modules:
        sys.modules["trl"] = MagicMock()

    # Mock datasets if not present
    if "datasets" not in sys.modules:
        sys.modules["datasets"] = MagicMock()

    return True

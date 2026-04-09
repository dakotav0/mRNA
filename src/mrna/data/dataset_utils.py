"""
Dataset utility components for formatting parsing and subset loading HF datasets.
"""

from typing import List, Optional


def extract_text(example: dict, text_column: str, fallback_columns: List[str]) -> str:
    """Pull text from an example dict, trying fallback column names.

    Handles list-valued columns (e.g. daily_dialog's 'dialog' field which is
    List[str]) by joining turns with a space.
    """

    def _coerce(v) -> str:
        if isinstance(v, list):
            return " ".join(str(t) for t in v if t)
        return str(v)

    if text_column in example and example[text_column]:
        return _coerce(example[text_column])
    for col in fallback_columns:
        if col in example and example[col]:
            return _coerce(example[col])
    # Last resort: concatenate all string-valued fields
    parts = [str(v) for v in example.values() if isinstance(v, str) and v.strip()]
    return " ".join(parts[:3])


def extract_text2(
    example: dict, col1: str, col2: Optional[str], fallback_columns: List[str]
) -> str:
    """Concatenate two columns for richer domain signal (e.g. Q+A instead of Q-only)."""
    t1 = extract_text(example, col1, fallback_columns)
    if not col2 or col2 not in example or not example[col2]:
        return t1

    def _coerce(v) -> str:
        if isinstance(v, list):
            return " ".join(str(t) for t in v if t)
        return str(v)

    t2 = _coerce(example[col2])
    return f"{t1} {t2}".strip()

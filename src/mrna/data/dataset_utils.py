"""
Dataset utility components for formatting, parsing, and smart loading.
Consolidates logic for local vs remote dataset resolution.
"""

import os
from pathlib import Path
from typing import List, Optional, Union

import datasets

# ---------------------------------------------------------------------------
# Formatting Templates (Moved from adapter.py to avoid pickle issues)
# ---------------------------------------------------------------------------
CHAT_PROMPT = """Below is a question about {topic}. Write a thorough and accurate response.

### Question:
{question}

### Response:
{answer}"""

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


def get_dataset_formatter(
    dataset_id: str, eos_token: str, text_column: Optional[str] = None
):
    """Probes dataset and returns appropriate formatting function.

    Accepts ``eos_token`` as a plain string (not the full tokenizer) so the
    returned closure is picklable by multiprocessing DataLoader workers.
    """
    if os.path.isdir(dataset_id):
        try:
            ds = datasets.load_from_disk(dataset_id)
            if isinstance(ds, datasets.DatasetDict):
                ds = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
            sample = [ds[0]]
        except Exception:
            # Fallback for raw files (e.g. PIPPA JSONL)
            local_p = Path(dataset_id)
            raw_files = (
                list(local_p.glob("*.jsonl"))
                + list(local_p.glob("*.json"))
                + list(local_p.glob("*.csv"))
            )
            if raw_files:
                fmt = raw_files[0].suffix[1:]
                if fmt == "jsonl":
                    fmt = "json"
                ds = datasets.load_dataset(
                    fmt,
                    data_files=[str(f) for f in raw_files],
                    split="train",
                    streaming=True,
                )
                sample = list(ds.take(1))
            else:
                raise ValueError(f"Could not resolve dataset columns at {dataset_id}")
    else:
        sample = list(
            datasets.load_dataset(dataset_id, split="train", streaming=True).take(1)
        )
    cols = list(sample[0].keys())

    if text_column:
        # Use the smart extract_text for the specific column provided
        def column_format(ex):
            # Batch processing: ex is a dict of lists
            batch_size = len(next(iter(ex.values())))
            out = []
            for i in range(batch_size):
                item = {k: v[i] for k, v in ex.items()}
                txt = extract_text(item, text_column, [])
                out.append(txt + eos_token)
            return {"text": out}

        return column_format

    elif "message_1" in cols and "message_2" in cols:

        def chat_format(ex):
            q_list = ex["message_1"]
            a_list = ex["message_2"]
            t_list = ex.get("topic")
            if t_list is None:
                t_list = ["science"] * len(q_list)

            return {
                "text": [
                    CHAT_PROMPT.format(topic=t or "science", question=q, answer=a)
                    + eos_token
                    for t, q, a in zip(t_list, q_list, a_list)
                ]
            }

        return chat_format
    elif "instruction" in cols:

        def alpaca_format(ex):
            i_list = ex["instruction"]
            inp_list = ex.get("input", [""] * len(i_list))
            o_list = ex["output"]
            return {
                "text": [
                    ALPACA_PROMPT.format(instruction=instr, input=inp, output=out)
                    + eos_token
                    for instr, inp, out in zip(i_list, inp_list, o_list)
                ]
            }

        return alpaca_format
    else:
        # Final fallback: desperate scavenger for each item in the batch
        def generic_format(ex):
            batch_size = len(next(iter(ex.values())))
            out = []
            for i in range(batch_size):
                item = {k: v[i] for k, v in ex.items()}
                txt = extract_text(item, [], ["text", "content", "message"])
                out.append(txt + eos_token)
            return {"text": out}

        return generic_format


def load_smart_dataset(
    dataset_id: str,
    root_path: Path,
    download: bool = False,
    split: str = "train",
    streaming: bool = True,
):
    """
    Resolves local vs remote dataset paths.
    If download is True, ensures local persistence.
    Returns (dataset, is_streaming, resolved_id)
    """
    local_p = root_path / dataset_id

    if local_p.exists() and any(local_p.iterdir()):
        print(f"[Data] Found local directory at {local_p}")
        try:
            ds = datasets.load_from_disk(str(local_p))
            if isinstance(ds, datasets.DatasetDict):
                ds = ds[split] if split in ds else ds[list(ds.keys())[0]]
            return ds, False, str(local_p)
        except Exception as e:
            print(
                "[Data] Local directory is not an Arrow dataset. Attempting raw file load..."
            )
            # Check for common raw formats
            raw_files = (
                list(local_p.glob("*.jsonl"))
                + list(local_p.glob("*.json"))
                + list(local_p.glob("*.csv"))
            )
            if raw_files:
                fmt = raw_files[0].suffix[1:]
                if fmt == "jsonl":
                    fmt = "json"
                print(f"[Data] Loading raw {fmt} files from {local_p}")
                ds = datasets.load_dataset(
                    fmt, data_files=[str(f) for f in raw_files], split="train"
                )
                return ds, False, str(local_p)
            else:
                print(
                    f"[Data] Warning: No supported raw files found in {local_p}. Falling back to remote."
                )

    if download:
        print(f"[Data] Downloading {dataset_id} to {local_p}...")
        os.makedirs(local_p.parent, exist_ok=True)
        # Handle DatasetDict correctly before saving
        ds = datasets.load_dataset(dataset_id, trust_remote_code=True)
        if isinstance(ds, datasets.DatasetDict):
            if split in ds:
                # Save only the requested split or the whole Dict?
                # Standards suggest saving the Dict and letting load_from_disk handle it.
                ds.save_to_disk(str(local_p))
            else:
                ds.save_to_disk(str(local_p))
        else:
            ds.save_to_disk(str(local_p))

        return load_smart_dataset(dataset_id, root_path, False, split, False)

    print(f"[Data] Accessing {dataset_id} (Streaming: {streaming})")
    try:
        ds = datasets.load_dataset(
            dataset_id, split=split, streaming=streaming, trust_remote_code=True
        )
    except Exception as e:
        print(
            f"[Data] Warning: Standard load failed ({e}). Trying without split/streaming..."
        )
        ds = datasets.load_dataset(dataset_id, trust_remote_code=True)
        if isinstance(ds, datasets.DatasetDict):
            ds = ds[split] if split in ds else ds[list(ds.keys())[0]]

    return ds, streaming, dataset_id


def extract_text(
    example: dict, text_column: Union[str, List[str]], fallback_columns: List[str]
) -> str:
    """
    Extracts text from example, supporting multi-column concatenation
    and nested formats (List[Dict], ShareGPT, etc.).
    """

    def _flatten(v) -> str:
        if v is None:
            return ""
        if isinstance(v, list):
            # 1. Handle List of Dictionaries (Conversation turns)
            if v and isinstance(v[0], dict):
                parts = []
                for turn in v:
                    # Look for content in common variant keys
                    content = (
                        turn.get("content")
                        or turn.get("message")
                        or turn.get("value")
                        or turn.get("text")
                        or turn.get("prompt")
                    )
                    if content:
                        parts.append(str(content))
                return "\n\n".join(parts)
            # 2. Handle simple list
            return "\n\n".join(str(t) for t in v if t)

        if isinstance(v, dict):
            # Attempt to find a known text key
            content = (
                v.get("content")
                or v.get("message")
                or v.get("text")
                or v.get("value")
                or v.get("prompt")
            )
            if content:
                return str(content)
            # If no obvious key, join top-level string values
            return " ".join(str(sv) for sv in v.values() if isinstance(sv, str))

        return str(v)

    # 1. Handle Explicit Column List or Q+A Concatenation
    cols_to_join = []
    if isinstance(text_column, list):
        cols_to_join = text_column
    elif "message_1" in example and "message_2" in example:
        cols_to_join = ["message_1", "message_2"]
    elif "conversation_a" in example:
        # Specific helper for Chatbot Arena
        cols_to_join = ["conversation_a"]
    elif text_column in example:
        cols_to_join = [text_column]

    if cols_to_join:
        parts = []
        for col in cols_to_join:
            val = example.get(col)
            if val:
                parts.append(_flatten(val))
        if parts:
            return "\n\n".join(parts)

    # 2. Fallback Scavenging
    for col in fallback_columns:
        if col in example and example[col]:
            return _flatten(example[col])

    # Final desperate attempt: scavenge for any usable text content
    # Look for long strings first
    cand = [str(v) for v in example.values() if isinstance(v, str) and len(v) > 20]
    if cand:
        return "\n\n".join(cand[:3])

    # If no strings, check if any top-level key contains a list of objects (Dialogue)
    for v in example.values():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            flat = _flatten(v)
            if len(flat) > 20:
                return flat

    return ""

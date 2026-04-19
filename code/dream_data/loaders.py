"""File loaders for the dream pipeline.

Currently:
- load_last_llm_record: mirrors load_last_jsonl_record_for_id in llm_analysis.py
  without importing the LLM module (which pulls in pydantic / requests).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def load_last_llm_record(
    jsonl_path: str | Path, gutenberg_id: int
) -> Optional[dict]:
    """Return the LAST JSONL record matching ``gutenberg_id``, or None.

    The JSONL file is appended to by ``llm_analysis.py``; later records
    supersede earlier ones (e.g. when re-running analysis). Malformed JSON
    lines are silently skipped — matching the forgiving semantics of
    ``llm_analysis.load_last_jsonl_record_for_id``.

    Parameters
    ----------
    jsonl_path : str or Path
        Path to the JSONL file (typically ``exploration_output/llm_analysis.jsonl``).
    gutenberg_id : int
        Gutenberg ID to match against ``record["gutenberg_id"]``.

    Returns
    -------
    dict or None
        The last matching record as a dict, or ``None`` if the file is
        missing or no record matches.
    """
    p = Path(jsonl_path)
    if not p.exists():
        return None

    last: Optional[dict] = None
    with p.open(encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if record.get("gutenberg_id") == gutenberg_id:
                last = record
    return last

"""Chunking utilities shared across the dream pipeline.

We treat a "chunk" as the unit that drives:
- retrieval queries (CLIP -> FAISS)
- one LLM `visual_scene` entry
- one SDXL keyframe

The current corpus is stanza-delimited when aligned `pg_raw` text is
available, but some poems have no blank line markers. For those, we fall
back to fixed-size chunking (mirrors `clip_pipeline.retrieve`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Chunk:
    chunk_index: int
    text: str
    split_mode: str  # "stanza" or "fixed"


def _join_nonempty(lines: Iterable[str]) -> str:
    parts = [str(l).strip() for l in lines if str(l).strip()]
    return " / ".join(parts)


def _split_into_stanzas_text(lines: list[str]) -> list[str]:
    stanzas: list[str] = []
    current: list[str] = []
    for line in lines:
        if str(line).strip() == "":
            if current:
                stanzas.append(_join_nonempty(current))
                current = []
        else:
            current.append(str(line).strip())
    if current:
        stanzas.append(_join_nonempty(current))
    return [s for s in stanzas if s.strip()]


def _chunk_lines_text(lines: list[str], chunk_size: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    nonempty = [str(l).strip() for l in lines if str(l).strip()]
    chunks: list[str] = []
    for i in range(0, len(nonempty), chunk_size):
        chunks.append(_join_nonempty(nonempty[i : i + chunk_size]))
    return [c for c in chunks if c.strip()]


def split_poem(lines: list[str], fallback_chunk_size: int = 8) -> list[Chunk]:
    """Split poem lines into canonical numbered chunks.

    Strategy:
    - Attempt stanza splitting by blank lines.
    - If that yields a single chunk AND the original input had no blank
      line markers, fall back to fixed-size chunking.

    Output text uses the same " / " join convention as `clip_pipeline`.
    """
    stanzas = _split_into_stanzas_text(lines)
    has_blank_markers = any(str(l).strip() == "" for l in lines)

    if len(stanzas) == 1 and not has_blank_markers:
        chunks = _chunk_lines_text(lines, fallback_chunk_size)
        return [
            Chunk(chunk_index=i, text=txt, split_mode="fixed")
            for i, txt in enumerate(chunks)
        ]

    return [
        Chunk(chunk_index=i, text=txt, split_mode="stanza")
        for i, txt in enumerate(stanzas)
    ]


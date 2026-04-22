"""dream_chunks — canonical poem chunking for retrieval + LLM analysis.

This module is the single source of truth for splitting a poem's raw
lines into numbered chunks. Both `clip_pipeline.retrieve` and
`llm_analysis.py` should consume the same chunk list so downstream
pairing is explainable and stable.
"""

from .chunking import Chunk, split_poem

__all__ = ["Chunk", "split_poem"]


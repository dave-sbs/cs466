"""CLI-ish tests for scripts/run_dream_9825.py.

These are intentionally CPU-safe: they must not import diffusers or touch CUDA.
"""

from __future__ import annotations

from scripts import run_dream_9825


def test_build_parser_has_expected_flags():
    p = run_dream_9825.build_parser()
    help_text = p.format_help()
    assert "--use-rife" in help_text
    assert "--run-dir" in help_text
    assert "--data-root" in help_text
    assert "--no-mp4" in help_text


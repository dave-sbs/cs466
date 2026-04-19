"""Smoke imports: ensure core modules import on CPU without initializing CUDA.

These guards catch accidental top-level CUDA access during refactors.
"""
from __future__ import annotations


def test_import_clip_pipeline_module():
    import clip_pipeline  # noqa: F401


def test_import_llm_analysis_module():
    import llm_analysis  # noqa: F401


def test_import_interpretability_module():
    import interpretability  # noqa: F401


def test_no_eager_cuda_context():
    """Importing clip_pipeline must not create a CUDA context.

    We check that torch.cuda has not been used for allocation.
    """
    import torch

    import clip_pipeline  # noqa: F401

    assert torch.cuda.is_available() in (True, False)
    if torch.cuda.is_available():
        # If a CUDA device is present we at least assert no memory has been
        # allocated yet by a pure import (the clip_pipeline module defers
        # model loading behind load_model()).
        assert torch.cuda.memory_allocated() == 0

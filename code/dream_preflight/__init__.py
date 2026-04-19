"""dream_preflight — verify dream pipeline inputs before GPU work.

See ``python -m dream_preflight --help``.
"""
from __future__ import annotations

from .checks import PreflightReport, run_preflight

__all__ = ["PreflightReport", "run_preflight"]

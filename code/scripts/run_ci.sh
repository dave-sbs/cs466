#!/usr/bin/env bash
# Run the default CI test suite (CPU-only, no ffmpeg/GPU/network).
# Invoke from any directory; this script cd's into code/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export KMP_DUPLICATE_LIB_OK=TRUE

cd "${CODE_DIR}"
exec pytest tests/ -q -m "not ffmpeg and not gpu and not network and not slow and not colab_manual" "$@"

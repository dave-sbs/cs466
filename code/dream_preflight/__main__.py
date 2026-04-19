"""CLI entrypoint: ``python -m dream_preflight --gutenberg-id 9825 --data-root ...``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .checks import run_preflight


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dream_preflight",
        description=(
            "Validate the local or Drive layout required by the dream "
            "pipeline for one Gutenberg poem ID."
        ),
    )
    p.add_argument(
        "--gutenberg-id",
        type=int,
        required=True,
        help="Gutenberg poem ID to check (e.g. 9825).",
    )
    p.add_argument(
        "--data-root",
        type=str,
        required=True,
        help=(
            "Directory containing exploration_output/ and output/ and "
            "data/images/ subtrees (typically DATA_ROOT in Colab or the "
            "'code/' dir locally)."
        ),
    )
    p.add_argument(
        "--llm-jsonl",
        type=str,
        default=None,
        help=(
            "Override path to llm_analysis.jsonl. Defaults to "
            "<data-root>/exploration_output/llm_analysis.jsonl."
        ),
    )
    p.add_argument(
        "--manifest",
        type=str,
        default=None,
        help=(
            "Override path to retrieval_manifest.json. Defaults to "
            "<data-root>/output/retrieval_results/poem_<id>/retrieval_manifest.json."
        ),
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit a stable JSON report to stdout instead of human text.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    report = run_preflight(
        gutenberg_id=args.gutenberg_id,
        data_root=args.data_root,
        llm_jsonl=args.llm_jsonl,
        manifest_path=args.manifest,
    )

    if args.json:
        # Stable: sorted keys, fixed separators.
        sys.stdout.write(
            json.dumps(report.to_dict(), sort_keys=True, indent=2) + "\n"
        )
    else:
        status = "OK" if report.ok else "FAIL"
        print(f"[preflight:{status}] gutenberg_id={report.gutenberg_id}")
        print(f"  data_root: {report.data_root}")
        print(f"  scenes:    {report.num_scenes}")
        print(f"  chunks:    {report.num_chunks}")
        print(f"  images checked: {report.num_image_paths_checked}")
        for w in report.warnings:
            print(f"  [warn] {w}")
        for e in report.errors:
            print(f"  [err]  {e}")

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

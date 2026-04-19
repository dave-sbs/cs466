"""Batch alignment report over a list of Gutenberg IDs.

Reads alignment JSON artifacts (``exploration_output/pg_raw/alignment_<id>.json``)
and writes a single CSV + JSON summary so reviewers can see which
poems are safe inputs to the dream pipeline.

Usage
-----

    python -m scripts.alignment_report \\
        --ids-csv curation/curated_ids.csv \\
        --pg-raw exploration_output/pg_raw \\
        --out exploration_output/alignment_report

Produces:
    <out>.csv
    <out>.json   (stable sorted-key JSON)

The script intentionally does not re-run the aligner — it only reads
artifacts already written by ``fetch_raw_gutenberg.py``. That keeps it
cheap and deterministic for CI.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class AlignmentRow:
    gutenberg_id: int
    status: str  # "ok" | "failed" | "missing"
    match_rate: float | None
    stanza_count: int | None
    warning: str | None


def load_alignment_json(pg_raw_dir: Path, gid: int) -> dict | None:
    p = pg_raw_dir / f"alignment_{gid}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def summarise(
    gids: Iterable[int], pg_raw_dir: Path
) -> list[AlignmentRow]:
    rows: list[AlignmentRow] = []
    for gid in gids:
        data = load_alignment_json(pg_raw_dir, gid)
        if data is None:
            rows.append(
                AlignmentRow(
                    gutenberg_id=gid,
                    status="missing",
                    match_rate=None,
                    stanza_count=None,
                    warning=f"no alignment_{gid}.json under {pg_raw_dir}",
                )
            )
            continue
        status = data.get("status")
        if status not in ("ok", "failed"):
            status = "failed" if data.get("match_rate", 0) < 0.98 else "ok"
        rows.append(
            AlignmentRow(
                gutenberg_id=gid,
                status=status,
                match_rate=float(data["match_rate"]) if data.get("match_rate") is not None else None,
                stanza_count=int(data["stanza_count"]) if data.get("stanza_count") is not None else None,
                warning=data.get("warning"),
            )
        )
    return rows


def _read_ids_csv(path: Path) -> list[int]:
    ids: list[int] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("gutenberg_id") or row.get("id")
            if raw is None:
                continue
            try:
                ids.append(int(raw))
            except ValueError:
                continue
    return ids


def write_report(rows: list[AlignmentRow], out_prefix: Path) -> tuple[Path, Path]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    json_path = out_prefix.with_suffix(".json")

    # CSV
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gutenberg_id", "status", "match_rate", "stanza_count", "warning"])
        for r in rows:
            w.writerow(
                [
                    r.gutenberg_id,
                    r.status,
                    "" if r.match_rate is None else f"{r.match_rate:.4f}",
                    "" if r.stanza_count is None else r.stanza_count,
                    r.warning or "",
                ]
            )

    # JSON (sorted)
    payload = {"rows": [asdict(r) for r in rows]}
    json_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return csv_path, json_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="alignment_report")
    p.add_argument(
        "--ids-csv",
        type=Path,
        required=True,
        help="CSV with a 'gutenberg_id' column (e.g. curation/curated_ids.csv).",
    )
    p.add_argument(
        "--pg-raw",
        type=Path,
        required=True,
        help="Directory containing alignment_<id>.json (exploration_output/pg_raw).",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path prefix; writes <out>.csv and <out>.json.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only summarise the first N IDs.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.ids_csv.exists():
        print(f"ids_csv not found: {args.ids_csv}", file=sys.stderr)
        return 2
    if not args.pg_raw.exists():
        print(f"pg_raw dir not found: {args.pg_raw}", file=sys.stderr)
        return 2

    gids = _read_ids_csv(args.ids_csv)
    if args.limit is not None:
        gids = gids[: args.limit]

    rows = summarise(gids, args.pg_raw)
    csv_path, json_path = write_report(rows, args.out)

    ok = sum(1 for r in rows if r.status == "ok")
    failed = sum(1 for r in rows if r.status == "failed")
    missing = sum(1 for r in rows if r.status == "missing")
    print(
        f"alignment_report: {len(rows)} ids -> "
        f"{ok} ok, {failed} failed, {missing} missing "
        f"({csv_path}, {json_path})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

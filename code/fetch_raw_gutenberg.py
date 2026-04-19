#!/usr/bin/env python3
"""
fetch_raw_gutenberg.py — Download Project Gutenberg raw .txt files and align
them to parquet lines so blank-line stanza structure can be recovered.

Steps:
    python fetch_raw_gutenberg.py fetch --ids 10469 1545
    python fetch_raw_gutenberg.py align --ids 10469 1545
    python fetch_raw_gutenberg.py all --ids ...   # fetch then align

Defaults --ids to a small validation set when omitted.

Outputs:
    exploration_output/pg_raw/_cache/<id>.txt     — cached PG download
    exploration_output/pg_raw/poem_<id>.txt       — aligned text (blanks preserved)
    exploration_output/pg_raw/alignment_<id>.json — stats and diagnostics
    exploration_output/pg_raw/alignment_summary.csv — after batch align
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

# ─── Paths (relative to code/ when run from code/) ───────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "exploration_output"
PG_RAW_DIR = OUTPUT_DIR / "pg_raw"
CACHE_DIR = PG_RAW_DIR / "_cache"
SHORTLIST_PATH = OUTPUT_DIR / "shortlist.csv"

PARQUET_URL = (
    "hf://datasets/biglam/gutenberg-poetry-corpus/data/"
    "train-00000-of-00001-fa9fb9e1f16eed7e.parquet"
)

USER_AGENT = (
    "CS466-final-project/1.0 (educational; "
    "https://www.gutenberg.org/policy/robot_access.html)"
)

DEFAULT_VALIDATION_IDS = (10469, 1545, 163, 24449, 1974, 1322)

# Alignment: flag likely anthology if book is huge vs parquet line count
# When raw has far more non-blank lines than parquet, alignment may grab the wrong section.
ANTHOLOGY_NONBLANK_RATIO = 2.0
# Known large anthology-style IDs from project validation (defer poem file).
KNOWN_ANTHOLOGY_IDS = frozenset({1322})
MATCH_THRESHOLD = 0.98


def gutenberg_txt_url(gutenberg_id: int) -> str:
    return f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt"


def load_df():
    print("Loading Gutenberg Poetry Corpus parquet...")
    df = pd.read_parquet(PARQUET_URL)
    print(f"  Loaded {len(df):,} rows.\n")
    return df


def strip_project_gutenberg_boilerplate(text: str) -> tuple[str, bool]:
    """
    Return body between START and END markers. If markers missing, return full text.
    """
    start_re = re.compile(
        r"\*\*\*\s*START OF (?:THE )?PROJECT GUTENBERG EBOOK[^*]*\*\*\*",
        re.IGNORECASE,
    )
    end_re = re.compile(
        r"\*\*\*\s*END OF (?:THE )?PROJECT GUTENBERG EBOOK[^*]*\*\*\*",
        re.IGNORECASE,
    )
    m_start = start_re.search(text)
    m_end = end_re.search(text)
    if m_start and m_end and m_end.start() > m_start.end():
        body = text[m_start.end() : m_end.start()]
        return body, True
    return text, False


def normalize_line(s: str) -> str:
    """Lowercase, NFKC, drop punctuation, collapse whitespace — for alignment."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def endpoint_trim(
    raw_lines: list[str],
    parquet_lines: list[str],
) -> tuple[list[str] | None, str | None]:
    """
    Trim raw_lines to cover [first_parquet_line, last_parquet_line] (inclusive).

    Returns (trimmed_lines, reason). If trimmed_lines is None, reason is one of:
      - 'empty_parquet'
      - 'first_line_not_in_raw'
      - 'last_line_not_in_raw'
    """
    if not parquet_lines:
        return None, "empty_parquet"

    first_span, _ = _find_next_parquet_line(raw_lines, 0, parquet_lines[0])
    if first_span is None:
        return None, "first_line_not_in_raw"

    first_start, first_end = first_span

    last_span, _ = _find_next_parquet_line(raw_lines, first_end + 1, parquet_lines[-1])
    if last_span is None:
        return None, "last_line_not_in_raw"

    _, last_end = last_span
    return raw_lines[first_start : last_end + 1], None


def _find_next_parquet_line(
    raw_lines: list[str],
    start_ri: int,
    pq: str,
) -> tuple[tuple[int, int] | None, int]:
    """
    Find inclusive raw line span [start, end] whose joined text matches parquet line pq.

    Corpus lines are a *subsequence* of PG lines (PG may have extra titles/TOC lines).
    Multiple physical PG lines may wrap one corpus row — merge consecutive non-blank
    lines until normalized text matches.
    """
    pn = normalize_line(pq)
    if not pn:
        return None, start_ri

    ri = start_ri
    while ri < len(raw_lines):
        if raw_lines[ri].strip() == "":
            ri += 1
            continue

        buf: list[str] = []
        rj = ri
        while rj < len(raw_lines) and raw_lines[rj].strip() != "":
            buf.append(raw_lines[rj])
            joined = " ".join(s.strip() for s in buf)
            jn = normalize_line(joined)
            if jn == pn:
                return (ri, rj), rj + 1
            if len(buf) == 1 and len(jn) > len(pn) and jn != pn:
                break
            if len(buf) >= 24:
                break
            rj += 1

        ri += 1

    return None, start_ri


def greedy_subsequence_align(
    raw_lines: list[str], parquet_lines: list[str]
) -> tuple[list[tuple[int, int]], list[dict] | None]:
    """Each parquet line -> inclusive (start, end) raw index span."""
    ri = 0
    spans: list[tuple[int, int]] = []

    for qi, pq in enumerate(parquet_lines):
        found, _ = _find_next_parquet_line(raw_lines, ri, pq)
        if found is None:
            return [], [
                {
                    "parquet_index": qi,
                    "reason": "no_raw_match",
                    "parquet_line": pq[:240],
                    "raw_scan_from": ri,
                }
            ]
        a, b = found
        spans.append((a, b))
        ri = b + 1

    return spans, None


def build_output_preserving_blanks(
    raw_lines: list[str], spans: list[tuple[int, int]]
) -> list[str]:
    """Emit matched spans; include only blank raw lines between spans (stanza breaks)."""
    if not spans:
        return []
    out: list[str] = []
    for k, (a, b) in enumerate(spans):
        for line in raw_lines[a : b + 1]:
            out.append(line)
        if k + 1 < len(spans):
            nxt = spans[k + 1][0]
            for r in range(b + 1, nxt):
                if raw_lines[r].strip() == "":
                    out.append(raw_lines[r])
    return out


def find_best_alignment(raw_lines: list[str], parquet_lines: list[str]) -> dict:
    """Align parquet lines to PG raw lines (subsequence match + wrapped lines)."""
    if not parquet_lines:
        return {
            "output_lines": [],
            "matched": 0,
            "total_parquet": 0,
            "match_rate": 0.0,
            "start_raw_index": None,
            "mismatches": [{"reason": "empty_parquet"}],
            "boilerplate_stripped": False,
        }

    spans, errs = greedy_subsequence_align(raw_lines, parquet_lines)
    if errs:
        qi = errs[0]["parquet_index"]
        return {
            "output_lines": [],
            "matched": qi,
            "total_parquet": len(parquet_lines),
            "match_rate": qi / len(parquet_lines) if parquet_lines else 0.0,
            "start_raw_index": None,
            "mismatches": errs,
            "boilerplate_stripped": True,
        }

    out_lines = build_output_preserving_blanks(raw_lines, spans)
    return {
        "output_lines": out_lines,
        "matched": len(parquet_lines),
        "total_parquet": len(parquet_lines),
        "match_rate": 1.0,
        "start_raw_index": spans[0][0] if spans else None,
        "mismatches": [],
        "end_raw_index": spans[-1][1] if spans else None,
        "raw_spans": [[a, b] for a, b in spans],
        "boilerplate_stripped": True,
    }


def count_stanzas(text_lines: list[str]) -> int:
    """Blank-line-separated stanzas (non-empty groups)."""
    stanzas = 0
    current = False
    for line in text_lines:
        if line.strip() == "":
            if current:
                stanzas += 1
                current = False
        else:
            current = True
    if current:
        stanzas += 1
    return stanzas


def fetch_one(
    gutenberg_id: int,
    sleep_s: float = 1.0,
    force: bool = False,
) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{gutenberg_id}.txt"
    meta_path = CACHE_DIR / f"{gutenberg_id}.meta.json"

    url = gutenberg_txt_url(gutenberg_id)
    headers = {"User-Agent": USER_AGENT}

    if cache_path.exists() and not force:
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                lm = meta.get("last_modified")
                etag = meta.get("etag")
                if lm:
                    headers["If-Modified-Since"] = lm
                if etag:
                    headers["If-None-Match"] = etag
            except OSError:
                pass

    req = Request(url, headers=headers, method="GET")
    try:
        with urlopen(req, timeout=60) as resp:
            status = getattr(resp, "status", 200)
            if status == 304:
                print(f"  {gutenberg_id}: cache still valid (304)")
                time.sleep(sleep_s)
                return cache_path
            raw_bytes = resp.read()
            last_mod = resp.headers.get("Last-Modified")
            etag = resp.headers.get("ETag")
    except HTTPError as e:
        if e.code == 304:
            print(f"  {gutenberg_id}: cache still valid (304)")
            time.sleep(sleep_s)
            return cache_path
        raise
    except URLError as e:
        print(f"  ERROR {gutenberg_id}: {e}", file=sys.stderr)
        raise

    text = raw_bytes.decode("utf-8", errors="replace")
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"last_modified": last_mod, "etag": etag, "url": url}, f, indent=2)
    print(f"  {gutenberg_id}: saved {len(text):,} chars -> {cache_path}")
    time.sleep(sleep_s)
    return cache_path


def align_one(
    gutenberg_id: int,
    df: pd.DataFrame | None = None,
    parquet_lines: list[str] | None = None,
) -> dict:
    """Align cached raw text to parquet lines; write poem_<id>.txt and alignment_<id>.json."""
    PG_RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{gutenberg_id}.txt"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache missing for {gutenberg_id}: run fetch first.")

    if parquet_lines is None:
        if df is None:
            df = load_df()
        sub = df[df["gutenberg_id"] == gutenberg_id]
        if sub.empty:
            raise ValueError(f"gutenberg_id {gutenberg_id} not in parquet")
        parquet_lines = sub["line"].astype(str).tolist()

    with open(cache_path, encoding="utf-8", errors="replace") as f:
        full_text = f.read()

    body, stripped = strip_project_gutenberg_boilerplate(full_text)
    raw_lines = body.splitlines()

    non_blank_raw = sum(1 for ln in raw_lines if ln.strip())
    plen = len(parquet_lines)
    # Only defer poem file for known multi-poem / validation IDs (see plan).
    likely_anthology = gutenberg_id in KNOWN_ANTHOLOGY_IDS
    large_book_hint = plen > 500 and non_blank_raw > plen * ANTHOLOGY_NONBLANK_RATIO

    trimmed_lines, trim_reason = endpoint_trim(raw_lines, parquet_lines)
    alignment_input = trimmed_lines if trimmed_lines is not None else raw_lines

    result = find_best_alignment(alignment_input, parquet_lines)
    result["gutenberg_id"] = gutenberg_id
    result["parquet_line_count"] = plen
    result["raw_line_count"] = len(raw_lines)
    result["raw_non_blank_count"] = non_blank_raw
    result["boilerplate_markers_found"] = stripped
    result["likely_anthology"] = likely_anthology
    result["large_book_hint"] = large_book_hint
    result["endpoint_trim_applied"] = trimmed_lines is not None
    result["endpoint_trim_reason"] = trim_reason

    out_lines = result["output_lines"]
    stanza_count = count_stanzas(out_lines) if out_lines else 0
    result["stanza_count"] = stanza_count

    poem_path = PG_RAW_DIR / f"poem_{gutenberg_id}.txt"
    align_path = PG_RAW_DIR / f"alignment_{gutenberg_id}.json"

    if likely_anthology:
        result["warning"] = (
            "likely_anthology: deferring poem_<id>.txt (anthology or oversized book vs parquet). "
            "See alignment JSON for match stats."
        )
        status = "skipped_anthology"
        if poem_path.exists():
            poem_path.unlink()
    elif result["match_rate"] >= MATCH_THRESHOLD and out_lines:
        with open(poem_path, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines))
            if not out_lines[-1].endswith("\n"):
                f.write("\n")
        result["poem_file"] = str(poem_path.relative_to(SCRIPT_DIR))
        status = "ok"
    else:
        if result["match_rate"] < MATCH_THRESHOLD:
            result["warning"] = (
                f"match_rate {result['match_rate']:.4f} < {MATCH_THRESHOLD}; "
                "poem file not written."
            )
        else:
            result["warning"] = "no output lines; poem file not written."
        status = "failed"
        if poem_path.exists():
            poem_path.unlink()

    result["status"] = status

    with open(align_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(
        f"  {gutenberg_id}: match {result['matched']}/{result['total_parquet']} "
        f"({result['match_rate']:.2%}), stanzas={stanza_count}, "
        f"likely_anthology={likely_anthology} -> {status}"
    )
    if result.get("warning"):
        print(f"    {result['warning']}")

    return result


def read_shortlist_ids() -> list[int]:
    if not SHORTLIST_PATH.exists():
        return []
    ids = []
    with open(SHORTLIST_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row["gutenberg_id"]))
    return ids


def read_shortlist_ids_by_line_count(max_lines: int = 500) -> list[int]:
    if not SHORTLIST_PATH.exists():
        return []
    ids: list[int] = []
    with open(SHORTLIST_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                gid = int(row["gutenberg_id"])
                lc = int(row["line_count"])
            except Exception:
                continue
            if 1 <= lc <= max_lines:
                ids.append(gid)
    return ids


def cmd_fetch(args):
    for gid in args.ids:
        fetch_one(gid, sleep_s=args.sleep, force=args.force)


def cmd_align(args):
    df = load_df()

    summary_rows = []
    for gid in args.ids:
        try:
            r = align_one(gid, df=df)
            summary_rows.append(
                {
                    "gutenberg_id": gid,
                    "match_rate": f"{r['match_rate']:.6f}",
                    "matched": r["matched"],
                    "total_parquet": r["total_parquet"],
                    "stanza_count": r.get("stanza_count", 0),
                    "likely_anthology": r.get("likely_anthology", False),
                    "status": r.get("status", ""),
                }
            )
        except Exception as e:
            print(f"  ERROR {gid}: {e}", file=sys.stderr)
            summary_rows.append(
                {
                    "gutenberg_id": gid,
                    "match_rate": "",
                    "matched": "",
                    "total_parquet": "",
                    "stanza_count": "",
                    "likely_anthology": "",
                    "status": f"error: {e}",
                }
            )

    if getattr(args, "summary", True) and summary_rows:
        PG_RAW_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = PG_RAW_DIR / "alignment_summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "gutenberg_id",
                    "match_rate",
                    "matched",
                    "total_parquet",
                    "stanza_count",
                    "likely_anthology",
                    "status",
                ],
            )
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nWrote {summary_path}")


def cmd_diagnose(args):
    """
    Diagnose stanza-preserving alignment for shortlist IDs with parquet line_count <= max_lines.

    Writes:
      - exploration_output/pg_raw/curated_ids.csv
      - exploration_output/pg_raw/skipped.csv
    """
    df = load_df()
    ids = read_shortlist_ids_by_line_count(max_lines=args.max_lines)
    if not ids:
        print(f"No shortlist IDs found with line_count <= {args.max_lines}.", file=sys.stderr)
        sys.exit(1)

    curated_rows: list[dict] = []
    skipped_rows: list[dict] = []

    for gid in ids:
        try:
            try:
                fetch_one(gid, sleep_s=args.sleep, force=False)
            except Exception as e:
                skipped_rows.append(
                    {"gutenberg_id": gid, "reason": "fetch_error", "detail": str(e)}
                )
                continue

            r = align_one(gid, df=df)

            if r.get("status") != "ok":
                trim_reason = r.get("endpoint_trim_reason")
                if trim_reason in ("first_line_not_in_raw", "last_line_not_in_raw"):
                    reason = trim_reason
                else:
                    reason = "low_match_rate"

                skipped_rows.append(
                    {
                        "gutenberg_id": gid,
                        "reason": reason,
                        "match_rate": f"{r.get('match_rate', 0.0):.6f}",
                        "stanza_count": r.get("stanza_count", ""),
                        "parquet_line_count": r.get("parquet_line_count", ""),
                        "endpoint_trim_reason": trim_reason or "",
                        "large_book_hint": r.get("large_book_hint", ""),
                        "status": r.get("status", ""),
                        "detail": r.get("warning", ""),
                    }
                )
                continue

            if r.get("stanza_count", 0) < 2:
                skipped_rows.append(
                    {
                        "gutenberg_id": gid,
                        "reason": "single_stanza",
                        "match_rate": f"{r.get('match_rate', 0.0):.6f}",
                        "stanza_count": r.get("stanza_count", ""),
                        "parquet_line_count": r.get("parquet_line_count", ""),
                        "endpoint_trim_reason": r.get("endpoint_trim_reason", "") or "",
                        "large_book_hint": r.get("large_book_hint", ""),
                        "status": r.get("status", ""),
                        "detail": "",
                    }
                )
                continue

            curated_rows.append(
                {
                    "gutenberg_id": gid,
                    "match_rate": f"{r.get('match_rate', 0.0):.6f}",
                    "stanza_count": r.get("stanza_count", ""),
                    "parquet_line_count": r.get("parquet_line_count", ""),
                }
            )
        except Exception as e:
            skipped_rows.append(
                {"gutenberg_id": gid, "reason": "error", "detail": str(e)}
            )

    PG_RAW_DIR.mkdir(parents=True, exist_ok=True)
    curated_path = PG_RAW_DIR / "curated_ids.csv"
    skipped_path = PG_RAW_DIR / "skipped.csv"

    with open(curated_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["gutenberg_id", "match_rate", "stanza_count", "parquet_line_count"],
        )
        w.writeheader()
        w.writerows(curated_rows)

    skipped_fields = [
        "gutenberg_id",
        "reason",
        "match_rate",
        "stanza_count",
        "parquet_line_count",
        "endpoint_trim_reason",
        "large_book_hint",
        "status",
        "detail",
    ]
    with open(skipped_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=skipped_fields)
        w.writeheader()
        for row in skipped_rows:
            w.writerow({k: row.get(k, "") for k in skipped_fields})

    print(f"\nWrote {curated_path} ({len(curated_rows)} passing IDs)")
    print(f"Wrote {skipped_path} ({len(skipped_rows)} skipped IDs)")


def cmd_all(args):
    cmd_fetch(args)
    cmd_align(args)


def main():
    parser = argparse.ArgumentParser(description="Fetch and align Project Gutenberg raw texts.")
    sub = parser.add_subparsers(dest="step", required=True)

    def add_id_args(p):
        p.add_argument(
            "--ids",
            type=int,
            nargs="*",
            default=None,
            help="Gutenberg IDs (default: validation set)",
        )
        p.add_argument(
            "--shortlist",
            action="store_true",
            help="Use gutenberg_id column from exploration_output/shortlist.csv",
        )
        p.add_argument("--sleep", type=float, default=1.0, help="Seconds between fetches")
        p.add_argument(
            "--force",
            action="store_true",
            help="Re-download even if cache exists",
        )

    p_fetch = sub.add_parser("fetch", help="Download PG .txt into cache")
    add_id_args(p_fetch)
    p_fetch.set_defaults(func=cmd_fetch)

    p_align = sub.add_parser("align", help="Align cached text to parquet")
    add_id_args(p_align)
    p_align.add_argument(
        "--summary",
        action="store_true",
        default=True,
        help="Write alignment_summary.csv (default: on)",
    )
    p_align.add_argument(
        "--no-summary",
        dest="summary",
        action="store_false",
        help="Do not write alignment_summary.csv",
    )
    p_align.set_defaults(func=cmd_align)

    p_all = sub.add_parser("all", help="fetch then align")
    add_id_args(p_all)
    p_all.add_argument(
        "--no-summary",
        dest="summary",
        action="store_false",
        default=True,
        help="Do not write alignment_summary.csv",
    )
    p_all.set_defaults(func=cmd_all)

    p_diag = sub.add_parser(
        "diagnose",
        help="Curate stanza-preserving poems from shortlist (<=500 parquet lines)",
    )
    p_diag.add_argument(
        "--max_lines",
        type=int,
        default=500,
        help="Max parquet line_count to include (default: 500)",
    )
    p_diag.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds between fetches (default: 1.0)",
    )
    p_diag.set_defaults(func=cmd_diagnose)

    args = parser.parse_args()

    if hasattr(args, "ids") and args.ids is None:
        if getattr(args, "shortlist", False):
            args.ids = read_shortlist_ids()
            if not args.ids:
                print("shortlist.csv missing or empty; using validation IDs.", file=sys.stderr)
                args.ids = list(DEFAULT_VALIDATION_IDS)
        else:
            args.ids = list(DEFAULT_VALIDATION_IDS)

    if args.step != "diagnose":
        if not getattr(args, "ids", None):
            print("No IDs to process.", file=sys.stderr)
            sys.exit(1)

    # Ensure summary for align/all unless disabled
    args.func(args)


if __name__ == "__main__":
    main()

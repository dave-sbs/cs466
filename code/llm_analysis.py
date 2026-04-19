"""
llm_analysis.py — LLM-based poem analysis and preprocessing via OpenRouter.

Sends poems to Gemini Flash for thematic tagging, non-poem detection,
visual scene extraction, mood arc mapping, and quality assessment.

Usage:
    python llm_analysis.py --step analyze     # Analyze shortlisted poems via Gemini Flash
    python llm_analysis.py --step summarize   # Generate CSV summary from JSONL results
    python llm_analysis.py --step report      # Print analysis statistics
    python llm_analysis.py                    # Run all steps
    python llm_analysis.py --gutenberg-id 9825 --step analyze   # Analyze one poem (local text or HF corpus)
    python llm_analysis.py --gutenberg-id 9825 --step summarize  # Print stored analysis for that ID

Options:
    --source shortlist|catalog   Which poem set to process (default: shortlist)
    --limit N                    Process at most N poems (for testing)
    --delay SECONDS              Delay between API calls (default: 0.5)
    --model MODEL                Override model (see DEFAULT_MODEL in code)
    --max-lines N                Max poem lines to send to LLM (default: 200)
    --gutenberg-id ID            Analyze or print summary for a single poem
    --force                      With --gutenberg-id and analyze: re-call API even if JSONL has this ID
"""

import os
import json
import csv
import time
import argparse
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path

import requests
import pandas as pd
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

load_dotenv()


# ─── Configuration ────────────────────────────────────────────────────────────

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3-flash-preview"
DEFAULT_DELAY = 0.5
DEFAULT_MAX_LINES = 200
MAX_RETRIES = 3

OUTPUT_DIR = "exploration_output"
JSONL_PATH = os.path.join(OUTPUT_DIR, "llm_analysis.jsonl")
SUMMARY_CSV_PATH = os.path.join(OUTPUT_DIR, "llm_analysis_summary.csv")
SHORTLIST_PATH = os.path.join(OUTPUT_DIR, "shortlist.csv")
CATALOG_PATH = os.path.join(OUTPUT_DIR, "corpus_catalog.csv")

# Same layout as clip_pipeline.load_poem_lines — prefer aligned pg_raw text when present.
PG_RAW_POEM_DIR = Path("exploration_output/pg_raw")
SAMPLES_STRATIFIED_DIR = Path("exploration_output/samples_stratified")
SAMPLES_DIR = Path("samples")


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class VisualScene(BaseModel):
    stanza_index: int
    scene_description: str
    dominant_colors: list[str]
    time_of_day: str
    season: str


class MoodPoint(BaseModel):
    position: str
    mood: str
    intensity: int


class PoemAnalysis(BaseModel):
    is_poem: bool
    content_type: str
    content_type_rationale: str
    title: str
    author: str
    genre: str
    is_collection: bool
    themes: list[str]
    primary_theme: str
    visual_scenes: list[VisualScene]
    mood_arc: list[MoodPoint]
    overall_mood: str
    nature_categories: list[str]
    primary_nature_setting: str
    language: str
    ocr_artifacts_detected: bool
    has_non_poem_content: bool
    non_poem_content_types: list[str]
    visualization_suitability: int
    visualization_rationale: str
    most_visual_stanzas: list[int]
    notable_lines: list[str]


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_df():
    """Load the Gutenberg Poetry Corpus from HuggingFace."""
    print("Loading Gutenberg Poetry Corpus from Hugging Face...")
    df = pd.read_parquet(
        "hf://datasets/biglam/gutenberg-poetry-corpus/data/"
        "train-00000-of-00001-fa9fb9e1f16eed7e.parquet"
    )
    print(f"  Loaded {len(df):,} lines across {df['gutenberg_id'].nunique()} poems.\n")
    return df


def build_poem_lookup(df: pd.DataFrame) -> dict[int, list[str]]:
    """Pre-group the dataframe by gutenberg_id for fast access."""
    return {gid: group["line"].tolist() for gid, group in df.groupby("gutenberg_id")}


def load_poem_lines_single(gutenberg_id: int) -> tuple[list[str], str]:
    """
    Load poem lines for one ID. Prefer aligned pg_raw, then stratified samples, then samples/.

    Returns (lines, source_label).
    """
    pg_path = PG_RAW_POEM_DIR / f"poem_{gutenberg_id}.txt"
    if pg_path.exists():
        with open(pg_path, encoding="utf-8") as f:
            return f.read().splitlines(), "pg_raw"

    for folder, label in (
        (SAMPLES_STRATIFIED_DIR, "samples_stratified"),
        (SAMPLES_DIR, "samples"),
    ):
        p = folder / f"poem_{gutenberg_id}.txt"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                lines = [line for line in f.read().splitlines() if line.strip()]
            return lines, label

    print("  No local poem file; loading corpus parquet (one ID)...")
    df = load_df()
    sub = df[df["gutenberg_id"] == gutenberg_id]
    if sub.empty:
        raise SystemExit(
            f"gutenberg_id {gutenberg_id} not found in HuggingFace corpus parquet "
            "and no local poem_{id}.txt under exploration_output/pg_raw or samples."
        )
    lines = sub["line"].astype(str).tolist()
    return lines, "hf_parquet"


def load_last_jsonl_record_for_id(jsonl_path: str, gutenberg_id: int) -> dict | None:
    """Return the last JSON object in the JSONL file for this gutenberg_id, or None."""
    if not os.path.exists(jsonl_path):
        return None
    last: dict | None = None
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("gutenberg_id") == gutenberg_id:
                last = record
    return last


def print_poem_analysis_summary(record: dict) -> None:
    """Print a readable summary of one analysis record (stdout)."""
    gid = record.get("gutenberg_id", "?")
    if record.get("llm_parse_error"):
        print(f"\n  Poem {gid}: parse error — {record.get('error', 'unknown')}\n")
        if record.get("raw_response"):
            print(f"  Raw (truncated): {str(record['raw_response'])[:500]}...\n")
        return

    print(f"\n  === LLM summary: Gutenberg ID {gid} ===\n")
    print(f"  Title:    {record.get('title', '')}")
    print(f"  Author:   {record.get('author', '')}")
    print(f"  Type:     {record.get('content_type', '')}  (is_poem={record.get('is_poem')})")
    print(f"  Genre:    {record.get('genre', '')}")
    print(f"  Theme:    {record.get('primary_theme', '')}")
    themes = record.get("themes") or []
    if themes:
        print(f"  Themes:   {', '.join(str(t) for t in themes)}")
    print(f"  Nature:   {record.get('primary_nature_setting', '')}")
    cats = record.get("nature_categories") or []
    if cats:
        print(f"            ({', '.join(str(c) for c in cats)})")
    print(f"  Mood:     {record.get('overall_mood', '')}")
    print(f"  Language: {record.get('language', '')}")
    print(f"  Viz score:{record.get('visualization_suitability', '')}/5 — {record.get('visualization_rationale', '')}")
    rationale = record.get("content_type_rationale") or ""
    if rationale:
        print(f"  Rationale:{rationale}")
    scenes = record.get("visual_scenes") or []
    if scenes:
        print("\n  Visual scenes (by stanza):")
        for s in scenes[:12]:
            idx = s.get("stanza_index", "?")
            desc = (s.get("scene_description") or "")[:200]
            cols = s.get("dominant_colors") or []
            print(f"    [{idx}] {desc}{'...' if len(str(s.get('scene_description', ''))) > 200 else ''}")
            if cols:
                print(f"         colors: {', '.join(str(c) for c in cols)}")
        if len(scenes) > 12:
            print(f"    ... ({len(scenes) - 12} more stanzas)")
    notable = record.get("notable_lines") or []
    if notable:
        print("\n  Notable lines:")
        for ln in notable[:8]:
            print(f"    • {ln}")
    print(f"\n  Model: {record.get('model', '')}  |  lines sent context: {record.get('line_count', '')}\n")


# ─── HTTP Client ──────────────────────────────────────────────────────────────

def get_api_key() -> str:
    """Read the OpenRouter API key from environment."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        print(
            "ERROR: OPENROUTER_API_KEY environment variable not set.\n"
            "  export OPENROUTER_API_KEY=<your-key>\n"
            "  Get a key at https://openrouter.ai/keys"
        )
        raise SystemExit(1)
    return key


def call_openrouter(
    messages: list[dict[str, str]],
    model: str,
    api_key: str,
) -> str:
    """Send a chat completion request to OpenRouter and return the response content."""
    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "Pangyo Poetry Pipeline",
        },
        json={
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


# ─── Prompt Construction ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a literary analyst specializing in poetry for visual media adaptation.
Given a poem excerpt, produce a JSON object with EXACTLY this structure (no extra keys):
{
  "is_poem": true or false,
  "content_type": "poem | collection | table_of_contents | preface | biography | disclaimer | index | mixed | other",
  "content_type_rationale": "1 sentence explaining classification",
  "title": "best guess at poem/collection title, or 'Unknown'",
  "author": "best guess at author, or 'Unknown'",
  "genre": "sonnet | ballad | ode | epic | elegy | lyric | narrative | hymn | free_verse | dramatic | other",
  "is_collection": true or false,
  "themes": ["theme1", "theme2"],
  "primary_theme": "single most dominant theme",
  "visual_scenes": [
    {
      "stanza_index": 0,
      "scene_description": "A CLIP-friendly visual description: concrete, photographic, no abstractions. Describe what a camera would see.",
      "dominant_colors": ["blue", "grey"],
      "time_of_day": "dawn | morning | midday | afternoon | dusk | evening | night | unspecified",
      "season": "spring | summer | autumn | winter | unspecified"
    }
  ],
  "mood_arc": [
    {"position": "opening", "mood": "melancholic", "intensity": 3},
    {"position": "middle", "mood": "hopeful", "intensity": 4},
    {"position": "closing", "mood": "serene", "intensity": 2}
  ],
  "overall_mood": "single word",
  "nature_categories": ["ocean", "forest", "mountain"],
  "primary_nature_setting": "the single most relevant nature category",
  "language": "English | archaic_English | dialect | non_English | mixed",
  "ocr_artifacts_detected": true or false,
  "has_non_poem_content": true or false,
  "non_poem_content_types": ["disclaimer", "advertisement", "table_of_contents"],
  "visualization_suitability": 1-5,
  "visualization_rationale": "1-2 sentences",
  "most_visual_stanzas": [0, 3, 5],
  "notable_lines": ["line1", "line2"]
}

Guidelines:
- For visual_scenes, create one entry per stanza (or per major section if the poem is long).
  Write scene_description as if describing a photograph — concrete objects, lighting, colors.
  Example: "A dark pine forest at twilight with mist rising between ancient trunks, faint moonlight filtering through branches"
- For mood_arc, always include exactly 3 entries: opening, middle, closing. Intensity is 1-5.
- For nature_categories, use specific categories: ocean, sea, river, lake, mountain, forest, meadow, desert, garden, sky, cliff, cave, island, snow, storm, field, swamp, prairie, shore, valley.
- For most_visual_stanzas, list the indices (0-based) of the 1-3 most visually evocative stanzas.
- If the text is NOT a poem (e.g. table of contents, preface, disclaimer), still fill all fields with best-effort values but set is_poem=false and content_type accordingly.

Respond ONLY with valid JSON."""


def build_messages(lines: list[str], gid: int, line_count: int, max_lines: int) -> list[dict[str, str]]:
    """Build the system + user message pair for a poem."""
    truncated = lines[:max_lines]
    poem_text = "\n".join(truncated)
    note = f" (showing first {max_lines} of {line_count} lines)" if line_count > max_lines else ""

    user_msg = (
        f"Analyze this poem (Gutenberg ID: {gid}, {line_count} lines{note}).\n"
        f"Focus on visual/cinematic potential for nature-themed video visualization.\n\n"
        f"{poem_text}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


# ─── LLM Call + Validation ────────────────────────────────────────────────────

def analyze_poem(
    lines: list[str],
    gid: int,
    line_count: int,
    model: str,
    api_key: str,
    max_lines: int,
) -> dict:
    """Call the LLM for a single poem. Returns a dict (validated or error-marked)."""
    messages = build_messages(lines, gid, line_count, max_lines)

    for attempt in range(MAX_RETRIES):
        try:
            raw_content = call_openrouter(messages, model, api_key)
            parsed = json.loads(raw_content)
            analysis = PoemAnalysis.model_validate(parsed)
            result = analysis.model_dump()
            result["gutenberg_id"] = gid
            result["line_count"] = line_count
            result["model"] = model
            result["timestamp"] = datetime.now(timezone.utc).isoformat()
            result["llm_parse_error"] = False
            return result

        except ValidationError as e:
            print(f"    Validation error (attempt {attempt + 1}): {e.error_count()} issues")
            if attempt == MAX_RETRIES - 1:
                return {
                    "gutenberg_id": gid,
                    "line_count": line_count,
                    "model": model,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "llm_parse_error": True,
                    "error": f"ValidationError: {e.error_count()} issues",
                    "raw_response": raw_content,
                }

        except json.JSONDecodeError as e:
            print(f"    JSON decode error (attempt {attempt + 1}): {e}")
            if attempt == MAX_RETRIES - 1:
                return {
                    "gutenberg_id": gid,
                    "line_count": line_count,
                    "model": model,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "llm_parse_error": True,
                    "error": f"JSONDecodeError: {e}",
                    "raw_response": raw_content if "raw_content" in dir() else "",
                }

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status == 429:
                wait = 2 ** (attempt + 1)
                print(f"    Rate limited (429). Waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"    HTTP error (attempt {attempt + 1}): {e}")
            if attempt == MAX_RETRIES - 1:
                return {
                    "gutenberg_id": gid,
                    "line_count": line_count,
                    "model": model,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "llm_parse_error": True,
                    "error": f"HTTPError: {status} {e}",
                }

        except requests.exceptions.ConnectionError as e:
            wait = 2 ** (attempt + 1)
            print(f"    Connection error (attempt {attempt + 1}). Waiting {wait}s...")
            time.sleep(wait)
            if attempt == MAX_RETRIES - 1:
                return {
                    "gutenberg_id": gid,
                    "line_count": line_count,
                    "model": model,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "llm_parse_error": True,
                    "error": f"ConnectionError: {e}",
                }

        # Backoff before retry on validation/JSON errors
        time.sleep(1)

    # Should not reach here, but just in case
    return {
        "gutenberg_id": gid,
        "llm_parse_error": True,
        "error": "Exhausted retries",
    }


# ─── Resumability ─────────────────────────────────────────────────────────────

def load_processed_ids(jsonl_path: str) -> set[int]:
    """Load set of already-processed gutenberg_ids from the JSONL manifest."""
    if not os.path.exists(jsonl_path):
        return set()
    ids = set()
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                ids.add(record["gutenberg_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def append_result(jsonl_path: str, result: dict) -> None:
    """Append a single analysis result as a JSONL line."""
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


# ─── Step: Analyze ────────────────────────────────────────────────────────────

def run_analysis(
    source: str,
    limit: int,
    delay: float,
    model: str,
    max_lines: int,
) -> None:
    """Run LLM analysis on shortlisted or cataloged poems."""
    print("=== LLM Analysis ===\n")

    # Load poem IDs from source
    if source == "shortlist":
        if not os.path.exists(SHORTLIST_PATH):
            print(f"  No shortlist at {SHORTLIST_PATH}. Run explore_corpus.py --step shortlist first.\n")
            return
        source_df = pd.read_csv(SHORTLIST_PATH)
        print(f"  Source: shortlist ({len(source_df)} poems)")
    else:
        if not os.path.exists(CATALOG_PATH):
            print(f"  No catalog at {CATALOG_PATH}. Run explore_corpus.py --step catalog first.\n")
            return
        source_df = pd.read_csv(CATALOG_PATH)
        print(f"  Source: catalog ({len(source_df)} poems)")

    api_key = get_api_key()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Resumability: skip already-processed poems
    processed = load_processed_ids(JSONL_PATH)
    if processed:
        print(f"  Resuming: {len(processed)} poems already analyzed.")

    # Build work list
    poem_ids = source_df["gutenberg_id"].tolist()
    line_counts = dict(zip(source_df["gutenberg_id"], source_df["line_count"]))
    pending = [gid for gid in poem_ids if gid not in processed]

    if limit and limit > 0:
        pending = pending[:limit]

    if not pending:
        print("  All poems already analyzed. Nothing to do.\n")
        return

    print(f"  Poems to analyze: {len(pending)}")
    print(f"  Model: {model}")
    print(f"  Delay: {delay}s between calls\n")

    # Load poem text from HuggingFace
    df = load_df()
    print("Building poem lookup table...")
    poems = build_poem_lookup(df)
    del df
    print(f"  {len(poems)} poems indexed.\n")

    errors = 0
    for i, gid in enumerate(pending):
        lines = poems.get(gid, [])
        lc = line_counts.get(gid, len(lines))

        if not lines:
            print(f"  [{i+1}/{len(pending)}] Poem {gid}: no text found, skipping.")
            continue

        print(f"  [{i+1}/{len(pending)}] Poem {gid} ({lc} lines)...", end=" ", flush=True)

        result = analyze_poem(lines, gid, lc, model, api_key, max_lines)
        append_result(JSONL_PATH, result)

        if result.get("llm_parse_error"):
            print(f"ERROR: {result.get('error', 'unknown')}")
            errors += 1
        else:
            print(
                f"ok  type={result['content_type']}  "
                f"theme={result['primary_theme']}  "
                f"viz={result['visualization_suitability']}/5"
            )

        if i < len(pending) - 1:
            time.sleep(delay)

    print(f"\n  Done. Results appended to {JSONL_PATH}")
    if errors:
        print(f"  {errors} poems had parse errors (see llm_parse_error=true in JSONL).")
    print()


def run_single_poem_analysis(
    gutenberg_id: int,
    model: str,
    max_lines: int,
    delay: float,
    force: bool,
) -> None:
    """Run LLM analysis for one gutenberg_id; append to JSONL; print a readable summary."""
    print(f"=== Single-poem LLM analysis (gutenberg_id={gutenberg_id}) ===\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not force:
        cached = load_last_jsonl_record_for_id(JSONL_PATH, gutenberg_id)
        if cached is not None:
            print("  Found existing analysis in JSONL (use --force to re-call the API).\n")
            print_poem_analysis_summary(cached)
            return

    api_key = get_api_key()
    lines, source = load_poem_lines_single(gutenberg_id)
    line_count = len(lines)
    print(f"  Loaded {line_count} lines (source={source}).\n")

    result = analyze_poem(lines, gutenberg_id, line_count, model, api_key, max_lines)
    append_result(JSONL_PATH, result)

    if result.get("llm_parse_error"):
        print(f"  ERROR: {result.get('error', 'unknown')}\n")
    else:
        print("  Appended to JSONL.\n")
        print_poem_analysis_summary(result)

    if delay and delay > 0:
        time.sleep(delay)


def run_summarize_single_poem(gutenberg_id: int) -> None:
    """Print stored LLM analysis for one ID from the JSONL manifest."""
    print(f"=== Summary for poem {gutenberg_id} ===\n")
    record = load_last_jsonl_record_for_id(JSONL_PATH, gutenberg_id)
    if record is None:
        print(
            f"  No analysis found for gutenberg_id {gutenberg_id} in {JSONL_PATH}.\n"
            f"  Run: python llm_analysis.py --gutenberg-id {gutenberg_id} --step analyze\n"
        )
        return
    print_poem_analysis_summary(record)


def run_report_single_poem(gutenberg_id: int) -> None:
    """Print a short report for one poem if present in the JSONL results."""
    print(f"=== Report: poem {gutenberg_id} ===\n")
    record = load_last_jsonl_record_for_id(JSONL_PATH, gutenberg_id)
    if record is None:
        print(f"  No record for {gutenberg_id} in {JSONL_PATH}.\n")
        return
    if record.get("llm_parse_error"):
        print("  Status: parse error (see JSONL line for details).\n")
        return
    print(f"  content_type={record.get('content_type')}  viz={record.get('visualization_suitability')}/5")
    print(f"  primary_theme={record.get('primary_theme')}")
    print(f"  primary_nature_setting={record.get('primary_nature_setting')}\n")


# ─── Step: Summarize ──────────────────────────────────────────────────────────

SUMMARY_COLUMNS = [
    "gutenberg_id", "line_count", "is_poem", "content_type", "title", "author",
    "genre", "is_collection", "primary_theme", "themes", "primary_nature_setting",
    "nature_categories", "overall_mood", "visualization_suitability", "language",
    "has_non_poem_content", "ocr_artifacts_detected", "llm_parse_error", "model",
]


def run_summarize() -> None:
    """Flatten the JSONL analysis results into a CSV for easy pandas integration."""
    print("=== Generating Summary CSV ===\n")

    if not os.path.exists(JSONL_PATH):
        print(f"  No analysis results at {JSONL_PATH}. Run --step analyze first.\n")
        return

    rows = []
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            row = {}
            for col in SUMMARY_COLUMNS:
                val = record.get(col)
                if isinstance(val, list):
                    val = " | ".join(str(v) for v in val)
                elif isinstance(val, bool):
                    val = val
                row[col] = val
            rows.append(row)

    if not rows:
        print("  No valid records found.\n")
        return

    with open(SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Wrote {len(rows)} rows to {SUMMARY_CSV_PATH}\n")


# ─── Step: Report ─────────────────────────────────────────────────────────────

def run_report() -> None:
    """Print statistics from the analysis results."""
    print("=== Analysis Report ===\n")

    if not os.path.exists(JSONL_PATH):
        print(f"  No analysis results at {JSONL_PATH}. Run --step analyze first.\n")
        return

    records = []
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    total = len(records)
    errors = sum(1 for r in records if r.get("llm_parse_error"))
    valid = [r for r in records if not r.get("llm_parse_error")]

    print(f"  Total analyzed: {total}")
    print(f"  Parse errors:   {errors}")
    print(f"  Valid results:  {len(valid)}\n")

    if not valid:
        return

    # Content type distribution
    print("  Content Type Distribution:")
    ct_counts = Counter(r.get("content_type", "unknown") for r in valid)
    for ct, count in ct_counts.most_common():
        print(f"    {ct:>20}: {count:>4}  ({100*count/len(valid):.1f}%)")
    print()

    # Non-poem content
    non_poem = [r for r in valid if not r.get("is_poem", True)]
    print(f"  Non-poem content flagged: {len(non_poem)}/{len(valid)}")
    if non_poem:
        for r in non_poem[:10]:
            print(f"    ID {r['gutenberg_id']:>6}: {r.get('content_type', '?')} — {r.get('content_type_rationale', '')[:60]}")
    print()

    # Theme distribution
    print("  Top 15 Themes:")
    theme_counts = Counter()
    for r in valid:
        for t in r.get("themes", []):
            theme_counts[t] += 1
    for theme, count in theme_counts.most_common(15):
        print(f"    {theme:>25}: {count:>4}")
    print()

    # Nature category distribution
    print("  Nature Category Distribution:")
    cat_counts = Counter()
    for r in valid:
        for c in r.get("nature_categories", []):
            cat_counts[c] += 1
    for cat, count in cat_counts.most_common():
        bar = "█" * (count * 40 // max(cat_counts.values())) if cat_counts else ""
        print(f"    {cat:>15}: {count:>4}  {bar}")
    print()

    # Visualization suitability histogram
    print("  Visualization Suitability (1-5):")
    viz_counts = Counter(r.get("visualization_suitability", 0) for r in valid)
    for score in range(1, 6):
        count = viz_counts.get(score, 0)
        bar = "█" * (count * 40 // max(max(viz_counts.values(), default=1), 1))
        print(f"    {score}: {count:>4}  {bar}")
    print()

    # Language issues
    lang_counts = Counter(r.get("language", "unknown") for r in valid)
    print("  Language Distribution:")
    for lang, count in lang_counts.most_common():
        print(f"    {lang:>20}: {count:>4}")

    ocr_issues = sum(1 for r in valid if r.get("ocr_artifacts_detected"))
    if ocr_issues:
        print(f"\n  OCR artifacts detected in {ocr_issues} poems.")

    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM-based poem analysis and preprocessing via OpenRouter.",
    )
    parser.add_argument(
        "--step",
        choices=["analyze", "summarize", "report", "all"],
        default="all",
        help="Pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--source",
        choices=["shortlist", "catalog"],
        default="shortlist",
        help="Which poem set to process (default: shortlist)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N poems; 0 = all (default: 0)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Delay in seconds between API calls (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenRouter model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=DEFAULT_MAX_LINES,
        help=f"Max poem lines to send to LLM (default: {DEFAULT_MAX_LINES})",
    )
    parser.add_argument(
        "--gutenberg-id",
        type=int,
        default=None,
        metavar="ID",
        help="Analyze or print summary for a single poem (bypasses shortlist/catalog)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --gutenberg-id: re-call the API even if this ID already exists in JSONL",
    )
    args = parser.parse_args()

    single_id = args.gutenberg_id

    if single_id is not None:
        if args.step in ("analyze", "all"):
            run_single_poem_analysis(
                gutenberg_id=single_id,
                model=args.model,
                max_lines=args.max_lines,
                delay=args.delay,
                force=args.force,
            )
        if args.step == "summarize":
            run_summarize_single_poem(single_id)
            run_summarize()
        elif args.step == "all":
            run_summarize()
        if args.step == "report":
            run_report_single_poem(single_id)
        elif args.step == "all":
            run_report()
    else:
        if args.step in ("analyze", "all"):
            run_analysis(
                source=args.source,
                limit=args.limit,
                delay=args.delay,
                model=args.model,
                max_lines=args.max_lines,
            )

        if args.step in ("summarize", "all"):
            run_summarize()

        if args.step in ("report", "all"):
            run_report()

    print("Done.")


if __name__ == "__main__":
    main()

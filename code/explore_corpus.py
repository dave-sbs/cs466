"""
explore_corpus.py — Systematic exploration of the Gutenberg Poetry Corpus.

Usage:
    python explore_corpus.py --step sample      # Stratified sampling across length buckets
    python explore_corpus.py --step catalog     # Build metadata catalog for all poems
    python explore_corpus.py --step prompts     # Generate LLM analysis prompts for samples
    python explore_corpus.py --step plots       # Generate visualizations
    python explore_corpus.py --step shortlist   # Filter candidate poems for the project
    python explore_corpus.py                    # Run all steps sequentially
"""

import os
import json
import csv
import random
import re
import argparse
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constants import POEM_LEN_DICT


# ─── Configuration ────────────────────────────────────────────────

BUCKETS = [
    (1, 50),
    (51, 100),
    (101, 250),
    (251, 500),
    (501, 1000),
    (1001, 2500),
    (2501, 5000),
    (5001, 10000),
    (10001, 60000),
]

BUCKET_LABELS = [f"{lo}-{hi}" for lo, hi in BUCKETS]

NATURE_IMAGERY = {
    "sky", "sun", "moon", "star", "stars", "cloud", "clouds", "rain", "wind",
    "storm", "thunder", "lightning", "snow", "ice", "frost", "fog", "mist",
    "sea", "ocean", "wave", "waves", "river", "stream", "lake", "pond",
    "mountain", "hill", "valley", "cliff", "rock", "stone", "cave",
    "forest", "wood", "woods", "tree", "trees", "leaf", "leaves", "branch",
    "flower", "flowers", "rose", "lily", "bloom", "blossom", "garden",
    "field", "meadow", "grass", "earth", "ground", "soil", "dust",
    "fire", "flame", "smoke", "ash", "ember",
    "bird", "eagle", "hawk", "dove", "swan", "nightingale", "lark",
    "dawn", "dusk", "twilight", "sunset", "sunrise", "morning", "evening",
    "night", "shadow", "shadows", "light", "darkness", "dark",
    "spring", "summer", "autumn", "winter", "season",
    "shore", "sand", "tide", "harbor", "harbour",
    "wilderness", "desert", "plain", "plains", "horizon",
}

ARCHAIC_MARKERS = {
    "thee", "thou", "thy", "thine", "hath", "doth", "dost", "hast",
    "wherefore", "whence", "hence", "thence", "ere", "oft",
    "ne'er", "methinks", "perchance", "forsooth", "prithee", "nay",
    "aye", "yea", "verily", "betwixt", "whilst", "unto", "wilt",
    "shalt", "wouldst", "couldst", "shouldst", "didst", "knowest",
    "cometh", "goeth", "speaketh", "saith", "quoth",
}

OUTPUT_DIR = "exploration_output"
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "samples_stratified")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
CATALOG_PATH = os.path.join(OUTPUT_DIR, "corpus_catalog.csv")
PROMPTS_PATH = os.path.join(OUTPUT_DIR, "llm_prompts.json")
SHORTLIST_PATH = os.path.join(OUTPUT_DIR, "shortlist.csv")


# ─── Data Loading ─────────────────────────────────────────────────

def load_df():
    print("Loading Gutenberg Poetry Corpus from Hugging Face...")
    df = pd.read_parquet(
        "hf://datasets/biglam/gutenberg-poetry-corpus/data/"
        "train-00000-of-00001-fa9fb9e1f16eed7e.parquet"
    )
    print(f"  Loaded {len(df):,} lines across {df['gutenberg_id'].nunique()} poems.\n")
    return df


def build_poem_lookup(df):
    """Pre-group the dataframe by gutenberg_id for fast access."""
    return {gid: group["line"].tolist() for gid, group in df.groupby("gutenberg_id")}


# ─── Helpers ──────────────────────────────────────────────────────

def bucket_for(line_count):
    for lo, hi in BUCKETS:
        if lo <= line_count <= hi:
            return f"{lo}-{hi}"
    return "unknown"


def invert_poem_len_dict():
    """Return {gutenberg_id: line_count}."""
    return {gid: length for length, ids in POEM_LEN_DICT.items() for gid in ids}


def write_lines(lines, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─── 1. Stratified Sampling ──────────────────────────────────────

def stratified_sample(poems, per_bucket=5, seed=42):
    """Sample poems from each length bucket and save to disk."""
    print("=== Stratified Sampling ===\n")
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    bucket_pools = {label: [] for label in BUCKET_LABELS}
    for length, ids in POEM_LEN_DICT.items():
        label = bucket_for(length)
        for gid in ids:
            bucket_pools[label].append((gid, length))

    rng = random.Random(seed)
    manifest = []

    for label in BUCKET_LABELS:
        pool = bucket_pools[label]
        n = min(per_bucket, len(pool))
        chosen = rng.sample(pool, n)
        print(f"  Bucket {label:>12}:  {len(pool):>4} available  ->  sampling {n}")

        for gid, length in chosen:
            lines = poems.get(gid, [])
            filepath = os.path.join(SAMPLES_DIR, f"poem_{gid}.txt")
            write_lines(lines, filepath)
            manifest.append({
                "gutenberg_id": gid,
                "line_count": length,
                "bucket": label,
                "file": filepath,
            })

    manifest_path = os.path.join(SAMPLES_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Saved {len(manifest)} poems to {SAMPLES_DIR}/")
    print(f"  Manifest: {manifest_path}\n")
    return manifest


# ─── 2. Metadata Extraction / Catalog ────────────────────────────

def compute_type_token_ratio(lines, max_lines=200):
    """Vocabulary richness: unique words / total words on a sample."""
    words = re.findall(r"[a-z']+", " ".join(lines[:max_lines]).lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def count_imagery_words(lines, max_lines=300):
    words = re.findall(r"[a-z]+", " ".join(lines[:max_lines]).lower())
    return sum(1 for w in words if w in NATURE_IMAGERY)


def estimate_era(lines, max_lines=300):
    words = re.findall(r"[a-z']+", " ".join(lines[:max_lines]).lower())
    if not words:
        return "unknown"
    ratio = sum(1 for w in words if w in ARCHAIC_MARKERS) / len(words)
    if ratio > 0.02:
        return "archaic"
    if ratio > 0.005:
        return "early_modern"
    return "modern"


def extract_metadata(lines, gid, line_count):
    first_5 = lines[:5]
    last_5 = lines[-5:] if len(lines) >= 5 else lines
    word_count = sum(len(line.split()) for line in lines)
    avg_chars = np.mean([len(l) for l in lines]) if lines else 0.0

    imagery = count_imagery_words(lines)
    return {
        "gutenberg_id": gid,
        "line_count": line_count,
        "word_count": word_count,
        "bucket": bucket_for(line_count),
        "first_line": first_5[0] if first_5 else "",
        "first_5_lines": " | ".join(first_5),
        "last_5_lines": " | ".join(last_5),
        "type_token_ratio": round(compute_type_token_ratio(lines), 4),
        "imagery_word_count": imagery,
        "imagery_density": round(imagery / max(line_count, 1), 4),
        "estimated_era": estimate_era(lines),
        "avg_line_length_chars": round(avg_chars, 1),
    }


def build_catalog(poems):
    """Build a CSV catalog with metadata for every poem in the corpus."""
    print("=== Building Corpus Catalog ===\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gid_to_len = invert_poem_len_dict()
    total = len(gid_to_len)
    rows = []

    for i, (gid, length) in enumerate(sorted(gid_to_len.items())):
        if (i + 1) % 200 == 0 or i == 0:
            print(f"  Processing poem {i + 1}/{total}  (id={gid})...")
        lines = poems.get(gid, [])
        rows.append(extract_metadata(lines, gid, length))

    fieldnames = list(rows[0].keys())
    with open(CATALOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Catalog saved to {CATALOG_PATH}  ({len(rows)} poems)\n")
    return rows


# ─── 3. LLM Prompt Generation ────────────────────────────────────

LLM_SYSTEM_PROMPT = """\
You are a literary analyst. Given a poem excerpt, produce a JSON object with:
{
  "title": "best guess at poem/collection title, or 'Unknown'",
  "author": "best guess at author, or 'Unknown'",
  "genre": "sonnet | ballad | ode | epic | elegy | lyric | narrative | hymn | free_verse | collection | other",
  "is_collection": true or false,
  "themes": ["theme1", "theme2"],
  "environmental_imagery_richness": 1-5,
  "musicality": 1-5,
  "recommended_for_visualization": "yes | no | maybe",
  "recommendation_rationale": "1-2 sentence explanation",
  "notable_lines": ["line1", "line2"]
}
Respond ONLY with valid JSON."""


def build_llm_prompt(poem_text, gid, line_count):
    max_preview = 200
    lines = poem_text.split("\n")
    truncated = "\n".join(lines[:max_preview])
    note = f" (showing first {max_preview} of {line_count} lines)" if line_count > max_preview else ""

    return {
        "gutenberg_id": gid,
        "line_count": line_count,
        "system": LLM_SYSTEM_PROMPT,
        "user": f"Analyze this poem (Gutenberg ID: {gid}, {line_count} lines{note}):\n\n{truncated}",
    }


def generate_llm_prompts(samples_dir=None):
    """Read sampled poems and produce structured prompts for LLM annotation."""
    print("=== Generating LLM Analysis Prompts ===\n")
    samples_dir = samples_dir or SAMPLES_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    manifest_path = os.path.join(samples_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"  No manifest at {manifest_path}. Run --step sample first.\n")
        return []

    with open(manifest_path) as f:
        manifest = json.load(f)

    prompts = []
    for entry in manifest:
        filepath = entry["file"]
        if not os.path.exists(filepath):
            continue
        with open(filepath, encoding="utf-8") as f:
            poem_text = f.read()
        prompts.append(build_llm_prompt(poem_text, entry["gutenberg_id"], entry["line_count"]))

    with open(PROMPTS_PATH, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"  Generated {len(prompts)} prompts -> {PROMPTS_PATH}")

    individual_dir = os.path.join(OUTPUT_DIR, "prompts_individual")
    os.makedirs(individual_dir, exist_ok=True)
    for p in prompts:
        path = os.path.join(individual_dir, f"prompt_{p['gutenberg_id']}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"SYSTEM:\n{p['system']}\n\nUSER:\n{p['user']}")

    print(f"  Individual prompt files -> {individual_dir}/\n")
    return prompts


# ─── 4. Visualization ────────────────────────────────────────────

def get_all_lengths():
    lengths = []
    for length, ids in POEM_LEN_DICT.items():
        lengths.extend([length] * len(ids))
    return np.array(lengths)


def plot_length_distribution():
    print("=== Generating Plots ===\n")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    lengths = get_all_lengths()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(lengths, bins=50, edgecolor="black", alpha=0.7, color="#4C72B0")
    axes[0].set_xlabel("Line Count")
    axes[0].set_ylabel("Number of Poems")
    axes[0].set_title("Poem Length Distribution (Linear)")
    axes[0].axvline(np.median(lengths), color="red", ls="--",
                    label=f"Median: {int(np.median(lengths))}")
    axes[0].axvline(np.mean(lengths), color="orange", ls="--",
                    label=f"Mean: {int(np.mean(lengths))}")
    axes[0].legend()

    axes[1].hist(lengths, bins=np.logspace(0, np.log10(lengths.max()), 50),
                 edgecolor="black", alpha=0.7, color="#4C72B0")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Line Count (log scale)")
    axes[1].set_ylabel("Number of Poems")
    axes[1].set_title("Poem Length Distribution (Log Scale)")
    axes[1].axvline(500, color="green", ls="--", alpha=0.7, label="500 lines")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "length_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  {path}")


def plot_bucket_breakdown():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    counts = []
    for lo, hi in BUCKETS:
        counts.append(sum(len(ids) for ln, ids in POEM_LEN_DICT.items() if lo <= ln <= hi))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ca02c" if 51 <= lo and hi <= 500 else "#4C72B0" for lo, hi in BUCKETS]
    bars = ax.bar(BUCKET_LABELS, counts, color=colors, edgecolor="black", alpha=0.85)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(count), ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Line Count Range")
    ax.set_ylabel("Number of Poems")
    ax.set_title("Poems per Length Bucket  (green = target range for visualization)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "bucket_breakdown.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  {path}")


def plot_catalog_analysis():
    """Scatter and distribution plots derived from the catalog CSV."""
    if not os.path.exists(CATALOG_PATH):
        print(f"  Skipping catalog plots ({CATALOG_PATH} not found).\n")
        return
    os.makedirs(PLOTS_DIR, exist_ok=True)
    catalog = pd.read_csv(CATALOG_PATH)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    era_counts = catalog["estimated_era"].value_counts()
    axes[0].pie(era_counts.values, labels=era_counts.index, autopct="%1.0f%%",
                colors=["#4C72B0", "#DD8452", "#55A868"])
    axes[0].set_title("Estimated Era Distribution")

    axes[1].hist(catalog["type_token_ratio"], bins=30, edgecolor="black",
                 alpha=0.7, color="#DD8452")
    axes[1].set_xlabel("Type-Token Ratio")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Vocabulary Richness (TTR)")

    sub = catalog[catalog["line_count"] < 2000]
    axes[2].scatter(sub["line_count"], sub["imagery_density"], alpha=0.5, s=15, color="#55A868")
    axes[2].set_xlabel("Line Count")
    axes[2].set_ylabel("Imagery Words / Line Count")
    axes[2].set_title("Imagery Density vs. Length")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "catalog_analysis.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  {path}\n")


# ─── 5. Shortlist Generation ─────────────────────────────────────

def generate_shortlist(min_lines=50, max_lines=500, min_imagery_density=0.03):
    """Filter the catalog to poems most suitable for visualization."""
    print("=== Generating Shortlist ===\n")
    if not os.path.exists(CATALOG_PATH):
        print(f"  No catalog at {CATALOG_PATH}. Run --step catalog first.\n")
        return None

    catalog = pd.read_csv(CATALOG_PATH)
    candidates = catalog[
        (catalog["line_count"] >= min_lines)
        & (catalog["line_count"] <= max_lines)
        & (catalog["imagery_density"] >= min_imagery_density)
    ].copy()

    candidates["score"] = (
        candidates["imagery_density"] * 50
        + candidates["type_token_ratio"] * 30
        + (candidates["line_count"].between(100, 300).astype(int)) * 20
    )
    candidates = candidates.sort_values("score", ascending=False)
    candidates.to_csv(SHORTLIST_PATH, index=False)

    print(f"  {len(candidates)} candidates  (from {len(catalog)} total)")
    print(f"  Filters: {min_lines}-{max_lines} lines, imagery >= {min_imagery_density}")
    print(f"  Saved to {SHORTLIST_PATH}\n")

    print("  Top 20 candidates:\n")
    for _, row in candidates.head(20).iterrows():
        first = row["first_line"][:50]
        print(
            f"    ID {row['gutenberg_id']:>6}  |  {row['line_count']:>4} lines  |  "
            f"img={row['imagery_density']:.3f}  ttr={row['type_token_ratio']:.3f}  "
            f"era={row['estimated_era']:>13}  |  \"{first}...\""
        )

    print()
    return candidates


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Systematic exploration of the Gutenberg Poetry Corpus",
    )
    parser.add_argument(
        "--step",
        choices=["sample", "catalog", "prompts", "plots", "shortlist", "all"],
        default="all",
    )
    parser.add_argument("--per-bucket", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    need_df = args.step in ("sample", "catalog", "all")
    poems = None
    if need_df:
        df = load_df()
        print("Building poem lookup table...")
        poems = build_poem_lookup(df)
        del df
        print(f"  {len(poems)} poems indexed.\n")

    if args.step in ("sample", "all"):
        stratified_sample(poems, per_bucket=args.per_bucket, seed=args.seed)

    if args.step in ("catalog", "all"):
        build_catalog(poems)

    if args.step in ("prompts", "all"):
        generate_llm_prompts()

    if args.step in ("plots", "all"):
        plot_length_distribution()
        plot_bucket_breakdown()
        plot_catalog_analysis()

    if args.step in ("shortlist", "all"):
        generate_shortlist()

    print("Done.")


if __name__ == "__main__":
    main()

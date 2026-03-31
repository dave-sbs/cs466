# Poetry as Multimodal Dream — Code

Corpus exploration and preprocessing pipeline for *Poetry as Multimodal Dream: Corpus-Driven Visualization of Linguistic and Sonic Emergence* (CS466 Final Project).

## Overview

This directory contains the data exploration and preprocessing work for Check-in 1. The goal is to select and characterize a set of poems from the Gutenberg Poetry Corpus that are well-suited for multimodal visualization — poems with rich environmental imagery, manageable length, and strong musicality.

## Project Structure

```
code/
├── constants.py              # Pre-computed poem length dictionary for the corpus
├── exploration.py            # Initial ad-hoc exploration: length stats and manual sampling
├── explore_corpus.py         # Systematic six-step corpus exploration pipeline (main script)
├── requirements.txt          # Python dependencies
├── samples/                  # Ad-hoc manually pulled poem files for early review
└── exploration_output/
    ├── corpus_catalog.csv    # Per-poem metadata for all poems in the corpus
    ├── shortlist.csv         # Ranked visualization candidates
    ├── llm_prompts.json      # LLM analysis prompts (consolidated)
    ├── plots/                # Diagnostic visualizations (PNG)
    ├── prompts_individual/   # Per-poem LLM analysis prompts (TXT)
    └── samples_stratified/   # Stratified poem samples + manifest.json
```

## Dataset

**Gutenberg Poetry Corpus** — `biglam/gutenberg-poetry-corpus` on Hugging Face.

A large collection of public-domain poetry from Project Gutenberg, stored as a line-level Parquet file. The corpus spans thousands of unique works across a wide range of lengths, eras, and traditions.

## Setup

```bash
pip install -r requirements.txt
```

> The corpus is streamed directly from Hugging Face on first load — no manual download required.

## Usage

### Systematic Pipeline (`explore_corpus.py`)

Run all six steps end-to-end:

```bash
python explore_corpus.py
```

Or run individual steps:

```bash
python explore_corpus.py --step sample     # Stratified sampling across length buckets
python explore_corpus.py --step catalog    # Build full-corpus metadata catalog
python explore_corpus.py --step prompts    # Generate LLM analysis prompts for samples
python explore_corpus.py --step plots      # Generate diagnostic visualizations
python explore_corpus.py --step shortlist  # Filter and rank visualization candidates
```

#### Pipeline Steps

| Step | Output | Description |
|------|--------|-------------|
| `sample` | `samples_stratified/` | 5 poems per length bucket (9 buckets), saved to text files with a JSON manifest |
| `catalog` | `corpus_catalog.csv` | Per-poem features: line/word count, type-token ratio, imagery density, estimated era |
| `prompts` | `llm_prompts.json`, `prompts_individual/` | Structured prompts for LLM literary annotation of sampled poems |
| `plots` | `plots/` | Length distribution histogram, bucket breakdown bar chart, catalog scatter/TTR plots |
| `shortlist` | `shortlist.csv` | Poems ranked by `50×imagery_density + 30×TTR + 20×(100≤lines≤300)` |

### Initial Exploration (`exploration.py`)

A simpler script for quick ad-hoc analysis:

```bash
python exploration.py
```

Prints summary statistics (min, max, mean, median, percentiles, bucket counts) from the pre-computed length dictionary without loading the full corpus.

## Poem Selection Criteria

Target range for visualization candidates: **51–500 lines**.

Poems are filtered and scored by:
- **Imagery density** (`imagery_word_count / line_count`) ≥ 0.03 — ensures rich environmental/nature vocabulary
- **Type-token ratio** — favors vocabulary richness over repetition
- **Length** — 100–300 lines receives a score bonus as the ideal visualization length

The ranked `shortlist.csv` is the primary artifact for poem selection in subsequent pipeline stages.

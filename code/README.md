# Poetry as Multimodal Dream — Code

Pipeline for *Poetry as Multimodal Dream: Corpus-Driven Visualization of Linguistic and Sonic Emergence* (CS466 Final Project).

The system selects poems from the Gutenberg Poetry Corpus, retrieves semantically aligned nature images via CLIP + FAISS, and produces qualitative evaluation galleries for visual inspection — forming the foundation for dream-like video visualization.

## Project Structure

```
code/
├── constants.py              # Pre-computed poem length dictionary
├── exploration.py            # Ad-hoc corpus stats and manual sampling
├── explore_corpus.py         # Systematic corpus exploration pipeline
├── download_images.py        # Download nature images from HuggingFace
├── clip_pipeline.py          # CLIP embedding + FAISS retrieval pipeline
├── evaluate_retrieval.py     # HTML gallery generator for qualitative evaluation
├── requirements.txt          # Python dependencies
├── data/
│   ├── images/               # Downloaded images (not tracked in git)
│   ├── embeddings.npy        # CLIP image embeddings (not tracked in git)
│   ├── faiss_index.bin       # FAISS index (not tracked in git)
│   └── image_ids.json        # Ordered list of image filenames
├── exploration_output/
│   ├── corpus_catalog.csv    # Per-poem metadata for all corpus poems
│   ├── shortlist.csv         # Ranked visualization candidates
│   ├── llm_prompts.json      # LLM analysis prompts (consolidated)
│   ├── plots/                # Diagnostic visualizations (PNG)
│   ├── prompts_individual/   # Per-poem LLM analysis prompts (TXT)
│   └── samples_stratified/   # Stratified poem samples + manifest.json
└── output/
    ├── retrieval_results/    # Per-poem retrieved images (not tracked in git)
    └── galleries/            # HTML evaluation galleries (not tracked in git)
```

## Datasets

| Dataset | Source | Description |
|---------|--------|-------------|
| **Gutenberg Poetry Corpus** | `biglam/gutenberg-poetry-corpus` on HuggingFace | Public-domain poetry, stored as a line-level Parquet file. Streamed on first load — no manual download required. |
| **Nature Image Dataset** | `mertcobanov/nature-dataset` on HuggingFace | ~50K nature/landscape images with captions. Downloaded locally to `data/images/`. |

## Setup

```bash
pip install -r requirements.txt
```

> On macOS, if you encounter an OpenMP conflict (`OMP: Error #15`), prefix commands with `KMP_DUPLICATE_LIB_OK=TRUE`.

## Full Pipeline

Run the steps in order:

### Step 1 — Corpus Exploration

Build the corpus catalog, generate stratified samples, and produce a ranked shortlist of visualization candidates:

```bash
python explore_corpus.py          # Run all steps end-to-end
```

Or run individual steps:

```bash
python explore_corpus.py --step sample     # Stratified sampling across length buckets
python explore_corpus.py --step catalog    # Build full-corpus metadata catalog
python explore_corpus.py --step prompts    # Generate LLM analysis prompts for samples
python explore_corpus.py --step plots      # Generate diagnostic visualizations
python explore_corpus.py --step shortlist  # Filter and rank visualization candidates
```

**Outputs:** `exploration_output/shortlist.csv`, `exploration_output/corpus_catalog.csv`, `exploration_output/samples_stratified/`

---

### Step 2 — Download Images

Download nature images from HuggingFace to `data/images/`:

```bash
python download_images.py                   # Download first 2000 images (default)
python download_images.py --limit 500       # Smaller prototype set
python download_images.py --limit 0         # Download all ~50K images
```

The script is resumable — re-running skips already-downloaded images via `data/images/manifest.csv`.

**Outputs:** `data/images/*.jpg`, `data/images/manifest.csv`

---

### Step 3 — Build CLIP Embeddings + FAISS Index

Encode all images with CLIP and build a similarity index:

```bash
python clip_pipeline.py --step embed-images   # Encode images → data/embeddings.npy
python clip_pipeline.py --step build-index    # Build FAISS index → data/faiss_index.bin
```

Or run both at once:

```bash
python clip_pipeline.py
```

**Outputs:** `data/embeddings.npy`, `data/image_ids.json`, `data/faiss_index.bin`

---

### Step 4 — Retrieve Images for a Poem

Retrieve the top-K most visually similar images for each stanza of a poem:

```bash
# By Gutenberg ID (poem must be extracted first — see Step 1)
python clip_pipeline.py --step retrieve --gutenberg_id 24449

# By free-form text query
python clip_pipeline.py --step retrieve --text "dark forest at dusk, mist rising"

# With custom chunk size and top-K
python clip_pipeline.py --step retrieve --gutenberg_id 24449 --top_k 10 --chunk_size 8
```

Poems are split into stanzas (or fixed-size chunks). Each chunk is encoded with CLIP's text encoder and queried against the FAISS index.

**Outputs:** `output/retrieval_results/<poem_id>/`

---

### Step 5 — Qualitative Evaluation

Generate self-contained HTML galleries for visual inspection of retrieval quality:

```bash
python evaluate_retrieval.py                          # All shortlisted poems
python evaluate_retrieval.py --top_n 5                # Top 5 from shortlist
python evaluate_retrieval.py --ids 24449 9825 36305   # Specific poems by ID
python evaluate_retrieval.py --top_k 5 --chunk_size 6
```

Each gallery shows poem stanzas alongside their top-K retrieved images with cosine similarity scores. An `index.html` links all generated galleries.

**Outputs:** `output/galleries/<poem_id>.html`, `output/galleries/index.html`

---

## Poem Selection Criteria

Target range: **51–500 lines**.

Poems are scored by:
- **Imagery density** (`imagery_word_count / line_count`) ≥ 0.03 — rich environmental/nature vocabulary
- **Type-token ratio** — vocabulary richness over repetition
- **Length** — 100–300 lines receives a score bonus as the ideal visualization length

Ranking formula: `50×imagery_density + 30×TTR + 20×(100≤lines≤300)`

The ranked `shortlist.csv` is the primary artifact for poem selection.

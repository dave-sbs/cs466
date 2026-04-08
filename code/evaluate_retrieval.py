"""
evaluate_retrieval.py — Qualitative evaluation of CLIP retrieval via HTML galleries.

For each poem in the shortlist (or a specified set), runs retrieval via clip_pipeline.py
and generates a self-contained HTML gallery showing each stanza alongside its top-K
retrieved images with similarity scores.

Usage:
    python evaluate_retrieval.py                          # Run all shortlisted poems
    python evaluate_retrieval.py --ids 24449 9825 36305   # Specific Gutenberg IDs
    python evaluate_retrieval.py --top_n 10               # Only top 10 from shortlist
    python evaluate_retrieval.py --top_k 5 --chunk_size 6
"""

import os
import json
import argparse
import base64
import shutil
from pathlib import Path

import pandas as pd

from clip_pipeline import retrieve, IMAGES_DIR, OUTPUT_DIR


# ─── Configuration ────────────────────────────────────────────────────────────

SHORTLIST_PATH = Path("exploration_output/shortlist.csv")
GALLERY_DIR = Path("output/galleries")


# ─── HTML Generation ──────────────────────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --bg: #0f0f13;
    --surface: #1a1a24;
    --border: #2e2e42;
    --accent: #7c6af7;
    --accent2: #c084fc;
    --text: #e2e2f0;
    --muted: #888899;
    --score-high: #4ade80;
    --score-mid: #facc15;
    --score-low: #f87171;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Georgia', serif;
    padding: 2rem;
    line-height: 1.6;
  }}

  header {{
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
  }}

  header h1 {{
    font-size: 1.6rem;
    font-weight: normal;
    letter-spacing: 0.02em;
    color: var(--accent2);
  }}

  header .meta {{
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.4rem;
  }}

  .stats {{
    display: flex;
    gap: 2rem;
    margin-bottom: 2rem;
    font-size: 0.85rem;
    color: var(--muted);
  }}

  .stats span {{ color: var(--text); font-weight: bold; }}

  .chunk {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 2.5rem;
  }}

  .chunk-header {{
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin-bottom: 1rem;
  }}

  .chunk-number {{
    font-size: 0.72rem;
    font-family: monospace;
    background: var(--accent);
    color: #fff;
    padding: 2px 8px;
    border-radius: 3px;
    flex-shrink: 0;
  }}

  .chunk-text {{
    font-style: italic;
    font-size: 0.95rem;
    color: var(--text);
    border-left: 3px solid var(--accent);
    padding-left: 1rem;
    line-height: 1.8;
    white-space: pre-wrap;
  }}

  .images-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 1rem;
    margin-top: 1.2rem;
  }}

  .image-card {{
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    background: var(--bg);
    transition: transform 0.15s ease, border-color 0.15s ease;
  }}

  .image-card:hover {{
    transform: translateY(-2px);
    border-color: var(--accent);
  }}

  .image-card img {{
    width: 100%;
    height: 150px;
    object-fit: cover;
    display: block;
  }}

  .image-card .card-footer {{
    padding: 6px 8px;
    font-size: 0.7rem;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}

  .score {{
    font-family: monospace;
    font-weight: bold;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.68rem;
  }}

  .score-high {{ background: #14532d; color: var(--score-high); }}
  .score-mid  {{ background: #422006; color: var(--score-mid); }}
  .score-low  {{ background: #450a0a; color: var(--score-low); }}

  .image-card .label {{
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 110px;
  }}

  .missing-img {{
    height: 150px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    color: var(--muted);
    background: #111118;
  }}

  /* ── Attribution styles ── */

  .poem-line {{
    display: block;
    padding: 1px 4px;
    border-radius: 2px;
    transition: background 0.2s ease;
  }}

  .line-attribution-bar {{
    display: flex;
    height: 6px;
    margin: 0 4px;
    border-radius: 2px;
    overflow: hidden;
    gap: 1px;
  }}

  .attr-segment {{
    flex: 1;
    border-radius: 1px;
    min-width: 4px;
  }}

  footer {{
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    font-size: 0.75rem;
    color: var(--muted);
    text-align: center;
  }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="meta">{meta}</div>
</header>

<div class="stats">
  <div>Chunks: <span>{num_chunks}</span></div>
  <div>Top-K per chunk: <span>{top_k}</span></div>
  <div>Total retrieved: <span>{total_retrieved}</span></div>
</div>

{chunks_html}

<footer>Generated by evaluate_retrieval.py &mdash; CS466 Poetry Multimodal Dream</footer>
<script>
document.querySelectorAll('.image-card[data-line-scores]').forEach(card => {{
  const chunk = card.closest('.chunk');
  const lines = chunk.querySelectorAll('.poem-line');
  if (!lines.length) return;

  card.addEventListener('mouseenter', () => {{
    const scores = JSON.parse(card.dataset.lineScores);
    const maxS = Math.max(...scores);
    const minS = Math.min(...scores);
    const span = maxS - minS || 1;
    lines.forEach((el, i) => {{
      const norm = (scores[i] - minS) / span;
      el.style.background = 'rgba(124,106,247,' + (0.08 + norm * 0.45) + ')';
    }});
  }});

  card.addEventListener('mouseleave', () => {{
    lines.forEach(el => {{
      el.style.background = el.dataset.defaultBg || 'transparent';
    }});
  }});
}});
</script>
</body>
</html>
"""

CHUNK_TEMPLATE = """\
<div class="chunk">
  <div class="chunk-header">
    <span class="chunk-number">STANZA {chunk_num}</span>
    <div class="chunk-text">{chunk_text}</div>
  </div>
  <div class="images-grid">
    {images_html}
  </div>
</div>
"""

IMAGE_CARD_TEMPLATE = """\
<div class="image-card"{data_attr}>
  {img_tag}
  {attribution_bar}
  <div class="card-footer">
    <span class="label" title="{label}">{label}</span>
    <span class="score {score_class}">{score:.3f}</span>
  </div>
</div>
"""


def score_class(score: float) -> str:
    if score >= 0.25:
        return "score-high"
    if score >= 0.18:
        return "score-mid"
    return "score-low"


def build_attribution_bar(line_scores: list[float], line_texts: list[str]) -> str:
    """Build a small horizontal bar showing per-line contribution."""
    min_s = min(line_scores)
    max_s = max(line_scores)
    span = max_s - min_s if max_s > min_s else 1.0
    segments = []
    for i, s in enumerate(line_scores):
        norm = (s - min_s) / span
        opacity = 0.15 + norm * 0.85
        title = f"L{i+1}: {line_texts[i][:40]} ({s:.3f})"
        segments.append(
            f'<div class="attr-segment" style="opacity:{opacity:.2f};background:var(--accent)" title="{title}"></div>'
        )
    return f'<div class="line-attribution-bar">{"".join(segments)}</div>'


def render_stanza_with_attributions(query_text: str, line_texts: list[str], all_line_scores: list[list[float]]) -> str:
    """Render stanza text with per-line background coloring based on max attribution."""
    # Compute max attribution per line across all retrieved images
    num_lines = len(line_texts)
    max_per_line = [0.0] * num_lines
    for scores in all_line_scores:
        for j in range(min(len(scores), num_lines)):
            if scores[j] > max_per_line[j]:
                max_per_line[j] = scores[j]

    min_s = min(max_per_line) if max_per_line else 0
    max_s = max(max_per_line) if max_per_line else 0
    span = max_s - min_s if max_s > min_s else 1.0

    parts = []
    for i, line in enumerate(line_texts):
        norm = (max_per_line[i] - min_s) / span
        opacity = 0.05 + norm * 0.35
        bg = f"rgba(124,106,247,{opacity:.2f})"
        parts.append(
            f'<span class="poem-line" data-line-idx="{i}" data-default-bg="{bg}" style="background:{bg}">{line}</span>'
        )
    return "\n".join(parts)


def img_tag_for(image_path: Path, output_file: str, chunk_dir: Path) -> str:
    """
    Try to embed image as base64 (self-contained HTML).
    Fall back to a relative path reference, then a placeholder if missing.
    """
    # Check if the copied image exists in the retrieval output dir
    copied = chunk_dir / output_file
    src = copied if copied.exists() else image_path

    if src.exists():
        try:
            with open(src, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            ext = src.suffix.lower().lstrip(".")
            mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
            return f'<img src="data:{mime};base64,{data}" alt="retrieved image" loading="lazy">'
        except Exception:
            pass
    return '<div class="missing-img">image not found</div>'


def build_gallery_html(manifest: dict, chunk_dir: Path, poem_name: str) -> str:
    chunks_html_parts = []
    has_attributions = "line_texts" in manifest.get("results", [{}])[0]

    for chunk in manifest["results"]:
        line_texts = chunk.get("line_texts")
        # Collect all line_attributions for this chunk (one per retrieved image)
        all_line_scores = [
            item["line_attributions"]
            for item in chunk["top_k"]
            if "line_attributions" in item
        ]

        images_html_parts = []
        for item in chunk["top_k"]:
            img_path = IMAGES_DIR / item["image_id"]
            label = Path(item["image_id"]).parent.name or Path(item["image_id"]).stem
            sc = score_class(item["score"])
            img_html = img_tag_for(img_path, item["output_file"], chunk_dir)

            # Build attribution bar and data attribute if available
            attribution_bar = ""
            data_attr = ""
            if "line_attributions" in item and line_texts:
                attribution_bar = build_attribution_bar(item["line_attributions"], line_texts)
                data_attr = f' data-line-scores=\'{json.dumps(item["line_attributions"])}\''

            images_html_parts.append(
                IMAGE_CARD_TEMPLATE.format(
                    img_tag=img_html,
                    label=label,
                    score=item["score"],
                    score_class=sc,
                    attribution_bar=attribution_bar,
                    data_attr=data_attr,
                )
            )

        # Render stanza text with or without attribution highlighting
        if has_attributions and line_texts and all_line_scores:
            chunk_text = render_stanza_with_attributions(
                chunk["query_text"], line_texts, all_line_scores
            )
        else:
            chunk_text = chunk["query_text"].replace(" / ", "\n")

        chunks_html_parts.append(
            CHUNK_TEMPLATE.format(
                chunk_num=chunk["chunk_index"] + 1,
                chunk_text=chunk_text,
                images_html="\n    ".join(images_html_parts),
            )
        )

    gid = manifest.get("gutenberg_id")
    title = f"Poem {gid} — CLIP Retrieval Gallery" if gid else f"{poem_name} — CLIP Retrieval"
    meta = f"Gutenberg ID: {gid} | {manifest['num_chunks']} stanzas | top-{manifest['top_k']} images each"
    total = manifest["num_chunks"] * manifest["top_k"]

    return HTML_TEMPLATE.format(
        title=title,
        meta=meta,
        num_chunks=manifest["num_chunks"],
        top_k=manifest["top_k"],
        total_retrieved=total,
        chunks_html="\n".join(chunks_html_parts),
    )


# ─── Main Evaluation Logic ────────────────────────────────────────────────────

def evaluate_poem(
    gutenberg_id: int,
    top_k: int,
    chunk_size: int,
    use_stanzas: bool,
    compute_attributions: bool = False,
) -> Path | None:
    print(f"\n--- Evaluating poem {gutenberg_id} ---")

    try:
        manifest = retrieve(
            gutenberg_id=gutenberg_id,
            top_k=top_k,
            chunk_size=chunk_size,
            use_stanzas=use_stanzas,
            compute_attributions=compute_attributions,
        )
    except FileNotFoundError as e:
        print(f"  Skipping: {e}")
        return None

    if manifest is None:
        return None

    poem_name = f"poem_{gutenberg_id}"
    chunk_dir = OUTPUT_DIR / poem_name
    html = build_gallery_html(manifest, chunk_dir, poem_name)

    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    gallery_path = GALLERY_DIR / f"{poem_name}_gallery.html"
    with open(gallery_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  Gallery saved -> {gallery_path}")
    return gallery_path


def build_index_page(galleries: list) -> Path:
    """Build a simple index HTML linking to all generated galleries."""
    links = "\n".join(
        f'<li><a href="{g.name}">{g.stem.replace("_gallery", "").replace("_", " ")}</a></li>'
        for g in galleries
    )
    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Retrieval Gallery Index</title>
<style>
  body {{ background:#0f0f13; color:#e2e2f0; font-family:Georgia,serif; padding:2rem; }}
  h1 {{ color:#c084fc; margin-bottom:1rem; font-size:1.4rem; font-weight:normal; }}
  ul {{ list-style:none; }}
  li {{ margin:0.5rem 0; }}
  a {{ color:#7c6af7; text-decoration:none; }}
  a:hover {{ text-decoration:underline; }}
</style>
</head>
<body>
<h1>Poetry CLIP Retrieval &mdash; Gallery Index</h1>
<ul>
{links}
</ul>
</body>
</html>
"""
    index_path = GALLERY_DIR / "index.html"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html)
    return index_path


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML retrieval galleries for shortlisted poems."
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        type=int,
        default=None,
        metavar="ID",
        help="Specific Gutenberg IDs to evaluate (default: use shortlist)",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=None,
        help="Only evaluate top N poems from shortlist (by score)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Images to retrieve per stanza (default: 5)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=8,
        help="Lines per chunk when not using stanza splitting (default: 8)",
    )
    parser.add_argument(
        "--no_stanzas",
        action="store_true",
        help="Use fixed-size chunks instead of stanza detection",
    )
    parser.add_argument(
        "--attributions",
        action="store_true",
        help="Compute and display per-line attribution scores",
    )
    args = parser.parse_args()

    if args.ids:
        gutenberg_ids = args.ids
    elif SHORTLIST_PATH.exists():
        df = pd.read_csv(SHORTLIST_PATH)
        if args.top_n:
            df = df.head(args.top_n)
        gutenberg_ids = df["gutenberg_id"].tolist()
        print(f"Loaded {len(gutenberg_ids)} poems from shortlist.")
    else:
        print(f"No shortlist found at {SHORTLIST_PATH}. Provide --ids or run explore_corpus.py first.")
        return

    galleries = []
    for gid in gutenberg_ids:
        path = evaluate_poem(
            gutenberg_id=int(gid),
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            use_stanzas=not args.no_stanzas,
            compute_attributions=args.attributions,
        )
        if path:
            galleries.append(path)

    if galleries:
        index_path = build_index_page(galleries)
        print(f"\nAll done. Generated {len(galleries)} galleries.")
        print(f"Index: {index_path}")
    else:
        print("\nNo galleries generated. Check that images are downloaded and FAISS index is built.")


if __name__ == "__main__":
    main()

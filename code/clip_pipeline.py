"""
clip_pipeline.py — CLIP embedding + FAISS retrieval pipeline.

Three sequential steps, each runnable independently:

    python clip_pipeline.py --step embed-images   # Encode all images with CLIP
    python clip_pipeline.py --step build-index    # Build FAISS index from embeddings
    python clip_pipeline.py --step retrieve       # Retrieve images for a poem
    python clip_pipeline.py                       # Run all three steps

Retrieve options:
    python clip_pipeline.py --step retrieve --gutenberg_id 24449
    python clip_pipeline.py --step retrieve --text "the sea at night, cold and restless"
    python clip_pipeline.py --step retrieve --top_k 10 --chunk_size 8
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel

import faiss

from interpretability import compute_line_attributions

from dream_chunks import split_poem


# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME = "openai/clip-vit-base-patch32"

IMAGES_DIR = Path("data/images")
DATA_DIR = Path("data")
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
IMAGE_IDS_PATH = DATA_DIR / "image_ids.json"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"

OUTPUT_DIR = Path("output/retrieval_results")

# Aligned Project Gutenberg text with blank-line stanza markers (see fetch_raw_gutenberg.py)
PG_RAW_POEM_DIR = Path("exploration_output/pg_raw")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model(device: str):
    print(f"Loading CLIP model ({MODEL_NAME}) on {device}...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    print("  Model loaded.\n")
    return model, processor


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─── Image Collection ─────────────────────────────────────────────────────────

def collect_images(images_dir: Path) -> list:
    """Recursively collect all image paths under images_dir."""
    paths = []
    for p in sorted(images_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            paths.append(p)
    return paths


# ─── Step 1: Embed Images ─────────────────────────────────────────────────────

def embed_images(batch_size: int = 32):
    """Encode all images in IMAGES_DIR with CLIP and save embeddings."""
    print("=== Step 1: Embedding Images ===\n")

    image_paths = collect_images(IMAGES_DIR)
    if not image_paths:
        print(f"No images found in {IMAGES_DIR}. Run download_images.py first.")
        return

    print(f"Found {len(image_paths)} images.\n")

    device = get_device()
    model, processor = load_model(device)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_embeddings = []
    image_ids = []
    failed = []

    for batch_start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_start : batch_start + batch_size]
        images = []
        valid_paths = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_paths.append(p)
            except Exception as e:
                print(f"  Skipping {p.name}: {e}")
                failed.append(str(p))

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            features = model.visual_projection(vision_outputs.pooler_output)
            # L2-normalize so cosine similarity == dot product
            features = torch.nn.functional.normalize(features, dim=-1)

        all_embeddings.append(features.cpu().numpy())
        image_ids.extend([str(p.relative_to(IMAGES_DIR)) for p in valid_paths])

        done = min(batch_start + batch_size, len(image_paths))
        print(f"  Embedded {done}/{len(image_paths)} images...", end="\r")

    print()  # newline after progress

    if not all_embeddings:
        print("No embeddings produced. Check your images.")
        return

    embeddings = np.concatenate(all_embeddings, axis=0).astype("float32")
    np.save(str(EMBEDDINGS_PATH), embeddings)

    with open(IMAGE_IDS_PATH, "w") as f:
        json.dump(image_ids, f, indent=2)

    print(f"\n  Embeddings shape: {embeddings.shape}")
    print(f"  Saved embeddings -> {EMBEDDINGS_PATH}")
    print(f"  Saved image ID map -> {IMAGE_IDS_PATH}")
    if failed:
        print(f"  Skipped {len(failed)} images due to errors.")
    print()


# ─── Step 2: Build FAISS Index ────────────────────────────────────────────────

def build_index():
    """Build a FAISS inner-product index from saved embeddings."""
    print("=== Step 2: Building FAISS Index ===\n")

    if not EMBEDDINGS_PATH.exists():
        print(f"No embeddings at {EMBEDDINGS_PATH}. Run --step embed-images first.")
        return

    embeddings = np.load(str(EMBEDDINGS_PATH)).astype("float32")
    print(f"  Loaded embeddings: {embeddings.shape}")

    dim = embeddings.shape[1]
    # Inner product index (works for cosine similarity on L2-normalized vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"  Index size: {index.ntotal} vectors")
    print(f"  Saved FAISS index -> {FAISS_INDEX_PATH}\n")


# ─── Poem Loading ─────────────────────────────────────────────────────────────

def load_poem_lines(gutenberg_id: int) -> tuple[list[str], str]:
    """
    Load poem lines and record source.

    Returns (lines, source) where source is:
      - 'pg_raw' — aligned Gutenberg text; may include blank lines between stanzas
      - 'samples_stratified' | 'samples' — parquet-derived excerpts (no stanza blanks)
    """
    pg_path = PG_RAW_POEM_DIR / f"poem_{gutenberg_id}.txt"
    if pg_path.exists():
        with open(pg_path, encoding="utf-8") as f:
            return f.read().splitlines(), "pg_raw"

    candidates = [
        (Path("exploration_output/samples_stratified"), "samples_stratified"),
        (Path("samples"), "samples"),
    ]
    for folder, label in candidates:
        p = folder / f"poem_{gutenberg_id}.txt"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                lines = [line for line in f.read().splitlines() if line.strip()]
            return lines, label

    raise FileNotFoundError(
        f"Poem {gutenberg_id} not found. "
        f"Run explore_corpus.py --step sample, or fetch_raw_gutenberg.py all --ids {gutenberg_id}."
    )


def chunk_lines(lines: list, chunk_size: int) -> list:
    """Split poem lines into chunks of chunk_size (non-empty lines only)."""
    nonempty = [line.strip() for line in lines if line.strip()]
    chunks = []
    for i in range(0, len(nonempty), chunk_size):
        chunk = nonempty[i : i + chunk_size]
        chunks.append(" / ".join(chunk))
    return chunks


def split_into_stanzas(lines: list) -> list:
    """Split by blank lines into stanzas."""
    stanzas = []
    current = []
    for line in lines:
        if line.strip() == "":
            if current:
                stanzas.append(" / ".join(current))
                current = []
        else:
            current.append(line.strip())
    if current:
        stanzas.append(" / ".join(current))
    if not stanzas:
        return []
    if len(stanzas) == 1 and not any(l.strip() == "" for l in lines):
        print(
            "[warn] split_into_stanzas: no blank lines in source; "
            "would produce a single whole-text chunk — caller should use chunk_lines.",
            file=sys.stderr,
        )
    return stanzas


# ─── Step 3: Retrieve ─────────────────────────────────────────────────────────

def retrieve(
    gutenberg_id: int = None,
    text: str = None,
    top_k: int = 5,
    chunk_size: int = 8,
    use_stanzas: bool = True,
    compute_attributions: bool = False,
):
    """Retrieve top-K images per chunk/stanza for a poem or free text."""
    print("=== Step 3: Retrieving Images ===\n")

    for p in [FAISS_INDEX_PATH, IMAGE_IDS_PATH]:
        if not p.exists():
            print(f"Missing {p}. Run --step build-index first.")
            return

    if compute_attributions and not EMBEDDINGS_PATH.exists():
        print(f"Missing {EMBEDDINGS_PATH}. Run --step embed-images first (needed for attributions).")
        return

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(IMAGE_IDS_PATH) as f:
        image_ids = json.load(f)

    all_image_embeddings = None
    if compute_attributions:
        all_image_embeddings = np.load(str(EMBEDDINGS_PATH)).astype("float32")

    device = get_device()
    model, processor = load_model(device)

    if text:
        queries = [text]
        poem_name = "custom_text"
        print(f"Query text: \"{text[:80]}{'...' if len(text) > 80 else ''}\"\n")
    elif gutenberg_id is not None:
        lines, text_source = load_poem_lines(gutenberg_id)
        n_nonempty = sum(1 for l in lines if l.strip())
        print(f"Poem {gutenberg_id}: {n_nonempty} non-empty lines (source={text_source})\n")

        if use_stanzas and text_source != "pg_raw":
            print(
                f"[warn] No aligned pg_raw text for poem {gutenberg_id}; "
                f"using fixed-size chunks (chunk_size={chunk_size}), not stanza split.",
                file=sys.stderr,
            )
            chunks = split_poem(lines, fallback_chunk_size=chunk_size)
            queries = [c.text for c in chunks if c.split_mode == "fixed"]
        else:
            chunks = split_poem(lines, fallback_chunk_size=chunk_size)
            if use_stanzas:
                stanzas = [c.text for c in chunks if c.split_mode == "stanza"]
                queries = stanzas if stanzas else [c.text for c in chunks]
            else:
                queries = [c.text for c in chunks]
        poem_name = f"poem_{gutenberg_id}"
    else:
        print("Provide --gutenberg_id or --text.")
        return

    out_dir = OUTPUT_DIR / poem_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, query in enumerate(queries):
        inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            text_outputs = model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            text_features = model.text_projection(text_outputs.pooler_output)
            text_features = torch.nn.functional.normalize(text_features, dim=-1)

        query_vec = text_features.cpu().numpy().astype("float32")
        scores, indices = index.search(query_vec, top_k)

        # Compute per-line attribution scores if requested
        line_attributions = None
        line_texts = None
        if compute_attributions:
            retrieved_img_embs = all_image_embeddings[indices[0]]  # shape [K, 512]
            line_attributions, line_texts = compute_line_attributions(
                model, processor, query, retrieved_img_embs, device
            )

        chunk_results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            img_rel = image_ids[idx]
            img_src = IMAGES_DIR / img_rel

            # Copy retrieved image into output dir for easy browsing
            dest_name = f"chunk{i:03d}_rank{rank+1}_{Path(img_rel).name}"
            dest = out_dir / dest_name
            if img_src.exists():
                shutil.copy2(str(img_src), str(dest))

            item = {
                "rank": rank + 1,
                "score": float(score),
                "image_id": img_rel,
                "output_file": dest_name,
            }
            if line_attributions is not None:
                item["line_attributions"] = line_attributions[rank]
            chunk_results.append(item)

        chunk_result = {
            "chunk_index": i,
            "query_text": query,
            "top_k": chunk_results,
        }
        if line_texts is not None:
            chunk_result["line_texts"] = line_texts
        results.append(chunk_result)

        print(f"  Chunk {i+1}/{len(queries)}: \"{query[:60]}{'...' if len(query) > 60 else ''}\"")
        for r in chunk_results:
            print(f"    #{r['rank']}  score={r['score']:.4f}  {r['image_id']}")

    manifest = {
        "poem_name": poem_name,
        "gutenberg_id": gutenberg_id,
        "num_chunks": len(queries),
        "top_k": top_k,
        "results": results,
    }
    manifest_path = out_dir / "retrieval_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Results saved to {out_dir}/")
    print(f"  Manifest: {manifest_path}\n")
    return manifest


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CLIP embedding and FAISS retrieval pipeline for poetry visualization."
    )
    parser.add_argument(
        "--step",
        choices=["embed-images", "build-index", "retrieve", "all"],
        default="all",
        help="Pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for image embedding (default: 32)",
    )
    parser.add_argument(
        "--gutenberg_id",
        type=int,
        default=None,
        help="Gutenberg ID of poem to retrieve images for",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Free text query for retrieval (overrides --gutenberg_id)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of images to retrieve per chunk (default: 5)",
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
        help="Compute per-line attribution scores for interpretability",
    )
    args = parser.parse_args()

    if args.step in ("embed-images", "all"):
        embed_images(batch_size=args.batch_size)

    if args.step in ("build-index", "all"):
        build_index()

    if args.step in ("retrieve", "all"):
        if args.step == "all" and args.gutenberg_id is None and args.text is None:
            print("Skipping retrieval in 'all' mode: provide --gutenberg_id or --text.\n")
        else:
            retrieve(
                gutenberg_id=args.gutenberg_id,
                text=args.text,
                top_k=args.top_k,
                chunk_size=args.chunk_size,
                use_stanzas=not args.no_stanzas,
                compute_attributions=args.attributions,
            )

    print("Done.")


if __name__ == "__main__":
    main()

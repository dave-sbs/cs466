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
import json
import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel

import faiss


# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME = "openai/clip-vit-base-patch32"

IMAGES_DIR = Path("data/images")
DATA_DIR = Path("data")
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
IMAGE_IDS_PATH = DATA_DIR / "image_ids.json"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"

OUTPUT_DIR = Path("output/retrieval_results")

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

def load_poem_lines(gutenberg_id: int) -> list:
    """Load poem lines from the stratified samples or general samples directory."""
    candidates = [
        Path("exploration_output/samples_stratified") / f"poem_{gutenberg_id}.txt",
        Path("samples") / f"poem_{gutenberg_id}.txt",
    ]
    for p in candidates:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return [line for line in f.read().splitlines() if line.strip()]
    raise FileNotFoundError(
        f"Poem {gutenberg_id} not found. "
        f"Run explore_corpus.py --step sample or exploration.py to extract it first."
    )


def chunk_lines(lines: list, chunk_size: int) -> list:
    """Split poem lines into chunks of chunk_size, preserving context."""
    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunk = lines[i : i + chunk_size]
        chunks.append(" / ".join(chunk))
    return chunks


def split_into_stanzas(lines: list) -> list:
    """Split by blank lines into stanzas; fall back to chunk_lines if no blanks."""
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
    return stanzas if stanzas else [" / ".join(lines)]


# ─── Step 3: Retrieve ─────────────────────────────────────────────────────────

def retrieve(
    gutenberg_id: int = None,
    text: str = None,
    top_k: int = 5,
    chunk_size: int = 8,
    use_stanzas: bool = True,
):
    """Retrieve top-K images per chunk/stanza for a poem or free text."""
    print("=== Step 3: Retrieving Images ===\n")

    for p in [FAISS_INDEX_PATH, IMAGE_IDS_PATH]:
        if not p.exists():
            print(f"Missing {p}. Run --step build-index first.")
            return

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(IMAGE_IDS_PATH) as f:
        image_ids = json.load(f)

    device = get_device()
    model, processor = load_model(device)

    if text:
        queries = [text]
        poem_name = "custom_text"
        print(f"Query text: \"{text[:80]}{'...' if len(text) > 80 else ''}\"\n")
    elif gutenberg_id is not None:
        lines = load_poem_lines(gutenberg_id)
        print(f"Poem {gutenberg_id}: {len(lines)} lines\n")
        queries = (
            split_into_stanzas(lines)
            if use_stanzas
            else chunk_lines(lines, chunk_size)
        )
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

        chunk_results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            img_rel = image_ids[idx]
            img_src = IMAGES_DIR / img_rel

            # Copy retrieved image into output dir for easy browsing
            dest_name = f"chunk{i:03d}_rank{rank+1}_{Path(img_rel).name}"
            dest = out_dir / dest_name
            if img_src.exists():
                shutil.copy2(str(img_src), str(dest))

            chunk_results.append({
                "rank": rank + 1,
                "score": float(score),
                "image_id": img_rel,
                "output_file": dest_name,
            })

        results.append({
            "chunk_index": i,
            "query_text": query,
            "top_k": chunk_results,
        })

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
            )

    print("Done.")


if __name__ == "__main__":
    main()

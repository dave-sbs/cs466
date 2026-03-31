"""
download_images.py — Download nature/landscape images from HuggingFace.

Uses the `mertcobanov/nature-dataset` dataset (50K nature images with captions).
Images are saved to data/images/ as JPEGs with a manifest.csv index.

Usage:
    python download_images.py                  # Download first 2000 images (default)
    python download_images.py --limit 500      # Smaller prototype set
    python download_images.py --limit 0        # Download all 50K images
    python download_images.py --output_dir data/images
"""

import os
import csv
import argparse
from pathlib import Path

from datasets import load_dataset
from PIL import Image


DATASET_NAME = "mertcobanov/nature-dataset"
DEFAULT_LIMIT = 2000
MANIFEST_FILENAME = "manifest.csv"
MANIFEST_FIELDS = ["filename", "caption", "index"]


def load_existing_manifest(manifest_path: Path) -> set:
    """Return set of filenames already saved (for resumability)."""
    if not manifest_path.exists():
        return set()
    with open(manifest_path, newline="", encoding="utf-8") as f:
        return {row["filename"] for row in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser(
        description=f"Download nature images from HuggingFace ({DATASET_NAME})."
    )
    parser.add_argument(
        "--output_dir",
        default="data/images",
        help="Directory to save images (default: data/images)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Max images to download (default: {DEFAULT_LIMIT}; 0 = all)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_FILENAME

    existing = load_existing_manifest(manifest_path)
    print(f"Resuming: {len(existing)} images already saved.\n")

    print(f"Loading dataset '{DATASET_NAME}' from HuggingFace...")
    dataset = load_dataset(DATASET_NAME, split="train")
    total = len(dataset)
    limit = args.limit if args.limit > 0 else total
    target = min(limit, total)
    print(f"Dataset size: {total:,} images. Target: {target:,} images.\n")

    append_mode = "a" if existing else "w"
    manifest_file = open(manifest_path, append_mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
    if not existing:
        writer.writeheader()

    saved = len(existing)
    skipped = 0

    for i, sample in enumerate(dataset):
        if saved - len(existing) + len(existing) >= target and i >= target:
            break
        if i >= target:
            break

        filename = f"{i:05d}.jpg"
        if filename in existing:
            skipped += 1
            continue

        dest = output_dir / filename
        try:
            img = sample["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = img.convert("RGB")
            img.save(dest, "JPEG", quality=90)
        except Exception as e:
            print(f"  Skipping index {i}: {e}")
            continue

        caption = sample.get("text", "") or ""
        writer.writerow({"filename": filename, "caption": caption, "index": i})
        existing.add(filename)
        saved += 1

        if saved % 100 == 0:
            manifest_file.flush()
            print(f"  Saved {saved}/{target} images...", end="\r")

    manifest_file.close()
    print(f"\nDone. {saved} total images in {output_dir}/")
    print(f"Manifest: {manifest_path}")
    if skipped:
        print(f"Skipped {skipped} already-downloaded images.")


if __name__ == "__main__":
    main()

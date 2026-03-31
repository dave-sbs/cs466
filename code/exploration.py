import os
import random
import pandas as pd
import numpy as np
from constants import POEM_LEN_DICT
from collections import defaultdict


def load_df():
    df = pd.read_parquet(
        "hf://datasets/biglam/gutenberg-poetry-corpus/data/train-00000-of-00001-fa9fb9e1f16eed7e.parquet"
    )
    return df


def write_to_file(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(data))


def sample_individual_poem(df, output_dir, gutenberg_id):
    poem = df[df["gutenberg_id"] == gutenberg_id]["line"].tolist()
    filename = os.path.join(output_dir, f"poem_{gutenberg_id}.txt")
    write_to_file(poem, filename)
    return filename


def get_poem_lengths(df):
    """Build a dict mapping line count -> list of gutenberg IDs."""
    gutenberg_ids = df["gutenberg_id"].unique()
    print(f"There are {len(gutenberg_ids)} unique gutenberg ids\n")

    poem_len_dict = defaultdict(list)
    for gid in gutenberg_ids:
        poem = df[df["gutenberg_id"] == gid]["line"]
        poem_len_dict[len(poem)].append(int(gid))

    print(poem_len_dict)


def analyze_dataset():
    """Print summary statistics about the poem length distribution."""
    lengths = sorted(POEM_LEN_DICT.keys())
    total_poems = sum(len(ids) for ids in POEM_LEN_DICT.values())

    all_lengths = []
    for length, ids in POEM_LEN_DICT.items():
        all_lengths.extend([length] * len(ids))
    all_lengths = np.array(all_lengths)

    print(f"Total poems:       {total_poems}")
    print(f"Unique lengths:    {len(lengths)}")
    print(f"Min length:        {all_lengths.min()} lines")
    print(f"Max length:        {all_lengths.max()} lines")
    print(f"Mean length:       {all_lengths.mean():.1f} lines")
    print(f"Median length:     {np.median(all_lengths):.0f} lines")
    print(f"Std dev:           {all_lengths.std():.1f} lines")
    print()

    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        print(f"  {p}th percentile: {int(np.percentile(all_lengths, p))} lines")

    print(f"\nLength distribution (buckets):")
    buckets = [
        (0, 50), (51, 100), (101, 250), (251, 500),
        (501, 1000), (1001, 2500), (2501, 5000),
        (5001, 10000), (10001, 60000),
    ]
    for lo, hi in buckets:
        count = sum(
            len(ids) for length, ids in POEM_LEN_DICT.items()
            if lo <= length <= hi
        )
        print(f"  {lo:>6}-{hi:>6} lines: {count} poems")


def sample_poems(df, n=5, min_lines=50, max_lines=500, seed=42, output_dir="samples"):
    """Sample n random poems within a line-count range and save each to a file."""
    candidates = []
    for length, ids in POEM_LEN_DICT.items():
        if min_lines <= length <= max_lines:
            for gid in ids:
                candidates.append((gid, length))

    if not candidates:
        print(f"No poems found with {min_lines}-{max_lines} lines.")
        return

    random.seed(seed)
    chosen = random.sample(candidates, min(n, len(candidates)))

    os.makedirs(output_dir, exist_ok=True)

    print(f"Sampling {len(chosen)} poems ({min_lines}-{max_lines} lines) -> {output_dir}/\n")
    for gid, length in chosen:
        lines = df[df["gutenberg_id"] == gid]["line"].tolist()
        filename = os.path.join(output_dir, f"poem_{gid}.txt")
        write_to_file(lines, filename)
        print(f"  poem_{gid}.txt  ({length} lines)")

    print(f"\nDone. Files saved to {output_dir}/")


def main():
    print("=== Dataset Analysis ===\n")
    analyze_dataset()

    # print("\n=== Sampling Individual Poem ===\n")
    # df = load_df()
    # sample_individual_poem(df, "samples", 24869)

    # print("\n=== Sampling Poems ===\n")
    # df = load_df()
    # sample_poems(df)


if __name__ == "__main__":
    main()

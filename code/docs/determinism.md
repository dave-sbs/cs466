# Determinism in the Dream Pipeline

> **TL;DR:** The CPU-mock path is bit-exact reproducible. The real
> SDXL + RIFE path on GPU is **not** bit-exact across machines and
> driver versions; we achieve *perceptual* reproducibility via seeded
> prompts and pinned model revisions.

## What we control

| Layer           | Determinism guarantee                                            |
|-----------------|-------------------------------------------------------------------|
| Seeds           | `stanza_seed(gutenberg_id, stanza_idx)` — splitmix-style hash      |
| Prompts         | Built from LLM JSONL (`visual_scene`, `dominant_colors`) — stable |
| Model identity  | `DreamRunConfig.model_id` + `revision` pinned to a HF commit SHA  |
| Keyframes       | `keyframe_manifest.json` records seed + sha256 per stanza         |
| Segment plan    | Pure function of `mood_arc` + `rife_depth` + `fps`               |
| Frames on disk  | Zero-padded filenames (`frame_000123.png`) for stable ffmpeg input|
| Metadata        | `meta.json` captures git_sha, library versions, CUDA device name  |

## What we do NOT control

- **cuDNN / cuBLAS** non-determinism: matrix multiplies on A100 vs T4
  can differ by a few ULP. Two runs on the same seed + same GPU *should*
  match, but we don't assert pixel-exactness in CI.
- **RIFE** forward pass uses `torch.nn.functional.grid_sample`, which is
  non-deterministic on CUDA by default. See
  [PyTorch determinism notes](https://pytorch.org/docs/stable/notes/randomness.html).
- **FFmpeg** H.264 encoding is deterministic given the same input and
  flags, but the x264 version baked into the Colab image is not pinned.
- **HuggingFace model weights**: if `revision` is `None`, we pick up
  whatever `main` points at. Always pin `revision` in the final video.

## How to reproduce a past run

1. Read `run_dir/meta.json` — the `git_sha`, `model_id`, `revision`,
   and `run_config` are everything you need.
2. `git checkout <git_sha>` in a matching worktree.
3. Recreate the venv from `requirements.txt` (versions pinned in the
   `meta.json` env block are informational; `requirements.txt` is the
   source of truth).
4. Re-run the CLI with the same `DreamRunConfig`.
5. On the mock path, keyframe sha256s **must** match bit-exactly.
6. On the GPU path, compare `keyframe_manifest.json` seeds and prompts,
   and eyeball the MP4 output — do not expect bitwise equality.

## Checking for drift

- `dream_preflight --json` + `jq` gives you a stable input shape.
- `keyframe_manifest.json` has a per-stanza `sha256` — diff two runs to
  see exactly which stanzas drifted.
- `meta.json` → `env.torch_version` / `cuda_device` often explains
  cross-machine drift at a glance.

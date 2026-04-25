# Wan 2.1 FLF2V (first-last-frame) — Colab setup

This project includes a small prototype wrapper around the official Diffusers
Wan 2.1 **First-Last-Frame-to-Video** pipeline.

Model: `Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers`  
Pipeline: `diffusers.WanImageToVideoPipeline`

## Colab notes

- **VRAM**: the 14B model is large. An **A100** is strongly recommended.
- **T4**: may OOM or be extremely slow unless you reduce resolution and lean on CPU offload.
- **HF auth**: if the model is gated in your environment, set `HF_TOKEN` in Colab Secrets.

## Install (Colab cell)

```bash
pip install -q --upgrade diffusers transformers accelerate safetensors ftfy torchvision
```

If you previously installed CPU-only torch by accident, reinstall the CUDA build
for your runtime (example for cu121):

```bash
pip uninstall -y -q torch torchvision torchaudio
pip install -q --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

Restart runtime after torch changes.

## Run one transition clip (prototype)

From the repo root (the directory that contains `code/`):

```python
%run code/scripts/run_wan_transition_9825.py \
  --first output/dream_runs/9825_sdxl_v1/keyframes/kf_000.png \
  --last output/dream_runs/9825_sdxl_v1/keyframes/kf_001.png \
  --prompt "snow and ice fragment into violet petals, then rebuild into spring forest, organic cinematic morphing transition, continuous one-shot camera movement" \
  --output output/wan_transitions/kf_000_to_001.mp4
```

## Run from Python

```python
from pathlib import Path
from dream_wan import WanFlf2vConfig, generate_wan_transition

out = generate_wan_transition(
    first_image=Path("output/dream_runs/9825_sdxl_v1/keyframes/kf_000.png"),
    last_image=Path("output/dream_runs/9825_sdxl_v1/keyframes/kf_001.png"),
    prompt="snow and ice fragment into violet petals, then rebuild into spring forest, organic cinematic morphing transition",
    output_path=Path("output/wan_transitions/kf_000_to_001.mp4"),
    cfg=WanFlf2vConfig(num_frames=81, guidance_scale=5.5),
)
print(out)
```


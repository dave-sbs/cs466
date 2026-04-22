# Practical-RIFE setup for the dream pipeline

The dream pipeline uses [Practical-RIFE](https://github.com/hzwer/Practical-RIFE)
for keyframe interpolation. It is not a PyPI package — install it by
cloning the repo and downloading model weights.

## Colab / Linux setup (recommended for MVP)

```bash
# 1. Clone
cd /content
git clone https://github.com/hzwer/Practical-RIFE.git
export RIFE_ROOT=/content/Practical-RIFE

# 2. Download weights. Practical-RIFE's README links a Google Drive
#    archive (commonly RIFE_trained_model_v4.*.zip). Unzip into
#    RIFE_ROOT/train_log/ so RIFE_ROOT/train_log/flownet.pkl exists.
# 3. Install deps
pip install --quiet torch==2.0.1 torchvision==0.15.2 \
    scipy tqdm numpy opencv-python
```

After setup, verify:

```bash
ls "$RIFE_ROOT"/train_log/flownet.pkl
```

## Using the prototype driver script

Once `RIFE_ROOT` is set and weights are installed, you can run the prototype
end-to-end script (poem 9825) with RIFE transitions:

```bash
%run code/scripts/run_dream_9825.py --use-rife --force
```

## Integration with dream_interp

```python
from dream_interp import RifeInterpolator, CrossfadeInterpolator

try:
    interp = RifeInterpolator(rife_root=os.environ["RIFE_ROOT"])
except Exception:
    interp = CrossfadeInterpolator()  # deterministic fallback
```

The orchestration module (`code/dream_render.py`) prefers RIFE when
`RIFE_ROOT` is set and falls back to crossfade otherwise.

## Notes

- RIFE's outputs on GPU are **not byte-identical** across runs even at
  the same seed; this is inherent to CUDA/cuDNN non-determinism. For
  reproducibility demos, use the mock/crossfade path — see
  `brainstorm/dream-pipeline-sprints.md` S10-T4.
- RIFE's license is non-commercial research; respect it when
  redistributing videos.

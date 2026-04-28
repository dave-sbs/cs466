# HuggingFace auth + model cache for the dream pipeline

SDXL weights (`stabilityai/stable-diffusion-xl-base-1.0`) are **gated**
on HuggingFace. Before the pipeline can download them you must:

1. Accept the license on the model page:
   https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
2. Create a read token:
   https://huggingface.co/settings/tokens
3. Authenticate in your environment.

## Colab

Use the **Secrets** sidebar in Colab (key icon) to store `HF_TOKEN`.
Do **not** print the token or commit it to a notebook cell.

```python
from google.colab import userdata
import os
import huggingface_hub

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
huggingface_hub.login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
```

## Local / CI

```bash
export HF_TOKEN=hf_xxx   # or use `huggingface-cli login`
export HF_HOME=/path/to/cache   # optional, default ~/.cache/huggingface
```

Environment variables recognized by diffusers / huggingface_hub:

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | Auth token for gated downloads |
| `HF_HOME` | Root for models, tokens, datasets cache |
| `HF_HUB_OFFLINE` | Set to `1` to prevent any network call |
| `TRANSFORMERS_CACHE` | Legacy alias for `HF_HOME/hub` |

## Pinning model revisions

For reproducibility the dream pipeline accepts an explicit `revision=` argument.
Record the commit SHA from the HuggingFace "Files and versions" tab.
It will appear in the generated `keyframe_manifest.json` and `meta.json`.

## Disk space

SDXL base (fp16) is ~7 GB. Colab Pro's `/root/.cache` is sufficient for
one model. Clear between runs if you add a refiner:

```python
!rm -rf ~/.cache/huggingface/hub/models--stabilityai--*
```

## Never commit

- `HF_TOKEN` in any form (notebook output, CLI history).
- `.cache/` directories.
- Downloaded `.safetensors` files.

`.gitignore` already excludes these. Run `nbstripout` before commit.

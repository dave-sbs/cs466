# Notebook & secret hygiene

This repo aims to keep the working tree free of secrets and large
generated artifacts. Follow these rules when committing.

## Secrets

- **Never** commit `HF_TOKEN`, `OPENROUTER_API_KEY`, Google Drive tokens,
  or any `.env` file.
- Prefer environment variables or the Colab "Secrets" sidebar.
- Before pushing, `grep -RniI "hf_\|sk-\|Bearer " . --exclude-dir=venv`
  on a freshly modified working tree to catch accidents.

## Notebooks

- Clear all outputs before committing:
  ```bash
  pip install --quiet nbstripout
  nbstripout code/*.ipynb
  ```
- Pin large `pip install` cells with `==` where practical; comment the
  reason (usually: reproducibility / Colab nightly drift).
- Do **not** print API tokens in notebook cells. If a cell calls
  `huggingface_hub.login()`, wrap it so the token is read from
  `os.environ` without echoing.

## Outputs

- Generated images, frames, `.mp4`, run manifests, and HF/RIFE caches
  are gitignored (see `.gitignore`). If a new output directory is
  introduced, add it to `.gitignore` in the same commit.

## Licensing checklist

- Gutenberg corpus: public domain, but attribute the source dataset in
  `code/README.md`.
- Image dataset: record its license (check HF dataset card) alongside
  any distributed sample.
- SDXL: redistributing generated media must respect the Stability AI
  license; document in user-facing output copy.

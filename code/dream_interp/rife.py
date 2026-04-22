"""RifeInterpolator — Practical-RIFE bridge.

Heavy imports (``torch``, ``sys.path.append(RIFE_ROOT)``) happen inside
``__call__`` so that importing ``dream_interp`` on CPU CI does not
require RIFE to be installed.

Set ``RIFE_ROOT`` to the directory containing Practical-RIFE's
``inference.py`` and model weights. See ``docs/rife_setup.md``.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from PIL import Image

from .base import Interpolator, InterpolatorError, assert_within_frame_cap
from dream_frames.plan import rife_intermediate_count


class RifeInterpolator:
    """Interpolator backed by Practical-RIFE (https://github.com/hzwer/Practical-RIFE).

    Heavy imports (torch + Practical-RIFE modules) happen lazily in the
    first call so CPU-only CI can import ``dream_interp``.
    """

    def __init__(self, rife_root: str | Path | None = None) -> None:
        self.rife_root = Path(rife_root) if rife_root else None
        self._model = None
        self._model_lock = threading.Lock()
        self._path_patched = False

    def _resolve_root(self) -> Path:
        root = self.rife_root or os.environ.get("RIFE_ROOT")
        if not root:
            raise InterpolatorError(
                "RIFE not configured: pass rife_root=... or set RIFE_ROOT "
                "env var. See code/docs/rife_setup.md."
            )
        root_p = Path(root)
        if not root_p.exists():
            raise InterpolatorError(
                f"RIFE_ROOT does not exist: {root_p}"
            )
        return root_p

    def _validate_install(self, root: Path) -> None:
        # Fail fast on missing weights without importing any RIFE code.
        flownet = root / "train_log" / "flownet.pkl"
        if not flownet.exists():
            raise InterpolatorError(
                "Practical-RIFE weights not found. Expected "
                f"{flownet} to exist. See code/docs/rife_setup.md."
            )

    def _ensure_rife_on_path(self, root: Path) -> None:
        if self._path_patched:
            return
        sys.path.insert(0, str(root))
        self._path_patched = True

    def _load_model(self, root: Path):
        """Load and cache the RIFE model, returning the model object."""
        # Double-checked locking: avoids repeated loads in the same process.
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model

            self._ensure_rife_on_path(root)
            try:
                import torch  # noqa: WPS433
            except Exception as e:  # pragma: no cover
                raise InterpolatorError("torch is required for RIFE") from e

            # Practical-RIFE has moved imports across versions; try a few.
            Model = None
            import_errors: list[str] = []
            for mod_path in (
                "train_log.RIFE_HDv3",
                "model.RIFE_HDv3",
                "model.RIFE",
            ):
                try:
                    mod = __import__(mod_path, fromlist=["Model"])
                    Model = getattr(mod, "Model", None)
                    if Model is not None:
                        break
                except Exception as e:  # pragma: no cover
                    import_errors.append(f"{mod_path}: {e}")

            if Model is None:
                raise InterpolatorError(
                    "Unable to import Practical-RIFE Model. "
                    "Tried: train_log.RIFE_HDv3, model.RIFE_HDv3, model.RIFE. "
                    f"Errors: {import_errors}"
                )

            model = Model()
            try:
                model.load_model(str(root / "train_log"), -1)
            except Exception as e:  # pragma: no cover
                raise InterpolatorError(
                    "Failed to load RIFE weights from train_log/. "
                    "Ensure flownet.pkl exists under RIFE_ROOT/train_log."
                ) from e

            try:
                model.eval()
            except Exception:
                pass
            try:
                model.device()
            except Exception:
                # Some versions expose device via torch only; ignore.
                pass

            # Store for subsequent calls.
            self._model = model
            return self._model

    @staticmethod
    def _pad_to_32(t, *, torch):
        _, _, h, w = t.shape
        pad_h = (32 - (h % 32)) % 32
        pad_w = (32 - (w % 32)) % 32
        if pad_h == 0 and pad_w == 0:
            return t, h, w
        # pad format: (left, right, top, bottom)
        import torch.nn.functional as F  # noqa: WPS433

        padded = F.pad(t, (0, pad_w, 0, pad_h), mode="replicate")
        return padded, h, w

    @staticmethod
    def _pil_to_tensor(img: Image.Image, *, torch, device):
        if img.mode != "RGB":
            img = img.convert("RGB")
        import numpy as np  # noqa: WPS433

        arr = np.asarray(img).astype("float32") / 255.0  # HWC
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1CHW
        return t.to(device=device)

    @staticmethod
    def _tensor_to_pil(t, *, torch, orig_h: int, orig_w: int) -> Image.Image:
        t = t.detach().clamp(0.0, 1.0)
        t = t[0, :, :orig_h, :orig_w]  # CHW crop
        t = (t * 255.0).round().to(dtype=torch.uint8)
        import numpy as np  # noqa: WPS433

        arr = t.permute(1, 2, 0).cpu().numpy()  # HWC
        return Image.fromarray(np.asarray(arr), mode="RGB")

    def _recurse(self, model, a_t, b_t, *, depth: int):
        if depth <= 0:
            return []
        mid = model.inference(a_t, b_t)
        left = self._recurse(model, a_t, mid, depth=depth - 1)
        right = self._recurse(model, mid, b_t, depth=depth - 1)
        return left + [mid] + right

    def __call__(
        self,
        a: Image.Image,
        b: Image.Image,
        out_dir: Path,
        depth: int,
        prefix: str = "mid_",
    ) -> list[Path]:
        # Fail fast on bad inputs before any heavy import.
        n_mids = rife_intermediate_count(depth)
        assert_within_frame_cap(n_mids)
        if a.size != b.size:
            raise InterpolatorError(
                f"image sizes differ: {a.size} vs {b.size}"
            )

        root = self._resolve_root()
        self._validate_install(root)

        try:
            import torch  # noqa: WPS433
        except Exception as e:  # pragma: no cover
            raise InterpolatorError("torch is required for RIFE") from e

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self._load_model(root)

        out_dir.mkdir(parents=True, exist_ok=True)

        a_t = self._pil_to_tensor(a, torch=torch, device=device)
        b_t = self._pil_to_tensor(b, torch=torch, device=device)
        a_t, h, w = self._pad_to_32(a_t, torch=torch)
        b_t, _, _ = self._pad_to_32(b_t, torch=torch)

        mids = self._recurse(model, a_t, b_t, depth=depth)
        paths: list[Path] = []
        for i, t in enumerate(mids):
            img = self._tensor_to_pil(t, torch=torch, orig_h=h, orig_w=w)
            p = out_dir / f"{prefix}{i:04d}.png"
            img.save(p, format="PNG")
            paths.append(p)
        return paths


_proto_check: Interpolator = RifeInterpolator()

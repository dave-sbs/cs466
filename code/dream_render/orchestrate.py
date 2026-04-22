"""End-to-end orchestration: LLM record -> MP4.

Responsibilities:

1. Load + validate the LLM record and retrieval manifest.
2. Resolve top-1 init images per stanza.
3. Generate one SDXL keyframe per stanza (or use a mock/crossfade when
   ``DREAM_MOCK_GPU=1``). Skip stanzas whose keyframe_manifest entry
   exists + SHA matches (resume).
4. Lay down hold frames via Ken Burns + transition frames via the
   interpolator.
5. Call ffmpeg to assemble the MP4.
6. Emit keyframe_manifest.json, frame_manifest.json, and (optionally)
   meta.json.

Keeping this a single function is intentional: the notebook calls
``render_dream_video(config)`` in one cell.
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from dream_data import (
    DreamDataError,
    build_sdxl_prompt,
    load_last_llm_record,
    load_retrieval_manifest,
    mood_to_strength,
    pair_scenes_with_chunks,
    resolve_top1_image_path,
    stanza_intensity,
    stanza_seed,
    validate_llm_record,
    DEFAULT_NEGATIVE_PROMPT,
)
from dream_frames import (
    apply_ken_burns,
    build_segment_plan,
)
from dream_ffmpeg import (
    FfmpegError,
    ffmpeg_render_frames_to_mp4,
    frame_filename,
)
from dream_interp import CrossfadeInterpolator, Interpolator
from dream_sdxl import (
    KeyframeEntry,
    KeyframeManifest,
    KeyframeProvider,
    MockKeyframeProvider,
    generate_keyframe,
    load_keyframe_manifest,
    sha256_file,
    write_keyframe_manifest,
)
from .artifacts import DreamArtifacts, FrameManifest, write_frame_manifest
from dream_meta import write_meta


log = logging.getLogger("dream_render")


@dataclass
class RenderConfig:
    """Fully-specified inputs for one dream run.

    Keep this small: paths + toggles only. Numerical pipeline knobs live
    at call-time with sensible defaults so the notebook cell is short.
    """

    gutenberg_id: int
    data_root: Path
    run_dir: Path
    llm_jsonl: Path | None = None
    manifest_path: Path | None = None
    fps: int = 30
    rife_depth: int = 4
    max_zoom: float = 1.05
    image_size: tuple[int, int] = (1024, 1024)
    start_stanza: int = 0
    end_stanza: int | None = None
    force: bool = False
    encode_mp4: bool = True
    fade_in_seconds: float = 0.5
    fade_out_seconds: float = 0.5
    num_inference_steps: int = 30
    guidance_scale: float = 7.0
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    revision: str | None = None

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        self.run_dir = Path(self.run_dir)
        if self.llm_jsonl is not None:
            self.llm_jsonl = Path(self.llm_jsonl)
        if self.manifest_path is not None:
            self.manifest_path = Path(self.manifest_path)


def _default_llm_jsonl(cfg: RenderConfig) -> Path:
    return cfg.data_root / "exploration_output" / "llm_analysis.jsonl"


def _default_manifest(cfg: RenderConfig) -> Path:
    return (
        cfg.data_root
        / "output"
        / "retrieval_results"
        / f"poem_{cfg.gutenberg_id}"
        / "retrieval_manifest.json"
    )


def _select_provider(
    *, mock_gpu: bool, provider_factory: Callable[[], KeyframeProvider] | None
) -> KeyframeProvider:
    if mock_gpu:
        log.info("DREAM_MOCK_GPU=1 -> using MockKeyframeProvider")
        return MockKeyframeProvider()
    if provider_factory is not None:
        return provider_factory()
    # Default: also fall back to mock. Real SDXL requires the caller to
    # pass a provider_factory (keeps this module diffusers-free).
    log.info("No provider_factory supplied; defaulting to MockKeyframeProvider")
    return MockKeyframeProvider()


def _select_interpolator(
    *, interpolator: Interpolator | None
) -> Interpolator:
    return interpolator if interpolator is not None else CrossfadeInterpolator()


def _load_init_image(path: Path, size: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.LANCZOS)
    return img


def render_dream_video(
    cfg: RenderConfig,
    *,
    provider_factory: Callable[[], KeyframeProvider] | None = None,
    interpolator: Interpolator | None = None,
    mock_gpu: bool | None = None,
) -> DreamArtifacts:
    """Run the full pipeline and return ``DreamArtifacts``.

    Parameters
    ----------
    cfg : RenderConfig
        All input paths + knobs.
    provider_factory : callable, optional
        Returns a ``KeyframeProvider``. Called once. If ``None`` and
        ``mock_gpu=False``, the mock provider is used (never SDXL from
        this module — the notebook supplies the real factory).
    interpolator : Interpolator, optional
        Defaults to ``CrossfadeInterpolator()``.
    mock_gpu : bool, optional
        Override ``DREAM_MOCK_GPU`` env detection.

    Raises
    ------
    DreamDataError
        On any invalid input (missing LLM record, pair mismatch, ...).
    FfmpegError
        If ``encode_mp4=True`` and ffmpeg fails.
    """
    if mock_gpu is None:
        mock_gpu = os.environ.get("DREAM_MOCK_GPU") == "1"

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    keyframes_dir = run_dir / "keyframes"
    frames_dir = run_dir / "frames"
    keyframes_dir.mkdir(exist_ok=True)
    frames_dir.mkdir(exist_ok=True)
    keyframe_manifest_path = run_dir / "keyframe_manifest.json"
    frame_manifest_path = run_dir / "frame_manifest.json"

    # 1. Load + validate
    jsonl_p = cfg.llm_jsonl or _default_llm_jsonl(cfg)
    manifest_p = cfg.manifest_path or _default_manifest(cfg)

    record = load_last_llm_record(jsonl_p, gutenberg_id=cfg.gutenberg_id)
    if record is None:
        raise DreamDataError(
            f"no LLM record for gutenberg_id={cfg.gutenberg_id} in {jsonl_p}"
        )
    validate_llm_record(record)
    manifest = load_retrieval_manifest(manifest_p)
    pairs = pair_scenes_with_chunks(record, manifest)

    num_stanzas = len(pairs)
    end = cfg.end_stanza if cfg.end_stanza is not None else num_stanzas
    stanza_range = range(cfg.start_stanza, min(end, num_stanzas))
    if not stanza_range:
        raise DreamDataError(
            f"empty stanza range [{cfg.start_stanza}, {end}) for "
            f"num_stanzas={num_stanzas}"
        )

    provider = _select_provider(
        mock_gpu=mock_gpu, provider_factory=provider_factory
    )
    interp = _select_interpolator(interpolator=interpolator)

    # 2. Resume: load prior manifest if present
    existing: dict[int, KeyframeEntry] = {}
    if keyframe_manifest_path.exists() and not cfg.force:
        try:
            prior = load_keyframe_manifest(keyframe_manifest_path)
            existing = {e.stanza_index: e for e in prior.entries}
        except Exception as e:
            log.warning("ignoring unreadable prior manifest: %s", e)

    # 3. Generate keyframes.
    # We key keyframe_images by *local* position in ``stanza_range`` so
    # they line up with indices in the SegmentPlan (which is built from
    # a list of intensities).
    intensities: list[int] = []
    entries: list[KeyframeEntry] = []
    keyframe_images: list[Path] = []

    for local_idx, stanza_idx in enumerate(stanza_range):
        scene, chunk = pairs[stanza_idx]
        intensity = stanza_intensity(stanza_idx, num_stanzas, record["mood_arc"])
        intensities.append(intensity)
        strength = mood_to_strength(intensity)
        seed = stanza_seed(cfg.gutenberg_id, stanza_idx)
        prompt, prompt_2 = build_sdxl_prompt(scene)
        image_id = chunk["top_k"][0]["image_id"]
        init_path = resolve_top1_image_path(cfg.data_root, image_id)
        out_path = keyframes_dir / f"kf_{stanza_idx:03d}.png"

        log.info(
            "stanza %d/%d: intensity=%d strength=%.2f seed=%d image_id=%s",
            stanza_idx,
            num_stanzas - 1,
            intensity,
            strength,
            seed,
            image_id,
        )
        log.info(
            "stanza %d/%d: prompt=%r",
            stanza_idx,
            num_stanzas - 1,
            (prompt[:200] + "..." if len(prompt) > 200 else prompt),
        )

        prior = existing.get(stanza_idx)
        if (
            prior is not None
            and out_path.exists()
            and prior.done
            and prior.sha256 == sha256_file(out_path)
        ):
            log.info("stanza %d: skipping (resume)", stanza_idx)
            entries.append(prior)
            keyframe_images.append(out_path)
            continue

        init_image = _load_init_image(init_path, cfg.image_size)
        generate_keyframe(
            provider,
            prompt=prompt,
            init_image=init_image,
            strength=strength,
            seed=seed,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            prompt_2=prompt_2,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            output_path=out_path,
        )
        entries.append(
            KeyframeEntry(
                stanza_index=stanza_idx,
                image_id=image_id,
                prompt=prompt,
                prompt_2=prompt_2,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                strength=strength,
                seed=seed,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                output_path=str(out_path),
                sha256=sha256_file(out_path),
                done=True,
            )
        )
        keyframe_images.append(out_path)
        log.info(
            "stanza %d/%d: wrote %s sha256=%s",
            stanza_idx,
            num_stanzas - 1,
            out_path.name,
            entries[-1].sha256[:8],
        )

    # 4. Write keyframe manifest (after each keyframe loop — guards resume).
    write_keyframe_manifest(
        keyframe_manifest_path,
        KeyframeManifest(
            gutenberg_id=cfg.gutenberg_id,
            model_id=cfg.model_id,
            revision=cfg.revision,
            entries=entries,
        ),
    )

    # 5. Build segment plan; materialize frames
    plan = build_segment_plan(intensities, rife_depth=cfg.rife_depth, fps=cfg.fps)

    # Clean old frames on a full run (force=True) to avoid stale files.
    if cfg.force:
        for f in frames_dir.glob("frame_*.png"):
            f.unlink()

    frame_idx = 0
    rendered_segments: list[dict[str, Any]] = []
    loaded_keyframes: list[Image.Image] = [
        Image.open(p).convert("RGB") for p in keyframe_images
    ]
    stanza_ids = list(stanza_range)

    for seg in plan.segments:
        seg_start = frame_idx
        if seg.kind == "hold":
            kf = loaded_keyframes[seg.stanza_from]
            for f in apply_ken_burns(kf, seg.frame_count, max_zoom=cfg.max_zoom):
                f.save(frames_dir / frame_filename(frame_idx), format="PNG")
                frame_idx += 1
        else:  # transition
            a = loaded_keyframes[seg.stanza_from]
            b = loaded_keyframes[seg.stanza_to]  # type: ignore[index]
            mids = interp(a, b, frames_dir / "__tmp_mids", cfg.rife_depth)
            for src in mids:
                src.replace(frames_dir / frame_filename(frame_idx))
                frame_idx += 1
            shutil.rmtree(frames_dir / "__tmp_mids", ignore_errors=True)
        rendered_segments.append(
            {
                "kind": seg.kind,
                "stanza_from": stanza_ids[seg.stanza_from],
                "stanza_to": (
                    stanza_ids[seg.stanza_to]
                    if seg.stanza_to is not None
                    else None
                ),
                "frame_start": seg_start,
                "frame_end": frame_idx - 1,
                "frame_count": seg.frame_count,
            }
        )

    frame_manifest = FrameManifest(
        total_frames=frame_idx,
        fps=cfg.fps,
        segments=rendered_segments,
    )
    write_frame_manifest(frame_manifest_path, frame_manifest)

    assert frame_idx == plan.total_frames, (
        f"rendered {frame_idx} frames but plan said {plan.total_frames}"
    )

    # 6. MP4
    mp4_path: Path | None = None
    if cfg.encode_mp4:
        mp4_path = run_dir / f"dream_poem_{cfg.gutenberg_id}.mp4"
        ffmpeg_render_frames_to_mp4(
            frames_dir,
            mp4_path,
            fps=cfg.fps,
            total_frames=frame_idx,
            fade_in_seconds=cfg.fade_in_seconds,
            fade_out_seconds=cfg.fade_out_seconds,
        )

    # 7. meta.json for the run (S10-T0/T2/T3).
    meta_path = run_dir / "meta.json"
    run_config_dict = {
        "gutenberg_id": cfg.gutenberg_id,
        "data_root": str(cfg.data_root),
        "run_dir": str(cfg.run_dir),
        "fps": cfg.fps,
        "rife_depth": cfg.rife_depth,
        "max_zoom": cfg.max_zoom,
        "image_size": list(cfg.image_size),
        "num_inference_steps": cfg.num_inference_steps,
        "guidance_scale": cfg.guidance_scale,
        "fade_in_seconds": cfg.fade_in_seconds,
        "fade_out_seconds": cfg.fade_out_seconds,
        "start_stanza": cfg.start_stanza,
        "end_stanza": cfg.end_stanza,
        "force": cfg.force,
        "mock_gpu": mock_gpu,
    }
    write_meta(
        meta_path,
        gutenberg_id=cfg.gutenberg_id,
        model_id=cfg.model_id,
        revision=cfg.revision,
        run_config=run_config_dict,
        artifacts={
            "run_dir": str(run_dir),
            "frames_dir": str(frames_dir),
            "keyframes_dir": str(keyframes_dir),
            "keyframe_manifest": str(keyframe_manifest_path),
            "frame_manifest": str(frame_manifest_path),
            "mp4": str(mp4_path) if mp4_path else "",
        },
    )

    return DreamArtifacts(
        run_dir=run_dir,
        frames_dir=frames_dir,
        keyframes_dir=keyframes_dir,
        keyframe_manifest=keyframe_manifest_path,
        frame_manifest=frame_manifest_path,
        mp4=mp4_path,
        meta=meta_path,
    )

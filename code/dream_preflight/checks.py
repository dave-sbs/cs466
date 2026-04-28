"""Core preflight checks (no argparse / no print statements).

Kept separate from ``__main__`` so tests can drive the logic directly
and the same ``run_preflight`` function can be called from a Colab cell.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dream_data import (
    DreamDataError,
    load_last_llm_record,
    load_retrieval_manifest,
    pair_scenes_with_chunks,
    resolve_top1_image_path,
    stanza_intensity,
    validate_llm_record,
)


DEFAULT_LLM_JSONL_REL = "exploration_output/llm_analysis.jsonl"
DEFAULT_MANIFEST_REL_TEMPLATE = "output/retrieval_results/poem_{gid}/retrieval_manifest.json"


@dataclass
class PreflightReport:
    """Structured result of a preflight run.

    ``ok`` is ``True`` iff ``errors`` is empty. ``warnings`` does not
    affect ``ok``.
    """

    ok: bool = True
    gutenberg_id: int | None = None
    data_root: str | None = None
    num_scenes: int | None = None
    num_chunks: int | None = None
    num_image_paths_checked: int = 0
    plan_total_frames: int | None = None
    plan_duration_seconds: float | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_dict(self) -> dict[str, Any]:
        """Return a dict with sorted keys for stable JSON output."""
        d = asdict(self)
        return {k: d[k] for k in sorted(d.keys())}


def _default_llm_jsonl(data_root: Path) -> Path:
    return data_root / DEFAULT_LLM_JSONL_REL


def _default_manifest(data_root: Path, gutenberg_id: int) -> Path:
    return data_root / DEFAULT_MANIFEST_REL_TEMPLATE.format(gid=gutenberg_id)


def run_preflight(
    *,
    gutenberg_id: int,
    data_root: str | Path,
    llm_jsonl: str | Path | None = None,
    manifest_path: str | Path | None = None,
    plan: bool = False,
    rife_depth: int = 4,
    fps: int = 30,
) -> PreflightReport:
    """Run all preflight checks and return a ``PreflightReport``.

    Never raises — every failure is recorded as an error on the report.
    """
    report = PreflightReport(
        gutenberg_id=gutenberg_id,
        data_root=str(data_root),
    )

    data_root_p = Path(data_root)
    if not data_root_p.exists():
        report.add_error(f"data_root does not exist: {data_root_p}")
        return report

    jsonl_p = Path(llm_jsonl) if llm_jsonl else _default_llm_jsonl(data_root_p)
    manifest_p = (
        Path(manifest_path)
        if manifest_path
        else _default_manifest(data_root_p, gutenberg_id)
    )

    # ----- LLM record -----
    if not jsonl_p.exists():
        report.add_error(f"llm_analysis.jsonl not found at {jsonl_p}")
        record = None
    else:
        record = load_last_llm_record(jsonl_p, gutenberg_id=gutenberg_id)
        if record is None:
            report.add_error(
                f"no LLM record with gutenberg_id={gutenberg_id} in {jsonl_p}"
            )
        else:
            try:
                validate_llm_record(record)
            except DreamDataError as e:
                report.add_error(f"LLM record failed validation: {e}")
                record = None
            else:
                report.num_scenes = len(record["visual_scenes"])

    # ----- Manifest -----
    manifest: dict[str, Any] | None = None
    if not manifest_p.exists():
        report.add_error(f"retrieval_manifest.json not found at {manifest_p}")
    else:
        try:
            manifest = load_retrieval_manifest(manifest_p)
        except DreamDataError as e:
            report.add_error(f"manifest failed to load: {e}")
        else:
            report.num_chunks = len(manifest["results"])

    # ----- Pairing + images -----
    if record is not None and manifest is not None:
        try:
            pairs = pair_scenes_with_chunks(record, manifest)
        except DreamDataError as e:
            report.add_error(f"pairing failed: {e}")
        else:
            missing: list[str] = []
            for scene, chunk in pairs:
                top_k = chunk.get("top_k") or []
                if not top_k:
                    report.add_error(
                        f"chunk {chunk.get('chunk_index')} has empty top_k"
                    )
                    continue
                image_id = top_k[0].get("image_id")
                if not image_id:
                    report.add_error(
                        f"chunk {chunk.get('chunk_index')} top_k[0] missing image_id"
                    )
                    continue
                img_path = resolve_top1_image_path(data_root_p, image_id)
                report.num_image_paths_checked += 1
                if not img_path.exists():
                    missing.append(str(img_path))
            if missing:
                if len(missing) <= 5:
                    for m in missing:
                        report.add_error(f"image not found: {m}")
                else:
                    report.add_error(
                        f"{len(missing)} image paths missing "
                        f"(first: {missing[0]})"
                    )

    # ----- Consistency warning -----
    if (
        report.num_scenes is not None
        and report.num_chunks is not None
        and report.num_scenes != report.num_chunks
    ):
        # Already reported as error above; leave.
        pass

    # ----- Plan estimate (S2-T7) -----
    if plan and record is not None and report.num_scenes:
        try:
            from dream_frames import build_segment_plan

            intensities = [
                stanza_intensity(i, report.num_scenes, record["mood_arc"])
                for i in range(report.num_scenes)
            ]
            seg_plan = build_segment_plan(
                intensities, rife_depth=rife_depth, fps=fps
            )
            report.plan_total_frames = seg_plan.total_frames
            report.plan_duration_seconds = seg_plan.duration_seconds
        except Exception as e:  # pragma: no cover - guarded, surfaced as error
            report.add_error(f"plan estimate failed: {e}")

    return report

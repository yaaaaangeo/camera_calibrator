"""Leak-safe comparison of baseline and Best Subset calibrations.

The two calibration files are treated as immutable.  Only a board pose is
estimated on each frozen hold-out frame; K and D are never optimized here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Callable

import cv2
import numpy as np

from calibration.calibration_io import StandardCalibration, load_standard_calibration
from calibration.models.common import (
    compute_regional_error,
    project_points_for_model,
    regional_edge_average,
    solve_pnp_for_model_robust,
)
from calibration.residual_stats import compute_residual_stats
from calibration.straightness import compute_straightness_residual
from calibration.types import CameraConfig, CameraModelType, Dataset, PatternConfig


class DataLeakageError(ValueError):
    """Raised before evaluation when training and frozen hold-out overlap."""


@dataclass(frozen=True)
class ComparisonTolerance:
    relative: float = 0.05
    absolute_px: float = 0.10
    success_rate_drop: float = 0.05
    minimum_common_frames: int = 5

    def __post_init__(self) -> None:
        if self.relative < 0 or self.absolute_px < 0 or self.success_rate_drop < 0:
            raise ValueError("comparison tolerance는 음수일 수 없습니다.")
        if self.minimum_common_frames < 1:
            raise ValueError("minimum_common_frames는 1 이상이어야 합니다.")


@dataclass
class SplitManifest:
    baseline_training_scene_ids: list[str]
    subset_training_scene_ids: list[str]
    holdout_scene_ids: list[str]
    seed: int | None = None
    source_path: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def _first_list(data: dict[str, Any], *names: str) -> list[str] | None:
    for name in names:
        value = data.get(name)
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
    return None


def load_split_manifest(path: str | Path) -> SplitManifest:
    """Load JSON/YAML manifests while accepting existing train/test aliases."""
    import yaml

    manifest_path = Path(path)
    text = manifest_path.read_text(encoding="utf-8")
    data = json.loads(text) if manifest_path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("split manifest의 최상위는 객체여야 합니다.")
    split = data.get("split") if isinstance(data.get("split"), dict) else data
    baseline = _first_list(
        split, "baseline_training_scene_ids", "training_pool_scene_ids",
        "training_scene_ids", "train_ids",
    )
    subset = _first_list(split, "subset_training_scene_ids", "subset_scene_ids", "best_subset_scene_ids")
    holdout = _first_list(split, "holdout_scene_ids", "test_scene_ids", "test_ids")
    missing = [
        label for label, value in (("baseline training", baseline), ("subset training", subset), ("hold-out", holdout))
        if value is None
    ]
    if missing:
        raise ValueError("split manifest에 필수 scene ID 목록이 없습니다: " + ", ".join(missing))
    return SplitManifest(
        baseline_training_scene_ids=baseline or [],
        subset_training_scene_ids=subset or [],
        holdout_scene_ids=holdout or [],
        seed=split.get("seed", data.get("seed")),
        source_path=str(manifest_path.resolve()),
        raw=data,
    )


def validate_split_manifest(manifest: SplitManifest) -> dict[str, Any]:
    baseline = set(manifest.baseline_training_scene_ids)
    subset = set(manifest.subset_training_scene_ids)
    holdout = set(manifest.holdout_scene_ids)
    overlap = {
        "baseline_holdout": sorted(baseline & holdout),
        "subset_holdout": sorted(subset & holdout),
    }
    if overlap["baseline_holdout"] or overlap["subset_holdout"]:
        raise DataLeakageError(
            "데이터 누수 감지: training/hold-out scene ID가 겹칩니다: "
            + json.dumps(overlap, ensure_ascii=False)
        )
    if not baseline or not subset or not holdout:
        raise ValueError("baseline training, subset training, hold-out 목록은 모두 비어 있지 않아야 합니다.")
    if not subset.issubset(baseline):
        extra = sorted(subset - baseline)
        raise DataLeakageError(
            "Best Subset은 frozen hold-out을 제외한 baseline training pool의 부분집합이어야 합니다: "
            + ", ".join(extra)
        )
    return {"passed": True, "overlap": overlap}


def _sha256(path: str | None) -> str | None:
    if not path:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
            timeout=3,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _model(calibration: StandardCalibration) -> CameraModelType:
    if calibration.model_name is None:
        raise ValueError(f"{calibration.label}: calibration model 정보가 없습니다.")
    return calibration.model_name


def _check_compatibility(
    baseline: StandardCalibration,
    candidate: StandardCalibration,
    camera: CameraConfig,
    pattern: PatternConfig,
) -> CameraModelType:
    baseline_model, candidate_model = _model(baseline), _model(candidate)
    if baseline_model != candidate_model:
        raise ValueError(f"모델이 다릅니다: baseline={baseline_model.value}, candidate={candidate_model.value}")
    for label, calibration in (("baseline", baseline), ("candidate", candidate)):
        if calibration.width is not None and calibration.height is not None:
            if (calibration.width, calibration.height) != (camera.width, camera.height):
                raise ValueError(
                    f"{label} 해상도 {calibration.width}x{calibration.height}와 "
                    f"hold-out 해상도 {camera.width}x{camera.height}가 다릅니다."
                )
    if (
        baseline.width is not None and candidate.width is not None
        and (baseline.width, baseline.height) != (candidate.width, candidate.height)
    ):
        raise ValueError("baseline과 candidate의 해상도가 다릅니다.")
    for key in ("pattern_type", "pattern_squares_x", "pattern_squares_y", "pattern_square_size", "pattern_marker_size", "pattern_dictionary"):
        left, right = baseline.metadata.get(key), candidate.metadata.get(key)
        if left is not None and right is not None and left != right:
            raise ValueError(f"baseline과 candidate의 target geometry가 다릅니다: {key}")
    expected = {
        "pattern_type": pattern.type.value if hasattr(pattern.type, "value") else str(pattern.type),
        "pattern_squares_x": pattern.squares_x,
        "pattern_squares_y": pattern.squares_y,
        "pattern_square_size": pattern.square_size,
        "pattern_marker_size": pattern.marker_size,
        "pattern_dictionary": pattern.dictionary,
    }
    for label, calibration in (("baseline", baseline), ("candidate", candidate)):
        for key, wanted in expected.items():
            actual = calibration.metadata.get(key)
            if actual is not None and wanted is not None and actual != wanted:
                raise ValueError(f"{label} calibration과 hold-out target geometry가 다릅니다: {key}")
    return baseline_model


def _evaluate_model(
    dataset: Dataset,
    holdout_ids: list[str],
    calibration: StandardCalibration,
    camera: CameraConfig,
    pattern: PatternConfig,
    progress: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    model = _model(calibration)
    by_id = {frame.image_info.image_id: frame for frame in dataset.frames}
    per_frame: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    pooled_errors: list[float] = []
    pooled_xs: list[float] = []
    pooled_ys: list[float] = []
    successful_frames = []

    for index, frame_id in enumerate(holdout_ids, 1):
        if cancelled and cancelled():
            raise RuntimeError("비교가 취소되었습니다.")
        if progress:
            progress(f"{calibration.label}: hold-out {index}/{len(holdout_ids)}")
        frame = by_id.get(frame_id)
        if frame is None:
            failures.append({"frame_id": frame_id, "stage": "detection", "reason": "manifest scene not found in dataset"})
            continue
        det = frame.detection
        if det is None or not det.success or det.corners is None or det.object_points is None:
            reason = det.failure_reason if det is not None else "detection result missing"
            failures.append({"frame_id": frame_id, "stage": "detection", "reason": reason or "corner detection failed"})
            continue
        ok, rvec, tvec, reason = solve_pnp_for_model_robust(
            det.object_points, det.corners, calibration.camera_matrix, calibration.distortion, model
        )
        if not ok:
            failures.append({"frame_id": frame_id, "stage": "pose_estimation", "reason": reason})
            continue
        try:
            projected = project_points_for_model(
                det.object_points, rvec, tvec, calibration.camera_matrix, calibration.distortion, model
            )
        except cv2.error as exc:
            failures.append({"frame_id": frame_id, "stage": "pose_estimation", "reason": f"projection failed: {exc}"})
            continue
        detected = np.asarray(det.corners, dtype=np.float64).reshape(-1, 2)
        point_errors = np.hypot(*(detected - projected).T)
        stats = compute_residual_stats(point_errors)
        regional = compute_regional_error(detected[:, 0], detected[:, 1], point_errors, (camera.width, camera.height))
        edge = regional_edge_average(regional)
        straightness, _ = compute_straightness_residual(
            [frame], pattern, calibration.camera_matrix, calibration.distortion, model
        )
        per_frame[frame_id] = {
            "rms": stats.rmse, "median": stats.median, "p95": stats.p95,
            "p99": stats.p99, "max": stats.max, "edge_rms": edge,
            "straightness": straightness,
            # Kept only until common-frame pooled metrics are computed, then
            # removed before serialization.
            "_point_errors": point_errors.tolist(),
            "_xs": detected[:, 0].tolist(),
            "_ys": detected[:, 1].tolist(),
        }
        pooled_errors.extend(point_errors.tolist())
        pooled_xs.extend(detected[:, 0].tolist())
        pooled_ys.extend(detected[:, 1].tolist())
        successful_frames.append(frame)

    def aggregate(ids: list[str]) -> dict[str, Any]:
        rows = [per_frame[fid] for fid in ids if fid in per_frame]
        if not rows:
            return {key: None for key in ("holdout_rms", "median", "p95", "p99", "max", "edge_rms", "straightness")}
        # Headline values use corner-level pooled residuals for all-success; common
        # subsets are aggregated from paired per-frame values below.
        def rms_of(key: str) -> float | None:
            vals = [row[key] for row in rows if row.get(key) is not None]
            return float(np.sqrt(np.mean(np.square(vals)))) if vals else None
        def median_of(key: str) -> float | None:
            vals = [row[key] for row in rows if row.get(key) is not None]
            return float(np.median(vals)) if vals else None
        return {
            "holdout_rms": rms_of("rms"), "median": median_of("median"),
            "p95": median_of("p95"), "p99": median_of("p99"), "max": max((r["max"] for r in rows), default=None),
            "edge_rms": rms_of("edge_rms"), "straightness": median_of("straightness"),
        }

    all_ids = list(per_frame)
    metrics = aggregate(all_ids)
    if pooled_errors:
        stats = compute_residual_stats(pooled_errors)
        regional = compute_regional_error(pooled_xs, pooled_ys, pooled_errors, (camera.width, camera.height))
        metrics.update({
            "holdout_rms": stats.rmse, "median": stats.median, "p95": stats.p95,
            "p99": stats.p99, "max": stats.max, "edge_rms": regional_edge_average(regional),
        })
        metrics["straightness"] = compute_straightness_residual(
            successful_frames, pattern, calibration.camera_matrix, calibration.distortion, model
        )[0]
    detection_failures = sum(row["stage"] == "detection" for row in failures)
    pose_failures = sum(row["stage"] == "pose_estimation" for row in failures)
    return {
        "metrics": metrics, "per_frame": per_frame, "failures": failures,
        "total_frames": len(holdout_ids), "evaluated_frames": len(per_frame),
        "failed_detection_frames": detection_failures,
        "failed_pose_estimation_frames": pose_failures,
        "success_rate": len(per_frame) / len(holdout_ids) if holdout_ids else 0.0,
        "train_rms": calibration.metadata.get("rms_reprojection_error"),
    }


def classify_metric(baseline: float | None, candidate: float | None, tolerance: ComparisonTolerance) -> dict[str, Any]:
    if baseline is None or candidate is None or not (math.isfinite(baseline) and math.isfinite(candidate)):
        return {"baseline": baseline, "candidate": candidate, "delta": None, "relative_change_pct": None, "tolerance": None, "status": "Unavailable", "winner": "N/A"}
    delta = candidate - baseline
    allowed = max(tolerance.absolute_px, abs(baseline) * tolerance.relative)
    status = "Improved" if delta < 0 else ("Maintained" if delta <= allowed else "Regressed")
    relative = (delta / baseline * 100.0) if baseline != 0 else (0.0 if delta == 0 else None)
    winner = "Best Subset" if delta < 0 else ("Baseline" if delta > allowed else "Tie / maintained")
    return {"baseline": baseline, "candidate": candidate, "delta": delta, "relative_change_pct": relative, "tolerance": allowed, "status": status, "winner": winner}


def _paired_statistics(baseline: dict[str, Any], candidate: dict[str, Any], common: list[str]) -> dict[str, Any]:
    deltas = [candidate[frame_id]["rms"] - baseline[frame_id]["rms"] for frame_id in common]
    wins = sum(delta < -1e-12 for delta in deltas)
    losses = sum(delta > 1e-12 for delta in deltas)
    ties = len(deltas) - wins - losses
    ci = None
    evidence = "insufficient evidence" if len(deltas) < 5 else "descriptive"
    if len(deltas) >= 5:
        rng = np.random.default_rng(42)
        arr = np.asarray(deltas)
        samples = rng.choice(arr, size=(2000, arr.size), replace=True)
        ci = [float(v) for v in np.percentile(np.median(samples, axis=1), [2.5, 97.5])]
        evidence = "bootstrap 95% CI excludes zero" if not (ci[0] <= 0 <= ci[1]) else "bootstrap 95% CI includes zero"
    return {
        "common_frame_count": len(common), "subset_wins": wins, "baseline_wins": losses,
        "ties": ties, "median_delta": float(np.median(deltas)) if deltas else None,
        "bootstrap_median_delta_95_ci": ci, "evidence": evidence,
    }


def compare_subset_calibrations(
    dataset: Dataset,
    baseline: StandardCalibration | str,
    candidate: StandardCalibration | str,
    manifest: SplitManifest | str,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    tolerance: ComparisonTolerance = ComparisonTolerance(),
    progress: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Compare immutable K/D on one frozen hold-out and return raw precision data."""
    baseline = load_standard_calibration(baseline) if isinstance(baseline, str) else baseline
    candidate = load_standard_calibration(candidate) if isinstance(candidate, str) else candidate
    manifest = load_split_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
    leakage = validate_split_manifest(manifest)
    recorded_subset = candidate.metadata.get("subset_scene_ids")
    if recorded_subset is not None and set(map(str, recorded_subset)) != set(manifest.subset_training_scene_ids):
        raise DataLeakageError(
            "candidate YAML의 subset_scene_ids와 split manifest의 subset training IDs가 다릅니다."
        )
    model = _check_compatibility(baseline, candidate, camera_config, pattern_config)
    baseline_eval = _evaluate_model(dataset, manifest.holdout_scene_ids, baseline, camera_config, pattern_config, progress, cancelled)
    candidate_eval = _evaluate_model(dataset, manifest.holdout_scene_ids, candidate, camera_config, pattern_config, progress, cancelled)
    common = [fid for fid in manifest.holdout_scene_ids if fid in baseline_eval["per_frame"] and fid in candidate_eval["per_frame"]]

    def common_metrics(evaluation: dict[str, Any]) -> dict[str, Any]:
        rows = evaluation["per_frame"]
        def values(key): return [rows[fid][key] for fid in common if rows[fid].get(key) is not None]
        def rms(key):
            vals = values(key)
            return float(np.sqrt(np.mean(np.square(vals)))) if vals else None
        def med(key):
            vals = values(key)
            return float(np.median(vals)) if vals else None
        errors = [value for fid in common for value in rows[fid].get("_point_errors", [])]
        xs = [value for fid in common for value in rows[fid].get("_xs", [])]
        ys = [value for fid in common for value in rows[fid].get("_ys", [])]
        stats = compute_residual_stats(errors)
        regional = compute_regional_error(xs, ys, errors, (camera_config.width, camera_config.height))
        return {
            "holdout_rms": stats.rmse if stats.n else rms("rms"),
            "median": stats.median if stats.n else med("median"),
            "p95": stats.p95 if stats.n else med("p95"),
            "p99": stats.p99 if stats.n else med("p99"),
            "max": stats.max if stats.n else (max(values("max")) if values("max") else None),
            "edge_rms": regional_edge_average(regional) if stats.n else rms("edge_rms"),
            "straightness": med("straightness"),
        }

    baseline_eval["common_metrics"] = common_metrics(baseline_eval)
    candidate_eval["common_metrics"] = common_metrics(candidate_eval)
    comparisons = {
        metric: classify_metric(baseline_eval["common_metrics"].get(metric), candidate_eval["common_metrics"].get(metric), tolerance)
        for metric in ("holdout_rms", "median", "p95", "p99", "max", "edge_rms", "straightness")
    }
    comparisons["train_rms"] = classify_metric(baseline_eval.get("train_rms"), candidate_eval.get("train_rms"), tolerance)
    rate_delta = candidate_eval["success_rate"] - baseline_eval["success_rate"]
    comparisons["success_rate"] = {
        "baseline": baseline_eval["success_rate"], "candidate": candidate_eval["success_rate"],
        "delta": rate_delta, "relative_change_pct": rate_delta * 100.0,
        "tolerance": tolerance.success_rate_drop,
        "status": "Regressed" if rate_delta < -tolerance.success_rate_drop else ("Improved" if rate_delta > 0 else "Maintained"),
        "winner": "Best Subset" if rate_delta > 0 else ("Baseline" if rate_delta < -tolerance.success_rate_drop else "Tie / maintained"),
    }
    reasons: list[str] = []
    rms_status = comparisons["holdout_rms"]["status"]
    edge_status = comparisons["edge_rms"]["status"]
    if rms_status == "Regressed" or edge_status == "Regressed":
        verdict = "FAIL"
        reasons.append(f"Hold-out RMS={rms_status}, Edge RMS={edge_status}")
    elif rms_status == "Unavailable" or edge_status == "Unavailable":
        verdict = "FAIL"
        reasons.append("공정한 판정에 필요한 Hold-out RMS 또는 Edge RMS가 없습니다.")
    elif comparisons["success_rate"]["status"] == "Regressed":
        verdict = "WARNING"
        reasons.append(f"성공률이 {abs(rate_delta) * 100:.1f}%p 감소했습니다.")
    elif comparisons["p95"]["status"] == "Regressed":
        verdict = "WARNING"
        reasons.append("전체 RMS는 유지됐지만 P95가 허용치를 초과했습니다.")
    else:
        verdict = "PASS"
        reasons.append(f"Hold-out RMS={rms_status}, Edge RMS={edge_status}, 누수 없음")
    if len(common) < tolerance.minimum_common_frames and verdict == "PASS":
        verdict = "WARNING"
        reasons.append(f"공통 평가 프레임이 {len(common)}장으로 최소 {tolerance.minimum_common_frames}장보다 적습니다.")

    per_frame = []
    for fid in manifest.holdout_scene_ids:
        left, right = baseline_eval["per_frame"].get(fid), candidate_eval["per_frame"].get(fid)
        row: dict[str, Any] = {"frame_id": fid, "common_success": left is not None and right is not None}
        for label, value in (("baseline", left), ("candidate", right)):
            for metric in ("rms", "edge_rms", "p95", "p99", "straightness"):
                row[f"{label}_{metric}"] = value.get(metric) if value else None
        row["rms_delta"] = right["rms"] - left["rms"] if left and right else None
        per_frame.append(row)
    for evaluation in (baseline_eval, candidate_eval):
        for row in evaluation["per_frame"].values():
            row.pop("_point_errors", None)
            row.pop("_xs", None)
            row.pop("_ys", None)
    provenance = {
        "baseline_training_scene_ids": manifest.baseline_training_scene_ids,
        "subset_training_scene_ids": manifest.subset_training_scene_ids,
        "holdout_scene_ids": manifest.holdout_scene_ids,
        "counts": {"baseline_training": len(manifest.baseline_training_scene_ids), "subset_training": len(manifest.subset_training_scene_ids), "holdout": len(manifest.holdout_scene_ids)},
        "baseline_model": {"path": baseline.source_path, "sha256": _sha256(baseline.source_path)},
        "candidate_model": {"path": candidate.source_path, "sha256": _sha256(candidate.source_path)},
        "split_seed": manifest.seed, "split_manifest_path": manifest.source_path,
        "overlap_check": leakage, "evaluated_at": datetime.now(timezone.utc).isoformat(), "git_commit_sha": _git_sha(),
    }
    required_provenance = {
        "baseline model path/hash": provenance["baseline_model"]["path"] and provenance["baseline_model"]["sha256"],
        "candidate model path/hash": provenance["candidate_model"]["path"] and provenance["candidate_model"]["sha256"],
        "split seed": provenance["split_seed"],
        "split manifest path": provenance["split_manifest_path"],
        "Git commit SHA": provenance["git_commit_sha"],
    }
    missing_provenance = [name for name, value in required_provenance.items() if value is None or value == ""]
    provenance["missing_required_fields"] = missing_provenance
    if missing_provenance:
        verdict = "FAIL"
        reasons.append("필수 provenance 누락: " + ", ".join(missing_provenance))
    return {
        "schema_version": 1, "verdict": verdict, "verdict_reasons": reasons,
        "model": model.value, "tolerance": asdict(tolerance),
        "edge_roi": {"definition": "existing regional thirds", "normalized_boundaries": [1 / 3, 2 / 3], "regions": ["left", "right", "top", "bottom", "corner"]},
        "provenance": provenance, "baseline": baseline_eval, "candidate": candidate_eval,
        "comparison": comparisons, "paired": _paired_statistics(baseline_eval["per_frame"], candidate_eval["per_frame"], common),
        "per_frame": per_frame,
    }


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def write_subset_comparison_outputs(result: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write raw JSON plus presentation-rounded CSV/Markdown artifacts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out / "subset_comparison_summary.json",
        "summary_csv": out / "subset_comparison_summary.csv",
        "per_frame_csv": out / "subset_comparison_per_frame.csv",
        "failures_csv": out / "subset_comparison_failures.csv",
        "report": out / "subset_comparison_report.md",
    }
    paths["json"].write_text(json.dumps(_clean_json(result), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with paths["summary_csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "baseline", "best_subset", "delta", "change_pct", "tolerance", "status", "winner"])
        for metric, row in result["comparison"].items():
            fmt = lambda value: "" if value is None else f"{value:.3f}" if isinstance(value, (int, float)) else value
            writer.writerow([metric, fmt(row.get("baseline")), fmt(row.get("candidate")), fmt(row.get("delta")), fmt(row.get("relative_change_pct")), fmt(row.get("tolerance")), row.get("status"), row.get("winner")])
    fields = list(result["per_frame"][0]) if result["per_frame"] else ["frame_id", "common_success"]
    with paths["per_frame_csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for raw in result["per_frame"]:
            writer.writerow({key: f"{value:.3f}" if isinstance(value, float) else ("" if value is None else value) for key, value in raw.items()})
    with paths["failures_csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "frame_id", "stage", "reason"])
        writer.writeheader()
        for model in ("baseline", "candidate"):
            for failure in result[model]["failures"]:
                writer.writerow({"model": model, **failure})
    report_lines = [
        "# Best Subset Frozen Hold-out Comparison", "", f"**Overall verdict: {result['verdict']}** — {' '.join(result['verdict_reasons'])}", "",
        "## Data split and leakage check", "", f"- Overlap check: PASS", f"- Baseline training: {result['provenance']['counts']['baseline_training']} scenes", f"- Best Subset training: {result['provenance']['counts']['subset_training']} scenes", f"- Frozen hold-out: {result['provenance']['counts']['holdout']} scenes", f"- Split seed: {result['provenance']['split_seed']}", "",
        "## Model files", "", f"- Baseline: `{result['provenance']['baseline_model']['path']}` (`{result['provenance']['baseline_model']['sha256']}`)", f"- Best Subset: `{result['provenance']['candidate_model']['path']}` (`{result['provenance']['candidate_model']['sha256']}`)", "",
        "## Comparison on common successful frames", "", "| Metric | Baseline | Best Subset | Delta | Change | Result |", "|---|---:|---:|---:|---:|---|",
    ]
    for metric, row in result["comparison"].items():
        fmt = lambda value: "N/A" if value is None else f"{value:.3f}"
        report_lines.append(f"| {metric} | {fmt(row.get('baseline'))} | {fmt(row.get('candidate'))} | {fmt(row.get('delta'))} | {fmt(row.get('relative_change_pct'))}% | {row.get('status')} |")
    report_lines += ["", "## Evaluation success", "", f"- Baseline: {result['baseline']['evaluated_frames']}/{result['baseline']['total_frames']} ({result['baseline']['success_rate'] * 100:.1f}%)", f"- Best Subset: {result['candidate']['evaluated_frames']}/{result['candidate']['total_frames']} ({result['candidate']['success_rate'] * 100:.1f}%)", f"- Common pairwise frames: {result['paired']['common_frame_count']}", f"- Paired evidence: {result['paired']['evidence']}", "", "## Failed frames", ""]
    for model in ("baseline", "candidate"):
        for failure in result[model]["failures"]:
            report_lines.append(f"- {model} / `{failure['frame_id']}` / {failure['stage']}: {failure['reason']}")
    if not result["baseline"]["failures"] and not result["candidate"]["failures"]:
        report_lines.append("- None")
    report_lines += ["", "## Interpretation cautions", "", "Train RMS is descriptive only. The verdict prioritizes frozen hold-out RMS and Edge RMS. K/D were fixed; only per-frame board pose was estimated. Do not tune the subset after inspecting this hold-out result."]
    paths["report"].write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {key: str(value) for key, value in paths.items()}

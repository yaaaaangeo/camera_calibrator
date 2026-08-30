"""
Baseline regression tests for the synthetic calibration pipeline.

The baseline JSON is intentionally small and checked into the repository.  When
an algorithm change is intentional, refresh it with:

    $env:UPDATE_BASELINE = "1"; pytest tests/test_baseline_regression.py -q

Normal test runs compare the current full-pipeline metrics against that frozen
snapshot and fail on meaningful drift.
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

from calibration.compare import run_all_models
from calibration.frame_quality import compute_dataset_quality_score, compute_frame_quality_scores
from calibration.image_quality import evaluate_dataset_image_quality
from calibration.models.common import infer_image_size
from calibration.quality import analyze_dataset_quality, coverage_percentage
from calibration.recommender import compute_final_result, compute_model_scores
from calibration.validation import validate_all_models
from calibration.json_utils import json_safe
from export.json_export import build_export_dict

pytestmark = pytest.mark.slow

BASELINE_PATH = Path(__file__).parent / "baselines" / "synthetic_pipeline_baseline.json"
EXPORT_BASELINE_PATH = Path(__file__).parent / "baselines" / "synthetic_calibration_export_baseline.json"

METRIC_TOLERANCES = {
    "train_rms": (0.08, 0.02),
    "test_rms": (0.08, 0.03),
    "test_p95": (0.08, 0.04),
    "edge_rms": (0.10, 0.04),
    "straightness": (0.10, 0.02),
    "radial_edge": (0.10, 0.04),
    "aic": (0.03, 8.0),
    "bic": (0.03, 8.0),
    "model_score": (0.10, 0.04),
    "observability_score": (0.12, 3.0),
    "undistortion_score": (0.08, 3.0),
    "valid_pixel_ratio": (0.03, 0.02),
    "roi_loss_ratio": (0.06, 0.03),
    "selection_confidence": (0.08, 4.0),
}


def _round_or_none(value: Any, ndigits: int = 6) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, ndigits)


def _canonical_json(value: Any) -> Any:
    """Make the exported JSON baseline stable enough for regression testing."""
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 6)
    if isinstance(value, dict):
        return {k: _canonical_json(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_canonical_json(v) for v in value]
    return value


def _p95(validation_result) -> float | None:
    stats = validation_result.test_residual_stats if validation_result else None
    return _round_or_none(stats.p95 if stats else None)


def _radial_edge(calibration_result) -> float | None:
    profile = calibration_result.radial_profile
    if not profile or not profile.bins:
        return None
    edge_bins = profile.bins[-2:] if len(profile.bins) >= 2 else profile.bins
    values = [b.mean_error for b in edge_bins if b.mean_error is not None]
    if not values:
        return None
    return _round_or_none(sum(values) / len(values))


def _pipeline_artifacts(synthetic_dataset, camera_config, pattern_config) -> dict[str, Any]:
    dataset = copy.deepcopy(synthetic_dataset)
    analyze_dataset_quality(dataset, camera_config)
    image_size = infer_image_size(dataset, camera_config)

    compute_frame_quality_scores(dataset, pattern_config, image_size, use_reprojection=False)
    results = run_all_models(dataset, camera_config)
    calibration_results = {r.model_name: r for r in results}
    compute_frame_quality_scores(dataset, pattern_config, image_size, use_reprojection=True)

    coverage_pct = coverage_percentage(dataset.coverage_grid) if dataset.coverage_grid else None
    _, duplicate_groups = evaluate_dataset_image_quality(dataset)
    duplicate_ratio = (
        sum(len(g.image_ids) for g in duplicate_groups) / max(1, len(dataset.enabled_frames))
        if duplicate_groups else 0.0
    )
    dataset.quality_score = compute_dataset_quality_score(dataset, coverage_pct, duplicate_ratio)

    validation_results = validate_all_models(dataset, camera_config, pattern_config, test_ratio=0.25)
    scores = compute_model_scores(calibration_results, validation_results)
    chosen = next((s.model_name for s in scores if s.is_recommended), results[0].model_name)
    final_result = compute_final_result(
        chosen,
        calibration_results,
        validation_results,
        dataset_coverage_pct=coverage_pct,
        scores=scores,
        coverage_grid=dataset.coverage_grid,
        dataset_diversity=dataset.diversity,
    )

    return {
        "dataset": dataset,
        "calibration_results": calibration_results,
        "validation_results": validation_results,
        "scores": scores,
        "chosen": chosen,
        "final_result": final_result,
        "coverage_pct": coverage_pct,
    }


def _snapshot_from_artifacts(artifacts: dict[str, Any]) -> dict[str, Any]:
    dataset = artifacts["dataset"]
    calibration_results = artifacts["calibration_results"]
    validation_results = artifacts["validation_results"]
    scores = artifacts["scores"]
    final_result = artifacts["final_result"]
    coverage_pct = artifacts["coverage_pct"]

    by_model: dict[str, dict[str, Any]] = {}
    scores_by_model = {s.model_name: s for s in scores}
    for model_name, cal in calibration_results.items():
        val = validation_results.get(model_name)
        score = scores_by_model.get(model_name)
        obs = cal.observability
        undist = cal.undistortion_quality
        by_model[model_name.value] = {
            "success": cal.success,
            "train_rms": _round_or_none(cal.rms_error),
            "test_rms": _round_or_none(val.test_rms if val else None),
            "test_p95": _p95(val),
            "edge_rms": _round_or_none(val.edge_rms if val else None),
            "straightness": _round_or_none(val.straightness_residual if val else None),
            "radial_edge": _radial_edge(cal),
            "aic": _round_or_none(score.aic if score else None),
            "bic": _round_or_none(score.bic if score else None),
            "model_score": _round_or_none(score.score if score else None),
            "score_components": {
                k: _round_or_none(v) for k, v in sorted((score.components if score else {}).items())
            },
            "selection_confidence": _round_or_none(score.selection_confidence if score else None),
            "selection_confidence_level": score.selection_confidence_level if score else None,
            "selection_reasons": list(score.selection_reasons if score else []),
            "observability_score": _round_or_none(obs.observability_score if obs else None),
            "observability_grade": obs.observability_grade if obs else None,
            "observability_rank": obs.rank if obs else None,
            "observability_cols": obs.jacobian_cols if obs else None,
            "max_abs_correlation": _round_or_none(obs.max_abs_correlation if obs else None),
            "undistortion_score": _round_or_none(undist.quality_score if undist else None),
            "undistortion_grade": undist.quality_grade.value if undist else None,
            "valid_pixel_ratio": _round_or_none(undist.valid_pixel_ratio if undist else None),
            "roi_loss_ratio": _round_or_none(undist.roi_loss_ratio if undist else None),
        }

    confidence = final_result.confidence
    return {
        "baseline_version": 1,
        "dataset": {
            "num_total": dataset.num_total,
            "num_detected": dataset.num_detected,
            "coverage_pct": _round_or_none(coverage_pct),
            "diversity_overall": _round_or_none(dataset.diversity.overall if dataset.diversity else None),
            "quality_score": _round_or_none(dataset.quality_score.overall if dataset.quality_score else None),
        },
        "chosen_model": final_result.chosen_model.value,
        "overall_grade": final_result.overall_grade.value,
        "final_confidence": {
            "score": _round_or_none(confidence.score if confidence else None),
            "level": confidence.level if confidence else None,
        },
        "models": dict(sorted(by_model.items())),
    }


def _export_snapshot_from_artifacts(
    artifacts: dict[str, Any],
    camera_config,
    pattern_config,
) -> dict[str, Any]:
    payload = build_export_dict(
        camera_config,
        pattern_config,
        artifacts["dataset"],
        artifacts["calibration_results"],
        artifacts["validation_results"],
        artifacts["chosen"],
        final_result=artifacts["final_result"],
        model_scores=artifacts["scores"],
    )
    safe_payload = json_safe(payload, ndarray_wrapper=False)
    safe_payload["generated_at"] = "<normalized>"
    safe_payload = _canonical_json(safe_payload)

    models = safe_payload["models"]
    holdout = safe_payload["cross_validation"]["holdout"]
    model_scores = safe_payload.get("model_scores", [])
    final_result = safe_payload.get("final_result") or {}
    final_summary = safe_payload.get("final_calibration_summary") or {}
    bootstrap = safe_payload.get("bootstrap_stability") or {}

    return {
        "export_format_version": safe_payload["export_format_version"],
        "top_level_keys": sorted(safe_payload.keys()),
        "camera": safe_payload["camera"],
        "pattern": safe_payload["pattern"],
        "dataset": {
            "keys": sorted(safe_payload["dataset"].keys()),
            "num_total": safe_payload["dataset"]["num_total"],
            "num_detected": safe_payload["dataset"]["num_detected"],
            "num_used": safe_payload["dataset"]["num_used"],
        },
        "chosen_model": safe_payload["chosen_model"],
        "models": {
            model: {
                "keys": sorted(entry.keys()),
                "success": entry["success"],
                "has_camera_matrix": "camera_matrix" in entry,
                "has_residual_stats": "residual_stats" in entry,
                "has_observability": "observability" in entry,
                "has_undistortion_quality": "undistortion_quality" in entry,
                "distortion_coefficient_count": entry.get("distortion_coefficient_count"),
            }
            for model, entry in sorted(models.items())
        },
        "cross_validation": {
            "keys": sorted(safe_payload["cross_validation"].keys()),
            "holdout_models": sorted(holdout.keys()),
            "holdout": {
                model: {
                    "keys": sorted(entry.keys()),
                    "success": entry["success"],
                    "num_train_frames": len(entry["train_frame_ids"]),
                    "num_test_frames": len(entry["test_frame_ids"]),
                    "has_train_residual_stats": entry["train_residual_stats"] is not None,
                    "has_test_residual_stats": entry["test_residual_stats"] is not None,
                }
                for model, entry in sorted(holdout.items())
            },
        },
        "bootstrap_stability": {
            model: {
                "keys": sorted(entry.keys()),
                "available": entry["available"],
                "method": entry["method"],
                "has_distortion_stats": bool(entry["distortion_stats"]),
            }
            for model, entry in sorted(bootstrap.items())
        },
        "model_scores": [
            {
                "keys": sorted(entry.keys()),
                "model": entry["model"],
                "is_recommended": entry["is_recommended"],
                "component_keys": sorted((entry.get("components") or {}).keys()),
                "has_aic": entry.get("aic") is not None,
                "has_bic": entry.get("bic") is not None,
                "has_selection_reasons": bool(entry.get("selection_reasons")),
            }
            for entry in model_scores
        ],
        "final_result": {
            "keys": sorted(final_result.keys()),
            "chosen_model": final_result.get("chosen_model"),
            "overall_grade": final_result.get("overall_grade"),
            "has_confidence": final_result.get("confidence") is not None,
            "has_diagnosis": final_result.get("diagnosis") is not None,
        },
        "final_calibration_summary": {
            "keys": sorted(final_summary.keys()),
            "chosen_model": final_summary.get("chosen_model"),
            "overall_grade": final_summary.get("overall_grade"),
            "has_confidence": final_summary.get("confidence") is not None,
            "has_diagnosis": final_summary.get("diagnosis") is not None,
            "has_observability": final_summary.get("observability") is not None,
            "has_undistortion_quality": final_summary.get("undistortion_quality") is not None,
        },
    }


def _snapshot(synthetic_dataset, camera_config, pattern_config) -> dict[str, Any]:
    return _snapshot_from_artifacts(_pipeline_artifacts(synthetic_dataset, camera_config, pattern_config))


def _assert_close(path: str, current: float | None, expected: float | None, rel: float, abs_: float) -> None:
    if current is None and expected is None:
        return
    assert current is not None, f"{path}: current metric is missing"
    assert expected is not None, f"{path}: baseline metric is missing"
    assert math.isclose(current, expected, rel_tol=rel, abs_tol=abs_), (
        f"{path}: current={current}, baseline={expected}, tolerance=rel {rel}, abs {abs_}"
    )


def _compare_snapshot(current: dict[str, Any], baseline: dict[str, Any]) -> None:
    assert current["baseline_version"] == baseline["baseline_version"]
    assert current["dataset"]["num_total"] == baseline["dataset"]["num_total"]
    assert current["dataset"]["num_detected"] == baseline["dataset"]["num_detected"]
    _assert_close("dataset.coverage_pct", current["dataset"]["coverage_pct"], baseline["dataset"]["coverage_pct"], 0.0, 0.1)
    _assert_close("dataset.quality_score", current["dataset"]["quality_score"], baseline["dataset"]["quality_score"], 0.02, 1.0)

    assert current["chosen_model"] == baseline["chosen_model"]
    assert current["overall_grade"] == baseline["overall_grade"]
    _assert_close(
        "final_confidence.score",
        current["final_confidence"]["score"],
        baseline["final_confidence"]["score"],
        0.06,
        3.0,
    )
    assert current["final_confidence"]["level"] == baseline["final_confidence"]["level"]

    assert set(current["models"]) == set(baseline["models"])
    for model_name, current_model in current["models"].items():
        baseline_model = baseline["models"][model_name]
        assert current_model["success"] == baseline_model["success"], model_name
        assert current_model["selection_confidence_level"] == baseline_model["selection_confidence_level"]
        assert current_model["observability_grade"] == baseline_model["observability_grade"]
        assert current_model["undistortion_grade"] == baseline_model["undistortion_grade"]
        if model_name == current["chosen_model"]:
            assert current_model["selection_reasons"], f"{model_name}: selection reasons disappeared"

        for metric, (rel, abs_) in METRIC_TOLERANCES.items():
            _assert_close(
                f"models.{model_name}.{metric}",
                current_model[metric],
                baseline_model[metric],
                rel,
                abs_,
            )

        assert set(current_model["score_components"]) == set(baseline_model["score_components"])
        for component, current_value in current_model["score_components"].items():
            _assert_close(
                f"models.{model_name}.score_components.{component}",
                current_value,
                baseline_model["score_components"][component],
                0.12,
                0.03,
            )


def test_synthetic_pipeline_matches_baseline(synthetic_dataset, camera_config, pattern_config):
    current = _snapshot(synthetic_dataset, camera_config, pattern_config)

    if os.getenv("UPDATE_BASELINE") == "1":
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip(f"updated baseline snapshot: {BASELINE_PATH}")

    assert BASELINE_PATH.exists(), (
        "baseline snapshot is missing. Run with UPDATE_BASELINE=1 to create it."
    )
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    _compare_snapshot(current, baseline)


def test_synthetic_calibration_json_export_matches_baseline(synthetic_dataset, camera_config, pattern_config):
    artifacts = _pipeline_artifacts(synthetic_dataset, camera_config, pattern_config)
    current = _export_snapshot_from_artifacts(artifacts, camera_config, pattern_config)

    if os.getenv("UPDATE_BASELINE") == "1":
        EXPORT_BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        EXPORT_BASELINE_PATH.write_text(
            json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        pytest.skip(f"updated calibration JSON export baseline: {EXPORT_BASELINE_PATH}")

    assert EXPORT_BASELINE_PATH.exists(), (
        "calibration JSON export baseline is missing. Run with UPDATE_BASELINE=1 to create it."
    )
    baseline = json.loads(EXPORT_BASELINE_PATH.read_text(encoding="utf-8"))
    assert current == baseline

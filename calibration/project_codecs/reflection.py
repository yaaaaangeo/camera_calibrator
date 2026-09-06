"""
camera_calibrator.calibration.project_codecs.reflection
==============================================================

Phase D-2 안정화 - Reflection 평가 결과 관련 dict -> dataclass 디코더.
`calibration/project_io.py`에서 로직 변경 없이 그대로 옮겨왔다. Ghost/
Windshield Geometry 재구성 함수와 완전히 독립적이다(서로 import하지 않음).
"""

from __future__ import annotations

from calibration.windshield.reflection.types import (
    ReflectionDatasetResult,
    ReflectionEvaluationResult,
    ReflectionRegionMetrics,
    ReflectionSpatialCell,
)

def _reflection_region_metrics_from_dict(d) -> ReflectionRegionMetrics:
    d = d or {}
    return ReflectionRegionMetrics(
        mean_strength=d.get("mean_strength", 0.0),
        p95_strength=d.get("p95_strength", 0.0),
        coverage=d.get("coverage", 0.0),
    )


def _reflection_spatial_cell_from_dict(d) -> ReflectionSpatialCell:
    return ReflectionSpatialCell(
        row=d.get("row", 0),
        col=d.get("col", 0),
        mean_strength=d.get("mean_strength", 0.0),
        p95_strength=d.get("p95_strength", 0.0),
        coverage=d.get("coverage", 0.0),
    )


def _reflection_evaluation_result_from_dict(d) -> ReflectionEvaluationResult:
    mode = d.get("mode", "reference")
    mean_strength = d.get("mean_strength", 0.0)
    median_strength = d.get("median_strength", 0.0)
    p95_strength = d.get("p95_strength", 0.0)
    p99_strength = d.get("p99_strength", 0.0)
    max_strength = d.get("max_strength", 0.0)
    coverage = d.get("coverage", 0.0)
    return ReflectionEvaluationResult(
        mode=mode,
        metric_version=d.get("metric_version", 1),
        pair_id=d.get("pair_id", ""),
        reflection_mean=d.get("reflection_mean", mean_strength if mode == "reference" else None),
        reflection_median=d.get("reflection_median", median_strength if mode == "reference" else None),
        reflection_p95=d.get("reflection_p95", p95_strength if mode == "reference" else None),
        reflection_p99=d.get("reflection_p99", p99_strength if mode == "reference" else None),
        reflection_max=d.get("reflection_max", max_strength if mode == "reference" else None),
        reflection_coverage=d.get("reflection_coverage", coverage if mode == "reference" else None),
        reflection_likelihood=d.get("reflection_likelihood", mean_strength if mode == "no_reference" else None),
        no_reference_is_likelihood=d.get("no_reference_is_likelihood", mode == "no_reference"),
        mean_strength=mean_strength,
        median_strength=median_strength,
        p95_strength=p95_strength,
        p99_strength=p99_strength,
        max_strength=max_strength,
        positive_mean_strength=d.get("positive_mean_strength", 0.0),
        positive_p95_strength=d.get("positive_p95_strength", 0.0),
        coverage=coverage,
        coverage_threshold=d.get("coverage_threshold", 0.08),
        saturation_threshold=d.get("saturation_threshold", 250.0),
        glare_luminance_threshold=d.get("glare_luminance_threshold", 220.0),
        glare_contrast_threshold=d.get("glare_contrast_threshold", 12.0),
        severity_score=None if mode == "no_reference" else d.get("severity_score"),
        saturation_coverage=d.get("saturation_coverage", 0.0),
        glare_coverage=d.get("glare_coverage"),
        glare_strength=d.get("glare_strength"),
        contrast_retention=d.get("contrast_retention"),
        edge_retention=d.get("edge_retention"),
        bottom_roi_mean_strength=d.get("bottom_roi_mean_strength"),
        bottom_roi_coverage=d.get("bottom_roi_coverage"),
        regional_metrics={
            k: _reflection_region_metrics_from_dict(v)
            for k, v in d.get("regional_metrics", {}).items()
        },
        spatial_map=[_reflection_spatial_cell_from_dict(c) for c in d.get("spatial_map", [])],
        alignment_score=d.get("alignment_score"),
        alignment_error_px=d.get("alignment_error_px"),
        alignment_status=d.get("alignment_status", "not_run"),
        alignment_method=d.get("alignment_method", "none"),
        alignment_translation_x_px=d.get("alignment_translation_x_px"),
        alignment_translation_y_px=d.get("alignment_translation_y_px"),
        alignment_rotation_deg=d.get("alignment_rotation_deg"),
        alignment_scale=d.get("alignment_scale"),
        alignment_shear_deg=d.get("alignment_shear_deg"),
        photometric_normalized=d.get("photometric_normalized", False),
        photometric_gain=d.get("photometric_gain"),
        photometric_bias=d.get("photometric_bias"),
        heatmap_rows=d.get("heatmap_rows", 0),
        heatmap_cols=d.get("heatmap_cols", 0),
        downsampled_reflection_map=d.get("downsampled_reflection_map", []),
        warning_message=d.get("warning_message"),
        error_message=d.get("error_message"),
        success=d.get("success", True),
    )


def _reflection_dataset_result_from_dict(d) -> ReflectionDatasetResult:
    mode = d.get("mode", "reference")
    return ReflectionDatasetResult(
        mode=mode,
        metric_version=d.get("metric_version", 1),
        pair_results=[_reflection_evaluation_result_from_dict(r) for r in d.get("pair_results", [])],
        reference_mean_strength=d.get("reference_mean_strength", d.get("mean_strength") if mode == "reference" else None),
        reference_p95_strength=d.get("reference_p95_strength", d.get("p95_strength") if mode == "reference" else None),
        reference_coverage=d.get("reference_coverage", d.get("coverage") if mode == "reference" else None),
        mean_reflection_likelihood=d.get("mean_reflection_likelihood", d.get("mean_strength") if mode == "no_reference" else None),
        p95_reflection_likelihood=d.get("p95_reflection_likelihood", d.get("p95_strength") if mode == "no_reference" else None),
        mean_strength=d.get("mean_strength", 0.0),
        median_strength=d.get("median_strength", 0.0),
        p95_strength=d.get("p95_strength", 0.0),
        worst_pair_id=d.get("worst_pair_id"),
        coverage=d.get("coverage", 0.0),
        severity_score=None if mode == "no_reference" else d.get("severity_score"),
        by_day_night=d.get("by_day_night", {}),
        num_valid_pairs=d.get("num_valid_pairs", 0),
        num_invalid_pairs=d.get("num_invalid_pairs", 0),
        success=d.get("success", True),
        warning_message=d.get("warning_message"),
        error_message=d.get("error_message"),
    )



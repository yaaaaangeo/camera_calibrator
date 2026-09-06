"""
camera_calibrator.calibration.project_codecs.ghost
==============================================================

Phase D-2 안정화 - Ghost/Double Image 평가 결과 관련 dict -> dataclass
디코더. `calibration/project_io.py`에서 로직 변경 없이 그대로 옮겨왔다.
Reflection과 완전히 독립적이다(서로 import하지 않음).
"""

from __future__ import annotations

import numpy as np

from calibration.project_codecs.common import _arr
from calibration.windshield.ghost.types import (
    GhostDatasetResult,
    GhostEvaluationResult,
    GhostField,
    GhostFieldDiagnostics,
    GhostPointDetection,
    GhostRegionMetrics,
    GhostSpatialCell,
)

# ---------------------------------------------------------------------------
# Ghost / Double Image (STEP 8) - Reflection과 완전히 별도의 재구성 함수다.
# `ghost_results`/`ghost_models`는 windshield_results/reflection_results와
# 절대 섞이지 않는다.
# ---------------------------------------------------------------------------

def _ghost_point_detection_from_dict(d) -> GhostPointDetection:
    return GhostPointDetection(
        main_x=d.get("main_x", 0.0),
        main_y=d.get("main_y", 0.0),
        ghost_x=d.get("ghost_x"),
        ghost_y=d.get("ghost_y"),
        offset_x_px=d.get("offset_x_px"),
        offset_y_px=d.get("offset_y_px"),
        distance_px=d.get("distance_px"),
        angular_separation_deg=d.get("angular_separation_deg"),
        strength_ratio=d.get("strength_ratio"),
        detected=d.get("detected", False),
        pair_residual_px=d.get("pair_residual_px"),
        pair_energy_ratio=d.get("pair_energy_ratio"),
        pair_energy_residual=d.get("pair_energy_residual"),
    )


def _ghost_region_metrics_from_dict(d) -> GhostRegionMetrics:
    d = d or {}
    return GhostRegionMetrics(
        mean_distance_px=d.get("mean_distance_px", 0.0),
        mean_strength_ratio=d.get("mean_strength_ratio", 0.0),
        detection_rate=d.get("detection_rate", 0.0),
    )


def _ghost_spatial_cell_from_dict(d) -> GhostSpatialCell:
    return GhostSpatialCell(
        row=d.get("row", 0),
        col=d.get("col", 0),
        mean_offset_x_px=d.get("mean_offset_x_px"),
        mean_offset_y_px=d.get("mean_offset_y_px"),
        mean_distance_px=d.get("mean_distance_px"),
        mean_strength_ratio=d.get("mean_strength_ratio"),
        sample_count=d.get("sample_count", 0),
        mad_offset_x_px=d.get("mad_offset_x_px"),
        mad_offset_y_px=d.get("mad_offset_y_px"),
        mad_strength=d.get("mad_strength"),
        outlier_rejected_count=d.get("outlier_rejected_count", 0),
        is_filled=d.get("is_filled", False),
    )


def _ghost_evaluation_result_from_dict(d) -> GhostEvaluationResult:
    return GhostEvaluationResult(
        success=d.get("success", True),
        mode=d.get("mode", "point_source"),
        metric_version=d.get("metric_version", 1),
        pair_id=d.get("pair_id", ""),
        detection_count=d.get("detection_count", 0),
        candidate_count=d.get("candidate_count", 0),
        detection_rate=d.get("detection_rate"),
        mean_offset_x_px=d.get("mean_offset_x_px"),
        mean_offset_y_px=d.get("mean_offset_y_px"),
        median_distance_px=d.get("median_distance_px"),
        p95_distance_px=d.get("p95_distance_px"),
        mean_angular_separation_deg=d.get("mean_angular_separation_deg"),
        p95_angular_separation_deg=d.get("p95_angular_separation_deg"),
        mean_strength_ratio=d.get("mean_strength_ratio"),
        p95_strength_ratio=d.get("p95_strength_ratio"),
        edge_offset_median_px=d.get("edge_offset_median_px"),
        edge_offset_p95_px=d.get("edge_offset_p95_px"),
        ghost_likelihood=d.get("ghost_likelihood"),
        is_likelihood=d.get("is_likelihood", False),
        likelihood_row_detection_rate=d.get("likelihood_row_detection_rate"),
        likelihood_column_detection_rate=d.get("likelihood_column_detection_rate"),
        regional_metrics={
            k: _ghost_region_metrics_from_dict(v)
            for k, v in d.get("regional_metrics", {}).items()
        },
        spatial_map=[_ghost_spatial_cell_from_dict(c) for c in d.get("spatial_map", [])],
        detections=[_ghost_point_detection_from_dict(p) for p in d.get("detections", [])],
        warning_message=d.get("warning_message"),
        error_message=d.get("error_message"),
    )


def _ghost_dataset_result_from_dict(d) -> GhostDatasetResult:
    return GhostDatasetResult(
        mode=d.get("mode", "point_source"),
        metric_version=d.get("metric_version", 1),
        per_frame=[_ghost_evaluation_result_from_dict(r) for r in d.get("per_frame", [])],
        mean_distance_px=d.get("mean_distance_px"),
        p95_distance_px=d.get("p95_distance_px"),
        mean_strength=d.get("mean_strength"),
        p95_strength=d.get("p95_strength"),
        mean_ghost_likelihood=d.get("mean_ghost_likelihood"),
        median_ghost_likelihood=d.get("median_ghost_likelihood"),
        p95_ghost_likelihood=d.get("p95_ghost_likelihood"),
        worst_frame_id=d.get("worst_frame_id"),
        image_width=d.get("image_width"),
        image_height=d.get("image_height"),
        num_input_frames=d.get("num_input_frames", 0),
        num_valid_frames=d.get("num_valid_frames", 0),
        num_failed_frames=d.get("num_failed_frames", 0),
        success=d.get("success", True),
        warning_message=d.get("warning_message"),
        error_message=d.get("error_message"),
    )


def _ghost_field_diagnostics_from_dict(d) -> Optional[GhostFieldDiagnostics]:
    if not d:
        return None
    return GhostFieldDiagnostics(
        num_frames=d.get("num_frames", 0),
        num_detections=d.get("num_detections", 0),
        grid_rows=d.get("grid_rows", 0),
        grid_cols=d.get("grid_cols", 0),
        samples_per_cell=list(d.get("samples_per_cell", [])),
        global_median_dx=d.get("global_median_dx"),
        global_median_dy=d.get("global_median_dy"),
        global_median_strength=d.get("global_median_strength"),
        fit_stability=d.get("fit_stability"),
    )


def _ghost_field_from_dict(d) -> GhostField:
    return GhostField(
        offset_x=_arr(d.get("offset_x"), np.float32),
        offset_y=_arr(d.get("offset_y"), np.float32),
        strength=_arr(d.get("strength"), np.float32),
        image_width=d.get("image_width", 0.0),
        image_height=d.get("image_height", 0.0),
        model_version=d.get("model_version", 1),
        diagnostics=_ghost_field_diagnostics_from_dict(d.get("diagnostics")),
    )


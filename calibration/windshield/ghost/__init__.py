"""
calibration.windshield.ghost
==============================================================

STEP 8 - Ghost / Double Image. Reflection(calibration.windshield.reflection,
calibration.windshield.reflection_suppression)과 완전히 독립된 패키지다 -
Ghost는 "같은 exterior scene의 shifted/warped duplicate"이고, Reflection은
"다른 scene(내부)의 overlay"이기 때문에 코드/점수/모델을 절대 공유하지
않는다.
"""

from __future__ import annotations

from calibration.windshield.ghost.config import GHOST_METRIC_VERSION, GHOST_MODEL_VERSION
from calibration.windshield.ghost.edge_detector import extract_edge_profiles
from calibration.windshield.ghost.evaluator import (
    evaluate_ghost_dataset,
    evaluate_ghost_dataset_from_paths,
    evaluate_ghost_edge_target,
    evaluate_ghost_general_likelihood,
    evaluate_ghost_image,
    evaluate_ghost_point_source,
)
from calibration.windshield.ghost.point_detector import estimate_dominant_energy_ratio, estimate_dominant_ghost_vector
from calibration.windshield.ghost.spatial_model import build_robust_spatial_map_from_detections, fill_empty_spatial_cells
from calibration.windshield.ghost.suppression import (
    build_dense_fields,
    build_suppression_evaluation,
    compute_reconstruction_metrics,
    fit_ghost_field_constant,
    fit_ghost_field_from_dataset,
    fit_ghost_field_from_spatial_map,
    load_ghost_model,
    save_ghost_model,
    suppress_ghost,
)
from calibration.windshield.ghost.types import (
    GhostDatasetResult,
    GhostEvaluationConfig,
    GhostEvaluationResult,
    GhostField,
    GhostFieldDiagnostics,
    GhostPointDetection,
    GhostReconstructionMetrics,
    GhostRegionMetrics,
    GhostSpatialCell,
    GhostSuppressionEvaluation,
    GhostSuppressionResult,
)
from calibration.windshield.ghost.visualization import (
    render_ghost_point_overlay,
    render_strength_heatmap_image,
    render_vector_field_image,
)

__all__ = [
    "GHOST_METRIC_VERSION",
    "GHOST_MODEL_VERSION",
    "evaluate_ghost_point_source",
    "evaluate_ghost_edge_target",
    "evaluate_ghost_general_likelihood",
    "evaluate_ghost_image",
    "evaluate_ghost_dataset",
    "evaluate_ghost_dataset_from_paths",
    "extract_edge_profiles",
    "estimate_dominant_ghost_vector",
    "estimate_dominant_energy_ratio",
    "build_robust_spatial_map_from_detections",
    "fill_empty_spatial_cells",
    "suppress_ghost",
    "build_dense_fields",
    "build_suppression_evaluation",
    "fit_ghost_field_constant",
    "fit_ghost_field_from_spatial_map",
    "fit_ghost_field_from_dataset",
    "compute_reconstruction_metrics",
    "save_ghost_model",
    "load_ghost_model",
    "GhostEvaluationConfig",
    "GhostPointDetection",
    "GhostRegionMetrics",
    "GhostSpatialCell",
    "GhostEvaluationResult",
    "GhostDatasetResult",
    "GhostField",
    "GhostFieldDiagnostics",
    "GhostReconstructionMetrics",
    "GhostSuppressionResult",
    "GhostSuppressionEvaluation",
    "render_ghost_point_overlay",
    "render_vector_field_image",
    "render_strength_heatmap_image",
]

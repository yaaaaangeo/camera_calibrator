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
from calibration.windshield.ghost.evaluator import (
    evaluate_ghost_dataset,
    evaluate_ghost_edge_target,
    evaluate_ghost_general_likelihood,
    evaluate_ghost_point_source,
)
from calibration.windshield.ghost.suppression import (
    build_dense_fields,
    fit_ghost_field_constant,
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
    GhostPointDetection,
    GhostRegionMetrics,
    GhostSpatialCell,
    GhostSuppressionEvaluation,
    GhostSuppressionResult,
)

__all__ = [
    "GHOST_METRIC_VERSION",
    "GHOST_MODEL_VERSION",
    "evaluate_ghost_point_source",
    "evaluate_ghost_edge_target",
    "evaluate_ghost_general_likelihood",
    "evaluate_ghost_dataset",
    "suppress_ghost",
    "build_dense_fields",
    "fit_ghost_field_constant",
    "fit_ghost_field_from_spatial_map",
    "save_ghost_model",
    "load_ghost_model",
    "GhostEvaluationConfig",
    "GhostPointDetection",
    "GhostRegionMetrics",
    "GhostSpatialCell",
    "GhostEvaluationResult",
    "GhostDatasetResult",
    "GhostField",
    "GhostSuppressionResult",
    "GhostSuppressionEvaluation",
]

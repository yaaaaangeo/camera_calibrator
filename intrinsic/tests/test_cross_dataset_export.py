from __future__ import annotations

import numpy as np

from calibration.project_io import project_from_dict, project_to_dict
from calibration.recommender import compute_final_result
from calibration.types import (
    CalibrationProject,
    CalibrationResult,
    CameraModelType,
    CrossDatasetValidationResult,
    Dataset,
    PatternConfig,
    PatternType,
)
from export.json_export import build_export_dict
from export.report import generate_html_report


def _cal() -> CalibrationResult:
    return CalibrationResult(
        model_name=CameraModelType.PINHOLE,
        camera_matrix=np.eye(3),
        distortion=np.zeros((5, 1)),
        rms_error=0.2,
        success=True,
    )


def _cross() -> CrossDatasetValidationResult:
    return CrossDatasetValidationResult(
        source_dataset_id="Dataset A",
        target_dataset_id="Dataset B",
        model_name=CameraModelType.PINHOLE,
        train_rms=0.2,
        test_rms=0.35,
        test_p95=0.5,
        edge_rms=0.4,
        generalization_gap=0.15,
        num_test_frames=12,
        success=True,
    )


def test_cross_dataset_results_roundtrip_project_json_and_html(camera_config):
    pattern = PatternConfig(PatternType.CHESSBOARD, squares_x=5, squares_y=4, square_size=0.04)
    cal = _cal()
    calibration_results = {CameraModelType.PINHOLE: cal}
    final = compute_final_result(CameraModelType.PINHOLE, calibration_results, {})
    cross = [_cross()]
    project = CalibrationProject(
        project_name="cross",
        camera_config=camera_config,
        pattern_config=pattern,
        calibration_results=calibration_results,
        final_result=final,
        cross_dataset_results=cross,
    )

    restored = project_from_dict(project_to_dict(project))
    payload = build_export_dict(
        camera_config,
        pattern,
        Dataset(),
        calibration_results,
        {},
        CameraModelType.PINHOLE,
        final_result=final,
        cross_dataset_results=cross,
    )
    html = generate_html_report(
        "cross",
        camera_config,
        pattern,
        Dataset(),
        calibration_results,
        {},
        final,
        cross_dataset_results=cross,
    )

    assert restored.cross_dataset_results[0].target_dataset_id == "Dataset B"
    assert restored.final_result.confidence is not None
    assert payload["cross_dataset_validation"][0].target_dataset_id == "Dataset B"
    assert payload["final_result"]["confidence"].score >= 0.0
    assert "Cross-Dataset Generalization" in html
    assert "Final Calibration Confidence" in html
    assert "Dataset B" in html
    assert "Target P95" in html

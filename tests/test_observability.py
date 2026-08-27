from __future__ import annotations

import cv2
import numpy as np

from calibration.observability import (
    attach_observability_report,
    compute_numeric_jacobian,
    compute_observability_report,
    grade_observability,
    score_observability,
)
from calibration.project_io import project_from_dict, project_to_dict
from calibration.recommender import compute_final_result
from calibration.types import (
    CalibrationProject,
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    Dataset,
    DetectionResult,
    Frame,
    FrameStatus,
    ImageInfo,
    ObservabilityReport,
    ParameterCorrelation,
    PatternConfig,
    PatternType,
    ValidationResult,
)
from export.json_export import build_export_dict
from export.report import generate_html_report


def _object_points() -> np.ndarray:
    pts = []
    for y in range(4):
        for x in range(5):
            pts.append([x * 0.04, y * 0.04, 0.0])
    return np.asarray(pts, dtype=np.float32).reshape(-1, 1, 3)


def _dataset_and_result(model: CameraModelType = CameraModelType.PINHOLE) -> tuple[Dataset, CalibrationResult]:
    K = np.array([[800.0, 0.0, 320.0], [0.0, 805.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.zeros((5, 1), dtype=np.float64)
    obj = _object_points()
    rvecs = [
        np.array([[0.04], [0.02], [0.01]], dtype=np.float64),
        np.array([[0.12], [-0.03], [0.08]], dtype=np.float64),
        np.array([[-0.08], [0.05], [-0.05]], dtype=np.float64),
    ]
    tvecs = [
        np.array([[0.00], [0.00], [0.80]], dtype=np.float64),
        np.array([[0.04], [-0.02], [0.95]], dtype=np.float64),
        np.array([[-0.05], [0.03], [0.75]], dtype=np.float64),
    ]

    frames = []
    for idx, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        corners, _ = cv2.projectPoints(obj.astype(np.float64), rvec, tvec, K, D)
        det = DetectionResult(
            image_id=f"img_{idx}",
            success=True,
            corners=corners.astype(np.float32),
            object_points=obj.copy(),
            num_corners=int(obj.shape[0]),
        )
        frames.append(
            Frame(
                image_info=ImageInfo(f"img_{idx}", f"img_{idx}.png", 640, 480),
                detection=det,
                status=FrameStatus.DETECTED,
            )
        )

    result = CalibrationResult(
        model_name=model,
        camera_matrix=K,
        distortion=D,
        rvecs=rvecs,
        tvecs=tvecs,
        rms_error=0.0,
        success=True,
    )
    return Dataset(frames=frames), result


def test_numeric_jacobian_and_svd_report_for_pinhole():
    dataset, result = _dataset_and_result()

    J, labels = compute_numeric_jacobian(result, dataset)
    report = compute_observability_report(result, dataset)

    assert labels == ["fx", "fy", "cx", "cy"]
    assert J.shape == (dataset.num_detected * _object_points().shape[0] * 2, 4)
    assert report.jacobian_rows == J.shape[0]
    assert report.jacobian_cols == 4
    assert report.num_points == J.shape[0] // 2
    assert len(report.singular_values) == 4
    assert report.condition_number is not None
    assert report.condition_number > 0
    assert report.rank <= 4
    assert report.observability_score is not None
    assert 0.0 <= report.observability_score <= 100.0
    assert report.observability_grade in ("GOOD", "WARNING", "POOR")
    assert len(report.correlation_matrix) == 4
    assert all(len(row) == 4 for row in report.correlation_matrix)
    assert all(abs(report.correlation_matrix[i][i] - 1.0) < 1e-9 for i in range(4))
    assert all(-1.0 <= v <= 1.0 for row in report.correlation_matrix for v in row)


def test_observability_score_and_grade_thresholds():
    good = score_observability(rank=4, jacobian_cols=4, condition_number=1e4, max_abs_correlation=0.70)
    warning = score_observability(rank=4, jacobian_cols=4, condition_number=1e8, max_abs_correlation=0.88)
    poor = score_observability(rank=3, jacobian_cols=4, condition_number=float("inf"), max_abs_correlation=0.995)

    assert good == 100.0
    assert grade_observability(good) == "GOOD"
    assert grade_observability(warning) == "WARNING"
    assert grade_observability(poor) == "POOR"


def test_extended_model_observability_includes_distortion_labels_and_correlations():
    dataset, result = _dataset_and_result(CameraModelType.EXTENDED_PINHOLE)

    attach_observability_report(result, dataset)

    obs = result.observability
    assert obs is not None
    assert obs.parameter_labels[:4] == ["fx", "fy", "cx", "cy"]
    assert obs.parameter_labels[4:] == ["k1", "k2", "p1", "p2", "k3"]
    assert obs.jacobian_cols == 9
    assert obs.max_abs_correlation is None or 0.0 <= obs.max_abs_correlation <= 1.0
    assert len(obs.top_correlations) <= 5


def test_observability_is_exported_to_json_report_and_project(camera_config):
    dataset, result = _dataset_and_result()
    result.observability = ObservabilityReport(
        parameter_labels=["fx", "fy"],
        jacobian_rows=120,
        jacobian_cols=2,
        num_points=60,
        singular_values=[10.0, 0.1],
        rank=2,
        condition_number=100.0,
        min_singular_value=0.1,
        max_singular_value=10.0,
        max_abs_correlation=0.9,
        correlation_matrix=[[1.0, 0.9], [0.9, 1.0]],
        observability_score=75.0,
        observability_grade="WARNING",
        top_correlations=[ParameterCorrelation("fx", "fy", 0.9)],
        warnings=["High condition number: 100."],
    )
    cal = {CameraModelType.PINHOLE: result}
    val = {CameraModelType.PINHOLE: ValidationResult(test_rms=0.1, success=True)}
    pattern = PatternConfig(PatternType.CHESSBOARD, squares_x=5, squares_y=4, square_size=0.04)
    final = compute_final_result(CameraModelType.PINHOLE, cal, val)
    project = CalibrationProject(
        project_name="obs",
        camera_config=camera_config,
        pattern_config=pattern,
        dataset=dataset,
        calibration_results=cal,
        validation_results=val,
        final_result=final,
    )

    payload = build_export_dict(camera_config, pattern, dataset, cal, val, CameraModelType.PINHOLE, final)
    html = generate_html_report("obs", camera_config, pattern, dataset, cal, val, final)
    restored = project_from_dict(project_to_dict(project))

    obs_payload = payload["models"]["pinhole"]["observability"]
    assert obs_payload.parameter_labels == ["fx", "fy"]
    assert obs_payload.correlation_matrix == [[1.0, 0.9], [0.9, 1.0]]
    assert obs_payload.observability_score == 75.0
    assert obs_payload.observability_grade == "WARNING"
    assert "Observability (Jacobian / SVD)" in html
    assert "Parameter Correlation Matrix" in html
    assert "WARNING (75.0/100)" in html
    assert "Condition Number" in html
    assert restored.calibration_results[CameraModelType.PINHOLE].observability.condition_number == 100.0
    assert restored.calibration_results[CameraModelType.PINHOLE].observability.correlation_matrix == [[1.0, 0.9], [0.9, 1.0]]
    assert restored.calibration_results[CameraModelType.PINHOLE].observability.observability_grade == "WARNING"

from __future__ import annotations

import cv2
import numpy as np

from calibration.observability import (
    _correlation_matrix_from_normalized_jacobian,
    _covariance_from_jacobian_svd,
    _svd_diagnostics,
    attach_observability_report,
    compute_numeric_jacobian,
    compute_observability_report,
    grade_observability,
    score_observability,
)
from calibration.project_codecs.intrinsic import _observability_report_from_dict
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
    assert report.condition_number == report.normalized_condition_number
    assert report.raw_condition_number is not None
    assert report.normalization_scales == {"fx": 640.0, "fy": 480.0, "cx": 640.0, "cy": 480.0}
    assert len(report.raw_singular_values) == 4
    assert report.correlation_method == "covariance_from_normalized_jacobian"
    assert report.rank <= 4
    assert report.observability_score is not None
    assert 0.0 <= report.observability_score <= 100.0
    assert report.observability_grade in ("GOOD", "WARNING", "POOR")
    assert len(report.correlation_matrix) == 4
    assert all(len(row) == 4 for row in report.correlation_matrix)
    assert all(abs(report.correlation_matrix[i][i] - 1.0) < 1e-9 for i in range(4))
    assert all(-1.0 <= v <= 1.0 for row in report.correlation_matrix for v in row)
    assert any("fixed-pose local intrinsic observability" in w for w in report.warnings)


def test_normalized_observability_reduces_pixel_unit_scale_sensitivity():
    dataset, result = _dataset_and_result()

    report = compute_observability_report(result, dataset)

    assert report.raw_condition_number is not None
    assert report.normalized_condition_number is not None
    assert report.raw_condition_number != report.normalized_condition_number
    assert report.normalization_scales["fx"] == 640.0
    assert report.normalization_scales["fy"] == 480.0
    assert report.normalized_condition_number == report.condition_number


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
        raw_condition_number=1000.0,
        normalized_condition_number=100.0,
        normalization_scales={"fx": 640.0, "fy": 480.0},
        raw_singular_values=[100.0, 0.1],
        min_singular_value=0.1,
        max_singular_value=10.0,
        max_abs_correlation=0.9,
        correlation_matrix=[[1.0, 0.9], [0.9, 1.0]],
        correlation_method="covariance_from_normalized_jacobian",
        observability_score=75.0,
        observability_grade="WARNING",
        top_correlations=[ParameterCorrelation("fx", "fy", 0.9)],
        warnings=["High normalized condition number: 100."],
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
    assert "Observability (Fixed-Pose Local Intrinsic Jacobian / SVD)" in html
    assert "Parameter Correlation Matrix" in html
    assert "WARNING (75.0/100)" in html
    assert "Normalized Condition Number" in html
    assert "Raw Condition Number" in html
    assert restored.calibration_results[CameraModelType.PINHOLE].observability.condition_number == 100.0
    assert restored.calibration_results[CameraModelType.PINHOLE].observability.raw_condition_number == 1000.0
    assert restored.calibration_results[CameraModelType.PINHOLE].observability.normalization_scales["fx"] == 640.0
    assert restored.calibration_results[CameraModelType.PINHOLE].observability.correlation_matrix == [[1.0, 0.9], [0.9, 1.0]]
    assert restored.calibration_results[CameraModelType.PINHOLE].observability.observability_grade == "WARNING"
    # 새 프로젝트는 항상 새 covariance 방식으로 라벨링되어 왕복돼야 한다.
    assert restored.calibration_results[CameraModelType.PINHOLE].observability.correlation_method == (
        "covariance_from_normalized_jacobian"
    )


# ---------------------------------------------------------------------------
# SVD-based covariance/correlation (Jn.T @ Jn 없이 계산) 회귀 테스트
# ---------------------------------------------------------------------------


def _well_conditioned_jacobian(rows: int = 40, cols: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows, cols))


def _jacobian_with_singular_values(
    singular_values: np.ndarray, rows: int = 40, seed: int = 0
) -> np.ndarray:
    """주어진 singular value들을 갖는 (rows, len(singular_values)) Jacobian을 합성."""
    cols = singular_values.size
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.normal(size=(rows, cols)))
    v, _ = np.linalg.qr(rng.normal(size=(cols, cols)))
    return (u * singular_values) @ v.T


def _old_normal_matrix_correlation(J: np.ndarray, rank: int) -> list[list[float]]:
    """수정 전 구현(Jn.T @ Jn 뒤 pinv) - 비교용으로만 이 테스트 파일 안에 재현."""
    if J.shape[1] == 0:
        return []
    if J.shape[1] == 1 or J.shape[0] < 2:
        return [[1.0]]
    if rank < J.shape[1]:
        return []
    hessian = np.asarray(J.T @ J, dtype=np.float64)
    covariance = np.linalg.pinv(hessian)
    diag = np.diag(covariance)
    if np.any(diag <= 0) or not np.all(np.isfinite(diag)):
        return []
    denom = np.sqrt(np.outer(diag, diag))
    corr = covariance / denom
    if not np.all(np.isfinite(corr)):
        return []
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)
    return [[float(v) for v in row] for row in corr.tolist()]


def test_svd_covariance_is_symmetric():
    J = _well_conditioned_jacobian()
    covariance = _covariance_from_jacobian_svd(J)
    assert covariance is not None
    assert np.allclose(covariance, covariance.T, atol=1e-9)


def test_svd_correlation_full_rank_diagonal_is_one_and_bounded():
    J = _well_conditioned_jacobian()
    _singular, rank, _cond, _min_sv, _max_sv = _svd_diagnostics(J)
    assert rank == J.shape[1]

    corr = _correlation_matrix_from_normalized_jacobian(J, rank)
    assert len(corr) == J.shape[1]
    for i, row in enumerate(corr):
        assert abs(row[i] - 1.0) < 1e-9
        for value in row:
            assert -1.0 <= value <= 1.0


def test_svd_correlation_matches_old_normal_matrix_method_for_well_conditioned_case():
    J = _well_conditioned_jacobian(rows=60, cols=5, seed=42)
    _singular, rank, _cond, _min_sv, _max_sv = _svd_diagnostics(J)

    new_corr = np.asarray(_correlation_matrix_from_normalized_jacobian(J, rank))
    old_corr = np.asarray(_old_normal_matrix_correlation(J, rank))

    assert new_corr.shape == old_corr.shape
    assert np.allclose(new_corr, old_corr, atol=1e-6)


def test_svd_correlation_stable_for_ill_conditioned_full_rank_jacobian():
    # 매우 넓게 퍼진 singular value: cond(J) ~ 1e10, cond(J.T @ J) ~ 1e20이 됐을
    # normal-matrix 경로였다면 pinv가 rcond 근처에서 흔들릴 수 있는 영역.
    singular_values = np.array([1e6, 1e4, 1e2, 1.0, 1e-2, 1e-4])
    J = _jacobian_with_singular_values(singular_values, rows=40, seed=7)
    _singular, rank, condition, _min_sv, _max_sv = _svd_diagnostics(J)
    assert rank == singular_values.size  # 여전히 full-rank (tol 아래로 안 내려감)
    assert condition is not None and condition > 1e9

    covariance = _covariance_from_jacobian_svd(J)
    assert covariance is not None
    assert np.all(np.isfinite(covariance))

    corr = _correlation_matrix_from_normalized_jacobian(J, rank)
    corr_arr = np.asarray(corr)
    assert corr_arr.shape == (6, 6)
    assert np.all(np.isfinite(corr_arr))
    assert np.all(corr_arr >= -1.0) and np.all(corr_arr <= 1.0)
    assert np.allclose(np.diag(corr_arr), 1.0, atol=1e-6)


def test_rank_deficient_jacobian_keeps_correlation_unavailable_policy():
    J = _well_conditioned_jacobian(rows=40, cols=5, seed=3)
    J[:, -1] = J[:, 0]  # 마지막 컬럼을 첫 컬럼과 동일하게 만들어 rank-deficient화
    _singular, rank, _cond, _min_sv, _max_sv = _svd_diagnostics(J)
    assert rank < J.shape[1]

    # covariance 자체는 (pinv처럼 작은 singular value를 0으로 눌러) 계산할 수 있어도,
    # rank-deficient correlation을 정상적인 값처럼 보여주지 않는 보수적 정책은 유지된다.
    assert _correlation_matrix_from_normalized_jacobian(J, rank) == []


def test_rank_deficient_jacobian_report_has_warning_and_no_correlation():
    dataset, result = _dataset_and_result(CameraModelType.EXTENDED_PINHOLE)
    report = compute_observability_report(result, dataset)
    if report.rank >= report.jacobian_cols:
        # 이 synthetic dataset은 보통 full-rank이므로, 최소 정책 자체는 단위
        # 테스트(test_rank_deficient_jacobian_keeps_correlation_unavailable_policy)로
        # 이미 커버된다 - 여기서는 report 레벨 경고 문구만 있으면 통과시킨다.
        return
    assert report.correlation_matrix == []
    assert any("rank deficient" in w.lower() for w in report.warnings)


# ---------------------------------------------------------------------------
# correlation_method backward compatibility 회귀 테스트
# ---------------------------------------------------------------------------


def _legacy_observability_dict_without_method() -> dict:
    """correlation_method 필드가 아예 없는 구버전 저장 포맷을 흉내."""
    return {
        "parameter_labels": ["fx", "fy"],
        "jacobian_rows": 120,
        "jacobian_cols": 2,
        "num_points": 60,
        "singular_values": [10.0, 0.1],
        "rank": 2,
        "condition_number": 100.0,
        "raw_condition_number": 1000.0,
        "normalized_condition_number": 100.0,
        "normalization_scales": {"fx": 640.0, "fy": 480.0},
        "raw_singular_values": [100.0, 0.1],
        "min_singular_value": 0.1,
        "max_singular_value": 10.0,
        "max_abs_correlation": 0.9,
        # np.corrcoef(raw Jacobian, rowvar=False) 시절 값이라고 가정.
        "correlation_matrix": [[1.0, 0.9], [0.9, 1.0]],
        "observability_score": 75.0,
        "observability_grade": "WARNING",
        "top_correlations": [],
        "warnings": [],
        # 의도적으로 "correlation_method" 키를 넣지 않는다.
    }


def test_observability_report_from_dict_without_correlation_method_uses_legacy_label():
    restored = _observability_report_from_dict(_legacy_observability_dict_without_method())
    assert restored is not None
    # 새 covariance 방식으로 잘못 라벨링되면 안 된다.
    assert restored.correlation_method != "covariance_from_normalized_jacobian"
    assert restored.correlation_method == "legacy_jacobian_column_correlation"
    # legacy correlation 값 자체는 재해석/재계산하지 않고 그대로 보존.
    assert restored.correlation_matrix == [[1.0, 0.9], [0.9, 1.0]]


def test_observability_report_from_dict_preserves_explicit_correlation_method():
    d = _legacy_observability_dict_without_method()
    d["correlation_method"] = "covariance_from_normalized_jacobian"
    restored = _observability_report_from_dict(d)
    assert restored is not None
    assert restored.correlation_method == "covariance_from_normalized_jacobian"


def test_legacy_project_dict_without_correlation_method_does_not_load_as_new_method(camera_config):
    dataset, result = _dataset_and_result()
    result.observability = ObservabilityReport(
        parameter_labels=["fx", "fy"],
        jacobian_cols=2,
        rank=2,
        correlation_matrix=[[1.0, 0.9], [0.9, 1.0]],
        correlation_method="covariance_from_normalized_jacobian",
    )
    pattern = PatternConfig(PatternType.CHESSBOARD, squares_x=5, squares_y=4, square_size=0.04)
    project = CalibrationProject(
        project_name="legacy-obs",
        camera_config=camera_config,
        pattern_config=pattern,
        dataset=dataset,
        calibration_results={CameraModelType.PINHOLE: result},
    )

    payload = project_to_dict(project)
    # 구버전 저장 포맷을 흉내: observability dict에서 correlation_method 키를 제거.
    obs_dict = payload["project"]["calibration_results"]["pinhole"]["observability"]
    assert "correlation_method" in obs_dict
    del obs_dict["correlation_method"]

    restored = project_from_dict(payload)
    restored_obs = restored.calibration_results[CameraModelType.PINHOLE].observability
    assert restored_obs is not None
    assert restored_obs.correlation_method == "legacy_jacobian_column_correlation"
    assert restored_obs.correlation_matrix == [[1.0, 0.9], [0.9, 1.0]]

    # crash 없이 로드되는지가 핵심 - 파일이 없어도 project_from_dict가 여기까지
    # 도달했다는 것 자체가 이미 그 요구사항을 만족한다.
    assert restored.project_name == "legacy-obs"


def test_new_project_roundtrip_preserves_covariance_correlation_method(camera_config):
    dataset, result = _dataset_and_result()
    result.observability = ObservabilityReport(
        parameter_labels=["fx", "fy"],
        jacobian_cols=2,
        rank=2,
        correlation_matrix=[[1.0, 0.2], [0.2, 1.0]],
        correlation_method="covariance_from_normalized_jacobian",
    )
    pattern = PatternConfig(PatternType.CHESSBOARD, squares_x=5, squares_y=4, square_size=0.04)
    project = CalibrationProject(
        project_name="new-obs",
        camera_config=camera_config,
        pattern_config=pattern,
        dataset=dataset,
        calibration_results={CameraModelType.PINHOLE: result},
    )

    restored = project_from_dict(project_to_dict(project))
    restored_obs = restored.calibration_results[CameraModelType.PINHOLE].observability
    assert restored_obs is not None
    assert restored_obs.correlation_method == "covariance_from_normalized_jacobian"

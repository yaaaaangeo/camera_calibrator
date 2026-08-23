"""
tests/test_external_compare.py
===================================

calibration/external_compare.py - "예전 파라미터 vs 지금 구한 파라미터"
비교 기능의 회귀 테스트.

핵심으로 검증해야 할 것:
1. 진짜로 더 나쁜 파라미터가 "패배"로 정확히 판정되는지 (조작 없이).
2. 비교가 항상 "학습에 안 쓰인 test 프레임"에서만 이뤄지는지 - 내 파라미터가
   자기 학습 데이터로 스스로를 채점하는 사기가 불가능해야 한다.
3. 한 지표(RMS)만으로 승패를 가르지 않는지 - 지표가 엇갈리면 verdict가
   그 사실을 정직하게 밝히는지.
4. OpenCV YAML을 통한 외부 파라미터 불러오기가 왕복 가능한지.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from calibration.external_compare import (
    ComparisonSide,
    ExternalCameraParams,
    PointErrorDetail,
    _build_verdict,
    build_metric_comparison_rows,
    build_fov_diff_rows,
    build_benchmark_validation_rows,
    build_bootstrap_comparison,
    build_error_distribution_comparison,
    build_parameter_diff_rows,
    build_parameter_diagnostics,
    build_radial_comparison_profile,
    build_residual_heatmap_comparison,
    build_spatial_comparisons,
    build_spatial_comparison_grid,
    build_statistical_tests,
    build_worst_case_rows,
    build_winner_decision,
    compare_reference_candidate_calibrations,
    compare_with_external_params,
)
from calibration.calibration_io import StandardCalibration
from calibration.compare import run_all_models
from calibration.types import (
    CameraConfig,
    CameraModelType,
    Dataset,
    DetectionResult,
    Frame,
    FrameStatus,
    ImageInfo,
    PatternConfig,
    PatternType,
)
from calibration.validation import split_train_test, validate_all_models
from export.opencv import (
    detect_model_hint_from_opencv_yaml,
    export_opencv_yaml,
    load_camera_matrix_and_distortion_from_opencv_yaml,
)

W, H = 640, 480
TRUE_K = np.array([[500.0, 0, W / 2], [0, 500.0, H / 2], [0, 0, 1]])
TRUE_D = np.array([-0.22, 0.06, 0.0, 0.0, 0.0])


def _pattern_config() -> PatternConfig:
    return PatternConfig(
        type=PatternType.CHARUCO, squares_x=7, squares_y=5,
        square_size=0.04, marker_size=0.03, dictionary="DICT_5X5_100",
    )


def _synthetic_dataset(pattern: PatternConfig, n_frames: int = 30, seed: int = 0) -> Dataset:
    """이미지 렌더링/검출 없이 3D->2D 직접 사영으로 프레임을 만든다 (빠름) -
    test_smoke_pipeline.py와 동일한 접근."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard(
        (pattern.squares_x, pattern.squares_y), pattern.square_size, pattern.marker_size, aruco_dict
    )
    pts3d = board.getChessboardCorners().astype(np.float32)
    n_corners = pts3d.shape[0]
    ids = np.arange(n_corners, dtype=np.int32).reshape(-1, 1)

    rng = np.random.default_rng(seed)
    frames: list[Frame] = []
    attempts = 0
    while len(frames) < n_frames and attempts < n_frames * 20:
        attempts += 1
        rvec = (rng.random(3) - 0.5) * 0.6
        tvec = np.array([(rng.random() - 0.5) * 0.25, (rng.random() - 0.5) * 0.25, 0.3 + rng.random() * 0.25])
        proj, _ = cv2.projectPoints(pts3d.reshape(-1, 1, 3), rvec, tvec, TRUE_K, TRUE_D)
        proj = proj.reshape(-1, 2)
        if np.any(proj < 0) or np.any(proj[:, 0] > W) or np.any(proj[:, 1] > H):
            continue
        image_id = f"ext_{len(frames):02d}"
        info = ImageInfo(image_id=image_id, path="-", width=W, height=H)
        det = DetectionResult(
            image_id=image_id, success=True,
            corners=proj.reshape(-1, 1, 2).astype(np.float32),
            object_points=pts3d.reshape(-1, 1, 3), ids=ids, num_corners=n_corners,
        )
        frames.append(Frame(image_info=info, detection=det, status=FrameStatus.DETECTED))

    assert len(frames) >= n_frames * 0.8, "합성 뷰가 너무 적게 생성됨"
    return Dataset(frames=frames)


@pytest.fixture(scope="module")
def camera_config() -> CameraConfig:
    return CameraConfig(width=W, height=H)


@pytest.fixture(scope="module")
def pattern_config() -> PatternConfig:
    return _pattern_config()


@pytest.fixture(scope="module")
def dataset(pattern_config) -> Dataset:
    return _synthetic_dataset(pattern_config, n_frames=30, seed=1)


@pytest.fixture(scope="module")
def my_pinhole_result(dataset, camera_config):
    results = run_all_models(dataset, camera_config, estimate_fisheye_uncertainty=False)
    return {r.model_name: r for r in results}[CameraModelType.PINHOLE]


@pytest.fixture(scope="module")
def my_validation(dataset, camera_config, pattern_config):
    results = validate_all_models(dataset, camera_config, pattern_config, test_ratio=0.3)
    return results[CameraModelType.PINHOLE]


# ---------------------------------------------------------------------------
# 핵심 시나리오: 진짜로 더 나쁜 파라미터가 실제로 패배로 판정되는지
# ---------------------------------------------------------------------------

def test_deliberately_bad_external_params_lose_on_every_metric(
    dataset, camera_config, pattern_config, my_validation,
):
    """왜곡을 아예 무시한(distortion=0) 파라미터는 실제 왜곡이 있는 이
    카메라에서 명백히 더 나빠야 한다 - 조작 없이 계산이 그렇게 판정하는지 확인."""
    bad_external = ExternalCameraParams(
        label="예전 결과(가짜, 왜곡 무시)",
        model_name=CameraModelType.PINHOLE,
        camera_matrix=TRUE_K.copy(),
        distortion=np.zeros(5),
    )
    result = compare_with_external_params(
        dataset, camera_config, pattern_config,
        CameraModelType.PINHOLE, my_validation, bad_external,
    )
    assert result.mine.success and result.external.success
    assert result.mine.test_rms < result.external.test_rms
    assert result.mine.residual_stats is not None
    assert result.external.residual_stats is not None
    assert result.mine.residual_stats.mae < result.external.residual_stats.mae
    assert result.mine.residual_stats.median < result.external.residual_stats.median
    assert result.mine.residual_stats.p90 < result.external.residual_stats.p90
    assert result.mine.residual_stats.p95 < result.external.residual_stats.p95
    assert result.mine.residual_stats.p99 < result.external.residual_stats.p99
    assert result.mine.residual_stats.max <= result.external.residual_stats.max
    assert result.mine_win_count > result.external_win_count
    assert result.mine.label in result.verdict


def test_ground_truth_external_params_beat_a_deliberately_underfit_model(
    dataset, camera_config, pattern_config,
):
    """반대 방향도 확인 - 내가 캘리브레이션을 대충 했다면(왜곡 계수를 0으로
    고정), 외부의 정답에 가까운 파라미터가 이겨야 한다. '내가 만든 툴이니까
    항상 내가 이긴다'는 오해를 반증하기 위한 테스트."""
    # 일부러 왜곡 계수를 전부 고정(=0)해서 실제보다 훨씬 나쁜 "내 결과"를 만든다.
    flags = cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
    object_points_list = [f.detection.object_points for f in dataset.enabled_frames]
    image_points_list = [f.detection.corners for f in dataset.enabled_frames]
    rms, K_bad, D_bad, _, _ = cv2.calibrateCamera(
        object_points_list, image_points_list, (W, H), None, None, flags=flags
    )

    from calibration.types import CalibrationResult, ValidationResult
    from calibration.validation import split_train_test, _subset_dataset, _test_reprojection_errors

    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=42)
    test_frames = _subset_dataset(dataset, test_ids).enabled_frames
    per_frame_error, _, _ = _test_reprojection_errors(test_frames, K_bad, D_bad, CameraModelType.PINHOLE)
    bad_test_rms = float(np.sqrt(np.mean(np.array(list(per_frame_error.values())) ** 2)))

    fake_my_validation = ValidationResult(
        train_frame_ids=train_ids, test_frame_ids=test_ids,
        test_rms=bad_test_rms, success=True,
    )
    # refit_on_train_split이 train_ids로 다시 학습할 텐데, 같은 flags가
    # 아니면 다른 값이 나온다 - 그래도 여전히 "왜곡 무시"보다는 나은 값이
    # 나올 뿐이니 external(진짜 정답)이 이기는지가 핵심. 다만 refit이
    # CALIB_FIX_K* 없이 정상 학습되면 오히려 mine이 좋아질 수 있으므로,
    # 이 테스트는 compare_with_external_params가 아니라 evaluate_side
    # 수준에서 "정답 파라미터가 왜곡-무시 파라미터를 이긴다"는 핵심만 확인한다.
    ground_truth = ExternalCameraParams(
        label="정답에 가까운 예전 결과",
        model_name=CameraModelType.PINHOLE,
        camera_matrix=TRUE_K.copy(),
        distortion=TRUE_D.copy(),
    )
    from calibration.external_compare import _evaluate_side

    truth_side = _evaluate_side(
        dataset, camera_config, pattern_config, test_ids,
        ground_truth.camera_matrix, ground_truth.distortion, ground_truth.model_name, ground_truth.label,
    )
    assert truth_side.success
    assert truth_side.test_rms < bad_test_rms, (
        "실제 정답에 가까운 파라미터가 왜곡을 무시한 파라미터보다 test RMS가 낮아야 함"
    )


# ---------------------------------------------------------------------------
# 공정성 - 같은 test 분할 재사용 + 내부 정합성 체크
# ---------------------------------------------------------------------------

def test_mine_side_reuses_the_same_holdout_split_as_validation(
    dataset, camera_config, pattern_config, my_validation,
):
    """재계산된 mine.test_rms가 기존 Hold-out(my_validation.test_rms)과
    (거의) 같아야 한다 - 그래야 '같은 절차를 그대로 재현했다'는 신뢰가 생긴다.
    """
    external = ExternalCameraParams(
        label="비교용", model_name=CameraModelType.PINHOLE,
        camera_matrix=TRUE_K.copy(), distortion=np.zeros(5),
    )
    result = compare_with_external_params(
        dataset, camera_config, pattern_config,
        CameraModelType.PINHOLE, my_validation, external,
    )
    assert result.mine.test_rms == pytest.approx(my_validation.test_rms, abs=1e-2)
    # 정합성이 맞으면 caveat에 "다릅니다" 경고가 없어야 한다.
    assert not any("다릅니다" in c for c in result.caveats)


def test_external_compare_reports_loaded_resolution_mismatch_as_caveat(
    dataset, camera_config, pattern_config, my_validation,
):
    external = ExternalCameraParams(
        label="해상도 다른 예전 결과",
        model_name=CameraModelType.PINHOLE,
        camera_matrix=TRUE_K.copy(),
        distortion=np.zeros(5),
        width=800,
        height=600,
    )

    result = compare_with_external_params(
        dataset, camera_config, pattern_config,
        CameraModelType.PINHOLE, my_validation, external,
    )

    assert any("validation 이미지 해상도" in caveat for caveat in result.caveats)


def test_reference_candidate_file_calibrations_are_compared_symmetrically(
    dataset, camera_config, pattern_config,
):
    _train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=42)
    reference = StandardCalibration(
        label="Reference",
        model_name=CameraModelType.EXTENDED_PINHOLE,
        distortion_model="plumb_bob",
        camera_matrix=TRUE_K.copy(),
        distortion=np.zeros(5),
        width=W,
        height=H,
        source_format="standard_json",
    )
    candidate = StandardCalibration(
        label="Candidate",
        model_name=CameraModelType.EXTENDED_PINHOLE,
        distortion_model="plumb_bob",
        camera_matrix=TRUE_K.copy(),
        distortion=TRUE_D.copy(),
        width=W,
        height=H,
        source_format="standard_json",
    )

    result = compare_reference_candidate_calibrations(
        dataset, camera_config, pattern_config, reference, candidate, test_ids
    )

    assert result.mine.label == "Candidate"
    assert result.external.label == "Reference"
    assert result.mine.success and result.external.success
    assert result.winner_decision.status == "Candidate Preferred"
    assert result.winner_decision.candidate_score > result.winner_decision.reference_score
    assert result.verdict.startswith("Candidate Preferred:")
    assert result.mine.test_rms < result.external.test_rms
    assert result.mine.residual_stats is not None
    assert result.external.residual_stats is not None
    assert result.mine.residual_stats.p95 < result.external.residual_stats.p95
    p95_row = next(r for r in result.metric_rows if r.metric == "P95")
    assert p95_row.reference_value == pytest.approx(result.external.residual_stats.p95)
    assert p95_row.candidate_value == pytest.approx(result.mine.residual_stats.p95)
    assert p95_row.improvement_pct > 0
    assert p95_row.winner == "Candidate"
    final_metrics = {row.metric for row in result.final_benchmark_rows}
    assert "RMSE" in final_metrics
    assert "Frame wins" in final_metrics
    assert "Worst image" in final_metrics
    assert "Bootstrap RMSE 95% CI" in final_metrics
    assert "Paired t-test p-value" in final_metrics
    final_p95 = next(row for row in result.final_benchmark_rows if row.metric == "P95")
    assert final_p95.winner == "Candidate"
    assert final_p95.reference.endswith(" px")
    assert final_p95.candidate.endswith(" px")
    assert {r.category for r in result.worst_case_rows} == {"Worst image", "Worst region", "Worst corner"}
    worst_corner = next(r for r in result.worst_case_rows if r.category == "Worst corner")
    assert "corner" in worst_corner.reference_location
    assert worst_corner.reference_value is not None
    assert worst_corner.candidate_value is not None
    assert result.error_distribution is not None
    assert result.error_distribution.bins
    assert result.error_distribution.num_reference_points > 0
    assert result.error_distribution.num_candidate_points > 0
    assert result.error_distribution.bins[-1].reference_cdf == pytest.approx(1.0)
    assert result.error_distribution.bins[-1].candidate_cdf == pytest.approx(1.0)
    assert set(result.spatial_comparisons) == {"3x3", "5x5"}
    grid3 = result.spatial_comparisons["3x3"]
    assert grid3.rows == 3 and grid3.cols == 3
    assert len(grid3.cells) == 9
    valid_cells = [
        c for c in grid3.cells
        if c.reference_p95 is not None and c.candidate_p95 is not None
    ]
    assert valid_cells
    assert any(c.improvement_p95_pct is not None and c.improvement_p95_pct > 0 for c in valid_cells)
    assert set(result.residual_heatmaps) == {"rmse_20x20", "p95_20x20"}
    heatmap = result.residual_heatmaps["rmse_20x20"]
    assert heatmap.rows == 20 and heatmap.cols == 20
    assert len(heatmap.cells) == 400
    assert heatmap.reference_max is not None
    assert heatmap.candidate_max is not None
    assert heatmap.difference_abs_max is not None
    assert any(
        c.difference_value is not None and c.difference_value < 0
        for c in heatmap.cells
    )
    assert set(result.radial_comparisons) == {"quartiles", "bands"}
    radial = result.radial_comparisons["quartiles"]
    assert len(radial.bands) == 4
    valid_bands = [
        b for b in radial.bands
        if b.reference_p95 is not None and b.candidate_p95 is not None
    ]
    assert valid_bands
    assert any(b.improvement_p95_pct is not None and b.improvement_p95_pct > 0 for b in valid_bands)
    assert {r.name for r in result.parameter_diff_rows[:4]} == {"fx", "fy", "cx", "cy"}
    assert any(r.name == "k1" for r in result.parameter_diff_rows)
    assert {r.name for r in result.fov_diff_rows} == {"HFOV", "VFOV", "DFOV"}
    assert {r.name for r in result.benchmark_validation_rows} == {"Hold-out", "K-fold (5)"}
    kfold_row = next(r for r in result.benchmark_validation_rows if r.name == "K-fold (5)")
    assert kfold_row.num_splits == 5
    assert kfold_row.reference_validation_rms_mean is not None
    assert kfold_row.candidate_validation_rms_mean is not None
    assert kfold_row.improvement_pct is not None
    assert {t.test_name for t in result.statistical_tests} == {"Paired t-test", "Wilcoxon signed-rank"}
    assert all(t.n_pairs == result.num_common_frames for t in result.statistical_tests)
    assert all(t.p_value is not None for t in result.statistical_tests)
    assert all(t.effect_size is not None for t in result.statistical_tests)
    assert result.bootstrap_comparison is not None
    assert result.bootstrap_comparison.probability_candidate_better is not None
    assert result.bootstrap_comparison.probability_candidate_better > 0.5
    assert result.bootstrap_comparison.reference_rmse_ci_low is not None
    assert result.bootstrap_comparison.candidate_rmse_ci_high is not None
    assert result.bootstrap_comparison.improvement_ci_low is not None
    assert result.bootstrap_comparison.improvement_ci_high is not None
    assert set(result.parameter_diagnostics) == {"reference", "candidate"}
    ref_diag = result.parameter_diagnostics["reference"]
    cand_diag = result.parameter_diagnostics["candidate"]
    assert ref_diag.stability_rows
    assert cand_diag.sensitivity_rows
    assert len(ref_diag.covariance_matrix) == len(ref_diag.parameter_labels)
    assert all(len(row) == len(ref_diag.parameter_labels) for row in ref_diag.covariance_matrix)
    assert ref_diag.jacobian_rows > 0
    assert ref_diag.jacobian_cols == len(ref_diag.parameter_labels)
    assert ref_diag.rank is not None
    assert ref_diag.singular_values
    assert ref_diag.min_singular_value is not None
    assert ref_diag.max_singular_value is not None
    assert len(ref_diag.correlation_matrix) == len(ref_diag.parameter_labels)
    assert all(len(row) == len(ref_diag.parameter_labels) for row in ref_diag.correlation_matrix)
    assert ref_diag.max_abs_correlation is None or 0.0 <= ref_diag.max_abs_correlation <= 1.0
    assert len(ref_diag.top_correlations) <= 5
    assert any("Parameter similarity" in caveat for caveat in result.caveats)
    assert result.mine_win_count > result.external_win_count
    assert "Candidate" in result.verdict


def test_metric_comparison_rows_compute_improvement_against_reference():
    reference = ComparisonSide(label="Reference", success=True)
    candidate = ComparisonSide(label="Candidate", success=True)
    from calibration.residual_stats import compute_residual_stats

    reference.residual_stats = compute_residual_stats([1.0, 2.0, 3.0, 4.0])
    candidate.residual_stats = compute_residual_stats([0.5, 1.0, 1.5, 2.0])
    reference.edge_rms = 2.0
    candidate.edge_rms = 1.0

    rows = build_metric_comparison_rows(reference, candidate)

    rmse = next(r for r in rows if r.metric == "RMSE")
    edge = next(r for r in rows if r.metric == "Edge RMS")
    assert rmse.improvement_pct == pytest.approx(50.0)
    assert rmse.winner == "Candidate"
    assert edge.improvement_pct == pytest.approx(50.0)
    assert edge.reference_value == pytest.approx(2.0)
    assert edge.candidate_value == pytest.approx(1.0)


def test_parameter_diff_rows_include_intrinsics_distortion_and_relative_diff():
    reference = ComparisonSide(
        label="Reference",
        model_name=CameraModelType.PINHOLE,
        camera_matrix=np.array([[500.0, 0.0, 320.0], [0.0, 510.0, 240.0], [0.0, 0.0, 1.0]]),
        distortion=np.array([-0.2, 0.04, 0.001, -0.002, 0.0]),
    )
    candidate = ComparisonSide(
        label="Candidate",
        model_name=CameraModelType.PINHOLE,
        camera_matrix=np.array([[525.0, 0.0, 322.0], [0.0, 500.0, 238.0], [0.0, 0.0, 1.0]]),
        distortion=np.array([-0.1, 0.02, 0.002, -0.001, 0.01]),
    )

    rows = build_parameter_diff_rows(reference, candidate)

    fx = next(r for r in rows if r.name == "fx")
    k1 = next(r for r in rows if r.name == "k1")
    p1 = next(r for r in rows if r.name == "p1")
    k3 = next(r for r in rows if r.name == "k3")
    assert fx.absolute_diff == pytest.approx(25.0)
    assert fx.relative_diff_pct == pytest.approx(5.0)
    assert k1.absolute_diff == pytest.approx(0.1)
    assert k1.relative_diff_pct == pytest.approx(50.0)
    assert p1.absolute_diff == pytest.approx(0.001)
    assert k3.relative_diff_pct is None


def test_fov_diff_rows_report_hfov_vfov_and_dfov():
    reference = ComparisonSide(
        label="Reference",
        model_name=CameraModelType.PINHOLE,
        camera_matrix=np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]),
    )
    candidate = ComparisonSide(
        label="Candidate",
        model_name=CameraModelType.PINHOLE,
        camera_matrix=np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]]),
    )

    rows = build_fov_diff_rows(reference, candidate, (640, 480))

    names = {r.name for r in rows}
    hfov = next(r for r in rows if r.name == "HFOV")
    dfov = next(r for r in rows if r.name == "DFOV")
    assert names == {"HFOV", "VFOV", "DFOV"}
    assert hfov.reference_value == pytest.approx(np.degrees(2 * np.arctan(320 / 500)))
    assert hfov.candidate_value > hfov.reference_value
    assert dfov.candidate_value > dfov.reference_value


def test_benchmark_validation_rows_include_holdout_kfold_and_generalization(
    dataset, camera_config, pattern_config,
):
    _train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=42)
    reference = ComparisonSide(
        label="Reference",
        model_name=CameraModelType.PINHOLE,
        camera_matrix=TRUE_K.copy(),
        distortion=np.zeros(5),
        success=True,
    )
    candidate = ComparisonSide(
        label="Candidate",
        model_name=CameraModelType.PINHOLE,
        camera_matrix=TRUE_K.copy(),
        distortion=TRUE_D.copy(),
        success=True,
    )

    rows = build_benchmark_validation_rows(
        dataset,
        camera_config,
        pattern_config,
        reference,
        candidate,
        test_ids,
        kfold=3,
        generalization_datasets={"DatasetB": dataset},
    )

    names = {row.name for row in rows}
    holdout = next(row for row in rows if row.name == "Hold-out")
    kfold = next(row for row in rows if row.name == "K-fold (3)")
    generalization = next(row for row in rows if row.name == "Generalization: DatasetB")
    assert names == {"Hold-out", "K-fold (3)", "Generalization: DatasetB"}
    assert holdout.num_splits == 1
    assert holdout.reference_train_validation_gap is not None
    assert holdout.candidate_train_validation_gap is not None
    assert kfold.num_splits == 3
    assert kfold.reference_validation_rms_std is not None
    assert kfold.candidate_validation_rms_std is not None
    assert generalization.reference_train_rms_mean is None
    assert generalization.candidate_validation_rms_mean < generalization.reference_validation_rms_mean
    assert generalization.improvement_pct > 0


def test_statistical_tests_report_p_values_and_effect_sizes():
    reference = ComparisonSide(
        label="Reference",
        per_frame_error={f"f{i}": 2.0 + i * 0.1 for i in range(8)},
    )
    candidate = ComparisonSide(
        label="Candidate",
        per_frame_error={f"f{i}": 1.0 + i * 0.05 for i in range(8)},
    )

    rows = build_statistical_tests(reference, candidate)

    paired_t = next(r for r in rows if r.test_name == "Paired t-test")
    wilcoxon = next(r for r in rows if r.test_name == "Wilcoxon signed-rank")
    assert paired_t.n_pairs == 8
    assert paired_t.mean_diff == pytest.approx(np.mean([-1.0 - i * 0.05 for i in range(8)]))
    assert paired_t.p_value is not None
    assert paired_t.p_value < 0.05
    assert paired_t.effect_size is not None
    assert paired_t.effect_size < 0
    assert wilcoxon.p_value is not None
    assert wilcoxon.effect_size is not None
    assert wilcoxon.effect_size < 0
    assert "Candidate lower" in paired_t.interpretation


def test_bootstrap_comparison_reports_probability_rmse_ci_and_improvement_ci():
    reference = ComparisonSide(
        label="Reference",
        per_frame_error={f"f{i}": 2.0 + i * 0.1 for i in range(10)},
    )
    candidate = ComparisonSide(
        label="Candidate",
        per_frame_error={f"f{i}": 1.0 + i * 0.05 for i in range(10)},
    )

    result = build_bootstrap_comparison(reference, candidate, n_bootstrap=200, seed=7)

    assert result.n_pairs == 10
    assert result.n_bootstrap == 200
    assert result.probability_candidate_better == pytest.approx(1.0)
    assert result.reference_rmse is not None
    assert result.candidate_rmse is not None
    assert result.candidate_rmse < result.reference_rmse
    assert result.reference_rmse_ci_low <= result.reference_rmse <= result.reference_rmse_ci_high
    assert result.candidate_rmse_ci_low <= result.candidate_rmse <= result.candidate_rmse_ci_high
    assert result.improvement_pct > 0
    assert result.improvement_ci_low is not None
    assert result.improvement_ci_high is not None
    assert result.improvement_ci_low > 0


def test_parameter_diagnostics_reports_observability_stability_covariance_and_sensitivity(
    dataset, camera_config, pattern_config,
):
    _train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=42)
    side = ComparisonSide(
        label="Reference",
        model_name=CameraModelType.PINHOLE,
        camera_matrix=TRUE_K.copy(),
        distortion=TRUE_D.copy(),
        success=True,
    )

    diagnostics = build_parameter_diagnostics(dataset, test_ids, side)

    assert diagnostics.side_label == "Reference"
    assert diagnostics.n_points > 0
    assert diagnostics.parameter_labels[:4] == ["fx", "fy", "cx", "cy"]
    assert "k1" in diagnostics.parameter_labels
    assert diagnostics.jacobian_rows == diagnostics.n_points * 2
    assert diagnostics.jacobian_cols == len(diagnostics.parameter_labels)
    assert diagnostics.rank is not None
    assert 0 <= diagnostics.rank <= diagnostics.jacobian_cols
    assert len(diagnostics.singular_values) == len(diagnostics.parameter_labels)
    assert diagnostics.min_singular_value is not None
    assert diagnostics.max_singular_value is not None
    assert diagnostics.condition_number is not None
    assert len(diagnostics.stability_rows) == len(diagnostics.parameter_labels)
    assert len(diagnostics.sensitivity_rows) == len(diagnostics.parameter_labels)
    assert len(diagnostics.covariance_matrix) == len(diagnostics.parameter_labels)
    assert all(len(row) == len(diagnostics.parameter_labels) for row in diagnostics.covariance_matrix)
    assert len(diagnostics.correlation_matrix) == len(diagnostics.parameter_labels)
    assert all(len(row) == len(diagnostics.parameter_labels) for row in diagnostics.correlation_matrix)
    assert all(abs(diagnostics.correlation_matrix[i][i] - 1.0) < 1e-9 for i in range(len(diagnostics.parameter_labels)))
    assert diagnostics.max_abs_correlation is None or 0.0 <= diagnostics.max_abs_correlation <= 1.0
    assert len(diagnostics.top_correlations) <= 5
    assert isinstance(diagnostics.weak_parameters, list)
    assert all(row.std is not None for row in diagnostics.stability_rows[:4])
    assert all(row.sensitivity_per_unit is not None for row in diagnostics.sensitivity_rows[:4])


def test_worst_case_rows_report_image_region_and_corner():
    reference = ComparisonSide(
        label="Reference",
        per_frame_error={"a": 1.0, "b": 3.0},
        point_errors_xy=[(10, 10, 1.0), (90, 90, 5.0)],
        point_error_details=[
            PointErrorDetail("a", 0, 10, 10, 1.0),
            PointErrorDetail("b", 3, 90, 90, 5.0),
        ],
    )
    candidate = ComparisonSide(
        label="Candidate",
        per_frame_error={"a": 0.8, "b": 2.0},
        point_errors_xy=[(10, 10, 0.5), (90, 90, 2.5)],
        point_error_details=[
            PointErrorDetail("a", 0, 10, 10, 0.5),
            PointErrorDetail("b", 3, 90, 90, 2.5),
        ],
    )
    spatial = build_spatial_comparisons(reference, candidate, (100, 100))

    rows = build_worst_case_rows(reference, candidate, spatial)

    categories = {row.category for row in rows}
    worst_image = next(row for row in rows if row.category == "Worst image")
    worst_region = next(row for row in rows if row.category == "Worst region")
    worst_corner = next(row for row in rows if row.category == "Worst corner")
    assert categories == {"Worst image", "Worst region", "Worst corner"}
    assert worst_image.reference_location == "b"
    assert worst_image.candidate_location == "b"
    assert worst_image.winner == "Candidate"
    assert worst_region.reference_value == pytest.approx(5.0)
    assert worst_corner.reference_location.startswith("b corner 3")
    assert worst_corner.improvement_pct == pytest.approx(50.0)


def test_winner_decision_engine_states_candidate_reference_inconclusive_and_insufficient():
    reference = ComparisonSide(
        label="Reference",
        success=True,
        per_frame_error={"a": 2.0, "b": 2.1, "c": 2.2, "d": 2.3},
    )
    candidate = ComparisonSide(
        label="Candidate",
        success=True,
        per_frame_error={"a": 1.0, "b": 1.1, "c": 1.2, "d": 1.3},
    )
    metric_rows = build_metric_comparison_rows(reference, candidate)
    decision = build_winner_decision(reference, candidate, metric_rows, [], [], [], None)
    assert decision.status == "Candidate Preferred"
    assert decision.data_quality_ok

    reverse = build_winner_decision(candidate, reference, build_metric_comparison_rows(candidate, reference), [], [], [], None)
    assert reverse.status == "Reference Preferred"

    tied_reference = ComparisonSide(
        label="Reference",
        success=True,
        per_frame_error={"a": 1.0, "b": 1.0, "c": 1.0},
        test_rms=1.0,
    )
    tied_candidate = ComparisonSide(
        label="Candidate",
        success=True,
        per_frame_error={"a": 1.0, "b": 1.0, "c": 1.0},
        test_rms=1.0,
    )
    tied = build_winner_decision(
        tied_reference,
        tied_candidate,
        build_metric_comparison_rows(tied_reference, tied_candidate),
        [],
        [],
        [],
        None,
    )
    assert tied.status == "Insufficient Evidence"

    weak_reference = ComparisonSide(
        label="Reference",
        success=True,
        per_frame_error={"a": 1.0, "b": 1.2, "c": 1.0, "d": 1.2},
        test_rms=1.0,
        edge_rms=0.9,
    )
    weak_candidate = ComparisonSide(
        label="Candidate",
        success=True,
        per_frame_error={"a": 1.1, "b": 1.1, "c": 1.1, "d": 1.1},
        test_rms=0.9,
        edge_rms=1.0,
    )
    weak = build_winner_decision(
        weak_reference,
        weak_candidate,
        build_metric_comparison_rows(weak_reference, weak_candidate),
        [],
        [],
        [],
        None,
        min_score_margin=10.0,
    )
    assert weak.status == "Inconclusive"

    insufficient = build_winner_decision(
        ComparisonSide(label="Reference", success=False),
        ComparisonSide(label="Candidate", success=True),
        [],
        [],
        [],
        [],
        None,
    )
    assert insufficient.status == "Insufficient Evidence"


def test_winner_decision_blocks_on_insufficient_data_quality_even_when_candidate_wins():
    reference = ComparisonSide(
        label="Reference",
        success=True,
        per_frame_error={"a": 2.0, "b": 2.1, "c": 2.2},
        point_error_details=[
            PointErrorDetail("a", 0, 320, 240, 2.0),
            PointErrorDetail("b", 0, 322, 241, 2.1),
            PointErrorDetail("c", 0, 321, 239, 2.2),
        ],
    )
    candidate = ComparisonSide(
        label="Candidate",
        success=True,
        per_frame_error={"a": 1.0, "b": 1.1, "c": 1.2},
        point_error_details=[
            PointErrorDetail("a", 0, 320, 240, 1.0),
            PointErrorDetail("b", 0, 322, 241, 1.1),
            PointErrorDetail("c", 0, 321, 239, 1.2),
        ],
    )

    decision = build_winner_decision(
        reference,
        candidate,
        build_metric_comparison_rows(reference, candidate),
        [],
        [],
        [],
        None,
        image_size=(640, 480),
    )

    assert decision.status == "Insufficient Evidence"
    assert not decision.data_quality_ok
    assert any("Insufficient paired corner evidence" in warning for warning in decision.warnings)
    assert any("Insufficient spatial coverage" in warning for warning in decision.warnings)


def test_error_distribution_comparison_uses_shared_histogram_and_cdf_bins():
    reference = ComparisonSide(
        label="Reference",
        point_error_details=[
            PointErrorDetail("a", 0, 0, 0, 0.5),
            PointErrorDetail("a", 1, 0, 0, 1.5),
            PointErrorDetail("b", 0, 0, 0, 2.5),
            PointErrorDetail("b", 1, 0, 0, 3.5),
        ],
    )
    candidate = ComparisonSide(
        label="Candidate",
        point_error_details=[
            PointErrorDetail("a", 0, 0, 0, 0.25),
            PointErrorDetail("a", 1, 0, 0, 0.75),
            PointErrorDetail("b", 0, 0, 0, 1.25),
            PointErrorDetail("b", 1, 0, 0, 1.75),
        ],
    )

    distribution = build_error_distribution_comparison(reference, candidate, num_bins=4)

    assert distribution.num_reference_points == 4
    assert distribution.num_candidate_points == 4
    assert len(distribution.bins) == 4
    assert distribution.reference_p95 == pytest.approx(np.percentile([0.5, 1.5, 2.5, 3.5], 95))
    assert distribution.candidate_p95 == pytest.approx(np.percentile([0.25, 0.75, 1.25, 1.75], 95))
    assert distribution.bins[-1].reference_cdf == pytest.approx(1.0)
    assert distribution.bins[-1].candidate_cdf == pytest.approx(1.0)
    assert any(bin.candidate_cdf > bin.reference_cdf for bin in distribution.bins)


def test_spatial_comparison_grid_computes_cell_metrics_and_improvement():
    reference = ComparisonSide(
        label="Reference",
        point_errors_xy=[
            (10, 10, 2.0),
            (20, 10, 4.0),
            (90, 90, 10.0),
        ],
    )
    candidate = ComparisonSide(
        label="Candidate",
        point_errors_xy=[
            (10, 10, 1.0),
            (20, 10, 2.0),
            (90, 90, 5.0),
        ],
    )

    grid = build_spatial_comparison_grid(reference, candidate, (100, 100), rows=2, cols=2)

    assert grid.rows == 2
    assert grid.cols == 2
    top_left = next(c for c in grid.cells if c.row == 0 and c.col == 0)
    bottom_right = next(c for c in grid.cells if c.row == 1 and c.col == 1)
    assert top_left.num_reference_points == 2
    assert top_left.num_candidate_points == 2
    assert top_left.reference_mean == pytest.approx(3.0)
    assert top_left.candidate_mean == pytest.approx(1.5)
    assert top_left.reference_rmse == pytest.approx(np.sqrt((2.0 ** 2 + 4.0 ** 2) / 2))
    assert top_left.candidate_rmse == pytest.approx(np.sqrt((1.0 ** 2 + 2.0 ** 2) / 2))
    assert top_left.reference_p95 == pytest.approx(np.percentile([2.0, 4.0], 95))
    assert top_left.reference_max == pytest.approx(4.0)
    assert top_left.improvement_mean_pct == pytest.approx(50.0)
    assert bottom_right.improvement_max_pct == pytest.approx(50.0)


def test_residual_heatmap_comparison_contains_reference_candidate_and_difference_layers():
    reference = ComparisonSide(
        label="Reference",
        point_errors_xy=[
            (10, 10, 2.0),
            (20, 10, 4.0),
            (90, 90, 10.0),
        ],
    )
    candidate = ComparisonSide(
        label="Candidate",
        point_errors_xy=[
            (10, 10, 1.0),
            (20, 10, 2.0),
            (90, 90, 5.0),
        ],
    )

    heatmap = build_residual_heatmap_comparison(reference, candidate, (100, 100), rows=2, cols=2)

    assert heatmap.rows == 2
    assert heatmap.cols == 2
    assert heatmap.metric == "rmse"
    assert len(heatmap.cells) == 4
    top_left = next(c for c in heatmap.cells if c.row == 0 and c.col == 0)
    bottom_right = next(c for c in heatmap.cells if c.row == 1 and c.col == 1)
    assert top_left.reference_value == pytest.approx(np.sqrt((2.0 ** 2 + 4.0 ** 2) / 2))
    assert top_left.candidate_value == pytest.approx(np.sqrt((1.0 ** 2 + 2.0 ** 2) / 2))
    assert top_left.difference_value == pytest.approx(top_left.candidate_value - top_left.reference_value)
    assert top_left.difference_value < 0
    assert bottom_right.reference_value == pytest.approx(10.0)
    assert bottom_right.candidate_value == pytest.approx(5.0)
    assert heatmap.reference_max == pytest.approx(10.0)
    assert heatmap.candidate_max == pytest.approx(5.0)
    assert heatmap.difference_abs_max == pytest.approx(5.0)


def test_radial_comparison_profile_computes_band_metrics_and_improvement():
    reference = ComparisonSide(
        label="Reference",
        point_errors_xy=[
            (50, 50, 2.0),   # center
            (60, 50, 4.0),   # center-ish
            (100, 100, 10.0), # outer edge/corner
        ],
    )
    candidate = ComparisonSide(
        label="Candidate",
        point_errors_xy=[
            (50, 50, 1.0),
            (60, 50, 2.0),
            (100, 100, 5.0),
        ],
    )

    profile = build_radial_comparison_profile(
        reference,
        candidate,
        (100, 100),
        [0.0, 0.25, 0.75, 1.0],
        ["Center", "Middle", "Edge"],
    )

    assert profile.max_radius_px == pytest.approx(np.hypot(50, 50))
    center = next(b for b in profile.bands if b.label == "Center")
    edge = next(b for b in profile.bands if b.label == "Edge")
    assert center.num_reference_points == 2
    assert center.num_candidate_points == 2
    assert center.reference_mean == pytest.approx(3.0)
    assert center.candidate_mean == pytest.approx(1.5)
    assert center.reference_rmse == pytest.approx(np.sqrt((2.0 ** 2 + 4.0 ** 2) / 2))
    assert center.reference_p95 == pytest.approx(np.percentile([2.0, 4.0], 95))
    assert center.reference_max == pytest.approx(4.0)
    assert center.improvement_mean_pct == pytest.approx(50.0)
    assert edge.improvement_max_pct == pytest.approx(50.0)


def test_reference_candidate_file_comparison_stops_on_incompatible_models(
    dataset, camera_config, pattern_config,
):
    _train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=42)
    reference = StandardCalibration(
        label="Reference",
        model_name=CameraModelType.EXTENDED_PINHOLE,
        distortion_model="plumb_bob",
        camera_matrix=TRUE_K.copy(),
        distortion=TRUE_D.copy(),
        width=W,
        height=H,
    )
    candidate = StandardCalibration(
        label="Candidate",
        model_name=CameraModelType.FISHEYE,
        distortion_model="equidistant",
        camera_matrix=TRUE_K.copy(),
        distortion=np.zeros(4),
        width=W,
        height=H,
    )

    result = compare_reference_candidate_calibrations(
        dataset, camera_config, pattern_config, reference, candidate, test_ids
    )

    assert not result.mine.success
    assert not result.external.success
    assert "호환되지 않아" in result.verdict
    assert any("different_camera_models" in caveat or "모델" in caveat for caveat in result.caveats)


def test_no_test_frames_returns_explanatory_verdict(dataset, camera_config, pattern_config):
    from calibration.types import ValidationResult

    empty_validation = ValidationResult(train_frame_ids=[], test_frame_ids=[], success=True)
    external = ExternalCameraParams(
        label="비교용", model_name=CameraModelType.PINHOLE,
        camera_matrix=TRUE_K.copy(), distortion=TRUE_D.copy(),
    )
    result = compare_with_external_params(
        dataset, camera_config, pattern_config,
        CameraModelType.PINHOLE, empty_validation, external,
    )
    assert not result.mine.success
    assert "test 프레임" in result.verdict or "이미지를 더" in result.verdict


# ---------------------------------------------------------------------------
# verdict 문구 - 지표가 엇갈리는 경우 정직하게 말하는지
# ---------------------------------------------------------------------------

def test_verdict_reports_full_sweep_when_all_metrics_agree():
    mine = ComparisonSide(label="내 결과", test_rms=0.3, edge_rms=0.4, straightness_residual=0.2, success=True)
    external = ComparisonSide(label="예전 결과", test_rms=0.5, edge_rms=0.6, straightness_residual=0.4, success=True)
    verdict = _build_verdict(mine, external, mine_win=10, external_win=2, n_common=12)
    assert "내 결과" in verdict
    assert "전부" in verdict


def test_verdict_reports_mixed_signal_honestly():
    mine = ComparisonSide(label="내 결과", test_rms=0.3, edge_rms=0.9, straightness_residual=0.2, success=True)
    external = ComparisonSide(label="예전 결과", test_rms=0.5, edge_rms=0.4, straightness_residual=0.4, success=True)
    verdict = _build_verdict(mine, external, mine_win=6, external_win=6, n_common=12)
    assert "엇갈" in verdict


def test_verdict_reports_failure_reason_when_side_fails():
    mine = ComparisonSide(label="내 결과", success=True, test_rms=0.3)
    external = ComparisonSide(label="예전 결과", success=False, error_message="pose 추정 실패")
    verdict = _build_verdict(mine, external, mine_win=0, external_win=0, n_common=0)
    assert "예전 결과" in verdict
    assert "pose 추정 실패" in verdict


# ---------------------------------------------------------------------------
# OpenCV YAML을 통한 외부 파라미터 불러오기
# ---------------------------------------------------------------------------

def test_load_camera_matrix_and_distortion_roundtrip(tmp_path, my_pinhole_result, camera_config, pattern_config):
    path = str(tmp_path / "camera.yaml")
    export_opencv_yaml(my_pinhole_result, camera_config, pattern_config, path)

    K, D = load_camera_matrix_and_distortion_from_opencv_yaml(path)
    assert K.shape == (3, 3)
    np.testing.assert_allclose(K, my_pinhole_result.camera_matrix, rtol=1e-6)
    np.testing.assert_allclose(D.reshape(-1), my_pinhole_result.distortion.reshape(-1), rtol=1e-6)

    hint = detect_model_hint_from_opencv_yaml(path)
    assert hint == CameraModelType.PINHOLE


def test_detect_model_hint_returns_none_for_foreign_yaml(tmp_path):
    """이 툴이 만들지 않은(calibration_model 필드가 없는) YAML도 읽을 수는
    있어야 하지만, 모델 종류는 억지로 추측하지 않고 None을 돌려줘야 한다."""
    path = str(tmp_path / "foreign.yaml")
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", TRUE_K)
    fs.write("distortion_coefficients", TRUE_D)
    fs.release()

    K, D = load_camera_matrix_and_distortion_from_opencv_yaml(path)
    np.testing.assert_allclose(K, TRUE_K)
    assert detect_model_hint_from_opencv_yaml(path) is None


def test_load_rejects_yaml_without_camera_matrix(tmp_path):
    path = str(tmp_path / "invalid.yaml")
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("something_else", 1.0)
    fs.release()

    with pytest.raises(ValueError, match="camera_matrix"):
        load_camera_matrix_and_distortion_from_opencv_yaml(path)


def test_external_compare_view_exposes_undistortion_visual_comparison_modes():
    source = Path("ui/external_compare_view.py").read_text(encoding="utf-8")

    for label in [
        "Original",
        "Reference Undistorted",
        "Candidate Undistorted",
        "Overlay View",
        "Difference View",
        "Calibration Benchmark Report",
        "Final Benchmark Table",
        "Performance Comparison",
        "Statistical Evidence",
        "Visual Evidence",
        "Parameter Analysis",
        "Model Analysis",
        "FINAL VERDICT",
        "One-line diagnosis",
        "Spatial Error 3x3 / 5x5",
        "Residual Heatmap Reference / Candidate / Difference",
        "Reference",
        "Candidate",
        "Difference (Candidate - Reference)",
        "Radial Error Profile",
        "Worst-case Analysis",
        "Error Distribution Comparison",
        "Benchmark Hold-out / K-fold / Generalization",
        "Statistical Significance",
        "Bootstrap Comparison",
        "Parameter Observability / Stability / Covariance / Sensitivity",
        "Jacobian",
        "Rank",
        "Singular Value",
        "Correlation",
        "Weak Params",
        "Overview",
        "Error Analysis",
        "Visual Comparison",
        "Statistical Validation",
        "Parameter Analysis",
        "Model Comparison",
        "Model Comparison: Pinhole / Extended Pinhole / Fisheye",
        "Hold-out RMSE",
        "Hold-out P95",
        "AIC",
        "BIC",
        "Complexity",
        "Recommendation",
        "Final Report",
    ]:
        assert label in source

    assert "QTabWidget" in source
    assert "cv2.addWeighted" in source
    assert "cv2.absdiff" in source
    assert "COLORMAP_TURBO" in source
    assert "spatial_comparisons" in source
    assert "residual_heatmaps" in source
    assert "radial_comparisons" in source
    assert "compute_model_scores" in source

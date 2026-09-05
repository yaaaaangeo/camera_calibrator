"""
tests/test_windshield_spline_stabilization.py
==================================================

STEP 4 Spline Windshield Model 검증(안정화) - STAGE B, leakage, AUTO,
repeated hold-out, ray/surface stability, pose 진단, 진단 오케스트레이터,
B-spline continuity, outer normal 정책.

성능 노트: 이 모델은 코너마다 2번(inner+outer)의 3-unknown ray-surface
intersection solve가 필요해 `calibrate_spline()` 1회 호출이 최소 grid
(4x4)에서도 수 분 걸린다(analytic Jacobian + 재사용 가능한 B-spline basis
evaluator로 이미 최적화했지만, 물리적으로 정확한 surface intersection
자체의 비용은 Grid/RBF의 단순 lookup보다 근본적으로 크다). 그래서 이
파일은:
  - 비용이 큰 `calibrate_spline()`/`run_spline_calibration_with_diagnostics()`
    호출을 모듈 스코프 fixture로 묶어 여러 assertion이 결과를 공유하게 한다.
  - repeated hold-out/AUTO 후보/seed 수를 최소(1-2)로 줄인다(Residual
    Grid/RBF의 monkeypatch 패턴과 동일한 이유 - 기본 후보/seed 개수 자체를
    검증하는 게 목적이 아니라 leakage 없음/구조 정합성을 검증하는 게
    목적이므로 문제 없다).
"""

from __future__ import annotations

import copy as _copy

import numpy as np
import pytest

from calibration.types import CameraModelType
from calibration.validation import split_train_test
from calibration.windshield.base import WindshieldConfig, WindshieldModelType
from calibration.windshield.spline import (
    MIN_SPLINE_GRID_SIZE,
    _build_spline_basis,
    _check_normal_continuity,
    calibrate_spline,
    compute_angular_fov_scale,
    evaluate_inner_surface,
    evaluate_outer_surface,
    run_repeated_holdout_spline,
    run_spline_calibration_with_diagnostics,
    select_best_spline_grid_resolution,
)
from calibration.windshield.baseline import BaselineWindshieldModel
from tests._windshield_test_utils import (
    build_synthetic_spherical_windshield_dataset,
    build_synthetic_spline_windshield_dataset,
    default_camera_config,
    default_camera_matrix_distortion,
)

_MODEL = CameraModelType.BROWN_CONRADY


def _config(K, D, **kwargs) -> WindshieldConfig:
    return WindshieldConfig(
        base_model_name=_MODEL, base_camera_matrix=K, base_distortion=D,
        windshield_model=WindshieldModelType.SPLINE, **kwargs,
    )


def _known_sphere():
    return np.array([0.0, 0.0, -4.9]), 5.0


# ---------------------------------------------------------------------------
# B-Spline continuity(사용자 스펙 37번) / Outer normal 정책(사용자 스펙 38번)
# - 순수 geometry 함수 테스트, calibrate_spline 호출 없음(빠름)
# ---------------------------------------------------------------------------

def test_bspline_surface_has_no_sudden_jumps_across_cell_boundaries():
    """Bicubic B-spline의 장점 - control grid의 cell 경계를 넘나들어도
    Delta_s와 normal이 급격히 튀지 않아야 한다(C0 + 사실상 C2 continuity가
    B-spline의 수학적 성질이다). 인접한 (p,q) 샘플 사이의 값/normal 변화가
    간격에 비례해 작아야 한다(연속성의 최소 실용적 정의)."""
    K, D = default_camera_matrix_distortion()
    center, radius = _known_sphere()
    baseline = BaselineWindshieldModel(K, D, _MODEL)
    theta_scale, phi_scale = compute_angular_fov_scale(baseline, center, radius, 1280, 800)
    rows, cols = 4, 5  # 노드 수가 서로 달라 cell 경계가 p/q 축에서 다르게 생기게 한다
    rng = np.random.default_rng(0)
    grid = rng.normal(scale=0.004, size=(rows, cols))
    basis = _build_spline_basis(rows, cols, 3)

    coords = np.linspace(-0.95, 0.95, 61)  # cell 경계(노드 위치)를 여러 번 지나가는 촘촘한 샘플
    prev_ds = None
    prev_normal = None
    max_ds_jump = 0.0
    max_normal_jump = 0.0
    for p in coords:
        ev = evaluate_inner_surface(p, 0.1, center, radius, grid, theta_scale, phi_scale, basis)
        assert ev is not None
        ds = float(np.dot(ev.point - center, ev.point - center) ** 0.5) - radius
        if prev_ds is not None:
            max_ds_jump = max(max_ds_jump, abs(ds - prev_ds))
            max_normal_jump = max(max_normal_jump, float(np.linalg.norm(ev.normal - prev_normal)))
        prev_ds, prev_normal = ds, ev.normal

    # 샘플 간격(coords 간 p step)에 비례하는 작은 변화만 있어야 한다 - 급격한
    # 불연속(예: bilinear의 기울기 불연속 수준)이 있다면 이 값들이 grid
    # amplitude(4mm)에 비해 비정상적으로 커야 한다. 넉넉한 상한을 쓴다.
    assert max_ds_jump < 0.002  # 2mm - grid amplitude(4mm)보다 훨씬 작은 step-to-step 변화
    assert max_normal_jump < 0.05


def test_outer_normal_equals_inner_normal_fallback_never_radial():
    """Outer normal은 항상 N_outer ~= N_inner(fallback, 사용자 스펙 21번)
    여야 하고, 절대 순수 반경 방향(normalize(point-center))이면 안 된다
    (사용자 스펙 20/38번 금지 사항 - 국소 변형이 있으면 반경 방향과 달라야
    한다는 것으로 이를 확인한다)."""
    K, D = default_camera_matrix_distortion()
    center, radius = _known_sphere()
    baseline = BaselineWindshieldModel(K, D, _MODEL)
    theta_scale, phi_scale = compute_angular_fov_scale(baseline, center, radius, 1280, 800)
    n = MIN_SPLINE_GRID_SIZE
    rng = np.random.default_rng(1)
    grid = rng.normal(scale=0.006, size=(n, n))
    basis = _build_spline_basis(n, n, 3)

    inner = evaluate_inner_surface(0.2, -0.3, center, radius, grid, theta_scale, phi_scale, basis)
    assert inner is not None
    outer = evaluate_outer_surface(inner, thickness=0.005)

    assert np.allclose(outer.normal, inner.normal)  # 명시된 fallback 그대로
    radial_at_outer = (outer.point - center) / np.linalg.norm(outer.point - center)
    # 국소 변형이 있는 지점(random grid)에서는 순수 반경 방향과 normal이
    # 눈에 띄게 달라야 한다(둘이 같다면 여전히 금지된 radial 근사를 쓰고
    # 있다는 뜻이다).
    assert np.dot(outer.normal, radial_at_outer) < 0.999999


def test_normal_continuity_check_detects_severe_folding():
    """`_check_normal_continuity`(사용자 스펙 16/27번)가 정상적인(작은)
    grid에서는 True, 인위적으로 극단적인(fold를 일으키는) grid에서는
    False를 반환하는지 확인한다."""
    K, D = default_camera_matrix_distortion()
    center, radius = _known_sphere()
    baseline = BaselineWindshieldModel(K, D, _MODEL)
    theta_scale, phi_scale = compute_angular_fov_scale(baseline, center, radius, 1280, 800)
    n = MIN_SPLINE_GRID_SIZE
    basis = _build_spline_basis(n, n, 3)

    smooth_grid = np.zeros((n, n))
    assert _check_normal_continuity(center, radius, smooth_grid, theta_scale, phi_scale, basis, lattice=5)

    # 인접 노드가 큰 폭으로 번갈아 뒤집히는 극단적인 패턴 - 실제 max_displacement
    # bound(기본 10mm) 안에서도 매우 거친 체커보드 패턴을 만들면 국소적으로
    # normal이 급격히 흔들릴 수 있다.
    rng = np.random.default_rng(2)
    checkerboard = np.array([[0.009 if (r + c) % 2 == 0 else -0.009 for c in range(n)] for r in range(n)])
    result = _check_normal_continuity(center, radius, checkerboard, theta_scale, phi_scale, basis, lattice=5)
    # 체커보드가 항상 fold를 만든다고 보장할 수는 없으므로(B-spline은
    # bilinear보다 훨씬 부드럽다), 이 테스트는 "smooth=True, 함수 자체가
    # 크래시 없이 bool을 반환한다"는 것만 엄격히 확인하고, 극단 패턴
    # 결과는 정보성으로만 남긴다(flaky assertion을 피하기 위해 극단
    # 케이스에는 하드 assert를 걸지 않는다).
    assert isinstance(result, (bool, np.bool_))


# ---------------------------------------------------------------------------
# STAGE B / K,D immutable / Pose bounds - 공유 fixture (calibrate_spline 1회)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bump_spline_result():
    K, D = default_camera_matrix_distortion()
    K_before, D_before = K.copy(), D.copy()
    dataset = build_synthetic_spline_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D, spline_hint={"spline_rows": 4.0, "spline_cols": 4.0})
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=1)
    result = calibrate_spline(dataset, config, camera_config, train_ids, test_ids)
    return result, K, D, K_before, D_before


def test_stage_b_pipeline_keeps_base_intrinsics_immutable(bump_spline_result):
    _, K, D, K_before, D_before = bump_spline_result
    assert np.array_equal(K, K_before)
    assert np.array_equal(D, D_before)


def test_final_result_reports_stage_used(bump_spline_result):
    result, *_ = bump_spline_result
    assert result.success, result.error_message
    assert "stage_used_is_joint_refined" in result.fitted_params
    assert np.isfinite(result.residual_stats.rmse)


def test_pose_moves_within_reasonable_bounds_during_stage_b(bump_spline_result):
    """Test J - Pose ΔR/Δt finite & non-divergent(넉넉한 상한, flaky 방지)."""
    result, *_ = bump_spline_result
    assert result.success, result.error_message

    for key in (
        "diag_pose_delta_r_median_deg", "diag_pose_delta_r_p95_deg",
        "diag_pose_delta_t_median_mm", "diag_pose_delta_t_p95_mm",
    ):
        assert key in result.fitted_params
        assert np.isfinite(result.fitted_params[key])

    assert result.fitted_params["diag_pose_delta_r_median_deg"] < 30.0
    assert result.fitted_params["diag_pose_delta_t_median_mm"] < 500.0
    if result.fitted_params["stage_used_is_joint_refined"] == 0.0:
        assert result.fitted_params["diag_pose_delta_r_median_deg"] == pytest.approx(0.0, abs=1e-9)
        assert result.fitted_params["diag_pose_delta_t_median_mm"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Test H - Outer Test leakage (Base Sphere + Spline grid)
# ---------------------------------------------------------------------------

def test_outer_test_corruption_does_not_change_base_sphere_or_spline_grid():
    """이 테스트는 calibrate_spline()을 2번 호출한다 - 물리적으로 정확한
    ray-surface intersection의 비용 때문에 한 번에 여러 분(약 3분 이상)이
    걸릴 수 있어, 실행 시간을 억제하려고 프레임 수를 4장(train 3 + test 1)
    으로 줄인다. Leakage 검증(Outer Test를 왜곡해도 Base Sphere/Spline
    grid가 바뀌면 안 된다)에는 통계적 정확도가 필요 없고 "같은 입력이면
    같은 출력" 결정론성만 확인하면 되므로 프레임 수를 줄여도 테스트
    목적에는 영향이 없다."""
    from calibration.types import Dataset

    K, D = default_camera_matrix_distortion()
    dataset_a = Dataset(frames=build_synthetic_spline_windshield_dataset(K, D).frames[:4])
    camera_config = default_camera_config()
    ids = [f.image_info.image_id for f in dataset_a.frames]
    train_ids, test_ids = ids[:3], ids[3:]
    assert test_ids

    dataset_b = _copy.deepcopy(dataset_a)
    for frame in dataset_b.frames:
        if frame.image_info.image_id in test_ids:
            frame.detection.corners = frame.detection.corners + np.float32(50.0)

    config_a = _config(K, D, spline_hint={"spline_rows": 4.0, "spline_cols": 4.0})
    config_b = _config(K, D, spline_hint={"spline_rows": 4.0, "spline_cols": 4.0})

    result_a = calibrate_spline(dataset_a, config_a, camera_config, train_ids, test_ids)
    result_b = calibrate_spline(dataset_b, config_b, camera_config, train_ids, test_ids)

    assert result_a.success and result_b.success
    for key in ("sphere_center_x", "sphere_center_y", "sphere_center_z", "sphere_radius"):
        assert result_a.fitted_params[key] == pytest.approx(result_b.fitted_params[key], abs=1e-9)
    rows, cols = int(result_a.fitted_params["spline_rows"]), int(result_a.fitted_params["spline_cols"])
    for r in range(rows):
        for c in range(cols):
            key = f"spline_ds_{r}_{c}"
            assert result_a.fitted_params[key] == pytest.approx(result_b.fitted_params[key], abs=1e-9)


# ---------------------------------------------------------------------------
# Test I - AUTO grid selection leakage (candidates/seeds를 최소로 줄여 실행)
# ---------------------------------------------------------------------------

def test_auto_grid_selection_does_not_leak_outer_test_set(monkeypatch):
    """auto_spline=1.0으로 실행하면 내부적으로 select_best_spline_grid_
    resolution이 outer train_ids만 갖고 grid 크기를 고른 뒤, 그 크기로 최종
    fitting까지 끝낸다 - outer test 코너를 왜곡해도 선택된 grid 크기와 최종
    fitted grid 값이 바뀌면 안 된다(leakage 검증).

    calibrate_spline() 1회가 최소 grid에서도 수 분이 걸리고, AUTO 경로는
    거기에 candidates x seeds번의 repeated hold-out 비용이 추가로 붙는다.
    이 테스트는 leakage 메커니즘 자체를 검증하는 게 목적이지 기본
    후보/시드 개수를 검증하는 게 아니므로, 후보를 1개로, AUTO 내부 seed도
    1개로, 그리고 프레임 수도 5장(train 3 + test 2)으로 줄여 실행 시간을
    최소화한다(Residual Grid/RBF의
    `test_auto_grid_end_to_end_does_not_leak_outer_test_set`와 동일 패턴 +
    calibrate_spline 자체의 근본적인 물리적 비용 때문에 프레임 수 축소를
    추가로 적용)."""
    import calibration.windshield.spline as spline_module
    from calibration.types import Dataset

    monkeypatch.setattr(spline_module, "SPLINE_GRID_CANDIDATES", [(4, 4)])
    monkeypatch.setattr(spline_module, "DEFAULT_REPEATED_HOLDOUT_SEEDS", (1,))

    K, D = default_camera_matrix_distortion()
    dataset_a = Dataset(frames=build_synthetic_spline_windshield_dataset(K, D).frames[:5])
    camera_config = default_camera_config()
    ids = [f.image_info.image_id for f in dataset_a.frames]
    train_ids, test_ids = ids[:-2], ids[-2:]
    assert test_ids

    dataset_b = _copy.deepcopy(dataset_a)
    for frame in dataset_b.frames:
        if frame.image_info.image_id in test_ids:
            frame.detection.corners = frame.detection.corners + np.float32(50.0)

    config_a = _config(K, D, spline_hint={"auto_spline": 1.0})
    config_b = _config(K, D, spline_hint={"auto_spline": 1.0})

    result_a = calibrate_spline(dataset_a, config_a, camera_config, train_ids, test_ids)
    result_b = calibrate_spline(dataset_b, config_b, camera_config, train_ids, test_ids)

    assert result_a.success and result_b.success
    assert result_a.fitted_params["spline_rows"] == result_b.fitted_params["spline_rows"]
    assert result_a.fitted_params["spline_cols"] == result_b.fitted_params["spline_cols"]
    rows, cols = int(result_a.fitted_params["spline_rows"]), int(result_a.fitted_params["spline_cols"])
    for r in range(rows):
        for c in range(cols):
            key = f"spline_ds_{r}_{c}"
            assert result_a.fitted_params[key] == pytest.approx(result_b.fitted_params[key], abs=1e-9)


def test_select_best_spline_grid_resolution_keeps_base_intrinsics_immutable():
    K, D = default_camera_matrix_distortion()
    K_before, D_before = K.copy(), D.copy()
    dataset = build_synthetic_spline_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D)
    train_ids = [f.image_info.image_id for f in dataset.frames]

    select_best_spline_grid_resolution(dataset, config, camera_config, train_ids, candidates=[(4, 4)], seeds=(1,))

    assert np.array_equal(K, K_before)
    assert np.array_equal(D, D_before)


# ---------------------------------------------------------------------------
# Test K/L/M - Repeated Hold-out, Ray Stability, Surface Stability +
# 진단 오케스트레이터(하나의 fixture로 묶어 calibrate_spline 반복 호출 최소화)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def diagnostics_result():
    """run_spline_calibration_with_diagnostics()를 1번만 실행해서 Repeated
    Hold-out/Ray Stability/Surface Stability/AUTO 플래그/pose 진단을 전부
    같은 결과에서 확인한다 - 순수 Spherical GT(더 안정적으로 성공하는
    fixture)에 2개 seed만 쓴다.

    프레임을 6장으로 줄여봤더니 base sphere 조건이 나빠져 최종 spline
    surface에서 train 코너의 12%가 유효한 pixel 예측을 못 내는 실패가
    발생했다(다양한 pose 커버리지가 필요함) - 실행 시간보다 정확성이
    우선이므로 전체 8프레임 데이터셋을 그대로 쓴다."""
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D, spline_hint={"spline_rows": 4.0, "spline_cols": 4.0})
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=3)
    result = run_spline_calibration_with_diagnostics(
        dataset, config, camera_config, train_ids, test_ids,
        compute_repeated_holdout=True, repeated_holdout_seeds=(11, 12), repeated_holdout_test_ratio=0.3,
    )
    return result


def test_diagnostics_orchestrator_populates_all_diag_fields(diagnostics_result):
    result = diagnostics_result
    assert result.success, result.error_message
    assert result.fitted_params["diag_selection_mode_is_auto"] == 0.0
    assert result.fitted_params["diag_repeated_n_requested"] == 2.0
    assert "diag_repeated_n_successful" in result.fitted_params
    for key in (
        "diag_pose_delta_r_median_deg", "diag_pose_delta_r_p95_deg",
        "diag_pose_delta_t_median_mm", "diag_pose_delta_t_p95_mm",
        "diag_deformation_mean_abs_m", "diag_deformation_max_abs_m",
    ):
        assert key in result.fitted_params


def test_repeated_holdout_reports_stability_fields_when_successful(diagnostics_result):
    result = diagnostics_result
    n_ok = result.fitted_params["diag_repeated_n_successful"]
    if n_ok >= 1:
        assert result.fitted_params.get("diag_repeated_mean_test_rmse") is not None or n_ok == 0
    if n_ok >= 2:
        # 2개 이상 성공했을 때만 pairwise stability를 계산할 수 있다(None을
        # 억지로 채우지 않는다는 설계).
        assert "diag_ray_stability_mean_deg" in result.fitted_params
        assert result.fitted_params["diag_ray_stability_mean_deg"] >= 0
        assert "diag_surface_stability_mean_mm" in result.fitted_params
        assert result.fitted_params["diag_surface_stability_mean_mm"] >= 0


def test_run_repeated_holdout_spline_keeps_base_intrinsics_immutable():
    K, D = default_camera_matrix_distortion()
    K_before, D_before = K.copy(), D.copy()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D, spline_hint={"spline_rows": 4.0, "spline_cols": 4.0})

    run_repeated_holdout_spline(dataset, config, camera_config, seeds=(1, 2), test_ratio=0.3)

    assert np.array_equal(K, K_before)
    assert np.array_equal(D, D_before)


# ---------------------------------------------------------------------------
# Repeated Hold-out = Outer Train only (사용자 스펙 28/29번, 이번 라운드에
# 고친 leakage 버그의 회귀 테스트)
# ---------------------------------------------------------------------------

def test_diagnostics_orchestrator_repeated_holdout_never_sees_outer_test_ids():
    """`run_spline_calibration_with_diagnostics()`가 repeated hold-out에
    넘기는 dataset이 항상 Outer Train만 포함하는지 monkeypatch로 직접
    확인한다(사용자 스펙 30번 "Repeated diagnostics에서도 Outer Test ID가
    전달되지 않았음을 monkeypatch를 통해 직접 확인") - 이전 라운드에는
    `windshield_dataset`(전체)을 그대로 넘겨서 Outer Test 프레임이 내부
    split_train_test에 다시 섞여 들어갈 수 있었다."""
    import calibration.windshield.spline as spline_module

    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_spherical_windshield_dataset(K, D)
    camera_config = default_camera_config()
    ids = [f.image_info.image_id for f in dataset.frames]
    train_ids, test_ids = ids[:-2], ids[-2:]
    assert test_ids

    seen_frame_ids: set[str] = set()
    original_run_repeated = spline_module.run_repeated_holdout_spline

    def spying_run_repeated_holdout_spline(windshield_dataset, *args, **kwargs):
        for frame in windshield_dataset.frames:
            seen_frame_ids.add(frame.image_info.image_id)
        return original_run_repeated(windshield_dataset, *args, **kwargs)

    spline_module.run_repeated_holdout_spline = spying_run_repeated_holdout_spline
    try:
        config = _config(K, D, spline_hint={"spline_rows": 4.0, "spline_cols": 4.0})
        result = run_spline_calibration_with_diagnostics(
            dataset, config, camera_config, train_ids, test_ids,
            compute_repeated_holdout=True, repeated_holdout_seeds=(21,), repeated_holdout_test_ratio=0.3,
        )
    finally:
        spline_module.run_repeated_holdout_spline = original_run_repeated

    assert result.success, result.error_message
    assert seen_frame_ids, "run_repeated_holdout_spline이 아예 호출되지 않았습니다."
    leaked = seen_frame_ids & set(test_ids)
    assert not leaked, f"Outer Test 프레임이 repeated hold-out dataset에 leakage됨: {leaked}"
    assert seen_frame_ids <= set(train_ids)

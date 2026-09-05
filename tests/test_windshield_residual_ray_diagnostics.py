"""
tests/test_windshield_residual_ray_diagnostics.py
========================================================

Residual Ray Grid STEP 3-A "실사용 마무리" 라운드 검증:
  - run_residual_ray_calibration_with_diagnostics (사용자 스펙 2번 - UI가
    호출할 단일 진단 orchestrator, backend가 diagnostics를 채워 넣는다)
  - K,D immutability, outer-test leakage가 새 orchestrator를 거쳐도 여전히
    지켜지는지(사용자 스펙 5/6번)

compute_ray_stability_deg 자체의 단위 테스트는 STEP 3-B에서
tests/test_windshield_residual_common.py로 옮겼다 - 그 함수가 raw grid
배열이 아니라 WindshieldModel 인스턴스 리스트를 받는 형태로 일반화되어
Grid/RBF 양쪽이 공유하게 됐기 때문이다.
"""

from __future__ import annotations

import copy as _copy
import time

import numpy as np
import pytest

from calibration.types import CameraModelType
from calibration.validation import split_train_test
from calibration.windshield.base import WindshieldConfig, WindshieldModelType
from calibration.windshield.residual_ray import run_residual_ray_calibration_with_diagnostics
from tests._windshield_test_utils import (
    build_synthetic_residual_ray_dataset,
    default_camera_config,
    default_camera_matrix_distortion,
    default_residual_delta_fn,
)

_MODEL = CameraModelType.BROWN_CONRADY


def _config(K, D, **kwargs) -> WindshieldConfig:
    return WindshieldConfig(
        base_model_name=_MODEL, base_camera_matrix=K, base_distortion=D,
        windshield_model=WindshieldModelType.RESIDUAL_RAY, **kwargs,
    )


# ---------------------------------------------------------------------------
# run_residual_ray_calibration_with_diagnostics
# ---------------------------------------------------------------------------

def test_diagnostics_orchestrator_populates_all_diag_fields_manual_grid():
    K, D = default_camera_matrix_distortion()
    delta_fn = default_residual_delta_fn(K)
    dataset = build_synthetic_residual_ray_dataset(K, D, delta_fn)
    camera_config = default_camera_config()
    config = _config(K, D, residual_ray_hint={"grid_rows": 3.0, "grid_cols": 4.0, "auto_grid": 0.0})
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=3)

    start = time.monotonic()
    result = run_residual_ray_calibration_with_diagnostics(
        dataset, config, camera_config, train_ids, test_ids,
        compute_repeated_holdout=True, repeated_holdout_seeds=(11, 12, 13), repeated_holdout_test_ratio=0.3,
    )
    elapsed = time.monotonic() - start

    assert result.success, result.error_message
    assert result.fitted_params["diag_selection_mode_is_auto"] == 0.0
    assert "diag_repeated_n_requested" in result.fitted_params
    assert result.fitted_params["diag_repeated_n_requested"] == 3.0
    assert "diag_repeated_n_successful" in result.fitted_params
    assert result.fitted_params["diag_repeated_n_successful"] >= 0.0
    # 3개의 diagnostic seed만 요청했으므로(내부적으로 calibrate_residual_ray를
    # 그 개수만큼만 호출), 지수적으로 커지지 않고 넉넉한 시간 안에 끝나야 한다.
    assert elapsed < 120.0

    for key in (
        "diag_pose_delta_r_median_deg", "diag_pose_delta_r_p95_deg",
        "diag_pose_delta_t_median_mm", "diag_pose_delta_t_p95_mm",
    ):
        assert key in result.fitted_params


def test_diagnostics_orchestrator_auto_grid_uses_resolved_size_for_holdout():
    """AUTO 모드로 실행해도 repeated hold-out 내부에서는 AUTO가 다시
    실행되지 않고, 본 계산이 고른 grid_rows/grid_cols로 고정돼야 한다(안 그러면
    비용 폭발 + split마다 다른 해상도로 stability를 비교하게 되는 문제)."""
    import calibration.windshield.residual_ray as residual_ray_module

    K, D = default_camera_matrix_distortion()
    delta_fn = default_residual_delta_fn(K)
    dataset = build_synthetic_residual_ray_dataset(K, D, delta_fn)
    camera_config = default_camera_config()
    config = _config(K, D, residual_ray_hint={"auto_grid": 1.0})
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=3)

    original_select = residual_ray_module.select_best_grid_resolution
    call_count = {"n": 0}

    def counting_select(*args, **kwargs):
        call_count["n"] += 1
        return original_select(*args, **kwargs)

    residual_ray_module.select_best_grid_resolution = counting_select
    try:
        result = run_residual_ray_calibration_with_diagnostics(
            dataset, config, camera_config, train_ids, test_ids,
            compute_repeated_holdout=True, repeated_holdout_seeds=(21, 22), repeated_holdout_test_ratio=0.3,
        )
    finally:
        residual_ray_module.select_best_grid_resolution = original_select

    assert result.success, result.error_message
    assert result.fitted_params["diag_selection_mode_is_auto"] == 1.0
    # AUTO 선택은 본 계산(calibrate_residual_ray 최초 1회) 안에서만 일어나야
    # 하고, 그 뒤 repeated hold-out의 각 seed에서는 다시 호출되면 안 된다.
    assert call_count["n"] == 1


def test_diagnostics_orchestrator_keeps_base_intrinsics_immutable():
    K, D = default_camera_matrix_distortion()
    K_before, D_before = K.copy(), D.copy()
    delta_fn = default_residual_delta_fn(K)
    dataset = build_synthetic_residual_ray_dataset(K, D, delta_fn)
    camera_config = default_camera_config()
    config = _config(K, D, residual_ray_hint={"grid_rows": 3.0, "grid_cols": 4.0, "auto_grid": 0.0})
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=3)

    run_residual_ray_calibration_with_diagnostics(
        dataset, config, camera_config, train_ids, test_ids,
        compute_repeated_holdout=True, repeated_holdout_seeds=(31, 32), repeated_holdout_test_ratio=0.3,
    )

    assert np.array_equal(K, K_before)
    assert np.array_equal(D, D_before)


def test_diagnostics_orchestrator_does_not_leak_outer_test_set():
    """AUTO grid를 orchestrator 경유로 실행해도, outer test set을 오염시켜도
    선택된 grid 크기와 최종 fitted grid 값이 바뀌면 안 된다(기존 leakage
    테스트를 orchestrator까지 확장)."""
    import calibration.windshield.residual_ray as residual_ray_module

    residual_ray_module.DEFAULT_GRID_CANDIDATES_BACKUP = getattr(
        residual_ray_module, "DEFAULT_GRID_CANDIDATES_BACKUP", None
    )
    original_candidates = residual_ray_module.DEFAULT_GRID_CANDIDATES
    residual_ray_module.DEFAULT_GRID_CANDIDATES = [(3, 4), (4, 6)]
    try:
        K, D = default_camera_matrix_distortion()
        delta_fn = default_residual_delta_fn(K)
        dataset_a = build_synthetic_residual_ray_dataset(K, D, delta_fn)
        camera_config = default_camera_config()
        ids = [f.image_info.image_id for f in dataset_a.frames]
        train_ids, test_ids = ids[:-2], ids[-2:]
        assert test_ids

        dataset_b = _copy.deepcopy(dataset_a)
        for frame in dataset_b.frames:
            if frame.image_info.image_id in test_ids:
                frame.detection.corners = frame.detection.corners + np.float32(50.0)

        config_a = _config(K, D, residual_ray_hint={"auto_grid": 1.0})
        config_b = _config(K, D, residual_ray_hint={"auto_grid": 1.0})

        result_a = run_residual_ray_calibration_with_diagnostics(
            dataset_a, config_a, camera_config, train_ids, test_ids,
            compute_repeated_holdout=False,
        )
        result_b = run_residual_ray_calibration_with_diagnostics(
            dataset_b, config_b, camera_config, train_ids, test_ids,
            compute_repeated_holdout=False,
        )

        assert result_a.success and result_b.success
        assert result_a.fitted_params["grid_rows"] == result_b.fitted_params["grid_rows"]
        assert result_a.fitted_params["grid_cols"] == result_b.fitted_params["grid_cols"]
        rows, cols = int(result_a.fitted_params["grid_rows"]), int(result_a.fitted_params["grid_cols"])
        for r in range(rows):
            for c in range(cols):
                for axis in ("dx", "dy", "dz"):
                    key = f"grid_{axis}_{r}_{c}"
                    assert result_a.fitted_params[key] == pytest.approx(result_b.fitted_params[key], abs=1e-9)
    finally:
        residual_ray_module.DEFAULT_GRID_CANDIDATES = original_candidates

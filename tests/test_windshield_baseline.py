"""
tests/test_windshield_baseline.py
======================================

Phase 1(Baseline) 핵심 검증 - 사용자 스펙 23번 "Baseline" 테스트 항목:
  * 기존 K,D loading
  * K,D immutable / fixed 확인 (test_windshield_base_projection.py에서 별도 검증)
  * identical projection에서 residual ≈ 0 확인
  * 요구된 통계(RMS/Median/P95/P99/Max/Regional/Radial/Spatial) 전부 채워지는지
  * Train/Test 분할에 leakage가 없는지
"""

from __future__ import annotations

import numpy as np
import pytest

from calibration.types import CameraModelType
from calibration.validation import split_train_test
from calibration.windshield.base import WindshieldConfig
from calibration.windshield.baseline import calibrate_baseline
from tests._windshield_test_utils import (
    build_synthetic_windshield_dataset,
    default_camera_config,
    default_camera_matrix_distortion,
)


def _config(K: np.ndarray, D: np.ndarray) -> WindshieldConfig:
    return WindshieldConfig(base_model_name=CameraModelType.BROWN_CONRADY, base_camera_matrix=K, base_distortion=D)


def test_baseline_residual_near_zero_for_identical_projection():
    """Base K,D가 실제로 코너를 생성한 K,D와 정확히 같으면(Windshield 영향
    없음을 흉내낸 상황), Baseline residual은 부동소수점 오차 수준이어야 한다."""
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D)
    train_ids = [f.image_info.image_id for f in dataset.frames]

    result = calibrate_baseline(dataset, config, camera_config, train_ids, [])

    assert result.success
    assert result.residual_stats is not None
    assert result.residual_stats.rmse < 1e-3
    assert result.fitted_params == {}


def test_baseline_populates_all_required_statistics():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D)
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=1)

    result = calibrate_baseline(dataset, config, camera_config, train_ids, test_ids)

    assert result.success
    stats = result.residual_stats
    assert stats is not None
    for attr in ("rmse", "median", "p95", "p99", "max", "mae", "std"):
        assert getattr(stats, attr) is not None
    assert result.test_residual_stats is not None
    assert result.regional_error is not None
    assert result.radial_profile is not None and result.radial_profile.bins
    assert result.radial_bands is not None and result.radial_bands.bins
    assert result.spatial_error_map is not None
    assert any(c.num_points > 0 for c in result.spatial_error_map.cells)
    assert result.mean_dx is not None
    assert result.mean_dy is not None


def test_baseline_detects_systematic_nonrigid_distortion():
    """균일한 픽셀 평행이동은 프레임별 solvePnP가 거의 다 흡수해버리므로,
    windshield 굴절처럼 "단일 rvec/tvec로는 설명 안 되는" 비강체(non-rigid)
    패턴을 흉내내야 Baseline residual이 뚜렷하게 남는다 - 새들(saddle) 형태의
    변위(dy += k*(x-cx)*(y-cy))를 코너에 주입한다."""
    K, D = default_camera_matrix_distortion()
    camera_config = default_camera_config()
    config = _config(K, D)

    baseline_ds = build_synthetic_windshield_dataset(K, D)
    train_ids = [f.image_info.image_id for f in baseline_ds.frames]
    baseline_result = calibrate_baseline(baseline_ds, config, camera_config, train_ids, [])

    distorted_ds = build_synthetic_windshield_dataset(K, D, shear_k=0.02)
    distorted_result = calibrate_baseline(distorted_ds, config, camera_config, train_ids, [])

    assert distorted_result.success
    assert distorted_result.residual_stats.rmse > 100 * baseline_result.residual_stats.rmse
    assert distorted_result.residual_stats.rmse > 0.1


def test_baseline_train_test_split_has_no_leakage():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D)

    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=7)
    assert train_ids and test_ids
    assert set(train_ids).isdisjoint(test_ids)

    result = calibrate_baseline(dataset, config, camera_config, train_ids, test_ids)

    assert result.train_frame_ids == train_ids
    assert result.test_frame_ids == test_ids
    # Train residual_stats는 train_ids만으로, Test residual_stats는 test_ids만으로
    # 계산됐어야 한다 - 둘 다 채워져 있고 n이 0이 아니어야 실제로 각자의 데이터로
    # 계산됐다는 근거가 된다.
    assert result.residual_stats.n > 0
    assert result.test_residual_stats.n > 0


def test_baseline_fails_gracefully_with_no_train_frames():
    K, D = default_camera_matrix_distortion()
    dataset = build_synthetic_windshield_dataset(K, D)
    camera_config = default_camera_config()
    config = _config(K, D)

    result = calibrate_baseline(dataset, config, camera_config, [], [])

    assert result.success is False
    assert result.error_message

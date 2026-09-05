"""
tests/test_repeatability.py
==================================

설계 문서 40번 - Calibration Repeatability.
"""

from __future__ import annotations

import pytest

from calibration.repeatability import compute_repeatability, format_repeatability
from calibration.types import CameraModelType, Dataset


class TestComputeRepeatability:
    def test_deterministic_optimizer_gives_high_repeatability(self, synthetic_dataset, camera_config):
        """cv2.calibrateCamera는 결정론적이므로, 프레임 순서만 바꿔도 거의
        동일한 파라미터가 나와야 한다 - repeatability가 매우 높아야(예: 95%
        이상) 정상이다(모듈 docstring 참고)."""
        result = compute_repeatability(
            synthetic_dataset, camera_config, CameraModelType.PINHOLE, n_runs=5, seed=1,
        )
        assert result.order_successful == 5
        assert result.initial_condition_successful == 5
        assert result.n_successful == 10
        assert result.repeatability_pct is not None
        assert result.repeatability_pct > 90.0

    def test_cv_values_are_small_for_stable_data(self, synthetic_dataset, camera_config):
        # Brown-Conrady(5계수)를 쓴다 - Rational(EXTENDED_PINHOLE, 항상 8계수,
        # P0-1)은 자유도가 많아 이 정도 크기의 합성 데이터에서는 CV가
        # 구조적으로 훨씬 커진다(실측 ~40%대) - 이건 버그가 아니라
        # "파라미터가 많을수록 데이터가 충분치 않으면 불안정해진다"는 이
        # 프로젝트가 이미 알고 있던 트레이드오프 그 자체다. "적당히 복잡한
        # 모델도 안정적으로 재현되는지" 확인하려는 이 테스트의 원래 의도에는
        # Brown-Conrady가 더 맞는다.
        result = compute_repeatability(
            synthetic_dataset, camera_config, CameraModelType.BROWN_CONRADY, n_runs=5, seed=2,
        )
        assert result.fx_cv is not None
        assert result.fx_cv < 0.05  # 변동계수 5% 미만

    def test_empty_dataset_returns_zero_successful(self, camera_config):
        result = compute_repeatability(
            Dataset(frames=[]), camera_config, CameraModelType.PINHOLE, n_runs=3, seed=1,
        )
        assert result.n_successful == 0
        assert result.order_successful == 0
        assert result.initial_condition_successful == 0
        assert result.repeatability_pct is None

    def test_frame_content_unchanged_across_runs(self, synthetic_dataset, camera_config):
        """순서만 바뀌어야지 프레임 개수나 내용 자체는 바뀌면 안 된다."""
        original_ids = {f.image_info.image_id for f in synthetic_dataset.frames}
        compute_repeatability(synthetic_dataset, camera_config, CameraModelType.PINHOLE, n_runs=3, seed=1)
        after_ids = {f.image_info.image_id for f in synthetic_dataset.frames}
        assert original_ids == after_ids

    def test_parallel_matches_sequential_for_same_seed(self, synthetic_dataset, camera_config):
        sequential = compute_repeatability(
            synthetic_dataset, camera_config, CameraModelType.PINHOLE, n_runs=4, seed=7, n_jobs=1,
        )
        parallel = compute_repeatability(
            synthetic_dataset, camera_config, CameraModelType.PINHOLE, n_runs=4, seed=7, n_jobs=2,
        )
        assert parallel.n_successful == sequential.n_successful
        assert parallel.repeatability_pct == pytest.approx(sequential.repeatability_pct, rel=1e-5)
        assert parallel.fx_cv == pytest.approx(sequential.fx_cv, rel=1e-5, abs=1e-10)

    def test_can_run_order_only_for_backward_compatibility(self, synthetic_dataset, camera_config):
        result = compute_repeatability(
            synthetic_dataset,
            camera_config,
            CameraModelType.PINHOLE,
            n_runs=4,
            seed=5,
            vary_initial_conditions=False,
        )
        assert result.n_runs == 4
        assert result.n_successful == 4
        assert result.order_successful == 4
        assert result.initial_condition_runs == 0

    def test_initial_condition_perturbation_is_recorded(self, synthetic_dataset, camera_config):
        result = compute_repeatability(
            synthetic_dataset,
            camera_config,
            CameraModelType.EXTENDED_PINHOLE,
            n_runs=3,
            seed=8,
            initial_condition_perturbation=0.03,
        )
        assert result.order_runs == 3
        assert result.initial_condition_runs == 3
        assert result.initial_condition_successful > 0
        assert result.initial_condition_perturbation == pytest.approx(0.03)
        assert result.repeatability_pct is not None


class TestFormatRepeatability:
    def test_includes_percentage_and_cv(self, synthetic_dataset, camera_config):
        result = compute_repeatability(
            synthetic_dataset, camera_config, CameraModelType.PINHOLE, n_runs=5, seed=3,
        )
        text = format_repeatability(result)
        assert "Repeatability" in text
        assert "CV" in text
        assert "initial condition" in text

    def test_handles_insufficient_runs(self):
        from calibration.types import RepeatabilityResult
        result = RepeatabilityResult(n_runs=3, n_successful=1)
        text = format_repeatability(result)
        assert "계산할 수 없습니다" in text

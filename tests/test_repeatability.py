"""
tests/test_repeatability.py
==================================

설계 문서 40번 - Calibration Repeatability.
"""

from __future__ import annotations

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
        assert result.n_successful == 5
        assert result.repeatability_pct is not None
        assert result.repeatability_pct > 90.0

    def test_cv_values_are_small_for_stable_data(self, synthetic_dataset, camera_config):
        result = compute_repeatability(
            synthetic_dataset, camera_config, CameraModelType.EXTENDED_PINHOLE, n_runs=5, seed=2,
        )
        assert result.fx_cv is not None
        assert result.fx_cv < 0.05  # 변동계수 5% 미만

    def test_empty_dataset_returns_zero_successful(self, camera_config):
        result = compute_repeatability(
            Dataset(frames=[]), camera_config, CameraModelType.PINHOLE, n_runs=3, seed=1,
        )
        assert result.n_successful == 0
        assert result.repeatability_pct is None

    def test_frame_content_unchanged_across_runs(self, synthetic_dataset, camera_config):
        """순서만 바뀌어야지 프레임 개수나 내용 자체는 바뀌면 안 된다."""
        original_ids = {f.image_info.image_id for f in synthetic_dataset.frames}
        compute_repeatability(synthetic_dataset, camera_config, CameraModelType.PINHOLE, n_runs=3, seed=1)
        after_ids = {f.image_info.image_id for f in synthetic_dataset.frames}
        assert original_ids == after_ids


class TestFormatRepeatability:
    def test_includes_percentage_and_cv(self, synthetic_dataset, camera_config):
        result = compute_repeatability(
            synthetic_dataset, camera_config, CameraModelType.PINHOLE, n_runs=5, seed=3,
        )
        text = format_repeatability(result)
        assert "Repeatability" in text
        assert "CV" in text

    def test_handles_insufficient_runs(self):
        from calibration.types import RepeatabilityResult
        result = RepeatabilityResult(n_runs=3, n_successful=1)
        text = format_repeatability(result)
        assert "계산할 수 없습니다" in text

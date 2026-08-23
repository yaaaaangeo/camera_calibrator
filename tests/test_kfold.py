"""
tests/test_kfold.py
==========================

설계 문서 18/19번 - K-Fold / Repeated K-Fold Cross Validation.
"""

from __future__ import annotations

import pytest

from calibration.kfold import (
    compute_kfold_validation,
    compute_repeated_kfold,
    format_kfold_result,
    format_repeated_kfold_result,
    split_k_folds,
)
from calibration.cache import ValidationCache
from calibration.types import CameraModelType, ResidualStats, ValidationResult


class TestSplitKFolds:
    def test_folds_do_not_overlap(self, synthetic_dataset, camera_config):
        folds = split_k_folds(synthetic_dataset, camera_config, k=5, seed=1)
        seen = set()
        for fold in folds:
            for fid in fold:
                assert fid not in seen, "같은 프레임이 여러 폴드에 들어감"
                seen.add(fid)

    def test_folds_cover_all_usable_frames(self, synthetic_dataset, camera_config):
        folds = split_k_folds(synthetic_dataset, camera_config, k=5, seed=1)
        total_in_folds = sum(len(f) for f in folds)
        usable = [
            f for f in synthetic_dataset.enabled_frames
            if f.detection and f.detection.success and f.detection.num_corners >= 4
        ]
        assert total_in_folds == len(usable)

    def test_correct_number_of_folds(self, synthetic_dataset, camera_config):
        folds = split_k_folds(synthetic_dataset, camera_config, k=4, seed=1)
        assert len(folds) == 4

    def test_folds_are_reasonably_balanced(self, synthetic_dataset, camera_config):
        folds = split_k_folds(synthetic_dataset, camera_config, k=5, seed=1)
        sizes = [len(f) for f in folds]
        assert max(sizes) - min(sizes) <= 2

    def test_empty_dataset_returns_empty_folds(self, camera_config):
        from calibration.types import Dataset
        folds = split_k_folds(Dataset(frames=[]), camera_config, k=5, seed=1)
        assert len(folds) == 5
        assert all(f == [] for f in folds)


class TestComputeKFoldValidation:
    def test_returns_stats_across_folds(self, synthetic_dataset, camera_config, pattern_config):
        result = compute_kfold_validation(
            synthetic_dataset, camera_config, pattern_config, CameraModelType.PINHOLE, k=4, seed=7,
        )
        assert result.k == 4
        assert result.n_successful_folds > 0
        assert result.mean_test_rms is not None
        assert result.min_test_rms <= result.mean_test_rms <= result.max_test_rms

    def test_each_fold_is_a_valid_holdout(self, synthetic_dataset, camera_config, pattern_config):
        result = compute_kfold_validation(
            synthetic_dataset, camera_config, pattern_config, CameraModelType.PINHOLE, k=4, seed=3,
        )
        for vr in result.fold_validation_results:
            assert set(vr.train_frame_ids).isdisjoint(set(vr.test_frame_ids))

    def test_format_no_crash(self, synthetic_dataset, camera_config, pattern_config):
        result = compute_kfold_validation(
            synthetic_dataset, camera_config, pattern_config, CameraModelType.EXTENDED_PINHOLE, k=4, seed=9,
        )
        text = format_kfold_result(result)
        assert "Fold" in text

    def test_parallel_matches_sequential(self, synthetic_dataset, camera_config, pattern_config):
        sequential = compute_kfold_validation(
            synthetic_dataset, camera_config, pattern_config, CameraModelType.PINHOLE,
            k=4, seed=11, n_jobs=1, cache=None,
        )
        parallel = compute_kfold_validation(
            synthetic_dataset, camera_config, pattern_config, CameraModelType.PINHOLE,
            k=4, seed=11, n_jobs=2, cache=None,
        )
        assert parallel.n_successful_folds == sequential.n_successful_folds
        assert parallel.mean_test_rms == pytest.approx(sequential.mean_test_rms, rel=1e-5)
        assert parallel.mean_test_p95 == pytest.approx(sequential.mean_test_p95, rel=1e-5)

    def test_cache_reuses_fold_validation(self, monkeypatch, synthetic_dataset, camera_config, pattern_config):
        import calibration.validation as validation

        calls = {"count": 0}

        def fake_validate_holdout(dataset, camera_config, pattern_config, model, train_ids, test_ids, **kwargs):
            calls["count"] += 1
            value = float(len(test_ids))
            return ValidationResult(
                train_frame_ids=list(train_ids),
                test_frame_ids=list(test_ids),
                train_rms=1.0,
                test_rms=value,
                test_residual_stats=ResidualStats(p95=value + 0.5),
                success=True,
            )

        monkeypatch.setattr(validation, "validate_holdout", fake_validate_holdout)
        cache = ValidationCache()
        first = compute_kfold_validation(
            synthetic_dataset, camera_config, pattern_config, CameraModelType.PINHOLE,
            k=4, seed=13, cache=cache,
        )
        second = compute_kfold_validation(
            synthetic_dataset, camera_config, pattern_config, CameraModelType.PINHOLE,
            k=4, seed=13, cache=cache,
        )

        assert calls["count"] == len(first.fold_validation_results)
        assert second.mean_test_rms == first.mean_test_rms


class TestComputeRepeatedKFold:
    def test_aggregates_across_repeats(self, synthetic_dataset, camera_config, pattern_config):
        result = compute_repeated_kfold(
            synthetic_dataset, camera_config, pattern_config, CameraModelType.PINHOLE,
            k=4, n_repeats=3, base_seed=1,
        )
        assert result.n_repeats == 3
        assert len(result.kfold_results) == 3
        assert result.n_successful_runs > 0
        assert result.mean_test_rms is not None

    def test_parallel_repeats_preserve_result_order(self, synthetic_dataset, camera_config, pattern_config):
        sequential = compute_repeated_kfold(
            synthetic_dataset, camera_config, pattern_config, CameraModelType.PINHOLE,
            k=4, n_repeats=2, base_seed=17, n_jobs=1, cache=None,
        )
        parallel = compute_repeated_kfold(
            synthetic_dataset, camera_config, pattern_config, CameraModelType.PINHOLE,
            k=4, n_repeats=2, base_seed=17, n_jobs=2, cache=None,
        )
        assert parallel.n_successful_runs == sequential.n_successful_runs
        assert parallel.mean_test_rms == pytest.approx(sequential.mean_test_rms, rel=1e-5)

    def test_format_no_crash(self, synthetic_dataset, camera_config, pattern_config):
        result = compute_repeated_kfold(
            synthetic_dataset, camera_config, pattern_config, CameraModelType.FISHEYE,
            k=4, n_repeats=2, base_seed=5,
        )
        text = format_repeated_kfold_result(result)
        assert "Repeated" in text

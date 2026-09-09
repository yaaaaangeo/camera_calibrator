"""
tests/test_kfold.py
==========================

설계 문서 18/19번 - K-Fold / Repeated K-Fold Cross Validation.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from calibration.kfold import (
    KFoldProgressEvent,
    compute_kfold_validation,
    compute_repeated_kfold,
    format_kfold_result,
    format_repeated_kfold_result,
    format_model_comparison_table,
    model_comparison_rows,
    run_repeated_kfold_all_models,
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


# ---------------------------------------------------------------------------
# 논문용 Multi-Metric 확장(Test Edge RMS / Test Straightness) 회귀 테스트.
# ---------------------------------------------------------------------------

class TestMultiMetricAggregation:
    def test_edge_rms_and_straightness_aggregate_only_from_test_source(
        self, synthetic_dataset, camera_config, pattern_config, monkeypatch
    ):
        """edge_rms는 모든 성공 fold에서 그대로 평균에 들어가야 하고,
        straightness는 straightness_source == "test"인 fold만 들어가야 한다 -
        "train_fallback"과, straightness 자체가 None(계산 불가)인 fold는
        aggregate에서 제외되지만 KFoldResult 전체를 실패시키지는 않는다."""
        import calibration.validation as validation_module

        cycle = [
            (0.40, 0.05, "test"),
            (0.44, 0.09, "train_fallback"),  # straightness aggregate에서 제외돼야 함
            (0.38, 0.06, "test"),
            (0.42, None, None),  # straightness 계산 자체가 안 된 fold - missing
        ]
        assigned: list[tuple[float, float | None, str | None]] = []

        def fake_validate_holdout(dataset, camera_config, pattern_config, model, train_ids, test_ids, **kwargs):
            edge, straight, source = cycle[len(assigned) % len(cycle)]
            assigned.append((edge, straight, source))
            return ValidationResult(
                train_frame_ids=list(train_ids), test_frame_ids=list(test_ids),
                train_rms=1.0, test_rms=1.0,
                edge_rms=edge,
                straightness_residual=straight,
                straightness_source=source,
                test_residual_stats=ResidualStats(p95=1.5),
                success=True,
            )

        monkeypatch.setattr(validation_module, "validate_holdout", fake_validate_holdout)
        result = compute_kfold_validation(
            synthetic_dataset, camera_config, pattern_config, CameraModelType.PINHOLE,
            k=4, seed=1, n_jobs=1, cache=None,
        )

        assert assigned, "적어도 하나의 fold는 실행됐어야 한다"
        expected_edges = [e for e, _s, _src in assigned]
        expected_straight = [s for _e, s, src in assigned if src == "test" and s is not None]

        assert result.n_successful_folds == len(assigned)
        assert result.mean_edge_rms == pytest.approx(float(np.mean(expected_edges)))
        assert result.min_edge_rms == pytest.approx(min(expected_edges))
        assert result.max_edge_rms == pytest.approx(max(expected_edges))

        assert result.n_straightness_folds == len(expected_straight)
        if expected_straight:
            assert result.mean_test_straightness == pytest.approx(float(np.mean(expected_straight)))
        else:
            assert result.mean_test_straightness is None

    def test_missing_edge_and_straightness_in_all_folds_does_not_fail_kfold(
        self, synthetic_dataset, camera_config, pattern_config, monkeypatch
    ):
        """edge_rms/straightness가 모든 fold에서 missing이어도(계산 자체가 안 된
        경우) KFoldResult 전체가 실패로 처리되면 안 된다 - test_rms 기반
        n_successful_folds는 정상적으로 채워지고, missing metric은 그냥
        None(임의의 숫자나 worst-penalty로 채우지 않음)이어야 한다."""
        import calibration.validation as validation_module

        def fake_validate_holdout(dataset, camera_config, pattern_config, model, train_ids, test_ids, **kwargs):
            return ValidationResult(
                train_frame_ids=list(train_ids), test_frame_ids=list(test_ids),
                train_rms=1.0, test_rms=1.0, edge_rms=None,
                straightness_residual=None, straightness_source=None,
                test_residual_stats=ResidualStats(p95=1.5),
                success=True,
            )

        monkeypatch.setattr(validation_module, "validate_holdout", fake_validate_holdout)
        result = compute_kfold_validation(
            synthetic_dataset, camera_config, pattern_config, CameraModelType.PINHOLE,
            k=4, seed=1, n_jobs=1, cache=None,
        )
        assert result.n_successful_folds > 0
        assert result.mean_test_rms is not None
        assert result.mean_edge_rms is None
        assert result.n_straightness_folds == 0
        assert result.mean_test_straightness is None

    def test_repeated_kfold_aggregates_match_expected_statistics(
        self, synthetic_dataset, camera_config, pattern_config, monkeypatch
    ):
        """Repeated K-Fold의 mean/std가 monkeypatch로 완전히 통제한 synthetic
        fold 값들의 실제 numpy 통계와 정확히 일치해야 한다 - RMS/P95/Edge/
        Straightness 전부."""
        import calibration.validation as validation_module

        values = [0.30, 0.32, 0.28, 0.35, 0.31, 0.29, 0.33, 0.34, 0.27, 0.36, 0.30, 0.31]
        cursor = {"i": 0}
        assigned_rms: list[float] = []

        def fake_validate_holdout(dataset, camera_config, pattern_config, model, train_ids, test_ids, **kwargs):
            val = values[cursor["i"] % len(values)]
            cursor["i"] += 1
            assigned_rms.append(val)
            return ValidationResult(
                train_frame_ids=list(train_ids), test_frame_ids=list(test_ids),
                train_rms=1.0, test_rms=val,
                edge_rms=val + 0.1,
                straightness_residual=val * 0.1,
                straightness_source="test",
                test_residual_stats=ResidualStats(p95=val + 0.2),
                success=True,
            )

        monkeypatch.setattr(validation_module, "validate_holdout", fake_validate_holdout)
        result = compute_repeated_kfold(
            synthetic_dataset, camera_config, pattern_config, CameraModelType.PINHOLE,
            k=4, n_repeats=3, base_seed=1, n_jobs=1, cache=None,
        )

        assert len(assigned_rms) == result.n_successful_runs
        assert result.mean_test_rms == pytest.approx(float(np.mean(assigned_rms)))
        assert result.std_test_rms == pytest.approx(float(np.std(assigned_rms, ddof=1)))

        expected_edges = [v + 0.1 for v in assigned_rms]
        assert result.mean_edge_rms == pytest.approx(float(np.mean(expected_edges)))
        assert result.std_edge_rms == pytest.approx(float(np.std(expected_edges, ddof=1)))

        expected_straight = [v * 0.1 for v in assigned_rms]
        assert result.n_straightness_folds == len(assigned_rms)
        assert result.mean_test_straightness == pytest.approx(float(np.mean(expected_straight)))
        assert result.std_test_straightness == pytest.approx(float(np.std(expected_straight, ddof=1)))

        # 논문용 fold 상태 diagnostic - 전부 성공했으니 fully_successful == total.
        assert result.total_folds == 4 * 3
        assert result.fully_successful_folds == len(assigned_rms)
        assert result.partial_success_folds == 0
        assert result.failed_folds == result.total_folds - len(assigned_rms)

    def test_each_usable_frame_is_test_exactly_once_within_one_kfold(
        self, synthetic_dataset, camera_config, pattern_config
    ):
        """단일 K-Fold 안에서 각 프레임은 정확히 한 번만 test로 쓰여야 한다
        (겹치지 않는다는 것만이 아니라 "정확히 1회"까지 확인)."""
        result = compute_kfold_validation(
            synthetic_dataset, camera_config, pattern_config, CameraModelType.PINHOLE,
            k=4, seed=9, n_jobs=1, cache=None,
        )
        counts: Counter = Counter()
        for vr in result.fold_validation_results:
            counts.update(vr.test_frame_ids)
        assert counts, "적어도 하나의 fold가 test_frame_ids를 가져야 한다"
        assert all(c == 1 for c in counts.values()), f"frame이 두 번 이상 test된 경우가 있다: {counts}"

    def test_train_test_frame_overlap_is_zero_per_fold(self, synthetic_dataset, camera_config, pattern_config):
        result = compute_kfold_validation(
            synthetic_dataset, camera_config, pattern_config, CameraModelType.PINHOLE,
            k=4, seed=9, n_jobs=1, cache=None,
        )
        for vr in result.fold_validation_results:
            assert set(vr.train_frame_ids).isdisjoint(set(vr.test_frame_ids))


# ---------------------------------------------------------------------------
# Brown-Conrady/Rational/Fisheye가 정확히 같은 fold partition을 공유하는지
# (특정 모델이 유리한 fold만 받는 일이 없도록).
# ---------------------------------------------------------------------------

class TestSameFoldPartitionAcrossModels:
    def _test_id_partition(self, kfold_result) -> list[tuple[str, ...]]:
        return [tuple(sorted(vr.test_frame_ids)) for vr in kfold_result.fold_validation_results]

    def test_kfold_partition_identical_across_brown_rational_fisheye(
        self, synthetic_dataset, camera_config, pattern_config
    ):
        models = [CameraModelType.BROWN_CONRADY, CameraModelType.EXTENDED_PINHOLE, CameraModelType.FISHEYE]
        partitions = {
            m: self._test_id_partition(
                compute_kfold_validation(
                    synthetic_dataset, camera_config, pattern_config, m, k=3, seed=5, n_jobs=1, cache=None,
                )
            )
            for m in models
        }
        brown_partition = partitions[CameraModelType.BROWN_CONRADY]
        for m, p in partitions.items():
            assert p == brown_partition, f"{m}의 fold partition이 Brown-Conrady와 다르다"

    def test_run_repeated_kfold_all_models_shares_fold_partition(
        self, synthetic_dataset, camera_config, pattern_config
    ):
        results = run_repeated_kfold_all_models(
            synthetic_dataset, camera_config, pattern_config,
            k=3, n_repeats=2, base_seed=7, n_jobs=1, cache=None,
        )
        assert set(results.keys()) == {
            CameraModelType.BROWN_CONRADY, CameraModelType.EXTENDED_PINHOLE, CameraModelType.FISHEYE,
        }
        brown = results[CameraModelType.BROWN_CONRADY]
        for model, r in results.items():
            assert r.total_folds == brown.total_folds
            assert len(r.kfold_results) == len(brown.kfold_results)
            for i, (kf_b, kf_m) in enumerate(zip(brown.kfold_results, r.kfold_results)):
                partition_b = self._test_id_partition(kf_b)
                partition_m = self._test_id_partition(kf_m)
                assert partition_b == partition_m, f"repeat {i}에서 {model}의 fold partition이 다르다"

    def test_format_model_comparison_table_and_rows_no_crash(
        self, synthetic_dataset, camera_config, pattern_config
    ):
        results = run_repeated_kfold_all_models(
            synthetic_dataset, camera_config, pattern_config,
            k=3, n_repeats=2, base_seed=7, n_jobs=1, cache=None,
        )
        text = format_model_comparison_table(results)
        assert "Brown-Conrady" in text and "Fisheye" in text
        rows = model_comparison_rows(results)
        assert len(rows) == 3
        assert {r["model"] for r in rows} == {"brown_conrady", "extended_pinhole", "fisheye"}
        for r in rows:
            assert r["total_folds"] == 3 * 2
            assert r["fully_successful_folds"] + r["partial_success_folds"] + r["failed_folds"] == r["total_folds"]


# ---------------------------------------------------------------------------
# Fisheye pose fallback + K-Fold 통합 - 이전에는 cv2.fisheye.solvePnP 단일
# 실패가 곧바로 test frame 손실(0/12 성공)로 이어졌다. K-Fold도 그 영향을
# 받았을 것이므로, fallback이 있는 상태에서 fold들이 정상적으로 test
# metric을 만들어내는지 확인한다.
# ---------------------------------------------------------------------------

class TestFisheyeKFoldIntegration:
    def test_fisheye_kfold_produces_normal_test_metrics_on_synthetic_gt(self, pattern_config):
        from tests.test_validation import _build_synthetic_fisheye_dataset

        dataset, camera_config, true_K, true_D = _build_synthetic_fisheye_dataset(n_frames=24, seed=2)
        K_before, D_before = true_K.copy(), true_D.copy()

        result = compute_kfold_validation(
            dataset, camera_config, pattern_config, CameraModelType.FISHEYE,
            k=4, seed=3, n_jobs=1, cache=None,
        )

        assert result.n_successful_folds == 4, (
            "Fisheye synthetic GT 데이터에서 fold가 실패했다 - pose fallback 회귀 가능성"
        )
        assert result.mean_test_rms is not None and result.mean_test_rms < 5.0
        assert result.mean_edge_rms is not None
        for vr in result.fold_validation_results:
            assert not vr.failed_test_frame_ids, (
                f"fold의 test frame이 실패로 남음(fallback 회귀?): {vr.failed_test_frame_reasons}"
            )
        # 참고용 GT K,D 자체가 이 테스트 도중 변형되지 않았는지(공용 fixture가
        # 아니라 이 함수가 새로 만든 로컬 배열이지만, 혹시 모를 참조 공유
        # mutation을 잡기 위한 방어적 확인).
        assert (true_K == K_before).all()
        assert (true_D == D_before).all()


# ---------------------------------------------------------------------------
# Repeated K-Fold progress reporting (fold 단위 진행률 콜백).
#
# 75 fold(K=5 x Repeats=5 x 3모델) 계산이 오래 걸려도 사용자가 "멈춘 건지
# 계산 중인지" 알 수 있게 하는 게 목적이라, 여기서는 실제 cv2 calibration
# 대신 빠른 monkeypatch된 validate_holdout으로 progress event의 구조/순서/
# 카운터 정합성만 검증한다(계산 자체는 TestMultiMetricAggregation 등
# 다른 클래스가 이미 검증한다).
# ---------------------------------------------------------------------------

def _fast_fake_validate_holdout(dataset, camera_config, pattern_config, model, train_ids, test_ids, **kwargs):
    from calibration.types import ResidualStats, ValidationResult

    return ValidationResult(
        train_frame_ids=list(train_ids), test_frame_ids=list(test_ids),
        train_rms=1.0, test_rms=1.0, edge_rms=0.5,
        straightness_residual=0.05, straightness_source="test",
        test_residual_stats=ResidualStats(p95=1.5),
        success=True,
    )


class TestKFoldProgressReporting:
    def test_total_folds_for_5x5x3_models_is_75(
        self, synthetic_dataset, camera_config, pattern_config, monkeypatch
    ):
        import calibration.validation as validation_module
        monkeypatch.setattr(validation_module, "validate_holdout", _fast_fake_validate_holdout)

        events: list[KFoldProgressEvent] = []
        run_repeated_kfold_all_models(
            synthetic_dataset, camera_config, pattern_config,
            k=5, n_repeats=5, base_seed=1, n_jobs=1, cache=None,
            progress_callback=events.append,
        )
        all_completed = [e for e in events if e.stage == "all_completed"]
        assert len(all_completed) == 1
        assert all_completed[0].total_folds == 75
        assert all_completed[0].completed_folds == 75

    def test_progress_never_regresses_and_increments_by_exactly_one_per_fold(
        self, synthetic_dataset, camera_config, pattern_config, monkeypatch
    ):
        import calibration.validation as validation_module
        monkeypatch.setattr(validation_module, "validate_holdout", _fast_fake_validate_holdout)

        events: list[KFoldProgressEvent] = []
        run_repeated_kfold_all_models(
            synthetic_dataset, camera_config, pattern_config,
            k=5, n_repeats=5, base_seed=1, n_jobs=1, cache=None,
            progress_callback=events.append,
        )

        completed_seq = [e.completed_folds for e in events]
        assert completed_seq == sorted(completed_seq), "completed_folds가 역행했다"

        fold_events = [e for e in events if e.stage == "fold_completed"]
        assert len(fold_events) == 75, "정확히 75번의 fold_completed 이벤트가 있어야 한다"
        fold_completed_values = [e.completed_folds for e in fold_events]
        assert fold_completed_values == list(range(1, 76)), (
            "fold 하나가 끝날 때마다 completed_folds가 정확히 1씩 증가해야 한다"
        )

    def test_model_completed_fires_exactly_once_per_model_with_full_count(
        self, synthetic_dataset, camera_config, pattern_config, monkeypatch
    ):
        import calibration.validation as validation_module
        monkeypatch.setattr(validation_module, "validate_holdout", _fast_fake_validate_holdout)

        events: list[KFoldProgressEvent] = []
        run_repeated_kfold_all_models(
            synthetic_dataset, camera_config, pattern_config,
            k=5, n_repeats=5, base_seed=1, n_jobs=1, cache=None,
            progress_callback=events.append,
        )

        for model in (CameraModelType.BROWN_CONRADY, CameraModelType.EXTENDED_PINHOLE, CameraModelType.FISHEYE):
            model_completed = [e for e in events if e.stage == "model_completed" and e.model == model]
            assert len(model_completed) == 1, f"{model}의 model_completed 이벤트가 정확히 1개여야 한다"
            event = model_completed[0]
            assert event.model_completed_folds == 25
            assert event.model_total_folds == 25
            assert event.partial_result is not None
            assert event.partial_result.total_folds == 25
            assert event.partial_result.mean_test_rms == pytest.approx(1.0)

            started = [e for e in events if e.stage == "model_started" and e.model == model]
            assert len(started) == 1, f"{model}의 model_started 이벤트가 정확히 1개여야 한다"

    def test_model_order_and_status_progression_matches_brown_rational_fisheye(
        self, synthetic_dataset, camera_config, pattern_config, monkeypatch
    ):
        """모델 시작 순서가 Brown -> Rational -> Fisheye로 유지되는지, 그리고
        아직 시작하지 않은 모델에 대해서는 이벤트가 미리 나오지 않는지."""
        import calibration.validation as validation_module
        monkeypatch.setattr(validation_module, "validate_holdout", _fast_fake_validate_holdout)

        events: list[KFoldProgressEvent] = []
        run_repeated_kfold_all_models(
            synthetic_dataset, camera_config, pattern_config,
            k=5, n_repeats=5, base_seed=1, n_jobs=1, cache=None,
            progress_callback=events.append,
        )
        started_order = [e.model for e in events if e.stage == "model_started"]
        assert started_order == [
            CameraModelType.BROWN_CONRADY, CameraModelType.EXTENDED_PINHOLE, CameraModelType.FISHEYE,
        ]

        # Fisheye의 model_started보다 앞서는 이벤트에는 Fisheye의 fold_completed가
        # 있으면 안 된다 (아직 시작 안 한 모델의 진행 이벤트가 미리 새어나오면 안 됨).
        fisheye_start_idx = next(i for i, e in enumerate(events) if e.stage == "model_started" and e.model == CameraModelType.FISHEYE)
        for e in events[:fisheye_start_idx]:
            if e.stage == "fold_completed":
                assert e.model != CameraModelType.FISHEYE

    def test_progress_thread_safe_with_n_jobs_greater_than_one(
        self, synthetic_dataset, camera_config, pattern_config, monkeypatch
    ):
        """n_jobs>1(repeat 단위 병렬 실행)에서도 completed_folds가 역행하거나
        중복 값을 갖지 않아야 한다 - Lock으로 보호된 카운터인지 확인."""
        import calibration.validation as validation_module
        monkeypatch.setattr(validation_module, "validate_holdout", _fast_fake_validate_holdout)

        events: list[KFoldProgressEvent] = []
        run_repeated_kfold_all_models(
            synthetic_dataset, camera_config, pattern_config,
            k=5, n_repeats=5, base_seed=1, n_jobs=3, cache=None,
            progress_callback=events.append,
        )

        fold_events = [e for e in events if e.stage == "fold_completed"]
        assert len(fold_events) == 75
        fold_completed_values = [e.completed_folds for e in fold_events]
        # 병렬 실행이라 관찰 순서 자체는 완벽히 정렬 안 될 수 있어도(콜백이
        # 서로 다른 thread에서 호출됨), "중복 없이 1~75 각각 정확히 한 번씩"은
        # 반드시 성립해야 한다 - Lock이 없다면 두 thread가 같은 값을 relay하거나
        # 카운터를 잃어버릴 수 있다.
        assert sorted(fold_completed_values) == list(range(1, 76)), (
            "n_jobs>1에서 progress 카운터가 중복되거나 값을 잃어버렸다"
        )
        all_completed = [e for e in events if e.stage == "all_completed"]
        assert len(all_completed) == 1
        assert all_completed[0].completed_folds == 75

    def test_compute_repeated_kfold_single_model_progress_uses_model_scope(
        self, synthetic_dataset, camera_config, pattern_config, monkeypatch
    ):
        """compute_repeated_kfold를 단독으로(run_repeated_kfold_all_models
        없이) 호출하면, completed_folds/total_folds가 이 모델 하나의 진행과
        같아야 한다(모델이 하나뿐이므로 전역=모델 범위)."""
        import calibration.validation as validation_module
        monkeypatch.setattr(validation_module, "validate_holdout", _fast_fake_validate_holdout)

        events: list[KFoldProgressEvent] = []
        compute_repeated_kfold(
            synthetic_dataset, camera_config, pattern_config, CameraModelType.BROWN_CONRADY,
            k=5, n_repeats=5, base_seed=1, n_jobs=1, cache=None,
            progress_callback=events.append,
        )
        for e in events:
            assert e.completed_folds == e.model_completed_folds
            assert e.total_folds == e.model_total_folds == 25
            assert e.model == CameraModelType.BROWN_CONRADY

        fold_events = [e for e in events if e.stage == "fold_completed"]
        assert [e.completed_folds for e in fold_events] == list(range(1, 26))
        model_completed = [e for e in events if e.stage == "model_completed"]
        assert len(model_completed) == 1
        assert model_completed[0].partial_result is not None

    def test_progress_callback_default_none_is_fully_backward_compatible(
        self, synthetic_dataset, camera_config, pattern_config
    ):
        """progress_callback/fold_done_callback을 아예 안 넘기는 기존
        호출부가 (인자 개수/에러 없이) 그대로 동작해야 하고, 같은 seed면
        결과도 동일해야 한다 - progress 콜백 유무가 계산 결과에 영향을
        주면 안 된다."""
        result_first = compute_kfold_validation(
            synthetic_dataset, camera_config, pattern_config, CameraModelType.PINHOLE,
            k=4, seed=9, n_jobs=1, cache=None,
        )
        result_second = compute_kfold_validation(
            synthetic_dataset, camera_config, pattern_config, CameraModelType.PINHOLE,
            k=4, seed=9, n_jobs=1, cache=None,
        )
        assert result_first.mean_test_rms == pytest.approx(result_second.mean_test_rms)
        assert result_first.n_successful_folds == result_second.n_successful_folds

        repeated_result = compute_repeated_kfold(
            synthetic_dataset, camera_config, pattern_config, CameraModelType.PINHOLE,
            k=4, n_repeats=2, base_seed=9, n_jobs=1, cache=None,
        )
        assert repeated_result.n_successful_runs > 0

        all_models_result = run_repeated_kfold_all_models(
            synthetic_dataset, camera_config, pattern_config,
            k=3, n_repeats=1, base_seed=9, n_jobs=1, cache=None,
        )
        assert len(all_models_result) == 3

    def test_callback_exception_does_not_break_calibration(
        self, synthetic_dataset, camera_config, pattern_config, monkeypatch
    ):
        """progress callback이 예외를 던져도 K-Fold 계산 자체는 정상 완료돼야
        한다(콜백은 UI 쪽 코드일 수 있으므로 절대 계산을 막으면 안 됨)."""
        import calibration.validation as validation_module
        monkeypatch.setattr(validation_module, "validate_holdout", _fast_fake_validate_holdout)

        def _raising_callback(event):
            raise RuntimeError("boom - UI 쪽에서 터진 척하는 콜백")

        result = compute_repeated_kfold(
            synthetic_dataset, camera_config, pattern_config, CameraModelType.PINHOLE,
            k=5, n_repeats=5, base_seed=1, n_jobs=1, cache=None,
            progress_callback=_raising_callback,
        )
        assert result.n_successful_runs == 25
        assert result.mean_test_rms == pytest.approx(1.0)

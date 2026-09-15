"""
tests/test_paper_evidence.py
====================================

Paper Evidence 단계 회귀 테스트 - calibration/paper_evidence.py(fold-level
raw export, pairwise win count, ranking sensitivity, ROI, metadata) +
calibration/bootstrap.py의 paper_intrinsic_stability 분리 + Hold-out spatial
evidence(calibration/validation.py::compute_holdout_spatial_evidence).

핵심 확인 사항: 이 파일의 어떤 계산도 calibration 결과/metric 산식/fold
split/Fisheye pose fallback/Straightness TEST source 정책/추천 로직을
바꾸지 않는다는 것 - 전부 이미 계산된 값을 다른 모양으로 재배열만 한다.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import calibration.paper_evidence as pe
from calibration.bootstrap import compute_parameter_bootstrap
from calibration.kfold import compute_kfold_validation, run_repeated_kfold_all_models
from calibration.models.brown_conrady import calibrate_brown_conrady
from calibration.types import (
    CameraModelType,
    DistortionCoeffStat,
    ParameterUncertainty,
    ResidualStats,
    ValidationResult,
)
from calibration.validation import _subset_dataset, compute_holdout_spatial_evidence, split_train_test, validate_holdout


# ---------------------------------------------------------------------------
# 1) paper_intrinsic_stability vs all-parameter overall_stability 분리
# ---------------------------------------------------------------------------

class TestPaperIntrinsicStabilitySeparation:
    def test_paper_intrinsic_stability_uses_only_fx_fy_cx_cy(self, synthetic_dataset, camera_config):
        """distortion coefficient가 fx/fy/cx/cy보다 훨씬 불안정해도
        paper_intrinsic_stability는 절대 그 값에 끌려 내려가면 안 된다."""
        frames = [f for f in synthetic_dataset.enabled_frames if f.detection and f.detection.success][:8]
        object_points = [f.detection.object_points for f in frames]
        image_points = [f.detection.corners for f in frames]
        K_ref = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
        D_ref = np.array([1e-4, 1e-4, 0.0, 0.0, 0.0])  # 전부 near-zero distortion

        result = compute_parameter_bootstrap(
            object_points, image_points, (640, 480), CameraModelType.EXTENDED_PINHOLE,
            K_ref, D_ref, flags=0, n_bootstrap=6, rng_seed=1, n_jobs=1,
        )
        if result is None:
            pytest.skip("bootstrap 재표본이 충분히 성공하지 않음(합성 데이터 특성)")

        intrinsic_values = [result.fx_stability, result.fy_stability, result.cx_stability, result.cy_stability]
        intrinsic_values = [v for v in intrinsic_values if v is not None]
        if not intrinsic_values or result.paper_intrinsic_stability is None:
            pytest.skip("intrinsic stability가 계산되지 않음")
        assert result.paper_intrinsic_stability == pytest.approx(float(np.mean(intrinsic_values)))

    def test_paper_intrinsic_stability_ignores_low_distortion_stability_by_construction(self):
        """near-zero distortion coefficient 하나가 극단적으로 낮은
        stability_score를 가져도, paper_intrinsic_stability는 fx/fy/cx/cy만
        평균한 값과 정확히 같아야 한다(직접 dataclass를 구성해 확정적으로 확인)."""
        from calibration.bootstrap import _paper_stability_fields

        fx = {"stability": 95.0, "mean": 800.0, "std": 5.0}
        fy = {"stability": 92.0, "mean": 800.0, "std": 6.0}
        cx = {"stability": 90.0, "mean": 320.0, "std": 4.0}
        cy = {"stability": 88.0, "mean": 240.0, "std": 3.0}
        d_stats = [
            DistortionCoeffStat(index=0, label="k1", mean=0.1, std=0.001, stability_score=99.0, reference=0.1),
            DistortionCoeffStat(index=1, label="k3", mean=0.0001, std=0.0005, stability_score=0.5, reference=0.0001),
        ]
        fields = _paper_stability_fields(fx, fy, cx, cy, d_stats, 800.0, 800.0, 320.0, 240.0)
        expected_intrinsic = float(np.mean([95.0, 92.0, 90.0, 88.0]))
        assert fields["paper_intrinsic_stability"] == pytest.approx(expected_intrinsic)
        # distortion(0.5%짜리 k3 포함)은 완전히 별도 요약에만 영향을 준다.
        assert fields["distortion_stability_summary"] == pytest.approx(float(np.mean([99.0, 0.5])))
        assert fields["lowest_stability_parameter"] == "k3"

    def test_near_zero_reference_distortion_gets_diagnostic_flag(self):
        from calibration.bootstrap import _paper_stability_fields

        fx = {"stability": 95.0, "mean": 800.0, "std": 5.0}
        fy = {"stability": 92.0, "mean": 800.0, "std": 6.0}
        cx = {"stability": 90.0, "mean": 320.0, "std": 4.0}
        cy = {"stability": 88.0, "mean": 240.0, "std": 3.0}
        near_zero_stat = DistortionCoeffStat(index=0, label="k4", mean=0.00001, std=0.0002, stability_score=1.0, reference=0.00001)
        normal_stat = DistortionCoeffStat(index=1, label="k1", mean=0.12, std=0.001, stability_score=99.0, reference=0.12)
        d_stats = [near_zero_stat, normal_stat]

        _paper_stability_fields(fx, fy, cx, cy, d_stats, 800.0, 800.0, 320.0, 240.0)

        assert near_zero_stat.near_zero_reference is True
        assert near_zero_stat.diagnostic == "near-zero coefficient; relative CV unstable"
        assert normal_stat.near_zero_reference is False
        assert normal_stat.diagnostic is None

    def test_legacy_overall_stability_field_still_populated_and_serializes(self):
        """recommender.py가 여전히 소비하는 overall_stability(all-parameter)가
        새 필드 추가 이후에도 그대로 채워지고, project round-trip에서
        깨지지 않아야 한다."""
        from calibration.project_codecs.intrinsic import _param_uncertainty_from_dict
        from calibration.json_utils import json_safe
        import dataclasses

        pu = ParameterUncertainty(
            fx_std=2.0, fy_std=2.1, cx_std=1.0, cy_std=1.1, method="bootstrap",
            n_bootstrap_success=18, n_bootstrap_total=20,
            fx_mean=800.0, fy_mean=801.0, cx_mean=320.0, cy_mean=240.0,
            fx_stability=95.0, fy_stability=94.0, cx_stability=93.0, cy_stability=92.0,
            overall_stability=62.0,  # legacy all-parameter 값(예: Fisheye 62%)
            paper_intrinsic_stability=93.5,
            distortion_stats=[DistortionCoeffStat(index=0, label="k4", stability_score=10.0, reference=0.0001, near_zero_reference=True, diagnostic="near-zero coefficient; relative CV unstable")],
        )
        raw = json_safe(dataclasses.asdict(pu))
        restored = _param_uncertainty_from_dict(raw)

        assert restored.overall_stability == pytest.approx(62.0)
        assert restored.paper_intrinsic_stability == pytest.approx(93.5)
        assert restored.overall_stability != restored.paper_intrinsic_stability
        assert restored.n_bootstrap_total == 20
        assert len(restored.distortion_stats) == 1
        assert restored.distortion_stats[0].near_zero_reference is True

    def test_old_project_dict_without_paper_fields_restores_safely(self):
        """paper_intrinsic_stability 등 새 필드가 없는 구버전 project dict를
        로드해도 crash하지 않고 None(=계산 안 됨)으로 복원돼야 한다 -
        새 값으로 잘못 채워지면 안 된다."""
        from calibration.project_codecs.intrinsic import _param_uncertainty_from_dict

        legacy_dict = {
            "fx_std": 2.0, "fy_std": 2.1, "cx_std": 1.0, "cy_std": 1.1,
            "method": "bootstrap", "n_bootstrap_success": 18,
            "overall_stability": 62.0,
            # paper_intrinsic_stability 등은 아예 없음(구버전).
        }
        restored = _param_uncertainty_from_dict(legacy_dict)
        assert restored.overall_stability == pytest.approx(62.0)
        assert restored.paper_intrinsic_stability is None
        assert restored.n_bootstrap_total is None
        assert restored.distortion_stats == []


# ---------------------------------------------------------------------------
# 2) Repeated K-Fold fold-level raw export
# ---------------------------------------------------------------------------

class TestKFoldRawExport:
    def test_25_fold_rows_per_model(self, synthetic_dataset, camera_config, pattern_config):
        results = run_repeated_kfold_all_models(
            synthetic_dataset, camera_config, pattern_config,
            models=[CameraModelType.BROWN_CONRADY], k=5, n_repeats=5, base_seed=1, n_jobs=1, cache=None,
        )
        rows = pe.kfold_raw_rows(CameraModelType.BROWN_CONRADY, results[CameraModelType.BROWN_CONRADY])
        assert len(rows) == 25
        for expected_key in (
            "model", "repeat_index", "fold_index", "seed", "train_frame_count",
            "test_frame_count", "successful_test_frames", "failed_test_frames",
            "fold_status", "test_rms", "test_p95", "test_edge_rms",
            "test_straightness", "straightness_source",
        ):
            assert expected_key in rows[0]

    def test_repeat_level_summary_aggregates_five_folds_per_repeat(self, synthetic_dataset, camera_config, pattern_config):
        results = run_repeated_kfold_all_models(
            synthetic_dataset, camera_config, pattern_config,
            models=[CameraModelType.BROWN_CONRADY], k=5, n_repeats=5, base_seed=3, n_jobs=1, cache=None,
        )
        result = results[CameraModelType.BROWN_CONRADY]
        repeat_rows = pe.repeat_level_rows(CameraModelType.BROWN_CONRADY, result)
        assert len(repeat_rows) == 5
        fold_rows = pe.kfold_raw_rows(CameraModelType.BROWN_CONRADY, result)
        for repeat_row in repeat_rows:
            this_repeat_folds = [r for r in fold_rows if r["repeat_index"] == repeat_row["repeat_index"]]
            assert len(this_repeat_folds) == 5
            expected_mean = float(np.mean([f["test_rms"] for f in this_repeat_folds if f["test_rms"] is not None]))
            assert repeat_row["mean_test_rms"] == pytest.approx(expected_mean)

    def test_same_repeat_fold_index_shares_test_frame_ids_across_models(
        self, synthetic_dataset, camera_config, pattern_config
    ):
        results = run_repeated_kfold_all_models(
            synthetic_dataset, camera_config, pattern_config,
            k=3, n_repeats=2, base_seed=5, n_jobs=1, cache=None,
        )
        # split_manifest()는 모델 간 test_frame_ids가 다르면 예외를 던진다 -
        # 예외 없이 끝나는 것 자체가 "동일 fold partition" 검증이다.
        manifest = pe.split_manifest(results)
        assert manifest["folds"]
        assert set(manifest["models"]) == {"brown_conrady", "extended_pinhole", "fisheye"}


# ---------------------------------------------------------------------------
# 3) Pairwise fold win counts / ranking sensitivity (monkeypatched - 빠르고
#    결정론적인 fake fold 값으로 산술 자체만 검증)
# ---------------------------------------------------------------------------

def _fake_row(model: str, repeat_index: int, fold_index: int, test_rms: float | None, straightness=None, source=None) -> dict:
    return {
        "model": model, "repeat_index": repeat_index, "fold_index": fold_index, "seed": 1,
        "train_frame_count": 10, "test_frame_count": 3,
        "successful_test_frames": 3, "failed_test_frames": 0,
        "fold_status": "fully_successful",
        "test_rms": test_rms,
        "test_p95": test_rms * 1.5 if test_rms is not None else None,
        "test_edge_rms": test_rms * 1.2 if test_rms is not None else None,
        "test_straightness": straightness, "straightness_source": source,
    }


class TestPairwiseWinCounts:
    def test_wins_losses_ties_sum_to_comparable_folds(self):
        rows_by_model = {
            CameraModelType.BROWN_CONRADY: [
                _fake_row("brown_conrady", 0, 0, 1.0),
                _fake_row("brown_conrady", 0, 1, 2.0),
                _fake_row("brown_conrady", 0, 2, 3.0),
            ],
            CameraModelType.FISHEYE: [
                _fake_row("fisheye", 0, 0, 0.5),   # fisheye wins
                _fake_row("fisheye", 0, 1, 2.0),   # tie
                _fake_row("fisheye", 0, 2, 5.0),   # brown wins
            ],
        }
        wins = pe.pairwise_win_counts(rows_by_model, "test_rms")
        assert len(wins) == 1
        w = wins[0]
        assert w["wins_a"] + w["wins_b"] + w["ties"] == w["comparable_folds"]
        assert w["wins_a"] == 1  # brown wins fold 2
        assert w["wins_b"] == 1  # fisheye wins fold 0
        assert w["ties"] == 1

    def test_incomparable_folds_are_excluded(self):
        rows_by_model = {
            CameraModelType.BROWN_CONRADY: [_fake_row("brown_conrady", 0, 0, 1.0)],
            CameraModelType.FISHEYE: [_fake_row("fisheye", 0, 0, None)],  # missing value
        }
        wins = pe.pairwise_win_counts(rows_by_model, "test_rms")[0]
        assert wins["comparable_folds"] == 0
        assert wins["wins_a"] == wins["wins_b"] == wins["ties"] == 0

    def test_three_models_produce_three_pairs(self):
        rows_by_model = {
            m: [_fake_row(m.value, 0, 0, float(i + 1))]
            for i, m in enumerate([CameraModelType.BROWN_CONRADY, CameraModelType.EXTENDED_PINHOLE, CameraModelType.FISHEYE])
        }
        wins = pe.pairwise_win_counts(rows_by_model, "test_rms")
        assert len(wins) == 3  # C(3,2)


class TestRankingSensitivity:
    def test_detects_brown_rational_reversal_matching_reported_scenario(self):
        """실제 보고된 상황(Fisheye가 두 split 모두에서 1위, Brown/Rational
        순위만 뒤집힘: Single Hold-out Brown 4.252 < Rational 6.928,
        Repeated CV Rational 4.014 < Brown 6.270)을 그대로 재현해
        ranking_changed=True와 순서 자체를 확인한다."""
        single_holdout = {
            CameraModelType.BROWN_CONRADY: ValidationResult(test_rms=4.252, success=True),
            CameraModelType.EXTENDED_PINHOLE: ValidationResult(test_rms=6.928, success=True),
            CameraModelType.FISHEYE: ValidationResult(test_rms=3.011, success=True),
        }
        repeated = {
            CameraModelType.BROWN_CONRADY: type("R", (), {
                "mean_test_rms": 6.270, "mean_test_p95": None, "mean_edge_rms": None,
                "mean_test_straightness": None, "n_straightness_folds": 0,
            })(),
            CameraModelType.EXTENDED_PINHOLE: type("R", (), {
                "mean_test_rms": 4.014, "mean_test_p95": None, "mean_edge_rms": None,
                "mean_test_straightness": None, "n_straightness_folds": 0,
            })(),
            CameraModelType.FISHEYE: type("R", (), {
                "mean_test_rms": 1.769, "mean_test_p95": None, "mean_edge_rms": None,
                "mean_test_straightness": None, "n_straightness_folds": 0,
            })(),
        }
        sensitivity = pe.ranking_sensitivity(single_holdout, repeated)
        assert sensitivity["test_rms"]["ranking_changed"] is True
        single_order = [e["model"] for e in sensitivity["test_rms"]["single_holdout_ranking"]]
        repeated_order = [e["model"] for e in sensitivity["test_rms"]["repeated_cv_ranking"]]
        assert single_order == ["fisheye", "brown_conrady", "extended_pinhole"]
        assert repeated_order == ["fisheye", "extended_pinhole", "brown_conrady"]

    def test_train_fallback_straightness_excluded_from_ranking(self):
        single_holdout = {
            CameraModelType.BROWN_CONRADY: ValidationResult(
                straightness_residual=0.9, straightness_source="train_fallback", success=True,
            ),
        }
        repeated = {
            CameraModelType.BROWN_CONRADY: type("R", (), {
                "mean_test_rms": None, "mean_test_p95": None, "mean_edge_rms": None,
                "mean_test_straightness": 0.05, "n_straightness_folds": 0,  # missing
            })(),
        }
        sensitivity = pe.ranking_sensitivity(single_holdout, repeated)
        assert sensitivity["test_straightness"]["single_holdout_ranking"] == []
        assert sensitivity["test_straightness"]["repeated_cv_ranking"] == []

    def test_actual_reversal_case(self):
        """Brown이 Single Hold-out에서 이기고 Repeated CV에서는 지는 실제
        reversal 케이스."""
        single_holdout = {
            CameraModelType.BROWN_CONRADY: ValidationResult(test_rms=1.0, success=True),
            CameraModelType.FISHEYE: ValidationResult(test_rms=2.0, success=True),
        }
        repeated = {
            CameraModelType.BROWN_CONRADY: type("R", (), {
                "mean_test_rms": 5.0, "mean_test_p95": None, "mean_edge_rms": None,
                "mean_test_straightness": None, "n_straightness_folds": 0,
            })(),
            CameraModelType.FISHEYE: type("R", (), {
                "mean_test_rms": 2.0, "mean_test_p95": None, "mean_edge_rms": None,
                "mean_test_straightness": None, "n_straightness_folds": 0,
            })(),
        }
        sensitivity = pe.ranking_sensitivity(single_holdout, repeated)
        assert sensitivity["test_rms"]["ranking_changed"] is True
        assert [e["model"] for e in sensitivity["test_rms"]["single_holdout_ranking"]] == ["brown_conrady", "fisheye"]
        assert [e["model"] for e in sensitivity["test_rms"]["repeated_cv_ranking"]] == ["fisheye", "brown_conrady"]


# ---------------------------------------------------------------------------
# 4) ROI - 기본값 Full Image는 metric을 바꾸지 않는다
# ---------------------------------------------------------------------------

class TestEvaluationROI:
    def test_full_image_roi_keeps_all_points_and_matches_direct_rms(self):
        roi = pe.full_image_roi(640, 480)
        assert roi.is_full_image
        rows = [
            {"x": 10.0, "y": 10.0, "dx": 1.0, "dy": 0.0, "magnitude": 1.0},
            {"x": 630.0, "y": 470.0, "dx": 0.0, "dy": 2.0, "magnitude": 2.0},
        ]
        filtered = pe.filter_points_by_roi(rows, roi)
        assert filtered == rows  # Full Image는 전부 통과

        stats = pe.roi_recomputed_stats(filtered)
        direct_rms = float(np.sqrt(np.mean(np.array([1.0, 2.0]) ** 2)))
        assert stats["rms"] == pytest.approx(direct_rms)

    def test_narrower_roi_excludes_points_outside_and_is_applied_identically(self):
        roi = pe.EvaluationROI(x_min=0, y_min=0, x_max=320, y_max=240, image_width=640, image_height=480, label="Upper-Left Quadrant")
        assert not roi.is_full_image
        rows_model_a = [
            {"x": 10.0, "y": 10.0, "magnitude": 1.0},
            {"x": 500.0, "y": 400.0, "magnitude": 9.0},  # 밖
        ]
        rows_model_b = [
            {"x": 15.0, "y": 15.0, "magnitude": 3.0},
            {"x": 500.0, "y": 400.0, "magnitude": 9.0},  # 밖
        ]
        filtered_a = pe.filter_points_by_roi(rows_model_a, roi)
        filtered_b = pe.filter_points_by_roi(rows_model_b, roi)
        # 같은 ROI 정의가 두 모델 모두에 동일한 기준(0<=x<=320, 0<=y<=240)으로 적용됨.
        assert len(filtered_a) == 1 and filtered_a[0]["magnitude"] == 1.0
        assert len(filtered_b) == 1 and filtered_b[0]["magnitude"] == 3.0
        roi_dict = roi.to_dict()
        assert roi_dict["pixel"]["x_max"] == 320
        assert roi_dict["normalized"]["x_max"] == pytest.approx(320 / 640)


# ---------------------------------------------------------------------------
# 5) Hold-out spatial evidence - Train residual이 섞이지 않는지, K/D가
#    바뀌지 않는지.
# ---------------------------------------------------------------------------

class TestHoldoutSpatialEvidence:
    def test_uses_only_test_frames_not_train(self, synthetic_dataset, camera_config, pattern_config):
        dataset = synthetic_dataset
        train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=7)
        assert train_ids and test_ids

        train_dataset = _subset_dataset(dataset, train_ids)
        train_result = calibrate_brown_conrady(train_dataset, camera_config)
        assert train_result.success

        smap, rows = compute_holdout_spatial_evidence(
            dataset, camera_config, CameraModelType.BROWN_CONRADY, test_ids,
            train_result.camera_matrix, train_result.distortion, rows=4, cols=4,
        )
        frame_ids_used = {r["frame_id"] for r in rows}
        assert frame_ids_used, "spatial evidence row가 하나도 없음"
        assert frame_ids_used <= set(test_ids), "train frame이 hold-out spatial evidence에 섞여 들어감"
        assert frame_ids_used.isdisjoint(set(train_ids))

    def test_does_not_mutate_k_d(self, synthetic_dataset, camera_config, pattern_config):
        dataset = synthetic_dataset
        train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=7)
        train_result = calibrate_brown_conrady(_subset_dataset(dataset, train_ids), camera_config)
        K_before = train_result.camera_matrix.copy()
        D_before = train_result.distortion.copy()

        compute_holdout_spatial_evidence(
            dataset, camera_config, CameraModelType.BROWN_CONRADY, test_ids,
            train_result.camera_matrix, train_result.distortion, rows=4, cols=4,
        )
        assert np.array_equal(train_result.camera_matrix, K_before)
        assert np.array_equal(train_result.distortion, D_before)

    def test_grid_row_col_within_bounds(self, synthetic_dataset, camera_config, pattern_config):
        dataset = synthetic_dataset
        train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=7)
        train_result = calibrate_brown_conrady(_subset_dataset(dataset, train_ids), camera_config)
        _smap, rows = compute_holdout_spatial_evidence(
            dataset, camera_config, CameraModelType.BROWN_CONRADY, test_ids,
            train_result.camera_matrix, train_result.distortion, rows=4, cols=4,
        )
        for r in rows:
            assert 0 <= r["grid_row"] < 4
            assert 0 <= r["grid_col"] < 4
            assert r["magnitude"] == pytest.approx((r["dx"] ** 2 + r["dy"] ** 2) ** 0.5)


# ---------------------------------------------------------------------------
# 6) Export - CSV/JSON round trip
# ---------------------------------------------------------------------------

class TestExportPaperMetrics:
    def test_export_writes_all_expected_files_with_valid_content(self, tmp_path, synthetic_dataset, camera_config, pattern_config):
        train_ids, test_ids = split_train_test(synthetic_dataset, camera_config, test_ratio=0.3, seed=9)
        single = validate_holdout(
            synthetic_dataset, camera_config, pattern_config, CameraModelType.BROWN_CONRADY, train_ids, test_ids,
        )
        repeated = run_repeated_kfold_all_models(
            synthetic_dataset, camera_config, pattern_config,
            models=[CameraModelType.BROWN_CONRADY], k=3, n_repeats=2, base_seed=1, n_jobs=1, cache=None,
        )
        metadata = pe.build_paper_metadata(camera_config, pattern_config, synthetic_dataset, k=3, n_repeats=2, base_seed=1)

        written = pe.export_paper_metrics(
            tmp_path, {CameraModelType.BROWN_CONRADY: single}, repeated, {}, metadata,
        )
        for name in (
            "repeated_kfold_folds.csv", "repeated_kfold_repeats.csv", "pairwise_win_counts.csv",
            "stability_parameters.csv", "split_manifest.json", "paper_summary.csv", "paper_metrics.json",
        ):
            assert name in written
            path = tmp_path / name
            assert path.exists()
            assert path.stat().st_size > 0

        manifest = json.loads((tmp_path / "split_manifest.json").read_text(encoding="utf-8"))
        assert len(manifest["folds"]) == 6  # k=3 x n_repeats=2

        metrics = json.loads((tmp_path / "paper_metrics.json").read_text(encoding="utf-8"))
        assert "brown_conrady" in metrics["single_holdout"]
        assert metrics["metadata"]["k"] == 3
        assert metrics["metadata"]["n_repeats"] == 2
        assert metrics["metadata"]["evaluation_roi"]["is_full_image"] is True
        assert "camera_installation" not in metrics["metadata"]
        assert "target_location" not in metrics["metadata"]

    def test_paper_metadata_exports_installation_only_when_explicit(
        self, synthetic_dataset, camera_config, pattern_config
    ):
        metadata = pe.build_paper_metadata(
            camera_config,
            pattern_config,
            synthetic_dataset,
            k=3,
            n_repeats=2,
            base_seed=1,
            camera_installation="exterior mount",
            target_location="in front of camera",
        )

        exported = metadata.to_dict()
        assert exported["camera_installation"] == "exterior mount"
        assert exported["target_location"] == "in front of camera"

    def test_paper_summary_report_has_no_auto_generated_verdict(self, synthetic_dataset, camera_config, pattern_config):
        """보고서 텍스트에 "가장 좋다"/"실패" 같은 자동 판정 문구가 없는지
        가볍게 확인 - 정확한 문구 검열이 아니라, 최소한 이 함수가 절대
        쓰지 않기로 한 대표적인 결론성 단어들이 없는지 정도만 본다."""
        train_ids, test_ids = split_train_test(synthetic_dataset, camera_config, test_ratio=0.3, seed=9)
        single = validate_holdout(
            synthetic_dataset, camera_config, pattern_config, CameraModelType.BROWN_CONRADY, train_ids, test_ids,
        )
        repeated = run_repeated_kfold_all_models(
            synthetic_dataset, camera_config, pattern_config,
            models=[CameraModelType.BROWN_CONRADY], k=3, n_repeats=2, base_seed=1, n_jobs=1, cache=None,
        )
        report = pe.build_paper_summary_report({CameraModelType.BROWN_CONRADY: single}, repeated)
        for banned in ("best model", "가장 좋", "실패 모델", "recommended winner"):
            assert banned not in report

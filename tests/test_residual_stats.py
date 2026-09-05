"""
tests/test_residual_stats.py
===================================

설계 문서 11번(Reprojection Error 지표 확장)/12번(Residual Distribution 분석).
"""

from __future__ import annotations

import numpy as np
import pytest

from calibration.residual_stats import (
    compute_cdf,
    compute_residual_stats,
    format_cdf,
    format_residual_boxplot,
    format_residual_histogram,
    format_residual_stats,
)
from calibration.types import ResidualStats


class TestComputeResidualStats:
    def test_empty_input_returns_zero_n(self):
        stats = compute_residual_stats([])
        assert stats.n == 0
        assert stats.rmse is None

    def test_basic_statistics_are_correct(self):
        # 0,1,2,...,99 -> 알려진 통계량으로 공식 자체를 검증
        errors = list(range(100))
        stats = compute_residual_stats(errors)
        assert stats.n == 100
        assert stats.median == pytest.approx(49.5)
        assert stats.mae == pytest.approx(np.mean(errors))
        assert stats.min == 0.0
        assert stats.max == 99.0
        assert stats.rmse == pytest.approx(np.sqrt(np.mean(np.array(errors, dtype=float) ** 2)))
        assert stats.std == pytest.approx(np.std(errors))

    def test_percentiles_are_monotonic(self):
        rng = np.random.default_rng(0)
        errors = rng.exponential(scale=0.5, size=500)
        stats = compute_residual_stats(errors)
        assert stats.min <= stats.q1 <= stats.median <= stats.q3 <= stats.p90 <= stats.p95 <= stats.p99 <= stats.max

    def test_nan_and_inf_are_filtered_out(self):
        errors = [0.1, 0.2, float("nan"), 0.3, float("inf")]
        stats = compute_residual_stats(errors)
        assert stats.n == 3  # NaN/Inf 두 개는 제외

    def test_all_nan_returns_zero_n(self):
        stats = compute_residual_stats([float("nan"), float("inf")])
        assert stats.n == 0

    def test_outlier_count_detects_extreme_value(self):
        normal = [0.3, 0.31, 0.29, 0.32, 0.28, 0.30]
        with_outlier = normal + [50.0]
        stats = compute_residual_stats(with_outlier)
        assert stats.outlier_count == 1

    def test_no_outliers_when_all_similar(self):
        stats = compute_residual_stats([0.3, 0.31, 0.29, 0.32, 0.28, 0.30])
        assert stats.outlier_count == 0

    def test_histogram_counts_sum_to_n(self):
        rng = np.random.default_rng(1)
        errors = rng.normal(loc=1.0, scale=0.2, size=200)
        errors = np.abs(errors)
        stats = compute_residual_stats(errors, num_histogram_bins=10)
        assert sum(stats.histogram_counts) == stats.n
        assert len(stats.histogram_bin_edges) == len(stats.histogram_counts) + 1

    def test_sample_residuals_capped_and_sorted(self):
        errors = list(range(1000))  # 캡(500)보다 훨씬 많은 포인트
        stats = compute_residual_stats(errors)
        assert len(stats.sample_residuals) == 500
        assert stats.sample_residuals == sorted(stats.sample_residuals)

    def test_sample_residuals_keeps_all_when_under_cap(self):
        errors = [0.1, 0.2, 0.3]
        stats = compute_residual_stats(errors)
        assert len(stats.sample_residuals) == 3

    def test_sample_residuals_deterministic_across_calls(self):
        rng = np.random.default_rng(5)
        errors = rng.exponential(size=2000)
        stats1 = compute_residual_stats(errors)
        stats2 = compute_residual_stats(errors)
        assert stats1.sample_residuals == stats2.sample_residuals


class TestComputeCdf:
    def test_empty_stats_returns_empty_lists(self):
        edges, cum = compute_cdf(ResidualStats(n=0))
        assert edges == []
        assert cum == []

    def test_cdf_ends_at_one(self):
        errors = list(range(1, 101))
        stats = compute_residual_stats(errors, num_histogram_bins=10)
        edges, cum = compute_cdf(stats)
        assert len(edges) == len(cum)
        assert cum[-1] == pytest.approx(1.0)

    def test_cdf_is_monotonically_nondecreasing(self):
        rng = np.random.default_rng(2)
        errors = rng.exponential(scale=1.0, size=300)
        stats = compute_residual_stats(errors)
        _, cum = compute_cdf(stats)
        assert all(b >= a - 1e-9 for a, b in zip(cum, cum[1:]))


class TestFormatting:
    def test_format_residual_stats_includes_key_metrics(self):
        stats = compute_residual_stats([0.1, 0.2, 0.3, 0.4, 0.5])
        text = format_residual_stats(stats)
        for label in ("RMSE", "MAE", "Median", "Std", "P90", "P95", "P99", "Max", "Outliers"):
            assert label in text

    def test_format_residual_stats_handles_empty(self):
        text = format_residual_stats(ResidualStats(n=0))
        assert "없습니다" in text

    def test_format_histogram_no_crash(self):
        stats = compute_residual_stats([0.1, 0.2, 0.3, 0.4, 0.5], num_histogram_bins=5)
        text = format_residual_histogram(stats)
        assert "Histogram" in text

    def test_format_histogram_handles_empty(self):
        text = format_residual_histogram(ResidualStats(n=0))
        assert "없습니다" in text


class TestBoxplot:
    def test_format_boxplot_includes_five_number_summary(self):
        stats = compute_residual_stats([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        text = format_residual_boxplot(stats)
        for label in ("Min", "Q1", "Median", "Q3", "Max", "Outliers"):
            assert label in text

    def test_format_boxplot_handles_empty(self):
        text = format_residual_boxplot(ResidualStats(n=0))
        assert "없습니다" in text

    def test_format_boxplot_handles_constant_values(self):
        stats = compute_residual_stats([0.5, 0.5, 0.5, 0.5])
        text = format_residual_boxplot(stats)
        assert "동일합니다" in text


class TestCdfFormatting:
    def test_format_cdf_includes_percentile_markers(self):
        rng = np.random.default_rng(3)
        errors = rng.exponential(scale=0.5, size=300)
        stats = compute_residual_stats(errors)
        text = format_cdf(stats)
        assert "CDF" in text
        for label in ("P90", "P95", "P99"):
            assert label in text

    def test_format_cdf_ends_near_100_percent(self):
        errors = list(range(1, 101))
        stats = compute_residual_stats(errors, num_histogram_bins=10)
        text = format_cdf(stats)
        assert "100.0%" in text

    def test_format_cdf_handles_empty(self):
        text = format_cdf(ResidualStats(n=0))
        assert "없습니다" in text

    def test_format_cdf_row_count_matches_histogram_bins(self):
        stats = compute_residual_stats([0.1, 0.2, 0.3, 0.4, 0.5], num_histogram_bins=5)
        text = format_cdf(stats)
        # 헤더 1줄 + bin 5줄 + percentile 참고선 1줄
        assert len(text.splitlines()) == 1 + 5 + 1

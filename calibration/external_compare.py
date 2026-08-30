"""
camera_calibrator.calibration.external_compare
====================================================

사용자 요청 - "예전에 다른 사람/다른 툴로 같은 카메라를 캘리브레이션한
값"과 "이 툴로 지금 구한 값" 중 뭐가 더 정확한지, 주장이 아니라 누구나
확인 가능한 정량적 근거로 비교하는 기능.

--- 왜 "그냥 두 RMS 숫자를 나란히 놓고 비교"하면 안 되는가 ---

"내 RMS"는 보통 내 데이터셋으로 학습한 뒤 그 데이터셋에서 측정한 값이다.
"예전 파라미터"는 완전히 다른 촬영 세션에서 나왔으니, 애초에 "내 데이터셋
전체"가 그쪽 입장에서는 전부 처음 보는 데이터다. 그래서 "내 파라미터를
내 데이터로 잰 값" vs "예전 파라미터를 내 데이터에 그대로 적용해 잰 값"을
그냥 비교하면, 내가 유리한 조건에서 이긴 것처럼 보일 수 있다 - 이러면
"그야 네가 만든 툴이니까 네 편을 들겠지"라는 반박을 피할 수 없다.

그래서 이 모듈은 두 파라미터 모두에게 완전히 동일한 조건을 강제한다:
    1. 비교는 항상 "내가 캘리브레이션 학습에 전혀 쓰지 않은 프레임"
       (validate_holdout이 이미 떼어둔 test 분할)에서만 이뤄진다.
       - 외부 파라미터에게는 어차피 전체가 안 본 데이터라 손해볼 게 없다.
       - 내 파라미터도 같은 test 분할로 "다시 학습한" 버전
         (calibration/validation.py의 refit_on_train_split, 곧
         ValidationResult.test_rms를 만들어낸 바로 그 절차)을 써서, 이
         test 프레임을 학습에 훔쳐본 적이 없게 만든다.
    2. 두 쪽 다 intrinsic을 이 데이터로 다시 최적화하지 않는다 - solvePnP로
       pose만 새로 구하고 그대로 재투영한다(validation.py의 핵심 원칙과
       동일). 그래야 "파라미터 자체가 이 카메라를 얼마나 잘 설명하는지"만
       순수하게 잰다.
    3. 두 쪽에 동일한 계산 함수(_test_reprojection_errors,
       compute_regional_error, compute_straightness_residual)를 그대로
       적용한다 - "계산 방식이 달라서 결과가 갈렸다"는 반박 자체가
       원천적으로 불가능하게.

RMS 숫자 하나로 승패를 가르지 않는다("RMS가 가장 낮은 모델 = 정답은 절대
금지" 원칙을 두 파라미터 비교에도 동일하게 적용): Test RMS / Edge RMS(외곽)
/ Straightness Residual(직선성) 3개 독립 지표 + 프레임별 승-패 개수까지
전부 보여주고, 지표가 엇갈리면 엇갈린다고 정직하게 말한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np

from calibration.benchmark_compatibility import (
    validate_calibration_pair_compatibility,
    validate_single_calibration,
)
from calibration.calibration_io import StandardCalibration
from calibration.types import (
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    Dataset,
    PatternConfig,
    ResidualStats,
    ValidationResult,
)
from calibration.validation import (
    _subset_dataset,
    refit_on_train_split,
)
from calibration.models.common import compute_regional_error, distortion_coeff_labels, regional_edge_average
from calibration.residual_stats import compute_residual_stats
from calibration.straightness import compute_straightness_residual

_MODEL_LABELS = {
    CameraModelType.PINHOLE: "Ideal Pinhole",
    CameraModelType.BROWN_CONRADY: "Brown-Conrady",
    CameraModelType.EXTENDED_PINHOLE: "Rational",
    CameraModelType.FISHEYE: "Fisheye",
}


@dataclass
class ExternalCameraParams:
    """비교 대상이 되는 "다른 곳에서 구한" 카메라 파라미터.
    이 툴이 만든 결과가 아니어도 된다(수기 입력, 다른 소프트웨어의 OpenCV
    YAML 등) - label/source_note에 출처를 남겨 화면/리포트에서 항상
    "이건 어디서 온 값인지" 구분되게 한다.
    """
    label: str
    model_name: CameraModelType
    camera_matrix: np.ndarray   # 3x3
    distortion: np.ndarray
    source_note: str = ""       # 예: "2025-03 A업체 캘리브레이션, OpenCV YAML"
    width: int | None = None
    height: int | None = None
    distortion_model: str | None = None


@dataclass
class PointErrorDetail:
    frame_id: str
    corner_index: int
    x: float
    y: float
    error: float


@dataclass
class ComparisonSide:
    """한쪽(내 결과 또는 외부 결과)을 동일 조건(같은 test 프레임, 같은
    solvePnP-only 절차)으로 재평가한 결과."""
    label: str
    model_name: CameraModelType | None = None
    camera_matrix: np.ndarray | None = None
    distortion: np.ndarray | None = None
    test_rms: float | None = None
    residual_stats: ResidualStats | None = None
    edge_rms: float | None = None
    straightness_residual: float | None = None
    per_frame_error: dict[str, float] = field(default_factory=dict)
    point_errors_xy: list[tuple[float, float, float]] = field(default_factory=list)
    point_error_details: list[PointErrorDetail] = field(default_factory=list)
    failed_frame_ids: list[str] = field(default_factory=list)
    success: bool = False
    error_message: str | None = None


@dataclass
class SpatialMetricCell:
    row: int
    col: int
    num_reference_points: int = 0
    num_candidate_points: int = 0
    reference_mean: float | None = None
    candidate_mean: float | None = None
    improvement_mean_pct: float | None = None
    reference_rmse: float | None = None
    candidate_rmse: float | None = None
    improvement_rmse_pct: float | None = None
    reference_p95: float | None = None
    candidate_p95: float | None = None
    improvement_p95_pct: float | None = None
    reference_max: float | None = None
    candidate_max: float | None = None
    improvement_max_pct: float | None = None


@dataclass
class SpatialComparisonGrid:
    rows: int
    cols: int
    cells: list[SpatialMetricCell] = field(default_factory=list)


@dataclass
class HeatmapCell:
    row: int
    col: int
    reference_value: float | None = None
    candidate_value: float | None = None
    difference_value: float | None = None  # candidate - reference, positive means candidate is worse
    num_reference_points: int = 0
    num_candidate_points: int = 0


@dataclass
class ResidualHeatmapComparison:
    rows: int
    cols: int
    metric: str = "rmse"
    cells: list[HeatmapCell] = field(default_factory=list)
    reference_max: float | None = None
    candidate_max: float | None = None
    difference_abs_max: float | None = None


@dataclass
class RadialMetricBand:
    label: str
    radius_min_norm: float
    radius_max_norm: float
    num_reference_points: int = 0
    num_candidate_points: int = 0
    reference_mean: float | None = None
    candidate_mean: float | None = None
    improvement_mean_pct: float | None = None
    reference_rmse: float | None = None
    candidate_rmse: float | None = None
    improvement_rmse_pct: float | None = None
    reference_p95: float | None = None
    candidate_p95: float | None = None
    improvement_p95_pct: float | None = None
    reference_max: float | None = None
    candidate_max: float | None = None
    improvement_max_pct: float | None = None


@dataclass
class RadialComparisonProfile:
    bands: list[RadialMetricBand] = field(default_factory=list)
    max_radius_px: float = 0.0


@dataclass
class MetricComparisonRow:
    """Reference/Candidate 성능표 한 줄.

    improvement_pct는 낮을수록 좋은 오차 지표 전용으로 계산한다:
        (reference - candidate) / reference * 100
    그래서 양수면 candidate 쪽 개선, 음수면 candidate 쪽 악화다.
    기존 "내 결과 vs 외부 결과" 모드에서는 external을 reference,
    mine을 candidate로 해석한다.
    """
    metric: str
    reference_value: float | None
    candidate_value: float | None
    improvement_pct: float | None = None
    winner: str = "N/A"


@dataclass
class FinalBenchmarkRow:
    metric: str
    reference: str = "N/A"
    candidate: str = "N/A"
    improvement: str = "N/A"
    winner: str = "N/A"


@dataclass
class WinnerDecision:
    """Benchmark winner state machine output."""
    status: str = "Insufficient Evidence"
    candidate_score: float = 0.0
    reference_score: float = 0.0
    score_margin: float = 0.0
    data_quality_ok: bool = False
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class WorstCaseRow:
    category: str
    reference_location: str = "N/A"
    reference_value: float | None = None
    candidate_location: str = "N/A"
    candidate_value: float | None = None
    improvement_pct: float | None = None
    winner: str = "N/A"


@dataclass
class ErrorDistributionBin:
    bin_start: float
    bin_end: float
    reference_count: int = 0
    candidate_count: int = 0
    reference_density: float = 0.0
    candidate_density: float = 0.0
    reference_cdf: float = 0.0
    candidate_cdf: float = 0.0
    cdf_delta: float = 0.0  # candidate - reference


@dataclass
class ErrorDistributionComparison:
    bins: list[ErrorDistributionBin] = field(default_factory=list)
    num_reference_points: int = 0
    num_candidate_points: int = 0
    reference_median: float | None = None
    candidate_median: float | None = None
    reference_p95: float | None = None
    candidate_p95: float | None = None


@dataclass
class ParameterDiffRow:
    """Reference/Candidate parameter difference table row.

    relative_diff_pct는 Reference 기준:
        abs(candidate - reference) / abs(reference) * 100
    """
    name: str
    reference_value: float | None
    candidate_value: float | None
    absolute_diff: float | None = None
    relative_diff_pct: float | None = None
    unit: str = ""


@dataclass
class BenchmarkValidationRow:
    """Benchmark-only hold-out/k-fold/generalization summary.

    train_validation_gap = validation_rms_mean - train_rms_mean.
    improvement_pct uses lower-is-better validation RMS:
        (reference_validation - candidate_validation) / reference_validation * 100
    """
    name: str
    num_splits: int = 0
    reference_train_rms_mean: float | None = None
    reference_validation_rms_mean: float | None = None
    reference_validation_rms_std: float | None = None
    reference_train_validation_gap: float | None = None
    candidate_train_rms_mean: float | None = None
    candidate_validation_rms_mean: float | None = None
    candidate_validation_rms_std: float | None = None
    candidate_train_validation_gap: float | None = None
    improvement_pct: float | None = None


@dataclass
class StatisticalTestResult:
    """Paired Reference/Candidate statistical test over common frame RMS values.

    differences are candidate - reference. Negative mean_diff means Candidate
    has lower per-frame error.
    """
    test_name: str
    statistic: float | None = None
    p_value: float | None = None
    effect_size: float | None = None
    effect_size_name: str = ""
    n_pairs: int = 0
    mean_diff: float | None = None
    median_diff: float | None = None
    interpretation: str = "N/A"


@dataclass
class BenchmarkBootstrapResult:
    """Paired bootstrap over common frame RMS values.

    Each bootstrap sample resamples frame pairs together, preserving the
    Reference/Candidate pairing for the same image.
    """
    n_pairs: int = 0
    n_bootstrap: int = 0
    confidence_level: float = 0.95
    probability_candidate_better: float | None = None
    reference_rmse: float | None = None
    candidate_rmse: float | None = None
    reference_rmse_ci_low: float | None = None
    reference_rmse_ci_high: float | None = None
    candidate_rmse_ci_low: float | None = None
    candidate_rmse_ci_high: float | None = None
    improvement_pct: float | None = None
    improvement_ci_low: float | None = None
    improvement_ci_high: float | None = None


@dataclass
class ParameterStabilityRow:
    parameter: str
    value: float | None = None
    std: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    stability_score: float | None = None


@dataclass
class ParameterSensitivityRow:
    parameter: str
    value: float | None = None
    perturbation: float | None = None
    rmse_delta: float | None = None
    sensitivity_per_unit: float | None = None


@dataclass
class BenchmarkParameterDiagnostics:
    side_label: str
    parameter_labels: list[str] = field(default_factory=list)
    n_points: int = 0
    jacobian_rows: int = 0
    jacobian_cols: int = 0
    rank: int | None = None
    singular_values: list[float] = field(default_factory=list)
    min_singular_value: float | None = None
    max_singular_value: float | None = None
    covariance_matrix: list[list[float]] = field(default_factory=list)
    correlation_matrix: list[list[float]] = field(default_factory=list)
    max_abs_correlation: float | None = None
    top_correlations: list[tuple[str, str, float]] = field(default_factory=list)
    condition_number: float | None = None
    weak_parameters: list[str] = field(default_factory=list)
    stability_rows: list[ParameterStabilityRow] = field(default_factory=list)
    sensitivity_rows: list[ParameterSensitivityRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ExternalComparisonResult:
    mine: ComparisonSide
    external: ComparisonSide
    num_common_frames: int = 0
    mine_win_count: int = 0
    external_win_count: int = 0
    tie_count: int = 0
    verdict: str = ""
    winner_decision: WinnerDecision = field(default_factory=WinnerDecision)
    caveats: list[str] = field(default_factory=list)
    metric_rows: list[MetricComparisonRow] = field(default_factory=list)
    final_benchmark_rows: list[FinalBenchmarkRow] = field(default_factory=list)
    worst_case_rows: list[WorstCaseRow] = field(default_factory=list)
    error_distribution: ErrorDistributionComparison | None = None
    spatial_comparisons: dict[str, SpatialComparisonGrid] = field(default_factory=dict)
    residual_heatmaps: dict[str, ResidualHeatmapComparison] = field(default_factory=dict)
    radial_comparisons: dict[str, RadialComparisonProfile] = field(default_factory=dict)
    parameter_diff_rows: list[ParameterDiffRow] = field(default_factory=list)
    fov_diff_rows: list[ParameterDiffRow] = field(default_factory=list)
    benchmark_validation_rows: list[BenchmarkValidationRow] = field(default_factory=list)
    statistical_tests: list[StatisticalTestResult] = field(default_factory=list)
    bootstrap_comparison: BenchmarkBootstrapResult | None = None
    parameter_diagnostics: dict[str, BenchmarkParameterDiagnostics] = field(default_factory=dict)
    evaluation_source: str = "internal_holdout"  # internal_holdout | independent_benchmark
    confidence: str = "limited"                 # limited | high
    evaluation_mode: str = "auto"
    benchmark_image_count: int = 0
    benchmark_usable_frames: int = 0
    benchmark_overlap_count: int = 0
    benchmark_status: str = "not_provided"


def _metric_value(side: ComparisonSide, metric_key: str) -> float | None:
    if metric_key == "edge_rms":
        return side.edge_rms
    if metric_key == "straightness":
        return side.straightness_residual
    if metric_key == "frame_wins":
        return None
    stats = side.residual_stats
    if not stats:
        return side.test_rms if metric_key == "rmse" else None
    return getattr(stats, metric_key, None)


def _improvement_pct(reference_value: float | None, candidate_value: float | None) -> float | None:
    if reference_value is None or candidate_value is None:
        return None
    if not np.isfinite(reference_value) or not np.isfinite(candidate_value):
        return None
    if abs(reference_value) < 1e-12:
        return None
    return float((reference_value - candidate_value) / reference_value * 100.0)


def build_metric_comparison_rows(
    reference_side: ComparisonSide,
    candidate_side: ComparisonSide,
) -> list[MetricComparisonRow]:
    """Benchmark final table용 metric / Reference / Candidate / Improvement / Winner."""
    specs = [
        ("Mean", "mae"),
        ("Median", "median"),
        ("RMSE", "rmse"),
        ("Std", "std"),
        ("P90", "p90"),
        ("P95", "p95"),
        ("P99", "p99"),
        ("Max", "max"),
        ("Edge RMS", "edge_rms"),
        ("Straightness", "straightness"),
    ]
    rows: list[MetricComparisonRow] = []
    for label, key in specs:
        ref = _metric_value(reference_side, key)
        cand = _metric_value(candidate_side, key)
        improvement = _improvement_pct(ref, cand)
        if ref is None or cand is None:
            winner = "N/A"
        elif abs(ref - cand) < 1e-9:
            winner = "Tie"
        elif cand < ref:
            winner = candidate_side.label
        else:
            winner = reference_side.label
        rows.append(MetricComparisonRow(label, ref, cand, improvement, winner))
    return rows


def _fmt_final_value(value: float | None, unit: str = "") -> str:
    if value is None:
        return "N/A"
    suffix = f" {unit}" if unit else ""
    return f"{value:.3f}{suffix}"


def _fmt_final_pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.1f}%"


def _fmt_final_ci(value: float | None, low: float | None, high: float | None, unit: str = "") -> str:
    if value is None:
        return "N/A"
    suffix = f" {unit}" if unit else ""
    if low is None or high is None:
        return f"{value:.3f}{suffix}"
    return f"{value:.3f} [{low:.3f}, {high:.3f}]{suffix}"


def build_final_benchmark_rows(
    reference_side: ComparisonSide,
    candidate_side: ComparisonSide,
    metric_rows: list[MetricComparisonRow],
    worst_case_rows: list[WorstCaseRow],
    benchmark_validation_rows: list[BenchmarkValidationRow],
    statistical_tests: list[StatisticalTestResult],
    bootstrap_comparison: BenchmarkBootstrapResult | None,
) -> list[FinalBenchmarkRow]:
    """Document-style Metric / Reference / Candidate / Improvement / Winner table."""
    rows: list[FinalBenchmarkRow] = []

    for metric in metric_rows:
        rows.append(FinalBenchmarkRow(
            metric=metric.metric,
            reference=_fmt_final_value(metric.reference_value, "px"),
            candidate=_fmt_final_value(metric.candidate_value, "px"),
            improvement=_fmt_final_pct(metric.improvement_pct),
            winner=metric.winner,
        ))

    common_frame_ids = set(reference_side.per_frame_error) & set(candidate_side.per_frame_error)
    candidate_frame_wins = sum(
        candidate_side.per_frame_error[fid] < reference_side.per_frame_error[fid]
        for fid in common_frame_ids
    )
    reference_frame_wins = sum(
        reference_side.per_frame_error[fid] < candidate_side.per_frame_error[fid]
        for fid in common_frame_ids
    )
    rows.append(FinalBenchmarkRow(
        metric="Frame wins",
        reference=str(reference_frame_wins),
        candidate=str(candidate_frame_wins),
        improvement="N/A",
        winner=(
            candidate_side.label if candidate_frame_wins > reference_frame_wins
            else reference_side.label if reference_frame_wins > candidate_frame_wins
            else "Tie"
        ),
    ))

    for row in worst_case_rows:
        rows.append(FinalBenchmarkRow(
            metric=row.category,
            reference=f"{_fmt_final_value(row.reference_value, 'px')} @ {row.reference_location}",
            candidate=f"{_fmt_final_value(row.candidate_value, 'px')} @ {row.candidate_location}",
            improvement=_fmt_final_pct(row.improvement_pct),
            winner=row.winner,
        ))

    for row in benchmark_validation_rows:
        rows.append(FinalBenchmarkRow(
            metric=f"{row.name} validation RMSE",
            reference=_fmt_final_value(row.reference_validation_rms_mean, "px"),
            candidate=_fmt_final_value(row.candidate_validation_rms_mean, "px"),
            improvement=_fmt_final_pct(row.improvement_pct),
            winner=_winner_for_lower_is_better(
                row.reference_validation_rms_mean,
                row.candidate_validation_rms_mean,
                reference_side.label,
                candidate_side.label,
            ),
        ))
        if row.reference_train_validation_gap is not None or row.candidate_train_validation_gap is not None:
            rows.append(FinalBenchmarkRow(
                metric=f"{row.name} train-validation gap",
                reference=_fmt_final_value(row.reference_train_validation_gap, "px"),
                candidate=_fmt_final_value(row.candidate_train_validation_gap, "px"),
                improvement="N/A",
                winner=_winner_for_lower_is_better(
                    abs(row.reference_train_validation_gap) if row.reference_train_validation_gap is not None else None,
                    abs(row.candidate_train_validation_gap) if row.candidate_train_validation_gap is not None else None,
                    reference_side.label,
                    candidate_side.label,
                ),
            ))

    if bootstrap_comparison is not None:
        rows.append(FinalBenchmarkRow(
            metric="Bootstrap RMSE 95% CI",
            reference=_fmt_final_ci(
                bootstrap_comparison.reference_rmse,
                bootstrap_comparison.reference_rmse_ci_low,
                bootstrap_comparison.reference_rmse_ci_high,
                "px",
            ),
            candidate=_fmt_final_ci(
                bootstrap_comparison.candidate_rmse,
                bootstrap_comparison.candidate_rmse_ci_low,
                bootstrap_comparison.candidate_rmse_ci_high,
                "px",
            ),
            improvement=_fmt_final_ci(
                bootstrap_comparison.improvement_pct,
                bootstrap_comparison.improvement_ci_low,
                bootstrap_comparison.improvement_ci_high,
                "%",
            ),
            winner=_winner_for_lower_is_better(
                bootstrap_comparison.reference_rmse,
                bootstrap_comparison.candidate_rmse,
                reference_side.label,
                candidate_side.label,
            ),
        ))
        prob = bootstrap_comparison.probability_candidate_better
        rows.append(FinalBenchmarkRow(
            metric="P(Candidate Error < Reference Error)",
            reference="N/A",
            candidate="N/A" if prob is None else f"{prob * 100.0:.2f}%",
            improvement="N/A",
            winner=(
                "N/A" if prob is None
                else candidate_side.label if prob > 0.5
                else reference_side.label if prob < 0.5
                else "Tie"
            ),
        ))

    for test in statistical_tests:
        rows.append(FinalBenchmarkRow(
            metric=f"{test.test_name} p-value",
            reference="N/A",
            candidate="N/A",
            improvement="N/A" if test.p_value is None else f"p={test.p_value:.6g}",
            winner=(
                "N/A" if test.p_value is None or test.mean_diff is None
                else candidate_side.label if test.p_value < 0.05 and test.mean_diff < 0
                else reference_side.label if test.p_value < 0.05 and test.mean_diff > 0
                else "Not significant"
            ),
        ))

    return rows


def _add_decision_vote(
    decision: WinnerDecision,
    *,
    candidate_better: bool | None,
    weight: float,
    evidence: str,
) -> None:
    if candidate_better is None:
        return
    if candidate_better:
        decision.candidate_score += weight
    else:
        decision.reference_score += weight
    decision.evidence.append(evidence)


def _decision_from_numeric_metric(
    reference: float | None,
    candidate: float | None,
    *,
    min_relative_gap: float = 0.01,
) -> bool | None:
    if reference is None or candidate is None:
        return None
    if not np.isfinite(reference) or not np.isfinite(candidate):
        return None
    scale = max(abs(reference), abs(candidate), 1e-12)
    if abs(reference - candidate) / scale < min_relative_gap:
        return None
    return candidate < reference


def _occupied_grid_cells(
    side: ComparisonSide,
    image_size: tuple[int, int] | None,
    *,
    rows: int = 3,
    cols: int = 3,
) -> int | None:
    if image_size is None or not side.point_error_details:
        return None
    w, h = image_size
    if w <= 0 or h <= 0:
        return None
    occupied: set[tuple[int, int]] = set()
    cell_w = w / cols
    cell_h = h / rows
    for detail in side.point_error_details:
        col = int(np.clip(detail.x // cell_w, 0, cols - 1))
        row = int(np.clip(detail.y // cell_h, 0, rows - 1))
        occupied.add((row, col))
    return len(occupied)


def _data_quality_warnings(
    reference_side: ComparisonSide,
    candidate_side: ComparisonSide,
    common_frame_count: int,
    image_size: tuple[int, int] | None,
    *,
    min_frames: int = 3,
    min_points: int = 30,
    min_coverage_cells_3x3: int = 3,
) -> list[str]:
    warnings: list[str] = []
    if common_frame_count < min_frames:
        warnings.append(
            f"Insufficient paired frames: {common_frame_count} < {min_frames}."
        )

    if reference_side.point_error_details or candidate_side.point_error_details:
        ref_points = len(reference_side.point_error_details)
        cand_points = len(candidate_side.point_error_details)
        if min(ref_points, cand_points) < min_points:
            warnings.append(
                f"Insufficient paired corner evidence: Reference {ref_points}, "
                f"Candidate {cand_points}, required >= {min_points}."
            )

    ref_cells = _occupied_grid_cells(reference_side, image_size)
    cand_cells = _occupied_grid_cells(candidate_side, image_size)
    if ref_cells is not None and cand_cells is not None:
        if min(ref_cells, cand_cells) < min_coverage_cells_3x3:
            warnings.append(
                f"Insufficient spatial coverage: Reference {ref_cells}/9 cells, "
                f"Candidate {cand_cells}/9 cells, required >= {min_coverage_cells_3x3}."
            )
    return warnings


def build_winner_decision(
    reference_side: ComparisonSide,
    candidate_side: ComparisonSide,
    metric_rows: list[MetricComparisonRow],
    worst_case_rows: list[WorstCaseRow],
    benchmark_validation_rows: list[BenchmarkValidationRow],
    statistical_tests: list[StatisticalTestResult],
    bootstrap_comparison: BenchmarkBootstrapResult | None,
    *,
    image_size: tuple[int, int] | None = None,
    min_score_margin: float = 1.0,
    min_data_quality_frames: int = 3,
    min_data_quality_points: int = 30,
    min_data_quality_coverage_cells_3x3: int = 3,
) -> WinnerDecision:
    """Candidate/Reference/Inconclusive/Insufficient Evidence state machine."""
    decision = WinnerDecision()
    if not (reference_side.success and candidate_side.success):
        decision.warnings.append("One or both calibrations failed fixed-parameter validation.")
        return decision

    common_ids = set(reference_side.per_frame_error) & set(candidate_side.per_frame_error)
    data_quality_warnings = _data_quality_warnings(
        reference_side,
        candidate_side,
        len(common_ids),
        image_size,
        min_frames=min_data_quality_frames,
        min_points=min_data_quality_points,
        min_coverage_cells_3x3=min_data_quality_coverage_cells_3x3,
    )
    if data_quality_warnings:
        decision.warnings.extend(data_quality_warnings)
        return decision
    decision.data_quality_ok = True

    weights = {
        "RMSE": 2.0,
        "P95": 2.0,
        "Edge RMS": 1.5,
        "Straightness": 1.0,
    }
    for row in metric_rows:
        if row.metric not in weights:
            continue
        candidate_better = _decision_from_numeric_metric(row.reference_value, row.candidate_value)
        if candidate_better is not None:
            _add_decision_vote(
                decision,
                candidate_better=candidate_better,
                weight=weights[row.metric],
                evidence=f"{row.metric}: {row.winner}",
            )

    candidate_frame_wins = sum(
        candidate_side.per_frame_error[fid] < reference_side.per_frame_error[fid]
        for fid in common_ids
    )
    reference_frame_wins = sum(
        reference_side.per_frame_error[fid] < candidate_side.per_frame_error[fid]
        for fid in common_ids
    )
    if abs(candidate_frame_wins - reference_frame_wins) >= max(1, int(0.1 * len(common_ids))):
        _add_decision_vote(
            decision,
            candidate_better=candidate_frame_wins > reference_frame_wins,
            weight=1.0,
            evidence=f"Frame wins: Candidate {candidate_frame_wins}, Reference {reference_frame_wins}",
        )

    for row in benchmark_validation_rows:
        if row.name.startswith("K-fold") or row.name == "Hold-out" or row.name.startswith("Generalization"):
            candidate_better = _decision_from_numeric_metric(
                row.reference_validation_rms_mean,
                row.candidate_validation_rms_mean,
                min_relative_gap=0.005,
            )
            if candidate_better is not None:
                _add_decision_vote(
                    decision,
                    candidate_better=candidate_better,
                    weight=1.25 if row.name.startswith("K-fold") else 1.0,
                    evidence=f"{row.name}: validation RMSE",
                )

    for row in worst_case_rows:
        candidate_better = _decision_from_numeric_metric(row.reference_value, row.candidate_value)
        if candidate_better is not None:
            _add_decision_vote(
                decision,
                candidate_better=candidate_better,
                weight=0.75,
                evidence=f"{row.category}: {row.winner}",
            )

    if bootstrap_comparison and bootstrap_comparison.probability_candidate_better is not None:
        prob = bootstrap_comparison.probability_candidate_better
        if prob >= 0.8 or prob <= 0.2:
            _add_decision_vote(
                decision,
                candidate_better=prob > 0.5,
                weight=2.0,
                evidence=f"Bootstrap probability Candidate better = {prob:.3f}",
            )
        elif 0.45 <= prob <= 0.55:
            decision.warnings.append("Bootstrap probability is near 50%; practical winner is weak.")

    for test in statistical_tests:
        if test.p_value is None or test.mean_diff is None:
            continue
        if test.p_value < 0.05:
            _add_decision_vote(
                decision,
                candidate_better=test.mean_diff < 0,
                weight=1.5,
                evidence=f"{test.test_name}: p={test.p_value:.3g}",
            )

    total_score = decision.candidate_score + decision.reference_score
    decision.score_margin = abs(decision.candidate_score - decision.reference_score)
    if total_score <= 0:
        decision.status = "Insufficient Evidence"
        decision.warnings.append("No decisive metric evidence was available.")
    elif decision.score_margin < min_score_margin:
        decision.status = "Inconclusive"
        decision.warnings.append("Decision scores are too close for a preferred winner.")
    elif decision.candidate_score > decision.reference_score:
        decision.status = "Candidate Preferred"
    else:
        decision.status = "Reference Preferred"
    return decision


def _verdict_with_decision(decision: WinnerDecision, base_verdict: str) -> str:
    return f"{decision.status}: {base_verdict}"


def _winner_for_lower_is_better(
    reference_value: float | None,
    candidate_value: float | None,
    reference_label: str,
    candidate_label: str,
) -> str:
    if reference_value is None or candidate_value is None:
        return "N/A"
    if abs(reference_value - candidate_value) < 1e-9:
        return "Tie"
    return candidate_label if candidate_value < reference_value else reference_label


def _worst_frame(side: ComparisonSide) -> tuple[str, float] | None:
    valid = [
        (frame_id, float(error))
        for frame_id, error in side.per_frame_error.items()
        if np.isfinite(error)
    ]
    return max(valid, key=lambda item: item[1]) if valid else None


def _worst_region_from_grid(grid: SpatialComparisonGrid, *, reference: bool) -> tuple[str, float] | None:
    attr = "reference_max" if reference else "candidate_max"
    candidates = [
        (f"row {cell.row}, col {cell.col}", getattr(cell, attr))
        for cell in grid.cells
        if getattr(cell, attr) is not None and np.isfinite(getattr(cell, attr))
    ]
    if not candidates:
        return None
    location, value = max(candidates, key=lambda item: item[1])
    return location, float(value)


def _worst_corner(side: ComparisonSide) -> tuple[str, float] | None:
    if not side.point_error_details:
        return None
    detail = max(side.point_error_details, key=lambda item: item.error)
    return (
        f"{detail.frame_id} corner {detail.corner_index} ({detail.x:.1f}, {detail.y:.1f})",
        float(detail.error),
    )


def build_worst_case_rows(
    reference_side: ComparisonSide,
    candidate_side: ComparisonSide,
    spatial_comparisons: dict[str, SpatialComparisonGrid],
) -> list[WorstCaseRow]:
    rows: list[WorstCaseRow] = []

    def add_row(category: str, ref_item, cand_item) -> None:
        ref_loc, ref_val = ref_item if ref_item is not None else ("N/A", None)
        cand_loc, cand_val = cand_item if cand_item is not None else ("N/A", None)
        rows.append(WorstCaseRow(
            category=category,
            reference_location=ref_loc,
            reference_value=ref_val,
            candidate_location=cand_loc,
            candidate_value=cand_val,
            improvement_pct=_improvement_pct(ref_val, cand_val),
            winner=_winner_for_lower_is_better(ref_val, cand_val, reference_side.label, candidate_side.label),
        ))

    add_row("Worst image", _worst_frame(reference_side), _worst_frame(candidate_side))
    grid = spatial_comparisons.get("5x5") or spatial_comparisons.get("3x3")
    add_row(
        "Worst region",
        _worst_region_from_grid(grid, reference=True) if grid is not None else None,
        _worst_region_from_grid(grid, reference=False) if grid is not None else None,
    )
    add_row("Worst corner", _worst_corner(reference_side), _worst_corner(candidate_side))
    return rows


def _side_point_error_array(side: ComparisonSide) -> np.ndarray:
    if side.point_error_details:
        values = [detail.error for detail in side.point_error_details]
    elif side.point_errors_xy:
        values = [error for _x, _y, error in side.point_errors_xy]
    elif side.residual_stats and side.residual_stats.sample_residuals:
        values = side.residual_stats.sample_residuals
    else:
        values = list(side.per_frame_error.values())
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def build_error_distribution_comparison(
    reference_side: ComparisonSide,
    candidate_side: ComparisonSide,
    *,
    num_bins: int = 20,
) -> ErrorDistributionComparison:
    """Reference/Candidate histogram and CDF on shared bin edges."""
    reference = _side_point_error_array(reference_side)
    candidate = _side_point_error_array(candidate_side)
    result = ErrorDistributionComparison(
        num_reference_points=int(reference.size),
        num_candidate_points=int(candidate.size),
        reference_median=(
            reference_side.residual_stats.median if reference_side.residual_stats else
            (float(np.median(reference)) if reference.size else None)
        ),
        candidate_median=(
            candidate_side.residual_stats.median if candidate_side.residual_stats else
            (float(np.median(candidate)) if candidate.size else None)
        ),
        reference_p95=(
            reference_side.residual_stats.p95 if reference_side.residual_stats else
            (float(np.percentile(reference, 95)) if reference.size else None)
        ),
        candidate_p95=(
            candidate_side.residual_stats.p95 if candidate_side.residual_stats else
            (float(np.percentile(candidate, 95)) if candidate.size else None)
        ),
    )
    if reference.size == 0 and candidate.size == 0:
        return result

    combined = np.concatenate([reference, candidate])
    lo = float(np.min(combined))
    hi = float(np.max(combined))
    if abs(hi - lo) < 1e-12:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, max(1, int(num_bins)) + 1)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cand_counts, _ = np.histogram(candidate, bins=edges)
    ref_total = max(int(np.sum(ref_counts)), 1)
    cand_total = max(int(np.sum(cand_counts)), 1)
    ref_cdf = np.cumsum(ref_counts) / ref_total
    cand_cdf = np.cumsum(cand_counts) / cand_total

    result.bins = [
        ErrorDistributionBin(
            bin_start=float(edges[i]),
            bin_end=float(edges[i + 1]),
            reference_count=int(ref_counts[i]),
            candidate_count=int(cand_counts[i]),
            reference_density=float(ref_counts[i] / ref_total),
            candidate_density=float(cand_counts[i] / cand_total),
            reference_cdf=float(ref_cdf[i]),
            candidate_cdf=float(cand_cdf[i]),
            cdf_delta=float(cand_cdf[i] - ref_cdf[i]),
        )
        for i in range(len(edges) - 1)
    ]
    return result


def _relative_abs_diff_pct(reference_value: float | None, candidate_value: float | None) -> float | None:
    if reference_value is None or candidate_value is None:
        return None
    if not np.isfinite(reference_value) or not np.isfinite(candidate_value):
        return None
    if abs(reference_value) < 1e-12:
        return None
    return float(abs(candidate_value - reference_value) / abs(reference_value) * 100.0)


def _parameter_diff_row(
    name: str,
    reference_value: float | None,
    candidate_value: float | None,
    unit: str = "",
) -> ParameterDiffRow:
    absolute = None
    if reference_value is not None and candidate_value is not None:
        if np.isfinite(reference_value) and np.isfinite(candidate_value):
            absolute = float(candidate_value - reference_value)
    return ParameterDiffRow(
        name=name,
        reference_value=None if reference_value is None else float(reference_value),
        candidate_value=None if candidate_value is None else float(candidate_value),
        absolute_diff=absolute,
        relative_diff_pct=_relative_abs_diff_pct(reference_value, candidate_value),
        unit=unit,
    )


def _distortion_label(index: int, model: CameraModelType | None) -> str:
    if model == CameraModelType.FISHEYE:
        return f"k{index + 1}"
    labels = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
    return labels[index] if index < len(labels) else f"d{index + 1}"


def build_parameter_diff_rows(
    reference_side: ComparisonSide,
    candidate_side: ComparisonSide,
) -> list[ParameterDiffRow]:
    """fx/fy/cx/cy and distortion coefficient absolute/relative differences."""
    rows: list[ParameterDiffRow] = []
    ref_K = reference_side.camera_matrix
    cand_K = candidate_side.camera_matrix
    if ref_K is not None and cand_K is not None:
        specs = [
            ("fx", (0, 0)),
            ("fy", (1, 1)),
            ("cx", (0, 2)),
            ("cy", (1, 2)),
        ]
        for name, (r, c) in specs:
            rows.append(_parameter_diff_row(name, ref_K[r, c], cand_K[r, c], "px"))

    ref_D = np.asarray(reference_side.distortion, dtype=float).reshape(-1) if reference_side.distortion is not None else np.array([])
    cand_D = np.asarray(candidate_side.distortion, dtype=float).reshape(-1) if candidate_side.distortion is not None else np.array([])
    n = max(ref_D.size, cand_D.size)
    for i in range(n):
        ref = float(ref_D[i]) if i < ref_D.size else None
        cand = float(cand_D[i]) if i < cand_D.size else None
        rows.append(_parameter_diff_row(_distortion_label(i, reference_side.model_name), ref, cand))
    return rows


def _estimate_fov_deg(
    model: CameraModelType | None,
    camera_matrix: np.ndarray | None,
    image_size: tuple[int, int],
) -> tuple[float | None, float | None, float | None]:
    if camera_matrix is None:
        return None, None, None
    w, h = image_size
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    if fx <= 0 or fy <= 0 or w <= 0 or h <= 0:
        return None, None, None
    if model == CameraModelType.FISHEYE:
        hfov = math.degrees(w / fx)
        vfov = math.degrees(h / fy)
        dfov = math.degrees(math.hypot(w / fx, h / fy))
    else:
        hfov = math.degrees(2.0 * math.atan((w / 2.0) / fx))
        vfov = math.degrees(2.0 * math.atan((h / 2.0) / fy))
        dfov = math.degrees(2.0 * math.atan(math.hypot(w / 2.0, h / 2.0) / math.sqrt(fx * fy)))
    return hfov, vfov, dfov


def build_fov_diff_rows(
    reference_side: ComparisonSide,
    candidate_side: ComparisonSide,
    image_size: tuple[int, int],
) -> list[ParameterDiffRow]:
    ref_h, ref_v, ref_d = _estimate_fov_deg(reference_side.model_name, reference_side.camera_matrix, image_size)
    cand_h, cand_v, cand_d = _estimate_fov_deg(candidate_side.model_name, candidate_side.camera_matrix, image_size)
    return [
        _parameter_diff_row("HFOV", ref_h, cand_h, "deg"),
        _parameter_diff_row("VFOV", ref_v, cand_v, "deg"),
        _parameter_diff_row("DFOV", ref_d, cand_d, "deg"),
    ]


def _parameter_labels_for_side(side: ComparisonSide) -> list[str]:
    labels = ["fx", "fy", "cx", "cy"]
    if side.distortion is not None and side.model_name is not None:
        labels.extend(distortion_coeff_labels(side.model_name, int(np.asarray(side.distortion).size)))
    return labels


def _parameter_vector_for_side(side: ComparisonSide) -> np.ndarray:
    if side.camera_matrix is None:
        return np.array([], dtype=np.float64)
    values = [
        float(side.camera_matrix[0, 0]),
        float(side.camera_matrix[1, 1]),
        float(side.camera_matrix[0, 2]),
        float(side.camera_matrix[1, 2]),
    ]
    if side.distortion is not None:
        values.extend(np.asarray(side.distortion, dtype=np.float64).reshape(-1).tolist())
    return np.asarray(values, dtype=np.float64)


def _unpack_side_params(side: ComparisonSide, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    K = np.asarray(side.camera_matrix, dtype=np.float64).copy()
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = params[:4]
    if side.distortion is None:
        D = np.zeros((0, 1), dtype=np.float64)
    else:
        D = np.asarray(params[4:], dtype=np.float64).reshape(np.asarray(side.distortion).shape)
    return K, D


def _project_with_pose(
    model: CameraModelType,
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    rv = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tv = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    if model == CameraModelType.FISHEYE:
        projected, _ = cv2.fisheye.projectPoints(obj, rv, tv, K, D.reshape(-1, 1))
    else:
        projected, _ = cv2.projectPoints(obj, rv, tv, K, D.reshape(-1, 1))
    return np.asarray(projected, dtype=np.float64).reshape(-1, 2)


def _collect_fixed_pose_observations(
    dataset: Dataset,
    test_ids: list[str],
    side: ComparisonSide,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if side.camera_matrix is None or side.distortion is None or side.model_name is None:
        return []
    observations = []
    for frame in _subset_dataset(dataset, test_ids).enabled_frames:
        det = frame.detection
        if det is None or not det.success:
            continue
        obj = np.asarray(det.object_points, dtype=np.float64)
        img = np.asarray(det.corners, dtype=np.float64)
        try:
            if side.model_name == CameraModelType.FISHEYE:
                ok, rvec, tvec = cv2.fisheye.solvePnP(obj, img, side.camera_matrix, side.distortion)
            else:
                ok, rvec, tvec = cv2.solvePnP(obj, img, side.camera_matrix, side.distortion)
            if not ok:
                continue
        except cv2.error:
            continue
        observations.append((obj, img.reshape(-1, 2), rvec, tvec))
    return observations


def _diagnostic_residual_vector(
    observations: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    side: ComparisonSide,
    params: np.ndarray,
) -> np.ndarray:
    if side.model_name is None:
        return np.array([], dtype=np.float64)
    K, D = _unpack_side_params(side, params)
    chunks: list[np.ndarray] = []
    for obj, img, rvec, tvec in observations:
        projected = _project_with_pose(side.model_name, obj, rvec, tvec, K, D)
        chunks.append((img - projected).reshape(-1))
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)


def _stability_from_covariance(value: float, std: float | None) -> float | None:
    if std is None or not np.isfinite(std):
        return None
    scale = max(abs(float(value)), 1e-9)
    return float(max(0.0, min(100.0, 100.0 * (1.0 - abs(std) / scale))))


def _correlation_from_covariance(covariance: np.ndarray) -> tuple[list[list[float]], float | None, list[tuple[int, int, float]]]:
    if covariance.size == 0:
        return [], None, []
    diag = np.diag(covariance)
    std = np.sqrt(np.clip(diag, 0.0, None))
    denom = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.divide(covariance, denom, out=np.zeros_like(covariance, dtype=np.float64), where=denom > 0)
    corr = np.clip(corr, -1.0, 1.0)
    corr[~np.isfinite(corr)] = 0.0
    for i in range(min(corr.shape)):
        corr[i, i] = 1.0

    pairs: list[tuple[int, int, float]] = []
    max_abs = None
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[1]):
            value = float(corr[i, j])
            abs_value = abs(value)
            max_abs = abs_value if max_abs is None else max(max_abs, abs_value)
            pairs.append((i, j, value))
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    matrix = [[float(v) for v in row] for row in corr.tolist()]
    return matrix, max_abs, pairs[:5]


def build_parameter_diagnostics(
    dataset: Dataset,
    test_ids: list[str],
    side: ComparisonSide,
    *,
    relative_step: float = 1e-6,
) -> BenchmarkParameterDiagnostics:
    """Approximate covariance/stability/sensitivity for one fixed calibration side."""
    labels = _parameter_labels_for_side(side)
    diagnostics = BenchmarkParameterDiagnostics(side_label=side.label, parameter_labels=labels)
    params = _parameter_vector_for_side(side)
    observations = _collect_fixed_pose_observations(dataset, test_ids, side)
    if params.size == 0 or not observations:
        diagnostics.warnings.append("Parameter diagnostics could not be computed: no usable observations.")
        return diagnostics

    base_residual = _diagnostic_residual_vector(observations, side, params)
    diagnostics.n_points = int(base_residual.size // 2)
    if base_residual.size == 0:
        diagnostics.warnings.append("Parameter diagnostics could not be computed: empty residual vector.")
        return diagnostics

    J = np.empty((base_residual.size, params.size), dtype=np.float64)
    diagnostics.jacobian_rows = int(J.shape[0])
    diagnostics.jacobian_cols = int(J.shape[1])
    base_rmse = _rmse(base_residual)
    sensitivity_rows: list[ParameterSensitivityRow] = []
    for i, value in enumerate(params):
        step = max(abs(float(value)) * relative_step, relative_step)
        plus = params.copy()
        minus = params.copy()
        plus[i] += step
        minus[i] -= step
        plus_residual = _diagnostic_residual_vector(observations, side, plus)
        minus_residual = _diagnostic_residual_vector(observations, side, minus)
        J[:, i] = (plus_residual - minus_residual) / (2.0 * step)
        plus_rmse = _rmse(plus_residual)
        rmse_delta = float(plus_rmse - base_rmse)
        sensitivity_rows.append(ParameterSensitivityRow(
            parameter=labels[i] if i < len(labels) else f"p{i}",
            value=float(value),
            perturbation=float(step),
            rmse_delta=rmse_delta,
            sensitivity_per_unit=float(rmse_delta / step),
        ))

    try:
        singular = np.linalg.svd(J, compute_uv=False)
        diagnostics.singular_values = [float(v) for v in singular.tolist()]
        diagnostics.min_singular_value = float(np.min(singular)) if singular.size else None
        diagnostics.max_singular_value = float(np.max(singular)) if singular.size else None
        if singular.size:
            tol = float(max(J.shape) * np.finfo(np.float64).eps * singular[0])
            diagnostics.rank = int(np.sum(singular > tol))
        diagnostics.condition_number = (
            float(singular[0] / singular[-1]) if singular.size and singular[-1] > 0 else math.inf
        )
        dof = max(base_residual.size - params.size, 1)
        sigma2 = float(np.sum(base_residual ** 2) / dof)
        covariance = sigma2 * np.linalg.pinv(J.T @ J)
    except np.linalg.LinAlgError:
        diagnostics.warnings.append("Covariance matrix could not be computed: singular Jacobian.")
        covariance = np.full((params.size, params.size), np.nan, dtype=np.float64)

    covariance = np.asarray(covariance, dtype=np.float64)
    diagnostics.covariance_matrix = [
        [float(v) if np.isfinite(v) else float("nan") for v in row]
        for row in covariance.tolist()
    ]
    corr, max_abs_corr, top_corr_indices = _correlation_from_covariance(covariance)
    diagnostics.correlation_matrix = corr
    diagnostics.max_abs_correlation = max_abs_corr
    diagnostics.top_correlations = [
        (
            labels[i] if i < len(labels) else f"p{i}",
            labels[j] if j < len(labels) else f"p{j}",
            value,
        )
        for i, j, value in top_corr_indices
    ]

    diag = np.diag(covariance) if covariance.size else np.array([], dtype=float)
    stability_rows: list[ParameterStabilityRow] = []
    column_norms = np.linalg.norm(J, axis=0) if J.size else np.array([], dtype=float)
    max_col_norm = float(np.max(column_norms)) if column_norms.size else 0.0
    for i, value in enumerate(params):
        variance = float(diag[i]) if i < diag.size and np.isfinite(diag[i]) and diag[i] >= 0 else None
        std = math.sqrt(variance) if variance is not None else None
        label = labels[i] if i < len(labels) else f"p{i}"
        if max_col_norm > 0 and i < column_norms.size and column_norms[i] < max_col_norm * 1e-4:
            diagnostics.weak_parameters.append(label)
        elif _stability_from_covariance(float(value), std) is not None and _stability_from_covariance(float(value), std) < 25.0:
            diagnostics.weak_parameters.append(label)
        stability_rows.append(ParameterStabilityRow(
            parameter=label,
            value=float(value),
            std=std,
            ci_low=float(value - 1.96 * std) if std is not None else None,
            ci_high=float(value + 1.96 * std) if std is not None else None,
            stability_score=_stability_from_covariance(float(value), std),
        ))
    diagnostics.weak_parameters = sorted(set(diagnostics.weak_parameters), key=diagnostics.weak_parameters.index)
    if diagnostics.rank is not None and diagnostics.rank < diagnostics.jacobian_cols:
        diagnostics.warnings.append(
            f"Jacobian rank deficient: {diagnostics.rank}/{diagnostics.jacobian_cols}."
        )
    if diagnostics.max_abs_correlation is not None and diagnostics.max_abs_correlation >= 0.95:
        diagnostics.warnings.append(
            f"High parameter correlation detected: {diagnostics.max_abs_correlation:.3f}."
        )
    if diagnostics.weak_parameters:
        diagnostics.warnings.append(
            "Weakly observable parameters: " + ", ".join(diagnostics.weak_parameters)
        )
    diagnostics.stability_rows = stability_rows
    diagnostics.sensitivity_rows = sensitivity_rows
    return diagnostics


def build_parameter_diagnostics_pair(
    dataset: Dataset,
    test_ids: list[str],
    reference_side: ComparisonSide,
    candidate_side: ComparisonSide,
) -> dict[str, BenchmarkParameterDiagnostics]:
    return {
        "reference": build_parameter_diagnostics(dataset, test_ids, reference_side),
        "candidate": build_parameter_diagnostics(dataset, test_ids, candidate_side),
    }


def _usable_frame_ids(dataset: Dataset) -> list[str]:
    return [
        f.image_info.image_id
        for f in dataset.enabled_frames
        if f.detection and f.detection.success and f.detection.num_corners >= 4
    ]


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def _benchmark_validation_row_from_splits(
    name: str,
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    reference_side: ComparisonSide,
    candidate_side: ComparisonSide,
    splits: list[tuple[list[str], list[str]]],
) -> BenchmarkValidationRow:
    ref_train: list[float] = []
    ref_val: list[float] = []
    cand_train: list[float] = []
    cand_val: list[float] = []

    for train_ids, validation_ids in splits:
        if not validation_ids:
            continue
        ref_validation = _evaluate_side(
            dataset, camera_config, pattern_config, validation_ids,
            reference_side.camera_matrix, reference_side.distortion, reference_side.model_name, reference_side.label,
        )
        cand_validation = _evaluate_side(
            dataset, camera_config, pattern_config, validation_ids,
            candidate_side.camera_matrix, candidate_side.distortion, candidate_side.model_name, candidate_side.label,
        )
        if ref_validation.success and ref_validation.test_rms is not None:
            ref_val.append(ref_validation.test_rms)
        if cand_validation.success and cand_validation.test_rms is not None:
            cand_val.append(cand_validation.test_rms)

        if train_ids:
            ref_train_side = _evaluate_side(
                dataset, camera_config, pattern_config, train_ids,
                reference_side.camera_matrix, reference_side.distortion, reference_side.model_name, reference_side.label,
            )
            cand_train_side = _evaluate_side(
                dataset, camera_config, pattern_config, train_ids,
                candidate_side.camera_matrix, candidate_side.distortion, candidate_side.model_name, candidate_side.label,
            )
            if ref_train_side.success and ref_train_side.test_rms is not None:
                ref_train.append(ref_train_side.test_rms)
            if cand_train_side.success and cand_train_side.test_rms is not None:
                cand_train.append(cand_train_side.test_rms)

    ref_train_mean, _ref_train_std = _mean_std(ref_train)
    cand_train_mean, _cand_train_std = _mean_std(cand_train)
    ref_val_mean, ref_val_std = _mean_std(ref_val)
    cand_val_mean, cand_val_std = _mean_std(cand_val)
    ref_gap = (
        ref_val_mean - ref_train_mean
        if ref_val_mean is not None and ref_train_mean is not None
        else None
    )
    cand_gap = (
        cand_val_mean - cand_train_mean
        if cand_val_mean is not None and cand_train_mean is not None
        else None
    )
    return BenchmarkValidationRow(
        name=name,
        num_splits=len(ref_val) if len(ref_val) == len(cand_val) else max(len(ref_val), len(cand_val)),
        reference_train_rms_mean=ref_train_mean,
        reference_validation_rms_mean=ref_val_mean,
        reference_validation_rms_std=ref_val_std,
        reference_train_validation_gap=ref_gap,
        candidate_train_rms_mean=cand_train_mean,
        candidate_validation_rms_mean=cand_val_mean,
        candidate_validation_rms_std=cand_val_std,
        candidate_train_validation_gap=cand_gap,
        improvement_pct=_improvement_pct(ref_val_mean, cand_val_mean),
    )


def build_benchmark_validation_rows(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    reference_side: ComparisonSide,
    candidate_side: ComparisonSide,
    holdout_validation_ids: list[str],
    *,
    kfold: int = 5,
    seed: int = 42,
    generalization_datasets: dict[str, Dataset] | None = None,
) -> list[BenchmarkValidationRow]:
    """Reference/Candidate 전용 Hold-out, K-fold, Generalization comparison.

    모든 평가는 이미 주어진 K/D를 고정하고 pose만 다시 추정한다. 따라서 기존
    모델 선택용 k-fold처럼 fold마다 intrinsic을 재학습하지 않는다.
    """
    if not (reference_side.success and candidate_side.success):
        return []

    usable_ids = _usable_frame_ids(dataset)
    holdout_set = set(holdout_validation_ids)
    holdout_train_ids = [fid for fid in usable_ids if fid not in holdout_set]
    rows = [
        _benchmark_validation_row_from_splits(
            "Hold-out",
            dataset,
            camera_config,
            pattern_config,
            reference_side,
            candidate_side,
            [(holdout_train_ids, holdout_validation_ids)],
        )
    ]

    if kfold >= 2:
        from calibration.kfold import split_k_folds

        folds = split_k_folds(dataset, camera_config, k=kfold, seed=seed)
        splits = []
        for i, validation_ids in enumerate(folds):
            train_ids = [fid for j, fold_ids in enumerate(folds) if j != i for fid in fold_ids]
            if validation_ids:
                splits.append((train_ids, validation_ids))
        if splits:
            rows.append(
                _benchmark_validation_row_from_splits(
                    f"K-fold ({kfold})",
                    dataset,
                    camera_config,
                    pattern_config,
                    reference_side,
                    candidate_side,
                    splits,
                )
            )

    for target_name, target_dataset in (generalization_datasets or {}).items():
        validation_ids = _usable_frame_ids(target_dataset)
        rows.append(
            _benchmark_validation_row_from_splits(
                f"Generalization: {target_name}",
                target_dataset,
                camera_config,
                pattern_config,
                reference_side,
                candidate_side,
                [([], validation_ids)],
            )
        )
    return rows


def _normal_two_sided_p(z: float) -> float:
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _paired_t_test(diffs: np.ndarray) -> tuple[float | None, float | None]:
    n = diffs.size
    if n < 2:
        return None, None
    mean = float(np.mean(diffs))
    std = float(np.std(diffs, ddof=1))
    if std <= 0:
        return (0.0, 1.0) if abs(mean) < 1e-12 else (math.copysign(math.inf, mean), 0.0)

    try:
        from scipy import stats  # type: ignore

        result = stats.ttest_1samp(diffs, popmean=0.0, nan_policy="omit")
        return float(result.statistic), float(result.pvalue)
    except Exception:  # noqa: BLE001 - SciPy is optional
        t_stat = mean / (std / math.sqrt(n))
        return float(t_stat), _normal_two_sided_p(t_stat)


def _wilcoxon_signed_rank_test(diffs: np.ndarray) -> tuple[float | None, float | None, float | None]:
    nonzero = diffs[np.abs(diffs) > 1e-12]
    n = nonzero.size
    if n == 0:
        return 0.0, 1.0, 0.0

    try:
        from scipy import stats  # type: ignore

        result = stats.wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    except Exception:  # noqa: BLE001 - SciPy is optional
        ranks = _rankdata_average(np.abs(nonzero))
        w_plus = float(np.sum(ranks[nonzero > 0]))
        w_minus = float(np.sum(ranks[nonzero < 0]))
        statistic = min(w_plus, w_minus)
        expected = n * (n + 1) / 4.0
        variance = n * (n + 1) * (2 * n + 1) / 24.0
        if variance <= 0:
            p_value = 1.0
        else:
            z = (statistic - expected) / math.sqrt(variance)
            p_value = _normal_two_sided_p(z)

    ranks = _rankdata_average(np.abs(nonzero))
    rank_total = float(np.sum(ranks))
    if rank_total <= 0:
        rank_biserial = 0.0
    else:
        w_plus = float(np.sum(ranks[nonzero > 0]))
        w_minus = float(np.sum(ranks[nonzero < 0]))
        rank_biserial = float((w_plus - w_minus) / rank_total)
    return statistic, p_value, rank_biserial


def _interpret_statistical_result(p_value: float | None, effect_size: float | None) -> str:
    if p_value is None:
        return "Not enough paired frames."
    direction = "Candidate lower" if effect_size is not None and effect_size < 0 else "Reference lower"
    if effect_size is not None and abs(effect_size) < 1e-12:
        direction = "No directional difference"
    strength = "statistically significant" if p_value < 0.05 else "not statistically significant"
    return f"{direction}; {strength} (alpha=0.05)."


def build_statistical_tests(
    reference_side: ComparisonSide,
    candidate_side: ComparisonSide,
) -> list[StatisticalTestResult]:
    """Paired tests over common per-frame RMS errors.

    Uses frame-level paired samples because each value is computed on the same
    image for Reference and Candidate. Point-level residuals are intentionally
    not treated as independent samples here.
    """
    common_ids = sorted(set(reference_side.per_frame_error) & set(candidate_side.per_frame_error))
    pairs = [
        (reference_side.per_frame_error[fid], candidate_side.per_frame_error[fid])
        for fid in common_ids
        if np.isfinite(reference_side.per_frame_error[fid])
        and np.isfinite(candidate_side.per_frame_error[fid])
    ]
    if not pairs:
        return [
            StatisticalTestResult(test_name="Paired t-test"),
            StatisticalTestResult(test_name="Wilcoxon signed-rank"),
        ]

    reference = np.asarray([p[0] for p in pairs], dtype=float)
    candidate = np.asarray([p[1] for p in pairs], dtype=float)
    diffs = candidate - reference
    n = int(diffs.size)
    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))

    t_stat, t_p = _paired_t_test(diffs)
    std = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    cohens_dz = None if n < 2 or std <= 0 else float(mean_diff / std)
    w_stat, w_p, rank_biserial = _wilcoxon_signed_rank_test(diffs)

    return [
        StatisticalTestResult(
            test_name="Paired t-test",
            statistic=t_stat,
            p_value=t_p,
            effect_size=cohens_dz,
            effect_size_name="Cohen dz",
            n_pairs=n,
            mean_diff=mean_diff,
            median_diff=median_diff,
            interpretation=_interpret_statistical_result(t_p, mean_diff),
        ),
        StatisticalTestResult(
            test_name="Wilcoxon signed-rank",
            statistic=w_stat,
            p_value=w_p,
            effect_size=rank_biserial,
            effect_size_name="rank-biserial r",
            n_pairs=n,
            mean_diff=mean_diff,
            median_diff=median_diff,
            interpretation=_interpret_statistical_result(w_p, rank_biserial),
        ),
    ]


def _paired_frame_arrays(
    reference_side: ComparisonSide,
    candidate_side: ComparisonSide,
) -> tuple[np.ndarray, np.ndarray]:
    common_ids = sorted(set(reference_side.per_frame_error) & set(candidate_side.per_frame_error))
    pairs = [
        (reference_side.per_frame_error[fid], candidate_side.per_frame_error[fid])
        for fid in common_ids
        if np.isfinite(reference_side.per_frame_error[fid])
        and np.isfinite(candidate_side.per_frame_error[fid])
    ]
    if not pairs:
        return np.array([], dtype=float), np.array([], dtype=float)
    return (
        np.asarray([p[0] for p in pairs], dtype=float),
        np.asarray([p[1] for p in pairs], dtype=float),
    )


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values ** 2)))


def _percentile_ci(values: np.ndarray, confidence_level: float) -> tuple[float | None, float | None]:
    if values.size == 0:
        return None, None
    alpha = (1.0 - confidence_level) / 2.0
    return (
        float(np.percentile(values, alpha * 100.0)),
        float(np.percentile(values, (1.0 - alpha) * 100.0)),
    )


def build_bootstrap_comparison(
    reference_side: ComparisonSide,
    candidate_side: ComparisonSide,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> BenchmarkBootstrapResult:
    """Bootstrap P(Candidate Error < Reference Error), RMSE CI, improvement CI."""
    reference, candidate = _paired_frame_arrays(reference_side, candidate_side)
    n = int(reference.size)
    result = BenchmarkBootstrapResult(
        n_pairs=n,
        n_bootstrap=max(0, int(n_bootstrap)),
        confidence_level=float(confidence_level),
    )
    if n == 0 or n_bootstrap <= 0:
        return result

    result.reference_rmse = _rmse(reference)
    result.candidate_rmse = _rmse(candidate)
    result.improvement_pct = _improvement_pct(result.reference_rmse, result.candidate_rmse)

    rng = np.random.default_rng(seed)
    ref_samples = np.empty(n_bootstrap, dtype=float)
    cand_samples = np.empty(n_bootstrap, dtype=float)
    improvement_samples = np.empty(n_bootstrap, dtype=float)
    better_count = 0
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        ref_rmse = _rmse(reference[idx])
        cand_rmse = _rmse(candidate[idx])
        ref_samples[i] = ref_rmse
        cand_samples[i] = cand_rmse
        improvement = _improvement_pct(ref_rmse, cand_rmse)
        improvement_samples[i] = np.nan if improvement is None else improvement
        if cand_rmse < ref_rmse:
            better_count += 1

    result.probability_candidate_better = float(better_count / n_bootstrap)
    result.reference_rmse_ci_low, result.reference_rmse_ci_high = _percentile_ci(
        ref_samples, confidence_level
    )
    result.candidate_rmse_ci_low, result.candidate_rmse_ci_high = _percentile_ci(
        cand_samples, confidence_level
    )
    valid_improvements = improvement_samples[np.isfinite(improvement_samples)]
    result.improvement_ci_low, result.improvement_ci_high = _percentile_ci(
        valid_improvements, confidence_level
    )
    return result


def _cell_stats(values: list[float]) -> tuple[int, float | None, float | None, float | None, float | None]:
    if not values:
        return 0, None, None, None, None
    arr = np.asarray(values, dtype=float)
    return (
        int(arr.size),
        float(np.mean(arr)),
        float(np.sqrt(np.mean(arr ** 2))),
        float(np.percentile(arr, 95)),
        float(np.max(arr)),
    )


def build_spatial_comparison_grid(
    reference_side: ComparisonSide,
    candidate_side: ComparisonSide,
    image_size: tuple[int, int],
    *,
    rows: int,
    cols: int,
) -> SpatialComparisonGrid:
    """이미지를 rows x cols로 나눠 cell별 Reference/Candidate error를 비교한다."""
    w, h = image_size
    ref_buckets: dict[tuple[int, int], list[float]] = {(r, c): [] for r in range(rows) for c in range(cols)}
    cand_buckets: dict[tuple[int, int], list[float]] = {(r, c): [] for r in range(rows) for c in range(cols)}

    def bucket(points: list[tuple[float, float, float]], target: dict[tuple[int, int], list[float]]) -> None:
        if w <= 0 or h <= 0:
            return
        cell_w, cell_h = w / cols, h / rows
        for x, y, error in points:
            col = int(np.clip(x // cell_w, 0, cols - 1))
            row = int(np.clip(y // cell_h, 0, rows - 1))
            target[(row, col)].append(float(error))

    bucket(reference_side.point_errors_xy, ref_buckets)
    bucket(candidate_side.point_errors_xy, cand_buckets)

    cells: list[SpatialMetricCell] = []
    for r in range(rows):
        for c in range(cols):
            ref_n, ref_mean, ref_rmse, ref_p95, ref_max = _cell_stats(ref_buckets[(r, c)])
            cand_n, cand_mean, cand_rmse, cand_p95, cand_max = _cell_stats(cand_buckets[(r, c)])
            cells.append(SpatialMetricCell(
                row=r,
                col=c,
                num_reference_points=ref_n,
                num_candidate_points=cand_n,
                reference_mean=ref_mean,
                candidate_mean=cand_mean,
                improvement_mean_pct=_improvement_pct(ref_mean, cand_mean),
                reference_rmse=ref_rmse,
                candidate_rmse=cand_rmse,
                improvement_rmse_pct=_improvement_pct(ref_rmse, cand_rmse),
                reference_p95=ref_p95,
                candidate_p95=cand_p95,
                improvement_p95_pct=_improvement_pct(ref_p95, cand_p95),
                reference_max=ref_max,
                candidate_max=cand_max,
                improvement_max_pct=_improvement_pct(ref_max, cand_max),
            ))
    return SpatialComparisonGrid(rows=rows, cols=cols, cells=cells)


def build_spatial_comparisons(
    reference_side: ComparisonSide,
    candidate_side: ComparisonSide,
    image_size: tuple[int, int],
) -> dict[str, SpatialComparisonGrid]:
    return {
        "3x3": build_spatial_comparison_grid(reference_side, candidate_side, image_size, rows=3, cols=3),
        "5x5": build_spatial_comparison_grid(reference_side, candidate_side, image_size, rows=5, cols=5),
    }


def build_residual_heatmap_comparison(
    reference_side: ComparisonSide,
    candidate_side: ComparisonSide,
    image_size: tuple[int, int],
    *,
    rows: int = 20,
    cols: int = 20,
    metric: str = "rmse",
) -> ResidualHeatmapComparison:
    """Heatmap 렌더링용 Reference/Candidate/Difference grid를 만든다.

    difference_value = candidate_value - reference_value 이므로:
      - 양수: Candidate residual이 더 큼
      - 음수: Candidate residual이 더 작음
    """
    spatial = build_spatial_comparison_grid(reference_side, candidate_side, image_size, rows=rows, cols=cols)
    cells: list[HeatmapCell] = []
    ref_values: list[float] = []
    cand_values: list[float] = []
    diff_abs_values: list[float] = []

    ref_attr = f"reference_{metric}"
    cand_attr = f"candidate_{metric}"
    for cell in spatial.cells:
        ref = getattr(cell, ref_attr, None)
        cand = getattr(cell, cand_attr, None)
        diff = None if ref is None or cand is None else float(cand - ref)
        if ref is not None:
            ref_values.append(float(ref))
        if cand is not None:
            cand_values.append(float(cand))
        if diff is not None:
            diff_abs_values.append(abs(diff))
        cells.append(HeatmapCell(
            row=cell.row,
            col=cell.col,
            reference_value=ref,
            candidate_value=cand,
            difference_value=diff,
            num_reference_points=cell.num_reference_points,
            num_candidate_points=cell.num_candidate_points,
        ))

    return ResidualHeatmapComparison(
        rows=rows,
        cols=cols,
        metric=metric,
        cells=cells,
        reference_max=max(ref_values) if ref_values else None,
        candidate_max=max(cand_values) if cand_values else None,
        difference_abs_max=max(diff_abs_values) if diff_abs_values else None,
    )


def build_residual_heatmaps(
    reference_side: ComparisonSide,
    candidate_side: ComparisonSide,
    image_size: tuple[int, int],
) -> dict[str, ResidualHeatmapComparison]:
    return {
        "rmse_20x20": build_residual_heatmap_comparison(
            reference_side, candidate_side, image_size, rows=20, cols=20, metric="rmse"
        ),
        "p95_20x20": build_residual_heatmap_comparison(
            reference_side, candidate_side, image_size, rows=20, cols=20, metric="p95"
        ),
    }


def build_radial_comparison_profile(
    reference_side: ComparisonSide,
    candidate_side: ComparisonSide,
    image_size: tuple[int, int],
    boundaries: list[float],
    labels: list[str],
) -> RadialComparisonProfile:
    """Normalized radius band별 Reference/Candidate error를 비교한다."""
    w, h = image_size
    cx, cy = w / 2.0, h / 2.0
    max_radius = float(np.hypot(cx, cy))
    if max_radius <= 0:
        return RadialComparisonProfile(max_radius_px=max_radius)

    ref_buckets: list[list[float]] = [[] for _ in labels]
    cand_buckets: list[list[float]] = [[] for _ in labels]

    def bucket(points: list[tuple[float, float, float]], target: list[list[float]]) -> None:
        for x, y, error in points:
            rn = float(np.hypot(x - cx, y - cy) / max_radius)
            rn = min(max(rn, 0.0), 1.0)
            for i, _label in enumerate(labels):
                lo, hi = boundaries[i], boundaries[i + 1]
                in_band = (lo <= rn < hi) if i < len(labels) - 1 else (lo <= rn <= hi)
                if in_band:
                    target[i].append(float(error))
                    break

    bucket(reference_side.point_errors_xy, ref_buckets)
    bucket(candidate_side.point_errors_xy, cand_buckets)

    bands: list[RadialMetricBand] = []
    for i, label in enumerate(labels):
        ref_n, ref_mean, ref_rmse, ref_p95, ref_max = _cell_stats(ref_buckets[i])
        cand_n, cand_mean, cand_rmse, cand_p95, cand_max = _cell_stats(cand_buckets[i])
        bands.append(RadialMetricBand(
            label=label,
            radius_min_norm=float(boundaries[i]),
            radius_max_norm=float(boundaries[i + 1]),
            num_reference_points=ref_n,
            num_candidate_points=cand_n,
            reference_mean=ref_mean,
            candidate_mean=cand_mean,
            improvement_mean_pct=_improvement_pct(ref_mean, cand_mean),
            reference_rmse=ref_rmse,
            candidate_rmse=cand_rmse,
            improvement_rmse_pct=_improvement_pct(ref_rmse, cand_rmse),
            reference_p95=ref_p95,
            candidate_p95=cand_p95,
            improvement_p95_pct=_improvement_pct(ref_p95, cand_p95),
            reference_max=ref_max,
            candidate_max=cand_max,
            improvement_max_pct=_improvement_pct(ref_max, cand_max),
        ))
    return RadialComparisonProfile(bands=bands, max_radius_px=max_radius)


def build_radial_comparisons(
    reference_side: ComparisonSide,
    candidate_side: ComparisonSide,
    image_size: tuple[int, int],
) -> dict[str, RadialComparisonProfile]:
    return {
        "quartiles": build_radial_comparison_profile(
            reference_side,
            candidate_side,
            image_size,
            [0.0, 0.25, 0.50, 0.75, 1.0],
            ["0-25%", "25-50%", "50-75%", "75-100%"],
        ),
        "bands": build_radial_comparison_profile(
            reference_side,
            candidate_side,
            image_size,
            [0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0],
            ["Center", "Inner", "Middle", "Outer", "Edge", "Corner"],
        ),
    }


# ---------------------------------------------------------------------------
# 한쪽 평가
# ---------------------------------------------------------------------------

def _test_reprojection_errors_with_points(
    test_frames: list,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model: CameraModelType,
) -> tuple[
    dict[str, float],
    list[str],
    list[float],
    list[tuple[float, float, float]],
    list[PointErrorDetail],
]:
    """_test_reprojection_errors()와 같은 평가를 하되 spatial 비교용
    (observed x, observed y, residual magnitude)도 함께 반환한다.
    """
    errors: dict[str, float] = {}
    failed: list[str] = []
    point_errors: list[float] = []
    point_errors_xy: list[tuple[float, float, float]] = []
    point_error_details: list[PointErrorDetail] = []
    is_fisheye = model == CameraModelType.FISHEYE

    for frame in test_frames:
        det = frame.detection
        obj = det.object_points
        img = det.corners
        frame_id = frame.image_info.image_id

        try:
            if is_fisheye:
                obj64 = obj.astype(np.float64)
                img64 = img.astype(np.float64)
                ok, rvec, tvec = cv2.fisheye.solvePnP(obj64, img64, camera_matrix, distortion)
                if not ok:
                    raise cv2.error("fisheye.solvePnP returned False")
                projected, _ = cv2.fisheye.projectPoints(obj64, rvec, tvec, camera_matrix, distortion)
                detected = img64.reshape(-1, 2)
            else:
                ok, rvec, tvec = cv2.solvePnP(obj, img, camera_matrix, distortion)
                if not ok:
                    raise cv2.error("solvePnP returned False")
                projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, distortion)
                detected = img.reshape(-1, 2)
        except cv2.error:
            failed.append(frame_id)
            continue

        projected = projected.reshape(-1, 2)
        diff = detected - projected
        per_point = np.hypot(diff[:, 0], diff[:, 1])
        rms = float(np.sqrt(np.mean(per_point ** 2)))
        errors[frame_id] = rms
        point_errors.extend(per_point.tolist())
        point_errors_xy.extend(
            (float(x), float(y), float(error))
            for (x, y), error in zip(detected, per_point)
        )
        point_error_details.extend(
            PointErrorDetail(
                frame_id=frame_id,
                corner_index=int(i),
                x=float(x),
                y=float(y),
                error=float(error),
            )
            for i, ((x, y), error) in enumerate(zip(detected, per_point))
        )
        frame.reprojection_error = rms

    return errors, failed, point_errors, point_errors_xy, point_error_details


def _evaluate_side(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    test_ids: list[str],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model: CameraModelType,
    label: str,
) -> ComparisonSide:
    subset = _subset_dataset(dataset, test_ids)
    frames = subset.enabled_frames
    if not frames:
        return ComparisonSide(
            label=label, model_name=model, camera_matrix=camera_matrix, distortion=distortion,
            success=False, error_message="비교할 프레임이 없습니다.",
        )

    per_frame_error, failed, point_errors, point_errors_xy, point_error_details = _test_reprojection_errors_with_points(
        frames, camera_matrix, distortion, model
    )
    if not per_frame_error:
        return ComparisonSide(
            label=label, model_name=model, camera_matrix=camera_matrix, distortion=distortion,
            success=False,
            error_message=(
                "모든 프레임에서 pose 추정(solvePnP)이 실패했습니다 - "
                "카메라 모델 종류(Pinhole/Extended/Fisheye)나 파라미터가 "
                "이 촬영과 맞는지 다시 확인해 주세요."
            ),
            failed_frame_ids=failed,
        )

    test_rms = float(np.sqrt(np.mean(np.array(list(per_frame_error.values())) ** 2)))
    residual_stats = compute_residual_stats(point_errors)
    image_size = (camera_config.width, camera_config.height)
    scored_frames = [f for f in frames if f.image_info.image_id in per_frame_error]
    regional = compute_regional_error(scored_frames, per_frame_error, image_size)
    edge_rms = regional_edge_average(regional)
    straightness, _n_lines = compute_straightness_residual(
        scored_frames, pattern_config, camera_matrix, distortion, model,
    )

    return ComparisonSide(
        label=label, model_name=model, camera_matrix=camera_matrix, distortion=distortion,
        test_rms=test_rms, residual_stats=residual_stats, edge_rms=edge_rms, straightness_residual=straightness,
        per_frame_error=per_frame_error, point_errors_xy=point_errors_xy,
        point_error_details=point_error_details,
        failed_frame_ids=failed, success=True,
    )


# ---------------------------------------------------------------------------
# 메인 함수
# ---------------------------------------------------------------------------

def compare_with_external_params(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    my_model: CameraModelType,
    my_validation: ValidationResult,
    external: ExternalCameraParams,
    benchmark_kfold: int = 5,
    benchmark_bootstrap: int = 1000,
    generalization_datasets: dict[str, Dataset] | None = None,
) -> ExternalComparisonResult:
    """my_validation이 이미 확보해 둔 "학습에 전혀 안 쓰인" test 프레임
    집합에서, 내 파라미터(같은 test 분할로 다시 학습한 것 - my_validation.
    test_rms를 만들어낸 바로 그 절차)와 외부 파라미터를 완전히 동일한
    방식으로 재평가해 비교한다.
    """
    caveats: list[str] = []
    my_label = f"내 결과 ({_MODEL_LABELS.get(my_model, my_model)})"

    test_ids = my_validation.test_frame_ids
    if not test_ids:
        return ExternalComparisonResult(
            mine=ComparisonSide(label=my_label, success=False, error_message="Hold-out 검증에 test 프레임이 없습니다."),
            external=ComparisonSide(label=external.label, success=False, error_message="비교 기준(test 프레임)이 없습니다."),
            verdict=(
                "데이터셋이 너무 작아 Hold-out에 쓸 test 프레임이 없습니다. "
                "이미지를 더 모아서(최소 수십 장 권장) 다시 시도해 주세요."
            ),
        )

    mine_train_result = refit_on_train_split(
        dataset, camera_config, my_model, my_validation.train_frame_ids,
    )
    if not mine_train_result.success:
        return ExternalComparisonResult(
            mine=ComparisonSide(label=my_label, success=False, error_message=mine_train_result.error_message),
            external=ComparisonSide(label=external.label, success=False, error_message="비교할 수 없습니다."),
            verdict=f"내 파라미터를 test 분할 기준으로 다시 학습하지 못했습니다: {mine_train_result.error_message}",
        )

    mine_side = _evaluate_side(
        dataset, camera_config, pattern_config, test_ids,
        mine_train_result.camera_matrix, mine_train_result.distortion, my_model, my_label,
    )
    external_side = _evaluate_side(
        dataset, camera_config, pattern_config, test_ids,
        external.camera_matrix, external.distortion, external.model_name, external.label,
    )

    # 정합성 자체 점검 - 방금 재학습한 test_rms는 my_validation.test_rms와
    # (거의) 같아야 정상이다. 조용히 다르면 안 되고, 다르면 그 자체를
    # 사용자에게 밝힌다 (원인 예: 그 사이에 데이터셋/설정이 바뀌었음).
    if (
        mine_side.success and my_validation.test_rms is not None
        and abs(mine_side.test_rms - my_validation.test_rms) > 1e-2
    ):
        caveats.append(
            "내부 정합성 확인: 방금 재계산한 Test RMS"
            f"({mine_side.test_rms:.3f}px)가 이전 Hold-out 결과"
            f"({my_validation.test_rms:.3f}px)와 다릅니다 - 그 사이에 데이터셋/설정이 "
            "바뀌었을 수 있으니 [캘리브레이션 실행]을 다시 눌러 최신 상태로 비교해 보세요."
        )

    if external.source_note:
        caveats.append(f"'{external.label}' 출처: {external.source_note}")
    caveats.append(
        "Parameter similarity ≠ Calibration accuracy: fx/fy/cx/cy/k 차이가 작아도 "
        "hold-out residual, edge, radial, straightness 지표가 더 중요합니다."
    )

    external_standard = StandardCalibration(
        label=external.label,
        camera_matrix=external.camera_matrix,
        distortion=external.distortion,
        model_name=external.model_name,
        distortion_model=external.distortion_model,
        width=external.width or camera_config.width,
        height=external.height or camera_config.height,
        source_format="external_compare_input",
    )
    for issue in validate_single_calibration(
        external_standard,
        side=external.label,
        validation_image_size=(camera_config.width, camera_config.height),
    ):
        level = "ERROR" if issue.severity.value == "error" else "WARNING"
        caveats.append(f"[{level}] {issue.message}")

    mine_win = external_win = tie = 0
    common_ids = set(mine_side.per_frame_error) & set(external_side.per_frame_error)
    for fid in common_ids:
        m, e = mine_side.per_frame_error[fid], external_side.per_frame_error[fid]
        if abs(m - e) < 1e-9:
            tie += 1
        elif m < e:
            mine_win += 1
        else:
            external_win += 1

    base_verdict = _build_verdict(mine_side, external_side, mine_win, external_win, len(common_ids))
    image_size = (camera_config.width, camera_config.height)
    spatial_comparisons = build_spatial_comparisons(external_side, mine_side, image_size)
    metric_rows = build_metric_comparison_rows(external_side, mine_side)
    worst_case_rows = build_worst_case_rows(external_side, mine_side, spatial_comparisons)
    benchmark_validation_rows = build_benchmark_validation_rows(
        dataset,
        camera_config,
        pattern_config,
        external_side,
        mine_side,
        test_ids,
        kfold=benchmark_kfold,
        generalization_datasets=generalization_datasets,
    )
    statistical_tests = build_statistical_tests(external_side, mine_side)
    bootstrap_comparison = build_bootstrap_comparison(
        external_side, mine_side, n_bootstrap=benchmark_bootstrap
    )
    winner_decision = build_winner_decision(
        external_side,
        mine_side,
        metric_rows,
        worst_case_rows,
        benchmark_validation_rows,
        statistical_tests,
        bootstrap_comparison,
        image_size=image_size,
    )

    return ExternalComparisonResult(
        mine=mine_side,
        external=external_side,
        num_common_frames=len(common_ids),
        mine_win_count=mine_win,
        external_win_count=external_win,
        tie_count=tie,
        verdict=_verdict_with_decision(winner_decision, base_verdict),
        winner_decision=winner_decision,
        caveats=caveats,
        metric_rows=metric_rows,
        final_benchmark_rows=build_final_benchmark_rows(
            external_side,
            mine_side,
            metric_rows,
            worst_case_rows,
            benchmark_validation_rows,
            statistical_tests,
            bootstrap_comparison,
        ),
        worst_case_rows=worst_case_rows,
        error_distribution=build_error_distribution_comparison(external_side, mine_side),
        spatial_comparisons=spatial_comparisons,
        residual_heatmaps=build_residual_heatmaps(external_side, mine_side, image_size),
        radial_comparisons=build_radial_comparisons(external_side, mine_side, image_size),
        parameter_diff_rows=build_parameter_diff_rows(external_side, mine_side),
        fov_diff_rows=build_fov_diff_rows(external_side, mine_side, image_size),
        benchmark_validation_rows=benchmark_validation_rows,
        statistical_tests=statistical_tests,
        bootstrap_comparison=bootstrap_comparison,
        parameter_diagnostics=build_parameter_diagnostics_pair(
            dataset, test_ids, external_side, mine_side
        ),
    )


def _compatibility_caveats(reference: StandardCalibration, candidate: StandardCalibration, camera_config: CameraConfig) -> list[str]:
    report = validate_calibration_pair_compatibility(
        reference,
        candidate,
        validation_image_size=(camera_config.width, camera_config.height),
    )
    caveats = [f"Compatibility status: {report.status}"]
    for issue in report.issues:
        level = "ERROR" if issue.severity.value == "error" else "WARNING"
        caveats.append(f"[{level}] {issue.side}: {issue.message}")
    caveats.append(
        "Parameter similarity ≠ Calibration accuracy: fx/fy/cx/cy/k 차이가 작아도 "
        "hold-out residual, edge, radial, straightness 지표가 더 중요합니다."
    )
    return caveats


_MIN_INDEPENDENT_BENCHMARK_USABLE_FRAMES = 10


def _safe_sha256(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _dataset_paths_and_hashes(dataset: Dataset) -> tuple[set[str], set[str]]:
    paths: set[str] = set()
    hashes: set[str] = set()
    for frame in dataset.frames:
        path = frame.image_info.path
        if path:
            try:
                paths.add(str(Path(path).resolve()))
            except OSError:
                paths.add(str(Path(path).absolute()))
            digest = _safe_sha256(path)
            if digest:
                hashes.add(digest)
    return paths, hashes


def _benchmark_overlap_count(calibration_dataset: Dataset, benchmark_dataset: Dataset) -> int:
    calibration_paths, calibration_hashes = _dataset_paths_and_hashes(calibration_dataset)
    count = 0
    seen: set[str] = set()
    for frame in benchmark_dataset.frames:
        path = frame.image_info.path
        resolved = ""
        if path:
            try:
                resolved = str(Path(path).resolve())
            except OSError:
                resolved = str(Path(path).absolute())
        digest = _safe_sha256(path) if path else None
        key = digest or resolved or frame.image_info.image_id
        if key in seen:
            continue
        if (resolved and resolved in calibration_paths) or (digest and digest in calibration_hashes):
            count += 1
            seen.add(key)
    return count


def _select_evaluation_dataset(
    calibration_dataset: Dataset,
    internal_test_frame_ids: list[str],
    independent_benchmark_dataset: Dataset | None,
    evaluation_mode: str,
    caveats: list[str],
) -> tuple[Dataset, list[str], str, str, str, int, int, int]:
    """Return dataset/test ids/source/confidence/status/counts for pair comparison.

    Independent Benchmark is an optional confidence upgrade. It never blocks the
    internal hold-out path; invalid benchmark data falls back to hold-out.
    """
    benchmark_count = len(independent_benchmark_dataset.frames) if independent_benchmark_dataset else 0
    benchmark_ids = _usable_frame_ids(independent_benchmark_dataset) if independent_benchmark_dataset else []
    benchmark_usable = len(benchmark_ids)
    overlap = (
        _benchmark_overlap_count(calibration_dataset, independent_benchmark_dataset)
        if independent_benchmark_dataset is not None
        else 0
    )

    wants_benchmark = evaluation_mode == "independent_benchmark" or (
        evaluation_mode == "auto" and independent_benchmark_dataset is not None
    )
    if wants_benchmark and independent_benchmark_dataset is not None:
        if overlap > 0:
            caveats.append(
                "Independent Benchmark Leakage Detected: "
                f"{overlap} benchmark images overlap with the calibration dataset. "
                "HIGH confidence is not allowed; falling back to Internal Hold-out."
            )
        elif benchmark_usable < _MIN_INDEPENDENT_BENCHMARK_USABLE_FRAMES:
            caveats.append(
                "Insufficient Benchmark Evidence: "
                f"{benchmark_usable} usable benchmark frames "
                f"(< {_MIN_INDEPENDENT_BENCHMARK_USABLE_FRAMES}). "
                "Falling back to Internal Hold-out."
            )
        else:
            caveats.append(
                "Evaluation Source: Independent Benchmark. Reference/Candidate K,D are fixed; "
                "only board pose is re-estimated per benchmark frame."
            )
            return (
                independent_benchmark_dataset,
                benchmark_ids,
                "independent_benchmark",
                "high",
                "ok",
                benchmark_count,
                benchmark_usable,
                overlap,
            )

    caveats.append(
        "Evaluation Source: Internal Hold-out. Reference and Candidate were compared using held-out images "
        "from the Candidate calibration acquisition session. These images were not used for parameter "
        "estimation, but they originate from the same acquisition distribution. Provide an Independent "
        "Benchmark Dataset for higher-confidence comparison."
    )
    status = "not_provided" if independent_benchmark_dataset is None else "fallback"
    return (
        calibration_dataset,
        internal_test_frame_ids,
        "internal_holdout",
        "limited",
        status,
        benchmark_count,
        benchmark_usable,
        overlap,
    )


def compare_reference_candidate_calibrations(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    reference: StandardCalibration,
    candidate: StandardCalibration,
    test_frame_ids: list[str],
    independent_benchmark_dataset: Dataset | None = None,
    evaluation_mode: str = "auto",
    benchmark_kfold: int = 5,
    benchmark_bootstrap: int = 1000,
    generalization_datasets: dict[str, Dataset] | None = None,
) -> ExternalComparisonResult:
    """Reference와 Candidate를 둘 다 외부/표준 calibration 파일로 받아
    동일한 validation frame에서 독립 비교한다.

    기존 compare_with_external_params()는 "현재 툴 결과 vs 외부 결과" 모드라서
    candidate 쪽을 train split으로 다시 학습한다. 이 함수는 문서의 benchmark
    모드처럼 두 calibration 모두 고정된 K/D로만 평가한다.

    ExternalComparisonResult의 기존 필드명 호환을 위해:
      - mine      = Candidate
      - external  = Reference
      - mine_win_count     = Candidate win count
      - external_win_count = Reference win count
    로 채운다.
    """
    caveats = _compatibility_caveats(reference, candidate, camera_config)
    if not test_frame_ids and independent_benchmark_dataset is None:
        return ExternalComparisonResult(
            mine=ComparisonSide(label=candidate.label, success=False, error_message="비교 기준(test 프레임)이 없습니다."),
            external=ComparisonSide(label=reference.label, success=False, error_message="비교 기준(test 프레임)이 없습니다."),
            verdict=(
                "Reference/Candidate 파일 비교에 사용할 Hold-out test 프레임이 없습니다. "
                "validation split을 먼저 만들거나 이미지를 더 추가해 주세요."
            ),
            caveats=caveats,
            evaluation_mode=evaluation_mode,
        )

    compatibility_has_errors = any(c.startswith("[ERROR]") for c in caveats)
    if compatibility_has_errors:
        return ExternalComparisonResult(
            mine=ComparisonSide(label=candidate.label, success=False, error_message="compatibility 검사 실패"),
            external=ComparisonSide(label=reference.label, success=False, error_message="compatibility 검사 실패"),
            verdict="Reference/Candidate calibration이 호환되지 않아 공정한 비교를 중단했습니다.",
            caveats=caveats,
            evaluation_mode=evaluation_mode,
        )

    if reference.model_name is None or candidate.model_name is None:
        return ExternalComparisonResult(
            mine=ComparisonSide(label=candidate.label, success=False, error_message="camera model 누락"),
            external=ComparisonSide(label=reference.label, success=False, error_message="camera model 누락"),
            verdict="Reference/Candidate 중 camera model이 없는 파일이 있어 비교할 수 없습니다.",
            caveats=caveats,
            evaluation_mode=evaluation_mode,
        )

    (
        evaluation_dataset,
        evaluation_frame_ids,
        evaluation_source,
        confidence,
        benchmark_status,
        benchmark_image_count,
        benchmark_usable_frames,
        benchmark_overlap_count,
    ) = _select_evaluation_dataset(
        dataset,
        test_frame_ids,
        independent_benchmark_dataset,
        evaluation_mode,
        caveats,
    )

    reference_side = _evaluate_side(
        evaluation_dataset, camera_config, pattern_config, evaluation_frame_ids,
        reference.camera_matrix, reference.distortion, reference.model_name, reference.label,
    )
    candidate_side = _evaluate_side(
        evaluation_dataset, camera_config, pattern_config, evaluation_frame_ids,
        candidate.camera_matrix, candidate.distortion, candidate.model_name, candidate.label,
    )

    candidate_win = reference_win = tie = 0
    common_ids = set(candidate_side.per_frame_error) & set(reference_side.per_frame_error)
    for fid in common_ids:
        cand = candidate_side.per_frame_error[fid]
        ref = reference_side.per_frame_error[fid]
        if abs(cand - ref) < 1e-9:
            tie += 1
        elif cand < ref:
            candidate_win += 1
        else:
            reference_win += 1

    base_verdict = _build_verdict(candidate_side, reference_side, candidate_win, reference_win, len(common_ids))
    image_size = (camera_config.width, camera_config.height)
    spatial_comparisons = build_spatial_comparisons(reference_side, candidate_side, image_size)
    metric_rows = build_metric_comparison_rows(reference_side, candidate_side)
    worst_case_rows = build_worst_case_rows(reference_side, candidate_side, spatial_comparisons)
    benchmark_validation_rows = build_benchmark_validation_rows(
        evaluation_dataset,
        camera_config,
        pattern_config,
        reference_side,
        candidate_side,
        evaluation_frame_ids,
        kfold=benchmark_kfold,
        generalization_datasets=generalization_datasets,
    )
    statistical_tests = build_statistical_tests(reference_side, candidate_side)
    bootstrap_comparison = build_bootstrap_comparison(
        reference_side, candidate_side, n_bootstrap=benchmark_bootstrap
    )
    winner_decision = build_winner_decision(
        reference_side,
        candidate_side,
        metric_rows,
        worst_case_rows,
        benchmark_validation_rows,
        statistical_tests,
        bootstrap_comparison,
        image_size=image_size,
    )
    return ExternalComparisonResult(
        mine=candidate_side,
        external=reference_side,
        num_common_frames=len(common_ids),
        mine_win_count=candidate_win,
        external_win_count=reference_win,
        tie_count=tie,
        verdict=_verdict_with_decision(winner_decision, base_verdict),
        winner_decision=winner_decision,
        caveats=caveats,
        metric_rows=metric_rows,
        final_benchmark_rows=build_final_benchmark_rows(
            reference_side,
            candidate_side,
            metric_rows,
            worst_case_rows,
            benchmark_validation_rows,
            statistical_tests,
            bootstrap_comparison,
        ),
        worst_case_rows=worst_case_rows,
        error_distribution=build_error_distribution_comparison(reference_side, candidate_side),
        spatial_comparisons=spatial_comparisons,
        residual_heatmaps=build_residual_heatmaps(reference_side, candidate_side, image_size),
        radial_comparisons=build_radial_comparisons(reference_side, candidate_side, image_size),
        parameter_diff_rows=build_parameter_diff_rows(reference_side, candidate_side),
        fov_diff_rows=build_fov_diff_rows(reference_side, candidate_side, image_size),
        benchmark_validation_rows=benchmark_validation_rows,
        statistical_tests=statistical_tests,
        bootstrap_comparison=bootstrap_comparison,
        parameter_diagnostics=build_parameter_diagnostics_pair(
            evaluation_dataset, evaluation_frame_ids, reference_side, candidate_side
        ),
        evaluation_source=evaluation_source,
        confidence=confidence,
        evaluation_mode=evaluation_mode,
        benchmark_image_count=benchmark_image_count,
        benchmark_usable_frames=benchmark_usable_frames,
        benchmark_overlap_count=benchmark_overlap_count,
        benchmark_status=benchmark_status,
    )


# ---------------------------------------------------------------------------
# 한 줄 평 (verdict)
# ---------------------------------------------------------------------------

def _build_verdict(
    mine: ComparisonSide, external: ComparisonSide,
    mine_win: int, external_win: int, n_common: int,
) -> str:
    if not mine.success or not external.success:
        broken = mine if not mine.success else external
        return f"[{broken.label}] 비교할 수 없습니다: {broken.error_message}"

    # 지표 3개를 독립적으로 비교 - 하나만 보고 판정하지 않는다
    # ("RMS가 가장 낮은 모델 = 정답 절대 금지" 원칙을 여기도 동일 적용).
    metrics = [
        ("Test RMS", mine.test_rms, external.test_rms),
        ("Edge RMS(외곽)", mine.edge_rms, external.edge_rms),
        ("Straightness(직선성)", mine.straightness_residual, external.straightness_residual),
    ]
    mine_wins, external_wins, compared = 0, 0, 0
    for _name, mv, ev in metrics:
        if mv is None or ev is None or abs(mv - ev) < 1e-9:
            continue
        compared += 1
        if mv < ev:
            mine_wins += 1
        else:
            external_wins += 1

    if compared == 0:
        headline = "두 파라미터의 지표를 비교할 수 없습니다 (유효한 값이 부족합니다)."
    elif mine_wins == compared:
        headline = f"'{mine.label}'가 비교 가능한 지표 {compared}개 전부에서 더 정확합니다."
    elif external_wins == compared:
        headline = f"'{external.label}'가 비교 가능한 지표 {compared}개 전부에서 더 정확합니다."
    elif mine_wins > external_wins:
        headline = (
            f"'{mine.label}'가 {mine_wins}/{compared}개 지표에서 우세하지만 일부는 엇갈립니다 "
            "- 아래 표에서 어느 지표가 갈렸는지 확인하세요."
        )
    elif external_wins > mine_wins:
        headline = (
            f"'{external.label}'가 {external_wins}/{compared}개 지표에서 우세하지만 일부는 엇갈립니다 "
            "- 아래 표에서 어느 지표가 갈렸는지 확인하세요."
        )
    else:
        headline = "두 파라미터의 지표가 팽팽하게 엇갈려 어느 한쪽이 명확히 낫다고 보기 어렵습니다."

    frame_total = mine_win + external_win
    if frame_total > 0 and mine_win != external_win:
        winner_label = mine.label if mine_win > external_win else external.label
        headline += (
            f" (프레임별로 보면 공통 {n_common}장 중 {winner_label}가 더 낮은 오차를 낸 프레임이 "
            f"{max(mine_win, external_win)}장으로 더 많습니다.)"
        )

    return headline

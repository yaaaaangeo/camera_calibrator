"""
camera_calibrator.calibration.residual_stats
================================================

설계 문서 11번 - Reprojection Error 지표 확장 / 12번 - Residual Distribution 분석.

지금까지는 Train RMS 하나(그리고 Mean/Max 정도)만 봤다. 이 모듈은 코너
포인트 단위 재투영 오차를 모아 표준 통계량(RMSE, MAE, Median, Std, P90,
P95, P99, Max)과 histogram/CDF/박스플롯에 필요한 값을 전부 계산한다.

    RMSE = 0.35 px
    Median = 0.21 px
    P95 = 0.72 px
    P99 = 1.43 px
    Max = 3.21 px

radial_profile.py의 collect_per_point_residuals()가 이미 "모든 코너의
재투영 오차를 모으는" 계산을 하고 있으므로, 이 모듈은 그 결과(errors 배열)
위에 통계만 얹는다 - 투영/코너 매칭 로직을 중복하지 않는다.
"""

from __future__ import annotations

import numpy as np

from calibration.models.common import compute_mad_threshold
from calibration.radial_profile import collect_per_point_residuals
from calibration.types import CameraModelType, Frame, ResidualStats

_DEFAULT_NUM_HISTOGRAM_BINS = 20
_MAX_SAMPLE_RESIDUALS = 500


def compute_residual_stats(
    errors: np.ndarray | list[float], num_histogram_bins: int = _DEFAULT_NUM_HISTOGRAM_BINS
) -> ResidualStats:
    """재투영 오차 배열(코너 포인트 단위) 하나로부터 ResidualStats를 계산하는
    순수 함수. 투영 계산과 통계 계산을 분리해두면 단위 테스트가 훨씬 쉽다 -
    실제 카메라 모델 없이도 임의의 오차 배열로 통계 공식 자체를 검증할 수 있다.
    """
    arr = np.asarray(errors, dtype=float)
    arr = arr[np.isfinite(arr)]  # NaN/Inf가 섞여 있어도 통계 전체가 죽지 않게 방어

    if arr.size == 0:
        return ResidualStats(n=0)

    outlier_threshold = compute_mad_threshold(arr.tolist())
    outlier_count = int(np.count_nonzero(arr > outlier_threshold)) if outlier_threshold > 0 else 0

    counts, edges = np.histogram(arr, bins=num_histogram_bins)

    # 설계 문서 12번 "corner별 residual" - 대표 표본만 남긴다(모듈 docstring/
    # ResidualStats.sample_residuals 참고). 표본이 전체보다 적으면 재현
    # 가능하도록 고정 시드로 뽑는다 - 매번 다른 하위집합이 나오면 UI에서
    # 새로고침할 때마다 그림이 미묘하게 달라져 혼란스럽다.
    if arr.size <= _MAX_SAMPLE_RESIDUALS:
        sample = arr
    else:
        rng = np.random.default_rng(0)
        idx = rng.choice(arr.size, size=_MAX_SAMPLE_RESIDUALS, replace=False)
        sample = arr[idx]

    return ResidualStats(
        n=int(arr.size),
        rmse=float(np.sqrt(np.mean(arr ** 2))),
        mae=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr)),
        min=float(np.min(arr)),
        q1=float(np.percentile(arr, 25)),
        q3=float(np.percentile(arr, 75)),
        p90=float(np.percentile(arr, 90)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        max=float(np.max(arr)),
        outlier_count=outlier_count,
        histogram_bin_edges=edges.tolist(),
        histogram_counts=counts.tolist(),
        sample_residuals=sorted(sample.tolist()),
    )


def compute_residual_stats_for_calibration(
    frames: list[Frame],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    image_size: tuple[int, int],
    model: CameraModelType,
    num_histogram_bins: int = _DEFAULT_NUM_HISTOGRAM_BINS,
) -> ResidualStats:
    """calibrate_pinhole/calibrate_extended_pinhole/calibrate_fisheye가
    호출하는 편의 함수 - 코너 포인트 재투영(collect_per_point_residuals)부터
    통계 계산까지 한 번에 끝낸다.
    """
    _, errors = collect_per_point_residuals(
        frames, rvecs, tvecs, camera_matrix, distortion, image_size, model
    )
    return compute_residual_stats(errors, num_histogram_bins=num_histogram_bins)


def compute_cdf(stats: ResidualStats) -> tuple[list[float], list[float]]:
    """histogram_counts의 누적합으로 CDF(edges, cumulative_fraction)를 만든다.
    Box Plot/Histogram과 달리 별도 raw 데이터 없이 이미 저장된 히스토그램만
    으로 충분히 그릴 수 있다.
    """
    if not stats.histogram_counts or stats.n == 0:
        return [], []
    cumulative = np.cumsum(stats.histogram_counts) / stats.n
    # edges는 bin 개수+1 만큼 있고, CDF 값은 각 bin의 "오른쪽 끝"에서의 누적 비율이므로
    # edges[1:]과 짝을 맞춘다 (histogram_bin_edges[0]은 항상 누적 0%).
    return stats.histogram_bin_edges[1:], cumulative.tolist()


def format_cdf(stats: ResidualStats, width: int = 30) -> str:
    """설계 문서 12번 - CDF(누적분포함수)를 터미널용 ASCII로 표시.

        CDF (Cumulative Distribution Function)
        <= 0.12px  #####                         12%
        <= 0.23px  ##############                45%
        <= 0.35px  #######################       78%
        ...
        <= 0.92px  ############################# 100%

    P90/P95/P99가 각각 어느 지점에서 곡선을 지나는지 참고선으로 같이 표시한다 -
    히스토그램(구간별 개수)과 달리 CDF는 "이 값 이하가 전체의 몇 %인가"를
    바로 보여줘서, P95/P99 같은 percentile이 실제로 어떤 의미인지 한눈에
    이어서 이해하기 좋다.
    """
    edges, cumulative = compute_cdf(stats)
    if not edges:
        return "CDF를 계산할 데이터가 없습니다."

    lines = ["CDF (Cumulative Distribution Function)"]
    for edge, frac in zip(edges, cumulative):
        bar_len = int(round(frac * width))
        lines.append(f"<= {edge:6.2f}px {'#' * bar_len}{'.' * (width - bar_len)} {frac*100:5.1f}%")

    markers = []
    for label, value in (("P90", stats.p90), ("P95", stats.p95), ("P99", stats.p99)):
        if value is not None:
            markers.append(f"{label}={value:.3f}px")
    if markers:
        lines.append("  " + "  ".join(markers))

    return "\n".join(lines)


def format_residual_stats(stats: ResidualStats) -> str:
    """설계 문서 11번 출력 형식.

        RMSE = 0.35 px
        Median = 0.21 px
        P95 = 0.72 px
        P99 = 1.43 px
        Max = 3.21 px
    """
    if stats.n == 0:
        return "Residual 통계를 계산할 데이터가 없습니다."

    lines = [
        f"Residual Distribution (n={stats.n} points)",
        f"RMSE = {stats.rmse:.3f} px",
        f"MAE = {stats.mae:.3f} px",
        f"Median = {stats.median:.3f} px",
        f"Std = {stats.std:.3f} px",
        f"Min = {stats.min:.3f} px",
        f"Q1 = {stats.q1:.3f} px",
        f"Q3 = {stats.q3:.3f} px",
        f"P90 = {stats.p90:.3f} px",
        f"P95 = {stats.p95:.3f} px",
        f"P99 = {stats.p99:.3f} px",
        f"Max = {stats.max:.3f} px",
        f"Outliers (median+3*MAD 초과) = {stats.outlier_count}개",
    ]
    return "\n".join(lines)


def format_residual_histogram(stats: ResidualStats, width: int = 30) -> str:
    """터미널용 ASCII 히스토그램."""
    if not stats.histogram_counts:
        return "히스토그램 데이터가 없습니다."
    max_count = max(stats.histogram_counts) if stats.histogram_counts else 0
    lines = ["Residual Histogram"]
    for i, count in enumerate(stats.histogram_counts):
        lo, hi = stats.histogram_bin_edges[i], stats.histogram_bin_edges[i + 1]
        bar_len = int(round((count / max_count) * width)) if max_count > 0 else 0
        lines.append(f"{lo:6.2f}-{hi:6.2f}px {'#' * bar_len} {count}")
    return "\n".join(lines)


def format_residual_boxplot(stats: ResidualStats, width: int = 50) -> str:
    """설계 문서 12번 - Box Plot. min/q1/median/q3/max 다섯 값을 ASCII
    한 줄로 그린다 (whisker는 IQR*1.5가 아니라 min/max 그대로 - outlier는
    별도로 outlier_count로 표시하므로 여기서 따로 잘라내지 않는다).

        Min                Q1      Median      Q3                  Max
        |------------------[========|==========]-------------------|
        0.004              0.123    0.219      0.335               0.923
    """
    if stats.n == 0 or stats.min is None or stats.max is None:
        return "박스플롯을 그릴 데이터가 없습니다."

    lo, q1, med, q3, hi = stats.min, stats.q1, stats.median, stats.q3, stats.max
    span = hi - lo
    if span <= 0:
        return f"Box Plot: 모든 값이 {lo:.3f}px로 동일합니다 (n={stats.n})."

    def pos(v: float) -> int:
        return int(round((v - lo) / span * (width - 1)))

    p_q1, p_med, p_q3 = pos(q1), pos(med), pos(q3)
    chars = ["-"] * width
    for i in range(p_q1, p_q3 + 1):
        chars[i] = "="
    chars[0] = "|"
    chars[-1] = "|"
    chars[p_q1] = "["
    chars[p_q3] = "]"
    chars[p_med] = "|"
    bar = "".join(chars)

    lines = [
        f"Box Plot (n={stats.n} points)",
        bar,
        f"Min={lo:.3f}  Q1={q1:.3f}  Median={med:.3f}  Q3={q3:.3f}  Max={hi:.3f}  (px)",
        f"Outliers (median+3*MAD 초과) = {stats.outlier_count}개",
    ]
    return "\n".join(lines)

"""
camera_calibrator.calibration.observability
===========================================

Jacobian/SVD 기반 calibration observability 진단.

이미 구한 rvec/tvec를 고정하고 intrinsic/distortion 파라미터만 작은 폭으로
흔들어 residual Jacobian을 수치미분한다. 이 계층의 목적은 재최적화가 아니라
"현재 데이터셋이 어떤 파라미터를 잘 구속하는가"를 보여주는 것이다.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from calibration.models.common import collect_calibration_inputs, distortion_coeff_labels
from calibration.types import (
    CalibrationResult,
    CameraModelType,
    Dataset,
    ObservabilityReport,
    ParameterCorrelation,
)


def parameter_labels_for_result(result: CalibrationResult) -> list[str]:
    labels = ["fx", "fy", "cx", "cy"]
    if result.model_name != CameraModelType.PINHOLE and result.distortion is not None:
        labels.extend(distortion_coeff_labels(result.model_name, int(result.distortion.size)))
    return labels


def _parameter_vector(result: CalibrationResult) -> np.ndarray:
    if result.camera_matrix is None:
        return np.array([], dtype=np.float64)
    values = [
        float(result.camera_matrix[0, 0]),
        float(result.camera_matrix[1, 1]),
        float(result.camera_matrix[0, 2]),
        float(result.camera_matrix[1, 2]),
    ]
    if result.model_name != CameraModelType.PINHOLE and result.distortion is not None:
        values.extend(np.asarray(result.distortion, dtype=np.float64).ravel().tolist())
    return np.asarray(values, dtype=np.float64)


def _unpack_params(
    result: CalibrationResult, params: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    K = np.asarray(result.camera_matrix, dtype=np.float64).copy()
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = params[:4]

    if result.distortion is None:
        D = np.zeros((0, 1), dtype=np.float64)
    else:
        D = np.asarray(result.distortion, dtype=np.float64).copy()
        if result.model_name != CameraModelType.PINHOLE:
            D = np.asarray(params[4:], dtype=np.float64).reshape(D.shape)
    return K, D


def _project(
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


def _collect_observations(
    result: CalibrationResult, dataset: Dataset
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    frames, object_points, image_points = collect_calibration_inputs(dataset)
    observations = []
    for obj, img, rvec, tvec in zip(object_points, image_points, result.rvecs, result.tvecs):
        observations.append((
            np.asarray(obj, dtype=np.float64),
            np.asarray(img, dtype=np.float64).reshape(-1, 2),
            np.asarray(rvec, dtype=np.float64),
            np.asarray(tvec, dtype=np.float64),
        ))
    return observations


def residual_vector(
    result: CalibrationResult,
    dataset: Dataset,
    params: np.ndarray | None = None,
) -> np.ndarray:
    """현재 파라미터에서 image_points - projected_points residual 벡터를 반환."""
    if not result.success or result.camera_matrix is None:
        return np.array([], dtype=np.float64)
    p = _parameter_vector(result) if params is None else np.asarray(params, dtype=np.float64)
    K, D = _unpack_params(result, p)
    chunks: list[np.ndarray] = []
    for obj, img, rvec, tvec in _collect_observations(result, dataset):
        projected = _project(result.model_name, obj, rvec, tvec, K, D)
        chunks.append((img - projected).reshape(-1))
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)


def compute_numeric_jacobian(
    result: CalibrationResult,
    dataset: Dataset,
    relative_step: float = 1e-6,
) -> tuple[np.ndarray, list[str]]:
    """Intrinsic/distortion 파라미터에 대한 residual Jacobian을 중앙차분으로 계산."""
    params = _parameter_vector(result)
    labels = parameter_labels_for_result(result)
    if params.size == 0:
        return np.empty((0, 0), dtype=np.float64), labels

    base = residual_vector(result, dataset, params)
    if base.size == 0:
        return np.empty((0, params.size), dtype=np.float64), labels

    J = np.empty((base.size, params.size), dtype=np.float64)
    for i, value in enumerate(params):
        step = max(abs(float(value)) * relative_step, relative_step)
        plus = params.copy()
        minus = params.copy()
        plus[i] += step
        minus[i] -= step
        J[:, i] = (residual_vector(result, dataset, plus) - residual_vector(result, dataset, minus)) / (2.0 * step)
    return J, labels


def _correlation_matrix_from_jacobian(J: np.ndarray) -> list[list[float]]:
    if J.shape[1] == 0:
        return []
    if J.shape[1] == 1 or J.shape[0] < 2:
        return [[1.0]]
    corr = np.asarray(np.corrcoef(J, rowvar=False), dtype=np.float64)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)
    return [[float(v) for v in row] for row in corr.tolist()]


def _correlations_from_matrix(
    corr: list[list[float]],
    labels: list[str],
    top_n: int,
) -> tuple[float | None, list[ParameterCorrelation]]:
    if len(corr) < 2:
        return None, []
    pairs: list[ParameterCorrelation] = []
    max_abs = 0.0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            value = float(corr[i][j])
            if not math.isfinite(value):
                continue
            max_abs = max(max_abs, abs(value))
            pairs.append(ParameterCorrelation(labels[i], labels[j], value))
    pairs.sort(key=lambda p: abs(p.correlation), reverse=True)
    return max_abs, pairs[:top_n]


def grade_observability(score: float | None) -> str:
    if score is None:
        return "POOR"
    if score >= 80.0:
        return "GOOD"
    if score >= 50.0:
        return "WARNING"
    return "POOR"


def score_observability(
    *,
    rank: int,
    jacobian_cols: int,
    condition_number: float | None,
    max_abs_correlation: float | None,
) -> float:
    """Rank, condition, correlation을 0~100 관측가능성 점수로 압축."""
    penalties: list[float] = []
    if jacobian_cols <= 0:
        return 0.0
    if rank < jacobian_cols:
        penalties.append((jacobian_cols - rank) / jacobian_cols * 100.0)
    if condition_number is None:
        penalties.append(25.0)
    elif math.isinf(condition_number):
        penalties.append(100.0)
    elif condition_number > 0:
        # 1e4 이하는 거의 감점 없음, 1e12 이상은 condition만으로 100점 감점.
        penalties.append(max(0.0, min(100.0, (math.log10(condition_number) - 4.0) / 8.0 * 100.0)))
    if max_abs_correlation is None:
        penalties.append(10.0)
    else:
        # 0.8 이하 상관은 양호, 0.99 이상은 강한 coupling.
        penalties.append(max(0.0, min(100.0, (max_abs_correlation - 0.8) / 0.19 * 100.0)))
    return round(max(0.0, 100.0 - max(penalties, default=0.0)), 1)


def compute_observability_report(
    result: CalibrationResult,
    dataset: Dataset,
    *,
    condition_warning_threshold: float = 1e8,
    correlation_warning_threshold: float = 0.98,
    top_correlation_count: int = 5,
) -> ObservabilityReport:
    """CalibrationResult에 붙일 observability 요약을 계산."""
    J, labels = compute_numeric_jacobian(result, dataset)
    report = ObservabilityReport(
        parameter_labels=labels,
        jacobian_rows=int(J.shape[0]),
        jacobian_cols=int(J.shape[1]) if J.ndim == 2 else 0,
        num_points=int(J.shape[0] // 2) if J.ndim == 2 else 0,
    )
    if J.size == 0 or J.shape[0] == 0 or J.shape[1] == 0:
        report.observability_score = 0.0
        report.observability_grade = "POOR"
        report.warnings.append("Observability could not be computed: no usable residual Jacobian.")
        return report

    singular = np.linalg.svd(J, compute_uv=False)
    tol = float(np.finfo(float).eps * max(J.shape) * singular[0]) if singular.size else 0.0
    min_sv = float(singular[-1]) if singular.size else None
    max_sv = float(singular[0]) if singular.size else None
    if min_sv is not None and min_sv > 0:
        condition = float(max_sv / min_sv)
    else:
        condition = math.inf

    corr_matrix = _correlation_matrix_from_jacobian(J)
    max_corr, top_corr = _correlations_from_matrix(corr_matrix, labels, top_correlation_count)
    report.singular_values = [float(v) for v in singular.tolist()]
    report.rank = int(np.sum(singular > tol))
    report.condition_number = condition
    report.min_singular_value = min_sv
    report.max_singular_value = max_sv
    report.max_abs_correlation = max_corr
    report.correlation_matrix = corr_matrix
    report.top_correlations = top_corr
    report.observability_score = score_observability(
        rank=report.rank,
        jacobian_cols=report.jacobian_cols,
        condition_number=report.condition_number,
        max_abs_correlation=report.max_abs_correlation,
    )
    report.observability_grade = grade_observability(report.observability_score)

    if report.rank < report.jacobian_cols:
        report.warnings.append(
            f"Jacobian rank deficient: rank {report.rank}/{report.jacobian_cols}."
        )
    if math.isinf(condition) or condition >= condition_warning_threshold:
        report.warnings.append(f"High condition number: {condition:.3g}.")
    if max_corr is not None and max_corr >= correlation_warning_threshold:
        report.warnings.append(f"Strong parameter correlation detected: {max_corr:.3f}.")
    if report.observability_grade == "POOR":
        report.warnings.append(f"Observability grade is POOR ({report.observability_score:.1f}/100).")
    return report


def attach_observability_report(result: CalibrationResult, dataset: Dataset) -> CalibrationResult:
    """성공한 결과에는 report를 채우고, 실패 결과는 그대로 둔다."""
    if result.success:
        result.observability = compute_observability_report(result, dataset)
    return result

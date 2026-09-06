"""
camera_calibrator.calibration.windshield.runtime_projector
==============================================================

Phase C-1 안정화 - Batch/LUT Windshield Runtime Projector.

`build_projector(result)`(calibration/windshield/projection.py)가 만드는
정확한(exact) point-by-point `WindshieldModel`은 절대 바꾸지 않는다 - 이
모듈은 그것을 대체하지 않고, Camera-LiDAR 프로젝션처럼 포인트/픽셀 수가
매우 많은(수만~수백만) 런타임 사용처를 위한 "빠른 근사" 계층을 별도
API(`build_runtime_projector()`)로만 추가한다.

핵심 근사와 정확도 tradeoff(반드시 `validate_runtime_projector_vs_exact()`
로 실측해서 함께 보고해야 한다 - 이 모듈 스스로 정확도를 주장하지 않는다):

  - `unproject_pixels()`(픽셀 -> 광선 방향): `WindshieldModel.unproject_pixel()`
    자체가 이미 "광선 원점은 카메라 중심에 있다"고 가정하는 근사다(각 모델의
    docstring 참고, 예: spherical.py). 이 LUT는 그 근사를 그대로 유지한 채,
    dense grid에서 미리 계산해 둔 방향을 bilinear interpolation으로 빠르게
    조회할 뿐이다 - 격자 해상도가 유일한 추가 오차 원인이다.
  - `project_points()`(3D 포인트 -> 픽셀): 진짜 `project_point()`는 3D 위치
    전체(거리 포함)에 의존하는 root-solve다 - windshield 표면이 카메라에서
    몇 cm 떨어져 있어, 가까운 포인트일수록 parallax가 커진다. 이 LUT는
    "방향만" 쓰는 근사다(k-최근접 방향의 역거리가중평균) - LiDAR 포인트처럼
    windshield-camera 간격(수 cm)보다 훨씬 먼 포인트에서는 근사 오차가
    작을 것으로 기대되지만, 가까운 포인트에서는 오차가 커질 수 있다. 실측
    없이 이 가정을 신뢰하지 않는다.

Deep learning을 쓰지 않는다 - 기존 project 전반의 robust-statistics/격자
보간 스타일(Ghost의 spatial_model.py, Residual Grid의 bilinear 등)과
동일한 성격의 결정론적 lookup + interpolation이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from calibration.windshield.base import WindshieldCalibrationResult, WindshieldModel
from calibration.windshield.baseline import BaselineWindshieldModel
from calibration.windshield.projection import build_projector

DEFAULT_LUT_ROWS = 64
DEFAULT_LUT_COLS = 64
DEFAULT_KNN = 4


class RuntimeWindshieldProjector:
    """`WindshieldModel`을 감싸는 벡터화된 LUT 기반 빠른 런타임 프로젝터.

    Exact model을 대체하지 않는다 - 생성자가 그 model 인스턴스를 그대로
    `self.exact_model`에 보관해서, 필요하면 언제든 개별 포인트를 exact
    경로로 재계산하거나(`validate_runtime_projector_vs_exact()`가 이렇게
    한다) 비교할 수 있다.
    """

    def __init__(
        self,
        model: WindshieldModel,
        image_width: float,
        image_height: float,
        *,
        lut_rows: int = DEFAULT_LUT_ROWS,
        lut_cols: int = DEFAULT_LUT_COLS,
    ) -> None:
        if lut_rows < 2 or lut_cols < 2:
            raise ValueError(f"LUT는 최소 2x2 이상이어야 합니다 (got {lut_rows}x{lut_cols}).")
        self.exact_model = model
        self.image_width = float(image_width)
        self.image_height = float(image_height)
        self.lut_rows = int(lut_rows)
        self.lut_cols = int(lut_cols)

        us = (np.arange(self.lut_cols) + 0.5) / self.lut_cols * self.image_width
        vs = (np.arange(self.lut_rows) + 0.5) / self.lut_rows * self.image_height
        grid_u, grid_v = np.meshgrid(us, vs)  # (rows, cols), init-time만 사용(런타임 쿼리 경로 아님)

        directions = np.zeros((self.lut_rows, self.lut_cols, 3), dtype=np.float64)
        for r in range(self.lut_rows):
            for c in range(self.lut_cols):
                directions[r, c] = model.unproject_pixel(float(grid_u[r, c]), float(grid_v[r, c]))

        self._grid_u = grid_u
        self._grid_v = grid_v
        self._directions = directions

        flat_dirs = directions.reshape(-1, 3)
        flat_uv = np.stack([grid_u.reshape(-1), grid_v.reshape(-1)], axis=1)
        self._flat_dirs = flat_dirs
        self._flat_uv = flat_uv
        self._kdtree = cKDTree(flat_dirs)

    def unproject_pixels(self, pixels_uv: np.ndarray) -> np.ndarray:
        """(N,2) 픽셀 -> (N,3) 정규화된 광선 방향. LUT 격자를 bilinear
        interpolation으로 조회한다(root-solve 없음, 순수 배열 연산이라
        exact 경로보다 훨씬 빠르다)."""
        pixels_uv = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
        u = pixels_uv[:, 0]
        v = pixels_uv[:, 1]

        fx = np.clip(u / self.image_width * self.lut_cols - 0.5, 0, self.lut_cols - 1)
        fy = np.clip(v / self.image_height * self.lut_rows - 0.5, 0, self.lut_rows - 1)
        x0 = np.floor(fx).astype(int)
        y0 = np.floor(fy).astype(int)
        x1 = np.clip(x0 + 1, 0, self.lut_cols - 1)
        y1 = np.clip(y0 + 1, 0, self.lut_rows - 1)
        tx = (fx - x0)[:, None]
        ty = (fy - y0)[:, None]

        d00 = self._directions[y0, x0]
        d01 = self._directions[y0, x1]
        d10 = self._directions[y1, x0]
        d11 = self._directions[y1, x1]
        top = d00 * (1 - tx) + d01 * tx
        bottom = d10 * (1 - tx) + d11 * tx
        out = top * (1 - ty) + bottom * ty

        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return out / norms

    def project_points(self, points_xyz: np.ndarray, *, k: int = DEFAULT_KNN) -> np.ndarray:
        """(N,3) 카메라 좌표 3D 포인트 -> (N,2) 픽셀(근사, 모듈 docstring
        참고). 각 포인트의 방향(원점 기준 단위벡터)만 사용해 LUT의
        k-최근접 방향들을 역거리가중평균(inverse-distance weighting)한다."""
        points_xyz = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
        norms = np.linalg.norm(points_xyz, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        directions = points_xyz / norms

        k_eff = max(1, min(k, self._flat_dirs.shape[0]))
        dists, idx = self._kdtree.query(directions, k=k_eff)
        if k_eff == 1:
            idx = idx[:, None]
            dists = dists[:, None]
        weights = 1.0 / np.maximum(dists, 1e-9)
        weights /= weights.sum(axis=1, keepdims=True)
        neighbor_uv = self._flat_uv[idx]  # (N, k, 2)
        return np.sum(neighbor_uv * weights[:, :, None], axis=1)


def project_points_exact_batch(model: WindshieldModel, points_xyz: np.ndarray) -> np.ndarray:
    """Exact 계산 결과와 100% 동일한 값을 (N,3) 배열 하나로 한 번에 받는
    "Batch(Exact)" 티어 - LUT 근사가 아니다. `BaselineWindshieldModel`은
    내부적으로 진짜 벡터화된 `project_points_batch()`를 쓰지만(cv2 호출
    자체가 다중 포인트를 한 번에 받을 수 있음), root-solve 기반의 다른
    모델(Spherical/Residual Ray/Spline/Neural)은 point-by-point 계산을
    그대로 유지한 채 이 함수 안에서 반복 호출한다(호출자 쪽 Python 루프
    오버헤드만 줄인다 - 계산 자체를 새로 벡터화하지 않는다, 각 모델의
    기존 최적화 로직을 재작성하는 것은 이번 라운드의 범위가 아니다)."""
    points_xyz = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    if isinstance(model, BaselineWindshieldModel):
        return model.project_points_batch(points_xyz)
    return np.array(
        [model.project_point(float(p[0]), float(p[1]), float(p[2])) for p in points_xyz],
        dtype=np.float64,
    ).reshape(-1, 2)


def unproject_pixels_exact_batch(model: WindshieldModel, pixels_uv: np.ndarray) -> np.ndarray:
    """`project_points_exact_batch()`의 unproject 대응 - 역시 LUT가 아니라
    exact `unproject_pixel()`을 그대로 반복 호출하는 batch wrapper다."""
    pixels_uv = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
    return np.array(
        [model.unproject_pixel(float(uv[0]), float(uv[1])) for uv in pixels_uv],
        dtype=np.float64,
    ).reshape(-1, 3)


def build_runtime_projector(
    result: WindshieldCalibrationResult,
    image_width: float,
    image_height: float,
    *,
    lut_rows: int = DEFAULT_LUT_ROWS,
    lut_cols: int = DEFAULT_LUT_COLS,
) -> RuntimeWindshieldProjector:
    """기존 `build_projector(result)`(Exact, 이 함수는 절대 바꾸지 않는다)로
    정확한 모델을 만든 뒤, 그 위에 빠른 LUT 계층을 씌운다. `build_projector()`
    와는 별도의 entry point다 - 호출자는 정확도가 중요하면 `build_projector()`
    (Exact)를, 처리량이 중요하면 이 함수(근사)를 선택할 수 있다."""
    model = build_projector(result)
    return RuntimeWindshieldProjector(
        model, image_width, image_height, lut_rows=lut_rows, lut_cols=lut_cols
    )


@dataclass
class RuntimeProjectorValidationReport:
    """Exact projector 대비 LUT 근사의 오차 실측 결과(사용자 스펙 C-1번,
    "must validate against exact implementation... explicitly report any
    accuracy tradeoff"). `project_*`는 픽셀 단위, `unproject_*`는 각도(도)
    단위 - 서로 다른 물리량이라 하나로 섞지 않는다."""
    num_project_samples: int
    project_median_px: float
    project_p95_px: float
    project_p99_px: float
    project_max_px: float
    num_unproject_samples: int
    unproject_median_deg: float
    unproject_p95_deg: float
    unproject_p99_deg: float
    unproject_max_deg: float


def validate_runtime_projector_vs_exact(
    runtime: RuntimeWindshieldProjector,
    *,
    sample_points_xyz: np.ndarray,
    sample_pixels_uv: np.ndarray,
) -> RuntimeProjectorValidationReport:
    """`sample_points_xyz`/`sample_pixels_uv`로 exact model과 LUT 근사를
    직접 비교한다. 호출자가 자신의 실제 사용 범위(예: LiDAR 포인트의 실제
    거리 분포, 관심 픽셀 영역)에 맞는 샘플을 넘겨야 의미 있는 숫자가 나온다
    - 이 함수 자체는 "대표 샘플"을 임의로 만들어내지 않는다(허위로 낙관적인
    검증을 피하기 위함)."""
    exact = runtime.exact_model
    points = np.asarray(sample_points_xyz, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(sample_pixels_uv, dtype=np.float64).reshape(-1, 2)

    proj_err = np.zeros(0)
    if points.shape[0] > 0:
        exact_proj = np.array(
            [exact.project_point(float(p[0]), float(p[1]), float(p[2])) for p in points]
        )
        fast_proj = runtime.project_points(points)
        proj_err = np.linalg.norm(exact_proj - fast_proj, axis=1)

    ang_err_deg = np.zeros(0)
    if pixels.shape[0] > 0:
        exact_dirs = np.array(
            [exact.unproject_pixel(float(uv[0]), float(uv[1])) for uv in pixels]
        )
        fast_dirs = runtime.unproject_pixels(pixels)
        cos_sim = np.clip(np.sum(exact_dirs * fast_dirs, axis=1), -1.0, 1.0)
        ang_err_deg = np.degrees(np.arccos(cos_sim))

    def _stat(arr: np.ndarray, pct: float) -> float:
        return float(np.percentile(arr, pct)) if arr.size else 0.0

    return RuntimeProjectorValidationReport(
        num_project_samples=int(points.shape[0]),
        project_median_px=_stat(proj_err, 50.0),
        project_p95_px=_stat(proj_err, 95.0),
        project_p99_px=_stat(proj_err, 99.0),
        project_max_px=float(np.max(proj_err)) if proj_err.size else 0.0,
        num_unproject_samples=int(pixels.shape[0]),
        unproject_median_deg=_stat(ang_err_deg, 50.0),
        unproject_p95_deg=_stat(ang_err_deg, 95.0),
        unproject_p99_deg=_stat(ang_err_deg, 99.0),
        unproject_max_deg=float(np.max(ang_err_deg)) if ang_err_deg.size else 0.0,
    )

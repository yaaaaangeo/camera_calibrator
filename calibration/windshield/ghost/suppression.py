"""
calibration.windshield.ghost.suppression
==============================================================

STEP 8B - Ghost Suppression (deterministic, PyTorch 불필요).

큰 CNN부터 만들지 않는다(사용자 스펙 36-38번) - 대신 Ghost displacement/
strength를 먼저 추정하고, 그 추정을 이용해 **결정론적 역변환**으로
재구성한다:

    I = T + alpha * W_delta(T)
    T_0 = I
    T_{k+1} = clip(I - alpha * W_delta(T_k), 0, 1)      (3~5회 반복)

Reflection Suppression 모델은 여기서 절대 재사용하지 않는다 - Ghost와
Reflection은 서로 다른 image formation model이기 때문이다.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from calibration.windshield.ghost.config import (
    DEFAULT_MAX_CORRECTION,
    DEFAULT_SUPPRESSION_ITERATIONS,
    GHOST_MODEL_VERSION,
)
from calibration.windshield.ghost.synthetic import warp_shift_field
from calibration.windshield.ghost.types import GhostField, GhostSpatialCell, GhostSuppressionResult


def _resize_grid_to_dense(grid: np.ndarray, width: int, height: int) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.float32)
    if grid.ndim != 2 or grid.shape[0] < 1 or grid.shape[1] < 1:
        raise ValueError("GhostField 그리드는 최소 1x1 이상의 2D 배열이어야 합니다.")
    if grid.shape == (1, 1):
        return np.full((height, width), float(grid[0, 0]), dtype=np.float32)
    return cv2.resize(grid, (width, height), interpolation=cv2.INTER_LINEAR)


def build_dense_fields(ghost_field: GhostField) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`GhostField`의 성긴 grid(offset_x/offset_y/strength)를 이미지
    해상도의 dense map으로 bilinear 확대한다."""
    w = int(round(ghost_field.image_width))
    h = int(round(ghost_field.image_height))
    dx_dense = _resize_grid_to_dense(ghost_field.offset_x, w, h)
    dy_dense = _resize_grid_to_dense(ghost_field.offset_y, w, h)
    alpha_dense = _resize_grid_to_dense(ghost_field.strength, w, h)
    return dx_dense, dy_dense, alpha_dense


def suppress_ghost(
    image_bgr: np.ndarray,
    ghost_field: GhostField,
    *,
    iterations: int = DEFAULT_SUPPRESSION_ITERATIONS,
    max_correction: float = DEFAULT_MAX_CORRECTION,
) -> GhostSuppressionResult:
    """`suppress_ghost(image, ghost_model, config)` 공개 API(사용자 스펙
    41번). Deterministic iterative reconstruction으로 ghost를 제거한다.

    Safety Guard(사용자 스펙 47번): 입력에 NaN/Inf가 있거나, 처리 중
    예외가 나거나, `ghost_field`가 이미지 크기와 맞지 않으면 원본 이미지로
    그대로 fallback한다(never crash, never silently produce garbage).
    """
    try:
        if image_bgr is None or not np.all(np.isfinite(image_bgr)):
            return GhostSuppressionResult(
                success=False,
                suppressed_image=None if image_bgr is None else image_bgr.copy(),
                fell_back_to_original=True,
                warning_message="입력 이미지에 NaN/Inf가 있어 원본으로 fallback했습니다.",
            )

        h, w = image_bgr.shape[:2]
        if abs(w - ghost_field.image_width) > 1e-6 or abs(h - ghost_field.image_height) > 1e-6:
            return GhostSuppressionResult(
                success=False,
                suppressed_image=image_bgr.copy(),
                fell_back_to_original=True,
                warning_message=(
                    f"ghost_field 크기({ghost_field.image_width}x{ghost_field.image_height})가 "
                    f"입력 이미지 크기({w}x{h})와 다릅니다 - 원본으로 fallback했습니다."
                ),
            )

        dx_dense, dy_dense, alpha_dense = build_dense_fields(ghost_field)

        is_float01 = image_bgr.dtype != np.uint8 and image_bgr.max() <= 1.0 + 1e-6
        high = 1.0 if (image_bgr.dtype != np.uint8 and is_float01) else 255.0
        i_img = image_bgr.astype(np.float32)

        t_k = i_img.copy()
        alpha3 = alpha_dense[..., None] if i_img.ndim == 3 else alpha_dense
        correction_map = np.zeros_like(i_img)
        n_iter = max(0, int(iterations))
        for _ in range(n_iter):
            ghost_pred = warp_shift_field(t_k, dx_dense, dy_dense)
            correction_map = np.clip(alpha3 * ghost_pred, 0.0, max_correction * high)
            t_k = np.clip(i_img - correction_map, 0.0, high)

        if not np.all(np.isfinite(t_k)):
            return GhostSuppressionResult(
                success=False,
                suppressed_image=image_bgr.copy(),
                fell_back_to_original=True,
                warning_message="재구성 결과에 NaN/Inf가 발생해 원본으로 fallback했습니다.",
            )

        out_dtype = image_bgr.dtype
        suppressed = t_k.astype(out_dtype)
        predicted_ghost = (np.clip(alpha3 * warp_shift_field(t_k, dx_dense, dy_dense), 0.0, high)).astype(np.float32)

        return GhostSuppressionResult(
            success=True,
            suppressed_image=suppressed,
            predicted_ghost_image=predicted_ghost,
            correction_map=correction_map,
            iterations=n_iter,
            mean_correction=float(np.mean(correction_map)),
            max_correction=float(np.max(correction_map)) if correction_map.size else 0.0,
        )
    except Exception as exc:  # pragma: no cover - defensive safety guard
        return GhostSuppressionResult(
            success=False,
            suppressed_image=None if image_bgr is None else image_bgr.copy(),
            fell_back_to_original=True,
            error_message=f"ghost suppression 중 예외 발생: {exc}",
        )


def fit_ghost_field_constant(
    *,
    image_width: float,
    image_height: float,
    mean_offset_x_px: float,
    mean_offset_y_px: float,
    mean_strength_ratio: float,
    rows: int = 1,
    cols: int = 1,
) -> GhostField:
    """8A의 dataset 레벨 평균값을 전체 이미지에 균일하게 적용하는 상수
    `GhostField`를 만든다(가장 단순한 첫 버전)."""
    offset_x = np.full((rows, cols), mean_offset_x_px, dtype=np.float32)
    offset_y = np.full((rows, cols), mean_offset_y_px, dtype=np.float32)
    strength = np.full((rows, cols), mean_strength_ratio, dtype=np.float32)
    return GhostField(offset_x=offset_x, offset_y=offset_y, strength=strength, image_width=image_width, image_height=image_height)


def fit_ghost_field_from_spatial_map(
    spatial_map: list[GhostSpatialCell],
    *,
    image_width: float,
    image_height: float,
    rows: int,
    cols: int,
    default_offset_x: float = 0.0,
    default_offset_y: float = 0.0,
    default_strength: float = 0.0,
) -> GhostField:
    """8A의 `GhostSpatialCell` 리스트(사용자 스펙 12-14번)를 그대로
    `GhostField` grid로 옮긴다 - 별도의 gradient 기반 최적화 없이, 이미
    계산된 spatial map을 재사용하는 결정론적 "fit"이다. sample_count==0인
    빈 cell은 dataset 평균(또는 지정된 default)으로 채운다."""
    offset_x = np.full((rows, cols), default_offset_x, dtype=np.float32)
    offset_y = np.full((rows, cols), default_offset_y, dtype=np.float32)
    strength = np.full((rows, cols), default_strength, dtype=np.float32)
    for cell in spatial_map:
        if cell.sample_count <= 0:
            continue
        if 0 <= cell.row < rows and 0 <= cell.col < cols:
            offset_x[cell.row, cell.col] = cell.mean_offset_x_px if cell.mean_offset_x_px is not None else default_offset_x
            offset_y[cell.row, cell.col] = cell.mean_offset_y_px if cell.mean_offset_y_px is not None else default_offset_y
            strength[cell.row, cell.col] = cell.mean_strength_ratio if cell.mean_strength_ratio is not None else default_strength
    return GhostField(offset_x=offset_x, offset_y=offset_y, strength=strength, image_width=image_width, image_height=image_height)


def save_ghost_model(ghost_field: GhostField, path: str, *, metadata: Optional[dict] = None) -> str:
    """`ghost_model.yml` 하나로 GhostField 전체를 저장한다(사용자 스펙 42
    번) - PyTorch tensor가 없으므로 sibling 바이너리 파일이 필요 없다."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "ghost_model_version": ghost_field.model_version,
        "image_width": float(ghost_field.image_width),
        "image_height": float(ghost_field.image_height),
        "grid_rows": int(ghost_field.offset_x.shape[0]),
        "grid_cols": int(ghost_field.offset_x.shape[1]),
        "offset_x": ghost_field.offset_x.astype(float).tolist(),
        "offset_y": ghost_field.offset_y.astype(float).tolist(),
        "strength": ghost_field.strength.astype(float).tolist(),
        "metadata": metadata or {},
    }
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return str(p)


def load_ghost_model(path: str) -> GhostField:
    """`save_ghost_model()`이 만든 YAML을 되돌린다. 파일이 없으면
    `FileNotFoundError`를 그대로 낸다(silent fallback 금지)."""
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return GhostField(
        offset_x=np.array(data["offset_x"], dtype=np.float32),
        offset_y=np.array(data["offset_y"], dtype=np.float32),
        strength=np.array(data["strength"], dtype=np.float32),
        image_width=float(data["image_width"]),
        image_height=float(data["image_height"]),
        model_version=int(data.get("ghost_model_version", GHOST_MODEL_VERSION)),
    )

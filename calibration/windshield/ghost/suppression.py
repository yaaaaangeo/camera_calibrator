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
    DEFAULT_MAD_OUTLIER_K,
    DEFAULT_MAX_CORRECTION,
    DEFAULT_OVER_SUPPRESSION_CLEAN_WEIGHT,
    DEFAULT_OVER_SUPPRESSION_EDGE_WEIGHT,
    DEFAULT_SUPPRESSION_ITERATIONS,
    GHOST_MODEL_VERSION,
)
from calibration.windshield.ghost.spatial_model import build_robust_spatial_map_from_detections, fill_empty_spatial_cells
from calibration.windshield.ghost.synthetic import warp_shift_field
from calibration.windshield.ghost.types import (
    GhostDatasetResult,
    GhostField,
    GhostFieldDiagnostics,
    GhostReconstructionMetrics,
    GhostSpatialCell,
    GhostSuppressionResult,
)


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


def _edge_magnitude(image: np.ndarray) -> np.ndarray:
    """Reflection Suppression의 `_edge_energy`와 완전히 독립된 Ghost 전용
    구현이다(Ghost가 Reflection 코드를 재사용하지 않는다는 원칙을 detail
    retention metric에도 그대로 적용) - Sobel gradient magnitude."""
    gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _compute_detail_retention_metrics(
    original: np.ndarray, suppressed: np.ndarray, correction_map: np.ndarray, high: float,
) -> tuple[float, float, float]:
    """Ghost Strength Reduction만으로는 성공이 아니다(STEP 8 stabilization
    7번, "Ghost만 지우면 완료가 아니다") - main edge 유지 + clean 영역
    불필요 변화까지 함께 봐야 한다.

    - edge_retention = mean(|grad(suppressed)|) / (mean(|grad(original)|)+eps)
      1에 가까울수록 구조가 잘 유지된 것이다.
    - clean_region_change = correction이 median 이하로 작았던("거의 손대지
      않았어야 할") 절반 영역에서의 평균 correction 크기 - 넓게 퍼진 불필요한
      변화를 감지하기 위한 진단이다.
    - over_suppression_score = clean_region_change(0-1 정규화) +
      max(0, 1-edge_retention) - 0에 가까울수록 이상적이다.
    """
    eps = 1e-6
    edge_before = float(np.mean(_edge_magnitude(original)))
    edge_after = float(np.mean(_edge_magnitude(suppressed)))
    edge_retention = edge_after / (edge_before + eps)

    correction_gray = np.mean(correction_map, axis=-1) if correction_map.ndim == 3 else correction_map
    if correction_gray.size:
        median_correction = float(np.median(correction_gray))
        clean_mask = correction_gray <= median_correction
        clean_region_change = float(np.mean(correction_gray[clean_mask])) if np.any(clean_mask) else 0.0
    else:
        clean_region_change = 0.0

    over_suppression_score = (
        DEFAULT_OVER_SUPPRESSION_CLEAN_WEIGHT * (clean_region_change / max(high, eps))
        + DEFAULT_OVER_SUPPRESSION_EDGE_WEIGHT * max(0.0, 1.0 - edge_retention)
    )
    return clean_region_change, edge_retention, over_suppression_score


def compute_reconstruction_metrics(clean: np.ndarray, reconstructed: np.ndarray) -> GhostReconstructionMetrics:
    """Synthetic GT(clean 원본을 알고 있는 경우)에서만 쓰는 reconstruction
    품질 지표(STEP 8 stabilization 7-D번) - real footage 평가에는 GT가
    없으므로 쓰이지 않는다. `skimage` 등 새 dependency 없이 MAE/RMSE/PSNR만
    계산한다(SSIM은 이번 라운드에서 추가하지 않는다)."""
    a = clean.astype(np.float64)
    b = reconstructed.astype(np.float64)
    diff = a - b
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff * diff))
    rmse = float(np.sqrt(mse))
    max_val = 255.0 if clean.dtype == np.uint8 else float(np.max(a)) or 1.0
    psnr_db = None if mse <= 1e-12 else float(20.0 * np.log10(max_val) - 10.0 * np.log10(mse))
    return GhostReconstructionMetrics(mae=mae, rmse=rmse, psnr_db=psnr_db)


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

        clean_region_change, edge_retention, over_suppression_score = _compute_detail_retention_metrics(
            i_img, t_k, correction_map, high,
        )

        return GhostSuppressionResult(
            success=True,
            suppressed_image=suppressed,
            predicted_ghost_image=predicted_ghost,
            correction_map=correction_map,
            iterations=n_iter,
            mean_correction=float(np.mean(correction_map)),
            max_correction=float(np.max(correction_map)) if correction_map.size else 0.0,
            clean_region_change=clean_region_change,
            edge_retention=edge_retention,
            over_suppression_score=over_suppression_score,
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


def fit_ghost_field_from_dataset(
    dataset_result: GhostDatasetResult,
    *,
    image_width: float,
    image_height: float,
    rows: int,
    cols: int,
    mad_k: float = DEFAULT_MAD_OUTLIER_K,
) -> GhostField:
    """Dataset 전체(모든 frame의 모든 detection)에서 `GhostField`를
    fit한다(STEP 8 stabilization 3-G번, "반드시 `per_frame[0]`을 사용하지
    않는다"). 각 frame이 이미 계산해 둔 spatial map을 재사용하는 대신,
    frame 경계를 없애고 raw detection을 전부 풀링한 뒤 cell별로 다시
    robust median/MAD aggregation(`build_robust_spatial_map_from_detections`)
    을 적용한다 - 이렇게 해야 outlier가 섞인 소수의 frame이 최종 field를
    왜곡하지 않는다.

    빈 cell은 `fill_empty_spatial_cells()`(인접 valid cell -> 안 되면
    그대로)로 채운다. `GhostFieldDiagnostics`에 dataset 크기/coverage
    정보를 함께 기록한다."""
    all_detections = [det for frame in dataset_result.per_frame for det in frame.detections]

    cells = build_robust_spatial_map_from_detections(
        all_detections, image_width=image_width, image_height=image_height, rows=rows, cols=cols, mad_k=mad_k,
    )
    cells = fill_empty_spatial_cells(cells)

    offset_x = np.zeros((rows, cols), dtype=np.float32)
    offset_y = np.zeros((rows, cols), dtype=np.float32)
    strength = np.zeros((rows, cols), dtype=np.float32)
    samples_per_cell = [0] * (rows * cols)
    filled_cell_count = 0
    for cell in cells:
        offset_x[cell.row, cell.col] = cell.mean_offset_x_px or 0.0
        offset_y[cell.row, cell.col] = cell.mean_offset_y_px or 0.0
        strength[cell.row, cell.col] = cell.mean_strength_ratio or 0.0
        samples_per_cell[cell.row * cols + cell.col] = cell.sample_count
        if cell.sample_count > 0:
            filled_cell_count += 1

    detected_detections = [d for d in all_detections if d.detected]
    global_dx = float(np.median([d.offset_x_px for d in detected_detections])) if detected_detections else None
    global_dy = float(np.median([d.offset_y_px for d in detected_detections])) if detected_detections else None
    global_strength = (
        float(np.median([d.strength_ratio for d in detected_detections if d.strength_ratio is not None]))
        if any(d.strength_ratio is not None for d in detected_detections) else None
    )

    diagnostics = GhostFieldDiagnostics(
        num_frames=len(dataset_result.per_frame),
        num_detections=len(detected_detections),
        grid_rows=rows,
        grid_cols=cols,
        samples_per_cell=samples_per_cell,
        global_median_dx=global_dx,
        global_median_dy=global_dy,
        global_median_strength=global_strength,
        fit_stability=(filled_cell_count / (rows * cols)) if rows * cols > 0 else None,
    )

    return GhostField(
        offset_x=offset_x, offset_y=offset_y, strength=strength,
        image_width=image_width, image_height=image_height,
        diagnostics=diagnostics,
    )


def save_ghost_model(ghost_field: GhostField, path: str, *, metadata: Optional[dict] = None) -> str:
    """`ghost_model.yml` 하나로 GhostField 전체를 저장한다(사용자 스펙 42
    번) - PyTorch tensor가 없으므로 sibling 바이너리 파일이 필요 없다."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    diag = ghost_field.diagnostics
    data = {
        "ghost_model_version": ghost_field.model_version,
        "image_width": float(ghost_field.image_width),
        "image_height": float(ghost_field.image_height),
        "grid_rows": int(ghost_field.offset_x.shape[0]),
        "grid_cols": int(ghost_field.offset_x.shape[1]),
        "offset_x": ghost_field.offset_x.astype(float).tolist(),
        "offset_y": ghost_field.offset_y.astype(float).tolist(),
        "strength": ghost_field.strength.astype(float).tolist(),
        "diagnostics": None if diag is None else dataclasses.asdict(diag),
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
    diag_raw = data.get("diagnostics")
    diagnostics = GhostFieldDiagnostics(**diag_raw) if diag_raw else None
    return GhostField(
        offset_x=np.array(data["offset_x"], dtype=np.float32),
        offset_y=np.array(data["offset_y"], dtype=np.float32),
        strength=np.array(data["strength"], dtype=np.float32),
        image_width=float(data["image_width"]),
        image_height=float(data["image_height"]),
        model_version=int(data.get("ghost_model_version", GHOST_MODEL_VERSION)),
        diagnostics=diagnostics,
    )

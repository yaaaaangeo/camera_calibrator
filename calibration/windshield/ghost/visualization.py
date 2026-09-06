"""
calibration.windshield.ghost.visualization
==============================================================

STEP 8 stabilization 6번 - 실제 Ghost 시각화(단순 숫자 표가 아니라).

여기 있는 함수들은 전부 이미 계산된 `GhostPointDetection`/`GhostSpatialCell`
데이터를 그리기만 한다 - 별도의 분석/재계산을 하지 않는다(사용자 스펙
6-F번, "UI가 별도 분석 알고리즘을 수행하면 안 된다"). PySide6에 의존하지
않는 순수 NumPy/OpenCV 함수라 Qt 없이도 단위 테스트할 수 있다 - 실제 Qt
위젯(`ui/windshield_workspace.py`)은 이 함수들의 출력(BGR ndarray)을
QPixmap으로 바꿔 표시하기만 한다.

Vector Field(displacement)와 Strength Heatmap(세기)은 절대 한 이미지에
합치지 않는다 - 항상 별도의 두 이미지를 만든다.
"""

from __future__ import annotations

import cv2
import numpy as np

from calibration.windshield.ghost.types import GhostPointDetection, GhostSpatialCell

_MAIN_COLOR = (0, 255, 0)      # BGR - 초록
_GHOST_COLOR = (0, 0, 255)     # BGR - 빨강
_ARROW_COLOR = (0, 165, 255)   # BGR - 주황


def render_ghost_point_overlay(image_bgr: np.ndarray, detections: list[GhostPointDetection]) -> np.ndarray:
    """Original 이미지 위에 Main(큰 원)/Ghost(작은 원)/Main->Ghost(화살표)를
    그린다(사용자 스펙 6-C번). 검출되지 않은 main만 있는 경우도 큰 원만
    그려 "후보였지만 ghost를 못 찾음"을 보여준다."""
    canvas = image_bgr.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    for det in detections:
        main_pt = (int(round(det.main_x)), int(round(det.main_y)))
        cv2.circle(canvas, main_pt, 8, _MAIN_COLOR, thickness=2)
        if not det.detected or det.ghost_x is None or det.ghost_y is None:
            continue
        ghost_pt = (int(round(det.ghost_x)), int(round(det.ghost_y)))
        cv2.circle(canvas, ghost_pt, 4, _GHOST_COLOR, thickness=2)
        cv2.arrowedLine(canvas, main_pt, ghost_pt, _ARROW_COLOR, thickness=1, tipLength=0.3)
        if det.offset_x_px is not None and det.offset_y_px is not None:
            label = f"dx={det.offset_x_px:.1f},dy={det.offset_y_px:.1f}"
            if det.strength_ratio is not None:
                label += f" s={det.strength_ratio:.2f}"
            cv2.putText(
                canvas, label, (main_pt[0] + 10, main_pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _ARROW_COLOR, 1, cv2.LINE_AA,
            )
    return canvas


def render_vector_field_image(
    spatial_map: list[GhostSpatialCell], *, rows: int, cols: int, cell_size_px: int = 60,
) -> np.ndarray:
    """각 cell 중심에서 실제 화살표(방향=dx,dy / 길이=displacement
    magnitude에 비례)를 그린 이미지를 만든다(사용자 스펙 6-D번). Strength는
    여기 절대 섞지 않는다(사용자 스펙 6-E번, "Vector Field / Strength
    Heatmap을 같은 그림에 합치지 않는다")."""
    h, w = rows * cell_size_px, cols * cell_size_px
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    by_pos = {(c.row, c.col): c for c in spatial_map}
    max_mag = 1e-6
    for c in spatial_map:
        if c.sample_count > 0 and c.mean_offset_x_px is not None and c.mean_offset_y_px is not None:
            max_mag = max(max_mag, float(np.hypot(c.mean_offset_x_px, c.mean_offset_y_px)))

    for row in range(rows):
        for col in range(cols):
            cx = int((col + 0.5) * cell_size_px)
            cy = int((row + 0.5) * cell_size_px)
            cv2.rectangle(
                canvas, (col * cell_size_px, row * cell_size_px),
                ((col + 1) * cell_size_px - 1, (row + 1) * cell_size_px - 1), (200, 200, 200), 1,
            )
            cell = by_pos.get((row, col))
            if cell is None or cell.sample_count <= 0 or cell.mean_offset_x_px is None or cell.mean_offset_y_px is None:
                cv2.drawMarker(canvas, (cx, cy), (150, 150, 150), cv2.MARKER_TILTED_CROSS, 6, 1)
                continue
            dx, dy = cell.mean_offset_x_px, cell.mean_offset_y_px
            mag = float(np.hypot(dx, dy))
            scale = (cell_size_px * 0.4) / max_mag
            end = (int(round(cx + dx * scale)), int(round(cy + dy * scale)))
            cv2.arrowedLine(canvas, (cx, cy), end, _ARROW_COLOR, thickness=2, tipLength=0.3)
    return canvas


def render_strength_heatmap_image(
    spatial_map: list[GhostSpatialCell], *, rows: int, cols: int, cell_size_px: int = 60,
) -> np.ndarray:
    """Strength(0~1로 clip)를 컬러맵으로 그린 heatmap 이미지(사용자 스펙
    6-E번) - Vector Field와 완전히 별도 이미지다."""
    grid = np.zeros((rows, cols), dtype=np.float32)
    has_value = np.zeros((rows, cols), dtype=bool)
    for c in spatial_map:
        if c.sample_count > 0 and c.mean_strength_ratio is not None:
            grid[c.row, c.col] = float(np.clip(c.mean_strength_ratio, 0.0, 1.0))
            has_value[c.row, c.col] = True

    gray = np.clip(grid * 255.0, 0, 255).astype(np.uint8)
    dense = cv2.resize(gray, (cols * cell_size_px, rows * cell_size_px), interpolation=cv2.INTER_NEAREST)
    colored = cv2.applyColorMap(dense, cv2.COLORMAP_JET)

    for row in range(rows):
        for col in range(cols):
            x0, y0 = col * cell_size_px, row * cell_size_px
            x1, y1 = x0 + cell_size_px - 1, y0 + cell_size_px - 1
            cv2.rectangle(colored, (x0, y0), (x1, y1), (128, 128, 128), 1)
            if not has_value[row, col]:
                cv2.drawMarker(colored, (x0 + cell_size_px // 2, y0 + cell_size_px // 2), (128, 128, 128), cv2.MARKER_TILTED_CROSS, 6, 1)
    return colored

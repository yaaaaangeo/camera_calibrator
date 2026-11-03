"""image_selection exposure_mode: 기존 "global" 동작 유지 + "pattern" 모드 비교.

pattern 모드는 이미지 전체 clipping 대신 보드 영역 밝기와 코너 주변 포화
비율로 노출을 판정한다. 기본값은 기존과 같은 "global"이다.
"""

from __future__ import annotations

import copy
import glob
from pathlib import Path

import cv2
import numpy as np
import pytest

from calibration.detector import detect_dataset
from calibration.image_quality import compute_pattern_exposure_metrics
from calibration.image_selection import ImageSelectionConfig, select_calibration_images
from calibration.types import PatternConfig, PatternType

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_IMAGE = REPO_ROOT / "image" / "bag_0067_t1788945731.945.jpg"
# 저장소 image/ 의 실제 촬영 이미지 보드 (마커 ID 0~34 → 10x7, DICT_5X5_100).
# 실제 칸 크기는 기록이 없어 0.04 m로 둔다 - 검출/노출 판정에는 영향이 없다.
REAL_PATTERN = PatternConfig(
    type=PatternType.CHARUCO, squares_x=10, squares_y=7,
    square_size=0.04, marker_size=0.03, dictionary="DICT_5X5_100",
)


@pytest.fixture(scope="module")
def white_background_dataset(synthetic_distorted_dataset_dir, pattern_config):
    paths = sorted(glob.glob(f"{synthetic_distorted_dataset_dir}/*.jpg"))
    return detect_dataset(paths, pattern_config)


def test_default_mode_is_unchanged_global(white_background_dataset):
    assert ImageSelectionConfig().exposure_mode == "global"
    result = select_calibration_images(copy.deepcopy(white_background_dataset))
    # 기존 동작 기록: 흰 배경(전체 clipping > 15%) 합성 이미지는 전부 노출 탈락
    assert result.counts["exposure_rejected"] == result.counts["valid"] + result.counts["exposure_rejected"]
    assert result.selected_count == 0
    manifest = result.to_manifest()
    assert "exposure_mode" not in manifest
    assert all("corner_clip" not in image for image in manifest["images"])
    assert all(r.corner_clip is None for r in result.records)


def test_pattern_mode_keeps_sharp_board_on_white_background(white_background_dataset):
    result = select_calibration_images(
        copy.deepcopy(white_background_dataset), ImageSelectionConfig(exposure_mode="pattern"),
    )
    assert result.counts["exposure_rejected"] == 0
    assert result.selected_count == result.counts["valid"] > 0
    manifest = result.to_manifest()
    assert manifest["exposure_mode"] == "pattern"
    measured = [image for image in manifest["images"] if image["detection_success"]]
    assert all(image["corner_clip"] is not None and image["roi_brightness"] is not None for image in measured)


@pytest.mark.skipif(not REAL_IMAGE.exists(), reason="실제 촬영 이미지가 없는 checkout")
def test_pattern_mode_rejects_overexposed_real_capture(tmp_path):
    original = cv2.imread(str(REAL_IMAGE))
    cv2.imwrite(str(tmp_path / "orig.jpg"), original)
    # 이득 1.5: 실측에서 코너 p95 이동 > 10 px, 코너 23 → 10개로 감소한 과노출
    overexposed = np.clip(original.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
    cv2.imwrite(str(tmp_path / "over.jpg"), overexposed)
    dataset = detect_dataset([str(tmp_path / "orig.jpg"), str(tmp_path / "over.jpg")], REAL_PATTERN)
    result = select_calibration_images(dataset, ImageSelectionConfig(exposure_mode="pattern"))
    reasons = {r.image_id: r.exclusion_reason for r in result.records}
    assert reasons["orig"] == ""
    assert reasons["over"] == "exposure_rejected"
    corner_clip = {r.image_id: r.corner_clip for r in result.records}
    assert corner_clip["orig"] < 0.45 < corner_clip["over"]


def test_unknown_exposure_mode_is_rejected(white_background_dataset):
    with pytest.raises(ValueError, match="exposure_mode"):
        select_calibration_images(copy.deepcopy(white_background_dataset), ImageSelectionConfig(exposure_mode="auto"))


SQUARE_PX = 40
GRID_CORNERS = np.array(
    [[x, y] for y in range(SQUARE_PX, 5 * SQUARE_PX, SQUARE_PX) for x in range(SQUARE_PX, 5 * SQUARE_PX, SQUARE_PX)],
    dtype=np.float32,
)


def _checkerboard(white: int, black: int, background: int) -> np.ndarray:
    image = np.full((400, 400), background, dtype=np.uint8)
    for row in range(5):
        for col in range(5):
            value = white if (row + col) % 2 == 0 else black
            image[row * SQUARE_PX:(row + 1) * SQUARE_PX, col * SQUARE_PX:(col + 1) * SQUARE_PX] = value
    return cv2.GaussianBlur(image, (5, 5), 1.0)


def test_corner_clip_ignores_bright_background_but_sees_saturated_corners():
    on_white_background = compute_pattern_exposure_metrics(_checkerboard(220, 30, 255), GRID_CORNERS)
    saturated = compute_pattern_exposure_metrics(_checkerboard(255, 120, 255), GRID_CORNERS)
    assert on_white_background.patch_radius_px == SQUARE_PX // 4
    assert on_white_background.corner_clip_median == 0.0  # 배경 포화는 코너 판정에 들어가지 않음
    assert saturated.corner_clip_median > 0.3
    assert 100 < on_white_background.roi_brightness < 150  # 보드 영역만의 평균
    assert compute_pattern_exposure_metrics(_checkerboard(220, 30, 255), GRID_CORNERS[:3]) is None

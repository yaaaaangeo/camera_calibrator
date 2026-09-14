from __future__ import annotations

import numpy as np

from calibration.image_selection import (
    ImageSelectionConfig,
    allocate_position_targets,
    select_calibration_images,
)
from calibration.types import Dataset, DetectionResult, Frame, FrameStatus, ImageInfo


def _frame(
    image_id: str,
    *,
    center: tuple[float, float] = (960.0, 540.0),
    area: float = 0.18,
    tilt: float = 0.0,
    success: bool = True,
    sharpness: float = 100.0,
    brightness: float = 127.0,
    saturation: float = 0.0,
    phash: str | None = None,
) -> Frame:
    info = ImageInfo(
        image_id=image_id,
        path=f"/images/{image_id}.png",
        width=1920,
        height=1080,
        sharpness=sharpness,
        brightness=brightness,
        saturation=saturation,
        phash=phash or f"{int(image_id.strip('f') or 0):064x}",
    )
    detection = DetectionResult(
        image_id=image_id,
        success=success,
        corners=np.zeros((24, 1, 2), dtype=np.float32) if success else None,
        num_corners=24 if success else 0,
        board_center_px=center if success else None,
        board_area_ratio=area if success else None,
        board_tilt_deg=tilt if success else None,
        corner_confidence=1.0 if success else None,
        failure_reason=None if success else "not found",
    )
    return Frame(
        image_info=info,
        detection=detection,
        status=FrameStatus.DETECTED if success else FrameStatus.DETECTION_FAILED,
    )


def test_allocate_position_targets_default_35_profile():
    assert allocate_position_targets(35) == {
        "center": 5,
        "left": 5,
        "right": 5,
        "top": 4,
        "bottom": 4,
        "top-left": 3,
        "top-right": 3,
        "bottom-left": 3,
        "bottom-right": 3,
    }


def test_allocate_position_targets_sums_to_requested_count():
    for target in range(1, 57):
        assert sum(allocate_position_targets(target).values()) == target


def test_selection_rejects_failed_blurry_and_bad_exposure_frames():
    dataset = Dataset(frames=[
        _frame("f0", success=False),
        _frame("f1", sharpness=2.0),
        _frame("f2", brightness=245.0),
        _frame("f3"),
        _frame("f4", center=(200.0, 540.0)),
    ])

    result = select_calibration_images(dataset, ImageSelectionConfig(target_count=5))

    assert result.selected_count == 2
    assert set(result.selected_image_ids) == {"f3", "f4"}
    reasons = {record.image_id: record.exclusion_reason for record in result.records}
    assert reasons["f0"] == "detection_failed"
    assert reasons["f1"] == "blur_rejected"
    assert reasons["f2"] == "exposure_rejected"
    assert dataset.frames[3].image_info.path == "/images/f3.png"


def test_selection_is_deterministic_and_skips_near_duplicates_first():
    dataset = Dataset(frames=[
        _frame("f1", center=(250.0, 540.0), phash="0" * 64),
        _frame("f2", center=(252.0, 542.0), phash="0" * 64, sharpness=90.0),
        _frame("f3", center=(960.0, 540.0), phash="1" * 64),
        _frame("f4", center=(1660.0, 540.0), phash="2" * 64),
    ])

    first = select_calibration_images(dataset, ImageSelectionConfig(target_count=3))
    second = select_calibration_images(dataset, ImageSelectionConfig(target_count=3))

    assert first.selected_image_ids == second.selected_image_ids
    assert "f1" in first.selected_image_ids
    assert "f2" not in first.selected_image_ids
    duplicate_record = next(record for record in first.records if record.image_id == "f2")
    assert duplicate_record.exclusion_reason == "duplicate_scene"


def test_selection_caps_request_to_valid_pool_without_touching_source_paths():
    dataset = Dataset(frames=[
        _frame("f1", center=(200.0, 200.0)),
        _frame("f2", center=(960.0, 540.0)),
        _frame("f3", success=False),
    ])
    original_paths = [frame.image_info.path for frame in dataset.frames]

    result = select_calibration_images(dataset, ImageSelectionConfig(target_count=10))
    manifest = result.to_manifest()

    assert result.requested_count == 10
    assert result.target_count == 2
    assert result.selected_count == 2
    assert [frame.image_info.path for frame in dataset.frames] == original_paths
    assert all("train_holdout_assignment" in image for image in manifest["images"])
    assert {
        image["train_holdout_assignment"] for image in manifest["images"]
    } == {"pending_split", "excluded"}

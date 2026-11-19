from __future__ import annotations

import numpy as np

from calibration.models.object_releasing import (
    calibrate_object_releasing_brown_conrady,
    collect_object_releasing_inputs,
    expected_object_releasing_ids,
    expected_object_releasing_object_points,
)
from calibration.types import (
    CameraConfig,
    Dataset,
    DetectionResult,
    Frame,
    FrameStatus,
    ImageInfo,
    PatternConfig,
    PatternType,
)


def _frame(
    image_id: str,
    ids: list[int],
    *,
    pattern: PatternConfig,
    object_ids: list[int] | None = None,
) -> Frame:
    object_ids = object_ids or ids
    corners = np.array([[[float(i), float(i + 1)]] for i in range(len(ids))], dtype=np.float32)
    expected_obj = expected_object_releasing_object_points(pattern).reshape(-1, 1, 3)
    obj = np.asarray([expected_obj[i] for i in object_ids], dtype=np.float32)
    det = DetectionResult(
        image_id=image_id,
        success=True,
        corners=corners,
        object_points=obj,
        ids=np.asarray(ids, dtype=np.int32).reshape(-1, 1),
        num_corners=len(ids),
    )
    return Frame(
        image_info=ImageInfo(image_id=image_id, path=f"{image_id}.png", width=640, height=480),
        detection=det,
        status=FrameStatus.DETECTED,
    )


def _dataset(*frames: Frame) -> Dataset:
    return Dataset(frames=list(frames))


def _charuco_pattern() -> PatternConfig:
    return PatternConfig(type=PatternType.CHARUCO, squares_x=4, squares_y=3, square_size=0.04)


def _aprilgrid_pattern() -> PatternConfig:
    return PatternConfig(
        type=PatternType.APRILGRID,
        squares_x=2,
        squares_y=2,
        square_size=0.05,
        marker_size=0.035,
        dictionary="DICT_APRILTAG_36h11",
    )


def test_charuco_full_board_multiple_frames_pass():
    pattern = _charuco_pattern()
    ids = expected_object_releasing_ids(pattern).tolist()
    frames, obj, img, rejected = collect_object_releasing_inputs(
        _dataset(_frame("a", ids, pattern=pattern), _frame("b", ids, pattern=pattern)),
        pattern,
    )
    assert [f.image_info.image_id for f in frames] == ["a", "b"]
    assert len(obj) == len(img) == 2
    assert [d["accepted"] for d in rejected] == [True, True]


def test_charuco_object_releasing_public_calibration_is_disabled():
    pattern = _charuco_pattern()
    result = calibrate_object_releasing_brown_conrady(
        _dataset(),
        CameraConfig(width=640, height=480),
        pattern,
    )
    assert not result.success
    assert "disabled" in result.error_message
    assert "charuco" in result.error_message


def test_charuco_partial_board_frame_is_rejected():
    pattern = _charuco_pattern()
    full = expected_object_releasing_ids(pattern).tolist()
    partial = full[:-1]
    frames, _obj, _img, rejected = collect_object_releasing_inputs(
        _dataset(_frame("full", full, pattern=pattern), _frame("partial", partial, pattern=pattern)),
        pattern,
    )
    assert [f.image_info.image_id for f in frames] == ["full"]
    assert [d["image_id"] for d in rejected if not d["accepted"]] == ["partial"]
    partial_diag = next(d for d in rejected if d["image_id"] == "partial")
    assert partial_diag["expected_count"] == 6
    assert partial_diag["detected_count"] == 5
    assert partial_diag["missing_ids"] == [5]


def test_charuco_same_count_different_ids_is_rejected():
    pattern = _charuco_pattern()
    full = expected_object_releasing_ids(pattern).tolist()
    different = [1, 2, 3, 4, 5, 6]
    frames, _obj, _img, rejected = collect_object_releasing_inputs(
        _dataset(_frame("full", full, pattern=pattern), _frame("different", different, pattern=pattern, object_ids=full)),
        pattern,
    )
    assert [f.image_info.image_id for f in frames] == ["full"]
    different_diag = next(d for d in rejected if d["image_id"] == "different")
    assert not different_diag["accepted"]
    assert different_diag["missing_ids"] == [0]


def test_charuco_shuffled_ids_are_canonical_sorted_and_pass():
    pattern = _charuco_pattern()
    full = expected_object_releasing_ids(pattern).tolist()
    shuffled = [3, 1, 5, 0, 2, 4]
    frames, obj, img, rejected = collect_object_releasing_inputs(
        _dataset(_frame("shuffled", shuffled, pattern=pattern, object_ids=shuffled)),
        pattern,
    )
    assert [f.image_info.image_id for f in frames] == ["shuffled"]
    assert rejected[0]["accepted"]
    assert np.allclose(obj[0], expected_object_releasing_object_points(pattern))
    assert img[0].shape == (6, 1, 2)


def test_aprilgrid_full_board_multiple_frames_pass():
    pattern = _aprilgrid_pattern()
    ids = expected_object_releasing_ids(pattern).tolist()
    frames, obj, img, rejected = collect_object_releasing_inputs(
        _dataset(_frame("a", ids, pattern=pattern), _frame("b", ids, pattern=pattern)),
        pattern,
    )
    assert len(frames) == 2
    assert obj[0].shape == (16, 1, 3)
    assert img[0].shape == (16, 1, 2)
    assert [d["accepted"] for d in rejected] == [True, True]


def test_aprilgrid_object_releasing_public_calibration_is_disabled():
    pattern = _aprilgrid_pattern()
    result = calibrate_object_releasing_brown_conrady(
        _dataset(),
        CameraConfig(width=640, height=480),
        pattern,
    )
    assert not result.success
    assert "disabled" in result.error_message
    assert "apriltag_grid" in result.error_message


def test_aprilgrid_missing_tag_is_rejected():
    pattern = _aprilgrid_pattern()
    ids = expected_object_releasing_ids(pattern).tolist()
    missing_tag = [i for i in ids if i // 4 != 2]
    frames, _obj, _img, rejected = collect_object_releasing_inputs(
        _dataset(_frame("full", ids, pattern=pattern), _frame("missing_tag", missing_tag, pattern=pattern)),
        pattern,
    )
    assert [f.image_info.image_id for f in frames] == ["full"]
    missing_diag = next(d for d in rejected if d["image_id"] == "missing_tag")
    assert not missing_diag["accepted"]
    assert missing_diag["missing_tag_ids"] == [2]
    assert missing_diag["missing_ids"] == [8, 9, 10, 11]


def test_aprilgrid_tag_id_set_mismatch_is_rejected():
    pattern = _aprilgrid_pattern()
    ids = expected_object_releasing_ids(pattern).tolist()
    mismatch = ids[:-1] + [99]
    frames, _obj, _img, rejected = collect_object_releasing_inputs(
        _dataset(_frame("full", ids, pattern=pattern), _frame("mismatch", mismatch, pattern=pattern, object_ids=ids)),
        pattern,
    )
    assert [f.image_info.image_id for f in frames] == ["full"]
    mismatch_diag = next(d for d in rejected if d["image_id"] == "mismatch")
    assert not mismatch_diag["accepted"]
    assert mismatch_diag["missing_ids"] == [15]


def test_aprilgrid_tag_internal_corner_order_is_canonicalized_and_passes():
    pattern = _aprilgrid_pattern()
    ids = expected_object_releasing_ids(pattern).tolist()
    first_tag_reversed = [3, 2, 1, 0] + ids[4:]
    frames, obj, _img, rejected = collect_object_releasing_inputs(
        _dataset(_frame("reordered", first_tag_reversed, pattern=pattern, object_ids=first_tag_reversed)),
        pattern,
    )
    assert [f.image_info.image_id for f in frames] == ["reordered"]
    assert rejected[0]["accepted"]
    assert np.allclose(obj[0], expected_object_releasing_object_points(pattern))

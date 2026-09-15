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


def test_similar_background_hash_does_not_force_duplicate_when_pose_differs():
    """Test 3 (계획 문서 11번, Problem 5) - 배경(phash)은 동일하지만 보드
    pose가 크게 다른 두 프레임은 duplicate로 판정되면 안 된다. 차량 장착
    카메라처럼 배경이 고정된 환경을 시뮬레이션한다.
    """
    from calibration.image_selection import ImageSelectionRecord, _is_similar_scene

    same_hash = "a" * 64
    left = _frame("left", center=(200.0, 540.0), phash=same_hash)
    right = _frame("right", center=(1700.0, 540.0), phash=same_hash)
    records = {
        "left": ImageSelectionRecord(
            image_id="left", path="", position="left", distance="middle", tilt="front",
        ),
        "right": ImageSelectionRecord(
            image_id="right", path="", position="right", distance="middle", tilt="front",
        ),
    }
    cfg = ImageSelectionConfig()
    assert _is_similar_scene(left, right, records, cfg) is False

    dataset = Dataset(frames=[left, right])
    result = select_calibration_images(dataset, ImageSelectionConfig(target_count=2))
    assert set(result.selected_image_ids) == {"left", "right"}
    reasons = {r.image_id: r.exclusion_reason for r in result.records}
    assert reasons["left"] == ""
    assert reasons["right"] == ""


def test_similar_pose_and_hash_together_are_flagged_duplicate():
    """AND-gate의 반대쪽 절반 - pose도 거의 동일하고 hash도 매우 비슷하면
    여전히 duplicate로 판정돼야 한다(진짜 연사 중복은 계속 잡아야 함).
    """
    from calibration.image_selection import ImageSelectionRecord, _is_similar_scene

    same_hash = "b" * 64
    a = _frame("a", center=(960.0, 540.0), phash=same_hash)
    b = _frame("b", center=(965.0, 542.0), phash=same_hash)
    records = {
        "a": ImageSelectionRecord(image_id="a", path="", position="center", distance="middle", tilt="front"),
        "b": ImageSelectionRecord(image_id="b", path="", position="center", distance="middle", tilt="front"),
    }
    assert _is_similar_scene(a, b, records, ImageSelectionConfig()) is True


def test_transitive_pose_chain_does_not_collapse_genuine_diversity():
    """Test 4 (계획 문서 11번, Problem 9) - A≈B, B≈C(pairwise 유사)이지만
    A!=C(실제로는 서로 다른 pose)인 체인에서, A와 C가 강제로 같은 cluster에
    묶여 대표 하나만 선택되는 일이 없어야 한다.

    A/B/C는 모두 같은 position/distance/tilt 버킷(같은 "center" 3분면,
    area/tilt 고정)에 있고 phash도 동일해서, 예전 Union-Find 방식이라면
    sim(A,B)=True, sim(B,C)=True로 A/B/C가 transitive closure를 통해
    전부 하나의 cluster가 되어(cluster당 최대 1장 규칙) target_count=2여도
    실질적으로 diversity가 큰 A/C 조합을 선택할 방법이 없었다. 지금은
    cluster 멤버십이 선택을 막지 않고 실제 pose-space 거리로 직접
    비교하므로, quality가 동률일 때 진짜로 더 먼 A/C 조합이 선택돼야 한다.
    """
    same_hash = "c" * 64
    a = _frame("a", center=(700.0, 540.0), phash=same_hash)
    b = _frame("b", center=(830.0, 540.0), phash=same_hash)
    c = _frame("c", center=(960.0, 540.0), phash=same_hash)

    from calibration.image_selection import ImageSelectionRecord, _is_similar_scene
    cfg = ImageSelectionConfig()
    records = {
        fid: ImageSelectionRecord(image_id=fid, path="", position="center", distance="middle", tilt="front")
        for fid in ("a", "b", "c")
    }
    assert _is_similar_scene(a, b, records, cfg) is True
    assert _is_similar_scene(b, c, records, cfg) is True
    assert _is_similar_scene(a, c, records, cfg) is False  # the actual "A != C" fact

    dataset = Dataset(frames=[a, b, c])
    result = select_calibration_images(dataset, ImageSelectionConfig(target_count=2))
    assert set(result.selected_image_ids) == {"a", "c"}


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

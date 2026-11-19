"""
tests/test_image_quality.py
==================================

설계 문서 3-1번 - 이미지 품질 검사 (해상도/노출/saturation/블러/모션블러/
대비/중복 검출).
"""

from __future__ import annotations

import numpy as np

from calibration.image_quality import (
    ImageQualitySeverity,
    check_blur,
    check_contrast,
    check_exposure,
    check_motion_blur,
    check_resolution,
    check_saturation,
    compute_contrast,
    compute_motion_blur_score,
    compute_phash,
    compute_saturation,
    evaluate_image_quality,
    find_duplicate_groups,
    hamming_distance,
)
from calibration.types import Dataset, Frame, FrameStatus, ImageInfo


class TestComputeMetrics:
    def test_uniform_gray_image_has_zero_contrast(self):
        img = np.full((100, 100), 128, dtype=np.uint8)
        assert compute_contrast(img) == 0.0

    def test_high_contrast_image_has_nonzero_contrast(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 255
        assert compute_contrast(img) > 50.0

    def test_uniform_image_has_zero_saturation_clipping(self):
        img = np.full((100, 100), 128, dtype=np.uint8)
        assert compute_saturation(img) == 0.0

    def test_all_black_image_has_full_saturation_clipping(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        assert compute_saturation(img) == 1.0

    def test_motion_blur_score_is_high_for_directional_edges(self):
        """가로 방향으로만 줄무늬가 있는(세로 그래디언트만 강한) 이미지는
        가로/세로 그래디언트 분산 비율이 크게 벌어져야 한다."""
        img = np.zeros((200, 200), dtype=np.uint8)
        img[::4, :] = 255  # 가로 줄무늬 -> 세로 방향 그래디언트만 강함
        score = compute_motion_blur_score(img.astype(np.float64))
        assert score > 2.0

    def test_phash_identical_for_identical_images(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (64, 64), dtype=np.uint8)
        assert compute_phash(img) == compute_phash(img.copy())

    def test_phash_hamming_distance_zero_for_same_hash(self):
        h = "abcd1234"
        assert hamming_distance(h, h) == 0

    def test_phash_hamming_distance_handles_invalid_input(self):
        assert hamming_distance("not-hex", "1234") == 999


class TestThresholdChecks:
    def test_low_resolution_is_error(self):
        issues = check_resolution(320, 240)
        assert issues and issues[0].severity == ImageQualitySeverity.ERROR

    def test_sufficient_resolution_has_no_issue(self):
        assert check_resolution(1920, 1080) == []

    def test_too_dark_is_warning(self):
        issues = check_exposure(5.0)
        assert any(i.code == "too_dark" for i in issues)

    def test_too_bright_is_warning(self):
        issues = check_exposure(250.0)
        assert any(i.code == "too_bright" for i in issues)

    def test_normal_brightness_has_no_issue(self):
        assert check_exposure(120.0) == []

    def test_high_saturation_clipping_is_warning(self):
        issues = check_saturation(0.5)
        assert any(i.code == "exposure_clipping" for i in issues)

    def test_low_saturation_has_no_issue(self):
        assert check_saturation(0.01) == []

    def test_very_low_sharpness_is_error(self):
        issues = check_blur(2.0)
        assert issues and issues[0].severity == ImageQualitySeverity.ERROR

    def test_normal_sharpness_has_no_issue(self):
        assert check_blur(500.0) == []

    def test_high_motion_blur_ratio_is_warning(self):
        issues = check_motion_blur(5.0)
        assert any(i.code == "possible_motion_blur" for i in issues)

    def test_low_motion_blur_ratio_has_no_issue(self):
        assert check_motion_blur(1.2) == []

    def test_low_contrast_is_warning(self):
        issues = check_contrast(5.0)
        assert any(i.code == "low_contrast" for i in issues)


class TestEvaluateImageQuality:
    def test_good_image_has_no_issues(self):
        info = ImageInfo(
            image_id="img1", path="-", width=1920, height=1080,
            sharpness=500.0, brightness=120.0, contrast=60.0,
            saturation=0.01, motion_blur_score=1.1,
        )
        report = evaluate_image_quality(info)
        assert report.issues == []

    def test_bad_image_accumulates_multiple_issues(self):
        info = ImageInfo(
            image_id="img2", path="-", width=320, height=240,
            sharpness=1.0, brightness=2.0, contrast=1.0,
            saturation=0.9, motion_blur_score=10.0,
        )
        report = evaluate_image_quality(info)
        codes = {i.code for i in report.issues}
        assert "resolution_too_low" in codes
        assert "too_dark" in codes
        assert "too_blurry" in codes
        assert report.has_errors


def _frame_with_phash(image_id: str, phash: str) -> Frame:
    info = ImageInfo(image_id=image_id, path=f"/fake/{image_id}.jpg", width=100, height=100, phash=phash)
    return Frame(image_info=info, status=FrameStatus.DETECTED)


class TestDuplicateDetection:
    def test_identical_hashes_form_a_group(self):
        dataset = Dataset(frames=[
            _frame_with_phash("a", "ffffffffffffffff"),
            _frame_with_phash("b", "ffffffffffffffff"),
            _frame_with_phash("c", "0000000000000000"),
        ])
        groups = find_duplicate_groups(dataset)
        assert len(groups) == 1
        assert set(groups[0].image_ids) == {"a", "b"}
        assert groups[0].exact

    def test_no_duplicates_when_all_distinct(self):
        dataset = Dataset(frames=[
            _frame_with_phash("a", "ffffffffffffffff"),
            _frame_with_phash("b", "0000000000000000"),
        ])
        assert find_duplicate_groups(dataset) == []

    def test_near_duplicate_detected_but_not_exact(self):
        # 해밍 거리 4 (비트 4개만 다름) - near-duplicate 범위(<=8) 안, 완전동일(<=2) 밖
        dataset = Dataset(frames=[
            _frame_with_phash("a", "00000000000000f0"),
            _frame_with_phash("b", "000000000000000f"),
        ])
        groups = find_duplicate_groups(dataset)
        assert len(groups) == 1
        assert not groups[0].exact

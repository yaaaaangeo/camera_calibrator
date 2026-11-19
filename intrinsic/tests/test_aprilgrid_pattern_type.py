"""
tests/test_aprilgrid_pattern_type.py
====================================

AprilGrid is a first-class PatternType in config paths and detector dispatch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.cli import _build_pattern_config, build_arg_parser
from calibration.detector import build_detect_fn
from calibration.types import PatternConfig, PatternType


def _aprilgrid_args(**overrides):
    values = {
        "pattern": "apriltag_grid",
        "squares_x": 7,
        "squares_y": 5,
        "square_size": 0.04,
        "marker_size": 0.03,
        "dictionary": "DICT_APRILTAG_36h11",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _aprilgrid_pattern(**overrides):
    values = {
        "type": PatternType.APRILGRID,
        "squares_x": 4,
        "squares_y": 3,
        "square_size": 0.04,
        "marker_size": 0.028,
        "dictionary": "DICT_APRILTAG_36h11",
    }
    values.update(overrides)
    return PatternConfig(**values)


def _render_aprilgrid(pattern: PatternConfig, tag_px: int = 90, pitch_px: int = 130) -> np.ndarray:
    dict_id = getattr(cv2.aruco, pattern.dictionary)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    margin = 80
    width = margin * 2 + (pattern.squares_x - 1) * pitch_px + tag_px
    height = margin * 2 + (pattern.squares_y - 1) * pitch_px + tag_px
    gray = np.full((height, width), 255, dtype=np.uint8)

    for marker_id in range(pattern.squares_x * pattern.squares_y):
        row = marker_id // pattern.squares_x
        col = marker_id % pattern.squares_x
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, tag_px)
        y = margin + row * pitch_px
        x = margin + col * pitch_px
        gray[y:y + tag_px, x:x + tag_px] = marker

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_cli_build_pattern_config_accepts_aprilgrid():
    pattern = _build_pattern_config(_aprilgrid_args())

    assert pattern.type == PatternType.APRILGRID
    assert pattern.marker_size == pytest.approx(0.03)
    assert pattern.dictionary == "DICT_APRILTAG_36h11"


def test_cli_build_pattern_config_accepts_aprilgrid_alias():
    pattern = _build_pattern_config(_aprilgrid_args(pattern="aprilgrid"))

    assert pattern.type == PatternType.APRILGRID


def test_cli_build_pattern_config_requires_marker_size_for_aprilgrid():
    with pytest.raises(Exception, match="marker-size"):
        _build_pattern_config(_aprilgrid_args(marker_size=None))


def test_cli_build_pattern_config_requires_apriltag_dictionary_for_aprilgrid():
    with pytest.raises(Exception, match="AprilTag dictionary"):
        _build_pattern_config(_aprilgrid_args(dictionary="DICT_5X5_100"))


def test_cli_accepts_kalibr_export_option():
    parser = build_arg_parser()

    args = parser.parse_args([
        "--images", "images",
        "--squares-x", "7",
        "--squares-y", "5",
        "--square-size", "0.04",
        "--marker-size", "0.03",
        "--export", "kalibr",
        "--kalibr-camera-model", "pinhole-equi",
    ])

    assert args.export == ["kalibr"]
    assert args.kalibr_camera_model == "pinhole-equi"


def test_detector_detects_synthetic_aprilgrid():
    pattern = _aprilgrid_pattern()
    detect_fn = build_detect_fn(pattern)
    image = _render_aprilgrid(pattern)

    result = detect_fn(image, "synthetic_aprilgrid")

    assert result.success is True
    assert result.num_corners == pattern.squares_x * pattern.squares_y * 4
    assert result.corners.shape == (result.num_corners, 1, 2)
    assert result.object_points.shape == (result.num_corners, 1, 3)
    assert result.ids.shape == (result.num_corners, 1)
    assert result.corner_confidence == pytest.approx(1.0)
    assert result.board_area_ratio > 0.2


def test_detector_maps_aprilgrid_ids_to_row_major_object_points():
    pattern = _aprilgrid_pattern()
    detect_fn = build_detect_fn(pattern)

    result = detect_fn(_render_aprilgrid(pattern), "synthetic_aprilgrid")

    assert result.success is True
    by_corner_id = {
        int(corner_id): point.reshape(3)
        for corner_id, point in zip(result.ids.reshape(-1), result.object_points)
    }
    marker_5_top_left = by_corner_id[5 * 4 + 0]
    marker_5_bottom_right = by_corner_id[5 * 4 + 2]
    assert marker_5_top_left.tolist() == pytest.approx([0.04, 0.04, 0.0])
    assert marker_5_bottom_right.tolist() == pytest.approx([0.068, 0.068, 0.0])


def test_detector_rejects_non_apriltag_dictionary_for_aprilgrid():
    pattern = _aprilgrid_pattern(dictionary="DICT_5X5_100")

    with pytest.raises(ValueError, match="AprilTag dictionary"):
        build_detect_fn(pattern)


def test_ui_source_exposes_aprilgrid_pattern_type_without_importing_qt():
    source = Path("ui/main_window.py").read_text(encoding="utf-8")

    assert 'PatternType.APRILGRID' in source
    assert 'AprilGrid (AprilTag grid)' in source
    assert 'DICT_APRILTAG_36h11' in source
    assert 'pattern_type == PatternType.APRILGRID' in source

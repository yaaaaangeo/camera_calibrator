from __future__ import annotations

import pytest

from calibration.types import PatternConfig, PatternType
from export.kalibr import (
    build_kalibr_aprilgrid_target,
    build_kalibr_camera_calibration_command,
    export_kalibr_target_yaml,
)


def _aprilgrid_pattern(**overrides) -> PatternConfig:
    values = {
        "type": PatternType.APRILGRID,
        "squares_x": 7,
        "squares_y": 5,
        "square_size": 0.04,
        "marker_size": 0.03,
        "dictionary": "DICT_APRILTAG_36h11",
    }
    values.update(overrides)
    return PatternConfig(**values)


def test_build_kalibr_aprilgrid_target_converts_pitch_to_spacing_ratio():
    target = build_kalibr_aprilgrid_target(_aprilgrid_pattern())

    assert target == {
        "target_type": "aprilgrid",
        "tagCols": 7,
        "tagRows": 5,
        "tagSize": 0.03,
        "tagSpacing": pytest.approx(1 / 3),
    }


def test_export_kalibr_target_yaml_writes_official_field_names(tmp_path):
    path = tmp_path / "kalibr_aprilgrid.yaml"

    export_kalibr_target_yaml(_aprilgrid_pattern(), str(path))

    text = path.read_text(encoding="utf-8")
    assert "target_type: 'aprilgrid'" in text
    assert "tagCols: 7" in text
    assert "tagRows: 5" in text
    assert "tagSize: 0.03" in text
    assert "tagSpacing: 0.333333333333" in text


def test_export_kalibr_target_yaml_rejects_non_aprilgrid(tmp_path):
    pattern = _aprilgrid_pattern(type=PatternType.CHARUCO)

    with pytest.raises(ValueError, match="apriltag_grid"):
        export_kalibr_target_yaml(pattern, str(tmp_path / "target.yaml"))


def test_build_kalibr_camera_calibration_command_uses_bag_topic_model_and_target():
    command = build_kalibr_camera_calibration_command(
        bag_path="dataset.bag",
        topic="/cam/image_raw",
        target_yaml_path="kalibr_aprilgrid.yaml",
        camera_model="pinhole-equi",
    )

    assert command == (
        "kalibr_calibrate_cameras --bag dataset.bag --topics /cam/image_raw "
        "--models pinhole-equi --target kalibr_aprilgrid.yaml"
    )

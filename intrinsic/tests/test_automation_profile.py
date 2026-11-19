"""automation.profile: YAML 로드, 실행 전 검증, 우선순위, 기존 CLI 인자 변환."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from automation.profile import (
    ProfileError,
    build_cli_argv,
    load_profile,
    parse_cli_argv,
    resolve_profile,
)
from tests._automation_test_utils import base_profile, pattern_section



def _resolve(raw: dict, **overrides):
    return resolve_profile(raw, source_path="test", **overrides)


def _error(raw: dict, **overrides) -> str:
    with pytest.raises(ProfileError) as info:
        _resolve(raw, **overrides)
    return str(info.value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("key", ["corner_clip_max", "target_count"])
def test_nonfinite_selection_setting_is_rejected(key, value):
    raw = base_profile()
    raw["image_selection"][key] = value
    assert key in _error(raw)


@pytest.mark.parametrize("key", ["corner_clip_max", "saturation_max"])
def test_selection_fraction_above_one_is_rejected(key):
    raw = base_profile()
    raw["image_selection"][key] = 1.01
    assert key in _error(raw)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_timeout_is_rejected(value):
    raw = base_profile()
    raw["execution"] = {"timeout_sec": value}
    assert "timeout_sec" in _error(raw)


def test_default_profile_loads_and_matches_existing_cli_defaults():
    profile = load_profile("default", camera_model="SG3S-ISX031C")
    assert profile.source_path == "builtin:default"
    assert profile.camera["model"] == "SG3S-ISX031C"
    assert profile.calibration["seed"] == 42
    assert profile.calibration["test_ratio"] == 0.25
    assert profile.image_selection_enabled is True
    assert profile.image_selection.target_count == 35
    # 자동화 기본값: 재현성을 위해 OpenCV 단일 스레드, 노출 판정은 pattern
    assert profile.opencv_threads == 1
    assert profile.image_selection.exposure_mode == "pattern"
    # 기본 프로파일은 운영 임계값을 확정하지 않는다
    assert profile.acceptance.thresholds
    assert all(rule.limit is None for rule in profile.acceptance.thresholds.values())


def test_builtin_default_profile_is_package_data():
    from importlib import resources

    from automation.profile import builtin_profile_names

    assert "default" in builtin_profile_names()
    assert resources.files("automation").joinpath("profiles", "default.yaml").is_file()


def test_unknown_profile_name_lists_builtins():
    with pytest.raises(ProfileError, match="default"):
        load_profile("no-such-profile")


def test_profile_file_round_trip_through_yaml(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(base_profile()), encoding="utf-8")
    profile = load_profile(path)
    assert profile.pattern["squares_x"] == 7
    assert profile.source_path == str(path.resolve())


def test_missing_profile_file_is_reported(tmp_path):
    with pytest.raises(ProfileError, match="찾을 수 없습니다"):
        load_profile(tmp_path / "nope.yaml")


def test_invalid_yaml_is_reported(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("pattern: [unclosed", encoding="utf-8")
    with pytest.raises(ProfileError, match="YAML"):
        load_profile(path)


# ---------------------------------------------------------------------------
# Pattern validation (기존 CLI 규칙 + ID 개수)
# ---------------------------------------------------------------------------


def test_missing_required_pattern_dimension_is_blocked():
    pattern = pattern_section()
    del pattern["squares_x"]
    assert "pattern.squares_x는 필수" in _error(base_profile(pattern=pattern))


def test_marker_not_smaller_than_square_is_blocked_by_existing_cli_rule():
    pattern = pattern_section() | {"marker_size": 0.05}
    message = _error(base_profile(pattern=pattern))
    assert "--marker-size는 --square-size보다 작아야" in message


def test_charuco_without_marker_size_is_blocked():
    pattern = pattern_section()
    del pattern["marker_size"]
    assert "marker-size" in _error(base_profile(pattern=pattern))


def test_unknown_dictionary_is_blocked():
    pattern = pattern_section() | {"dictionary": "DICT_NOT_REAL"}
    assert "DICT_NOT_REAL" in _error(base_profile(pattern=pattern))


def test_board_needing_more_ids_than_dictionary_is_blocked():
    # 12x12 ChArUco = 72 markers > DICT_4X4_50
    pattern = pattern_section() | {"squares_x": 12, "squares_y": 12, "dictionary": "DICT_4X4_50"}
    message = _error(base_profile(pattern=pattern))
    assert "72개" in message and "50개" in message


def test_aprilgrid_requires_apriltag_dictionary():
    pattern = pattern_section() | {"type": "aprilgrid", "dictionary": "DICT_5X5_100"}
    assert "AprilTag" in _error(base_profile(pattern=pattern))


def test_unknown_pattern_type_is_blocked():
    pattern = pattern_section() | {"type": "hexagon"}
    assert "hexagon" in _error(base_profile(pattern=pattern))


def test_chessboard_does_not_need_marker_or_dictionary():
    pattern = {"type": "chessboard", "squares_x": 9, "squares_y": 6, "square_size": 0.025}
    profile = _resolve(base_profile(pattern=pattern))
    assert profile.pattern["type"] == "chessboard"


# ---------------------------------------------------------------------------
# Resolution / structure / calibration options
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("camera", [
    {"width": -1, "height": 720},
    {"width": 0, "height": 720},
    {"width": 1280.5, "height": 720},
    {"width": "1280", "height": 720},
])
def test_invalid_resolution_is_blocked(camera):
    assert "camera.width" in _error(base_profile(camera=camera))


def test_width_without_height_is_blocked():
    assert "함께 지정" in _error(base_profile(camera={"width": 1280}))


def test_unknown_keys_are_reported_not_ignored():
    raw = base_profile()
    raw["pattern"]["squares"] = 7
    raw["mystery"] = 1
    message = _error(raw)
    assert "squares" in message and "mystery" in message


def test_automation_owned_cli_options_cannot_be_set_in_profile():
    raw = base_profile(calibration={"images": ["x"], "output_dir": "y"})
    message = _error(raw)
    assert "images" in message and "output_dir" in message


@pytest.mark.parametrize("ratio", [0.0, 1.0, 1.5, -0.1, "0.25"])
def test_test_ratio_out_of_range_is_blocked(ratio):
    assert "test_ratio" in _error(base_profile(calibration={"test_ratio": ratio}))


def test_calibration_model_choices_are_checked_by_existing_cli_parser():
    message = _error(base_profile(calibration={"models": ["pinhole", "magic_lens"]}))
    assert "기존 CLI가 설정을 거부" in message


def test_forced_model_must_be_in_models_list_existing_cli_rule():
    message = _error(base_profile(calibration={"models": ["pinhole"], "model": "fisheye"}))
    assert "--model" in message


def test_image_selection_keys_and_types_are_checked():
    message = _error(base_profile(image_selection={"enabled": True, "target_count": 0, "unknown": 1}))
    assert "unknown" in message and "target_count" in message


def test_acceptance_threshold_direction_is_checked():
    raw = base_profile(acceptance={"thresholds": {"test_rms_px": {"min": 0.5}}})
    assert "'max' 기준" in _error(raw)


def test_multiple_problems_are_reported_together():
    raw = base_profile(camera={"width": -5, "height": 720}, calibration={"test_ratio": 2})
    message = _error(raw)
    assert "camera.width" in message and "test_ratio" in message


# ---------------------------------------------------------------------------
# Priority: command line > profile > existing CLI defaults
# ---------------------------------------------------------------------------


def test_command_line_overrides_profile():
    raw = base_profile(calibration={"seed": 7, "test_ratio": 0.25})
    raw["camera"] = {"model": "FROM-PROFILE"}
    raw["output"] = {"root": "/profile/root"}
    profile = _resolve(raw, camera_model="FROM-CLI", seed=123, output_root="/cli/root")
    assert profile.camera["model"] == "FROM-CLI"
    assert profile.calibration["seed"] == 123
    assert profile.output_root == "/cli/root"


def test_profile_value_used_when_command_line_absent():
    profile = _resolve(base_profile(calibration={"seed": 7}))
    assert profile.calibration["seed"] == 7


def test_existing_cli_defaults_apply_when_profile_omits_option():
    raw = base_profile(calibration={})
    profile = _resolve(raw)
    argv = build_cli_argv(profile, image_paths=["a.png"], sensor_name="cam",
                          save_project="p.ccproj", json_summary="s.json", log_file="l.log")
    assert "--seed" not in argv and "--test-ratio" not in argv
    args, _ = parse_cli_argv(argv)
    assert args.seed == 42 and args.test_ratio == 0.25


# ---------------------------------------------------------------------------
# CLI argv generation uses the existing parser
# ---------------------------------------------------------------------------


def test_built_argv_is_accepted_by_existing_cli_parser():
    raw = base_profile(calibration={
        "models": ["pinhole", "brown_conrady"], "seed": 5, "test_ratio": 0.3,
        "outlier": True, "jobs": 1,
    })
    raw["camera"] = {"model": "CAM", "width": 1280, "height": 720}
    profile = _resolve(raw)
    argv = build_cli_argv(profile, image_paths=["a.png", "b.png"], sensor_name="CAM_SN1",
                          save_project="p.ccproj", json_summary="s.json", log_file="l.log")
    assert "--output-dir" not in argv  # Output Session 레이아웃을 쓰기 위해 생략
    args, pattern = parse_cli_argv(argv)
    assert args.images == ["a.png", "b.png"]
    assert args.models == ["pinhole", "brown_conrady"]
    assert args.seed == 5 and args.test_ratio == 0.3 and args.outlier is True
    assert (args.width, args.height) == (1280, 720)
    assert args.sensor_name == "CAM_SN1"
    assert set(args.export) == {"opencv", "ros", "report", "json", "csv"}
    assert pattern.squares_x == 7 and pattern.dictionary == "DICT_5X5_100"


def test_false_boolean_option_is_not_emitted():
    profile = _resolve(base_profile(calibration={"outlier": False}))
    argv = build_cli_argv(profile, image_paths=["a.png"], sensor_name="c",
                          save_project="p", json_summary="s", log_file="l")
    assert "--outlier" not in argv


@pytest.mark.parametrize("threads", [0, -1, 1.5, "1", True])
def test_invalid_opencv_threads_is_blocked(threads):
    raw = base_profile()
    raw["execution"] = {"opencv_threads": threads}
    assert "opencv_threads" in _error(raw)


def test_opencv_threads_default_is_unchanged_cli_behaviour():
    from automation.runner import cli_command

    assert _resolve(base_profile()).opencv_threads is None
    assert cli_command(Path("c.yaml"))[1:] == ["-m", "app.cli", "--config", "c.yaml"]
    pinned = cli_command(Path("c.yaml"), 1)
    assert pinned[1] == "-c" and "cv2.setNumThreads(1)" in pinned[2] and pinned[-2:] == ["--config", "c.yaml"]

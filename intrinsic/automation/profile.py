"""Automation profile (YAML) loading, validation and CLI argv generation.

설정 중복을 피하기 위한 원칙:
  * ``pattern``과 ``calibration`` 섹션의 키는 기존 ``app.cli``의 argparse dest
    이름을 그대로 쓴다 (``--config`` 파일과 같은 이름 규칙). 자동화 계층은 이
    값들을 해석하지 않고 그대로 CLI 인자로 바꿔 넘긴다.
  * 실행 전 검증도 새 규칙을 만들지 않고 기존 CLI의 파서
    (``build_arg_parser``), 정규화(``_normalize_cli_args``), 패턴 검증
    (``_build_pattern_config``)을 그대로 호출한다. 그래서 "검증은 통과했는데
    실제 실행에서 거부"되는 불일치가 생기지 않는다.
  * 자동화 계층만 아는 정보(카메라 식별, 이미지 선별, 합격 기준, 출력
    위치)만 별도 섹션으로 둔다.

우선순위: 명령행 인자 > 프로파일 > 기존 CLI 기본값.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from automation.acceptance import AcceptancePolicy, AcceptancePolicyError, parse_policy

PROFILE_VERSION = 1

_TOP_LEVEL_KEYS = {
    "profile_version", "camera", "pattern", "calibration",
    "image_selection", "acceptance", "output", "execution",
}
_CAMERA_KEYS = {"model", "width", "height"}
# 기존 CLI dest 이름 그대로 ("type"만 가독성을 위한 별칭으로 --pattern에 대응)
_PATTERN_KEYS = {
    "type", "squares_x", "squares_y", "square_size", "marker_size",
    "dictionary", "circle_grid_type", "aprilgrid_variant",
}
# 계산 옵션으로 허용하는 기존 CLI dest. 입력/출력/모드 전환 옵션은 자동화
# 계층이 직접 관리하므로 프로파일에서 줄 수 없다.
_CALIBRATION_KEYS = {
    "models", "model", "test_ratio", "seed", "calibration_method",
    "outlier", "corner_outlier", "max_iterations", "diagnostic",
    "kfold", "kfold_repeats", "repeatability", "bootstrap", "bootstrap_ci",
    "n_bootstrap", "jobs", "cache_dir", "robust_optimization", "capture_guidance_mask",
}
_OUTPUT_KEYS = {"root", "exports"}
_EXECUTION_KEYS = {"timeout_sec", "opencv_threads"}
# 결과 수집과 판정에 반드시 필요한 Export (calibration.ccproj는 별도로 항상 저장)
DEFAULT_EXPORTS = ["opencv", "ros", "report", "json", "csv"]


class ProfileError(ValueError):
    """프로파일/입력 설정 오류. 계산을 시작하지 않는다."""


@dataclass
class AutomationProfile:
    source_path: str | None
    camera: dict[str, Any]
    pattern: dict[str, Any]
    calibration: dict[str, Any]
    image_selection_enabled: bool
    image_selection: Any  # calibration.image_selection.ImageSelectionConfig
    acceptance: AcceptancePolicy
    output_root: str | None
    exports: list[str]
    timeout_sec: float | None
    opencv_threads: int | None = None

    def to_dict(self) -> dict:
        return {
            "profile_version": PROFILE_VERSION,
            "source_path": self.source_path,
            "camera": dict(self.camera),
            "pattern": dict(self.pattern),
            "calibration": dict(self.calibration),
            "image_selection": {
                "enabled": self.image_selection_enabled,
                **dataclasses.asdict(self.image_selection),
            },
            "acceptance": self.acceptance.to_dict(),
            "output": {"root": self.output_root, "exports": list(self.exports)},
            "execution": {"timeout_sec": self.timeout_sec, "opencv_threads": self.opencv_threads},
        }


def builtin_profile_names() -> list[str]:
    """패키지에 포함된 프로파일 이름 (automation/profiles/*.yaml).

    설치(pip install .) 후에도 저장소 경로와 무관하게 ``--profile default``로 찾는다.
    """
    from importlib import resources

    folder = resources.files("automation").joinpath("profiles")
    return sorted(
        entry.name[: -len(".yaml")] for entry in folder.iterdir() if entry.name.endswith(".yaml")
    )


def _profile_text(spec: str | Path) -> tuple[str, str]:
    """``spec``이 존재하는 파일이면 그 파일, 아니면 내장 프로파일 이름으로 찾는다.

    반환: (YAML 텍스트, 출처 표기)
    """
    path = Path(spec).expanduser()
    if path.is_file():
        return path.read_text(encoding="utf-8"), str(path.resolve())
    name = str(spec)
    if name in builtin_profile_names():
        from importlib import resources

        text = resources.files("automation").joinpath("profiles", f"{name}.yaml").read_text(encoding="utf-8")
        return text, f"builtin:{name}"
    raise ProfileError(
        f"프로파일을 찾을 수 없습니다: {spec} "
        f"(파일 경로 또는 내장 프로파일 이름: {', '.join(builtin_profile_names())})"
    )


def read_profile_file(spec: str | Path) -> dict:
    return _read_profile(spec)[0]


def _read_profile(spec: str | Path) -> tuple[dict, str]:
    text, source = _profile_text(spec)
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ProfileError(f"프로파일 YAML을 읽을 수 없습니다: {source}\n  {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ProfileError("프로파일 최상위는 key: value 객체여야 합니다.")
    return data, source


def _section(raw: dict, name: str, allowed: set[str], problems: list[str]) -> dict:
    value = raw.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        problems.append(f"{name} 섹션은 key: value 객체여야 합니다.")
        return {}
    normalized = {str(k).replace("-", "_"): v for k, v in value.items()}
    unknown = set(normalized) - allowed
    if unknown:
        problems.append(
            f"{name} 섹션에 알 수 없는 키: {', '.join(sorted(unknown))} "
            f"(허용: {', '.join(sorted(allowed))})"
        )
    return {k: v for k, v in normalized.items() if k in allowed}


def _positive_int_or_none(value: Any, label: str, problems: list[str]) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        problems.append(f"{label}는 양의 정수여야 합니다 (입력: {value!r}).")
        return None
    return value


def _image_selection_config(section: dict, problems: list[str]):
    from calibration.image_selection import ImageSelectionConfig

    fields = {f.name: f for f in dataclasses.fields(ImageSelectionConfig)}
    enabled = section.get("enabled", True)
    if not isinstance(enabled, bool):
        problems.append("image_selection.enabled는 true/false여야 합니다.")
        enabled = True
    values: dict[str, Any] = {}
    defaults = ImageSelectionConfig()
    for key, value in section.items():
        if key == "enabled":
            continue
        if key not in fields:
            problems.append(
                f"image_selection에 알 수 없는 키: {key} "
                f"(허용: enabled, {', '.join(fields)})"
            )
            continue
        expected = type(getattr(defaults, key))
        if key == "exposure_mode":
            from calibration.image_selection import EXPOSURE_MODES

            if value not in EXPOSURE_MODES:
                problems.append(f"image_selection.exposure_mode는 {', '.join(EXPOSURE_MODES)} 중 하나여야 합니다.")
            else:
                values[key] = value
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"image_selection.{key}는 숫자여야 합니다 (입력: {value!r}).")
            continue
        if not math.isfinite(value):
            problems.append(f"image_selection.{key}는 유한한 숫자여야 합니다.")
            continue
        if expected is int and not isinstance(value, int):
            problems.append(f"image_selection.{key}는 정수여야 합니다 (입력: {value!r}).")
            continue
        if value < 0:
            problems.append(f"image_selection.{key}는 음수일 수 없습니다.")
            continue
        if key in ("corner_clip_max", "saturation_max") and value > 1:
            problems.append(f"image_selection.{key}는 0과 1 사이여야 합니다.")
            continue
        values[key] = expected(value)
    if values.get("target_count", defaults.target_count) < 1:
        problems.append("image_selection.target_count는 1 이상이어야 합니다.")
    return enabled, dataclasses.replace(defaults, **values)


def resolve_profile(
    raw: dict,
    *,
    source_path: str | None = None,
    camera_model: str | None = None,
    seed: int | None = None,
    output_root: str | None = None,
) -> AutomationProfile:
    """원본 프로파일 dict + 명령행 override → 검증된 AutomationProfile.

    오류는 한 번에 모아서 ProfileError로 보고한다 (하나 고치고 또 실패하는
    반복을 줄이기 위해).
    """
    raw = copy.deepcopy(raw)
    problems: list[str] = []
    unknown = set(raw) - _TOP_LEVEL_KEYS
    if unknown:
        problems.append(f"알 수 없는 최상위 키: {', '.join(sorted(unknown))}")
    version = raw.get("profile_version", PROFILE_VERSION)
    if version != PROFILE_VERSION:
        problems.append(f"지원하지 않는 profile_version: {version!r} (지원: {PROFILE_VERSION})")

    camera = _section(raw, "camera", _CAMERA_KEYS, problems)
    pattern = _section(raw, "pattern", _PATTERN_KEYS, problems)
    calibration = _section(raw, "calibration", _CALIBRATION_KEYS, problems)
    selection_raw = _section(raw, "image_selection", _image_selection_keys(), problems)
    output = _section(raw, "output", _OUTPUT_KEYS, problems)
    execution = _section(raw, "execution", _EXECUTION_KEYS, problems)

    # 명령행 override (우선순위: CLI > profile)
    if camera_model is not None:
        camera["model"] = camera_model
    if seed is not None:
        calibration["seed"] = seed
    if output_root is not None:
        output["root"] = output_root

    camera["width"] = _positive_int_or_none(camera.get("width"), "camera.width", problems)
    camera["height"] = _positive_int_or_none(camera.get("height"), "camera.height", problems)
    if (camera["width"] is None) != (camera["height"] is None):
        problems.append("camera.width와 camera.height는 함께 지정하거나 함께 생략해야 합니다.")

    for required in ("squares_x", "squares_y", "square_size"):
        if pattern.get(required) is None:
            problems.append(f"pattern.{required}는 필수입니다.")

    # 기존 CLI는 test_ratio 범위를 검사하지 않는다. 1 이상이면 Train이 비고
    # 0 이하면 Hold-out이 사라지므로 자동화 계층에서 먼저 막는다.
    ratio = calibration.get("test_ratio")
    if ratio is not None and (
        isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0.0 < ratio < 1.0
    ):
        problems.append(f"calibration.test_ratio는 0과 1 사이 숫자여야 합니다 (입력: {ratio!r}).")
    seed_value = calibration.get("seed")
    if seed_value is not None and (isinstance(seed_value, bool) or not isinstance(seed_value, int)):
        problems.append(f"calibration.seed는 정수여야 합니다 (입력: {seed_value!r}).")

    exports = output.get("exports", list(DEFAULT_EXPORTS))
    if not isinstance(exports, list) or not all(isinstance(e, str) for e in exports):
        problems.append("output.exports는 문자열 목록이어야 합니다.")
        exports = list(DEFAULT_EXPORTS)
    root = output.get("root")
    if root is not None and not isinstance(root, str):
        problems.append("output.root는 경로 문자열이어야 합니다.")
        root = None

    timeout = execution.get("timeout_sec")
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0):
        problems.append("execution.timeout_sec는 양수 또는 null이어야 합니다.")
        timeout = None
    # OpenCV 내부 병렬 처리는 실행마다 부동소수점 합산 순서가 달라져 K/RMS가
    # 1e-6 수준으로 흔들린다(실측). 1로 고정하면 같은 입력에서 비트 단위로 같은
    # 결과가 나온다. null = 기존 CLI/GUI와 같은 기본 스레드 수.
    threads = execution.get("opencv_threads")
    if threads is not None and (isinstance(threads, bool) or not isinstance(threads, int) or threads < 1):
        problems.append("execution.opencv_threads는 1 이상의 정수 또는 null이어야 합니다.")
        threads = None

    enabled, selection_config = _image_selection_config(selection_raw, problems)

    try:
        policy = parse_policy(raw.get("acceptance"))
    except AcceptancePolicyError as exc:
        problems.append(str(exc))
        policy = parse_policy(None)

    profile = AutomationProfile(
        source_path=source_path,
        camera=camera,
        pattern=pattern,
        calibration=calibration,
        image_selection_enabled=enabled,
        image_selection=selection_config,
        acceptance=policy,
        output_root=root,
        exports=list(dict.fromkeys(exports)),
        timeout_sec=None if timeout is None else float(timeout),
        opencv_threads=threads,
    )

    if not problems:
        # 구조가 맞을 때만 실제 CLI 규칙으로 의미 검증을 한다.
        problems.extend(validate_against_cli(profile))
    if problems:
        raise ProfileError("\n".join(f"- {p}" for p in problems))
    return profile


def _image_selection_keys() -> set[str]:
    from calibration.image_selection import ImageSelectionConfig

    return {"enabled", *(f.name for f in dataclasses.fields(ImageSelectionConfig))}


# ---------------------------------------------------------------------------
# 기존 CLI 옵션으로 변환 + 기존 CLI 규칙으로 검증
#
# 옵션은 한 dict(dest -> 값)에서 두 형태로 만든다.
#   * argv  : 검증용. argparse의 type/choices 검사를 그대로 받는다.
#   * config: 실행용. 기존 CLI ``--config`` 파일 (이미지 수백 장도 명령줄
#             길이 제한 없이 전달, 그대로 재실행 가능한 기록이 됨).
# 두 경로가 같은 Namespace를 만드는지는 runner가 실행 전에 대조한다.
# ---------------------------------------------------------------------------


def _option_for_dest(parser, dest: str):
    for action in parser._actions:  # argparse 공개 속성은 아니지만 app.cli.main도 같은 방식으로 dest를 수집한다
        if action.dest == dest and action.option_strings:
            return action
    raise KeyError(dest)


def _plain(value: Any) -> Any:
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return getattr(value, "value", value)


def build_cli_options(
    profile: AutomationProfile,
    *,
    image_paths: list[str],
    sensor_name: str,
    save_project: str,
    json_summary: str,
    log_file: str,
    session_dir: str | None = None,
) -> dict[str, Any]:
    """기존 ``app.cli`` dest 이름 → 값. ``--output-dir``는 넣지 않는다."""
    options: dict[str, Any] = {"images": list(image_paths)}
    pattern = dict(profile.pattern)
    if "type" in pattern:
        pattern["pattern"] = pattern.pop("type")
    for dest in ("pattern", "squares_x", "squares_y", "square_size", "marker_size",
                 "dictionary", "circle_grid_type", "aprilgrid_variant"):
        if pattern.get(dest) is not None:
            options[dest] = _plain(pattern[dest])
    if profile.camera.get("width") is not None:
        options["width"] = profile.camera["width"]
        options["height"] = profile.camera["height"]
    for dest in sorted(profile.calibration):
        value = profile.calibration[dest]
        if dest == "cache_dir" and value is not None:
            # 자식 프로세스는 세션 폴더에서 실행되므로 상대경로를 미리 고정한다
            value = str(Path(value).expanduser().resolve())
        if value is not None:
            options[dest] = _plain(value)
    options["sensor_name"] = sensor_name
    options["export"] = list(profile.exports)
    options["save_project"] = save_project
    options["json_summary"] = json_summary
    options["log_file"] = log_file
    if session_dir is not None:
        options["session_dir"] = session_dir
    return options


def options_to_argv(options: dict[str, Any]) -> list[str]:
    import argparse

    from app.cli import build_arg_parser

    parser = build_arg_parser()
    argv: list[str] = []
    for dest, value in options.items():
        action = _option_for_dest(parser, dest)
        flag = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if not isinstance(value, bool):
                raise ProfileError(f"{dest}는 true/false여야 합니다 (입력: {value!r}).")
            if value:
                argv.append(flag)
            continue
        if value is None:
            continue
        if action.nargs in ("+", "*"):
            argv.append(flag)
            argv.extend(str(v) for v in (value if isinstance(value, list) else [value]))
            continue
        argv.extend([flag, str(value)])
    return argv


def build_cli_argv(profile: AutomationProfile, **kwargs: Any) -> list[str]:
    return options_to_argv(build_cli_options(profile, **kwargs))


def write_cli_config(options: dict[str, Any], path: str | Path) -> Path:
    """기존 CLI ``--config`` 형식(최상위 dest: 값)으로 저장한다."""
    from automation.run_manifest import atomic_write_text

    return atomic_write_text(path, yaml.safe_dump(options, allow_unicode=True, sort_keys=False))


def parse_cli_argv(argv: list[str]):
    """기존 CLI의 파서/정규화/패턴 검증을 그대로 실행한다.

    반환: (args, pattern_config). 실패하면 CLI가 내는 메시지 그대로 ProfileError.
    """
    from app.cli import build_arg_parser

    return _parse_with_cli(build_arg_parser(), argv)


def parse_cli_config(path: str | Path):
    """``python -m app.cli --config <path>``와 같은 순서로 해석한다 (app.cli.main과 동일)."""
    from app.cli import CliError, _load_config_file, build_arg_parser

    parser = build_arg_parser()
    try:
        values = _load_config_file(str(path), {a.dest for a in parser._actions})
    except CliError as exc:
        raise ProfileError(f"기존 CLI가 config를 거부했습니다: {exc}") from None
    parser.set_defaults(**values)
    return _parse_with_cli(parser, ["--config", str(path)])


def _parse_with_cli(parser, argv: list[str]):
    from app.cli import CliError, _build_pattern_config, _normalize_cli_args

    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            args = _normalize_cli_args(parser.parse_args(argv), parser)
    except SystemExit as exc:
        message = stderr.getvalue().strip().splitlines()
        detail = message[-1] if message else f"argparse exit {exc.code}"
        raise ProfileError(f"기존 CLI가 설정을 거부했습니다: {detail}") from None
    for required in ("squares_x", "squares_y", "square_size"):
        if getattr(args, required) is None:
            raise ProfileError(f"pattern.{required}는 필수입니다.")
    try:
        pattern_config = _build_pattern_config(args)
    except CliError as exc:
        raise ProfileError(f"패턴 설정 오류: {exc}") from None
    return args, pattern_config


def check_pattern_identifiers(pattern_config) -> list[str]:
    """보드에 필요한 marker/tag ID 개수가 dictionary 크기 안에 드는지 확인하고,
    기존 검출기 생성 경로(build_detect_fn)로 보드를 실제로 만들어 본다."""
    import cv2

    from calibration.detector import build_detect_fn
    from calibration.types import PatternType

    problems: list[str] = []
    ptype = pattern_config.type
    needed = None
    if ptype == PatternType.CHARUCO:
        needed = (pattern_config.squares_x * pattern_config.squares_y) // 2
        what = "ChArUco 마커"
    elif ptype == PatternType.APRILGRID:
        needed = pattern_config.squares_x * pattern_config.squares_y
        what = "AprilGrid 태그"
    if needed is not None:
        dict_id = getattr(cv2.aruco, str(pattern_config.dictionary), None)
        if dict_id is None:
            problems.append(f"알 수 없는 dictionary: {pattern_config.dictionary}")
            return problems
        capacity = int(cv2.aruco.getPredefinedDictionary(dict_id).bytesList.shape[0])
        if needed > capacity:
            problems.append(
                f"{what} {needed}개가 필요하지만 {pattern_config.dictionary}에는 ID가 "
                f"{capacity}개뿐입니다 (squares {pattern_config.squares_x}x{pattern_config.squares_y})."
            )
            return problems
    try:
        build_detect_fn(pattern_config)
    except (ValueError, cv2.error) as exc:
        problems.append(f"패턴 검출기를 만들 수 없습니다: {exc}")
    return problems


def validate_against_cli(profile: AutomationProfile) -> list[str]:
    placeholder = "__automation_validation__"
    try:
        argv = build_cli_argv(
            profile,
            image_paths=[placeholder],
            sensor_name="validation",
            save_project=placeholder,
            json_summary=placeholder,
            log_file=placeholder,
        )
        _args, pattern_config = parse_cli_argv(argv)
    except ProfileError as exc:
        return [str(exc)]
    return check_pattern_identifiers(pattern_config)


def load_profile(spec: str | Path, **overrides: Any) -> AutomationProfile:
    """``spec``: 프로파일 파일 경로 또는 내장 프로파일 이름(예: ``default``)."""
    data, source = _read_profile(spec)
    return resolve_profile(data, source_path=source, **overrides)

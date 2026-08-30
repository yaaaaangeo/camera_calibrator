"""
camera_calibrator.export.kalibr
================================

Kalibr 연동용 export.

Kalibr의 AprilGrid 정의는 이 프로젝트의 PatternConfig와 거의 같지만,
간격 표현만 다르다. 이 프로젝트는 square_size를 "태그 origin 사이 pitch"
로 쓰고, Kalibr은 tagSpacing을 "태그 사이 빈 공간 / tagSize" 비율로 쓴다.
따라서:

    square_size = tagSize * (1 + tagSpacing)
    tagSpacing = (square_size - marker_size) / marker_size
"""

from __future__ import annotations

from pathlib import Path

from calibration.types import PatternConfig, PatternType


def build_kalibr_aprilgrid_target(pattern: PatternConfig) -> dict:
    """PatternConfig를 Kalibr aprilgrid target dict로 변환한다."""
    if pattern.type != PatternType.APRILGRID:
        raise ValueError("Kalibr AprilGrid target export는 apriltag_grid 패턴만 지원합니다.")
    if pattern.marker_size is None or pattern.marker_size <= 0:
        raise ValueError("Kalibr AprilGrid target export에는 marker_size가 필요합니다.")
    if pattern.square_size <= pattern.marker_size:
        raise ValueError("Kalibr AprilGrid target export에는 square_size > marker_size가 필요합니다.")

    tag_spacing = (pattern.square_size - pattern.marker_size) / pattern.marker_size
    return {
        "target_type": "aprilgrid",
        "tagCols": int(pattern.squares_x),
        "tagRows": int(pattern.squares_y),
        "tagSize": float(pattern.marker_size),
        "tagSpacing": float(tag_spacing),
    }


def export_kalibr_target_yaml(pattern: PatternConfig, path: str) -> str:
    """Kalibr calibration target YAML 파일을 쓴다.

    PyYAML 의존 없이 Kalibr 예시와 같은 간단한 YAML을 직접 쓴다.
    """
    target = build_kalibr_aprilgrid_target(pattern)
    text = (
        "target_type: 'aprilgrid'\n"
        f"tagCols: {target['tagCols']}\n"
        f"tagRows: {target['tagRows']}\n"
        f"tagSize: {target['tagSize']:.12g}\n"
        f"tagSpacing: {target['tagSpacing']:.12g}\n"
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return path


def build_kalibr_camera_calibration_command(
    *,
    bag_path: str,
    topic: str,
    target_yaml_path: str,
    camera_model: str = "pinhole-radtan",
) -> str:
    """Kalibr camera calibration 실행 예시 command를 만든다."""
    return (
        "kalibr_calibrate_cameras "
        f"--bag {bag_path} "
        f"--topics {topic} "
        f"--models {camera_model} "
        f"--target {target_yaml_path}"
    )

"""
scripts/generate_legacy_project_fixtures.py
=================================================

tests/assets/projects/*.ccproj (P1-D legacy migration fixtures) 생성 스크립트.

.ccproj의 JSON 스키마는 CalibrationProject dataclass 구조에서 나오고,
project_io.migrate_v1_to_v2()는 format_version과 "extended_pinhole" 키/
model_name 문자열, distortion 길이만 검사하므로 - 지금 project_to_dict()로
정상 직렬화한 뒤 format_version만 1로 되돌리고 필요하면 모델 키를
"extended_pinhole"로 바꿔치기하는 것으로 실제 구버전 파일과 동등한
fixture를 만들 수 있다 (다른 필드 스키마는 v1/v2 사이에 변경되지 않았음).

재실행 가능 - 매번 같은 4개 파일을 덮어쓴다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from calibration.project_io import project_to_dict
from calibration.types import (
    CalibrationProject,
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    PatternConfig,
    PatternType,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "assets" / "projects"


def _camera_config() -> CameraConfig:
    return CameraConfig(width=640, height=480)


def _pattern_config() -> PatternConfig:
    return PatternConfig(type=PatternType.CHARUCO, squares_x=7, squares_y=5, square_size=0.04, marker_size=0.03, dictionary="DICT_5X5_100")


def _camera_matrix() -> np.ndarray:
    return np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])


def _write(name: str, payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"wrote {path}")


def build_v1_pinhole() -> None:
    project = CalibrationProject(
        project_name="legacy-pinhole",
        camera_config=_camera_config(),
        pattern_config=_pattern_config(),
        calibration_results={
            CameraModelType.PINHOLE: CalibrationResult(
                model_name=CameraModelType.PINHOLE,
                camera_matrix=_camera_matrix(),
                distortion=np.zeros(0),
                rms_error=0.3,
                success=True,
            ),
        },
    )
    payload = project_to_dict(project)
    payload["format_version"] = 1
    _write("v1_pinhole.ccproj", payload)


def build_v1_extended_5coeff() -> None:
    project = CalibrationProject(
        project_name="legacy-extended-5coeff",
        camera_config=_camera_config(),
        pattern_config=_pattern_config(),
        calibration_results={
            CameraModelType.EXTENDED_PINHOLE: CalibrationResult(
                model_name=CameraModelType.EXTENDED_PINHOLE,
                camera_matrix=_camera_matrix(),
                distortion=np.array([-0.2, 0.05, 0.001, -0.001, 0.01]),  # k1,k2,p1,p2,k3 (5)
                rms_error=0.4,
                success=True,
            ),
        },
    )
    payload = project_to_dict(project)
    payload["format_version"] = 1
    _write("v1_extended_5coeff.ccproj", payload)


def build_v1_extended_rational() -> None:
    project = CalibrationProject(
        project_name="legacy-extended-rational",
        camera_config=_camera_config(),
        pattern_config=_pattern_config(),
        calibration_results={
            CameraModelType.EXTENDED_PINHOLE: CalibrationResult(
                model_name=CameraModelType.EXTENDED_PINHOLE,
                camera_matrix=_camera_matrix(),
                # k1,k2,p1,p2,k3,k4,k5,k6 (8) - 오늘날 Rational과 같은 의미
                distortion=np.array([-0.2, 0.05, 0.001, -0.001, 0.01, 0.02, -0.01, 0.003]),
                rms_error=0.35,
                success=True,
            ),
        },
    )
    payload = project_to_dict(project)
    payload["format_version"] = 1
    _write("v1_extended_rational.ccproj", payload)


def build_v2_project() -> None:
    project = CalibrationProject(
        project_name="current-format",
        camera_config=_camera_config(),
        pattern_config=_pattern_config(),
        calibration_results={
            CameraModelType.BROWN_CONRADY: CalibrationResult(
                model_name=CameraModelType.BROWN_CONRADY,
                camera_matrix=_camera_matrix(),
                distortion=np.array([-0.2, 0.05, 0.001, -0.001, 0.01]),
                rms_error=0.32,
                success=True,
            ),
        },
    )
    payload = project_to_dict(project)  # format_version은 현재 PROJECT_FORMAT_VERSION 그대로
    _write("v2_project.ccproj", payload)


if __name__ == "__main__":
    build_v1_pinhole()
    build_v1_extended_5coeff()
    build_v1_extended_rational()
    build_v2_project()

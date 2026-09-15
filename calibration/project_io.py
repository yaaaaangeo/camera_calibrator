"""
camera_calibrator.calibration.project_io
=============================================

설계 문서 16번 - `.ccproj` 프로젝트 저장/불러오기.

pickle 대신 JSON을 쓴다 - `.ccproj` 파일은 나중에 동료와 공유하거나
버전관리에 올릴 수도 있는 파일인데, pickle.load()는 신뢰할 수 없는 파일을
열 때 임의 코드 실행 위험이 있다. JSON은 느리고 코드가 더 필요하지만 안전하고,
이 프로젝트가 이미 YAML/JSON을 export 포맷으로 쓰는 철학과도 맞는다.

이미지 파일 자체는 프로젝트 파일 안에 복사해 넣지 않는다 (설계 문서 9번의
"파일을 삭제/복제하지 않고 메타데이터만 다룬다" 원칙과 같은 이유) - 경로만
저장하고, 불러올 때 존재 여부를 확인해 없어진 파일은 경고로만 알려준다
(크래시하지 않음).

직렬화 전략:
    save: dataclasses.asdict()로 중첩 dataclass를 재귀적으로 dict화한 뒤,
          _json_safe()로 numpy 배열/Enum/datetime/tuple을 JSON이 이해하는
          타입으로 변환.
    load: 일반 dict에서 다시 정확한 dataclass 인스턴스로 복원 - 이 방향은
          타입 정보가 없어져서 필드별로 명시적으로 재구성해야 한다
          (필드마다 numpy dtype, Enum 클래스가 다르므로 자동화하지 않음).

Phase D-2 안정화 - 도메인별 dict -> dataclass 디코더(Intrinsic/Windshield/
Reflection/Ghost 각각 ~수백 줄)는 `calibration/project_codecs/` 패키지로
옮겼다(로직 변경 없이 순수 이동). 이 파일은 이제 오케스트레이션만 담당한다:
    - `project_to_dict`/`save_project` (write)
    - `migrate_v1_to_v2` (v1 -> v2 마이그레이션)
    - `project_from_dict`/`load_project` (read, 각 codec의 top-level
      `_xxx_from_dict`를 호출해 `CalibrationProject`를 조립)

기존에 `from calibration.project_io import _windshield_calibration_result_from_dict`
처럼 private 함수를 직접 가져다 쓰던 코드(예: tests/test_windshield_project_io.py,
tests/test_windshield_ghost.py)가 계속 그대로 동작하도록, 이 파일은 그 함수들을
codec 모듈에서 import해서 이 모듈의 이름공간에 다시 노출한다(재-export) - 공개
API를 이유 없이 바꾸지 않는다는 원칙을 그대로 지킨다.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from calibration.json_utils import json_safe
from calibration.project_codecs.common import _dt
from calibration.project_codecs.ghost import (
    _ghost_dataset_result_from_dict,
    _ghost_evaluation_result_from_dict,
    _ghost_field_from_dict,
    _ghost_point_detection_from_dict,
    _ghost_spatial_cell_from_dict,
)
from calibration.project_codecs.intrinsic import (
    _calibration_result_from_dict,
    _camera_config_from_dict,
    _cross_dataset_result_from_dict,
    _dataset_from_dict,
    _final_result_from_dict,
    _model_score_from_dict,
    _object_releasing_validation_result_from_dict,
    _optimizer_result_from_dict,
    _outlier_result_from_dict,
    _pattern_config_from_dict,
    _scene_quality_analysis_from_dict,
    _standard_vs_object_releasing_comparison_from_dict,
    _subset_calibration_result_from_dict,
    _validation_result_from_dict,
)
from calibration.project_codecs.reflection import _reflection_dataset_result_from_dict
from calibration.project_codecs.windshield import (
    _windshield_calibration_result_from_dict,
    _windshield_config_from_dict,
)
from calibration.types import CalibrationProject, CameraModelType, ExportFormat
from calibration.windshield.base import windshield_result_key_from_storage, windshield_result_key_to_storage

PROJECT_FORMAT_VERSION = 2
PROJECT_EXTENSION = ".ccproj"

logger = logging.getLogger(__name__)


def project_to_dict(project: CalibrationProject) -> dict:
    raw = dataclasses.asdict(project)
    if raw.get("windshield_results"):
        raw["windshield_results"] = {
            windshield_result_key_to_storage(k): v
            for k, v in raw["windshield_results"].items()
        }
    safe = json_safe(raw)
    return {"format_version": PROJECT_FORMAT_VERSION, "project": safe}


def save_project(project: CalibrationProject, path: str) -> str:
    project.updated_at = datetime.now()
    payload = project_to_dict(project)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _raw_array_len(d) -> int | None:
    """migrate_v1_to_v2용 - 아직 dataclass로 복원하지 않은 raw JSON 값에서
    distortion 배열 길이만 알고 싶을 때. _arr()과 같은 언랩 규칙(_json_safe가
    만드는 {"__ndarray__": True, "data": [...]} 포맷, 구버전의 순수 list 둘 다)을
    따르되 dtype 변환은 하지 않는다 - 여기서는 길이만 필요하다.
    """
    if d is None:
        return None
    if isinstance(d, dict) and d.get("__ndarray__"):
        d = d.get("data")
    if not isinstance(d, list):
        return None
    try:
        return int(np.asarray(d).reshape(-1).size)
    except (TypeError, ValueError):
        return None


def _migrate_model_name_refs(container, field_name: str, should_rename: bool) -> None:
    """container[field_name]이 문자열 "extended_pinhole"이면(그리고
    should_rename이면) "brown_conrady"로 바꾼다. dict 하나에 대한 최소 단위
    연산 - migrate_v1_to_v2가 여러 중첩 위치(ModelScore/CrossDatasetValidation
    Result/DiagnosisReport 등)에 동일하게 적용하기 위한 헬퍼.
    """
    if not should_rename or container is None:
        return
    if container.get(field_name) == "extended_pinhole":
        container[field_name] = "brown_conrady"


def _apply_model_map_to_field(container, field_name: str, legacy_model_map: dict[str, str]) -> None:
    if not isinstance(container, dict):
        return
    value = container.get(field_name)
    if value in legacy_model_map:
        container[field_name] = legacy_model_map[value]


def _apply_model_map_to_keyed_dict(container: dict, legacy_model_map: dict[str, str]) -> None:
    for old, new in list(legacy_model_map.items()):
        if old == new or old not in container:
            continue
        if new in container:
            logger.warning(
                "Legacy project contains both %s and %s entries; keeping existing %s entry.",
                old, new, new,
            )
            continue
        container[new] = container.pop(old)


def migrate_v1_to_v2(payload: dict) -> dict:
    """v1 -> v2 마이그레이션.

    v1 시절 "extended_pinhole"은 실제로는 두 가지 다른 의미로 쓰였을 수 있다:
      - distortion 5계수(k1,k2,p1,p2,k3) -> 지금의 Brown-Conrady 역할이었음
      - distortion 8계수 이상(k1~k6,p1,p2) -> 지금의 extended_pinhole(Rational)과 동일

    문자열만 보고 바꾸지 않는다 - calibration_results["extended_pinhole"]의
    distortion 벡터 길이로 프로젝트 전체에 딱 한 번 실제 의미를 판별한
    (should_rename_extended_to_brown) 뒤, 그 판정을 프로젝트 안의 모든 model
    reference(calibration_results/validation_results/model_scores/
    cross_dataset_results/final_result 하위 전부)에 동일하게 적용한다 -
    한 프로젝트 안에서 "extended_pinhole"이 어떤 곳에서는 Brown, 다른 곳에서는
    Rational을 가리키는 모순된 상태가 생기지 않도록.

    ModelScore/CrossDatasetValidationResult/DiagnosisReport 자체에는
    distortion vector가 없으므로 개별적으로 재판별하지 않는다 - 위에서 정한
    프로젝트 단위 판정을 그대로 물려받는다(사용자 스펙 6번 "Model Score
    migration 주의" 항목과 동일한 원칙).

    "pinhole"/"fisheye"는 v1과 v2에서 의미가 같으므로 손대지 않는다.
    object_releasing_result가 v1 payload에 아예 없는 경우는 project_from_dict의
    기존 .get(...) 처리로 이미 None으로 정상 로드되므로 여기서 손댈 필요 없다.
    """
    project = payload.get("project", {})
    calibration_results: dict = project.get("calibration_results", {}) or {}
    validation_results: dict = project.get("validation_results", {}) or {}

    legacy_model_map: dict[str, str] = {}

    legacy_entry = calibration_results.get("extended_pinhole")
    if legacy_entry is not None:
        dist_len = _raw_array_len(legacy_entry.get("distortion"))
        if dist_len == 5:
            legacy_model_map["extended_pinhole"] = "brown_conrady"
            logger.info(
                "Migrated legacy project model: extended_pinhole (5 coeffs) -> brown_conrady"
            )
        elif dist_len is not None and dist_len >= 8:
            legacy_model_map["extended_pinhole"] = "extended_pinhole"
            logger.info(
                "Legacy project model extended_pinhole (%d coeffs) already matches current "
                "Rational meaning - kept as extended_pinhole.", dist_len
            )
        elif dist_len is not None:
            logger.warning(
                "Legacy project model extended_pinhole has an unexpected distortion length "
                "(%d) - left unchanged, please verify manually.", dist_len
            )
        else:
            logger.warning(
                "Legacy project model extended_pinhole has no readable distortion vector - "
                "left unchanged, please verify manually."
            )

    _apply_model_map_to_field(project.get("camera_config"), "model", legacy_model_map)

    _apply_model_map_to_keyed_dict(calibration_results, legacy_model_map)
    for result in calibration_results.values():
        _apply_model_map_to_field(result, "model_name", legacy_model_map)

    _apply_model_map_to_keyed_dict(validation_results, legacy_model_map)
    for result in validation_results.values():
        _apply_model_map_to_field(result, "model_name", legacy_model_map)

    # 최상위 model_scores / cross_dataset_results 리스트.
    for score in project.get("model_scores", []) or []:
        _apply_model_map_to_field(score, "model_name", legacy_model_map)
    for cross_result in project.get("cross_dataset_results", []) or []:
        _apply_model_map_to_field(cross_result, "model_name", legacy_model_map)
    _apply_model_map_to_field(project.get("object_releasing_result"), "model_name", legacy_model_map)

    object_releasing_comparison = project.get("standard_vs_object_releasing_comparison")
    if object_releasing_comparison:
        _apply_model_map_to_field(
            object_releasing_comparison.get("standard_result"),
            "model_name",
            legacy_model_map,
        )
        _apply_model_map_to_field(
            object_releasing_comparison.get("object_releasing_result"),
            "model_name",
            legacy_model_map,
        )

    final_result = project.get("final_result")
    if final_result:
        _apply_model_map_to_field(final_result, "chosen_model", legacy_model_map)
        _apply_model_map_to_field(final_result.get("calibration"), "model_name", legacy_model_map)
        _apply_model_map_to_field(final_result.get("validation"), "model_name", legacy_model_map)
        for score in final_result.get("model_scores", []) or []:
            _apply_model_map_to_field(score, "model_name", legacy_model_map)
        _apply_model_map_to_field(final_result.get("diagnosis"), "model_name", legacy_model_map)

    if legacy_model_map.get("extended_pinhole") == "brown_conrady":
        logger.info(
            "Migrated legacy project model references (model_scores/cross_dataset_results/"
            "final_result): extended_pinhole -> brown_conrady"
        )

    payload["format_version"] = PROJECT_FORMAT_VERSION
    return payload


def project_from_dict(payload: dict) -> CalibrationProject:
    version = payload.get("format_version")
    if version == 1:
        payload = migrate_v1_to_v2(payload)
        version = payload["format_version"]
    if version != PROJECT_FORMAT_VERSION:
        raise ValueError(
            f"지원하지 않는 프로젝트 파일 버전입니다: {version} "
            f"(이 버전의 툴은 {PROJECT_FORMAT_VERSION}만 지원)"
        )
    d = payload["project"]

    calibration_results = {
        CameraModelType(k): _calibration_result_from_dict(v)
        for k, v in d.get("calibration_results", {}).items()
    }
    validation_results = {
        CameraModelType(k): _validation_result_from_dict(v)
        for k, v in d.get("validation_results", {}).items()
    }
    export_paths = {ExportFormat(k): v for k, v in d.get("export_paths", {}).items()}

    return CalibrationProject(
        project_name=d["project_name"],
        camera_config=_camera_config_from_dict(d["camera_config"]),
        pattern_config=_pattern_config_from_dict(d["pattern_config"]),
        dataset=_dataset_from_dict(d.get("dataset", {})),
        calibration_results=calibration_results,
        object_releasing_result=(
            _calibration_result_from_dict(d["object_releasing_result"])
            if d.get("object_releasing_result") else None
        ),
        object_releasing_validation_result=_object_releasing_validation_result_from_dict(
            d.get("object_releasing_validation_result")
        ),
        standard_vs_object_releasing_comparison=_standard_vs_object_releasing_comparison_from_dict(
            d.get("standard_vs_object_releasing_comparison")
        ),
        validation_results=validation_results,
        optimizer_results={
            CameraModelType(k): _optimizer_result_from_dict(v)
            for k, v in d.get("optimizer_results", {}).items()
        },
        cross_dataset_results=[
            _cross_dataset_result_from_dict(r)
            for r in d.get("cross_dataset_results", [])
        ],
        model_scores=[_model_score_from_dict(s) for s in d.get("model_scores", [])],
        outlier_result=_outlier_result_from_dict(d.get("outlier_result")),
        scene_quality_analysis=_scene_quality_analysis_from_dict(d.get("scene_quality_analysis")),
        subset_calibration_result=_subset_calibration_result_from_dict(d.get("subset_calibration_result")),
        final_result=_final_result_from_dict(d.get("final_result")),
        windshield_config=_windshield_config_from_dict(d.get("windshield_config")),
        windshield_dataset=_dataset_from_dict(d["windshield_dataset"]) if d.get("windshield_dataset") else None,
        windshield_results={
            windshield_result_key_from_storage(k): _windshield_calibration_result_from_dict(v)
            for k, v in d.get("windshield_results", {}).items()
        },
        reflection_results={
            k: _reflection_dataset_result_from_dict(v)
            for k, v in d.get("reflection_results", {}).items()
        },
        ghost_results={
            k: _ghost_dataset_result_from_dict(v)
            for k, v in d.get("ghost_results", {}).items()
        },
        ghost_models={
            k: _ghost_field_from_dict(v)
            for k, v in d.get("ghost_models", {}).items()
        },
        created_at=_dt(d.get("created_at")),
        updated_at=_dt(d.get("updated_at")),
        export_paths=export_paths,
    )


def load_project(path: str) -> tuple[CalibrationProject, list[str]]:
    """.ccproj 파일을 읽어 CalibrationProject로 복원.

    Returns:
        (project, missing_image_paths) - 프로젝트 저장 이후 원본 이미지
        파일이 옮겨지거나 지워졌으면 missing_image_paths에 담겨서 반환된다
        (예외로 죽지 않고, 호출부가 사용자에게 경고할 수 있게 함 - 설계
        문서 9번과 같은 "파일 없어도 메타데이터는 살아있다" 원칙).
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    project = project_from_dict(payload)

    missing = [
        f.image_info.path
        for f in project.dataset.frames
        if not Path(f.image_info.path).exists()
    ]
    return project, missing

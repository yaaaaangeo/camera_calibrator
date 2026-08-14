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
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from calibration.json_utils import json_safe
from calibration.types import (
    CalibrationProject,
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    CoverageCell,
    Dataset,
    DetectionResult,
    DiversityScores,
    ExportFormat,
    FinalResult,
    Frame,
    FrameQuality,
    FrameStatus,
    ImageInfo,
    ModelScore,
    OutlierResult,
    ParameterUncertainty,
    PatternConfig,
    PatternType,
    QualityGrade,
    RadialBin,
    RadialErrorProfile,
    RegionalError,
    ValidationResult,
)

PROJECT_FORMAT_VERSION = 1
PROJECT_EXTENSION = ".ccproj"


def project_to_dict(project: CalibrationProject) -> dict:
    raw = dataclasses.asdict(project)
    safe = json_safe(raw)
    return {"format_version": PROJECT_FORMAT_VERSION, "project": safe}


def save_project(project: CalibrationProject, path: str) -> str:
    project.updated_at = datetime.now()
    payload = project_to_dict(project)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _arr(d, dtype) -> np.ndarray | None:
    """_json_safe가 만든 {"__ndarray__": True, "data": [...]} 구조 -> np.ndarray.
    구버전 파일 호환을 위해 그냥 리스트로 저장된 경우도 받아준다.
    """
    if d is None:
        return None
    if isinstance(d, dict) and d.get("__ndarray__"):
        return np.array(d["data"], dtype=dtype)
    return np.array(d, dtype=dtype)


def _dt(d) -> datetime:
    if d is None:
        return datetime.now()
    if isinstance(d, dict) and d.get("__datetime__"):
        return datetime.fromisoformat(d["iso"])
    return datetime.fromisoformat(d)


def _pattern_config_from_dict(d: dict) -> PatternConfig:
    return PatternConfig(
        type=PatternType(d["type"]),
        squares_x=d["squares_x"],
        squares_y=d["squares_y"],
        square_size=d["square_size"],
        marker_size=d.get("marker_size"),
        dictionary=d.get("dictionary"),
    )


def _camera_config_from_dict(d: dict) -> CameraConfig:
    return CameraConfig(
        width=d["width"],
        height=d["height"],
        fps=d.get("fps"),
        model=CameraModelType(d["model"]) if d.get("model") else None,
        sensor_name=d.get("sensor_name"),
        hfov_deg=d.get("hfov_deg"),
        vfov_deg=d.get("vfov_deg"),
    )


def _image_info_from_dict(d: dict) -> ImageInfo:
    return ImageInfo(
        image_id=d["image_id"], path=d["path"], width=d["width"], height=d["height"],
        sharpness=d.get("sharpness"), brightness=d.get("brightness"), exposure=d.get("exposure"),
    )


def _detection_result_from_dict(d) -> DetectionResult | None:
    if d is None:
        return None
    return DetectionResult(
        image_id=d["image_id"], success=d["success"],
        corners=_arr(d.get("corners"), np.float32),
        object_points=_arr(d.get("object_points"), np.float32),
        ids=_arr(d.get("ids"), np.int32),
        num_corners=d.get("num_corners", 0),
        board_area_ratio=d.get("board_area_ratio"),
        board_center_px=tuple(d["board_center_px"]) if d.get("board_center_px") else None,
        board_tilt_deg=d.get("board_tilt_deg"),
        failure_reason=d.get("failure_reason"),
    )


def _frame_quality_from_dict(d) -> FrameQuality | None:
    if d is None:
        return None
    return FrameQuality(
        detection_score=d.get("detection_score", 0.0),
        geometric_score=d.get("geometric_score", 0.0),
        overall_score=d.get("overall_score", 0.0),
        grade=QualityGrade(d.get("grade", "poor")),
    )


def _frame_from_dict(d: dict) -> Frame:
    return Frame(
        image_info=_image_info_from_dict(d["image_info"]),
        detection=_detection_result_from_dict(d.get("detection")),
        quality=_frame_quality_from_dict(d.get("quality")),
        status=FrameStatus(d.get("status", "pending")),
        disabled_reason=d.get("disabled_reason"),
        reprojection_error=d.get("reprojection_error"),
    )


def _coverage_cell_from_dict(d: dict) -> CoverageCell:
    return CoverageCell(row=d["row"], col=d["col"], corner_count=d.get("corner_count", 0),
                         coverage_score=d.get("coverage_score", 0.0))


def _diversity_scores_from_dict(d) -> DiversityScores | None:
    if d is None:
        return None
    return DiversityScores(
        position_coverage=d.get("position_coverage", 0.0),
        distance_diversity=d.get("distance_diversity", 0.0),
        rotation_diversity=d.get("rotation_diversity", 0.0),
        edge_coverage=d.get("edge_coverage", 0.0),
    )


def _dataset_from_dict(d: dict) -> Dataset:
    return Dataset(
        frames=[_frame_from_dict(f) for f in d.get("frames", [])],
        coverage_grid=[_coverage_cell_from_dict(c) for c in d.get("coverage_grid", [])],
        diversity=_diversity_scores_from_dict(d.get("diversity")),
    )


def _param_uncertainty_from_dict(d) -> ParameterUncertainty | None:
    if d is None:
        return None
    return ParameterUncertainty(
        fx_std=d.get("fx_std"), fy_std=d.get("fy_std"),
        cx_std=d.get("cx_std"), cy_std=d.get("cy_std"),
    )


def _regional_error_from_dict(d) -> RegionalError | None:
    if d is None:
        return None
    return RegionalError(
        center=d.get("center"), left=d.get("left"), right=d.get("right"),
        top=d.get("top"), bottom=d.get("bottom"), corner=d.get("corner"),
    )


def _radial_profile_from_dict(d) -> RadialErrorProfile | None:
    if d is None:
        return None
    bins = [
        RadialBin(
            radius_min=b["radius_min"], radius_max=b["radius_max"],
            mean_error=b.get("mean_error"), num_points=b.get("num_points", 0),
        )
        for b in d.get("bins", [])
    ]
    return RadialErrorProfile(bins=bins, max_radius=d.get("max_radius", 0.0))


def _calibration_result_from_dict(d: dict) -> CalibrationResult:
    return CalibrationResult(
        model_name=CameraModelType(d["model_name"]),
        camera_matrix=_arr(d.get("camera_matrix"), np.float64),
        distortion=_arr(d.get("distortion"), np.float64),
        rvecs=[_arr(r, np.float64) for r in d.get("rvecs", [])],
        tvecs=[_arr(t, np.float64) for t in d.get("tvecs", [])],
        rms_error=d.get("rms_error"),
        per_frame_error=d.get("per_frame_error", {}),
        regional_error=_regional_error_from_dict(d.get("regional_error")),
        radial_profile=_radial_profile_from_dict(d.get("radial_profile")),
        param_uncertainty=_param_uncertainty_from_dict(d.get("param_uncertainty")),
        success=d.get("success", False),
        error_message=d.get("error_message"),
    )


def _outlier_result_from_dict(d) -> OutlierResult | None:
    if d is None:
        return None
    return OutlierResult(
        threshold_used=d["threshold_used"],
        removed_frame_ids=d.get("removed_frame_ids", []),
        rms_before=d.get("rms_before"), rms_after=d.get("rms_after"),
        iterations=d.get("iterations", 0), max_iterations=d.get("max_iterations", 3),
    )


def _validation_result_from_dict(d: dict) -> ValidationResult:
    return ValidationResult(
        train_frame_ids=d.get("train_frame_ids", []),
        test_frame_ids=d.get("test_frame_ids", []),
        train_rms=d.get("train_rms"), test_rms=d.get("test_rms"), edge_rms=d.get("edge_rms"),
        straightness_residual=d.get("straightness_residual"),
        success=d.get("success", True), error_message=d.get("error_message"),
        failed_test_frame_ids=d.get("failed_test_frame_ids", []),
    )


def _model_score_from_dict(d: dict) -> ModelScore:
    return ModelScore(
        model_name=CameraModelType(d["model_name"]), score=d["score"],
        components=d.get("components", {}), is_recommended=d.get("is_recommended", False),
    )


def _final_result_from_dict(d) -> FinalResult | None:
    if d is None:
        return None
    return FinalResult(
        chosen_model=CameraModelType(d["chosen_model"]),
        calibration=_calibration_result_from_dict(d["calibration"]),
        validation=_validation_result_from_dict(d["validation"]) if d.get("validation") else None,
        outlier=_outlier_result_from_dict(d.get("outlier")),
        dataset_coverage_pct=d.get("dataset_coverage_pct"),
        overall_grade=QualityGrade(d.get("overall_grade", "warning")),
        model_scores=[_model_score_from_dict(s) for s in d.get("model_scores", [])],
    )


def project_from_dict(payload: dict) -> CalibrationProject:
    version = payload.get("format_version")
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
        validation_results=validation_results,
        model_scores=[_model_score_from_dict(s) for s in d.get("model_scores", [])],
        outlier_result=_outlier_result_from_dict(d.get("outlier_result")),
        final_result=_final_result_from_dict(d.get("final_result")),
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

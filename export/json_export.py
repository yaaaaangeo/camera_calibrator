"""
camera_calibrator.export.json_export
=========================================

설계 문서 11번 - "JSON도 지원하면 실무성이 크게 올라간다."

OpenCV YAML(export/opencv.py), ROS YAML(export/ros.py)은 각각 특정 도구용
스키마를 따르는 반면, 이 JSON은 범용이다 - 다른 파이썬/JS/무엇이든 스크립트가
`json.load()` 한 줄로 카메라 행렬, 왜곡 계수, 각종 오차 지표, 최종 등급까지
전부 프로그래밍적으로 읽어갈 수 있게 하는 게 목적이다.

HTML 리포트(export/report.py)와 겹치는 내용이 많지만 HTML은 "사람이 읽는
서사문"이고 이 JSON은 "기계가 읽는 평평한 구조"라 목적이 다르다 - 그래서
report.py의 문자열 포맷팅 함수들을 재사용하지 않고 독립적으로 구현한다.

numpy 배열은 calibration/json_utils.json_safe(ndarray_wrapper=False)로
평평한 중첩 리스트로 편다 - 이 프로젝트를 모르는 외부 도구가 읽을 파일이라
project_io.py(.ccproj)가 쓰는 __ndarray__ 래퍼는 여기선 안 쓴다.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from calibration.json_utils import json_safe
from calibration.types import (
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    Dataset,
    FinalResult,
    ModelScore,
    PatternConfig,
    ValidationResult,
)
from calibration.quality import coverage_percentage

_MODEL_ORDER = [CameraModelType.PINHOLE, CameraModelType.EXTENDED_PINHOLE, CameraModelType.FISHEYE]


def build_export_dict(
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    dataset: Dataset,
    calibration_results: dict[CameraModelType, CalibrationResult],
    validation_results: dict[CameraModelType, ValidationResult],
    chosen_model: CameraModelType,
    final_result: FinalResult | None = None,
    model_scores: list[ModelScore] | None = None,
) -> dict:
    """JSON으로 직렬화할 dict를 만든다. export_json()이 이걸 그대로 파일에 쓴다 -
    테스트에서 파일 I/O 없이 구조만 검증하고 싶을 때도 이 함수를 바로 쓸 수 있다.
    """
    chosen_cal = calibration_results.get(chosen_model)
    chosen_val = validation_results.get(chosen_model)

    payload = {
        "export_format_version": 1,
        "generated_at": datetime.now().isoformat(),
        "sensor_name": camera_config.sensor_name,
        "camera": {
            "width": camera_config.width,
            "height": camera_config.height,
            "fps": camera_config.fps,
        },
        "pattern": {
            "type": pattern_config.type.value,
            "squares_x": pattern_config.squares_x,
            "squares_y": pattern_config.squares_y,
            "square_size_m": pattern_config.square_size,
            "marker_size_m": pattern_config.marker_size,
            "dictionary": pattern_config.dictionary,
        },
        "dataset": {
            "num_total": dataset.num_total,
            "num_detected": dataset.num_detected,
            "num_used": dataset.num_enabled,
            "coverage_pct": coverage_percentage(dataset.coverage_grid) if dataset.coverage_grid else None,
        },
        "chosen_model": chosen_model.value,
        "models": {},
    }

    for m in _MODEL_ORDER:
        cal = calibration_results.get(m)
        val = validation_results.get(m)
        if cal is None:
            continue
        entry = {
            "success": cal.success,
            "error_message": cal.error_message,
        }
        if cal.success:
            entry.update({
                "camera_matrix": cal.camera_matrix,
                "distortion_coefficients": cal.distortion,
                "distortion_coefficient_count": int(cal.distortion.size) if cal.distortion is not None else None,
                "rms_reprojection_error_px": cal.rms_error,
                "per_frame_error_px": cal.per_frame_error,
                "regional_error_px": cal.regional_error,
                "radial_error_profile": cal.radial_profile,
                "parameter_uncertainty": cal.param_uncertainty,
            })
        if val is not None:
            entry["validation"] = {
                "success": val.success,
                "train_rms_px": val.train_rms,
                "test_rms_px": val.test_rms,
                "edge_rms_px": val.edge_rms,
                "line_straightness_residual_px": val.straightness_residual,
                "num_train_frames": len(val.train_frame_ids),
                "num_test_frames": len(val.test_frame_ids),
            }
        payload["models"][m.value] = entry

    if model_scores:
        payload["model_scores"] = [
            {"model": s.model_name.value, "score": s.score, "is_recommended": s.is_recommended,
             "components": s.components}
            for s in model_scores
        ]

    if final_result is not None:
        payload["final_result"] = {
            "chosen_model": final_result.chosen_model.value,
            "overall_grade": final_result.overall_grade.value,
            "dataset_coverage_pct": final_result.dataset_coverage_pct,
        }
        if final_result.outlier:
            payload["final_result"]["outlier"] = {
                "removed_frame_ids": final_result.outlier.removed_frame_ids,
                "threshold_used_px": final_result.outlier.threshold_used,
                "rms_before_px": final_result.outlier.rms_before,
                "rms_after_px": final_result.outlier.rms_after,
                "iterations": final_result.outlier.iterations,
            }
    elif chosen_val is not None:
        payload["chosen_model_summary"] = {
            "rms_error_px": chosen_cal.rms_error if chosen_cal else None,
            "test_rms_px": chosen_val.test_rms,
        }

    return payload


def export_json(
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    dataset: Dataset,
    calibration_results: dict[CameraModelType, CalibrationResult],
    validation_results: dict[CameraModelType, ValidationResult],
    chosen_model: CameraModelType,
    path: str,
    final_result: FinalResult | None = None,
    model_scores: list[ModelScore] | None = None,
) -> str:
    payload = build_export_dict(
        camera_config, pattern_config, dataset, calibration_results, validation_results,
        chosen_model, final_result, model_scores,
    )
    safe_payload = json_safe(payload, ndarray_wrapper=False)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe_payload, f, ensure_ascii=False, indent=2)
    return path

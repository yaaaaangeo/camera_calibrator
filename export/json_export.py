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
    CrossDatasetValidationResult,
    Dataset,
    FinalResult,
    KFoldResult,
    ModelScore,
    PatternConfig,
    RepeatedKFoldResult,
    ValidationResult,
)
from calibration.quality import coverage_percentage

_MODEL_ORDER = [CameraModelType.PINHOLE, CameraModelType.EXTENDED_PINHOLE, CameraModelType.FISHEYE]


def _avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _dataset_statistics(dataset: Dataset) -> dict:
    failures: dict[str, int] = {}
    corners: list[float] = []
    board_areas: list[float] = []
    tilts: list[float] = []
    frame_quality: list[float] = []
    component_quality = {
        "blur_score": [],
        "exposure_score": [],
        "corner_quality_score": [],
        "board_area_score": [],
        "edge_coverage_score": [],
        "pose_diversity_score": [],
    }
    for frame in dataset.frames:
        det = frame.detection
        if det and det.success:
            corners.append(float(det.num_corners))
            if det.board_area_ratio is not None:
                board_areas.append(float(det.board_area_ratio))
            if det.board_tilt_deg is not None:
                tilts.append(float(det.board_tilt_deg))
        elif det and not det.success:
            reason = det.failure_reason or "unknown"
            failures[reason] = failures.get(reason, 0) + 1
        if frame.quality:
            frame_quality.append(float(frame.quality.overall_score))
            for key in component_quality:
                value = getattr(frame.quality, key)
                if value is not None:
                    component_quality[key].append(float(value))

    return {
        "num_total": dataset.num_total,
        "num_detected": dataset.num_detected,
        "num_used": dataset.num_enabled,
        "num_failed": dataset.num_total - dataset.num_detected,
        "detection_success_rate_pct": dataset.num_detected / dataset.num_total * 100.0 if dataset.num_total else 0.0,
        "coverage_pct": coverage_percentage(dataset.coverage_grid) if dataset.coverage_grid else None,
        "avg_detected_corners": _avg(corners),
        "avg_board_area_ratio": _avg(board_areas),
        "avg_board_tilt_deg": _avg(tilts),
        "failure_reasons": failures,
        "avg_frame_quality": _avg(frame_quality),
        "frame_quality_components": {k: _avg(v) for k, v in component_quality.items()},
        "coverage_grid": dataset.coverage_grid,
        "diversity": dataset.diversity,
        "quality_score": dataset.quality_score,
    }


def _cross_validation_summary(
    validation_results: dict[CameraModelType, ValidationResult],
    kfold_result: KFoldResult | None,
    repeated_kfold_result: RepeatedKFoldResult | None,
) -> dict:
    holdout = {}
    for model, val in validation_results.items():
        holdout[model.value] = {
            "success": val.success,
            "train_frame_ids": val.train_frame_ids,
            "test_frame_ids": val.test_frame_ids,
            "train_rms_px": val.train_rms,
            "test_rms_px": val.test_rms,
            "train_residual_stats": val.train_residual_stats,
            "test_residual_stats": val.test_residual_stats,
            "edge_rms_px": val.edge_rms,
            "straightness_residual_px": val.straightness_residual,
            "straightness_breakdown": val.straightness_breakdown,
            "failed_test_frame_ids": val.failed_test_frame_ids,
            "error_message": val.error_message,
        }
    return {
        "holdout": holdout,
        "kfold": kfold_result,
        "repeated_kfold": repeated_kfold_result,
    }


def _bootstrap_stability_summary(calibration_results: dict[CameraModelType, CalibrationResult]) -> dict:
    summary = {}
    for model, cal in calibration_results.items():
        pu = cal.param_uncertainty_bootstrap or cal.param_uncertainty
        summary[model.value] = {
            "available": pu is not None,
            "method": pu.method if pu else None,
            "n_bootstrap_success": pu.n_bootstrap_success if pu else None,
            "fx_std": pu.fx_std if pu else None,
            "fy_std": pu.fy_std if pu else None,
            "cx_std": pu.cx_std if pu else None,
            "cy_std": pu.cy_std if pu else None,
            "overall_stability": pu.overall_stability if pu else None,
            "distortion_stats": pu.distortion_stats if pu else [],
        }
    return summary


def _final_calibration_summary(final_result: FinalResult | None) -> dict | None:
    if final_result is None:
        return None
    cal = final_result.calibration
    val = final_result.validation
    return {
        "chosen_model": final_result.chosen_model.value,
        "overall_grade": final_result.overall_grade.value,
        "confidence": final_result.confidence,
        "train_rms_px": cal.rms_error if cal else None,
        "test_rms_px": val.test_rms if val else None,
        "test_p95_px": val.test_residual_stats.p95 if val and val.test_residual_stats else None,
        "edge_rms_px": val.edge_rms if val else None,
        "straightness_residual_px": val.straightness_residual if val else None,
        "dataset_coverage_pct": final_result.dataset_coverage_pct,
        "observability": cal.observability if cal else None,
        "undistortion_quality": cal.undistortion_quality if cal else None,
        "diagnosis": final_result.diagnosis,
    }


def build_export_dict(
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    dataset: Dataset,
    calibration_results: dict[CameraModelType, CalibrationResult],
    validation_results: dict[CameraModelType, ValidationResult],
    chosen_model: CameraModelType,
    final_result: FinalResult | None = None,
    model_scores: list[ModelScore] | None = None,
    cross_dataset_results: list[CrossDatasetValidationResult] | None = None,
    kfold_result: KFoldResult | None = None,
    repeated_kfold_result: RepeatedKFoldResult | None = None,
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
        "dataset": _dataset_statistics(dataset),
        "chosen_model": chosen_model.value,
        "models": {},
        "cross_validation": _cross_validation_summary(validation_results, kfold_result, repeated_kfold_result),
        "bootstrap_stability": _bootstrap_stability_summary(calibration_results),
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
                "parameter_uncertainty_bootstrap": cal.param_uncertainty_bootstrap,
                "residual_stats": cal.residual_stats,
                "observability": cal.observability,
                "undistortion_quality": cal.undistortion_quality,
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
             "components": s.components, "parameter_count": s.parameter_count,
             "residual_sum_squares": s.residual_sum_squares,
             "num_observations": s.num_observations, "aic": s.aic, "bic": s.bic,
             "selection_confidence": s.selection_confidence,
             "selection_confidence_level": s.selection_confidence_level,
             "selection_confidence_reason": s.selection_confidence_reason,
             "selection_reasons": s.selection_reasons}
            for s in model_scores
        ]

    if cross_dataset_results:
        payload["cross_dataset_validation"] = cross_dataset_results

    if final_result is not None:
        payload["final_result"] = {
            "chosen_model": final_result.chosen_model.value,
            "overall_grade": final_result.overall_grade.value,
            "confidence": final_result.confidence,
            "dataset_coverage_pct": final_result.dataset_coverage_pct,
            "diagnosis": final_result.diagnosis,
        }
        payload["final_calibration_summary"] = _final_calibration_summary(final_result)
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
    cross_dataset_results: list[CrossDatasetValidationResult] | None = None,
    kfold_result: KFoldResult | None = None,
    repeated_kfold_result: RepeatedKFoldResult | None = None,
) -> str:
    payload = build_export_dict(
        camera_config, pattern_config, dataset, calibration_results, validation_results,
        chosen_model, final_result, model_scores, cross_dataset_results,
        kfold_result, repeated_kfold_result,
    )
    safe_payload = json_safe(payload, ndarray_wrapper=False)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe_payload, f, ensure_ascii=False, indent=2)
    return path

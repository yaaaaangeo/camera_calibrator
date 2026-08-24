"""
Stereo calibration export helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import html
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from calibration.calibration_io import StandardCalibration
from calibration.stereo import StereoCalibrationResult, StereoPairObservation, StereoValidationStats
from calibration.types import CameraModelType


@dataclass(frozen=True)
class StereoRoboticsExportOptions:
    parent_frame: str = "camera1"
    child_frame: str = "camera2"
    rotation_format: str = "quaternion"


def _list(arr: np.ndarray | None) -> list:
    if arr is None:
        return []
    return np.asarray(arr, dtype=float).tolist()


def _stats(stats: StereoValidationStats) -> dict[str, float | None]:
    return {
        "mean": stats.mean,
        "median": stats.median,
        "rmse": stats.rmse,
        "p95": stats.p95,
        "p99": stats.p99,
        "max": stats.max,
    }


def _stats_from_dict(data: dict[str, Any] | None) -> StereoValidationStats:
    if not data:
        return StereoValidationStats()
    return StereoValidationStats(
        mean=data.get("mean"),
        median=data.get("median"),
        rmse=data.get("rmse"),
        p95=data.get("p95"),
        p99=data.get("p99"),
        max=data.get("max"),
    )


def _array(data: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.asarray(data if data is not None else [], dtype=np.float64)
    if shape is not None and arr.size:
        return arr.reshape(shape)
    return arr


def stereo_pair_to_dict(pair: StereoPairObservation) -> dict[str, Any]:
    return {
        "pair_id": pair.pair_id,
        "object_points": _list(pair.object_points),
        "image_points_cam1": _list(pair.image_points_cam1),
        "image_points_cam2": _list(pair.image_points_cam2),
        "common_ids": np.asarray(pair.common_ids, dtype=int).reshape(-1).tolist(),
        "detected_points_cam1": _list(pair.detected_points_cam1),
        "detected_points_cam2": _list(pair.detected_points_cam2),
        "detected_ids_cam1": np.asarray(pair.detected_ids_cam1, dtype=int).reshape(-1).tolist()
        if pair.detected_ids_cam1 is not None else [],
        "detected_ids_cam2": np.asarray(pair.detected_ids_cam2, dtype=int).reshape(-1).tolist()
        if pair.detected_ids_cam2 is not None else [],
        "timestamp_cam1": pair.timestamp_cam1,
        "timestamp_cam2": pair.timestamp_cam2,
        "image_path_cam1": pair.image_path_cam1,
        "image_path_cam2": pair.image_path_cam2,
        "sync_delta_ms": pair.sync_delta_ms,
        "used": pair.used,
        "rejected_reason": pair.rejected_reason,
        "quality_score": pair.quality_score,
        "quality_status": pair.quality_status,
        "quality_components": dict(pair.quality_components),
        "quality_warnings": list(pair.quality_warnings),
    }


def stereo_pair_from_dict(payload: dict[str, Any]) -> StereoPairObservation:
    return StereoPairObservation(
        pair_id=payload.get("pair_id", "Pair"),
        object_points=_array(payload.get("object_points")).astype(np.float32).reshape(-1, 1, 3),
        image_points_cam1=_array(payload.get("image_points_cam1")).astype(np.float32).reshape(-1, 1, 2),
        image_points_cam2=_array(payload.get("image_points_cam2")).astype(np.float32).reshape(-1, 1, 2),
        common_ids=np.asarray(payload.get("common_ids", []), dtype=np.int32),
        detected_points_cam1=_array(payload.get("detected_points_cam1")).astype(np.float32).reshape(-1, 1, 2)
        if payload.get("detected_points_cam1") else None,
        detected_points_cam2=_array(payload.get("detected_points_cam2")).astype(np.float32).reshape(-1, 1, 2)
        if payload.get("detected_points_cam2") else None,
        detected_ids_cam1=np.asarray(payload.get("detected_ids_cam1", []), dtype=np.int32),
        detected_ids_cam2=np.asarray(payload.get("detected_ids_cam2", []), dtype=np.int32),
        timestamp_cam1=payload.get("timestamp_cam1"),
        timestamp_cam2=payload.get("timestamp_cam2"),
        image_path_cam1=payload.get("image_path_cam1"),
        image_path_cam2=payload.get("image_path_cam2"),
        sync_delta_ms=payload.get("sync_delta_ms"),
        used=bool(payload.get("used", True)),
        rejected_reason=payload.get("rejected_reason"),
        quality_score=float(payload.get("quality_score", 0.0)),
        quality_status=payload.get("quality_status", "Unknown"),
        quality_components=dict(payload.get("quality_components", {})),
        quality_warnings=list(payload.get("quality_warnings", [])),
    )


def stereo_pairs_to_dict(pairs: list[StereoPairObservation]) -> list[dict[str, Any]]:
    return [stereo_pair_to_dict(pair) for pair in pairs]


def stereo_pairs_from_dict(payload: list[dict[str, Any]] | None) -> list[StereoPairObservation]:
    return [stereo_pair_from_dict(pair) for pair in (payload or [])]


def _quaternion_xyzw_from_rotation(R: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (matrix[2, 1] - matrix[1, 2]) / s
        qy = (matrix[0, 2] - matrix[2, 0]) / s
        qz = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        qw = (matrix[2, 1] - matrix[1, 2]) / s
        qx = 0.25 * s
        qy = (matrix[0, 1] + matrix[1, 0]) / s
        qz = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        qw = (matrix[0, 2] - matrix[2, 0]) / s
        qx = (matrix[0, 1] + matrix[1, 0]) / s
        qy = 0.25 * s
        qz = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        qw = (matrix[1, 0] - matrix[0, 1]) / s
        qx = (matrix[0, 2] + matrix[2, 0]) / s
        qy = (matrix[1, 2] + matrix[2, 1]) / s
        qz = 0.25 * s
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm > 0:
        quat /= norm
    return tuple(float(v) for v in quat)


def _robotics_export_payload(
    result: StereoCalibrationResult,
    options: StereoRoboticsExportOptions,
) -> dict[str, Any]:
    roll, pitch, yaw = result.roll_pitch_yaw_deg
    t = result.t_cam2_from_cam1.reshape(3)
    roll_rad, pitch_rad, yaw_rad = (math.radians(v) for v in (roll, pitch, yaw))
    qx, qy, qz, qw = _quaternion_xyzw_from_rotation(result.R_cam2_from_cam1)
    parent = options.parent_frame.strip() or "camera1"
    child = options.child_frame.strip() or "camera2"
    ros2_quaternion_cmd = (
        "ros2 run tf2_ros static_transform_publisher "
        f"{t[0]:.9g} {t[1]:.9g} {t[2]:.9g} "
        f"{qx:.9g} {qy:.9g} {qz:.9g} {qw:.9g} {parent} {child}"
    )
    ros2_rpy_radians_cmd = (
        "ros2 run tf2_ros static_transform_publisher "
        f"{t[0]:.9g} {t[1]:.9g} {t[2]:.9g} "
        f"{yaw_rad:.9g} {pitch_rad:.9g} {roll_rad:.9g} {parent} {child}"
    )
    ros2_rpy_degrees_cmd = (
        "ros2 run tf2_ros static_transform_publisher "
        f"{t[0]:.9g} {t[1]:.9g} {t[2]:.9g} "
        f"{yaw:.9g} {pitch:.9g} {roll:.9g} {parent} {child}"
    )
    ros1_rpy_radians_cmd = (
        "rosrun tf2_ros static_transform_publisher "
        f"{t[0]:.9g} {t[1]:.9g} {t[2]:.9g} "
        f"{yaw_rad:.9g} {pitch_rad:.9g} {roll_rad:.9g} {parent} {child} 100"
    )
    if options.rotation_format == "rpy_degrees":
        selected = ros2_rpy_degrees_cmd
    elif options.rotation_format == "rpy_radians":
        selected = ros2_rpy_radians_cmd
    else:
        selected = ros2_quaternion_cmd
    return {
        "parent_frame": parent,
        "child_frame": child,
        "rotation_format": options.rotation_format,
        "translation_xyz": _list(t),
        "quaternion_xyzw": [qx, qy, qz, qw],
        "roll_pitch_yaw_deg": {"roll": roll, "pitch": pitch, "yaw": yaw},
        "roll_pitch_yaw_rad": {"roll": roll_rad, "pitch": pitch_rad, "yaw": yaw_rad},
        "selected_static_transform_publisher": selected,
        "static_transform_publisher_ros2_quaternion": ros2_quaternion_cmd,
        "static_transform_publisher_ros2_rpy_radians": ros2_rpy_radians_cmd,
        "static_transform_publisher_ros2_rpy_degrees": ros2_rpy_degrees_cmd,
        "static_transform_publisher_ros1_rpy_radians": ros1_rpy_radians_cmd,
        "note": "Frame names and rotation representation are export options; verify the exact argument order for your ROS distribution.",
    }


def stereo_result_to_dict(
    result: StereoCalibrationResult,
    robotics_options: StereoRoboticsExportOptions | None = None,
) -> dict[str, Any]:
    robotics_options = robotics_options or StereoRoboticsExportOptions()
    roll, pitch, yaw = result.roll_pitch_yaw_deg
    return {
        "format_version": 1,
        "convention": "P_cam2 = R_cam2_from_cam1 @ P_cam1 + t_cam2_from_cam1",
        "camera1": {
            "camera_matrix": _list(result.camera1.camera_matrix),
            "distortion_coefficients": _list(result.camera1.distortion.reshape(-1)),
            "resolution": [result.camera1.width, result.camera1.height],
            "model": result.camera1.model_name.value if result.camera1.model_name else None,
        },
        "camera2": {
            "camera_matrix": _list(result.camera2.camera_matrix),
            "distortion_coefficients": _list(result.camera2.distortion.reshape(-1)),
            "resolution": [result.camera2.width, result.camera2.height],
            "model": result.camera2.model_name.value if result.camera2.model_name else None,
        },
        "stereo": {
            "stereo_rms": result.stereo_rms,
            "R_cam2_from_cam1": _list(result.R_cam2_from_cam1),
            "t_cam2_from_cam1": _list(result.t_cam2_from_cam1.reshape(3)),
            "T_cam2_from_cam1": _list(result.T_cam2_from_cam1),
            "R_cam1_from_cam2": _list(result.R_cam1_from_cam2),
            "t_cam1_from_cam2": _list(result.t_cam1_from_cam2.reshape(3)),
            "T_cam1_from_cam2": _list(result.T_cam1_from_cam2),
            "baseline": result.baseline,
            "roll_pitch_yaw_deg": {
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
            },
            "essential_matrix": _list(result.E),
            "fundamental_matrix": _list(result.F),
        },
        "rectification": {
            "R1": _list(result.R1),
            "R2": _list(result.R2),
            "P1": _list(result.P1),
            "P2": _list(result.P2),
            "Q": _list(result.Q),
        },
        "validation": {
            "epipolar_error": _stats(result.epipolar_error),
            "sampson_distance": _stats(result.sampson_distance),
            "rectification_vertical_error": _stats(result.rectification_vertical_error),
            "holdout": {
                "train_pairs": result.holdout_train_pair_count,
                "validation_pairs": result.holdout_validation_pair_count,
                "training_epipolar_error": (
                    _stats(result.holdout_training_error)
                    if result.holdout_training_error is not None else None
                ),
                "validation_epipolar_error": (
                    _stats(result.holdout_validation_error)
                    if result.holdout_validation_error is not None else None
                ),
                "generalization_gap": result.holdout_generalization_gap,
            },
            "pairs": [
                {
                    "pair_id": row.pair_id,
                    "common_corners": row.common_corners,
                    "epipolar_mean": row.epipolar_mean,
                    "sampson_mean": row.sampson_mean,
                    "vertical_mean": row.vertical_mean,
                    "status": row.status,
                }
                for row in result.pair_validations
            ],
        },
        "dataset": {
            "used_pairs": result.used_pair_count,
            "rejected_pairs": result.rejected_pair_count,
            "total_common_corners": result.total_common_corners,
        },
        "capture_coach": dict(result.capture_coach),
        "sync_guard": dict(result.sync_guard),
        "calibration_audit": dict(result.calibration_audit),
        "evidence_report": dict(result.evidence_report),
        "robotics": _robotics_export_payload(result, robotics_options),
    }


def stereo_result_from_dict(payload: dict[str, Any]) -> StereoCalibrationResult:
    from calibration.stereo import StereoPairValidation, inverse_rt, baseline_from_t, euler_zyx_from_rotation

    cam1 = payload["camera1"]
    cam2 = payload["camera2"]
    stereo = payload["stereo"]
    rect = payload.get("rectification", {})
    val = payload.get("validation", {})

    def make_camera(name: str, data: dict[str, Any]) -> StandardCalibration:
        width, height = data.get("resolution", [None, None])
        model = CameraModelType(data["model"]) if data.get("model") else None
        return StandardCalibration(
            name,
            _array(data.get("camera_matrix"), (3, 3)),
            _array(data.get("distortion_coefficients")).reshape(-1, 1),
            model,
            width=width,
            height=height,
        )

    R = _array(stereo.get("R_cam2_from_cam1"), (3, 3))
    t = _array(stereo.get("t_cam2_from_cam1"), (3, 1))
    if stereo.get("R_cam1_from_cam2") is not None and stereo.get("t_cam1_from_cam2") is not None:
        R_inv = _array(stereo.get("R_cam1_from_cam2"), (3, 3))
        t_inv = _array(stereo.get("t_cam1_from_cam2"), (3, 1))
        T_inv = _array(stereo.get("T_cam1_from_cam2"), (4, 4))
    else:
        R_inv, t_inv, T_inv = inverse_rt(R, t)

    rpy = stereo.get("roll_pitch_yaw_deg", {})
    if isinstance(rpy, dict):
        roll_pitch_yaw = (float(rpy.get("roll", 0.0)), float(rpy.get("pitch", 0.0)), float(rpy.get("yaw", 0.0)))
    else:
        roll_pitch_yaw = tuple(rpy) if rpy else euler_zyx_from_rotation(R)

    holdout = val.get("holdout", {})
    return StereoCalibrationResult(
        camera1=make_camera("camera1", cam1),
        camera2=make_camera("camera2", cam2),
        image_size=tuple(cam1.get("resolution", [0, 0])),
        stereo_rms=float(stereo.get("stereo_rms", 0.0)),
        R_cam2_from_cam1=R,
        t_cam2_from_cam1=t,
        E=_array(stereo.get("essential_matrix"), (3, 3)),
        F=_array(stereo.get("fundamental_matrix"), (3, 3)),
        T_cam2_from_cam1=_array(stereo.get("T_cam2_from_cam1"), (4, 4)),
        R_cam1_from_cam2=R_inv,
        t_cam1_from_cam2=t_inv,
        T_cam1_from_cam2=T_inv,
        baseline=float(stereo.get("baseline", baseline_from_t(t))),
        roll_pitch_yaw_deg=roll_pitch_yaw,
        R1=_array(rect.get("R1"), (3, 3)) if rect.get("R1") else None,
        R2=_array(rect.get("R2"), (3, 3)) if rect.get("R2") else None,
        P1=_array(rect.get("P1"), (3, 4)) if rect.get("P1") else None,
        P2=_array(rect.get("P2"), (3, 4)) if rect.get("P2") else None,
        Q=_array(rect.get("Q"), (4, 4)) if rect.get("Q") else None,
        epipolar_error=_stats_from_dict(val.get("epipolar_error")),
        sampson_distance=_stats_from_dict(val.get("sampson_distance")),
        rectification_vertical_error=_stats_from_dict(val.get("rectification_vertical_error")),
        pair_validations=[
            StereoPairValidation(
                pair_id=row.get("pair_id", ""),
                common_corners=int(row.get("common_corners", 0)),
                epipolar_mean=row.get("epipolar_mean"),
                sampson_mean=row.get("sampson_mean"),
                vertical_mean=row.get("vertical_mean"),
                status=row.get("status", "Good"),
            )
            for row in val.get("pairs", [])
        ],
        holdout_training_error=_stats_from_dict(holdout.get("training_epipolar_error"))
        if holdout.get("training_epipolar_error") else None,
        holdout_validation_error=_stats_from_dict(holdout.get("validation_epipolar_error"))
        if holdout.get("validation_epipolar_error") else None,
        holdout_generalization_gap=holdout.get("generalization_gap"),
        holdout_train_pair_count=int(holdout.get("train_pairs", 0)),
        holdout_validation_pair_count=int(holdout.get("validation_pairs", 0)),
        used_pair_count=int(payload.get("dataset", {}).get("used_pairs", 0)),
        rejected_pair_count=int(payload.get("dataset", {}).get("rejected_pairs", 0)),
        total_common_corners=int(payload.get("dataset", {}).get("total_common_corners", 0)),
        capture_coach=dict(payload.get("capture_coach", {})),
        sync_guard=dict(payload.get("sync_guard", {})),
        calibration_audit=dict(payload.get("calibration_audit", {})),
        evidence_report=dict(payload.get("evidence_report", {})),
    )


def export_stereo_json(
    result: StereoCalibrationResult,
    path: str | Path,
    robotics_options: StereoRoboticsExportOptions | None = None,
) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(stereo_result_to_dict(result, robotics_options), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(out)


def export_stereo_yaml(
    result: StereoCalibrationResult,
    path: str | Path,
    robotics_options: StereoRoboticsExportOptions | None = None,
) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(stereo_result_to_dict(result, robotics_options), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return str(out)


def _kalibr_distortion_model(model: CameraModelType | None) -> str:
    if model == CameraModelType.FISHEYE:
        return "equidistant"
    return "radtan"


def export_stereo_kalibr_camchain(result: StereoCalibrationResult, path: str | Path) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def camera_payload(cal: StandardCalibration, *, include_transform: bool) -> dict[str, Any]:
        data: dict[str, Any] = {
            "camera_model": "pinhole",
            "intrinsics": [float(cal.fx), float(cal.fy), float(cal.cx), float(cal.cy)],
            "distortion_model": _kalibr_distortion_model(cal.model_name),
            "distortion_coeffs": np.asarray(cal.distortion, dtype=float).reshape(-1).tolist(),
            "resolution": [int(cal.width or result.image_size[0]), int(cal.height or result.image_size[1])],
            "rostopic": f"/{cal.camera_name or cal.label}/image_raw",
        }
        if include_transform:
            data["T_cn_cnm1"] = _list(result.T_cam2_from_cam1)
        return data

    payload = {
        "cam0": camera_payload(result.camera1, include_transform=False),
        "cam1": camera_payload(result.camera2, include_transform=True),
    }
    out.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return str(out)


def export_stereo_html(
    result: StereoCalibrationResult,
    path: str | Path,
    robotics_options: StereoRoboticsExportOptions | None = None,
) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = stereo_result_to_dict(result, robotics_options)
    stereo = payload["stereo"]
    validation = payload["validation"]
    dataset = payload["dataset"]
    capture = payload.get("capture_coach", {})
    sync = payload.get("sync_guard", {})
    audit = payload.get("calibration_audit", {})
    evidence = payload.get("evidence_report", {})

    def fmt(value: Any, digits: int = 4) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return html.escape(str(value))

    pair_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['pair_id'])}</td>"
        f"<td>{row['common_corners']}</td>"
        f"<td>{fmt(row['epipolar_mean'])}</td>"
        f"<td>{fmt(row['vertical_mean'])}</td>"
        f"<td>{html.escape(row['status'])}</td>"
        "</tr>"
        for row in validation["pairs"]
    ) or "<tr><td colspan='5'>No per-pair validation rows.</td></tr>"

    recommendations = "\n".join(
        f"<li>{html.escape(str(item))}</li>"
        for item in capture.get("recommendations", [])
    ) or "<li>No immediate capture recommendations.</li>"
    warnings = "\n".join(
        f"<li>{html.escape(str(item))}</li>"
        for item in evidence.get("warnings", [])
    ) or "<li>No evidence warnings.</li>"
    pose = audit.get("cross_camera_pose_consistency", {})
    recon = audit.get("reconstruction", {})

    matrix_block = html.escape(json.dumps(payload, ensure_ascii=False, indent=2))
    body = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>Stereo Calibration Report</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ margin: 0 0 12px; }}
    h1 {{ font-size: 26px; }}
    h2 {{ font-size: 18px; margin-top: 28px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #d7dde5; border-radius: 8px; padding: 12px; }}
    .label {{ color: #627183; font-size: 12px; text-transform: uppercase; }}
    .value {{ font-size: 22px; font-weight: 650; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th, td {{ border-bottom: 1px solid #e4e9ef; padding: 8px; text-align: left; }}
    th {{ background: #f5f7fa; }}
    pre {{ background: #111827; color: #eef2ff; padding: 16px; overflow: auto; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>Stereo Calibration Report</h1>
  <p>{html.escape(payload["convention"])}</p>
  <div class="grid">
    <div class="metric"><div class="label">Stereo RMS</div><div class="value">{fmt(stereo["stereo_rms"])} px</div></div>
    <div class="metric"><div class="label">Baseline</div><div class="value">{fmt(stereo["baseline"] * 1000.0, 2)} mm</div></div>
    <div class="metric"><div class="label">Epipolar RMSE</div><div class="value">{fmt(validation["epipolar_error"]["rmse"])} px</div></div>
    <div class="metric"><div class="label">Vertical RMSE</div><div class="value">{fmt(validation["rectification_vertical_error"]["rmse"])} px</div></div>
    <div class="metric"><div class="label">Evidence Confidence</div><div class="value">{html.escape(str(evidence.get("confidence", "N/A")))}</div></div>
  </div>

  <h2>Dataset</h2>
  <table>
    <tr><th>Used pairs</th><th>Rejected pairs</th><th>Total common corners</th></tr>
    <tr><td>{dataset["used_pairs"]}</td><td>{dataset["rejected_pairs"]}</td><td>{dataset["total_common_corners"]}</td></tr>
  </table>

  <h2>Capture Coach</h2>
  <div class="grid">
    <div class="metric"><div class="label">Dataset Quality</div><div class="value">{fmt(capture.get("dataset_quality_score"), 1)}%</div></div>
    <div class="metric"><div class="label">Joint Coverage</div><div class="value">{fmt(capture.get("joint_coverage_score"), 1)}%</div></div>
    <div class="metric"><div class="label">Usable / Target</div><div class="value">{capture.get("usable_pairs", 0)} / {capture.get("target_pairs", 50)}</div></div>
    <div class="metric"><div class="label">Dataset Ready</div><div class="value">{html.escape(str(capture.get("dataset_ready", False)))}</div></div>
  </div>
  <ul>{recommendations}</ul>

  <h2>Sync Guard</h2>
  <div class="grid">
    <div class="metric"><div class="label">Status</div><div class="value">{html.escape(str(sync.get("status", "N/A")))}</div></div>
    <div class="metric"><div class="label">Median Δt</div><div class="value">{fmt(sync.get("timestamp_delta_ms", {}).get("median"))} ms</div></div>
    <div class="metric"><div class="label">P95 Δt</div><div class="value">{fmt(sync.get("timestamp_delta_ms", {}).get("p95"))} ms</div></div>
    <div class="metric"><div class="label">Jitter</div><div class="value">{fmt(sync.get("jitter_ms"))} ms</div></div>
  </div>

  <h2>Calibration Auditor</h2>
  <div class="grid">
    <div class="metric"><div class="label">Pose Consistency P95</div><div class="value">{fmt(pose.get("translation_error_mm", {}).get("p95"))} mm</div></div>
    <div class="metric"><div class="label">3D Point Error RMSE</div><div class="value">{fmt(recon.get("point_to_pose_error_mm", {}).get("rmse"))} mm</div></div>
    <div class="metric"><div class="label">Plane Error RMSE</div><div class="value">{fmt(recon.get("plane_error_mm", {}).get("rmse"))} mm</div></div>
    <div class="metric"><div class="label">Evidence Model</div><div class="value">{html.escape(str(evidence.get("evidence_model", "N/A")))}</div></div>
  </div>
  <ul>{warnings}</ul>

  <h2>Transform</h2>
  <table>
    <tr><th>Tx</th><th>Ty</th><th>Tz</th><th>Roll</th><th>Pitch</th><th>Yaw</th></tr>
    <tr>
      <td>{fmt(stereo["t_cam2_from_cam1"][0])}</td>
      <td>{fmt(stereo["t_cam2_from_cam1"][1])}</td>
      <td>{fmt(stereo["t_cam2_from_cam1"][2])}</td>
      <td>{fmt(stereo["roll_pitch_yaw_deg"]["roll"])}</td>
      <td>{fmt(stereo["roll_pitch_yaw_deg"]["pitch"])}</td>
      <td>{fmt(stereo["roll_pitch_yaw_deg"]["yaw"])}</td>
    </tr>
  </table>

  <h2>Pair Validation</h2>
  <table>
    <tr><th>Pair</th><th>Common corners</th><th>Epipolar mean</th><th>Vertical mean</th><th>Status</th></tr>
    {pair_rows}
  </table>

  <h2>Full Payload</h2>
  <pre>{matrix_block}</pre>
</body>
</html>
"""
    out.write_text(body, encoding="utf-8")
    return str(out)

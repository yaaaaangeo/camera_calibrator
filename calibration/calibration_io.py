"""
camera_calibrator.calibration.calibration_io
============================================

외부 calibration 파일을 benchmark/비교용 내부 표준 포맷으로 정규화한다.

지원 입력:
- 이 프로젝트의 표준 JSON schema
- OpenCV FileStorage YAML(camera_matrix / distortion_coefficients)
- ROS CameraInfo YAML
- Kalibr camchain YAML(cam0/cam1...)

중요한 원칙: 파라미터 값이 들어 있는 파일 형식과 이후 평가 로직을 분리한다.
비교/benchmark 코드는 StandardCalibration만 보도록 만들면, 외부 파일 포맷이
늘어나도 평가 파이프라인은 바뀌지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from calibration.models.common import distortion_coeff_labels
from calibration.types import CameraModelType


@dataclass
class StandardCalibration:
    """Benchmark 입력으로 쓰는 내부 표준 calibration 표현."""

    label: str
    camera_matrix: np.ndarray
    distortion: np.ndarray
    model_name: CameraModelType | None = None
    distortion_model: str | None = None
    width: int | None = None
    height: int | None = None
    camera_name: str | None = None
    coefficient_order: list[str] = field(default_factory=list)
    source_format: str = "standard_json"
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "label": self.label,
            "camera": {
                "width": self.width,
                "height": self.height,
                "name": self.camera_name,
            },
            "model": {
                "type": self.model_name.value if self.model_name else None,
                "distortion_model": self.distortion_model,
            },
            "intrinsics": {
                "fx": self.fx,
                "fy": self.fy,
                "cx": self.cx,
                "cy": self.cy,
            },
            "distortion": {
                "coefficients": [float(v) for v in self.distortion.reshape(-1)],
                "coefficient_order": self.coefficient_order,
            },
            "source": {
                "format": self.source_format,
                "path": self.source_path,
            },
            "metadata": self.metadata,
        }


def _as_float_matrix(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"{name}가 {shape} 형태가 아닙니다: {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}에 NaN 또는 Inf가 포함되어 있습니다.")
    return arr


def _as_distortion_vector(value: Any, name: str = "distortion") -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1, 1)
    if arr.size == 0:
        raise ValueError(f"{name} 계수가 비어 있습니다.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 계수에 NaN 또는 Inf가 포함되어 있습니다.")
    return arr


def _camera_matrix_from_intrinsics(intrinsics: dict[str, Any]) -> np.ndarray:
    required = ("fx", "fy", "cx", "cy")
    missing = [k for k in required if k not in intrinsics]
    if missing:
        raise ValueError(f"intrinsics에 필수 키가 없습니다: {', '.join(missing)}")
    return np.array(
        [
            [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
            [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _parse_model(value: Any) -> CameraModelType | None:
    if value is None or value == "":
        return None
    text = str(value)
    aliases = {
        "pinhole": CameraModelType.PINHOLE,
        "extended": CameraModelType.EXTENDED_PINHOLE,
        "extended_pinhole": CameraModelType.EXTENDED_PINHOLE,
        "pinhole-radtan": CameraModelType.EXTENDED_PINHOLE,
        "pinhole-equi": CameraModelType.FISHEYE,
        "fisheye": CameraModelType.FISHEYE,
        "equidistant": CameraModelType.FISHEYE,
        "omni-radtan": CameraModelType.EXTENDED_PINHOLE,
    }
    return aliases.get(text.lower(), CameraModelType(text) if text in CameraModelType._value2member_map_ else None)


def _model_from_distortion_model(distortion_model: str | None) -> CameraModelType | None:
    if not distortion_model:
        return None
    text = distortion_model.lower()
    if text in ("equidistant", "fisheye"):
        return CameraModelType.FISHEYE
    if text in ("plumb_bob", "radtan", "rational_polynomial", "radtan8"):
        return CameraModelType.EXTENDED_PINHOLE
    if text in ("none", "no_distortion"):
        return CameraModelType.PINHOLE
    return None


def _order_for(model: CameraModelType | None, count: int) -> list[str]:
    if model is None:
        return [f"d{i}" for i in range(count)]
    return distortion_coeff_labels(model, count)


def _finalize(
    *,
    label: str,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model_name: CameraModelType | None,
    distortion_model: str | None,
    width: int | None,
    height: int | None,
    camera_name: str | None,
    coefficient_order: list[str] | None,
    source_format: str,
    source_path: str | None,
    metadata: dict[str, Any] | None = None,
) -> StandardCalibration:
    K = _as_float_matrix(camera_matrix, (3, 3), "camera_matrix")
    D = _as_distortion_vector(distortion, "distortion")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("fx/fy는 0보다 커야 합니다.")
    if width is not None and width <= 0:
        raise ValueError("image width는 0보다 커야 합니다.")
    if height is not None and height <= 0:
        raise ValueError("image height는 0보다 커야 합니다.")

    resolved_model = model_name or _model_from_distortion_model(distortion_model)
    order = coefficient_order or _order_for(resolved_model, int(D.size))
    return StandardCalibration(
        label=label,
        camera_matrix=K,
        distortion=D,
        model_name=resolved_model,
        distortion_model=distortion_model,
        width=width,
        height=height,
        camera_name=camera_name,
        coefficient_order=order,
        source_format=source_format,
        source_path=source_path,
        metadata=metadata or {},
    )


def _load_standard_dict(data: dict[str, Any], path: str, source_format: str) -> StandardCalibration:
    if "intrinsics" not in data:
        raise ValueError("표준 calibration schema에 intrinsics 섹션이 없습니다.")
    camera = data.get("camera", {})
    model = data.get("model", {})
    distortion = data.get("distortion", {})
    K = _camera_matrix_from_intrinsics(data["intrinsics"])
    coeffs = distortion.get("coefficients")
    if coeffs is None:
        raise ValueError("표준 calibration schema에 distortion.coefficients가 없습니다.")
    return _finalize(
        label=data.get("label") or Path(path).stem,
        camera_matrix=K,
        distortion=coeffs,
        model_name=_parse_model(model.get("type")),
        distortion_model=model.get("distortion_model"),
        width=camera.get("width"),
        height=camera.get("height"),
        camera_name=camera.get("name"),
        coefficient_order=distortion.get("coefficient_order"),
        source_format=source_format,
        source_path=path,
        metadata=data.get("metadata") or {},
    )


def load_standard_json(path: str) -> StandardCalibration:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("표준 calibration JSON의 최상위는 객체여야 합니다.")
    return _load_standard_dict(data, path, "standard_json")


def export_standard_json(calibration: StandardCalibration, path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_opencv_calibration(path: str) -> StandardCalibration:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    try:
        cm_node = fs.getNode("camera_matrix")
        d_node = fs.getNode("distortion_coefficients")
        if cm_node.empty() or d_node.empty():
            raise ValueError(
                "OpenCV YAML에서 'camera_matrix' 또는 'distortion_coefficients' 항목을 찾을 수 없습니다."
            )
        K = cm_node.mat()
        D = d_node.mat()
        model_raw = fs.getNode("calibration_model").string() if not fs.getNode("calibration_model").empty() else None
        width = int(fs.getNode("image_width").real()) if not fs.getNode("image_width").empty() else None
        height = int(fs.getNode("image_height").real()) if not fs.getNode("image_height").empty() else None
        rms = fs.getNode("rms_reprojection_error").real() if not fs.getNode("rms_reprojection_error").empty() else None
    finally:
        fs.release()

    model_name = _parse_model(model_raw)
    return _finalize(
        label=Path(path).stem,
        camera_matrix=K,
        distortion=D,
        model_name=model_name,
        distortion_model=None,
        width=width,
        height=height,
        camera_name=None,
        coefficient_order=None,
        source_format="opencv_yaml",
        source_path=path,
        metadata={"rms_reprojection_error": rms} if rms is not None else {},
    )


def _matrix_data(node: dict[str, Any], name: str) -> np.ndarray:
    if not isinstance(node, dict) or "data" not in node:
        raise ValueError(f"{name} 섹션에 data가 없습니다.")
    rows = int(node.get("rows", 1))
    cols = int(node.get("cols", len(node["data"])))
    data = np.asarray(node["data"], dtype=np.float64)
    if data.size != rows * cols:
        raise ValueError(f"{name}.data 개수와 rows/cols가 맞지 않습니다.")
    return data.reshape(rows, cols)


def load_ros_camera_info(path: str) -> StandardCalibration:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("ROS CameraInfo YAML의 최상위는 객체여야 합니다.")
    K = _matrix_data(data.get("camera_matrix"), "camera_matrix")
    D = _matrix_data(data.get("distortion_coefficients"), "distortion_coefficients")
    distortion_model = data.get("distortion_model")
    return _finalize(
        label=data.get("camera_name") or Path(path).stem,
        camera_matrix=K,
        distortion=D,
        model_name=_model_from_distortion_model(distortion_model),
        distortion_model=distortion_model,
        width=data.get("image_width"),
        height=data.get("image_height"),
        camera_name=data.get("camera_name"),
        coefficient_order=None,
        source_format="ros_camera_info_yaml",
        source_path=path,
    )


def load_kalibr_camchain(path: str, camera_key: str | None = None) -> StandardCalibration:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Kalibr camchain YAML의 최상위는 객체여야 합니다.")
    cam_keys = [k for k, v in data.items() if str(k).startswith("cam") and isinstance(v, dict)]
    if not cam_keys:
        raise ValueError("Kalibr camchain YAML에서 cam0/cam1 섹션을 찾을 수 없습니다.")
    key = camera_key or sorted(cam_keys)[0]
    if key not in data:
        raise ValueError(f"Kalibr camchain YAML에 {key} 섹션이 없습니다.")

    cam = data[key]
    intrinsics = cam.get("intrinsics")
    if not isinstance(intrinsics, list) or len(intrinsics) != 4:
        raise ValueError(f"{key}.intrinsics는 [fx, fy, cx, cy] 4개 값이어야 합니다.")
    K = _camera_matrix_from_intrinsics(
        {"fx": intrinsics[0], "fy": intrinsics[1], "cx": intrinsics[2], "cy": intrinsics[3]}
    )
    distortion_model = cam.get("distortion_model")
    resolution = cam.get("resolution") or [None, None]
    model_name = _parse_model(cam.get("camera_model"))
    distortion_based_model = _model_from_distortion_model(distortion_model)
    if model_name == CameraModelType.PINHOLE and distortion_based_model is not None:
        model_name = distortion_based_model

    return _finalize(
        label=cam.get("rostopic") or key,
        camera_matrix=K,
        distortion=cam.get("distortion_coeffs") or [],
        model_name=model_name or distortion_based_model,
        distortion_model=distortion_model,
        width=resolution[0],
        height=resolution[1],
        camera_name=key,
        coefficient_order=None,
        source_format="kalibr_camchain_yaml",
        source_path=path,
        metadata={"camera_key": key, "rostopic": cam.get("rostopic")},
    )


def _looks_like_opencv_filestorage(path: str) -> bool:
    try:
        first = Path(path).read_text(encoding="utf-8", errors="ignore").lstrip()[:80]
    except OSError:
        return False
    return first.startswith("%YAML") or "!!opencv-matrix" in first


def load_standard_calibration(path: str, *, camera_key: str | None = None) -> StandardCalibration:
    """파일 확장자/내용을 보고 알맞은 loader로 StandardCalibration을 만든다."""
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return load_standard_json(path)
    if _looks_like_opencv_filestorage(path):
        return load_opencv_calibration(path)

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Calibration 파일의 최상위는 객체여야 합니다.")
    if "intrinsics" in data and "distortion" in data:
        return _load_standard_dict(data, path, "standard_yaml")
    if "camera_matrix" in data and "distortion_coefficients" in data:
        return load_ros_camera_info(path)
    if any(str(k).startswith("cam") for k in data):
        return load_kalibr_camchain(path, camera_key=camera_key)
    raise ValueError(
        "알 수 없는 calibration 포맷입니다. 지원: standard JSON, OpenCV YAML, "
        "ROS CameraInfo YAML, Kalibr camchain YAML."
    )

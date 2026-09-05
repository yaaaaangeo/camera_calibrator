"""
camera_calibrator.export.windshield
========================================

Windshield Calibration 전용 YAML export. export/opencv.py::export_opencv_yaml
과 별개의 파일/스키마다 - Base Intrinsic(camera_matrix/distortion_coefficients)
YAML은 항상 그 기존 함수로만 만들고, 여기서는 절대 같은 파일에 덮어쓰거나
그 함수를 수정하지 않는다(사용자 스펙 19번 "Base Intrinsic과 Windshield
Calibration을 분리한다").

스키마:

    base_camera:
      camera_model: ...
      camera_matrix: ...
      distortion_coefficients: ...
    windshield:
      model: baseline | spherical | residual_ray | spline
      train_rms: ...
      test_rms: ...
      fitted_params: {}   # Baseline은 항상 비어 있음
      neural_state_dict_file: <sibling .pt 파일명>   # Residual Ray Neural(STEP 5)에서만 존재

Residual Ray Neural은 fitted_params(flat float dict)에 학습된 weight를
직접 펼쳐 넣지 않는다(사용자 스펙 36번, "float key 수천 개로 넣지 마라") -
`result.neural_state_dict_b64`(base64로 인코딩된 PyTorch state_dict)가
있으면 별도 sibling 파일(`<yaml 파일명 stem>_neural.pt`)에 raw bytes로
저장하고, YAML에는 그 파일명(문자열)만 `neural_state_dict_file` 키로
남긴다. 나머지 fitted_params(architecture 메타데이터/hyperparameter/진단
값)는 기존과 동일하게 flat float로 저장된다.
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2

from calibration.models.common import distortion_coeff_labels
from calibration.types import CameraConfig, CameraModelType
from calibration.windshield.base import WindshieldCalibrationResult, WindshieldModel, WindshieldModelType


def export_windshield_yaml(
    result: WindshieldCalibrationResult,
    camera_config: CameraConfig,
    path: str,
) -> str:
    """WindshieldCalibrationResult를 base_camera/windshield 두 섹션으로 저장한다.

    fitted_params는 모델과 무관하게 flat한 이름->float 값의 dict라는 계약만
    지키면(Baseline/Spherical 둘 다 이 계약을 지킨다) 스키마 변경 없이 그대로
    저장된다 - success=False인 결과만 export를 거부한다.
    """
    if not result.success or result.base_camera_matrix is None or result.base_distortion is None:
        raise ValueError(f"실패한 WindshieldCalibrationResult는 export할 수 없습니다: {result.error_message}")

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("format_version", 1)

    fs.startWriteStruct("base_camera", cv2.FileNode_MAP)
    fs.write("camera_model", result.base_model_name.value)
    fs.write("camera_matrix", result.base_camera_matrix)
    fs.write("distortion_coefficients", result.base_distortion)
    fs.write(
        "distortion_coefficient_order",
        ",".join(distortion_coeff_labels(result.base_model_name, int(result.base_distortion.size))),
    )
    fs.write("image_width", camera_config.width)
    fs.write("image_height", camera_config.height)
    fs.endWriteStruct()

    fs.startWriteStruct("windshield", cv2.FileNode_MAP)
    fs.write("model", result.windshield_model.value)
    fs.write("train_rms", float(result.residual_stats.rmse) if result.residual_stats and result.residual_stats.rmse is not None else -1.0)
    fs.write("test_rms", float(result.test_residual_stats.rmse) if result.test_residual_stats and result.test_residual_stats.rmse is not None else -1.0)
    fitted_param_keys = sorted(result.fitted_params.keys())
    fs.write("fitted_param_names", ",".join(fitted_param_keys))
    for key in fitted_param_keys:
        fs.write(f"fitted_param_{key}", float(result.fitted_params[key]))
    if result.neural_state_dict_b64:
        neural_filename = Path(path).stem + "_neural.pt"
        neural_path = Path(path).parent / neural_filename
        neural_path.write_bytes(base64.b64decode(result.neural_state_dict_b64.encode("ascii")))
        fs.write("neural_state_dict_file", neural_filename)
    fs.endWriteStruct()

    fs.release()
    return path


def load_windshield_yaml(path: str) -> dict:
    """저장한 파일을 다시 읽어 dict로 반환 (재현성 검증/round-trip 테스트용).

    fitted_param_names(콤마로 구분된 키 목록) + fitted_param_<key>(각 float
    값)를 다시 하나의 fitted_params dict로 복원한다 - export_windshield_yaml
    이 쓴 것과 대칭인 읽기.
    """
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    base_node = fs.getNode("base_camera")
    windshield_node = fs.getNode("windshield")

    names_raw = windshield_node.getNode("fitted_param_names").string() or ""
    fitted_param_keys = [k for k in names_raw.split(",") if k]
    fitted_params = {
        key: windshield_node.getNode(f"fitted_param_{key}").real() for key in fitted_param_keys
    }

    neural_state_dict_b64 = None
    neural_node = windshield_node.getNode("neural_state_dict_file")
    neural_filename = neural_node.string() if not neural_node.empty() else None
    if neural_filename:
        neural_path = Path(path).parent / neural_filename
        neural_state_dict_b64 = base64.b64encode(neural_path.read_bytes()).decode("ascii")

    data = {
        "base_camera": {
            "camera_model": base_node.getNode("camera_model").string(),
            "camera_matrix": base_node.getNode("camera_matrix").mat(),
            "distortion_coefficients": base_node.getNode("distortion_coefficients").mat(),
            "image_width": int(base_node.getNode("image_width").real()),
            "image_height": int(base_node.getNode("image_height").real()),
        },
        "windshield": {
            "model": windshield_node.getNode("model").string(),
            "train_rms": windshield_node.getNode("train_rms").real(),
            "test_rms": windshield_node.getNode("test_rms").real(),
            "fitted_params": fitted_params,
            "neural_state_dict_b64": neural_state_dict_b64,
        },
    }
    fs.release()
    return data


def windshield_model_from_yaml(path: str) -> WindshieldModel:
    """Windshield YAML을 다시 읽어 실행 가능한 WindshieldModel로 재구성한다
    (project_point/unproject_pixel 바로 호출 가능) - 저장만 하고 못 쓰는
    일회성 export가 아니라 runtime에서 재사용 가능해야 한다는 요구사항.

    BASELINE/SPHERICAL 분기 로직을 여기서 새로 만들지 않고,
    calibration.windshield.projection.build_projector() 하나에만 위임한다 -
    Calibration 도중에 만든 결과든 YAML에서 막 불러온 결과든 같은 dispatch
    경로 하나로 모델을 얻는다.
    """
    from calibration.windshield.projection import build_projector

    data = load_windshield_yaml(path)
    base = data["base_camera"]
    ws = data["windshield"]

    result = WindshieldCalibrationResult(
        windshield_model=WindshieldModelType(ws["model"]),
        base_model_name=CameraModelType(base["camera_model"]),
        base_camera_matrix=base["camera_matrix"],
        base_distortion=base["distortion_coefficients"],
        fitted_params=ws["fitted_params"],
        success=True,
        neural_state_dict_b64=ws.get("neural_state_dict_b64"),
    )
    return build_projector(result)

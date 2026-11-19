"""
shared.camera
=============

카메라 내부 파라미터 한 벌 - Camera Intrinsic 결과와 CAMERA–LiDAR 렌즈 프리셋을 잇는다.

* 렌즈 프리셋 파일은 하나다: ``extrinsic/gui/config/camera_intrinsics.yaml``
  (CAMERA–LiDAR 3단계가 카메라마다 이 목록에서 렌즈를 고른다).
* Camera Intrinsic 결과(``CalibrationResult``) 또는 결과 파일(OpenCV YAML·ROS camera_info·
  Kalibr·표준 JSON - ``calibration.calibration_io``)을 같은 ``LensParams`` 로 바꿔
  프리셋에 등록한다.

모델 대응: Ideal Pinhole·Brown-Conrady·Rational(8계수) -> ``pinhole`` (OpenCV 순서
k1 k2 p1 p2 k3 k4 k5 k6), Fisheye(Kannala-Brandt) -> ``fisheye`` (k1..k4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

LENS_PRESETS_PATH = (Path(__file__).resolve().parents[2] / "extrinsic" / "gui" / "config" / "camera_intrinsics.yaml")
COEFF_KEYS = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")


@dataclass
class LensParams:
    model: str  # "pinhole" | "fisheye"
    fx: float
    fy: float
    cx: float
    cy: float
    coeffs: dict = field(default_factory=dict)  # k1..k6, p1, p2 (없는 것은 0)
    width: int | None = None
    height: int | None = None

    def as_preset_fields(self) -> dict:
        out = {"model": self.model, "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}
        out.update({k: float(self.coeffs.get(k, 0.0)) for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")})
        return out


class KD:
    """K 와 D 를 *그대로* 들고 shared.camera_ops 에 넘기는 어댑터 (계수를 다시 정리하지 않는다).

    인트린직 계산(검증·잔차)은 결과와 비트 단위로 같아야 하므로 D 배열을 손대지 않는다.
    """

    def __init__(self, K, D, fisheye: bool):
        self._K = np.asarray(K, np.float64)
        self._D = np.asarray(D, np.float64)
        self.is_fisheye = bool(fisheye)
        self.model = "fisheye" if fisheye else "pinhole"

    def matrix(self):
        return self._K

    def dist(self):
        return self._D

    fx = property(lambda self: float(self._K[0, 0]))
    fy = property(lambda self: float(self._K[1, 1]))
    cx = property(lambda self: float(self._K[0, 2]))
    cy = property(lambda self: float(self._K[1, 2]))


def _is_fisheye(model) -> bool:
    text = str(getattr(model, "value", model) or "").lower()
    return "fisheye" in text or "equidistant" in text or "kannala" in text


def lens_from_kd(K, D, model=None, width: int | None = None, height: int | None = None) -> LensParams:
    """3x3 K 와 OpenCV 순서의 왜곡 계수 D, 모델 이름 -> LensParams."""
    K = np.asarray(K, np.float64).reshape(3, 3)
    d = [float(x) for x in np.asarray(D if D is not None else [], np.float64).ravel()]
    if _is_fisheye(model):
        coeffs = {f"k{i + 1}": v for i, v in enumerate(d[:4])}
        kind = "fisheye"
    else:
        coeffs = dict(zip(COEFF_KEYS, d[:8]))
        kind = "pinhole"
    return LensParams(kind, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), coeffs,
                      width=int(width) if width else None, height=int(height) if height else None)


def lens_from_calibration_result(result, width: int | None = None, height: int | None = None) -> LensParams:
    """Camera Intrinsic 의 CalibrationResult -> LensParams."""
    if result is None or result.camera_matrix is None:
        raise ValueError("캘리브레이션 결과가 없습니다")
    return lens_from_kd(result.camera_matrix, result.distortion, result.model_name, width, height)


def read_lens_file(path: str | Path) -> LensParams:
    """Camera Intrinsic 결과 파일(또는 다른 툴의 OpenCV YAML·ROS camera_info·Kalibr) -> LensParams."""
    from calibration.calibration_io import load_standard_calibration

    cal = load_standard_calibration(str(path))
    model = cal.model_name if cal.model_name is not None else cal.distortion_model
    return lens_from_kd(cal.camera_matrix, cal.distortion, model, cal.width, cal.height)


def load_preset_entries(path: Path | None = None) -> list[dict]:
    path = Path(path or LENS_PRESETS_PATH)
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(data.get("cameras") or [])


def add_lens_preset(name: str, lens: LensParams, note: str = "", path: Path | None = None,
                    replace: bool = True) -> Path:
    """렌즈 프리셋 목록에 `name` 으로 추가한다. 같은 이름이 있으면 `replace` 일 때 바꾼다."""
    name = name.strip()
    if not name:
        raise ValueError("프리셋 이름이 비어 있습니다")
    path = Path(path or LENS_PRESETS_PATH)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    data = data or {}
    entries = list(data.get("cameras") or [])
    entry = {"name": name, "note": note, **lens.as_preset_fields()}
    existing = [i for i, e in enumerate(entries) if str(e.get("name")) == name]
    if existing and not replace:
        raise ValueError(f"같은 이름의 프리셋이 이미 있습니다: {name}")
    if existing:
        entries[existing[0]] = entry
    else:
        entries.append(entry)
    data["cameras"] = entries
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path

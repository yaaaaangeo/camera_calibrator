"""Reading an extrinsic someone already has, to project with before (or against) a fresh one.

Three formats turn up around this rig, and they do not agree on direction:

    result.yaml      written by this tool. `T_camera_lidar` maps LiDAR points
                     into the camera optical frame -- the projection direction.
    extrinsic.txt    FAST-LIVO2: `Rcl`, `Pcl`, also LiDAR -> camera.
    *_calibration.yaml
                     the vehicle files: `sensors.<name>.extrinsic.matrix` with
                     `direction: T_parent_child`, parent the LiDAR, child the
                     camera. That maps camera coordinates into the LiDAR frame,
                     so it is inverted here. The file itself warns that using it
                     as published is "a silent error worth about 1900 px".

Everything comes back as a `solve.Solution` (R, t mapping LiDAR into the camera),
so it projects through exactly the same code as a fresh result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from gui.core.solve import Solution


@dataclass
class Extrinsic:
    name: str  # which camera, for files holding several
    solution: Solution
    source: str  # file and format, for the record
    note: str = ""


def _sol(R, t) -> Solution:
    return Solution(ok=True, R=np.asarray(R, float).reshape(3, 3), t=np.asarray(t, float).reshape(3))


def _from_T_camera_lidar(T) -> Solution:
    T = np.asarray(T, float).reshape(4, 4)
    return _sol(T[:3, :3], T[:3, 3])


def load(path: str | Path) -> list[Extrinsic]:
    """Every camera extrinsic the file holds. Raises ValueError if none."""
    path = Path(path)
    text = path.read_text(errors="replace")
    out: list[Extrinsic] = []

    if "Rcl" in text and "Pcl" in text:
        nums = lambda key: [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                                                         re.search(key + r"\s*:\s*\[([^\]]*)\]", text, re.S).group(1))]
        out.append(Extrinsic(path.stem, _sol(nums("Rcl"), nums("Pcl")), f"{path.name} (FAST-LIVO2)"))
        return out

    doc = yaml.safe_load(text) or {}
    if "T_camera_lidar" in doc:
        out.append(Extrinsic(doc.get("camera_topic") or path.parent.name,
                             _from_T_camera_lidar(doc["T_camera_lidar"]), f"{path.name} (room result)"))
    elif "R_lidar_to_camera" in doc:
        out.append(Extrinsic(doc.get("camera_topic") or path.parent.name,
                             _sol(doc["R_lidar_to_camera"], doc["t_lidar_to_camera"]), f"{path.name} (room result)"))
    for name, sensor in (doc.get("sensors") or {}).items():
        if not isinstance(sensor, dict) or sensor.get("type") != "camera":
            continue
        ex = sensor.get("extrinsic") or {}
        if "matrix" not in ex:
            continue
        M = np.asarray(ex["matrix"], float).reshape(4, 4)
        direction = ex.get("direction", "T_parent_child")
        T_cam_lidar = np.linalg.inv(M) if direction == "T_parent_child" else M
        out.append(Extrinsic(name, _from_T_camera_lidar(T_cam_lidar),
                             f"{path.name} (차량 YAML, {direction} 역변환)" if direction == "T_parent_child" else path.name,
                             note=f"parent {ex.get('parent', '?')}"))
    if not out:
        raise ValueError(f"{path.name}: extrinsic 을 찾지 못했습니다 (result.yaml / FAST-LIVO2 / 차량 YAML 형식 지원)")
    return out

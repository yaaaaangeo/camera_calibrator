"""
shared.vehicle
==============

차량 목록 - Camera Intrinsic 과 CAMERA–LiDAR 가 같이 쓴다.

앱을 열면 홈 화면에서 차량부터 고른다 (차량에 달지 않은 카메라는 "차량 없음").
차량마다 프로필 파일 하나가 있다 - CAMERA–LiDAR 가 쓰던 것과 같은 파일이다::

    extrinsic/calib_result/<차량>/vehicle.yaml
      vehicle: torres03
      lidar_topics: [...]                      CAMERA–LiDAR 가 채운다
      cameras: {<토픽>: {name: front, preset: torres03_front}}
      lenses: {front: torres03_front}          카메라 이름 -> 렌즈 프리셋 (인트린직 결과)
      roi: [...]  target: ...  layouts: ...  note: ...

* 카메라 이름 목록 = ``cameras`` 의 이름 + ``lenses`` 의 이름. 인트린직에서 새 이름을
  쓰면 ``lenses`` 에 (렌즈 없이) 추가돼 다음부터 목록에 나온다.
* 인트린직 결과를 "CAMERA–LiDAR 렌즈로 등록"하면 ``lenses[카메라]`` 와, 같은 이름을 가진
  ``cameras`` 항목의 ``preset`` 이 그 프리셋으로 바뀐다 - CAMERA–LiDAR 3단계가 자동으로 고른다.
* 결과 저장소(Git)에는 ``vehicles/<차량>/vehicle.yaml`` 과, 그 차량 카메라들이 쓰는 렌즈
  프리셋 ``vehicles/<차량>/lenses.yaml`` 이 같이 올라간다 - 다른 PC 가 그대로 가져온다.

이 모듈은 YAML 을 dict 로만 다룬다. CAMERA–LiDAR 의 VehicleProfile 도 ``lenses`` 를 읽고
쓰므로 어느 쪽이 저장해도 서로의 내용이 지워지지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from shared.naming import safe_name

RESULT_DIR = Path(__file__).resolve().parents[2] / "extrinsic" / "calib_result"  # = gui.core.project.RESULT_DIR
PROFILE = "vehicle.yaml"
NO_VEHICLE = ""  # "차량 없음" (카메라 단품)
NO_VEHICLE_LABEL = "차량 없음 (카메라 단품)"
NO_VEHICLE_PATH = "no_vehicle"  # Git 경로에서 {vehicle} 자리에 들어가는 이름


def vehicle_name(text: str) -> str:
    return safe_name(text.strip()) if text and text.strip() else NO_VEHICLE


def vehicle_dir(name: str, root: Path | None = None) -> Path:
    return Path(root or RESULT_DIR) / safe_name(name)


def profile_path(name: str, root: Path | None = None) -> Path:
    return vehicle_dir(name, root) / PROFILE


def list_vehicles(root: Path | None = None) -> list[str]:
    """프로필이 있는 차량, 최근에 쓴 것부터."""
    root = Path(root or RESULT_DIR)
    if not root.exists():
        return []
    found = [p for p in root.iterdir() if (p / PROFILE).is_file()]
    found.sort(key=lambda p: (p / PROFILE).stat().st_mtime, reverse=True)
    out = []
    for p in found:
        try:
            out.append(str((yaml.safe_load((p / PROFILE).read_text(encoding="utf-8")) or {}).get("vehicle") or p.name))
        except (yaml.YAMLError, OSError):
            out.append(p.name)
    return out


def load(name: str, root: Path | None = None) -> dict:
    p = profile_path(name, root)
    if not p.is_file():
        return {"vehicle": name}
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        data = {}
    data.setdefault("vehicle", name)
    return data


def save(name: str, data: dict, root: Path | None = None) -> Path:
    p = profile_path(name, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = dict(data)
    data["vehicle"] = data.get("vehicle") or name
    p.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def ensure(name: str, root: Path | None = None) -> Path:
    """차량 프로필이 없으면 빈 것으로 만든다."""
    p = profile_path(name, root)
    if not p.is_file():
        save(name, {"vehicle": name}, root)
    return p


def camera_names(name: str, root: Path | None = None) -> list[str]:
    """차량에 등록된 카메라 이름 (CAMERA–LiDAR 의 카메라 + 인트린직에서 쓴 이름)."""
    if not name:
        return []
    data = load(name, root)
    out: list[str] = []
    for entry in (data.get("cameras") or {}).values():
        n = str((entry or {}).get("name") or "")
        if n and n not in out:
            out.append(n)
    for n in (data.get("lenses") or {}):
        if str(n) not in out:
            out.append(str(n))
    return out


def lens_of(name: str, camera: str, root: Path | None = None) -> str:
    """차량 카메라의 렌즈 프리셋 이름 (없으면 '')."""
    if not name or not camera:
        return ""
    data = load(name, root)
    lens = str((data.get("lenses") or {}).get(camera) or "")
    if lens:
        return lens
    for entry in (data.get("cameras") or {}).values():
        if str((entry or {}).get("name")) == camera and (entry or {}).get("preset"):
            return str(entry["preset"])
    return ""


def add_camera(name: str, camera: str, root: Path | None = None) -> bool:
    """카메라 이름을 차량에 추가한다 (이미 있으면 그대로). 추가했으면 True."""
    camera = camera.strip()
    if not name or not camera or camera in camera_names(name, root):
        return False
    data = load(name, root)
    lenses = dict(data.get("lenses") or {})
    lenses[camera] = ""
    data["lenses"] = lenses
    save(name, data, root)
    return True


def set_lens(name: str, camera: str, preset: str, root: Path | None = None) -> None:
    """차량 카메라의 렌즈를 `preset` 으로 정한다 (CAMERA–LiDAR 가 이 렌즈를 자동으로 고른다)."""
    if not name or not camera:
        return
    data = load(name, root)
    lenses = dict(data.get("lenses") or {})
    lenses[camera] = preset
    data["lenses"] = lenses
    cams = dict(data.get("cameras") or {})
    for topic, entry in cams.items():
        if str((entry or {}).get("name")) == camera:
            cams[topic] = {**(entry or {}), "preset": preset}
    if cams:
        data["cameras"] = cams
    save(name, data, root)


def used_presets(name: str, root: Path | None = None) -> list[str]:
    """차량 카메라들이 쓰는 렌즈 프리셋 이름."""
    data = load(name, root)
    out = [str(v) for v in (data.get("lenses") or {}).values() if v]
    out += [str((e or {}).get("preset")) for e in (data.get("cameras") or {}).values() if (e or {}).get("preset")]
    return list(dict.fromkeys(out))


def summary(name: str, root: Path | None = None) -> str:
    """차량 한 줄 설명 (홈 화면용)."""
    if not name:
        return "결과는 차량 없이 카메라 이름으로만 저장됩니다. CAMERA–LiDAR 는 차량이 있어야 합니다."
    if not profile_path(name, root).is_file():
        return "새 차량 - 시작하면 만들어집니다."
    data = load(name, root)
    cams = camera_names(name, root)
    with_lens = [c for c in cams if lens_of(name, c, root)]
    lidar = ", ".join(data.get("lidar_topics") or []) or "—"
    return (f"카메라 {len(cams)}대 ({', '.join(cams) or '—'}) · 렌즈 등록 {len(with_lens)}대 · LiDAR {lidar}")

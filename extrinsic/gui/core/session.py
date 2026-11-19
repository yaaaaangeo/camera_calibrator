"""Vehicles and calibration sessions: where results go, and what is remembered.

    calib_result/
      <vehicle>/
        vehicle.yaml                 the profile: LiDAR topics, ROI, cameras
        2026-11-18_143205/           one session = one capture of the room
          session.yaml               when, what, and every camera's numbers
          capture/                   the capture as rosbag2 (mcap), replayable
          FRONT_LEFT/                one folder per camera
            extrinsic.txt  result.yaml  boards.yaml  overlay.png  report.txt  report.html

The profile is what makes the second calibration of a vehicle quicker than the
first: which topic is which camera, which preset its lens took, where in the
room the boards stand (the ROI) -- none of that changes between sessions.

A session is named after the moment of its capture, not the moment it was
saved: that is the date the extrinsic is true for.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from gui.core.project import RESULT_DIR, FilterBox
from shared.naming import safe_name as _safe_name

PROFILE = "vehicle.yaml"
SUMMARY = "session.yaml"
SESSION_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")


def safe_name(text: str) -> str:
    """A name usable as one path component: letters (any script), digits, - and _."""
    return _safe_name(text)


def camera_name_from_topic(topic: str) -> str:
    """A first guess at a camera's name from its topic: the parts that are not boilerplate.

    '/camera/left/image_raw/compressed' -> 'left', '/video0/image_raw/compressed'
    -> 'video0', '/sensor/camera/front_wide/compressed' -> 'front_wide'.
    """
    noise = {"", "camera", "cameras", "sensor", "sensors", "image", "image_raw", "image_rect", "image_color",
             "image_rect_color", "compressed", "raw", "color", "rgb"}
    parts = [p for p in topic.split("/") if p.lower() not in noise]
    return safe_name("_".join(parts) if parts else topic)


def session_id(stamp_ns: int | None = None) -> str:
    t = time.time() if not stamp_ns else stamp_ns / 1e9
    return time.strftime("%Y-%m-%d_%H%M%S", time.localtime(t))


def session_label(sid: str) -> str:
    """'2026-11-18_143205' -> '2026-11-18 14:32:05'."""
    if not SESSION_RE.match(sid):
        return sid
    d, t = sid.split("_")
    return f"{d} {t[:2]}:{t[2:4]}:{t[4:]}"


# =================================================================== profile


@dataclass
class CameraEntry:
    name: str  # position, e.g. FRONT_LEFT -- the result folder
    preset: str = ""  # intrinsics preset last used
    direction: str = ""  # which way it faces, a key of room.DIRECTIONS ('' = unknown)


@dataclass
class VehicleProfile:
    name: str
    lidar_topics: list = field(default_factory=list)
    cameras: dict = field(default_factory=dict)  # topic -> CameraEntry
    lenses: dict = field(default_factory=dict)  # camera name -> lens preset (Camera Intrinsic, shared.vehicle)
    roi: list = field(default_factory=list)  # FilterBox
    target: str = ""
    layouts: str = ""
    note: str = ""
    # Vehicle forward as an azimuth in the LiDAR frame, degrees: found from the
    # cameras with a direction that see several boards, kept for the next session.
    forward_deg: float | None = None

    @property
    def dir(self) -> Path:
        return vehicle_dir(self.name)

    def camera_name(self, topic: str) -> str:
        e = self.cameras.get(topic)
        return e.name if e else camera_name_from_topic(topic)

    def lens_for(self, topic: str, name: str = "") -> str:
        """The lens preset for this camera: the one picked here last time, else the one
        Camera Intrinsic registered for its name (both are set together on registration)."""
        e = self.cameras.get(topic)
        if e is not None and e.preset:
            return e.preset
        return str(self.lenses.get(name or self.camera_name(topic)) or "")

    def camera_direction(self, topic: str, name: str = "") -> str:
        """The direction set for this camera, else one read from its name or topic."""
        from gui.core.room import guess_direction

        e = self.cameras.get(topic)
        if e is not None and e.direction:
            return e.direction
        return guess_direction(name or (e.name if e else "")) or guess_direction(topic)

    def to_dict(self) -> dict:
        return {
            "vehicle": self.name,
            "lidar_topics": list(self.lidar_topics),
            "cameras": {t: {"name": e.name, "preset": e.preset, "direction": e.direction}
                        for t, e in self.cameras.items()},
            "lenses": dict(self.lenses),
            "roi": [_box_dict(b) for b in self.roi],
            "target": self.target,
            "layouts": self.layouts,
            "note": self.note,
            "forward_deg": None if self.forward_deg is None else round(float(self.forward_deg), 2),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VehicleProfile":
        cams = {t: CameraEntry(v.get("name") or camera_name_from_topic(t), v.get("preset", ""),
                               v.get("direction", "") or "")
                for t, v in (d.get("cameras") or {}).items()}
        roi = [FilterBox(**{k: float(v) for k, v in b.items() if k in FilterBox.__dataclass_fields__})
               for b in d.get("roi") or []]
        return cls(name=str(d.get("vehicle", "")), lidar_topics=list(d.get("lidar_topics") or []),
                   cameras=cams, lenses={str(k): str(v or "") for k, v in (d.get("lenses") or {}).items()},
                   roi=roi, target=d.get("target", "") or "", layouts=d.get("layouts", "") or "",
                   note=d.get("note", "") or "",
                   forward_deg=None if d.get("forward_deg") is None else float(d["forward_deg"]))

    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / PROFILE).write_text(yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False))


def _box_dict(b: FilterBox) -> dict:
    return {k: round(float(v), 4) for k, v in asdict(b).items()}


def vehicle_dir(name: str, root: Path | None = None) -> Path:
    return (root or RESULT_DIR) / safe_name(name)


def list_vehicles(root: Path | None = None) -> list[str]:
    """Vehicles that have a profile, most recently used first."""
    root = root or RESULT_DIR
    if not root.exists():
        return []
    found = [p for p in root.iterdir() if (p / PROFILE).exists()]
    found.sort(key=lambda p: (p / PROFILE).stat().st_mtime, reverse=True)
    out = []
    for p in found:
        try:
            out.append(str((yaml.safe_load((p / PROFILE).read_text()) or {}).get("vehicle") or p.name))
        except yaml.YAMLError:
            out.append(p.name)
    return out


def load_profile(name: str, root: Path | None = None) -> VehicleProfile:
    """The vehicle's profile, or a fresh one if it has none yet."""
    p = vehicle_dir(name, root) / PROFILE
    if p.exists():
        prof = VehicleProfile.from_dict(yaml.safe_load(p.read_text()) or {})
        prof.name = prof.name or name
        return prof
    return VehicleProfile(name=name)


# ================================================================== sessions


@dataclass
class SessionInfo:
    id: str
    path: Path
    cameras: list  # [(name, image px or None, ok)]
    note: str = ""

    @property
    def label(self) -> str:
        return session_label(self.id)


def list_sessions(name: str, root: Path | None = None) -> list[SessionInfo]:
    """The vehicle's past sessions, newest first, with each camera's headline number."""
    d = vehicle_dir(name, root)
    if not d.exists():
        return []
    out = []
    for p in sorted((q for q in d.iterdir() if q.is_dir() and SESSION_RE.match(q.name)), reverse=True):
        cams = []
        s = p / SUMMARY
        if s.exists():
            try:
                doc = yaml.safe_load(s.read_text()) or {}
            except yaml.YAMLError:
                doc = {}
            for c in doc.get("cameras") or []:
                cams.append((c.get("name", "?"), c.get("image_rmse_px"), bool(c.get("ok", True))))
            note = doc.get("note", "") or ""
        else:
            cams = [(q.name, None, (q / "result.yaml").exists()) for q in sorted(p.iterdir())
                    if q.is_dir() and q.name != "capture"]
            note = ""
        out.append(SessionInfo(p.name, p, cams, note))
    return out


def previous_extrinsics(name: str, root: Path | None = None) -> dict:
    """camera name -> (Extrinsic, session id) from the newest session that solved it.

    Last time's answer is the best guess for this time's: shown as the prior on
    the live view, it says at a glance whether anything has moved.
    """
    from gui.core import extrinsic_io

    out = {}
    for sess in list_sessions(name, root):
        for cam_dir in sess.path.iterdir():
            f = cam_dir / "result.yaml"
            if cam_dir.name in out or not f.exists():
                continue
            try:
                ex = extrinsic_io.load(f)[0]
            except (ValueError, OSError, yaml.YAMLError, KeyError, IndexError):
                continue
            ex.name = f"{cam_dir.name} ({session_label(sess.id)})"
            out[cam_dir.name] = (ex, sess.id)
    return out


# ==================================================================== saving


def pose_text(sol) -> str:
    """The result read several ways, since every consumer wants a different one."""
    import cv2

    R, t = sol.R, sol.t
    rvec = cv2.Rodrigues(R)[0].ravel()
    Rl = R.T  # camera axes in the LiDAR frame
    pos = -R.T @ t
    # camera_link-style yaw/pitch/roll of the optical frame in the LiDAR frame
    yaw = np.degrees(np.arctan2(Rl[1, 0], Rl[0, 0]))
    pitch = np.degrees(np.arcsin(-np.clip(Rl[2, 0], -1, 1)))
    roll = np.degrees(np.arctan2(Rl[2, 1], Rl[2, 2]))
    return (f"LiDAR→카메라  t = [{t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}] m   rvec = [{rvec[0]:+.5f}, {rvec[1]:+.5f}, {rvec[2]:+.5f}]\n"
            f"카메라 위치 (LiDAR 좌표)  [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}] m\n"
            f"광학 프레임 자세 (LiDAR 기준, ZYX)  yaw {yaw:+.2f}°  pitch {pitch:+.2f}°  roll {roll:+.2f}°")


ALL_FILES = ("extrinsic", "yaml", "boards", "overlay", "report")


def save_camera(out: Path, *, res, camera, camera_preset: str, image, cloud, intensity,
                meta: dict, files=ALL_FILES, excluded=()) -> list[str]:
    """Write one camera's result into `out`. Returns the file names written.

    `meta` goes at the top of result.yaml as is: vehicle, session, topics --
    whatever says where the numbers came from.
    """
    import cv2

    from gui.core import room
    from gui.core import solve as S
    from gui.core.room_overlay import OverlayStyle, render

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sol = res.solution
    h, w = image.shape[:2]
    written = []
    if "extrinsic" in files:
        (out / "extrinsic.txt").write_text(S.to_fast_livo2(sol, camera, w, h))
        written.append("extrinsic.txt")
    if "yaml" in files:
        T = sol.matrix
        doc = {
            **meta,
            "camera_preset": camera_preset, "camera_model": camera.model_name, "image_size": [int(w), int(h)],
            "T_camera_lidar": np.round(T, 9).tolist(),
            "T_lidar_camera": np.round(np.linalg.inv(T), 9).tolist(),
            "direction_note": "T_camera_lidar maps LiDAR points into the camera optical frame",
            "rmse_m": round(float(sol.rmse), 5), "image_rmse_px": round(float(res.image_rmse), 3),
            "solver": "weighted" if res.weighted else "kabsch",
            "holdout_px": {k: (None if not np.isfinite(v) else round(float(v), 3)) for k, v in res.holdout.items()},
            "boards": [{"pair": sid, "rmse_m": round(sol.per_scene_rmse.get(sid, 0.0), 5),
                        "px": round(res.per_board_px.get(sid, 0.0), 3),
                        **{k: round(v, 4) for k, v in res.loo.get(sid, {}).items()}}
                       for sid, _, _ in res.scenes],
            "excluded": sorted(excluded), "notes": res.notes,
        }
        (out / "result.yaml").write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False))
        written.append("result.yaml")
    if "boards" in files:
        doc = [{"pair": sid, "lidar_centres": np.round(L, 5).tolist(), "camera_centres": np.round(C, 5).tolist()}
               for sid, L, C in res.scenes]
        (out / "boards.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
        written.append("boards.yaml")
    if "overlay" in files:
        img, _ = render(image, camera, res, cloud, intensity, OverlayStyle())
        cv2.imwrite(str(out / "overlay.png"), img)
        written.append("overlay.png")
    if "report" in files:
        head = "\n".join(f"{k}: {v}" for k, v in meta.items() if isinstance(v, (str, int, float)))
        (out / "report.txt").write_text(head + "\n\n" + room.report(res) + "\n\n" + pose_text(sol) + "\n")
        (out / "report.html").write_text(
            room.report_html(res, meta, pose_text(sol), "overlay.png" if "overlay.png" in written else None),
            encoding="utf-8")
        written += ["report.txt", "report.html"]
    return written


def write_summary(session_path: Path, *, vehicle: str, sid: str, source: str, lidar_topics, n_sweeps: int,
                  roi, target: str, cameras: list[dict], note: str = "") -> Path:
    """session.yaml: one page that says what this session was and how each camera came out.

    Rewritten on every save, merging with what is there: cameras saved in an
    earlier save of the same session keep their entries.
    """
    session_path = Path(session_path)
    session_path.mkdir(parents=True, exist_ok=True)
    f = session_path / SUMMARY
    old = {}
    if f.exists():
        try:
            old = {c["name"]: c for c in (yaml.safe_load(f.read_text()) or {}).get("cameras") or []}
        except (yaml.YAMLError, KeyError, TypeError):
            old = {}
    for c in cameras:
        old[c["name"]] = c
    doc = {
        "vehicle": vehicle,
        "session": sid,
        "date": session_label(sid),
        "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "lidar_topics": list(lidar_topics),
        "lidar_sweeps": int(n_sweeps),
        "roi": [_box_dict(b) for b in roi or []],
        "target": target,
        "note": note,
        "cameras": sorted(old.values(), key=lambda c: c["name"]),
    }
    f.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False))
    return f

"""Session-oriented output paths and manifest bookkeeping.

Serializers deliberately do not live here.  This module only decides *where* an
existing exporter writes and records the resulting artifact, so export formats
and calibration numerics remain unchanged.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping


APPLICATION_NAME = "camera-calibrator"
APPLICATION_VERSION = "0.2.0"
DEFAULT_OUTPUT_ROOT = Path.home() / "CameraCalibratorOutputs"

SESSION_DIRECTORIES: tuple[str, ...] = (
    "project",
    "intrinsic",
    "intrinsic/subset",
    "intrinsic/kalibr",
    "validation/subset_comparison",
    "paper",
    "windshield/geometry",
    "windshield/reflection",
    "windshield/ghost",
    "reports",
)

_MODEL_FILENAMES = {
    "pinhole": "pinhole",
    "brown_conrady": "brown_conrady",
    "brown-conrady": "brown_conrady",
    "brown": "brown_conrady",
    "extended_pinhole": "rational",
    "rational": "rational",
    "fisheye": "fisheye",
}


def _plain(value: Any) -> Any:
    """Convert common project values to JSON-safe, stable primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    return str(value)


def _safe_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._-")


def model_filename(model: Any) -> str:
    raw = getattr(model, "value", model)
    key = str(raw).strip().lower().replace(" ", "_")
    return _MODEL_FILENAMES.get(key, _safe_component(key) or "model")


class OutputManager:
    """Own one output root and, lazily, one collision-safe export session."""

    def __init__(
        self,
        output_root: str | Path | None = None,
        *,
        camera_name: str = "",
        now: Callable[[], datetime] | None = None,
        session_dir: str | Path | None = None,
    ) -> None:
        self.output_root = Path(output_root or DEFAULT_OUTPUT_ROOT).expanduser()
        self._now = now or datetime.now
        self.session_dir: Path | None = None
        self.manifest: dict[str, Any] = {}
        if session_dir is not None:
            self._initialize(Path(session_dir).expanduser(), camera_name)

    @property
    def active(self) -> bool:
        return self.session_dir is not None

    @property
    def session_id(self) -> str | None:
        return self.session_dir.name if self.session_dir is not None else None

    @property
    def intrinsic_dir(self) -> Path:
        return self.directory("intrinsic")

    @property
    def validation_dir(self) -> Path:
        return self.directory("validation")

    @property
    def paper_dir(self) -> Path:
        return self.directory("paper")

    @property
    def windshield_dir(self) -> Path:
        return self.directory("windshield")

    @property
    def report_dir(self) -> Path:
        return self.directory("reports")

    @property
    def manifest_path(self) -> Path:
        return self.require_session() / "manifest.json"

    def ensure_session(self, camera_name: str = "") -> Path:
        if self.session_dir is not None:
            return self.session_dir
        moment = self._now()
        stem = moment.strftime("%Y-%m-%d_%H%M%S")
        camera = _safe_component(camera_name)
        if camera:
            stem += f"_{camera}"
        candidate = self.output_root / stem
        suffix = 2
        while candidate.exists():
            candidate = self.output_root / f"{stem}_{suffix:02d}"
            suffix += 1
        self._initialize(candidate, camera_name, created_at=moment)
        return candidate

    def require_session(self) -> Path:
        if self.session_dir is None:
            return self.ensure_session()
        return self.session_dir

    def ensure_directories(self) -> Path:
        session = self.require_session()
        for relative in SESSION_DIRECTORIES:
            (session / relative).mkdir(parents=True, exist_ok=True)
        return session

    def _initialize(
        self,
        session_dir: Path,
        camera_name: str,
        *,
        created_at: datetime | None = None,
    ) -> None:
        session_dir.mkdir(parents=True, exist_ok=True)
        for relative in SESSION_DIRECTORIES:
            (session_dir / relative).mkdir(parents=True, exist_ok=True)
        self.session_dir = session_dir.resolve()
        created = created_at or self._now()
        self.manifest = {
            "schema_version": 1,
            "session_id": self.session_dir.name,
            "created_at": created.astimezone().isoformat(timespec="seconds"),
            "application": {
                "name": APPLICATION_NAME,
                "version": APPLICATION_VERSION,
            },
            "camera": {"name": camera_name},
            "pattern": {},
            "calibration": {
                "available_models": [],
                "selected_model": None,
                "recommended_model": None,
            },
            "windshield": {"geometry_models": [], "selected_model": None},
            "paper_metrics_available": False,
            "exports": {},
        }
        self.write_manifest()

    def directory(self, relative: str) -> Path:
        target = self.require_session() / relative
        target.mkdir(parents=True, exist_ok=True)
        return target

    def get_path(self, relative: str, filename: str) -> Path:
        """Generic extension point for exporters without a dedicated helper."""
        return self.directory(relative) / filename

    def project_path(self, filename: str = "calibration.ccproj") -> Path:
        return self.directory("project") / filename

    def intrinsic_path(self, model: Any, *, subset: bool = False) -> Path:
        folder = "intrinsic/subset" if subset else "intrinsic"
        prefix = "camera_subset" if subset else "camera"
        return self.directory(folder) / f"{prefix}_{model_filename(model)}.yaml"

    def ros_path(self) -> Path:
        return self.directory("intrinsic") / "camera_info.yaml"

    def kalibr_path(self, filename: str) -> Path:
        return self.directory("intrinsic/kalibr") / filename

    def report_path(self, filename: str) -> Path:
        return self.directory("reports") / filename

    def paper_directory(self) -> Path:
        return self.directory("paper")

    def validation_directory(self) -> Path:
        return self.directory("validation/subset_comparison")

    def windshield_geometry_path(self, model: Any, variant: str = "") -> Path:
        # Windshield geometry is layered on the session's selected base camera,
        # so the geometry algorithm is the useful discriminator in this folder.
        safe_variant = _safe_component(variant.lower()) or "baseline"
        return self.directory("windshield/geometry") / f"{safe_variant}.yaml"

    def reflection_path(self, filename: str = "reflection_model.yaml") -> Path:
        return self.directory("windshield/reflection") / filename

    def ghost_path(self, filename: str = "ghost_suppression_model.json") -> Path:
        return self.directory("windshield/ghost") / filename

    def update_context(self, **values: Any) -> None:
        """Merge top-level session facts and persist them immediately."""
        for key, value in values.items():
            plain = _plain(value)
            if isinstance(plain, dict) and isinstance(self.manifest.get(key), dict):
                self.manifest[key].update(plain)
            else:
                self.manifest[key] = plain
        self.write_manifest()

    def record_export(
        self,
        key: str,
        path: str | Path | list[str | Path] | tuple[str | Path, ...],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        paths = list(path) if isinstance(path, (list, tuple)) else [path]
        relative_paths = [self._relative(Path(item)) for item in paths]
        entry: dict[str, Any] = {
            "exported_at": self._now().astimezone().isoformat(timespec="seconds"),
        }
        if len(relative_paths) == 1:
            entry["file"] = relative_paths[0]
        else:
            entry["files"] = relative_paths
        if metadata:
            entry["metadata"] = _plain(metadata)
        self.manifest.setdefault("exports", {})[key] = entry
        if key.startswith("windshield.geometry."):
            model = key.removeprefix("windshield.geometry.")
            models = self.manifest.setdefault("windshield", {}).setdefault("geometry_models", [])
            if model not in models:
                models.append(model)
            self.manifest["windshield"]["selected_model"] = model
        if key == "paper.metrics":
            self.manifest["paper_metrics_available"] = True
        self.write_manifest()

    def execute_export(self, key: str, operation: Callable[[], Any] | None) -> dict[str, Any]:
        """Run one isolated export; ``None`` means unavailable/skipped."""
        if operation is None:
            return {"key": key, "status": "skipped", "paths": [], "error": None}
        try:
            result = operation()
            if result is None:
                raise RuntimeError("exporter returned no output path")
            paths = list(result) if isinstance(result, (list, tuple)) else [result]
            self.record_export(key, paths)
            return {
                "key": key,
                "status": "exported",
                "paths": [str(path) for path in paths],
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - deliberate export isolation boundary.
            return {"key": key, "status": "failed", "paths": [], "error": str(exc)}

    def _relative(self, path: Path) -> str:
        absolute = path.expanduser().resolve()
        session = self.require_session()
        try:
            return absolute.relative_to(session).as_posix()
        except ValueError:
            return os.fspath(absolute)

    def write_manifest(self) -> None:
        destination = self.require_session() / "manifest.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

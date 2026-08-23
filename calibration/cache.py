"""Caches for expensive calibration diagnostics and model results."""

from __future__ import annotations

import copy
import hashlib
import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Hashable

import numpy as np

from calibration.types import CalibrationResult, CameraConfig, Dataset, PatternConfig, ValidationResult


def dataset_fingerprint(dataset: Dataset) -> tuple[Hashable, ...]:
    """Stable fingerprint for detected calibration inputs.

    This intentionally includes actual corner/object-point bytes, not just counts,
    so persistent caches cannot be reused for different images with similar shape.
    """
    items = []
    for frame in dataset.enabled_frames:
        det = frame.detection
        digest = hashlib.sha256()
        if det and det.corners is not None:
            arr = np.ascontiguousarray(det.corners)
            digest.update(str(arr.shape).encode("ascii"))
            digest.update(arr.tobytes())
        if det and det.object_points is not None:
            arr = np.ascontiguousarray(det.object_points)
            digest.update(str(arr.shape).encode("ascii"))
            digest.update(arr.tobytes())
        if det and det.excluded_corner_indices:
            digest.update(",".join(map(str, sorted(det.excluded_corner_indices))).encode("ascii"))
        items.append((
            frame.image_info.image_id,
            frame.status.value,
            bool(det and det.success),
            int(det.num_corners if det else 0),
            round(float(det.board_area_ratio or 0.0), 8) if det else 0.0,
            digest.hexdigest(),
        ))
    return tuple(items)


def camera_fingerprint(camera_config: CameraConfig) -> tuple[Any, ...]:
    return (
        camera_config.width,
        camera_config.height,
        camera_config.sensor_name,
        camera_config.model,
    )


def pattern_fingerprint(pattern_config: PatternConfig) -> tuple[Any, ...]:
    return (
        pattern_config.type.value,
        pattern_config.squares_x,
        pattern_config.squares_y,
        pattern_config.square_size,
        pattern_config.marker_size,
        pattern_config.dictionary,
    )


@dataclass
class ValidationCache:
    max_entries: int = 128
    _items: dict[tuple[Any, ...], ValidationResult] = field(default_factory=dict)
    _order: list[tuple[Any, ...]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get(self, key: tuple[Any, ...]) -> ValidationResult | None:
        with self._lock:
            value = self._items.get(key)
            return copy.deepcopy(value) if value is not None else None

    def set(self, key: tuple[Any, ...], value: ValidationResult) -> None:
        with self._lock:
            if key not in self._items:
                self._order.append(key)
            self._items[key] = copy.deepcopy(value)
            while len(self._order) > self.max_entries:
                oldest = self._order.pop(0)
                self._items.pop(oldest, None)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._order.clear()


KFOLD_VALIDATION_CACHE = ValidationCache()


def stable_cache_key(*parts: Any) -> str:
    payload = pickle.dumps(parts, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(payload).hexdigest()


@dataclass
class PersistentResultCache:
    """Small pickle-backed cache for expensive whole-model results.

    The caller supplies a fully qualified key. Values are deep-copied on read and
    write so cache hits cannot leak mutable state between pipeline stages.
    """
    root: str | Path
    namespace: str = "v1"

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / self.namespace / f"{key}.pkl"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with path.open("rb") as f:
                return copy.deepcopy(pickle.load(f))
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(copy.deepcopy(value), f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)


def model_results_cache_key(
    dataset: Dataset,
    camera_config: CameraConfig,
    use_rational_model: bool,
    estimate_fisheye_uncertainty: bool,
    bootstrap_jobs: int,
    models: tuple[str, ...] | None = None,
) -> str:
    return stable_cache_key(
        "run_all_models",
        dataset_fingerprint(dataset),
        camera_fingerprint(camera_config),
        bool(use_rational_model),
        bool(estimate_fisheye_uncertainty),
        int(bootstrap_jobs),
        models,
    )

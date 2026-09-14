"""Automatic image subset selection for calibration inputs.

This module intentionally stays independent from Qt.  It consumes the same
``Dataset`` produced by ``detect_dataset`` and marks frames as selected or
excluded without deleting or modifying source image files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from calibration.image_quality import hamming_distance
from calibration.types import Dataset, Frame, FrameStatus


POSITION_ORDER = [
    "center",
    "left",
    "right",
    "top",
    "bottom",
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
]

DEFAULT_POSITION_WEIGHTS = {
    "center": 5,
    "left": 5,
    "right": 5,
    "top": 4,
    "bottom": 4,
    "top-left": 3,
    "top-right": 3,
    "bottom-left": 3,
    "bottom-right": 3,
}


@dataclass(frozen=True)
class ImageSelectionConfig:
    target_count: int = 35
    blur_min_laplacian_var: float = 15.0
    brightness_min: float = 25.0
    brightness_max: float = 230.0
    saturation_max: float = 0.15
    duplicate_phash_distance: int = 14
    similar_center_distance: float = 0.08
    similar_area_delta: float = 0.04
    similar_tilt_delta_deg: float = 8.0
    seed: int = 42


@dataclass
class ImageSelectionRecord:
    image_id: str
    path: str
    selected: bool = False
    exclusion_reason: str = ""
    score: float = 0.0
    position: str = "unknown"
    distance: str = "unknown"
    tilt: str = "unknown"
    cluster_id: int | None = None
    train_holdout_assignment: str = "excluded"
    detection_success: bool = False
    num_corners: int = 0
    blur: float | None = None
    brightness: float | None = None
    saturation: float | None = None


@dataclass
class ImageSelectionResult:
    selected_paths: list[str]
    selected_image_ids: list[str]
    records: list[ImageSelectionRecord]
    target_count: int
    requested_count: int
    valid_count: int
    counts: dict[str, int] = field(default_factory=dict)
    position_counts: dict[str, int] = field(default_factory=dict)
    distance_counts: dict[str, int] = field(default_factory=dict)
    tilt_counts: dict[str, int] = field(default_factory=dict)
    position_targets: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def selected_count(self) -> int:
        return len(self.selected_paths)

    def summary_text(self) -> str:
        return (
            f"Auto selection: input {self.counts.get('total', 0)}, "
            f"detection failed {self.counts.get('detection_failed', 0)}, "
            f"blur rejected {self.counts.get('blur_rejected', 0)}, "
            f"exposure rejected {self.counts.get('exposure_rejected', 0)}, "
            f"duplicate/scene skipped {self.counts.get('duplicate_scene', 0)}, "
            f"selected {self.selected_count}/{self.requested_count}"
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "requested_count": self.requested_count,
            "target_count": self.target_count,
            "valid_count": self.valid_count,
            "selected_count": self.selected_count,
            "counts": self.counts,
            "position_targets": self.position_targets,
            "position_counts": self.position_counts,
            "distance_counts": self.distance_counts,
            "tilt_counts": self.tilt_counts,
            "warnings": self.warnings,
            "images": [
                {
                    "image_id": r.image_id,
                    "path": r.path,
                    "selected": r.selected,
                    "exclusion_reason": r.exclusion_reason,
                    "score": round(r.score, 6),
                    "position": r.position,
                    "distance": r.distance,
                    "tilt": r.tilt,
                    "cluster_id": r.cluster_id,
                    "train_holdout_assignment": r.train_holdout_assignment,
                    "detection_success": r.detection_success,
                    "num_corners": r.num_corners,
                    "blur": r.blur,
                    "brightness": r.brightness,
                    "saturation": r.saturation,
                }
                for r in self.records
            ],
        }


def allocate_position_targets(total: int) -> dict[str, int]:
    """Allocate the 35-image default profile to any target count."""
    total = max(0, int(total))
    if total == 0:
        return {key: 0 for key in POSITION_ORDER}
    weight_sum = sum(DEFAULT_POSITION_WEIGHTS.values())
    raw = {key: total * DEFAULT_POSITION_WEIGHTS[key] / weight_sum for key in POSITION_ORDER}
    targets = {key: int(np.floor(value)) for key, value in raw.items()}
    remaining = total - sum(targets.values())
    remainders = sorted(
        POSITION_ORDER,
        key=lambda key: (raw[key] - targets[key], DEFAULT_POSITION_WEIGHTS[key]),
        reverse=True,
    )
    for key in remainders[:remaining]:
        targets[key] += 1
    return targets


def write_selection_manifest(result: ImageSelectionResult, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result.to_manifest(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def select_calibration_images(
    dataset: Dataset,
    config: ImageSelectionConfig | None = None,
) -> ImageSelectionResult:
    cfg = config or ImageSelectionConfig()
    requested = max(1, int(cfg.target_count))
    records: dict[str, ImageSelectionRecord] = {}
    candidate_frames: list[Frame] = []

    counts = {
        "total": len(dataset.frames),
        "detection_failed": 0,
        "blur_rejected": 0,
        "exposure_rejected": 0,
        "valid": 0,
        "duplicate_scene": 0,
        "distribution_excluded": 0,
        "selected": 0,
    }

    for frame in dataset.frames:
        record = _base_record(frame)
        records[frame.image_info.image_id] = record

        det = frame.detection
        if det is None or not det.success:
            counts["detection_failed"] += 1
            record.exclusion_reason = "detection_failed"
            continue
        record.detection_success = True
        record.position = classify_position(frame)
        record.distance = classify_distance(frame)
        record.tilt = classify_tilt(frame)

        blur_reason = _blur_rejection_reason(frame, cfg)
        if blur_reason:
            counts["blur_rejected"] += 1
            record.exclusion_reason = blur_reason
            frame.disable(blur_reason, outlier=False)
            continue

        exposure_reason = _exposure_rejection_reason(frame, cfg)
        if exposure_reason:
            counts["exposure_rejected"] += 1
            record.exclusion_reason = exposure_reason
            frame.disable(exposure_reason, outlier=False)
            continue

        candidate_frames.append(frame)

    valid_count = len(candidate_frames)
    counts["valid"] = valid_count
    target = min(requested, valid_count)
    targets = allocate_position_targets(target)
    _score_candidates(candidate_frames, records)
    clusters = _assign_clusters(candidate_frames, records, cfg)
    selected_ids, duplicate_skips = _select_frames(candidate_frames, records, targets, target)

    for frame in candidate_frames:
        image_id = frame.image_info.image_id
        record = records[image_id]
        if image_id in selected_ids:
            record.selected = True
            record.exclusion_reason = ""
            record.train_holdout_assignment = "pending_split"
            frame.status = FrameStatus.DETECTED
            frame.disabled_reason = None
        else:
            reason = "duplicate_scene" if record.cluster_id in duplicate_skips else "distribution_excluded"
            record.exclusion_reason = reason
            record.train_holdout_assignment = "excluded"
            frame.disable(reason, outlier=False)
            counts[reason] += 1

    counts["duplicate_scene"] = max(counts["duplicate_scene"], len(duplicate_skips))
    counts["selected"] = len(selected_ids)

    selected_frames = [frame for frame in dataset.frames if frame.image_info.image_id in selected_ids]
    selected_paths = [frame.image_info.path for frame in selected_frames]
    selected_image_ids = [frame.image_info.image_id for frame in selected_frames]

    position_counts = _count(records[i].position for i in selected_ids)
    distance_counts = _count(records[i].distance for i in selected_ids)
    tilt_counts = _count(records[i].tilt for i in selected_ids)
    warnings = _build_warnings(requested, target, valid_count, targets, position_counts)

    return ImageSelectionResult(
        selected_paths=selected_paths,
        selected_image_ids=selected_image_ids,
        records=[records[f.image_info.image_id] for f in dataset.frames],
        target_count=target,
        requested_count=requested,
        valid_count=valid_count,
        counts=counts,
        position_counts=position_counts,
        distance_counts=distance_counts,
        tilt_counts=tilt_counts,
        position_targets=targets,
        warnings=warnings,
    )


def classify_position(frame: Frame) -> str:
    det = frame.detection
    if not det or not det.board_center_px or frame.image_info.width <= 0 or frame.image_info.height <= 0:
        return "unknown"
    x = det.board_center_px[0] / frame.image_info.width
    y = det.board_center_px[1] / frame.image_info.height
    horizontal = "left" if x < 1 / 3 else ("right" if x > 2 / 3 else "center")
    vertical = "top" if y < 1 / 3 else ("bottom" if y > 2 / 3 else "center")
    if horizontal == "center" and vertical == "center":
        return "center"
    if vertical == "center":
        return horizontal
    if horizontal == "center":
        return vertical
    return f"{vertical}-{horizontal}"


def classify_distance(frame: Frame) -> str:
    area = frame.detection.board_area_ratio if frame.detection else None
    if area is None:
        return "unknown"
    if area >= 0.30:
        return "close"
    if area >= 0.12:
        return "middle"
    return "far"


def classify_tilt(frame: Frame) -> str:
    tilt = frame.detection.board_tilt_deg if frame.detection else None
    if tilt is None:
        return "unknown"
    if tilt <= -12:
        return "left_tilt"
    if tilt >= 12:
        return "right_tilt"
    return "front"


def _base_record(frame: Frame) -> ImageSelectionRecord:
    det = frame.detection
    return ImageSelectionRecord(
        image_id=frame.image_info.image_id,
        path=frame.image_info.path,
        detection_success=bool(det and det.success),
        num_corners=det.num_corners if det else 0,
        blur=frame.image_info.sharpness,
        brightness=frame.image_info.brightness,
        saturation=frame.image_info.saturation,
    )


def _blur_rejection_reason(frame: Frame, cfg: ImageSelectionConfig) -> str:
    sharpness = frame.image_info.sharpness
    if sharpness is not None and sharpness < cfg.blur_min_laplacian_var:
        return "blur_rejected"
    return ""


def _exposure_rejection_reason(frame: Frame, cfg: ImageSelectionConfig) -> str:
    brightness = frame.image_info.brightness
    if brightness is not None and (brightness < cfg.brightness_min or brightness > cfg.brightness_max):
        return "exposure_rejected"
    saturation = frame.image_info.saturation
    if saturation is not None and saturation > cfg.saturation_max:
        return "exposure_rejected"
    return ""


def _score_candidates(frames: list[Frame], records: dict[str, ImageSelectionRecord]) -> None:
    sharpness_values = [f.image_info.sharpness for f in frames if f.image_info.sharpness is not None]
    max_sharpness = max(sharpness_values) if sharpness_values else 1.0
    max_corners = max((f.detection.num_corners for f in frames if f.detection), default=1)
    for frame in frames:
        det = frame.detection
        info = frame.image_info
        corner_score = min(1.0, (det.num_corners if det else 0) / max(1, max_corners))
        confidence = det.corner_confidence if det and det.corner_confidence is not None else corner_score
        sharpness = min(1.0, (info.sharpness or 0.0) / max(1.0, max_sharpness))
        exposure = 0.5 if info.brightness is None else max(0.0, 1.0 - abs(info.brightness - 127.0) / 127.0)
        area = det.board_area_ratio if det and det.board_area_ratio is not None else 0.0
        area_score = 1.0 if 0.10 <= area <= 0.55 else max(0.0, 1.0 - min(abs(area - 0.10), abs(area - 0.55)) / 0.45)
        score = 0.35 * confidence + 0.25 * sharpness + 0.20 * corner_score + 0.10 * exposure + 0.10 * area_score
        records[frame.image_info.image_id].score = float(score)


def _assign_clusters(
    frames: list[Frame],
    records: dict[str, ImageSelectionRecord],
    cfg: ImageSelectionConfig,
) -> dict[int, list[str]]:
    n = len(frames)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if _is_similar_scene(frames[i], frames[j], records, cfg):
                union(i, j)

    roots: dict[int, int] = {}
    clusters: dict[int, list[str]] = {}
    for idx, frame in enumerate(frames):
        root = find(idx)
        cluster_id = roots.setdefault(root, len(roots))
        image_id = frame.image_info.image_id
        records[image_id].cluster_id = cluster_id
        clusters.setdefault(cluster_id, []).append(image_id)
    return clusters


def _is_similar_scene(
    a: Frame,
    b: Frame,
    records: dict[str, ImageSelectionRecord],
    cfg: ImageSelectionConfig,
) -> bool:
    phash_a = a.image_info.phash
    phash_b = b.image_info.phash
    if phash_a and phash_b and hamming_distance(phash_a, phash_b) <= cfg.duplicate_phash_distance:
        return True

    rec_a = records[a.image_info.image_id]
    rec_b = records[b.image_info.image_id]
    if rec_a.position != rec_b.position or rec_a.distance != rec_b.distance or rec_a.tilt != rec_b.tilt:
        return False
    det_a = a.detection
    det_b = b.detection
    if not det_a or not det_b or not det_a.board_center_px or not det_b.board_center_px:
        return False
    center_a = (det_a.board_center_px[0] / a.image_info.width, det_a.board_center_px[1] / a.image_info.height)
    center_b = (det_b.board_center_px[0] / b.image_info.width, det_b.board_center_px[1] / b.image_info.height)
    center_dist = float(np.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1]))
    area_delta = abs((det_a.board_area_ratio or 0.0) - (det_b.board_area_ratio or 0.0))
    tilt_delta = abs((det_a.board_tilt_deg or 0.0) - (det_b.board_tilt_deg or 0.0))
    return (
        center_dist <= cfg.similar_center_distance
        and area_delta <= cfg.similar_area_delta
        and tilt_delta <= cfg.similar_tilt_delta_deg
    )


def _select_frames(
    frames: list[Frame],
    records: dict[str, ImageSelectionRecord],
    targets: dict[str, int],
    target: int,
) -> tuple[set[str], set[int]]:
    by_position: dict[str, list[Frame]] = {key: [] for key in POSITION_ORDER}
    for frame in frames:
        by_position.setdefault(records[frame.image_info.image_id].position, []).append(frame)

    selected: list[str] = []
    used_clusters: set[int] = set()
    selected_distances: set[str] = set()
    selected_tilts: set[str] = set()

    def order(items: list[Frame]) -> list[Frame]:
        return sorted(
            items,
            key=lambda frame: _selection_score(frame, records, selected_distances, selected_tilts),
            reverse=True,
        )

    def choose(items: list[Frame], limit: int, *, allow_used_cluster: bool = False) -> None:
        nonlocal selected
        for frame in order(items):
            if len(selected) >= target or limit <= 0:
                return
            image_id = frame.image_info.image_id
            if image_id in selected:
                continue
            cluster_id = records[image_id].cluster_id
            if not allow_used_cluster and cluster_id in used_clusters:
                continue
            selected.append(image_id)
            if cluster_id is not None:
                used_clusters.add(cluster_id)
            selected_distances.add(records[image_id].distance)
            selected_tilts.add(records[image_id].tilt)
            limit -= 1

    for position in POSITION_ORDER:
        choose(by_position.get(position, []), targets.get(position, 0))

    remaining = [frame for frame in frames if frame.image_info.image_id not in selected]
    choose(remaining, target - len(selected))
    if len(selected) < target:
        choose(remaining, target - len(selected), allow_used_cluster=True)

    selected_set = set(selected)
    duplicate_skips = {
        records[frame.image_info.image_id].cluster_id
        for frame in frames
        if frame.image_info.image_id not in selected_set
        and records[frame.image_info.image_id].cluster_id in used_clusters
    }
    return selected_set, {cluster for cluster in duplicate_skips if cluster is not None}


def _selection_score(
    frame: Frame,
    records: dict[str, ImageSelectionRecord],
    selected_distances: set[str],
    selected_tilts: set[str],
) -> float:
    record = records[frame.image_info.image_id]
    diversity_bonus = 0.0
    if record.distance not in selected_distances:
        diversity_bonus += 0.05
    if record.tilt not in selected_tilts:
        diversity_bonus += 0.05
    return record.score + diversity_bonus


def _count(values) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def _build_warnings(
    requested: int,
    target: int,
    valid_count: int,
    targets: dict[str, int],
    position_counts: dict[str, int],
) -> list[str]:
    warnings: list[str] = []
    if valid_count < requested:
        warnings.append(f"Only {valid_count} valid images were available for requested count {requested}.")
    if target < requested:
        warnings.append(f"Selected {target} images because the valid image pool was smaller than requested.")
    for position, quota in targets.items():
        chosen = position_counts.get(position, 0)
        if chosen < quota:
            warnings.append(f"{position}: selected {chosen}/{quota}; remaining slots were redistributed.")
    return warnings

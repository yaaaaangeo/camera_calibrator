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
    # board_area_ratio 기준 그대로 유지 - distance_m(estimate_rough_pose)는
    # object_points의 물리 단위(보통 미터, pattern.square_size에 좌우됨)를
    # 그대로 쓰므로 보드 물리 크기가 프로젝트마다 다르면 절대 미터 임계값의
    # 의미도 달라진다. area_ratio(이미지 대비 보드가 차지하는 비율)는 보드
    # 물리 크기와 무관하게 "화면에서 얼마나 크게/작게 보이는가"를 이미
    # 스스로 정규화해서 담고 있어 이 목적에는 더 안전한 지표다.
    area = frame.detection.board_area_ratio if frame.detection else None
    if area is None:
        return "unknown"
    if area >= 0.30:
        return "close"
    if area >= 0.12:
        return "middle"
    return "far"


def classify_tilt(frame: Frame) -> str:
    det = frame.detection
    # yaw_deg(estimate_rough_pose) 우선 - 문서화된 부호 규약: yaw<0=left-facing,
    # yaw>0=right-facing (models.common.estimate_rough_pose 참고). 계산 불가
    # 시에만 2D board_tilt_deg(minAreaRect, 실질적으로 roll 근사치) fallback -
    # 이 근사치를 3D facing 판정에 쓰지 않는 것이 이번 수정의 핵심이다.
    if det and det.yaw_deg is not None:
        if det.yaw_deg <= -12:
            return "left_tilt"
        if det.yaw_deg >= 12:
            return "right_tilt"
        return "front"
    tilt = det.board_tilt_deg if det else None
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
    """각 후보 프레임의 0~1 quality 점수를 records에 채운다.

    frame_quality.py(compute_frame_quality_scores)가 이미 이 데이터셋에서
    실행됐으면(ui/worker.py는 select_calibration_images 직전에 항상
    실행한다) frame.quality.overall_score(0~100)를 그대로 재사용한다 -
    "이 프레임이 얼마나 좋은가"를 두 모듈이 각자 다른 가중치로 중복
    계산하던 문제를 없앤다. 아직 계산되지 않은 경로(예: 단위 테스트가
    select_calibration_images만 단독으로 부르는 경우)를 위해 기존 자체
    계산을 fallback으로 남긴다.
    """
    sharpness_values = [f.image_info.sharpness for f in frames if f.image_info.sharpness is not None]
    max_sharpness = max(sharpness_values) if sharpness_values else 1.0
    max_corners = max((f.detection.num_corners for f in frames if f.detection), default=1)
    for frame in frames:
        det = frame.detection
        info = frame.image_info
        if frame.quality is not None:
            score = frame.quality.overall_score / 100.0
        else:
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
    """두 프레임이 사실상 같은 촬영(중복)인지 판정한다.

    이전에는 phash(전체 이미지 해시)가 가까우면 그것만으로(OR) 즉시
    duplicate로 판정했다. 차량에 카메라가 고정된 calibration 환경에서는
    배경이 거의 항상 동일하고 보드만 움직이므로, 이 방식은 pose가 전혀
    다른 두 프레임을 배경이 비슷하다는 이유만으로 중복 처리해 중요한
    pose diversity를 잃을 위험이 있었다(Problem 5). 이제 pose 근접이
    **필수 게이트**이고, hash는 pose가 이미 가까울 때만 참고하는 보조
    신호다 - pose가 다르면 hash가 아무리 비슷해도 duplicate로 보지 않는다.
    """
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
    # yaw_deg(진짜 3D 회전) 우선, 계산 불가 시에만 board_tilt_deg(2D) fallback -
    # classify_tilt()가 rec_a.tilt/rec_b.tilt 버킷을 매길 때 쓰는 신호와 일치시킨다.
    if det_a.yaw_deg is not None and det_b.yaw_deg is not None:
        tilt_delta = abs(det_a.yaw_deg - det_b.yaw_deg)
    else:
        tilt_delta = abs((det_a.board_tilt_deg or 0.0) - (det_b.board_tilt_deg or 0.0))
    pose_close = (
        center_dist <= cfg.similar_center_distance
        and area_delta <= cfg.similar_area_delta
        and tilt_delta <= cfg.similar_tilt_delta_deg
    )
    if not pose_close:
        return False

    phash_a = a.image_info.phash
    phash_b = b.image_info.phash
    if not phash_a or not phash_b:
        # 해시를 계산할 수 없으면 pose 근접만으로 판단한다 - 보조 신호가
        # 없다고 duplicate 판정 자체를 거부하면 진짜 중복(예: 연사)도 못
        # 잡게 된다.
        return True
    return hamming_distance(phash_a, phash_b) <= cfg.duplicate_phash_distance


def _diversity_feature_vector(frame: Frame) -> np.ndarray:
    """Position/distance/pose를 하나의 연속 6D 벡터로 담아, 이미 선택된
    프레임들과의 최소 유클리드 거리로 "얼마나 다른 자세인가"를 재는 데 쓴다
    (아래 _select_frames의 Farthest Point Sampling). scene_quality.py의
    _pose_vector와 같은 철학이지만, 이 모듈의 진입점(select_calibration_images)
    은 camera_config를 받지 않으므로 프레임 자신의 image_info.width/height로
    정규화한다(classify_position이 이미 쓰는 것과 같은 방식).
    """
    det = frame.detection
    info = frame.image_info
    center = det.board_center_px or (info.width / 2, info.height / 2)
    area = det.board_area_ratio if det.board_area_ratio is not None else 0.0
    if det.yaw_deg is not None:
        yaw = min(1.0, abs(det.yaw_deg) / 60.0)
        pitch = min(1.0, abs(det.pitch_deg) / 60.0)
        roll = min(1.0, abs(det.roll_deg) / 60.0)
    else:
        tilt = det.board_tilt_deg if det.board_tilt_deg is not None else 0.0
        yaw = min(1.0, abs(tilt) / 60.0)
        pitch = 0.0
        roll = 0.0
    return np.array([
        center[0] / max(1, info.width),
        center[1] / max(1, info.height),
        min(1.0, area / 0.55),
        yaw, pitch, roll,
    ], dtype=float)


# quality 점수와 pose-space diversity(이미 선택된 것들과의 최소 거리)의 가중치.
# 0.5/0.5로 균형을 맞춘다 - diversity에 너무 치우치면 quality가 나쁜 후보가
# "단지 멀리 있다"는 이유만으로 뽑히고, quality에 너무 치우치면 예전처럼
# 비슷한 좋은 프레임들만 반복 선택되어 pose 다양성을 잃는다.
_DIVERSITY_WEIGHT = 0.5


def _select_frames(
    frames: list[Frame],
    records: dict[str, ImageSelectionRecord],
    targets: dict[str, int],
    target: int,
) -> tuple[set[str], set[int]]:
    """Quality-aware Farthest Point Sampling으로 target장을 채운다.

    이전 방식(Union-Find 클러스터링으로 "cluster당 최대 1장"을 강제)은
    A≈B, B≈C(A,C는 실제로 pose가 많이 다름)인 연쇄에서 transitive closure로
    A/B/C가 전부 하나의 cluster로 묶이면, 그 cluster에서 대표 하나만 뽑혀
    A와 C가 담고 있던 서로 다른 diversity가 함께 사라지는 위험이 있었다
    (Problem 9 - "cluster당 최대 1장"이라는 하드 제약 자체가 transitive
    chaining의 원인이다. `_is_similar_scene`을 아무리 엄격하게 고쳐도(Problem 5
    AND-gate), pose가 연속적으로 이어지는 촬영에서는 여전히 A~B~C 체인이
    생길 수 있다 - 이건 threshold를 얼마나 좁히든 근본적으로 남는 구조적
    문제다).

    그래서 여기서는 cluster 멤버십으로 선택 자체를 막지 않는다. 매 스텝
    "이미 선택된 프레임들과의 최소 pose-space 거리"를 quality와 함께
    반영해서(scene_quality.py::recommend_best_subset과 동일한 패턴) 값이
    가장 큰 후보를 고른다 - 진짜 거의 동일한 프레임(pose 거리≈0)은 이미
    선택된 것과 diversity가 거의 0이 되어 자연스럽게 밀려나지만, A와 C처럼
    실제 pose 거리가 먼 프레임은 같은 cluster에 속해 있어도(체인을 통해서만
    연결) 정상적으로 함께 선택될 수 있다. `cluster_id`(_assign_clusters)는
    이제 선택을 막는 용도가 아니라, 선택되지 않은 프레임에
    "duplicate_scene"이라는 사유를 붙이는 보고용으로만 쓰인다.

    POSITION_ORDER 버킷(이미지 내 위치 커버리지 - pose 다양성과는 별개
    축)은 그대로 유지해 바깥 루프로 쓴다.
    """
    features = {frame.image_info.image_id: _diversity_feature_vector(frame) for frame in frames}
    by_position: dict[str, list[Frame]] = {key: [] for key in POSITION_ORDER}
    for frame in frames:
        by_position.setdefault(records[frame.image_info.image_id].position, []).append(frame)

    selected: list[str] = []

    def fps_pick(pool: list[Frame]) -> Frame | None:
        best_frame: Frame | None = None
        best_value: float | None = None
        for frame in pool:
            image_id = frame.image_info.image_id
            if image_id in selected:
                continue
            if selected:
                distance = min(
                    float(np.linalg.norm(features[image_id] - features[sid])) for sid in selected
                )
                diversity = min(1.0, distance)
            else:
                diversity = 1.0
            value = (1.0 - _DIVERSITY_WEIGHT) * records[image_id].score + _DIVERSITY_WEIGHT * diversity
            if best_value is None or value > best_value:
                best_frame, best_value = frame, value
        return best_frame

    def choose(pool: list[Frame], limit: int) -> None:
        nonlocal selected
        while limit > 0 and len(selected) < target:
            pick = fps_pick(pool)
            if pick is None:
                return
            selected.append(pick.image_info.image_id)
            limit -= 1

    for position in POSITION_ORDER:
        choose(by_position.get(position, []), targets.get(position, 0))

    remaining = [frame for frame in frames if frame.image_info.image_id not in selected]
    choose(remaining, target - len(selected))

    # cluster_id는 선택에 관여하지 않았으므로, 여기서 순수하게 "선택된
    # 프레임과 같은 cluster에 속했지만 선택되지 못한 프레임"을 골라
    # duplicate_scene으로 보고한다(선택 로직과 완전히 분리된 후처리).
    selected_set = set(selected)
    selected_clusters = {
        records[image_id].cluster_id
        for image_id in selected_set
        if records[image_id].cluster_id is not None
    }
    duplicate_skips = {
        records[frame.image_info.image_id].cluster_id
        for frame in frames
        if frame.image_info.image_id not in selected_set
        and records[frame.image_info.image_id].cluster_id in selected_clusters
    }
    return selected_set, duplicate_skips


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

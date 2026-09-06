"""
calibration.windshield.reflection_suppression.dataset
==============================================================

STEP 7 - Real paired dataset loader for training/validation.

STEP 6의 alignment + photometric normalization 파이프라인을 그대로
재사용한다(`calibration.windshield.reflection.alignment`/`metrics`) - pair의
alignment status가 허용 목록에 없으면(기본은 "good"만) 학습 후보에서
제외한다(사용자 스펙 17번, "잘못 정렬된 pair를 Neural에 넣으면 network가
reflection 제거가 아니라 image registration을 학습해버린다").

Scene-level split(사용자 스펙 18번): `SuppressionPair.scene_id`가 같은
pair는 항상 같은 split(train/val/test)에 속해야 leakage가 없다 - 이 모듈의
`scene_level_split()`이 그 규칙을 강제한다(연속 프레임을 프레임 단위로
train/test에 흩뿌리는 것을 방지).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from calibration.windshield.reflection.alignment import align_reference_to_normal
from calibration.windshield.reflection.metrics import normalize_reflection_map, to_luminance
from calibration.windshield.reflection_suppression.config import allowed_alignment_statuses


@dataclass
class SuppressionPair:
    normal_image_path: str
    reference_image_path: str
    pair_id: str = ""
    scene_id: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class PreparedSuppressionPair:
    """정렬 + photometric normalization까지 끝낸, 곧바로 학습에 쓸 수 있는
    (input, target) 쌍. `target_bgr`는 "Aligned + Photometric-normalized
    Reflection-reduced Reference"다(사용자 스펙 16번)."""
    pair_id: str
    scene_id: str
    input_bgr: np.ndarray
    target_bgr: np.ndarray
    alignment_status: str
    alignment_score: Optional[float]


def prepare_pair(
    pair: SuppressionPair,
    *,
    alignment_model: str = "translation",
    allow_warning_alignment: bool = False,
) -> Optional[PreparedSuppressionPair]:
    """이미지를 읽고 정렬/정규화한다. Alignment quality gate를 통과하지
    못하거나 이미지를 읽지 못하면 `None`을 반환한다(호출자가 그 pair를
    조용히 건너뛴다) - 잘못된 pair로 학습하는 것보다 안전하다."""
    normal = cv2.imread(pair.normal_image_path, cv2.IMREAD_COLOR)
    reference = cv2.imread(pair.reference_image_path, cv2.IMREAD_COLOR)
    if normal is None or reference is None:
        return None
    if normal.shape[:2] != reference.shape[:2]:
        return None

    normal_luma = to_luminance(normal)
    reference_luma = to_luminance(reference)
    alignment = align_reference_to_normal(normal_luma, reference_luma, method=alignment_model, enabled=True)
    if alignment.status not in allowed_alignment_statuses(allow_warning_alignment):
        return None

    h, w = normal.shape[:2]
    aligned_reference_bgr = cv2.warpAffine(
        reference, alignment.warp_matrix, (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT,
    )
    aligned_reference_luma = to_luminance(aligned_reference_bgr)
    _abs_map, _pos_map, gain, bias, _norm_luma = normalize_reflection_map(
        normal_luma, aligned_reference_luma, photometric_normalize=True,
    )
    target_bgr = np.clip(gain * aligned_reference_bgr.astype(np.float32) + bias, 0.0, 255.0).astype(np.uint8)

    return PreparedSuppressionPair(
        pair_id=pair.pair_id or pair.normal_image_path,
        scene_id=pair.scene_id or (pair.pair_id or pair.normal_image_path),
        input_bgr=normal,
        target_bgr=target_bgr,
        alignment_status=alignment.status,
        alignment_score=alignment.score,
    )


def scene_level_split(
    pairs: list[SuppressionPair],
    *,
    val_scene_ids: set[str],
    test_scene_ids: set[str],
) -> tuple[list[SuppressionPair], list[SuppressionPair], list[SuppressionPair]]:
    """scene_id 단위로 train/val/test를 나눈다 - 같은 scene_id를 가진 pair가
    train과 test에 동시에 들어가지 않는다는 것을 이 함수 하나가 보장한다."""
    overlap = val_scene_ids & test_scene_ids
    if overlap:
        raise ValueError(f"val/test scene id sets must be disjoint, overlap={overlap}")

    train, val, test = [], [], []
    for pair in pairs:
        if pair.scene_id in test_scene_ids:
            test.append(pair)
        elif pair.scene_id in val_scene_ids:
            val.append(pair)
        else:
            train.append(pair)
    return train, val, test


def load_manifest(path: str) -> tuple[list[SuppressionPair], set[str], set[str]]:
    """실제 paired dataset을 위한 공식 학습 entrypoint(안정화 라운드 항목
    3)가 읽는 YAML manifest. 스키마:

        pairs:
          - pair_id: p001
            scene_id: scene01
            normal: path/to/normal.png       # manifest 파일 기준 상대경로 허용
            reference: path/to/reference.png
        validation_scenes: [scene07]
        test_scenes: [scene08, scene09]

    상대경로는 manifest 파일이 있는 디렉터리를 기준으로 해석한다."""
    manifest_path = Path(path)
    with open(manifest_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    manifest_dir = manifest_path.parent

    def _resolve(p: str) -> str:
        candidate = Path(p)
        return str(candidate if candidate.is_absolute() else manifest_dir / candidate)

    pairs = []
    for entry in data.get("pairs", []) or []:
        pairs.append(SuppressionPair(
            normal_image_path=_resolve(entry["normal"]),
            reference_image_path=_resolve(entry["reference"]),
            pair_id=str(entry.get("pair_id", "")),
            scene_id=str(entry.get("scene_id", "")),
        ))

    val_scene_ids = {str(s) for s in (data.get("validation_scenes") or [])}
    test_scene_ids = {str(s) for s in (data.get("test_scenes") or [])}
    return pairs, val_scene_ids, test_scene_ids

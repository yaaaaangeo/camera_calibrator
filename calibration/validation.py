"""
camera_calibrator.calibration.validation
============================================

설계 문서 3.3번, 8번, 17번 Step8 - Hold-out (교차 검증).

핵심 원칙 (문서에서 반복 강조됨, 반드시 지켜야 함):
    "Test 이미지의 카메라 intrinsic을 다시 최적화하면 안 됨
     - 진짜 validation이 아니게 됨"

그래서 이 모듈의 흐름은:
    1. Train 프레임만으로 cv2.calibrateCamera / cv2.fisheye.calibrate 실행
       -> camera_matrix, distortion 확정
    2. Test 프레임에는 그 camera_matrix/distortion을 "고정"한 채,
       solvePnP로 각 프레임의 포즈(rvec, tvec)만 새로 구함
    3. 그 포즈로 projectPoints -> 재투영 오차 계산

Train/Test 분할은 무작위가 아니라 position(center/left/right/top/bottom/corner)
x distance(near/far) 기준 stratified split을 사용한다 (문서 3.3번 권장).
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from calibration.types import (
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    Dataset,
    Frame,
    ValidationResult,
)
from calibration.models.common import (
    MIN_FRAMES_REQUIRED,
    classify_regions,
    compute_regional_error,
    regional_edge_average,
)
from calibration.models.pinhole import calibrate_pinhole
from calibration.models.extended_pinhole import calibrate_extended_pinhole
from calibration.models.fisheye import calibrate_fisheye


# ---------------------------------------------------------------------------
# Stratified Train/Test Split (설계 문서 3.3번)
# ---------------------------------------------------------------------------

def _position_label(cx: float, cy: float, w: int, h: int) -> str:
    """common.classify_regions는 프레임이 여러 영역에 동시에 속할 수 있게 하지만,
    stratification에는 프레임당 '대표 라벨' 하나가 필요하므로 우선순위를 정해 하나만 고른다.
    우선순위: corner > center > (left/right/top/bottom 단일)
    """
    regions = set(classify_regions(cx, cy, w, h))
    if "corner" in regions:
        return "corner"
    if "center" in regions:
        return "center"
    for r in ("left", "right", "top", "bottom"):
        if r in regions:
            return r
    return "center"  # 이론상 도달하지 않음


def _distance_label(area_ratio: float, median_ratio: float) -> str:
    return "near" if area_ratio >= median_ratio else "far"


def _stratum_key(frame: Frame, image_size: tuple[int, int], median_ratio: float) -> str:
    w, h = image_size
    det = frame.detection
    cx, cy = det.board_center_px if det.board_center_px else (w / 2, h / 2)
    area_ratio = det.board_area_ratio if det.board_area_ratio is not None else median_ratio
    return f"{_position_label(cx, cy, w, h)}_{_distance_label(area_ratio, median_ratio)}"


def split_train_test(
    dataset: Dataset,
    camera_config: CameraConfig,
    test_ratio: float = 0.25,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """position x distance 기준 stratified split.

    각 stratum(예: "center_near") 안에서 test_ratio 비율만큼 test로 빼고
    나머지는 train으로 둔다. stratum 크기가 1이면 test로 빼지 않고 무조건
    train에 남긴다 (test에만 있고 train에는 대표 사례가 전혀 없는 극단은 피함).

    분할 후 train이 MIN_FRAMES_REQUIRED보다 적으면, test에서 프레임을
    다시 train으로 옮겨서라도 최소 요구치를 맞춘다 (validation 자체가
    불가능한 상태를 피하기 위함).
    """
    from calibration.models.common import infer_image_size

    usable = [
        f
        for f in dataset.enabled_frames
        if f.detection and f.detection.success and f.detection.num_corners >= 4
    ]
    if not usable:
        return [], []

    image_size = infer_image_size(dataset, camera_config)
    area_ratios = [
        f.detection.board_area_ratio
        for f in usable
        if f.detection.board_area_ratio is not None
    ]
    median_ratio = float(np.median(area_ratios)) if area_ratios else 0.0

    strata: dict[str, list[Frame]] = {}
    for f in usable:
        key = _stratum_key(f, image_size, median_ratio)
        strata.setdefault(key, []).append(f)

    rng = random.Random(seed)
    train_ids: list[str] = []
    test_ids: list[str] = []

    for key, frames in strata.items():
        frames = frames[:]  # 원본 순서 보존을 위해 복사 후 셔플
        rng.shuffle(frames)
        if len(frames) <= 1:
            train_ids.extend(f.image_info.image_id for f in frames)
            continue
        n_test = round(len(frames) * test_ratio)
        n_test = min(n_test, len(frames) - 1)  # stratum 전체를 test로 빼지 않음
        test_frames, train_frames = frames[:n_test], frames[n_test:]
        train_ids.extend(f.image_info.image_id for f in train_frames)
        test_ids.extend(f.image_info.image_id for f in test_frames)

    # 최소 train 프레임 수 보장: 부족하면 test에서 되가져온다
    if len(train_ids) < MIN_FRAMES_REQUIRED and test_ids:
        need = MIN_FRAMES_REQUIRED - len(train_ids)
        moved, remaining = test_ids[:need], test_ids[need:]
        train_ids.extend(moved)
        test_ids = remaining

    return train_ids, test_ids


def _subset_dataset(dataset: Dataset, frame_ids: list[str]) -> Dataset:
    id_set = set(frame_ids)
    frames = [f for f in dataset.frames if f.image_info.image_id in id_set]
    return Dataset(frames=frames)


# ---------------------------------------------------------------------------
# Test 프레임 재투영 오차 (intrinsic 고정, pose만 재추정)
# ---------------------------------------------------------------------------

def _test_reprojection_errors(
    test_frames: list[Frame],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model: CameraModelType,
) -> tuple[dict[str, float], list[str]]:
    """Train에서 확정된 camera_matrix/distortion을 고정한 채,
    각 test 프레임에 대해 solvePnP로 pose만 새로 구하고 재투영 오차를 계산.

    Returns:
        per_frame_error: {frame_id: rms_px}
        failed_frame_ids: solvePnP/투영이 실패한 프레임 id 목록
        (실패해도 예외를 던지지 않고 목록에 남겨 사용자가 원인 파악 가능하게 함)
    """
    errors: dict[str, float] = {}
    failed: list[str] = []
    is_fisheye = model == CameraModelType.FISHEYE

    for frame in test_frames:
        det = frame.detection
        obj = det.object_points
        img = det.corners
        frame_id = frame.image_info.image_id

        try:
            if is_fisheye:
                obj64 = obj.astype(np.float64)
                img64 = img.astype(np.float64)
                ok, rvec, tvec = cv2.fisheye.solvePnP(obj64, img64, camera_matrix, distortion)
                if not ok:
                    raise cv2.error("fisheye.solvePnP returned False")
                projected, _ = cv2.fisheye.projectPoints(obj64, rvec, tvec, camera_matrix, distortion)
                detected = img64.reshape(-1, 2)
            else:
                ok, rvec, tvec = cv2.solvePnP(obj, img, camera_matrix, distortion)
                if not ok:
                    raise cv2.error("solvePnP returned False")
                projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, distortion)
                detected = img.reshape(-1, 2)
        except cv2.error:
            failed.append(frame_id)
            continue

        projected = projected.reshape(-1, 2)
        diff = detected - projected
        rms = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
        errors[frame_id] = rms
        frame.reprojection_error = rms  # test 프레임에도 기록해 UI/coverage 등에서 재사용 가능

    return errors, failed


# ---------------------------------------------------------------------------
# 모델별 Train 실행 (calibrate_* 함수 매핑)
# ---------------------------------------------------------------------------

def _train_model(
    train_dataset: Dataset,
    camera_config: CameraConfig,
    model: CameraModelType,
    use_rational_model: bool = False,
    fisheye_initial_guess: CalibrationResult | None = None,
) -> CalibrationResult:
    if model == CameraModelType.PINHOLE:
        return calibrate_pinhole(train_dataset, camera_config)
    if model == CameraModelType.EXTENDED_PINHOLE:
        return calibrate_extended_pinhole(
            train_dataset, camera_config, use_rational_model=use_rational_model
        )
    if model == CameraModelType.FISHEYE:
        return calibrate_fisheye(
            train_dataset, camera_config, initial_guess=fisheye_initial_guess
        )
    raise ValueError(f"알 수 없는 모델: {model}")


# ---------------------------------------------------------------------------
# 메인 함수
# ---------------------------------------------------------------------------

def validate_holdout(
    dataset: Dataset,
    camera_config: CameraConfig,
    model: CameraModelType,
    train_ids: list[str],
    test_ids: list[str],
    use_rational_model: bool = False,
    fisheye_initial_guess: CalibrationResult | None = None,
) -> ValidationResult:
    """지정된 train/test 분할로 한 모델에 대해 Hold-out validation 수행.

    train_ids/test_ids를 인자로 받는 이유: 세 모델을 비교할 때 반드시
    "같은 분할"을 써야 공정하다 (validate_all_models가 분할을 한 번만
    수행해서 재사용하는 이유이기도 함).
    """
    if len(train_ids) < MIN_FRAMES_REQUIRED:
        return ValidationResult(
            train_frame_ids=train_ids,
            test_frame_ids=test_ids,
            success=False,
            error_message=f"Train 프레임이 {len(train_ids)}장뿐입니다 (최소 {MIN_FRAMES_REQUIRED}장 필요).",
        )

    train_dataset = _subset_dataset(dataset, train_ids)
    test_dataset = _subset_dataset(dataset, test_ids)

    train_result = _train_model(
        train_dataset,
        camera_config,
        model,
        use_rational_model=use_rational_model,
        fisheye_initial_guess=fisheye_initial_guess,
    )

    if not train_result.success:
        return ValidationResult(
            train_frame_ids=train_ids,
            test_frame_ids=test_ids,
            success=False,
            error_message=f"Train 캘리브레이션 실패: {train_result.error_message}",
        )

    if not test_ids:
        # Test 프레임이 없으면(데이터셋이 너무 작음) train 결과만 반환하고
        # test_rms는 None으로 남겨 "검증 못 함"을 명확히 구분한다.
        return ValidationResult(
            train_frame_ids=train_ids,
            test_frame_ids=[],
            train_rms=train_result.rms_error,
            success=True,
            error_message="Test 프레임이 없어 Hold-out 검증을 수행하지 못했습니다.",
        )

    test_frames = test_dataset.enabled_frames
    per_frame_error, failed_ids = _test_reprojection_errors(
        test_frames, train_result.camera_matrix, train_result.distortion, model
    )

    if not per_frame_error:
        return ValidationResult(
            train_frame_ids=train_ids,
            test_frame_ids=test_ids,
            train_rms=train_result.rms_error,
            success=False,
            error_message="모든 test 프레임에서 pose 추정(solvePnP)이 실패했습니다.",
            failed_test_frame_ids=failed_ids,
        )

    test_rms = float(np.sqrt(np.mean(np.array(list(per_frame_error.values())) ** 2)))
    image_size = camera_config.width, camera_config.height
    regional = compute_regional_error(
        [f for f in test_frames if f.image_info.image_id in per_frame_error],
        per_frame_error,
        image_size,
    )
    edge_rms = regional_edge_average(regional)

    return ValidationResult(
        train_frame_ids=train_ids,
        test_frame_ids=test_ids,
        train_rms=train_result.rms_error,
        test_rms=test_rms,
        edge_rms=edge_rms,
        straightness_residual=None,  # V2
        success=True,
        failed_test_frame_ids=failed_ids,
    )


def validate_all_models(
    dataset: Dataset,
    camera_config: CameraConfig,
    test_ratio: float = 0.25,
    seed: int = 42,
    use_rational_model: bool = False,
) -> dict[CameraModelType, ValidationResult]:
    """세 모델을 '동일한 train/test 분할'로 검증. 분할을 여기서 한 번만 하고
    세 모델 모두에 재사용해야 비교가 공정하다 (모델마다 다른 분할을 쓰면
    Test RMS 차이가 모델 성능 때문인지 분할 운(運) 때문인지 알 수 없게 된다).
    """
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio, seed)

    # Fisheye 발산 방지용 초기값은 "같은 train 분할"로 학습한 Pinhole 결과를 써야
    # 일관성이 있다. 전체 데이터로 학습한 Pinhole을 쓰면 Fisheye가 test 정보를
    # 간접적으로 훔쳐보는 셈이 되어 validation의 취지가 깨진다.
    train_dataset = _subset_dataset(dataset, train_ids)
    pinhole_train_for_init = calibrate_pinhole(train_dataset, camera_config)

    results: dict[CameraModelType, ValidationResult] = {}
    results[CameraModelType.PINHOLE] = validate_holdout(
        dataset, camera_config, CameraModelType.PINHOLE, train_ids, test_ids
    )
    results[CameraModelType.EXTENDED_PINHOLE] = validate_holdout(
        dataset,
        camera_config,
        CameraModelType.EXTENDED_PINHOLE,
        train_ids,
        test_ids,
        use_rational_model=use_rational_model,
    )
    results[CameraModelType.FISHEYE] = validate_holdout(
        dataset,
        camera_config,
        CameraModelType.FISHEYE,
        train_ids,
        test_ids,
        fisheye_initial_guess=pinhole_train_for_init,
    )
    return results


# ---------------------------------------------------------------------------
# 출력용 요약
# ---------------------------------------------------------------------------

def format_validation_table(results: dict[CameraModelType, ValidationResult]) -> str:
    """
                      Pinhole  Extended  Fisheye
        Train RMS        0.35      0.31     0.31
        Test RMS         0.42      0.35     0.39
        Edge RMS(test)   0.51      0.40     0.44
        Gap(Test-Train)  0.07      0.04     0.08
    """
    order = [CameraModelType.PINHOLE, CameraModelType.EXTENDED_PINHOLE, CameraModelType.FISHEYE]
    labels = {"pinhole": "Pinhole", "extended_pinhole": "Extended", "fisheye": "Fisheye"}
    cols = [labels[m.value] for m in order]
    col_w = max(10, max(len(c) for c in cols) + 2)

    def row(name: str, values: list[str]) -> str:
        return f"{name:<16}" + "".join(f"{v:>{col_w}}" for v in values)

    def fmt(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "N/A"

    lines = [row("", cols)]
    lines.append(row("Train RMS", [fmt(results[m].train_rms) if results[m].success else "FAIL" for m in order]))
    lines.append(row("Test RMS", [fmt(results[m].test_rms) for m in order]))
    lines.append(row("Edge RMS(test)", [fmt(results[m].edge_rms) for m in order]))

    gaps = []
    for m in order:
        r = results[m]
        if r.success and r.train_rms is not None and r.test_rms is not None:
            gaps.append(f"{r.test_rms - r.train_rms:.3f}")
        else:
            gaps.append("N/A")
    lines.append(row("Gap(Test-Train)", gaps))

    notes = []
    for m in order:
        r = results[m]
        if not r.success:
            notes.append(f"  [{labels[m.value]}] {r.error_message}")
        elif r.error_message:
            notes.append(f"  [{labels[m.value]}] {r.error_message}")
        if r.failed_test_frame_ids:
            notes.append(f"  [{labels[m.value]}] pose 추정 실패 프레임: {r.failed_test_frame_ids}")
    if notes:
        lines.append("")
        lines.extend(notes)

    return "\n".join(lines)

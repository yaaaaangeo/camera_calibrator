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
    OutlierResult,
    PatternConfig,
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
from calibration.residual_stats import compute_residual_stats
from calibration.straightness import compute_straightness_residual, compute_straightness_breakdown


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
) -> tuple[dict[str, float], list[str], list[float]]:
    """Train에서 확정된 camera_matrix/distortion을 고정한 채,
    각 test 프레임에 대해 solvePnP로 pose만 새로 구하고 재투영 오차를 계산.

    Returns:
        per_frame_error: {frame_id: rms_px} (프레임 단위 RMS)
        failed_frame_ids: solvePnP/투영이 실패한 프레임 id 목록
        (실패해도 예외를 던지지 않고 목록에 남겨 사용자가 원인 파악 가능하게 함)
        point_errors: 모든 코너 포인트 각각의 재투영 오차(Euclidean distance) -
            설계 문서 10/11번 - Test 쪽에도 MAE/Median/P90/P95/P99 등을 계산하려면
            프레임 단위 RMS만으로는 부족하고 코너 포인트 단위 원본 오차가 필요하다.
    """
    errors: dict[str, float] = {}
    failed: list[str] = []
    point_errors: list[float] = []
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
        per_point = np.hypot(diff[:, 0], diff[:, 1])
        rms = float(np.sqrt(np.mean(per_point ** 2)))
        errors[frame_id] = rms
        point_errors.extend(per_point.tolist())
        frame.reprojection_error = rms  # test 프레임에도 기록해 UI/coverage 등에서 재사용 가능

    return errors, failed, point_errors


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


def refit_on_train_split(
    dataset: Dataset,
    camera_config: CameraConfig,
    model: CameraModelType,
    train_ids: list[str],
    use_rational_model: bool = False,
    fisheye_initial_guess: CalibrationResult | None = None,
) -> CalibrationResult:
    """validate_holdout()이 내부적으로 만드는 것과 완전히 동일한
    "train 프레임만으로 학습한" CalibrationResult를 다시 만들어 돌려준다.

    쓰임새: 외부(다른 사람/다른 세션) 캘리브레이션 결과와 공정하게 비교하려면
    (calibration/external_compare.py), "내 파라미터" 쪽도 정확히 같은
    test 프레임에서 한 번도 안 본 상태로 평가해야 한다. ValidationResult는
    집계된 숫자(test_rms 등)만 들고 있고 그때 쓰인 camera_matrix/distortion
    자체는 버려지므로, 똑같은 절차를 다시 돌려 그 파라미터를 복원한다
    (계산 로직 자체는 새로 만들지 않고 _train_model을 그대로 재사용 -
    계산 로직 중복 금지 원칙).
    """
    train_dataset = _subset_dataset(dataset, train_ids)
    return _train_model(
        train_dataset, camera_config, model,
        use_rational_model=use_rational_model,
        fisheye_initial_guess=fisheye_initial_guess,
    )


# ---------------------------------------------------------------------------
# 메인 함수
# ---------------------------------------------------------------------------

def _evaluate_on_test(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    model: CameraModelType,
    train_ids: list[str],
    test_ids: list[str],
    train_result: CalibrationResult,
) -> ValidationResult:
    """이미 확정된(train_result) camera_matrix/distortion으로 test_ids만
    평가한다. 이 함수는 절대로 train_result를 다시 계산하거나 test 오차를
    바탕으로 무언가를 바꾸지 않는다 - 오직 "평가"만 한다(문서 9번 핵심 원칙:
    "Test: 절대 수정하지 않음 -> 최종 평가만 수행"). validate_holdout()과
    recalibrate_train_with_outlier_pruning() 둘 다 이 함수를 재사용해서,
    "test 평가"의 정의가 코드 두 곳에서 갈라지는 사고를 막는다.
    """
    if not test_ids:
        train_dataset = _subset_dataset(dataset, train_ids)
        straightness, _ = compute_straightness_residual(
            train_dataset.enabled_frames, pattern_config,
            train_result.camera_matrix, train_result.distortion, model,
        )
        straightness_breakdown = compute_straightness_breakdown(
            train_dataset.enabled_frames, pattern_config,
            train_result.camera_matrix, train_result.distortion, model,
        )
        return ValidationResult(
            train_frame_ids=train_ids,
            test_frame_ids=[],
            train_rms=train_result.rms_error,
            train_residual_stats=train_result.residual_stats,
            straightness_residual=straightness,
            straightness_breakdown=straightness_breakdown,
            success=True,
            error_message="Test 프레임이 없어 Hold-out 검증을 수행하지 못했습니다.",
        )

    test_dataset = _subset_dataset(dataset, test_ids)
    test_frames = test_dataset.enabled_frames
    per_frame_error, failed_ids, point_errors = _test_reprojection_errors(
        test_frames, train_result.camera_matrix, train_result.distortion, model
    )

    if not per_frame_error:
        return ValidationResult(
            train_frame_ids=train_ids,
            test_frame_ids=test_ids,
            train_rms=train_result.rms_error,
            train_residual_stats=train_result.residual_stats,
            success=False,
            error_message="모든 test 프레임에서 pose 추정(solvePnP)이 실패했습니다.",
            failed_test_frame_ids=failed_ids,
        )

    test_rms = float(np.sqrt(np.mean(np.array(list(per_frame_error.values())) ** 2)))
    test_residual_stats = compute_residual_stats(point_errors)
    image_size = camera_config.width, camera_config.height
    regional = compute_regional_error(
        [f for f in test_frames if f.image_info.image_id in per_frame_error],
        per_frame_error,
        image_size,
    )
    edge_rms = regional_edge_average(regional)

    # 설계 문서 3.4번 - Line Straightness. Test 프레임(학습에 쓰이지 않은
    # 이미지)에서 측정하는 게 원칙적으로 더 정직하지만, ChArUco 코너 개수가
    # 적은 소규모 데이터셋에서는 test 프레임만으로 직선(4점 이상)이 하나도
    # 안 나올 수 있다 - 그럴 땐 train 프레임으로 대체해서라도 값을 낸다
    # ("측정 안 함"보다 "약간 낙관적인 값"이 recommender.py 입장에서 더 유용).
    straightness, n_lines = compute_straightness_residual(
        test_frames, pattern_config, train_result.camera_matrix, train_result.distortion, model
    )
    straightness_source_frames = test_frames
    if straightness is None:
        straightness_source_frames = _subset_dataset(dataset, train_ids).enabled_frames
        straightness, n_lines = compute_straightness_residual(
            straightness_source_frames, pattern_config,
            train_result.camera_matrix, train_result.distortion, model,
        )
    straightness_breakdown = compute_straightness_breakdown(
        straightness_source_frames, pattern_config,
        train_result.camera_matrix, train_result.distortion, model,
    )

    return ValidationResult(
        train_frame_ids=train_ids,
        test_frame_ids=test_ids,
        train_rms=train_result.rms_error,
        test_rms=test_rms,
        edge_rms=edge_rms,
        straightness_residual=straightness,
        straightness_breakdown=straightness_breakdown,
        train_residual_stats=train_result.residual_stats,
        test_residual_stats=test_residual_stats,
        success=True,
        failed_test_frame_ids=failed_ids,
    )


def validate_holdout(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
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

    return _evaluate_on_test(
        dataset, camera_config, pattern_config, model, train_ids, test_ids, train_result
    )


def validate_all_models(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
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
        dataset, camera_config, pattern_config, CameraModelType.PINHOLE, train_ids, test_ids
    )
    results[CameraModelType.EXTENDED_PINHOLE] = validate_holdout(
        dataset,
        camera_config,
        pattern_config,
        CameraModelType.EXTENDED_PINHOLE,
        train_ids,
        test_ids,
        use_rational_model=use_rational_model,
    )
    results[CameraModelType.FISHEYE] = validate_holdout(
        dataset,
        camera_config,
        pattern_config,
        CameraModelType.FISHEYE,
        train_ids,
        test_ids,
        fisheye_initial_guess=pinhole_train_for_init,
    )
    return results


# ---------------------------------------------------------------------------
# 설계 문서 9번 - Train/Test Leakage 완전 제거
# ---------------------------------------------------------------------------
#
#   전체 데이터 -> Split -> Train -> Outlier Detection -> Calibration -> ...
#                        -> Test -> (절대 수정 안 함) -> 최종 평가만
#
# app/cli.py의 기존 --outlier 흐름(outlier.recalibrate_with_outlier_pruning을
# "분할 전" 전체 데이터셋에 적용한 뒤 validate_all_models를 다시 호출해
# train/test를 "다시" 나누는 방식)은 미묘한 leakage 위험이 있다: 이상치 판단이
# 그 시점엔 아직 존재하지도 않는 test_ids 소속 프레임들의 오차까지 함께 본
# "전체 데이터 fit" 결과를 근거로 이뤄지기 때문에, 나중에 test로 뽑힐 프레임이
# 이상치 판정에 간접적으로 영향을 준다. 아래 함수는 문서 9번이 요구하는 순서를
# 문자 그대로 지킨다: 먼저 분할하고, Train 프레임만으로 이상치를 찾고 제거하며
# 반복 재계산하고, Test는 마지막에 딱 한 번 "고정된 파라미터"로만 평가한다.

def recalibrate_train_with_outlier_pruning(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    model: CameraModelType,
    train_ids: list[str],
    test_ids: list[str],
    max_iterations: int = 3,
    k: float = 3.0,
    mad_scale: float = 1.0,
    user_threshold: float | None = None,
    use_rational_model: bool = False,
    fisheye_initial_guess: CalibrationResult | None = None,
) -> tuple[CalibrationResult, OutlierResult, ValidationResult]:
    """Split -> Train-only Outlier Detection -> Calibration -> Test 평가,
    순서를 강제하는 leakage-safe 버전.

    outlier.recalibrate_with_outlier_pruning()을 "train 프레임만 들어있는
    Dataset 부분집합"에 대해 그대로 재사용한다 - 그 함수는 자신에게 주어진
    Dataset 안에서만 재투영 오차를 계산하고 이상치를 뽑으므로, train만 담긴
    부분집합을 넘기면 이상치 탐지/제거 로직을 중복 구현하지 않고도 자동으로
    "train에서만" 이뤄진다. Test는 완전히 별개로, 최종적으로 확정된
    camera_matrix/distortion에 대해 _evaluate_on_test()로 딱 한 번만 평가한다
    (validate_holdout()과 동일한 평가 함수를 재사용 - "test 평가"의 정의가
    두 곳에서 갈라지지 않도록).

    주의: dataset은 in-place로 바뀐다(제거된 train 프레임이 DISABLED_OUTLIER로
    표시됨) - outlier.apply_outlier_removal()과 동일한 부수효과 계약을 따른다.
    test_ids에 해당하는 프레임은 이 함수 안에서 절대 status가 바뀌지 않는다.
    """
    # outlier.py는 이 함수와 반대 방향으로 model 함수들을 import하므로, 순환
    # 참조를 피하기 위해 함수 안에서 지연 import한다 (fisheye.py의
    # _fisheye_flag 지연 평가와 같은 이유의 패턴).
    from calibration.outlier import recalibrate_with_outlier_pruning

    if len(train_ids) < MIN_FRAMES_REQUIRED:
        empty_outlier = OutlierResult(
            threshold_used=0.0, removed_frame_ids=[], rms_before=None, rms_after=None,
            iterations=0, max_iterations=max_iterations,
        )
        failed = CalibrationResult(
            model_name=model, success=False,
            error_message=f"Train 프레임이 {len(train_ids)}장뿐입니다 (최소 {MIN_FRAMES_REQUIRED}장 필요).",
        )
        val = ValidationResult(
            train_frame_ids=train_ids, test_frame_ids=test_ids, success=False,
            error_message=failed.error_message,
        )
        return failed, empty_outlier, val

    train_dataset = _subset_dataset(dataset, train_ids)

    # 여기서 이상치 탐지/제거/반복 재계산이 전부 train_dataset "안에서만"
    # 이뤄진다 - test_ids 프레임은 train_dataset에 아예 포함되지 않으므로
    # 재투영 오차 계산에도, threshold 산정에도, 제거 후보에도 절대 등장하지 않는다.
    train_result, outlier_result = recalibrate_with_outlier_pruning(
        train_dataset, camera_config, model,
        max_iterations=max_iterations, k=k, mad_scale=mad_scale,
        user_threshold=user_threshold, use_rational_model=use_rational_model,
        fisheye_initial_guess=fisheye_initial_guess,
    )

    if not train_result.success:
        val = ValidationResult(
            train_frame_ids=train_ids, test_frame_ids=test_ids, success=False,
            error_message=f"Train 캘리브레이션 실패: {train_result.error_message}",
        )
        return train_result, outlier_result, val

    validation_result = _evaluate_on_test(
        dataset, camera_config, pattern_config, model, train_ids, test_ids, train_result
    )
    return train_result, outlier_result, validation_result


def recalibrate_train_with_corner_outlier_pruning(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    model: CameraModelType,
    train_ids: list[str],
    test_ids: list[str],
    max_iterations: int = 3,
    k: float = 3.0,
    mad_scale: float = 1.0,
    use_rational_model: bool = False,
    fisheye_initial_guess: CalibrationResult | None = None,
) -> tuple[CalibrationResult, "CornerOutlierResult", ValidationResult]:
    """recalibrate_train_with_outlier_pruning()의 corner-level 버전 - 완전히
    같은 구조(Split -> Train-only 이상치 탐지 -> Calibration -> Test 평가만)를
    프레임 단위 대신 코너 단위 이상치 제거에 적용한다.

    outlier.recalibrate_with_corner_outlier_pruning()을 train 부분집합에만
    적용하므로, 이상치 판단에 쓰이는 재투영 오차 계산도 train 프레임의 코너
    로만 이뤄진다 - test 프레임의 코너는 이 함수 안에서 전혀 등장하지 않는다.
    """
    from calibration.outlier import recalibrate_with_corner_outlier_pruning
    from calibration.types import CornerOutlierResult

    if len(train_ids) < MIN_FRAMES_REQUIRED:
        empty_outlier = CornerOutlierResult(threshold_used=0.0, iterations=0, max_iterations=max_iterations)
        failed = CalibrationResult(
            model_name=model, success=False,
            error_message=f"Train 프레임이 {len(train_ids)}장뿐입니다 (최소 {MIN_FRAMES_REQUIRED}장 필요).",
        )
        val = ValidationResult(
            train_frame_ids=train_ids, test_frame_ids=test_ids, success=False,
            error_message=failed.error_message,
        )
        return failed, empty_outlier, val

    train_dataset = _subset_dataset(dataset, train_ids)

    train_result, corner_outlier_result = recalibrate_with_corner_outlier_pruning(
        train_dataset, camera_config, model,
        max_iterations=max_iterations, k=k, mad_scale=mad_scale,
        use_rational_model=use_rational_model, fisheye_initial_guess=fisheye_initial_guess,
    )

    if not train_result.success:
        val = ValidationResult(
            train_frame_ids=train_ids, test_frame_ids=test_ids, success=False,
            error_message=f"Train 캘리브레이션 실패: {train_result.error_message}",
        )
        return train_result, corner_outlier_result, val

    validation_result = _evaluate_on_test(
        dataset, camera_config, pattern_config, model, train_ids, test_ids, train_result
    )
    return train_result, corner_outlier_result, validation_result


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
    lines.append(row("Straightness", [fmt(results[m].straightness_residual) for m in order]))

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


def format_train_test_residual_comparison(results: dict[CameraModelType, ValidationResult]) -> str:
    """설계 문서 10번 - Hold-out Validation 강화. Train RMS/Test RMS 하나씩만
    보여주던 format_validation_table()과 달리, Train과 Test 양쪽의
    MAE/Median/P90/P95/P99/Max를 나란히 보여준다 (코너 포인트 단위 -
    residual_stats.py, ValidationResult.train_residual_stats/test_residual_stats).

                          Pinhole             Extended            Fisheye
                       Train    Test       Train    Test       Train    Test
    MAE               0.254   0.301      0.220   0.267      0.220   0.268
    Median            0.219   0.255      0.180   0.221      0.180   0.219
    P90               0.510   0.601      0.457   0.540      0.444   0.535
    P95               0.605   0.720      0.547   0.640      0.542   0.638
    P99               0.770   0.890      0.648   0.760      0.652   0.755
    Max               0.923   1.050      0.898   0.990      0.901   0.985
    """
    order = [CameraModelType.PINHOLE, CameraModelType.EXTENDED_PINHOLE, CameraModelType.FISHEYE]
    labels = {"pinhole": "Pinhole", "extended_pinhole": "Extended", "fisheye": "Fisheye"}

    sub_w = 9
    header1 = " " * 16 + "".join(f"{labels[m.value]:^{sub_w*2}}" for m in order)
    header2 = " " * 16 + "".join(f"{'Train':>{sub_w}}{'Test':>{sub_w}}" for m in order)

    def fmt(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "N/A"

    def metric_row(name: str, attr: str) -> str:
        cells = []
        for m in order:
            r = results[m]
            train_stats = r.train_residual_stats
            test_stats = r.test_residual_stats
            train_v = getattr(train_stats, attr) if train_stats else None
            test_v = getattr(test_stats, attr) if test_stats else None
            cells.append(f"{fmt(train_v):>{sub_w}}{fmt(test_v):>{sub_w}}")
        return f"{name:<16}" + "".join(cells)

    lines = [
        header1,
        header2,
        metric_row("MAE", "mae"),
        metric_row("Median", "median"),
        metric_row("P90", "p90"),
        metric_row("P95", "p95"),
        metric_row("P99", "p99"),
        metric_row("Max", "max"),
    ]
    return "\n".join(lines)


def format_straightness_comparison(results: dict[CameraModelType, ValidationResult]) -> str:
    """설계 문서 15번 - "Line Straightness Error를 모델 비교에 넣는다".
    format_validation_table()의 단일 Straightness 행 대신, 방향별(수평/수직/
    대각선)·위치별(중앙/가장자리/코너) 잔차를 모델 3개 나란히 보여준다.

                    Pinhole  Extended  Fisheye
    Horizontal        0.245     0.185    0.190
    Vertical          0.251     0.190    0.195
    Diagonal          0.298     0.221    0.230
    Center line       0.201     0.160    0.165
    Edge line         0.280     0.210    0.215
    Corner line       0.298     0.221    0.230
    Overall           0.256     0.196    0.201
    """
    order = [CameraModelType.PINHOLE, CameraModelType.EXTENDED_PINHOLE, CameraModelType.FISHEYE]
    labels = {"pinhole": "Pinhole", "extended_pinhole": "Extended", "fisheye": "Fisheye"}
    col_w = 10

    def fmt(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "N/A"

    header = f"{'':<14}" + "".join(f"{labels[m.value]:>{col_w}}" for m in order)

    def metric_row(name: str, attr: str) -> str:
        cells = []
        for m in order:
            breakdown = results[m].straightness_breakdown
            v = getattr(breakdown, attr) if breakdown else None
            cells.append(f"{fmt(v):>{col_w}}")
        return f"{name:<14}" + "".join(cells)

    lines = [
        "Line Straightness Comparison",
        header,
        metric_row("Horizontal", "horizontal_error"),
        metric_row("Vertical", "vertical_error"),
        metric_row("Diagonal", "diagonal_error"),
        metric_row("Center line", "center_line_error"),
        metric_row("Edge line", "edge_line_error"),
        metric_row("Corner line", "corner_line_error"),
        metric_row("Overall", "overall_error"),
    ]
    return "\n".join(lines)

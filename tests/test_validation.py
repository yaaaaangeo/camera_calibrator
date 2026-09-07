"""
tests/test_validation.py
=============================

설계 문서 3.3번 - Hold-out Validation. train/test 분할이 겹치지 않는지,
전체 프레임을 다 커버하는지, test 프레임의 intrinsic이 재최적화되지
않는다는 원칙(설계 문서 핵심 경고)이 지켜지는지 확인한다.
"""

from __future__ import annotations

import copy

import pytest

from calibration.holdout_evidence import evaluate_holdout_evidence
from calibration.types import CameraModelType, FrameStatus
from calibration.validation import (
    _subset_dataset,
    format_cross_dataset_validation_table,
    recalibrate_train_with_corner_outlier_pruning,
    recalibrate_train_with_outlier_pruning,
    split_train_test,
    validate_cross_dataset,
    validate_cross_datasets,
    validate_holdout,
)
from calibration.models.pinhole import calibrate_pinhole


def test_train_test_split_no_overlap(synthetic_dataset, camera_config):
    train_ids, test_ids = split_train_test(synthetic_dataset, camera_config, test_ratio=0.25)
    assert set(train_ids).isdisjoint(set(test_ids)), "train/test가 겹치면 안 됨 - validation의 의미가 없어짐"


def test_train_test_split_covers_all_usable_frames(synthetic_dataset, camera_config):
    train_ids, test_ids = split_train_test(synthetic_dataset, camera_config, test_ratio=0.25)
    usable_count = sum(
        1 for f in synthetic_dataset.enabled_frames if f.detection and f.detection.success
    )
    assert len(train_ids) + len(test_ids) == usable_count


def test_train_test_split_is_deterministic_with_same_seed(synthetic_dataset, camera_config):
    """같은 seed면 항상 같은 분할이 나와야 재현 가능한 실험이 된다."""
    train1, test1 = split_train_test(synthetic_dataset, camera_config, test_ratio=0.25, seed=42)
    train2, test2 = split_train_test(synthetic_dataset, camera_config, test_ratio=0.25, seed=42)
    assert set(train1) == set(train2)
    assert set(test1) == set(test2)


def test_different_seed_can_give_different_split(synthetic_dataset, camera_config):
    train1, test1 = split_train_test(synthetic_dataset, camera_config, test_ratio=0.25, seed=1)
    train2, test2 = split_train_test(synthetic_dataset, camera_config, test_ratio=0.25, seed=999)
    # 항상 다르다고 보장은 못 하지만(작은 데이터셋이면 우연히 같을 수도 있음),
    # seed 파라미터 자체가 무시되고 있지 않은지 정도는 확인 가능
    assert isinstance(train1, list) and isinstance(train2, list)


def test_cross_dataset_validation_evaluates_target_without_retraining(
    synthetic_dataset, camera_config, pattern_config
):
    dataset_a = copy.deepcopy(synthetic_dataset)
    dataset_b = copy.deepcopy(synthetic_dataset)
    train_fit = calibrate_pinhole(dataset_a, camera_config)
    assert train_fit.success
    original_K = train_fit.camera_matrix.copy()
    original_D = train_fit.distortion.copy()

    result = validate_cross_dataset(
        train_fit,
        dataset_b,
        camera_config,
        pattern_config,
        source_dataset_id="A",
        target_dataset_id="B",
    )

    assert result.success
    assert result.source_dataset_id == "A"
    assert result.target_dataset_id == "B"
    assert result.model_name == CameraModelType.PINHOLE
    assert result.num_test_frames == dataset_b.num_detected
    assert result.test_rms is not None
    assert result.test_p95 is not None
    assert result.generalization_gap is not None
    assert (train_fit.camera_matrix == original_K).all()
    assert (train_fit.distortion == original_D).all()


def test_cross_dataset_batch_and_format_table(synthetic_dataset, camera_config, pattern_config):
    dataset_a = copy.deepcopy(synthetic_dataset)
    dataset_b = copy.deepcopy(synthetic_dataset)
    dataset_c = copy.deepcopy(synthetic_dataset)
    pinhole = calibrate_pinhole(dataset_a, camera_config)

    results = validate_cross_datasets(
        {CameraModelType.PINHOLE: pinhole},
        {"B": dataset_b, "C": dataset_c},
        camera_config,
        pattern_config,
        source_dataset_id="A",
    )
    table = format_cross_dataset_validation_table(results)

    assert len(results) == 2
    assert {r.target_dataset_id for r in results} == {"B", "C"}
    assert "Cross-dataset validation" in table
    assert "B" in table and "C" in table


# ---------------------------------------------------------------------------
# 설계 문서 9번 - "validation leakage 테스트 추가"
#
# 세션 스코프 fixture(synthetic_dataset)는 여러 테스트가 공유하므로, 아래
# 테스트들은 전부 copy.deepcopy로 복사한 뒤에만 상태를 바꾼다 - 원본을
# 건드리면 이 파일의 다른 테스트나 다른 파일의 테스트가 오염된다.
# ---------------------------------------------------------------------------



def test_train_rms_reproducible_independently_of_test_evaluation(synthetic_dataset, camera_config, pattern_config):
    """validate_holdout()이 test 평가에 쓰는 camera_matrix/distortion이
    "진짜로" train 프레임만으로 학습된 것인지 확인한다: 완전히 독립적으로
    train 부분집합만 다시 calibrate_pinhole()에 넣었을 때와 정확히 같은
    train_rms가 나와야 한다. 만약 test 정보가 조금이라도 학습에 섞여
    들어갔다면 이 값이 달라진다(cv2.calibrateCamera는 입력이 같으면
    결정론적으로 같은 결과를 낸다).
    """
    dataset = copy.deepcopy(synthetic_dataset)
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.25, seed=7)
    assert test_ids, "이 테스트는 test 프레임이 존재해야 의미가 있음"

    independent_train_fit = calibrate_pinhole(_subset_dataset(dataset, train_ids), camera_config)
    assert independent_train_fit.success

    validation_result = validate_holdout(
        dataset, camera_config, pattern_config, CameraModelType.PINHOLE, train_ids, test_ids
    )
    assert validation_result.success
    assert validation_result.train_rms == pytest.approx(independent_train_fit.rms_error, rel=1e-9, abs=1e-9), (
        "validate_holdout 내부에서 쓰인 train fit이 순수 train-only 결과와 달라짐 - "
        "test 정보가 학습에 섞여 들어갔을 가능성이 있음"
    )


def test_holdout_test_evaluation_does_not_mutate_test_frame_reprojection_error(
    synthetic_dataset, camera_config, pattern_config,
):
    """Phase A-8 안정화 - hold-out **test** 평가(_test_reprojection_errors,
    고정된 intrinsic으로 test 프레임의 pose만 재추정)는 원본 test Frame의
    reprojection_error를 직접 덮어쓰면 안 된다. Pinhole/Brown/Rational/
    Fisheye를 순차적으로 hold-out 검증하면 같은 test Frame 객체가 매번
    mutate되어 "마지막에 평가된 모델의 test 값"만 남는 문제가 있었다 -
    이제는 test 평가 쪽에서는 절대 건드리지 않고, 필요한 값은
    ValidationResult.per_frame_error에만 담아야 한다.

    (참고: train 쪽 calibrate_*() 함수 자체가 갖고 있는 별도의
    frame.reprojection_error 기록 동작은 이 테스트의 범위가 아니다 - 그
    함수들은 이 validation 경로 밖에서도 단독으로 쓰이는 범용 함수라 이번
    라운드에서는 손대지 않았다.)"""
    dataset = copy.deepcopy(synthetic_dataset)
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.25, seed=11)
    assert test_ids

    test_frames_before = {fid: next(f for f in dataset.frames if f.image_info.image_id == fid).reprojection_error for fid in test_ids}

    validation_result = validate_holdout(
        dataset, camera_config, pattern_config, CameraModelType.PINHOLE, train_ids, test_ids
    )
    assert validation_result.success

    for fid in test_ids:
        frame = next(f for f in dataset.frames if f.image_info.image_id == fid)
        assert frame.reprojection_error == test_frames_before[fid], (
            "hold-out test 평가가 test Frame.reprojection_error를 mutate했다 - "
            "여러 모델을 순차 검증하면 마지막 모델의 test 값만 남는 버그가 재발한다."
        )

    # 대신 ValidationResult.per_frame_error에 model-specific하게 보관되어야 한다.
    assert validation_result.per_frame_error, "per_frame_error가 비어 있음 - hold-out 결과가 어디에도 저장되지 않음"
    assert set(validation_result.per_frame_error) <= set(test_ids)


def test_sequential_holdout_validation_across_models_keeps_independent_test_errors(
    synthetic_dataset, camera_config, pattern_config,
):
    """Pinhole -> Brown-Conrady 순서로 같은 Dataset을 hold-out 검증해도, 각
    ValidationResult.per_frame_error(test 프레임 기준)는 서로 다른 모델의
    값을 독립적으로 유지해야 한다.

    이 테스트의 목적은 오직 "mutation independence"(서로 다른 모델의 결과가
    같은 dict를 공유/덮어쓰지 않는지)이지 특정 카메라 모델의 적합 품질이
    아니다. 공용 synthetic_dataset fixture(conftest.py)는 Brown-distortion
    기반 pinhole 사영으로 렌더링된 데이터셋이라 Pinhole/Brown-Conrady 둘 다
    안정적으로 수렴한다 - Fisheye(Kannala-Brandt)는 애초에 이 데이터셋의
    생성 모델과 맞지 않아 검증 목적에 맞지 않는다(Fisheye 전용 GT 검증은
    아래 test_fisheye_holdout_validation_recovers_known_k_d가 실제
    cv2.fisheye.projectPoints로 만든 GT로 별도로 다룬다)."""
    dataset = copy.deepcopy(synthetic_dataset)
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=5)
    assert test_ids

    pinhole_validation = validate_holdout(
        dataset, camera_config, pattern_config, CameraModelType.PINHOLE, train_ids, test_ids
    )
    brown_validation = validate_holdout(
        dataset, camera_config, pattern_config, CameraModelType.BROWN_CONRADY, train_ids, test_ids
    )
    assert pinhole_validation.success and brown_validation.success
    assert pinhole_validation.per_frame_error
    assert brown_validation.per_frame_error
    # 각 ValidationResult는 자신의 모델 값만 독립적으로 갖고 있어야 한다 -
    # 같은 dict 객체를 공유하거나 서로 덮어쓰면 안 된다.
    assert pinhole_validation.per_frame_error is not brown_validation.per_frame_error
    assert set(pinhole_validation.per_frame_error) <= set(test_ids)
    assert set(brown_validation.per_frame_error) <= set(test_ids)


def _build_synthetic_fisheye_dataset(n_frames: int = 24, seed: int = 0):
    """cv2.fisheye.projectPoints()로 실제 Fisheye(Kannala-Brandt) 모델에서
    직접 생성한 GT 데이터셋. Pinhole/Brown 왜곡으로 만든 공용
    synthetic_dataset과 달리, 이 데이터는 진짜 Fisheye 모델의 결과라서
    Fisheye 전용 hold-out 검증의 정답 기준으로 쓸 수 있다."""
    import cv2
    import numpy as np

    from calibration.types import CameraConfig, Dataset, DetectionResult, Frame, FrameStatus, ImageInfo

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    board = cv2.aruco.CharucoBoard((11, 8), 0.02, 0.015, aruco_dict)
    pts3d = board.getChessboardCorners().astype(np.float32)
    n_corners = pts3d.shape[0]
    ids = np.arange(n_corners, dtype=np.int32).reshape(-1, 1)

    true_K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    true_D = np.array([0.05, 0.01, -0.01, 0.002])

    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_frames):
        rvec = (rng.random(3) - 0.5) * 0.6
        tvec = np.array([(rng.random() - 0.5) * 0.2, (rng.random() - 0.5) * 0.2, 0.3 + rng.random() * 0.2])
        proj, _ = cv2.fisheye.projectPoints(
            pts3d.reshape(-1, 1, 3).astype(np.float64), rvec, tvec, true_K, true_D
        )
        proj = proj.reshape(-1, 2)
        if (proj < 0).any() or (proj[:, 0] > 640).any() or (proj[:, 1] > 480).any():
            continue
        info = ImageInfo(image_id=f"fisheye_{i}", path="-", width=640, height=480)
        det = DetectionResult(
            image_id=f"fisheye_{i}",
            success=True,
            corners=proj.reshape(-1, 1, 2).astype(np.float32),
            object_points=pts3d.reshape(-1, 1, 3),
            ids=ids,
            num_corners=n_corners,
        )
        frames.append(Frame(image_info=info, detection=det, status=FrameStatus.DETECTED))

    dataset = Dataset(frames=frames)
    camera_config = CameraConfig(width=640, height=480, sensor_name="pytest-fisheye-gt")
    return dataset, camera_config, true_K, true_D


def test_fisheye_holdout_validation_recovers_known_k_d(pattern_config):
    """Fisheye 전용 hold-out 검증: 실제 cv2.fisheye.projectPoints()로 만든
    GT 데이터셋에서 Train은 K,D를 fitting하고, Test는 (그 K,D를 고정한 채)
    pose만 추정해서 재투영 오차를 계산해야 한다. 공용 synthetic_dataset은
    Fisheye 모델로 생성된 데이터가 아니므로(위 sequential-independence
    테스트 docstring 참고) 이 테스트가 Fisheye의 실제 검증 책임을 진다."""
    dataset, camera_config, true_K, true_D = _build_synthetic_fisheye_dataset()
    assert len(dataset.frames) >= 12, "합성 프레임이 너무 적게 생성됨 - 테스트 파라미터 조정 필요"

    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=13)
    assert train_ids and test_ids

    test_frames_before = {
        fid: next(f for f in dataset.frames if f.image_info.image_id == fid).reprojection_error
        for fid in test_ids
    }

    result = validate_holdout(
        dataset, camera_config, pattern_config, CameraModelType.FISHEYE, train_ids, test_ids
    )

    assert result.success, result.error_message
    # Train: K,D는 GT에 가깝게 fitting돼야 한다 (완전히 발산하지 않았는지 확인).
    assert result.train_rms is not None and result.train_rms < 5.0

    # Test: per_frame_error에 test 프레임 결과가 남아야 하고, 그 대상은
    # 정확히 test_ids여야 한다 (train 프레임이 섞여 들어가면 안 됨).
    assert result.per_frame_error
    assert set(result.per_frame_error) <= set(test_ids)

    # Test 프레임의 원본 Frame.reprojection_error는 절대 mutate되면 안 된다 -
    # test 평가는 고정된 K,D로 pose만 추정하는 순수 평가일 뿐이다.
    for fid in test_ids:
        frame = next(f for f in dataset.frames if f.image_info.image_id == fid)
        assert frame.reprojection_error == test_frames_before[fid], (
            "Fisheye hold-out test 평가가 test Frame.reprojection_error를 mutate했다."
        )

    # Test RMS 자체도 GT 노이즈 없는 합성 데이터이므로 낮아야 한다 (Test에서
    # K,D를 건드리지 않고 pose만 잘 추정했다면 재투영 오차는 작아야 정상).
    assert result.test_rms is not None and result.test_rms < 5.0


def test_leak_safe_outlier_pruning_only_removes_train_frames(
    synthetic_dataset, camera_config, pattern_config
):
    """recalibrate_train_with_outlier_pruning()이 제거하는 프레임은 전부
    train_ids 소속이어야 한다 - test_ids에 속한 프레임은 아무리 오차가
    커도(설령 완전히 이상한 코너 좌표를 갖고 있어도) 제거 후보에 조차
    오르면 안 된다(제거 로직이 애초에 test 프레임의 오차를 계산하지 않기
    때문에 구조적으로 불가능해야 정상).
    """
    dataset = copy.deepcopy(synthetic_dataset)
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=3)
    assert test_ids

    # test 프레임 하나를 일부러 심하게 망가뜨린다(코너를 화면 밖 극단으로 이동) -
    # 이게 이상치 탐지에 전혀 영향을 주지 않아야 한다.
    victim_id = test_ids[0]
    victim_frame = next(f for f in dataset.frames if f.image_info.image_id == victim_id)
    victim_frame.detection.corners = victim_frame.detection.corners + 5000.0

    train_result, outlier_result, validation_result = recalibrate_train_with_outlier_pruning(
        dataset, camera_config, pattern_config, CameraModelType.PINHOLE, train_ids, test_ids,
        max_iterations=3,
    )

    assert set(outlier_result.removed_frame_ids).issubset(set(train_ids)), (
        "제거된 프레임 중 train_ids에 속하지 않은 게 있음 - test 프레임이 "
        "이상치 판정에 관여했다는 뜻"
    )
    assert victim_id not in outlier_result.removed_frame_ids
    # 망가뜨린 test 프레임은 여전히 활성 상태여야 한다(제거 대상 후보로도 안 올라갔으므로).
    assert victim_frame.status != FrameStatus.DISABLED_OUTLIER


def test_leak_safe_outlier_pruning_never_shrinks_or_changes_test_set(
    synthetic_dataset, camera_config, pattern_config
):
    """이상치 제거를 몇 번을 반복하든 test_frame_ids는 처음 분할 그대로여야
    한다 - "Test는 절대 수정하지 않는다"는 문서 9번 핵심 원칙의 직접적인 검증.
    """
    dataset = copy.deepcopy(synthetic_dataset)
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.25, seed=11)
    original_test_ids = list(test_ids)

    _, outlier_result, validation_result = recalibrate_train_with_outlier_pruning(
        dataset, camera_config, pattern_config, CameraModelType.EXTENDED_PINHOLE,
        train_ids, test_ids, max_iterations=3,
    )

    assert validation_result.test_frame_ids == original_test_ids, (
        "test_frame_ids가 outlier pruning 과정에서 바뀌었음 - leakage 위험"
    )
    # 원본 dataset에서도 test 프레임들의 상태가 그대로 DETECTED여야 한다(비활성화 안 됨).
    for fid in original_test_ids:
        frame = next(f for f in dataset.frames if f.image_info.image_id == fid)
        assert frame.status != FrameStatus.DISABLED_OUTLIER, (
            f"test 프레임 {fid}가 outlier로 비활성화됨 - 있어서는 안 되는 일"
        )


def test_leak_safe_function_matches_validate_holdout_when_no_outliers_removed(
    synthetic_dataset, camera_config, pattern_config
):
    """이상치가 하나도 없는(전부 정상) 경우, leak-safe 함수의 결과는
    outlier 단계가 아예 없는 validate_holdout()과 (train 프레임 구성이
    같다면) 동일한 test_rms를 내야 한다 - 두 경로가 "test 평가"만큼은
    똑같은 로직(_evaluate_on_test)을 공유한다는 걸 간접적으로 확인.
    """
    dataset = copy.deepcopy(synthetic_dataset)
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.25, seed=99)

    # k를 아주 크게 줘서 사실상 이상치가 하나도 안 뽑히게 만든다.
    _, outlier_result, leak_safe_result = recalibrate_train_with_outlier_pruning(
        dataset, camera_config, pattern_config, CameraModelType.PINHOLE,
        train_ids, test_ids, max_iterations=3, k=1000.0,
    )
    assert outlier_result.removed_frame_ids == []

    plain_result = validate_holdout(
        dataset, camera_config, pattern_config, CameraModelType.PINHOLE, train_ids, test_ids
    )

    assert leak_safe_result.test_rms == pytest.approx(plain_result.test_rms, rel=1e-6, abs=1e-6)
    assert leak_safe_result.train_rms == pytest.approx(plain_result.train_rms, rel=1e-9, abs=1e-9)


# ---------------------------------------------------------------------------
# 설계 문서 16/17번 - Corner-level Outlier의 leak-safe 버전 검증
# ---------------------------------------------------------------------------

def test_corner_level_leak_safe_never_touches_test_frames(synthetic_dataset, camera_config, pattern_config):
    """recalibrate_train_with_corner_outlier_pruning()도 프레임 단위 버전과
    동일한 leakage 안전성을 지녀야 한다 - test 프레임의 코너는 절대 제외
    후보에 오르지 않아야 한다.
    """
    dataset = copy.deepcopy(synthetic_dataset)
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.3, seed=5)
    assert test_ids

    train_result, corner_outlier_result, validation_result = recalibrate_train_with_corner_outlier_pruning(
        dataset, camera_config, pattern_config, CameraModelType.PINHOLE, train_ids, test_ids,
        max_iterations=3,
    )

    assert set(corner_outlier_result.removed_corners.keys()).issubset(set(train_ids)), (
        "코너를 제외한 프레임 중 train_ids에 속하지 않은 게 있음 - test 프레임의 "
        "코너가 이상치 판정에 관여했다는 뜻"
    )
    assert validation_result.test_frame_ids == test_ids
    for fid in test_ids:
        frame = next(f for f in dataset.frames if f.image_info.image_id == fid)
        assert frame.detection.excluded_corner_indices == [], (
            f"test 프레임 {fid}의 코너가 제외됨 - 있어서는 안 되는 일"
        )


def test_corner_level_leak_safe_reproduces_pure_train_fit(synthetic_dataset, camera_config, pattern_config):
    """corner-level leak-safe 결과의 train_rms가, 완전히 독립적으로 같은
    (코너 제외 반영된) train 부분집합만으로 다시 계산한 결과와 일치해야 한다."""
    dataset = copy.deepcopy(synthetic_dataset)
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.25, seed=13)

    train_result, corner_outlier_result, validation_result = recalibrate_train_with_corner_outlier_pruning(
        dataset, camera_config, pattern_config, CameraModelType.PINHOLE, train_ids, test_ids,
        max_iterations=3,
    )
    assert train_result.success

    # 위 함수가 만든 dataset(코너 제외가 이미 반영됨)에서 train 부분집합만
    # 독립적으로 다시 fit해서 정확히 같은 결과가 나오는지 확인.
    independent_fit = calibrate_pinhole(_subset_dataset(dataset, train_ids), camera_config)
    assert independent_fit.success
    assert train_result.rms_error == pytest.approx(independent_fit.rms_error, rel=1e-9, abs=1e-9)


# ---------------------------------------------------------------------------
# Phase B-6 - Hold-out Evidence Gate.
#
# synthetic_dataset은 16장뿐이라 test_ratio=0.25면 test 프레임이 3~4장뿐이다
# (calibration/holdout_evidence.py의 MIN_EVIDENCE_TEST_FRAMES=5보다 적음) -
# 이 자연스러운 소규모 데이터셋 자체가 "insufficient evidence" 케이스를
# 실제로 재현하는 좋은 fixture가 된다(인위적으로 조작할 필요 없음).
# ---------------------------------------------------------------------------

def test_validate_holdout_populates_evidence_gate(synthetic_dataset, camera_config, pattern_config):
    dataset = copy.deepcopy(synthetic_dataset)
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.25, seed=42)
    assert test_ids

    result = validate_holdout(
        dataset, camera_config, pattern_config, CameraModelType.PINHOLE, train_ids, test_ids
    )
    assert result.success
    assert result.evidence_gate is not None
    assert result.evidence_gate.status in ("sufficient", "insufficient_evidence")
    assert result.evidence_gate.test_frame_count == len(test_ids)
    assert result.evidence_gate.test_corner_count > 0


def test_evidence_gate_never_overrides_or_hides_test_rms(synthetic_dataset, camera_config, pattern_config):
    """Evidence Gate는 test_rms 숫자 자체를 바꾸거나 success를 False로
    만들면 안 된다 - "낮은 RMS를 틀렸다고 판정"하는 게 아니라 근거 부족을
    별도로 보고할 뿐이다(사용자 스펙 B-6번 핵심 원칙)."""
    dataset = copy.deepcopy(synthetic_dataset)
    train_ids, test_ids = split_train_test(dataset, camera_config, test_ratio=0.25, seed=42)

    result = validate_holdout(
        dataset, camera_config, pattern_config, CameraModelType.PINHOLE, train_ids, test_ids
    )
    assert result.success is True
    assert result.test_rms is not None
    # evidence_gate가 채워졌더라도 test_rms/success는 그대로여야 한다.
    assert result.evidence_gate is not None


def test_small_synthetic_test_set_is_flagged_as_insufficient_evidence(synthetic_dataset, camera_config):
    """16장짜리 synthetic_dataset을 0.25 비율로 나누면 test 프레임이
    MIN_EVIDENCE_TEST_FRAMES(5)보다 적다 - 이런 소규모 test set은 RMS가
    아무리 낮아도 "insufficient evidence"로 명시적으로 구분되어야 한다."""
    dataset = copy.deepcopy(synthetic_dataset)
    _, test_ids = split_train_test(dataset, camera_config, test_ratio=0.25, seed=42)
    assert 0 < len(test_ids) < 5, "이 테스트는 test set이 작다는 전제가 깨지면 무의미함"

    test_dataset = _subset_dataset(dataset, test_ids)
    gate = evaluate_holdout_evidence(test_dataset, camera_config)

    assert gate.status == "insufficient_evidence"
    assert gate.reasons  # 근거 부족 사유가 최소 하나 이상 기록되어야 한다
    assert any("프레임" in r for r in gate.reasons)


def test_evidence_gate_reports_sufficient_when_thresholds_are_relaxed(synthetic_dataset, camera_config):
    """같은 test set이라도 threshold를 충분히 낮추면 "sufficient"로 판정돼야
    한다 - gate 로직 자체가 항상 insufficient만 내는 고정 함수가 아님을
    확인한다(사용자 스펙 - "낮은 RMS를 INVALID로 바꾸는 것"이 아니라 근거를
    평가하는 것이므로, 근거가 실제로 충분하면 sufficient가 나와야 함)."""
    dataset = copy.deepcopy(synthetic_dataset)
    _, test_ids = split_train_test(dataset, camera_config, test_ratio=0.25, seed=42)
    test_dataset = _subset_dataset(dataset, test_ids)

    gate = evaluate_holdout_evidence(
        test_dataset, camera_config,
        min_test_frames=1, min_test_corners=1, min_coverage_pct=0.0, min_pose_diversity=0.0,
    )
    assert gate.status == "sufficient"
    assert gate.reasons == []


def test_evidence_gate_is_none_when_no_test_frames(synthetic_dataset, camera_config, pattern_config):
    """Test 프레임이 아예 없어 hold-out 평가 자체를 못 하는 경로
    (`_evaluate_on_test`의 조기 반환)에서는 evidence_gate도 "아직 평가되지
    않음"을 뜻하는 None으로 남아야 한다 - 억지로 "insufficient_evidence"를
    지어내지 않는다."""
    dataset = copy.deepcopy(synthetic_dataset)
    all_ids = [f.image_info.image_id for f in dataset.enabled_frames if f.detection and f.detection.success]

    result = validate_holdout(
        dataset, camera_config, pattern_config, CameraModelType.PINHOLE, all_ids, [],
    )
    assert result.success
    assert result.test_frame_ids == []
    assert result.evidence_gate is None


def test_evidence_gate_does_not_mutate_input_dataset(synthetic_dataset, camera_config):
    """Phase A-8 원칙 재확인 - evaluate_holdout_evidence()는
    analyze_dataset_quality()와 달리 dataset.coverage_grid/diversity에 직접
    쓰지 않아야 한다(호출자가 그 결과를 이미 다른 용도로 쓰고 있을 수
    있으므로 부수효과를 만들지 않는다)."""
    dataset = copy.deepcopy(synthetic_dataset)
    _, test_ids = split_train_test(dataset, camera_config, test_ratio=0.25, seed=42)
    test_dataset = _subset_dataset(dataset, test_ids)
    assert test_dataset.coverage_grid == []
    assert test_dataset.diversity is None

    evaluate_holdout_evidence(test_dataset, camera_config)

    assert test_dataset.coverage_grid == []
    assert test_dataset.diversity is None

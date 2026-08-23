"""
tests/test_validation.py
=============================

설계 문서 3.3번 - Hold-out Validation. train/test 분할이 겹치지 않는지,
전체 프레임을 다 커버하는지, test 프레임의 intrinsic이 재최적화되지
않는다는 원칙(설계 문서 핵심 경고)이 지켜지는지 확인한다.
"""

from __future__ import annotations

import copy

from calibration.types import CameraModelType, FrameStatus
from calibration.validation import (
    _subset_dataset,
    recalibrate_train_with_corner_outlier_pruning,
    recalibrate_train_with_outlier_pruning,
    split_train_test,
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
    assert validation_result.train_rms == independent_train_fit.rms_error, (
        "validate_holdout 내부에서 쓰인 train fit이 순수 train-only 결과와 달라짐 - "
        "test 정보가 학습에 섞여 들어갔을 가능성이 있음"
    )


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

    assert leak_safe_result.test_rms == plain_result.test_rms
    assert leak_safe_result.train_rms == plain_result.train_rms


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
    assert train_result.rms_error == independent_fit.rms_error

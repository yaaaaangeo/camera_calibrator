"""
camera_calibrator.calibration.holdout_evidence
==============================================================

Phase B-6 안정화 - Hold-out Evidence Gate.

Hold-out RMS(test_rms)가 낮다는 사실 하나만으로 "이 캘리브레이션은
검증됐다"고 말할 수는 없다: test 프레임이 2~3장뿐이거나, 코너가 몇 개
안 되거나, 이미지의 한 구석에만 몰려 있거나, 자세(pose) 다양성이 거의
없으면 그 낮은 RMS는 "우연히 좋게 나온 숫자"일 뿐 신뢰할 수 있는
evidence가 아니다.

이 모듈은 새로운 임의 threshold/지표를 발명하지 않는다 - 이미 검증되어
쓰이고 있는 dataset quality/coverage 로직(calibration/quality.py의
`compute_coverage_grid`/`compute_diversity_scores`)을 test 프레임
부분집합에 그대로 재사용해서 "이 Hold-out 결과를 얼마나 믿어도 되는가"를
판단한다.

핵심 원칙: RMS 숫자 자체를 바꾸거나 무효화하지 않는다. `status` 필드로
"VALID(낮은 RMS + 충분한 근거)"와 "INSUFFICIENT EVIDENCE(근거 부족, 판단
보류)"를 구분해서 보고할 뿐이다.
"""

from __future__ import annotations

from calibration.quality import compute_coverage_grid, compute_diversity_scores, coverage_percentage
from calibration.types import CameraConfig, Dataset, HoldoutEvidenceGate

# 초기 default일 뿐 실데이터로 재조정 필요 - 다른 모듈들의 임의
# threshold(DEFAULT_MAD_OUTLIER_K, alignment translation ratio 등)와 동일한
# 성격이다. MIN_FRAMES_REQUIRED(calibration/models/common.py)가 "calibration
# 자체가 가능한 최소"라면, 아래 값들은 "그 hold-out 결과를 신뢰할 만한
# 최소"라 그보다 더 엄격할 수 있다.
MIN_EVIDENCE_TEST_FRAMES = 5
MIN_EVIDENCE_TEST_CORNERS = 60
MIN_EVIDENCE_COVERAGE_PCT = 40.0
MIN_EVIDENCE_POSE_DIVERSITY = 0.25


def evaluate_holdout_evidence(
    test_dataset: Dataset,
    camera_config: CameraConfig,
    *,
    min_test_frames: int = MIN_EVIDENCE_TEST_FRAMES,
    min_test_corners: int = MIN_EVIDENCE_TEST_CORNERS,
    min_coverage_pct: float = MIN_EVIDENCE_COVERAGE_PCT,
    min_pose_diversity: float = MIN_EVIDENCE_POSE_DIVERSITY,
) -> HoldoutEvidenceGate:
    """`test_dataset`(이미 hold-out 평가에 쓰인 test 프레임 부분집합)에
    대해, 그 test 결과를 뒷받침할 근거가 충분한지 판정한다.

    Phase A-8 원칙("Calibration/Validation 함수는 가능하면 입력 Dataset을
    mutate하지 않는다")을 지키기 위해 `analyze_dataset_quality()`처럼
    `dataset.coverage_grid`/`dataset.diversity`에 직접 쓰지 않고, 계산
    결과를 별도의 `HoldoutEvidenceGate`에만 담아 반환한다."""
    frames = [f for f in test_dataset.enabled_frames if f.detection and f.detection.success]
    frame_count = len(frames)
    corner_count = sum(f.detection.num_corners for f in frames)

    cells = compute_coverage_grid(test_dataset, camera_config)
    coverage_pct = coverage_percentage(cells)
    diversity = compute_diversity_scores(test_dataset, cells)
    pose_diversity = diversity.overall

    reasons: list[str] = []
    if frame_count < min_test_frames:
        reasons.append(
            f"Test 프레임 수가 부족합니다({frame_count}장, 최소 {min_test_frames}장 권장)."
        )
    if corner_count < min_test_corners:
        reasons.append(
            f"Test 코너 수가 부족합니다({corner_count}개, 최소 {min_test_corners}개 권장)."
        )
    if coverage_pct < min_coverage_pct:
        reasons.append(
            f"Test 프레임의 공간 coverage가 낮습니다({coverage_pct:.0f}%, 최소 {min_coverage_pct:.0f}% 권장)."
        )
    if pose_diversity < min_pose_diversity:
        reasons.append(
            f"Test 프레임의 자세(pose) 다양성이 낮습니다({pose_diversity:.2f}, 최소 {min_pose_diversity:.2f} 권장)."
        )

    status = "insufficient_evidence" if reasons else "sufficient"
    return HoldoutEvidenceGate(
        status=status,
        test_frame_count=frame_count,
        test_corner_count=corner_count,
        test_coverage_pct=coverage_pct,
        test_pose_diversity=pose_diversity,
        reasons=reasons,
    )

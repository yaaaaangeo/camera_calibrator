"""
camera_calibrator.calibration.outlier
=========================================

설계 문서 9번 - Outlier 관리.

원칙 (문서에서 반복 강조):
    "자동 삭제가 아니라 추천 + 사용자 확인이 원칙이다."
    "파일 자체를 삭제하지 않는다. 이미지는 그대로 두고 메타데이터에서
     비활성화 처리한다."

그래서 이 모듈의 함수들은 두 층위로 나뉜다:
    - recommend_outliers(): 순수 계산. 판단만 하고 아무것도 바꾸지 않는다.
    - apply_outlier_removal(): Frame.status를 DISABLED_OUTLIER로 바꾸는
      실제 부수효과. UI에서 사용자가 "[제외하고 다시 계산]"을 눌렀을 때만
      호출되어야 한다.
    - recalibrate_with_outlier_pruning(): 위 둘을 반복 적용하는 자동화
      버전이지만, 이것도 "자동 삭제"가 아니라 파이프라인 내부에서 쓰는
      제한된 반복(max_iterations=3)일 뿐 - 최종 채택 여부는 여전히
      OutlierResult를 사용자에게 보여준 뒤 결정하는 것을 전제로 한다.
"""

from __future__ import annotations

import numpy as np

from calibration.types import (
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    Dataset,
    OutlierResult,
)
from calibration.models.common import MIN_FRAMES_REQUIRED
from calibration.models.pinhole import calibrate_pinhole
from calibration.models.extended_pinhole import calibrate_extended_pinhole
from calibration.models.fisheye import calibrate_fisheye


# ---------------------------------------------------------------------------
# Threshold 계산 (설계 문서 9번: threshold = median(error) + 3*MAD)
# ---------------------------------------------------------------------------

def compute_mad_threshold(errors: list[float], k: float = 3.0, mad_scale: float = 1.0) -> float:
    """threshold = median(error) + k * MAD

    mad_scale: 정규분포 가정 하에서 MAD를 표준편차와 비슷한 스케일로 맞추려면
    1.4826을 곱하는 게 통계적으로 흔한 관례다. 문서 원문 공식은 스케일링 없이
    그대로("median + 3*MAD")이므로 기본값은 1.0으로 두되, 필요하면
    UI 고급 옵션에서 1.4826으로 바꿀 수 있게 열어둔다.
    """
    if not errors:
        return 0.0
    arr = np.asarray(errors, dtype=float)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    return median + k * mad_scale * mad


def recommend_outliers(
    per_frame_error: dict[str, float],
    k: float = 3.0,
    mad_scale: float = 1.0,
    user_threshold: float | None = None,
) -> tuple[list[str], float]:
    """이상치로 의심되는 frame_id 목록(오차 큰 순으로 정렬)과 사용된 threshold 반환.

    아무것도 disable하지 않는다 - 순수 함수.
    """
    if not per_frame_error:
        return [], 0.0

    threshold = (
        user_threshold
        if user_threshold is not None
        else compute_mad_threshold(list(per_frame_error.values()), k=k, mad_scale=mad_scale)
    )

    flagged = [
        (frame_id, err) for frame_id, err in per_frame_error.items() if err > threshold
    ]
    flagged.sort(key=lambda kv: kv[1], reverse=True)
    return [frame_id for frame_id, _ in flagged], threshold


# ---------------------------------------------------------------------------
# 실제 비활성화 (부수효과 있음 - 사용자 확인 후에만 호출)
# ---------------------------------------------------------------------------

def apply_outlier_removal(
    dataset: Dataset,
    frame_ids: list[str],
    reason: str = "high_reprojection_error",
) -> None:
    """설계 문서 9번 JSON 예시와 동일한 의미:
        { "image": "img004.jpg", "enabled": false, "reason": "high_reprojection_error" }

    파일이나 Frame 객체 자체는 지우지 않고 status만 DISABLED_OUTLIER로 바꾼다.
    """
    id_set = set(frame_ids)
    for frame in dataset.frames:
        if frame.image_info.image_id in id_set:
            frame.disable(reason=reason, outlier=True)


def restore_frame(dataset: Dataset, frame_id: str) -> None:
    """사용자가 "역시 이 프레임은 살릴래" 할 때를 위한 되돌리기.
    파일을 지운 적이 없으므로 상태만 되돌리면 그만이다.
    """
    from calibration.types import FrameStatus

    for frame in dataset.frames:
        if frame.image_info.image_id == frame_id:
            frame.status = FrameStatus.DETECTED
            frame.disabled_reason = None


# ---------------------------------------------------------------------------
# 모델 디스패치 (compare.py / validation.py와 동일한 패턴)
# ---------------------------------------------------------------------------

def _calibrate_by_model(
    dataset: Dataset,
    camera_config: CameraConfig,
    model: CameraModelType,
    use_rational_model: bool = False,
    fisheye_initial_guess: CalibrationResult | None = None,
) -> CalibrationResult:
    if model == CameraModelType.PINHOLE:
        return calibrate_pinhole(dataset, camera_config)
    if model == CameraModelType.EXTENDED_PINHOLE:
        return calibrate_extended_pinhole(dataset, camera_config, use_rational_model=use_rational_model)
    if model == CameraModelType.FISHEYE:
        return calibrate_fisheye(dataset, camera_config, initial_guess=fisheye_initial_guess)
    raise ValueError(f"알 수 없는 모델: {model}")


# ---------------------------------------------------------------------------
# 반복 재계산 (설계 문서 9번, 17번 Step6/7 - max_iterations=3)
# ---------------------------------------------------------------------------

def recalibrate_with_outlier_pruning(
    dataset: Dataset,
    camera_config: CameraConfig,
    model: CameraModelType,
    max_iterations: int = 3,
    k: float = 3.0,
    mad_scale: float = 1.0,
    user_threshold: float | None = None,
    use_rational_model: bool = False,
    fisheye_initial_guess: CalibrationResult | None = None,
) -> tuple[CalibrationResult, OutlierResult]:
    """이상치를 반복적으로 찾아 제거하며 재계산.

    안전장치:
    - 한 번에 제거해서 남는 프레임이 MIN_FRAMES_REQUIRED 밑으로 떨어지면
      그만큼만 제거하거나(우선순위: 오차 큰 프레임부터) 아예 멈춘다.
    - max_iterations를 넘기지 않는다 (무한 반복 방지).
    - 최종적으로 실패(success=False)한 재계산 결과는 채택하지 않고,
      마지막으로 성공했던 결과로 되돌린다.
    """
    result = _calibrate_by_model(
        dataset, camera_config, model,
        use_rational_model=use_rational_model,
        fisheye_initial_guess=fisheye_initial_guess,
    )

    if not result.success:
        return result, OutlierResult(
            threshold_used=0.0, removed_frame_ids=[], rms_before=None, rms_after=None,
            iterations=0, max_iterations=max_iterations,
        )

    rms_before = result.rms_error
    last_good_result = result
    removed_all: list[str] = []
    last_threshold = 0.0
    iterations = 0

    for _ in range(max_iterations):
        candidates, threshold = recommend_outliers(
            last_good_result.per_frame_error, k=k, mad_scale=mad_scale, user_threshold=user_threshold
        )
        if not candidates:
            break
        last_threshold = threshold

        # 안전장치: 최소 프레임 수 보장. 오차가 큰 순으로 정렬되어 있으므로
        # 앞에서부터 허용 가능한 만큼만 자른다.
        current_usable = len(last_good_result.per_frame_error)
        max_removable = current_usable - MIN_FRAMES_REQUIRED
        if max_removable <= 0:
            break
        candidates = candidates[:max_removable]

        apply_outlier_removal(dataset, candidates, reason="high_reprojection_error (auto)")
        removed_all.extend(candidates)
        iterations += 1

        new_result = _calibrate_by_model(
            dataset, camera_config, model,
            use_rational_model=use_rational_model,
            fisheye_initial_guess=fisheye_initial_guess,
        )
        if not new_result.success:
            # 이번 제거가 오히려 계산 자체를 깨뜨렸다면 되돌리고 멈춘다.
            for frame_id in candidates:
                restore_frame(dataset, frame_id)
            removed_all = removed_all[: -len(candidates)]
            iterations -= 1
            break

        last_good_result = new_result

    outlier_result = OutlierResult(
        threshold_used=last_threshold,
        removed_frame_ids=removed_all,
        rms_before=rms_before,
        rms_after=last_good_result.rms_error,
        iterations=iterations,
        max_iterations=max_iterations,
    )
    return last_good_result, outlier_result


# ---------------------------------------------------------------------------
# 출력용 요약
# ---------------------------------------------------------------------------

def format_outlier_summary(outlier_result: OutlierResult) -> str:
    if not outlier_result.removed_frame_ids:
        return "이상치로 판단된 프레임이 없습니다."

    before = f"{outlier_result.rms_before:.3f}" if outlier_result.rms_before is not None else "N/A"
    after = f"{outlier_result.rms_after:.3f}" if outlier_result.rms_after is not None else "N/A"
    lines = [
        f"제외된 프레임 ({outlier_result.iterations}회 반복, threshold={outlier_result.threshold_used:.3f}px):",
    ]
    for frame_id in outlier_result.removed_frame_ids:
        lines.append(f"  - {frame_id}")
    lines.append(f"RMS {before}px -> {after}px 로 개선")
    return "\n".join(lines)

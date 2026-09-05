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
    CornerOutlierResult,
    Dataset,
    Frame,
    OutlierResult,
)
from calibration.models.common import MIN_FRAMES_REQUIRED, MIN_CORNERS_PER_FRAME, compute_mad_threshold, collect_calibration_inputs
from calibration.models.pinhole import calibrate_pinhole
from calibration.models.brown_conrady import calibrate_brown_conrady
from calibration.models.extended_pinhole import calibrate_extended_pinhole
from calibration.models.fisheye import calibrate_fisheye
from calibration.radial_profile import _project


# ---------------------------------------------------------------------------
# Threshold 계산 (설계 문서 9번: threshold = median(error) + 3*MAD)
# ---------------------------------------------------------------------------
# compute_mad_threshold()의 실제 정의는 models/common.py로 옮겨졌다 (순환 참조
# 회피 - residual_stats.py docstring 참고). 기존에 `from calibration.outlier
# import compute_mad_threshold`로 쓰던 코드가 계속 동작하도록 여기서 재노출한다.


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
    reasons: dict[str, str] | None = None,
) -> None:
    """설계 문서 9번 JSON 예시와 동일한 의미:
        { "image": "img004.jpg", "enabled": false, "reason": "high_reprojection_error" }

    파일이나 Frame 객체 자체는 지우지 않고 status만 DISABLED_OUTLIER로 바꾼다.

    reasons: 설계 문서 16번 "왜 제거됐는지 기록" - frame_id별로 다른(더 상세한,
    실제 오차/threshold 수치가 들어간) 사유 문자열을 주고 싶을 때 쓴다. 없으면
    전부 reason 하나로 통일해서 기록한다(기존 동작과 동일, 하위 호환).
    """
    id_set = set(frame_ids)
    for frame in dataset.frames:
        if frame.image_info.image_id in id_set:
            frame_reason = reasons.get(frame.image_info.image_id, reason) if reasons else reason
            frame.disable(reason=frame_reason, outlier=True)


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
    fisheye_initial_guess: CalibrationResult | None = None,
) -> CalibrationResult:
    if model == CameraModelType.PINHOLE:
        return calibrate_pinhole(dataset, camera_config)
    if model == CameraModelType.BROWN_CONRADY:
        return calibrate_brown_conrady(dataset, camera_config)
    if model == CameraModelType.EXTENDED_PINHOLE:
        return calibrate_extended_pinhole(dataset, camera_config)
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
    fisheye_initial_guess: CalibrationResult | None = None,
) -> tuple[CalibrationResult, OutlierResult]:
    """이상치를 반복적으로 찾아 제거하며 재계산.

    안전장치:
    - 한 번에 제거해서 남는 프레임이 MIN_FRAMES_REQUIRED 밑으로 떨어지면
      그만큼만 제거하거나(우선순위: 오차 큰 프레임부터) 아예 멈춘다.
    - max_iterations를 넘기지 않는다 (무한 반복 방지).
    - 최종적으로 실패(success=False)한 재계산 결과는 채택하지 않고,
      마지막으로 성공했던 결과로 되돌린다.

    모델 의미는 초기 계산과 반복 재계산 내내 고정이다(_calibrate_by_model이
    각 CameraModelType에 대해 항상 같은 함수를 호출) - 예를 들어 Rational로
    시작했는데 재계산 후 Brown 5계수가 되는 일은 구조적으로 불가능하다.
    """
    result = _calibrate_by_model(
        dataset, camera_config, model,
        fisheye_initial_guess=fisheye_initial_guess,
    )

    if not result.success:
        return result, OutlierResult(
            threshold_used=0.0, removed_frame_ids=[], rms_before=None, rms_after=None,
            iterations=0, max_iterations=max_iterations,
        )

    rms_before = result.rms_error
    p95_before = result.residual_stats.p95 if result.residual_stats else None
    camera_matrix_before = result.camera_matrix.copy() if result.camera_matrix is not None else None
    distortion_before = result.distortion.copy() if result.distortion is not None else None
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

        # 설계 문서 16번 "왜 제거됐는지 기록" - 프레임마다 실제 오차값과
        # threshold를 그대로 사유 문자열에 남긴다. "high_reprojection_error
        # (auto)"라는 뭉뚱그린 한마디보다 나중에 사람이 검토할 때 훨씬 유용하다.
        reasons = {
            fid: (
                f"high_reprojection_error (auto): frame RMS={last_good_result.per_frame_error[fid]:.3f}px "
                f"> threshold={threshold:.3f}px (median+{k:.1f}*MAD)"
            )
            for fid in candidates
        }
        apply_outlier_removal(dataset, candidates, reasons=reasons)
        removed_all.extend(candidates)
        iterations += 1

        new_result = _calibrate_by_model(
            dataset, camera_config, model,
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

    p95_after = last_good_result.residual_stats.p95 if last_good_result.residual_stats else None
    camera_matrix_after = last_good_result.camera_matrix.copy() if last_good_result.camera_matrix is not None else None
    distortion_after = last_good_result.distortion.copy() if last_good_result.distortion is not None else None

    outlier_result = OutlierResult(
        threshold_used=last_threshold,
        removed_frame_ids=removed_all,
        rms_before=rms_before,
        rms_after=last_good_result.rms_error,
        iterations=iterations,
        max_iterations=max_iterations,
        p95_before=p95_before,
        p95_after=p95_after,
        camera_matrix_before=camera_matrix_before,
        camera_matrix_after=camera_matrix_after,
        distortion_before=distortion_before,
        distortion_after=distortion_after,
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


def _format_before_after_metrics(
    rms_before, rms_after, p95_before, p95_after,
    camera_matrix_before, camera_matrix_after, distortion_before, distortion_after,
    header: str,
) -> str:
    """OutlierResult(frame-level)와 CornerOutlierResult(corner-level)가 필드
    구성이 완전히 같으므로(설계 문서 17번 요구사항을 둘 다 만족시키려고
    일부러 맞춰뒀다 - types.py CornerOutlierResult docstring 참고), 실제
    포맷팅 로직은 여기 하나로 합쳐서 두 군데서 재사용한다.
    """
    def fmt(v: float | None, digits: int = 3) -> str:
        return f"{v:.{digits}f}" if v is not None else "N/A"

    lines = [header]
    lines.append(f"{'RMSE':<10}{fmt(rms_before):>10} -> {fmt(rms_after):>10} px")
    lines.append(f"{'P95':<10}{fmt(p95_before):>10} -> {fmt(p95_after):>10} px")

    K0, K1 = camera_matrix_before, camera_matrix_after
    if K0 is not None and K1 is not None:
        lines.append(f"{'fx':<10}{fmt(float(K0[0, 0])):>10} -> {fmt(float(K1[0, 0])):>10}")
        lines.append(f"{'fy':<10}{fmt(float(K0[1, 1])):>10} -> {fmt(float(K1[1, 1])):>10}")
        lines.append(f"{'cx':<10}{fmt(float(K0[0, 2])):>10} -> {fmt(float(K1[0, 2])):>10}")
        lines.append(f"{'cy':<10}{fmt(float(K0[1, 2])):>10} -> {fmt(float(K1[1, 2])):>10}")

    D0, D1 = distortion_before, distortion_after
    if D0 is not None and D1 is not None and D0.size == D1.size and D0.size > 0:
        delta_norm = float(np.linalg.norm(D1.flatten() - D0.flatten()))
        lines.append(f"distortion 변화(L2 노름): {delta_norm:.4f}")

    return "\n".join(lines)


def format_outlier_before_after(outlier_result: OutlierResult) -> str:
    """설계 문서 17번 - "Outlier 제거 전후 효과 측정". RMS 하나가 아니라
    P95, fx/fy, distortion 계수까지 전/후를 나란히 보여준다.

        Outlier Removal: Before -> After
        RMSE               0.720 ->  0.340 px
        P95                1.410 ->  0.650 px
        fx              1102.30 -> 1098.71
        fy              1099.80 -> 1096.20
        distortion 변화(L2 노름): 0.0231
    """
    if not outlier_result.removed_frame_ids:
        return "이상치가 제거되지 않아 비교할 전/후 차이가 없습니다."

    return _format_before_after_metrics(
        outlier_result.rms_before, outlier_result.rms_after,
        outlier_result.p95_before, outlier_result.p95_after,
        outlier_result.camera_matrix_before, outlier_result.camera_matrix_after,
        outlier_result.distortion_before, outlier_result.distortion_after,
        header="Outlier Removal: Before -> After",
    )


def format_corner_outlier_before_after(result: CornerOutlierResult) -> str:
    """format_outlier_before_after()의 corner-level 버전 - 설계 문서 17번
    요구사항(RMSE/P95/parameter 변화)을 corner-level 제거에도 동일하게 적용."""
    if not result.removed_corners:
        return "이상치로 제거된 코너가 없어 비교할 전/후 차이가 없습니다."

    return _format_before_after_metrics(
        result.rms_before, result.rms_after,
        result.p95_before, result.p95_after,
        result.camera_matrix_before, result.camera_matrix_after,
        result.distortion_before, result.distortion_after,
        header="Corner Outlier Removal: Before -> After",
    )


# ---------------------------------------------------------------------------
# 설계 문서 16번 - Corner-level Outlier Detection
# ---------------------------------------------------------------------------
# 프레임 단위 이상치 제거(위)는 "이 프레임 전체가 나쁘다"는 판단이지만, 실제로는
# 한 프레임 안에서 코너 몇 개만 (예: 보드 모서리가 살짝 가려졌거나 그림자가
# 져서) 튀고 나머지는 멀쩡한 경우가 흔하다. 그럴 땐 프레임 전체를 버리는 것보다
# 문제되는 코너 몇 개만 calibration 입력에서 빼는 게 데이터를 더 아낀다.

def compute_per_corner_errors(
    frame: Frame,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model: CameraModelType,
) -> np.ndarray:
    """이 프레임 하나의 코너마다 재투영 오차(스칼라, px)를 계산."""
    det = frame.detection
    if not det or det.object_points is None or det.corners is None:
        return np.array([])
    try:
        projected = _project(det.object_points, rvec, tvec, camera_matrix, distortion, model)
    except Exception:  # noqa: BLE001 - cv2.error 등 어떤 투영 실패든 조용히 빈 배열 반환
        return np.array([])

    detected = det.corners.reshape(-1, 2)
    if detected.shape[0] != projected.shape[0]:
        return np.array([])
    diff = detected - projected
    return np.hypot(diff[:, 0], diff[:, 1])


def recommend_corner_outliers(
    frames: list[Frame],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model: CameraModelType,
    k: float = 3.0,
    mad_scale: float = 1.0,
) -> tuple[dict[str, list[int]], float]:
    """설계 문서 16번 - corner-level outlier 추천. 순수 함수(아무것도 바꾸지 않음).

    threshold는 "이 계산에 참여한 모든 코너"를 한데 모아 median+k*MAD로 딱
    하나만 계산한다 - 프레임마다 다른 threshold를 쓰면 "이 프레임 기준으론
    정상인데 다른 프레임 기준으론 이상치"라는 애매함이 생기므로, 프레임 단위
    recommend_outliers()와 같은 철학을 코너 단위에 그대로 적용한다.

    Returns:
        (frame_id -> 이상치로 의심되는 코너 인덱스 목록, 사용된 threshold)
    """
    per_frame_errors: dict[str, np.ndarray] = {}
    all_errors: list[float] = []

    for frame, rvec, tvec in zip(frames, rvecs, tvecs):
        errors = compute_per_corner_errors(frame, rvec, tvec, camera_matrix, distortion, model)
        if errors.size == 0:
            continue
        per_frame_errors[frame.image_info.image_id] = errors
        all_errors.extend(errors.tolist())

    if not all_errors:
        return {}, 0.0

    threshold = compute_mad_threshold(all_errors, k=k, mad_scale=mad_scale)

    result: dict[str, list[int]] = {}
    for frame_id, errors in per_frame_errors.items():
        idx = np.where(errors > threshold)[0]
        if idx.size > 0:
            result[frame_id] = idx.tolist()
    return result, threshold


def apply_corner_outlier_removal(
    dataset: Dataset,
    corner_outliers: dict[str, list[int]],
) -> None:
    """추천된 코너 인덱스를 DetectionResult.excluded_corner_indices에 반영한다.

    프레임 status는 건드리지 않는다 - 프레임은 여전히 "활성"이고, 다음
    calibration 입력을 만들 때(models/common.collect_calibration_inputs)만
    이 인덱스들이 빠진다. 이미 제외돼 있던 인덱스와 합집합을 취해서, 여러
    번 반복 호출해도(예: recalibrate_with_corner_outlier_pruning의 반복 루프)
    이전에 뺀 코너가 되살아나지 않는다.
    """
    lookup = {f.image_info.image_id: f for f in dataset.frames}
    for frame_id, indices in corner_outliers.items():
        frame = lookup.get(frame_id)
        if frame is None or frame.detection is None:
            continue
        existing = set(frame.detection.excluded_corner_indices)
        existing.update(indices)
        frame.detection.excluded_corner_indices = sorted(existing)


def recalibrate_with_corner_outlier_pruning(
    dataset: Dataset,
    camera_config: CameraConfig,
    model: CameraModelType,
    max_iterations: int = 3,
    k: float = 3.0,
    mad_scale: float = 1.0,
    fisheye_initial_guess: CalibrationResult | None = None,
) -> tuple[CalibrationResult, CornerOutlierResult]:
    """recalibrate_with_outlier_pruning()의 corner-level 버전 - 프레임을 통째로
    빼는 대신 문제 코너만 골라 뺀다. 안전장치(최소 프레임 수, 반복 상한,
    실패 시 롤백)는 프레임 단위 버전과 동일한 원칙을 그대로 따른다.
    """
    result = _calibrate_by_model(
        dataset, camera_config, model,
        fisheye_initial_guess=fisheye_initial_guess,
    )
    if not result.success:
        return result, CornerOutlierResult(threshold_used=0.0, iterations=0, max_iterations=max_iterations)

    rms_before = result.rms_error
    p95_before = result.residual_stats.p95 if result.residual_stats else None
    camera_matrix_before = result.camera_matrix.copy() if result.camera_matrix is not None else None
    distortion_before = result.distortion.copy() if result.distortion is not None else None
    last_good_result = result
    removed_all: dict[str, list[int]] = {}
    last_threshold = 0.0
    iterations = 0

    for _ in range(max_iterations):
        frames, _, _ = collect_calibration_inputs(dataset)
        candidates, threshold = recommend_corner_outliers(
            frames, last_good_result.rvecs, last_good_result.tvecs,
            last_good_result.camera_matrix, last_good_result.distortion, model,
            k=k, mad_scale=mad_scale,
        )
        if not candidates:
            break
        last_threshold = threshold

        # 안전장치: 한 프레임에서 코너를 너무 많이 빼서 MIN_CORNERS_PER_FRAME
        # 밑으로 떨어지면 그 프레임 전체가 자동으로 다음 반복에서 빠진다
        # (collect_calibration_inputs가 이미 그렇게 처리함) - 여기서는 그보다
        # 한 걸음 더 보수적으로, 남는 코너가 MIN_CORNERS_PER_FRAME 밑으로
        # 떨어질 인덱스 조합은 이번 반복에서 아예 적용하지 않는다.
        safe_candidates = {}
        for fid, indices in candidates.items():
            frame = next((f for f in frames if f.image_info.image_id == fid), None)
            if frame is None:
                continue
            remaining = frame.detection.num_corners - len(
                set(indices) | set(frame.detection.excluded_corner_indices)
            )
            if remaining >= MIN_CORNERS_PER_FRAME:  # 코너 최소치와 별개로 보수적 여유
                safe_candidates[fid] = indices
        if not safe_candidates:
            break

        apply_corner_outlier_removal(dataset, safe_candidates)
        for fid, indices in safe_candidates.items():
            removed_all.setdefault(fid, [])
            removed_all[fid] = sorted(set(removed_all[fid]) | set(indices))
        iterations += 1

        new_result = _calibrate_by_model(
            dataset, camera_config, model,
            fisheye_initial_guess=fisheye_initial_guess,
        )
        if not new_result.success:
            # 롤백: 이번 반복에서 새로 제외한 인덱스만 되돌린다.
            for fid, indices in safe_candidates.items():
                frame = next((f for f in dataset.frames if f.image_info.image_id == fid), None)
                if frame and frame.detection:
                    frame.detection.excluded_corner_indices = [
                        i for i in frame.detection.excluded_corner_indices if i not in indices
                    ]
            iterations -= 1
            break

        last_good_result = new_result

    corner_outlier_result = CornerOutlierResult(
        threshold_used=last_threshold,
        removed_corners=removed_all,
        rms_before=rms_before,
        rms_after=last_good_result.rms_error,
        iterations=iterations,
        max_iterations=max_iterations,
        p95_before=p95_before,
        p95_after=last_good_result.residual_stats.p95 if last_good_result.residual_stats else None,
        camera_matrix_before=camera_matrix_before,
        camera_matrix_after=last_good_result.camera_matrix.copy() if last_good_result.camera_matrix is not None else None,
        distortion_before=distortion_before,
        distortion_after=last_good_result.distortion.copy() if last_good_result.distortion is not None else None,
    )
    return last_good_result, corner_outlier_result


def format_corner_outlier_summary(result: CornerOutlierResult) -> str:
    if not result.removed_corners:
        return "이상치로 판단된 코너가 없습니다."

    before = f"{result.rms_before:.3f}" if result.rms_before is not None else "N/A"
    after = f"{result.rms_after:.3f}" if result.rms_after is not None else "N/A"
    lines = [
        f"제외된 코너 (총 {result.total_corners_removed}개, {result.iterations}회 반복, "
        f"threshold={result.threshold_used:.3f}px):",
    ]
    for frame_id, indices in result.removed_corners.items():
        lines.append(f"  - {frame_id}: 코너 인덱스 {indices}")
    lines.append(f"RMS {before}px -> {after}px 로 개선")
    return "\n".join(lines)

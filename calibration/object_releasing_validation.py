"""
camera_calibrator.calibration.object_releasing_validation
==============================================================

Object-Releasing 전용 Hold-out Validation + Standard Brown-Conrady와의 공정 비교.

핵심 원칙(사용자 요구사항, 반드시 지켜야 함):
    Train: K, D, Refined Target Geometry를 모두 확정한다.
    Test:  K/D/Refined Geometry를 전부 고정한 채 pose(solvePnP)만 다시 구하고,
           reprojection error만 계산한다. calibrateCameraRO/-Extended를
           Test 프레임에 대해 다시 호출하는 일은 절대 없다.

calibration/validation.py의 Standard Hold-out(_evaluate_on_test)을 그대로
재사용하지 않는 이유: 그쪽은 각 test 프레임의 nominal object_points를 쓰지만,
Object-Releasing은 Train에서 나온 refined_object_points(물리적 타겟 하나에 대한
공유 형상)를 모든 test 프레임에 공통으로 써야 하기 때문에 계산 경로가 다르다.
split_train_test/_subset_dataset 등 분할/평가 인프라는 그대로 재사용한다.

Object-Releasing은 full-board 검출만 지원하므로(Checkerboard/Circle Grid),
분할 이전에 collect_object_releasing_inputs()로 eligible(=full-board) 프레임만
추려낸다 - Standard Hold-out의 split_train_test()가 쓰는 "usable" 기준(코너
4개 이상)은 이 요구사항을 반영하지 않는다.
"""

from __future__ import annotations

import cv2
import numpy as np

from calibration.models.common import MIN_FRAMES_REQUIRED, fmt_optional
from calibration.models.object_releasing import (
    calibrate_object_releasing_brown_conrady,
    collect_object_releasing_inputs,
)
from calibration.residual_stats import compute_residual_stats
from calibration.types import (
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    Dataset,
    ObjectReleasingValidationResult,
    PatternConfig,
    StandardVsObjectReleasingComparison,
)
from calibration.validation import (
    _subset_dataset,
    refit_on_train_split,
    split_train_test,
    validate_holdout,
)

# MIN_FRAMES_REQUIRED(=3)은 Object-Releasing 캘리브레이션 자체가 요구하는
# 최소 프레임 수(train 쪽). Hold-out은 여기에 test 쪽 최소 2장을 더 요구한다 -
# test가 1장뿐이면 P95/P99 같은 percentile이 사실상 의미가 없다. 사용자
# 스펙의 예시("Full-board frames < 5")와도 일치한다.
MIN_FULL_BOARD_FRAMES_FOR_HOLDOUT = MIN_FRAMES_REQUIRED + 2

_INTRINSIC_KEYS = ("fx", "fy", "cx", "cy")
_DISTORTION_LABELS = ("k1", "k2", "p1", "p2", "k3")
# Target geometry refinement이 이 비율(명목 spacing 대비)을 넘으면 경고 -
# 임의의 hard threshold이므로 여기 명시하고 조정 가능하게 둔다.
_LARGE_REFINEMENT_RATIO = 0.02


def _eligible_full_board_frame_ids(
    dataset: Dataset, pattern_config: PatternConfig
) -> tuple[list[str], list[str], dict[str, str]]:
    """collect_object_releasing_inputs()의 diagnostics를 그대로 재사용해
    "full-board로 인정된 프레임"과 "제외된 프레임 + 이유"를 나눈다 - 이유 없이
    조용히 skip하는 것을 막기 위해 이유를 항상 함께 들고 다닌다.
    """
    accepted_frames, _obj, _img, diagnostics = collect_object_releasing_inputs(dataset, pattern_config)
    eligible_ids = [f.image_info.image_id for f in accepted_frames]
    excluded_ids = [d["image_id"] for d in diagnostics if not d.get("accepted")]
    excluded_reasons = {
        d["image_id"]: (d.get("reject_reason") or "not a full-board detection")
        for d in diagnostics
        if not d.get("accepted")
    }
    return eligible_ids, excluded_ids, excluded_reasons


def _run_object_releasing_train_test(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    train_ids: list[str],
    test_ids: list[str],
) -> tuple[CalibrationResult | None, ObjectReleasingValidationResult]:
    """지정된 train/test 분할로 Object-Releasing Hold-out을 수행한다.

    Train에서만 calibrate_object_releasing_brown_conrady()를 (딱 한 번) 호출해
    K/D/refined_object_points를 확정하고, Test는 그 셋을 고정한 채 solvePnP +
    projectPoints로만 평가한다. 반환하는 CalibrationResult는 P1-B(Standard와의
    비교)에서 fx/fy/... 파라미터를 표시하는 데 재사용하기 위함이고, RO 캘리브레이션을
    두 번 돌리지 않기 위해 여기서 만든 것을 그대로 돌려준다.
    """
    if len(train_ids) < MIN_FRAMES_REQUIRED:
        return None, ObjectReleasingValidationResult(
            success=False,
            train_frame_ids=train_ids,
            test_frame_ids=test_ids,
            error_message=(
                f"Train 프레임이 {len(train_ids)}장뿐입니다 (최소 {MIN_FRAMES_REQUIRED}장 필요)."
            ),
        )
    if not test_ids:
        return None, ObjectReleasingValidationResult(
            success=False,
            train_frame_ids=train_ids,
            test_frame_ids=test_ids,
            error_message="Test 프레임이 없어 Object-Releasing Hold-out을 수행하지 못했습니다.",
        )

    train_dataset = _subset_dataset(dataset, train_ids)
    train_result = calibrate_object_releasing_brown_conrady(train_dataset, camera_config, pattern_config)

    if (
        not train_result.success
        or train_result.camera_matrix is None
        or train_result.distortion is None
        or train_result.refined_object_points is None
    ):
        return train_result, ObjectReleasingValidationResult(
            success=False,
            train_frame_ids=train_ids,
            test_frame_ids=test_ids,
            error_message=f"Train 캘리브레이션 실패: {train_result.error_message}",
        )

    camera_matrix = train_result.camera_matrix
    distortion = train_result.distortion
    refined_points = np.asarray(train_result.refined_object_points, dtype=np.float64).reshape(-1, 1, 3)
    n_expected = len(refined_points)

    # Test 프레임들의 image point를 Train과 동일한 canonical(ID 정렬) 순서로
    # 얻는다. 이 순서는 pattern_config에서만 결정되므로(어떤 frame 부분집합을
    # 넣든 동일) refined_points와 index가 그대로 대응한다.
    test_dataset = _subset_dataset(dataset, test_ids)
    test_frames, _test_obj, test_image_points, test_diagnostics = collect_object_releasing_inputs(
        test_dataset, pattern_config
    )
    canonical_image_points = {
        frame.image_info.image_id: img for frame, img in zip(test_frames, test_image_points)
    }

    failed_test_frame_ids: list[str] = []
    failed_test_reasons: dict[str, str] = {}
    # eligible pool 단계에서 이미 걸러졌어야 하지만, 방어적으로 한 번 더
    # 확인하고 - 실패하면 이유와 함께 기록한다(조용히 skip하지 않음).
    for diag in test_diagnostics:
        if not diag.get("accepted"):
            frame_id = diag["image_id"]
            failed_test_frame_ids.append(frame_id)
            failed_test_reasons[frame_id] = diag.get("reject_reason") or "full-board re-check failed"

    per_frame_rms: dict[str, float] = {}
    point_errors: list[float] = []
    for frame_id in test_ids:
        img = canonical_image_points.get(frame_id)
        if img is None:
            continue  # 이미 위에서 failed_test_frame_ids에 기록됨

        img = np.asarray(img, dtype=np.float64).reshape(-1, 1, 2)
        if len(img) != n_expected:
            failed_test_frame_ids.append(frame_id)
            failed_test_reasons[frame_id] = (
                f"point count mismatch: got {len(img)}, expected {n_expected}"
            )
            continue

        try:
            ok, rvec, tvec = cv2.solvePnP(refined_points, img, camera_matrix, distortion)
            if not ok:
                raise cv2.error("solvePnP returned False")
            projected, _ = cv2.projectPoints(refined_points, rvec, tvec, camera_matrix, distortion)
        except cv2.error:
            failed_test_frame_ids.append(frame_id)
            failed_test_reasons[frame_id] = "solvePnP/projectPoints failed to converge"
            continue

        projected = projected.reshape(-1, 2)
        detected = img.reshape(-1, 2)
        diff = detected - projected
        per_point = np.hypot(diff[:, 0], diff[:, 1])
        rms = float(np.sqrt(np.mean(per_point ** 2)))
        per_frame_rms[frame_id] = rms
        point_errors.extend(per_point.tolist())

    if not per_frame_rms:
        return train_result, ObjectReleasingValidationResult(
            success=False,
            train_frame_ids=train_ids,
            test_frame_ids=test_ids,
            train_rms=train_result.rms_error,
            target_geometry_refinement=train_result.target_geometry_refinement,
            failed_test_frame_ids=failed_test_frame_ids,
            failed_test_reasons=failed_test_reasons,
            error_message="모든 test 프레임에서 pose 추정(solvePnP)이 실패했습니다.",
        )

    test_rms = float(np.sqrt(np.mean(np.array(list(per_frame_rms.values())) ** 2)))
    test_residual_stats = compute_residual_stats(point_errors)

    return train_result, ObjectReleasingValidationResult(
        success=True,
        train_frame_ids=train_ids,
        test_frame_ids=test_ids,
        train_rms=train_result.rms_error,
        test_rms=test_rms,
        test_residual_stats=test_residual_stats,
        target_geometry_refinement=train_result.target_geometry_refinement,
        failed_test_frame_ids=failed_test_frame_ids,
        failed_test_reasons=failed_test_reasons,
    )


def validate_object_releasing_holdout(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    *,
    test_ratio: float = 0.25,
    seed: int = 42,
) -> ObjectReleasingValidationResult:
    """Object-Releasing 전용 Hold-out Validation.

    Full-board 프레임만 모아 stratified split을 한 번 만들고, Train에서만
    calibrate_object_releasing_brown_conrady()를 호출해 K/D/refined geometry를
    확정한다. Test는 그 값을 고정한 채 pose만 다시 구해 평가한다
    (_run_object_releasing_train_test 참고).
    """
    eligible_ids, excluded_ids, excluded_reasons = _eligible_full_board_frame_ids(dataset, pattern_config)

    if len(eligible_ids) < MIN_FULL_BOARD_FRAMES_FOR_HOLDOUT:
        return ObjectReleasingValidationResult(
            success=False,
            excluded_frame_ids=excluded_ids,
            excluded_reasons=excluded_reasons,
            error_message=(
                "Insufficient full-board frames for hold-out validation "
                f"({len(eligible_ids)} < {MIN_FULL_BOARD_FRAMES_FOR_HOLDOUT})."
            ),
        )

    eligible_subset = _subset_dataset(dataset, eligible_ids)
    train_ids, test_ids = split_train_test(eligible_subset, camera_config, test_ratio, seed)

    _train_result, validation = _run_object_releasing_train_test(
        dataset, camera_config, pattern_config, train_ids, test_ids
    )
    validation.excluded_frame_ids = excluded_ids
    validation.excluded_reasons = excluded_reasons
    return validation


def _intrinsics_delta(standard: CalibrationResult, ro: CalibrationResult) -> dict[str, float]:
    """ro - standard, fx/fy/cx/cy + k1/k2/p1/p2/k3. 두 결과 모두 Brown-Conrady
    (5계수) 레이아웃을 쓰므로 인덱스가 그대로 대응한다.
    """
    delta: dict[str, float] = {}
    if standard.camera_matrix is not None and ro.camera_matrix is not None:
        sk = np.asarray(standard.camera_matrix, dtype=np.float64).reshape(3, 3)
        rk = np.asarray(ro.camera_matrix, dtype=np.float64).reshape(3, 3)
        delta["fx"] = float(rk[0, 0] - sk[0, 0])
        delta["fy"] = float(rk[1, 1] - sk[1, 1])
        delta["cx"] = float(rk[0, 2] - sk[0, 2])
        delta["cy"] = float(rk[1, 2] - sk[1, 2])
    if standard.distortion is not None and ro.distortion is not None:
        sd = np.asarray(standard.distortion, dtype=np.float64).reshape(-1)
        rd = np.asarray(ro.distortion, dtype=np.float64).reshape(-1)
        for i, label in enumerate(_DISTORTION_LABELS):
            if i < len(sd) and i < len(rd):
                delta[label] = float(rd[i] - sd[i])
    return delta


def _build_warnings(
    standard_result: CalibrationResult | None,
    standard_validation,
    ro_result: CalibrationResult | None,
    ro_validation: ObjectReleasingValidationResult,
    pattern_config: PatternConfig,
) -> list[str]:
    """사실만 기술하는 경고. "RO가 더 정확하다" 같은 자동 판정 문구는 절대
    만들지 않는다 - 사용자 스펙의 명시적 금지 사항.
    """
    warnings: list[str] = []

    if (
        standard_result is not None and standard_result.rms_error
        and ro_result is not None and ro_result.rms_error is not None
        and standard_validation is not None and standard_validation.test_rms is not None
        and ro_validation.test_rms is not None
    ):
        train_improve = (standard_result.rms_error - ro_result.rms_error) / standard_result.rms_error
        holdout_improve = 0.0
        if standard_validation.test_rms:
            holdout_improve = (
                (standard_validation.test_rms - ro_validation.test_rms) / standard_validation.test_rms
            )
        if train_improve > 0.10 and holdout_improve <= 0.0:
            warnings.append(
                "Possible overfitting: Object-Releasing improved train RMS by "
                f"{train_improve * 100:.1f}% but did not improve hold-out RMSE."
            )

    geom = (ro_result.target_geometry_refinement if ro_result else None) or {}
    max_disp = geom.get("max_displacement")
    if max_disp is not None and pattern_config.square_size:
        threshold = _LARGE_REFINEMENT_RATIO * pattern_config.square_size
        if max_disp > threshold:
            warnings.append(
                f"Large target geometry refinement detected (max={max_disp:.5g}, "
                f"> {_LARGE_REFINEMENT_RATIO * 100:.0f}% of nominal spacing). "
                "Check target dimensions, rigidity, and detection quality."
            )

    return warnings


def compare_standard_vs_object_releasing_brown(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    *,
    test_ratio: float = 0.25,
    seed: int = 42,
) -> StandardVsObjectReleasingComparison:
    """Standard Brown-Conrady와 Object-Releasing Brown-Conrady를 "같은
    full-board eligible 데이터셋" + "같은 train/test 분할"로 공정하게 비교한다.

    두 arm에 서로 다른 데이터셋/분할을 쓰는 비교는 절대 만들지 않는다 - 그런
    비교 결과는 UI/CLI 어디에도 노출하지 않는 것이 사용자 스펙의 명시적 요구다.
    """
    eligible_ids, excluded_ids, excluded_reasons = _eligible_full_board_frame_ids(dataset, pattern_config)

    if len(eligible_ids) < MIN_FULL_BOARD_FRAMES_FOR_HOLDOUT:
        return StandardVsObjectReleasingComparison(
            success=False,
            eligible_frame_ids=eligible_ids,
            error_message=(
                "Insufficient full-board frames for a fair Standard vs Object-Releasing "
                f"comparison ({len(eligible_ids)} < {MIN_FULL_BOARD_FRAMES_FOR_HOLDOUT})."
            ),
        )

    eligible_subset = _subset_dataset(dataset, eligible_ids)
    train_ids, test_ids = split_train_test(eligible_subset, camera_config, test_ratio, seed)

    if len(train_ids) < MIN_FRAMES_REQUIRED or not test_ids:
        return StandardVsObjectReleasingComparison(
            success=False,
            eligible_frame_ids=eligible_ids,
            train_frame_ids=train_ids,
            test_frame_ids=test_ids,
            error_message="Insufficient frames after split for a fair comparison.",
        )

    # Standard arm: 기존, 이미 테스트된 validate_holdout()/refit_on_train_split()을
    # 그대로 재사용 - 같은 train_ids/test_ids를 넘기는 것만으로 "같은 분할" 요구를
    # 충족한다.
    standard_validation = validate_holdout(
        dataset, camera_config, pattern_config, CameraModelType.BROWN_CONRADY, train_ids, test_ids
    )
    standard_result = None
    if standard_validation.success:
        standard_result = refit_on_train_split(dataset, camera_config, CameraModelType.BROWN_CONRADY, train_ids)

    # RO arm: 같은 train_ids/test_ids로 P1-A와 동일한 helper를 사용.
    ro_result, ro_validation = _run_object_releasing_train_test(
        dataset, camera_config, pattern_config, train_ids, test_ids
    )

    intrinsics_delta: dict[str, float] = {}
    if standard_result is not None and standard_result.success and ro_result is not None and ro_result.success:
        intrinsics_delta = _intrinsics_delta(standard_result, ro_result)

    warnings = _build_warnings(standard_result, standard_validation, ro_result, ro_validation, pattern_config)

    success = bool(standard_validation.success and ro_validation.success)
    error_message = None
    if not success:
        parts = []
        if not standard_validation.success:
            parts.append(f"Standard: {standard_validation.error_message}")
        if not ro_validation.success:
            parts.append(f"Object-Releasing: {ro_validation.error_message}")
        error_message = "; ".join(parts) or "Comparison failed."

    return StandardVsObjectReleasingComparison(
        success=success,
        error_message=error_message,
        eligible_frame_ids=eligible_ids,
        train_frame_ids=train_ids,
        test_frame_ids=test_ids,
        standard_result=standard_result,
        standard_validation=standard_validation,
        object_releasing_result=ro_result,
        object_releasing_validation=ro_validation,
        intrinsics_delta=intrinsics_delta,
        warnings=warnings,
    )


def format_standard_vs_object_releasing_table(comparison: StandardVsObjectReleasingComparison) -> str:
    """CLI/로그용 2열 비교표. compare.py::format_comparison_table과 같은 스타일."""
    if not comparison.success:
        return f"Standard vs Object-Releasing comparison unavailable: {comparison.error_message}"

    labels = ["Standard Brown", "Object-Releasing"]
    col_w = max(18, max(len(l) for l in labels) + 2)

    def row(name: str, values: list[str]) -> str:
        return f"{name:<24}" + "".join(f"{v:>{col_w}}" for v in values)

    sr, sv = comparison.standard_result, comparison.standard_validation
    rr, rv = comparison.object_releasing_result, comparison.object_releasing_validation

    lines = [
        row("", labels),
        row("Train RMS", [fmt_optional(sr.rms_error if sr else None), fmt_optional(rr.rms_error if rr else None)]),
        row("Hold-out RMSE", [fmt_optional(sv.test_rms if sv else None), fmt_optional(rv.test_rms if rv else None)]),
        row("Median", [
            fmt_optional(sv.test_residual_stats.median if sv and sv.test_residual_stats else None),
            fmt_optional(rv.test_residual_stats.median if rv and rv.test_residual_stats else None),
        ]),
        row("P95", [
            fmt_optional(sv.test_residual_stats.p95 if sv and sv.test_residual_stats else None),
            fmt_optional(rv.test_residual_stats.p95 if rv and rv.test_residual_stats else None),
        ]),
        row("P99", [
            fmt_optional(sv.test_residual_stats.p99 if sv and sv.test_residual_stats else None),
            fmt_optional(rv.test_residual_stats.p99 if rv and rv.test_residual_stats else None),
        ]),
        row("Max", [
            fmt_optional(sv.test_residual_stats.max if sv and sv.test_residual_stats else None),
            fmt_optional(rv.test_residual_stats.max if rv and rv.test_residual_stats else None),
        ]),
    ]

    for key in _INTRINSIC_KEYS + _DISTORTION_LABELS:
        s_matrix = None
        if key in _INTRINSIC_KEYS and sr and sr.camera_matrix is not None:
            idx = {"fx": (0, 0), "fy": (1, 1), "cx": (0, 2), "cy": (1, 2)}[key]
            s_matrix = float(np.asarray(sr.camera_matrix).reshape(3, 3)[idx])
        r_matrix = None
        if key in _INTRINSIC_KEYS and rr and rr.camera_matrix is not None:
            idx = {"fx": (0, 0), "fy": (1, 1), "cx": (0, 2), "cy": (1, 2)}[key]
            r_matrix = float(np.asarray(rr.camera_matrix).reshape(3, 3)[idx])
        if key in _DISTORTION_LABELS:
            i = _DISTORTION_LABELS.index(key)
            if sr and sr.distortion is not None and i < len(np.asarray(sr.distortion).reshape(-1)):
                s_matrix = float(np.asarray(sr.distortion).reshape(-1)[i])
            if rr and rr.distortion is not None and i < len(np.asarray(rr.distortion).reshape(-1)):
                r_matrix = float(np.asarray(rr.distortion).reshape(-1)[i])
        lines.append(row(key, [fmt_optional(s_matrix), fmt_optional(r_matrix)]))

    geom = (rr.target_geometry_refinement if rr else None) or {}
    if geom:
        lines.append("")
        lines.append("Target geometry refinement (Object-Releasing):")
        lines.append(
            f"  mean={fmt_optional(geom.get('mean_displacement'))}  "
            f"median={fmt_optional(geom.get('median_displacement'))}  "
            f"p95={fmt_optional(geom.get('p95_displacement'))}  "
            f"max={fmt_optional(geom.get('max_displacement'))}"
        )

    if comparison.warnings:
        lines.append("")
        for w in comparison.warnings:
            lines.append(f"  [warning] {w}")

    return "\n".join(lines)

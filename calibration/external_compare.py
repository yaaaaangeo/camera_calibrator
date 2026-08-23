"""
camera_calibrator.calibration.external_compare
====================================================

사용자 요청 - "예전에 다른 사람/다른 툴로 같은 카메라를 캘리브레이션한
값"과 "이 툴로 지금 구한 값" 중 뭐가 더 정확한지, 주장이 아니라 누구나
확인 가능한 정량적 근거로 비교하는 기능.

--- 왜 "그냥 두 RMS 숫자를 나란히 놓고 비교"하면 안 되는가 ---

"내 RMS"는 보통 내 데이터셋으로 학습한 뒤 그 데이터셋에서 측정한 값이다.
"예전 파라미터"는 완전히 다른 촬영 세션에서 나왔으니, 애초에 "내 데이터셋
전체"가 그쪽 입장에서는 전부 처음 보는 데이터다. 그래서 "내 파라미터를
내 데이터로 잰 값" vs "예전 파라미터를 내 데이터에 그대로 적용해 잰 값"을
그냥 비교하면, 내가 유리한 조건에서 이긴 것처럼 보일 수 있다 - 이러면
"그야 네가 만든 툴이니까 네 편을 들겠지"라는 반박을 피할 수 없다.

그래서 이 모듈은 두 파라미터 모두에게 완전히 동일한 조건을 강제한다:
    1. 비교는 항상 "내가 캘리브레이션 학습에 전혀 쓰지 않은 프레임"
       (validate_holdout이 이미 떼어둔 test 분할)에서만 이뤄진다.
       - 외부 파라미터에게는 어차피 전체가 안 본 데이터라 손해볼 게 없다.
       - 내 파라미터도 같은 test 분할로 "다시 학습한" 버전
         (calibration/validation.py의 refit_on_train_split, 곧
         ValidationResult.test_rms를 만들어낸 바로 그 절차)을 써서, 이
         test 프레임을 학습에 훔쳐본 적이 없게 만든다.
    2. 두 쪽 다 intrinsic을 이 데이터로 다시 최적화하지 않는다 - solvePnP로
       pose만 새로 구하고 그대로 재투영한다(validation.py의 핵심 원칙과
       동일). 그래야 "파라미터 자체가 이 카메라를 얼마나 잘 설명하는지"만
       순수하게 잰다.
    3. 두 쪽에 동일한 계산 함수(_test_reprojection_errors,
       compute_regional_error, compute_straightness_residual)를 그대로
       적용한다 - "계산 방식이 달라서 결과가 갈렸다"는 반박 자체가
       원천적으로 불가능하게.

RMS 숫자 하나로 승패를 가르지 않는다("RMS가 가장 낮은 모델 = 정답은 절대
금지" 원칙을 두 파라미터 비교에도 동일하게 적용): Test RMS / Edge RMS(외곽)
/ Straightness Residual(직선성) 3개 독립 지표 + 프레임별 승-패 개수까지
전부 보여주고, 지표가 엇갈리면 엇갈린다고 정직하게 말한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from calibration.types import (
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    Dataset,
    PatternConfig,
    ValidationResult,
)
from calibration.validation import (
    _subset_dataset,
    _test_reprojection_errors,
    refit_on_train_split,
)
from calibration.models.common import compute_regional_error, regional_edge_average
from calibration.straightness import compute_straightness_residual

_MODEL_LABELS = {
    CameraModelType.PINHOLE: "Pinhole",
    CameraModelType.EXTENDED_PINHOLE: "Extended Pinhole",
    CameraModelType.FISHEYE: "Fisheye",
}


@dataclass
class ExternalCameraParams:
    """비교 대상이 되는 "다른 곳에서 구한" 카메라 파라미터.
    이 툴이 만든 결과가 아니어도 된다(수기 입력, 다른 소프트웨어의 OpenCV
    YAML 등) - label/source_note에 출처를 남겨 화면/리포트에서 항상
    "이건 어디서 온 값인지" 구분되게 한다.
    """
    label: str
    model_name: CameraModelType
    camera_matrix: np.ndarray   # 3x3
    distortion: np.ndarray
    source_note: str = ""       # 예: "2025-03 A업체 캘리브레이션, OpenCV YAML"


@dataclass
class ComparisonSide:
    """한쪽(내 결과 또는 외부 결과)을 동일 조건(같은 test 프레임, 같은
    solvePnP-only 절차)으로 재평가한 결과."""
    label: str
    model_name: CameraModelType | None = None
    camera_matrix: np.ndarray | None = None
    distortion: np.ndarray | None = None
    test_rms: float | None = None
    edge_rms: float | None = None
    straightness_residual: float | None = None
    per_frame_error: dict[str, float] = field(default_factory=dict)
    failed_frame_ids: list[str] = field(default_factory=list)
    success: bool = False
    error_message: str | None = None


@dataclass
class ExternalComparisonResult:
    mine: ComparisonSide
    external: ComparisonSide
    num_common_frames: int = 0
    mine_win_count: int = 0
    external_win_count: int = 0
    tie_count: int = 0
    verdict: str = ""
    caveats: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 한쪽 평가
# ---------------------------------------------------------------------------

def _evaluate_side(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    test_ids: list[str],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    model: CameraModelType,
    label: str,
) -> ComparisonSide:
    subset = _subset_dataset(dataset, test_ids)
    frames = subset.enabled_frames
    if not frames:
        return ComparisonSide(
            label=label, model_name=model, camera_matrix=camera_matrix, distortion=distortion,
            success=False, error_message="비교할 프레임이 없습니다.",
        )

    per_frame_error, failed, _point_errors = _test_reprojection_errors(frames, camera_matrix, distortion, model)
    if not per_frame_error:
        return ComparisonSide(
            label=label, model_name=model, camera_matrix=camera_matrix, distortion=distortion,
            success=False,
            error_message=(
                "모든 프레임에서 pose 추정(solvePnP)이 실패했습니다 - "
                "카메라 모델 종류(Pinhole/Extended/Fisheye)나 파라미터가 "
                "이 촬영과 맞는지 다시 확인해 주세요."
            ),
            failed_frame_ids=failed,
        )

    test_rms = float(np.sqrt(np.mean(np.array(list(per_frame_error.values())) ** 2)))
    image_size = (camera_config.width, camera_config.height)
    scored_frames = [f for f in frames if f.image_info.image_id in per_frame_error]
    regional = compute_regional_error(scored_frames, per_frame_error, image_size)
    edge_rms = regional_edge_average(regional)
    straightness, _n_lines = compute_straightness_residual(
        scored_frames, pattern_config, camera_matrix, distortion, model,
    )

    return ComparisonSide(
        label=label, model_name=model, camera_matrix=camera_matrix, distortion=distortion,
        test_rms=test_rms, edge_rms=edge_rms, straightness_residual=straightness,
        per_frame_error=per_frame_error, failed_frame_ids=failed, success=True,
    )


# ---------------------------------------------------------------------------
# 메인 함수
# ---------------------------------------------------------------------------

def compare_with_external_params(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    my_model: CameraModelType,
    my_validation: ValidationResult,
    external: ExternalCameraParams,
    use_rational_model: bool = False,
) -> ExternalComparisonResult:
    """my_validation이 이미 확보해 둔 "학습에 전혀 안 쓰인" test 프레임
    집합에서, 내 파라미터(같은 test 분할로 다시 학습한 것 - my_validation.
    test_rms를 만들어낸 바로 그 절차)와 외부 파라미터를 완전히 동일한
    방식으로 재평가해 비교한다.
    """
    caveats: list[str] = []
    my_label = f"내 결과 ({_MODEL_LABELS.get(my_model, my_model)})"

    test_ids = my_validation.test_frame_ids
    if not test_ids:
        return ExternalComparisonResult(
            mine=ComparisonSide(label=my_label, success=False, error_message="Hold-out 검증에 test 프레임이 없습니다."),
            external=ComparisonSide(label=external.label, success=False, error_message="비교 기준(test 프레임)이 없습니다."),
            verdict=(
                "데이터셋이 너무 작아 Hold-out에 쓸 test 프레임이 없습니다. "
                "이미지를 더 모아서(최소 수십 장 권장) 다시 시도해 주세요."
            ),
        )

    mine_train_result = refit_on_train_split(
        dataset, camera_config, my_model, my_validation.train_frame_ids,
        use_rational_model=use_rational_model,
    )
    if not mine_train_result.success:
        return ExternalComparisonResult(
            mine=ComparisonSide(label=my_label, success=False, error_message=mine_train_result.error_message),
            external=ComparisonSide(label=external.label, success=False, error_message="비교할 수 없습니다."),
            verdict=f"내 파라미터를 test 분할 기준으로 다시 학습하지 못했습니다: {mine_train_result.error_message}",
        )

    mine_side = _evaluate_side(
        dataset, camera_config, pattern_config, test_ids,
        mine_train_result.camera_matrix, mine_train_result.distortion, my_model, my_label,
    )
    external_side = _evaluate_side(
        dataset, camera_config, pattern_config, test_ids,
        external.camera_matrix, external.distortion, external.model_name, external.label,
    )

    # 정합성 자체 점검 - 방금 재학습한 test_rms는 my_validation.test_rms와
    # (거의) 같아야 정상이다. 조용히 다르면 안 되고, 다르면 그 자체를
    # 사용자에게 밝힌다 (원인 예: 그 사이에 데이터셋/설정이 바뀌었음).
    if (
        mine_side.success and my_validation.test_rms is not None
        and abs(mine_side.test_rms - my_validation.test_rms) > 1e-2
    ):
        caveats.append(
            "내부 정합성 확인: 방금 재계산한 Test RMS"
            f"({mine_side.test_rms:.3f}px)가 이전 Hold-out 결과"
            f"({my_validation.test_rms:.3f}px)와 다릅니다 - 그 사이에 데이터셋/설정이 "
            "바뀌었을 수 있으니 [캘리브레이션 실행]을 다시 눌러 최신 상태로 비교해 보세요."
        )

    if external.source_note:
        caveats.append(f"'{external.label}' 출처: {external.source_note}")

    mine_win = external_win = tie = 0
    common_ids = set(mine_side.per_frame_error) & set(external_side.per_frame_error)
    for fid in common_ids:
        m, e = mine_side.per_frame_error[fid], external_side.per_frame_error[fid]
        if abs(m - e) < 1e-9:
            tie += 1
        elif m < e:
            mine_win += 1
        else:
            external_win += 1

    verdict = _build_verdict(mine_side, external_side, mine_win, external_win, len(common_ids))

    return ExternalComparisonResult(
        mine=mine_side,
        external=external_side,
        num_common_frames=len(common_ids),
        mine_win_count=mine_win,
        external_win_count=external_win,
        tie_count=tie,
        verdict=verdict,
        caveats=caveats,
    )


# ---------------------------------------------------------------------------
# 한 줄 평 (verdict)
# ---------------------------------------------------------------------------

def _build_verdict(
    mine: ComparisonSide, external: ComparisonSide,
    mine_win: int, external_win: int, n_common: int,
) -> str:
    if not mine.success or not external.success:
        broken = mine if not mine.success else external
        return f"[{broken.label}] 비교할 수 없습니다: {broken.error_message}"

    # 지표 3개를 독립적으로 비교 - 하나만 보고 판정하지 않는다
    # ("RMS가 가장 낮은 모델 = 정답 절대 금지" 원칙을 여기도 동일 적용).
    metrics = [
        ("Test RMS", mine.test_rms, external.test_rms),
        ("Edge RMS(외곽)", mine.edge_rms, external.edge_rms),
        ("Straightness(직선성)", mine.straightness_residual, external.straightness_residual),
    ]
    mine_wins, external_wins, compared = 0, 0, 0
    for _name, mv, ev in metrics:
        if mv is None or ev is None or abs(mv - ev) < 1e-9:
            continue
        compared += 1
        if mv < ev:
            mine_wins += 1
        else:
            external_wins += 1

    if compared == 0:
        headline = "두 파라미터의 지표를 비교할 수 없습니다 (유효한 값이 부족합니다)."
    elif mine_wins == compared:
        headline = f"'{mine.label}'가 비교 가능한 지표 {compared}개 전부에서 더 정확합니다."
    elif external_wins == compared:
        headline = f"'{external.label}'가 비교 가능한 지표 {compared}개 전부에서 더 정확합니다."
    elif mine_wins > external_wins:
        headline = (
            f"'{mine.label}'가 {mine_wins}/{compared}개 지표에서 우세하지만 일부는 엇갈립니다 "
            "- 아래 표에서 어느 지표가 갈렸는지 확인하세요."
        )
    elif external_wins > mine_wins:
        headline = (
            f"'{external.label}'가 {external_wins}/{compared}개 지표에서 우세하지만 일부는 엇갈립니다 "
            "- 아래 표에서 어느 지표가 갈렸는지 확인하세요."
        )
    else:
        headline = "두 파라미터의 지표가 팽팽하게 엇갈려 어느 한쪽이 명확히 낫다고 보기 어렵습니다."

    frame_total = mine_win + external_win
    if frame_total > 0 and mine_win != external_win:
        winner_label = mine.label if mine_win > external_win else external.label
        headline += (
            f" (프레임별로 보면 공통 {n_common}장 중 {winner_label}가 더 낮은 오차를 낸 프레임이 "
            f"{max(mine_win, external_win)}장으로 더 많습니다.)"
        )

    return headline

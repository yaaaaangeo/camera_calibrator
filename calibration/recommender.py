"""
camera_calibrator.calibration.recommender
=============================================

설계 문서 8번 - 모델 자동 추천 로직.

    "RMS가 가장 낮은 모델 = 정답은 절대 금지."
    Score = w1*E_train + w2*E_test + w3*E_edge + w4*E_line + w5*P

이 모듈은 의도적으로 validation_results(Hold-out, 5단계)를 필수 인자로
요구한다 - Train RMS만 갖고 추천하는 걸 API 차원에서 막기 위해서다.
straightness(E_line)는 V2 기능이라 아직 없으므로, 있으면 쓰고 없으면
그 항목은 점수에 영향을 주지 않게(중립 0점) 처리한다.

추천은 "정답 지정"이 아니라 "근거 제시"다 (설계 문서 8번 "추천과 선택의 분리").
그래서 이 모듈의 결과는 항상 ModelScore 리스트 + 사람이 읽을 수 있는 근거
문장이며, 최종 모델을 실제로 쓸지는 여전히 호출하는 쪽(UI/사용자)의 몫이다.
"""

from __future__ import annotations

from calibration.types import (
    CalibrationResult,
    CameraModelType,
    ModelScore,
    ModelScoreWeights,
    ValidationResult,
)
from calibration.models.common import regional_edge_average

# 모델별 명목 자유도(free parameter) 개수. 실제 CalibrationResult.distortion
# 배열 길이로 역산하지 않고 여기서 명시하는 이유: Pinhole은 distortion 배열이
# [0,0,0,0,0]으로 채워져 있어도 "고정된 0"이지 "추정된 0"이 아니므로 자유도는 0이다.
_FREE_PARAMS = {
    CameraModelType.PINHOLE: 0,
    CameraModelType.EXTENDED_PINHOLE: 5,   # k1,k2,p1,p2,k3 (rational이면 아래서 8로 대체)
    CameraModelType.FISHEYE: 4,             # k1~k4
}
_FREE_PARAMS_RATIONAL = 8  # k1~k6, p1, p2


def _complexity_counts(use_rational_model: bool) -> dict[CameraModelType, int]:
    counts = dict(_FREE_PARAMS)
    if use_rational_model:
        counts[CameraModelType.EXTENDED_PINHOLE] = _FREE_PARAMS_RATIONAL
    return counts


def _normalize(values: dict[CameraModelType, float | None]) -> dict[CameraModelType, float]:
    """0(가장 좋음)~1(가장 나쁨)로 min-max 정규화.

    - 값이 전부 None(해당 지표를 아무 모델도 갖고 있지 않음, 예: straightness
      V2 미구현)이면 전부 0.0을 줘서 그 항목이 점수에 영향을 주지 않게 한다.
    - 일부만 None(그 모델만 실패했거나 데이터가 없음)이면 그 모델에는 1.0(최악)을
      줘서 불이익을 준다 - "몰라서 좋아 보이는" 상황을 방지.
    """
    valid = {k: v for k, v in values.items() if v is not None}
    if not valid:
        return {k: 0.0 for k in values}

    lo, hi = min(valid.values()), max(valid.values())
    rng = hi - lo
    out: dict[CameraModelType, float] = {}
    for k, v in values.items():
        if v is None:
            out[k] = 1.0
        else:
            out[k] = (v - lo) / rng if rng > 0 else 0.0
    return out


def compute_model_scores(
    calibration_results: dict[CameraModelType, CalibrationResult],
    validation_results: dict[CameraModelType, ValidationResult],
    weights: ModelScoreWeights = ModelScoreWeights(),
    use_rational_model: bool = False,
) -> list[ModelScore]:
    """세 모델의 Score를 계산하고, 가장 낮은(=좋은) 모델에 is_recommended=True 표시.

    Args:
        calibration_results: 모델별 (Outlier pruning까지 마친) 최종 CalibrationResult.
            E_train, E_edge(fallback)의 출처.
        validation_results: 모델별 ValidationResult (5단계 Hold-out 결과).
            E_test, E_edge(우선 출처)의 출처. 반드시 "같은 train/test 분할"로
            계산된 것이어야 공정하다 (validate_all_models 참고).
    """
    models = list(calibration_results.keys())
    complexity_counts = _complexity_counts(use_rational_model)

    e_train = {m: (calibration_results[m].rms_error if calibration_results[m].success else None) for m in models}

    e_test = {}
    for m in models:
        vr = validation_results.get(m)
        e_test[m] = vr.test_rms if (vr and vr.success) else None

    # Edge는 Hold-out(test)에서 나온 값을 우선 쓴다 - "새로운 데이터의 외곽 오차"가
    # "학습 데이터 외곽 오차"보다 일반화 성능을 더 잘 반영하기 때문.
    # 없으면 Train 쪽 regional_error로 대체.
    e_edge = {}
    for m in models:
        vr = validation_results.get(m)
        if vr and vr.success and vr.edge_rms is not None:
            e_edge[m] = vr.edge_rms
        elif calibration_results[m].success and calibration_results[m].regional_error:
            e_edge[m] = regional_edge_average(calibration_results[m].regional_error)
        else:
            e_edge[m] = None

    # Straightness (V2, 아직 미구현) - 전부 None -> _normalize가 중립 0으로 처리
    e_line = {}
    for m in models:
        vr = validation_results.get(m)
        e_line[m] = vr.straightness_residual if vr else None

    complexity = {m: float(complexity_counts.get(m, 0)) for m in models}

    n_train = _normalize(e_train)
    n_test = _normalize(e_test)
    n_edge = _normalize(e_edge)
    n_line = _normalize(e_line)
    n_complexity = _normalize(complexity)

    scores: list[ModelScore] = []
    for m in models:
        components = {
            "train": weights.w_train * n_train[m],
            "test": weights.w_test * n_test[m],
            "edge": weights.w_edge * n_edge[m],
            "line": weights.w_line * n_line[m],
            "complexity": weights.w_complexity * n_complexity[m],
        }
        total = sum(components.values())
        scores.append(ModelScore(model_name=m, score=total, components=components))

    # 학습 자체가 실패한 모델은 추천 후보에서 제외
    eligible = [s for s in scores if calibration_results[s.model_name].success]
    if eligible:
        # 점수가 같으면 더 단순한 모델을 우선한다 (Occam's razor).
        best = min(eligible, key=lambda s: (round(s.score, 6), complexity_counts.get(s.model_name, 0)))
        best.is_recommended = True

    return scores


# ---------------------------------------------------------------------------
# 근거 문장 생성 (설계 문서 8번 "근거 있는 추천")
# ---------------------------------------------------------------------------

_LABELS = {
    CameraModelType.PINHOLE: "Pinhole",
    CameraModelType.EXTENDED_PINHOLE: "Extended Pinhole",
    CameraModelType.FISHEYE: "Fisheye",
}


def build_recommendation_message(
    scores: list[ModelScore],
    calibration_results: dict[CameraModelType, CalibrationResult],
    validation_results: dict[CameraModelType, ValidationResult],
) -> str:
    """"⭐ Recommended: Extended Pinhole - Test RMS/Edge RMS 기준 가장 우수"
    형태의 사람이 읽는 근거 문장. 어디까지나 근거 제시이지 강제가 아니다.
    """
    recommended = next((s for s in scores if s.is_recommended), None)
    if recommended is None:
        return "모든 모델의 캘리브레이션이 실패해 추천할 수 없습니다."

    model = recommended.model_name
    label = _LABELS[model]
    train_rms = calibration_results[model].rms_error
    vr = validation_results.get(model)
    test_rms = vr.test_rms if vr else None

    # "line"(직선성, V2 미구현)처럼 아무 모델도 값을 갖지 못한 지표는 전부 0으로
    # 정규화되어 "가장 유리한 항목"처럼 보이지만, 실제로는 측정된 적이 없다.
    # 측정이 하나라도 있었던 지표만 근거 후보로 인정한다.
    line_measured = any(
        (validation_results.get(m) and validation_results[m].straightness_residual is not None)
        for m in validation_results
    )
    metric_names_kr = {"train": "Train RMS", "test": "Test RMS", "edge": "Edge RMS", "complexity": "복잡도"}
    if line_measured:
        metric_names_kr["line"] = "직선성"

    reasons = []
    # 가장 기여도가 낮은(=가장 유리했던) 항목을 근거로 든다
    sorted_components = sorted(
        ((k, v) for k, v in recommended.components.items() if k in metric_names_kr),
        key=lambda kv: kv[1],
    )
    top_reasons = [metric_names_kr[k] for k, v in sorted_components[:2] if v <= 0.15]
    if top_reasons:
        reasons.append(" / ".join(top_reasons) + " 기준에서 가장 우수")

    detail = []
    if train_rms is not None:
        detail.append(f"Train {train_rms:.3f}px")
    if test_rms is not None:
        detail.append(f"Test {test_rms:.3f}px")

    msg = f"⭐ 추천 모델: {label} (Score={recommended.score:.3f})"
    if reasons:
        msg += f" — {', '.join(reasons)}"
    if detail:
        msg += f" [{', '.join(detail)}]"
    msg += "\n※ 이 추천은 근거 제시일 뿐입니다. ROS 파이프라인 호환성 등의 이유로 다른 모델을 선택해도 됩니다."
    return msg


def format_score_table(
    scores: list[ModelScore],
    calibration_results: dict[CameraModelType, CalibrationResult],
    validation_results: dict[CameraModelType, ValidationResult],
) -> str:
    """설계 문서 8번 모델 비교 예시 형식.

                     Pinhole   Extended   Fisheye
    Train RMS          0.91      0.42       0.38
    Test RMS           1.03      0.49       0.44
    Edge RMS           1.72      0.61       0.48
    Complexity           ★        ★★       ★★★
    Score              0.62      0.18       0.31
    Recommendation                 ⭐
    """
    order = [s.model_name for s in scores]
    labels = [_LABELS[m] for m in order]
    col_w = max(10, max(len(l) for l in labels) + 2)

    def row(name: str, values: list[str]) -> str:
        return f"{name:<14}" + "".join(f"{v:>{col_w}}" for v in values)

    def fmt(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "N/A"

    train_vals = [fmt(calibration_results[m].rms_error) if calibration_results[m].success else "FAIL" for m in order]
    test_vals = [fmt(validation_results.get(m).test_rms if validation_results.get(m) else None) for m in order]
    # edge는 components(가중치 적용된 정규화 값)가 아니라 원본 px 단위로 다시 뽑아서 보여준다.
    raw_edge = []
    for m in order:
        vr = validation_results.get(m)
        if vr and vr.success and vr.edge_rms is not None:
            raw_edge.append(fmt(vr.edge_rms))
        elif calibration_results[m].success and calibration_results[m].regional_error:
            raw_edge.append(fmt(regional_edge_average(calibration_results[m].regional_error)))
        else:
            raw_edge.append("N/A")

    score_vals = [f"{s.score:.3f}" for s in scores]
    rec_vals = ["⭐" if s.is_recommended else "" for s in scores]

    lines = [
        row("", labels),
        row("Train RMS", train_vals),
        row("Test RMS", test_vals),
        row("Edge RMS", raw_edge),
        row("Score", score_vals),
        row("Recommend", rec_vals),
    ]
    return "\n".join(lines)

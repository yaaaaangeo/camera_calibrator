"""
tests/test_reflection_suppression.py
==============================================================

STEP 7 - Reflection Suppression core tests.

Maps to the user's required test list (stabilization round item 14):
  A. Real Pair equation          -> test_real_pair_loss_supervises_predicted_correction_not_reflection_alone
  B. Synthetic supervision       -> test_synthetic_sample_supervises_reflection_and_alpha_directly
  C. Reference evaluation        -> test_evaluate_suppression_reports_before_after_using_step6_evaluator
  D. No-reference evaluation     -> test_evaluate_suppression_no_reference_uses_likelihood_terminology_only
  E. Scene split                 -> test_scene_level_split_*
  F. Outer Test leakage          -> test_outer_test_change_does_not_affect_trained_state_dict
  G. Known reflection improvement -> test_deterministic_known_reflection_case_improves_after_training
  H. Identity                    -> test_zero_alpha_model_reproduces_input_identity, test_over_suppression_score_is_near_zero_for_clean_identity_model
  I. Correction bound            -> test_correction_never_exceeds_configured_max
  J. NaN/Inf                     -> test_non_finite_model_output_falls_back_to_original
  K. Model exception             -> test_model_exception_falls_back_to_original_and_reports_error
  L. Low reflection guard        -> test_low_reflection_guard_skips_suppression_when_alpha_is_low
  M. Strong local reflection     -> test_low_reflection_guard_does_not_skip_strong_local_reflection
  N. Save/load                   -> test_save_and_load_model_round_trips
  (O/P/Q/R/S are covered by test_reflection_suppression_torch_lazy_import.py /
   test_windshield_reflection_ui_architecture.py)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from calibration.windshield.reflection_suppression.dataset import SuppressionPair, prepare_pair, scene_level_split
from calibration.windshield.reflection_suppression.evaluation import evaluate_suppression
from calibration.windshield.reflection_suppression.model import build_model
from calibration.windshield.reflection_suppression.runtime import (
    ReflectionSuppressionModel,
    SuppressionRuntimeConfig,
    load_suppression_model,
    save_suppression_model,
    suppress_reflection,
)
from calibration.windshield.reflection_suppression.synthetic import make_identity_sample, make_synthetic_reflection_sample
from calibration.windshield.reflection_suppression.training import (
    LossWeights,
    TrainingSample,
    compute_batch_losses,
    train_suppression_from_dataset,
    train_suppression_model,
)
from calibration.windshield.reflection_suppression.types import SuppressionModelMetadata

_RES = 48


def _zero_bias_model(alpha_bias: float, reflection_bias: float = 0.0) -> ReflectionSuppressionModel:
    """모든 파라미터를 0으로 만들고 out_conv의 bias만 지정한다 - weight가
    전부 0이면 어떤 입력이 들어와도 그 앞 activation은 무시되고 bias만
    남으므로, 입력과 완전히 무관한 상수 alpha/reflection을 내는 결정론적
    모델을 만들 수 있다(A/D/Hallucination guard 테스트용)."""
    net = build_model()
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        net.out_conv.bias[0:3] = reflection_bias
        net.out_conv.bias[3] = alpha_bias
    return ReflectionSuppressionModel(net.state_dict())


def _random_clean_image(rng: np.random.Generator, res: int = _RES) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=(res, res, 3)).astype(np.float32)


def _no_guard_config(**overrides) -> SuppressionRuntimeConfig:
    """Low Reflection Guard(mean/P95/coverage)를 전부 비활성화한
    SuppressionRuntimeConfig - guard 로직 자체를 테스트하는 게 아닌 다른
    테스트에서 skip 경로로 새지 않도록 한다."""
    base = dict(min_mean_alpha=None, min_alpha_p95=None, min_alpha_coverage=None)
    base.update(overrides)
    return SuppressionRuntimeConfig(**base)


def _constant_output_model(reflection_bias: float, alpha_bias: float):
    """모든 파라미터를 0으로 만들고 out_conv bias만 지정한 raw `nn.Module`
    (ReflectionSuppressionModel로 감싸지 않은 net 자체) - loss 계산 함수를
    직접 테스트할 때, 입력과 무관한 상수 reflection_hat/alpha_hat을 손으로
    재계산할 수 있게 한다."""
    net = build_model()
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        net.out_conv.bias[0:3] = reflection_bias
        net.out_conv.bias[3] = alpha_bias
    return net


# ---------------------------------------------------------------------------
# Test A - Real Pair equation(CRITICAL, 안정화 라운드 항목 1) - I = T + aR
# 이므로 I-T ≈ aR(correction)이지 I-T = R이 아니다. Real pair(정확한 GT가
# 없는) 샘플은 alpha_hat*reflection_hat 전체를 pseudo_correction과
# 비교해야지, reflection_hat 단독을 비교하면 안 된다.
# ---------------------------------------------------------------------------

def test_real_pair_loss_supervises_predicted_correction_not_reflection_alone():
    from calibration.windshield.reflection_suppression.config import CHARBONNIER_EPS

    net = _constant_output_model(reflection_bias=1.0, alpha_bias=-0.5)
    rng = np.random.default_rng(20)
    observed = _random_clean_image(rng)
    target = _random_clean_image(rng)
    # reflection_gt/alpha_gt를 지정하지 않았다 = 정확한 GT가 없는 real pair 샘플.
    real_pair_sample = TrainingSample(observed=observed, target_clean=target)

    losses = compute_batch_losses(net, [real_pair_sample], LossWeights())

    assert losses["reflection_gt"] is None
    assert losses["alpha_gt"] is None
    assert losses["real_correction"] is not None

    reflection_hat_val = 1.0 / (1.0 + np.exp(-1.0))
    alpha_hat_val = 1.0 / (1.0 + np.exp(0.5))
    predicted_correction = alpha_hat_val * reflection_hat_val  # alpha_hat*reflection_hat, 상수
    pseudo_correction = np.clip(observed - target, 0.0, None)  # max(I-T, 0)
    diff = predicted_correction - pseudo_correction
    expected = float(np.mean(np.sqrt(diff ** 2 + CHARBONNIER_EPS ** 2)))

    assert losses["real_correction"] == pytest.approx(expected, rel=1e-3)

    # 만약 (버그가 있었던 이전 방식처럼) reflection_hat 단독을 pseudo target과
    # 비교했다면 다른 값이 나왔을 것이다 - 두 계산이 실제로 다른 값을
    # 낸다는 것 자체가 이 테스트가 뭔가 의미 있게 구분하고 있다는 증거다.
    pseudo_reflection_only = np.clip(observed - target, 0.0, None)
    diff_buggy = reflection_hat_val - pseudo_reflection_only
    buggy_value = float(np.mean(np.sqrt(diff_buggy ** 2 + CHARBONNIER_EPS ** 2)))
    assert losses["real_correction"] != pytest.approx(buggy_value, rel=1e-3)


# ---------------------------------------------------------------------------
# Test B - Synthetic supervision 유지: GT가 있는 샘플은 reflection_hat/
# alpha_hat 각각을 R_GT/alpha_GT와 직접 비교해야 한다.
# ---------------------------------------------------------------------------

def test_synthetic_sample_supervises_reflection_and_alpha_directly():
    from calibration.windshield.reflection_suppression.config import CHARBONNIER_EPS

    net = _constant_output_model(reflection_bias=1.0, alpha_bias=-0.5)
    rng = np.random.default_rng(21)
    clean = _random_clean_image(rng)
    interior = _random_clean_image(rng)
    s = make_synthetic_reflection_sample(clean, interior, rng)
    synthetic_sample = TrainingSample(s.observed, s.clean, s.reflection, s.alpha)

    losses = compute_batch_losses(net, [synthetic_sample], LossWeights())

    assert losses["reflection_gt"] is not None
    assert losses["alpha_gt"] is not None
    assert losses["real_correction"] is None

    reflection_hat_val = 1.0 / (1.0 + np.exp(-1.0))
    alpha_hat_val = 1.0 / (1.0 + np.exp(0.5))
    expected_reflection = float(np.mean(np.sqrt((reflection_hat_val - s.reflection) ** 2 + CHARBONNIER_EPS ** 2)))
    expected_alpha = float(np.mean(np.sqrt((alpha_hat_val - s.alpha) ** 2 + CHARBONNIER_EPS ** 2)))

    assert losses["reflection_gt"] == pytest.approx(expected_reflection, rel=1e-3)
    assert losses["alpha_gt"] == pytest.approx(expected_alpha, rel=1e-3)


# ---------------------------------------------------------------------------
# Test A(구 버전 번호) - Zero Reflection: alpha ~ 0 -> output ~ input
# ---------------------------------------------------------------------------

def test_zero_alpha_model_reproduces_input_identity():
    model = _zero_bias_model(alpha_bias=-20.0)
    rng = np.random.default_rng(0)
    clean = (_random_clean_image(rng) * 255.0).astype(np.uint8)

    result = suppress_reflection(clean, model, _no_guard_config())

    assert result.success
    assert result.mean_alpha < 1e-6
    assert np.max(np.abs(result.suppressed_image.astype(np.int16) - clean.astype(np.int16))) <= 1


# ---------------------------------------------------------------------------
# Test D - Correction Bound
# ---------------------------------------------------------------------------

def test_correction_never_exceeds_configured_max():
    """alpha~1, reflection~1인 최악의 모델이어도 correction은 항상
    max_correction 이하여야 한다."""
    model = _zero_bias_model(alpha_bias=20.0, reflection_bias=20.0)
    rng = np.random.default_rng(1)
    image = (_random_clean_image(rng) * 255.0).astype(np.uint8)

    # uint8 round-trip(0-255 <-> [0,1] float) 자체가 최대 1-2/255 정도의
    # 양자화 오차를 만들 수 있으므로, 그 오차를 감안한 여유(margin)를 둔다 -
    # correction 자체(`result.max_correction`, float 도메인)는 여유 없이
    # 엄격하게 확인한다.
    quantization_margin = 2.0 / 255.0
    for max_correction in (0.05, 0.15, 0.30):
        cfg = _no_guard_config(strength=1.0, max_correction=max_correction)
        result = suppress_reflection(image, model, cfg)
        assert result.success
        assert result.max_correction <= max_correction + 1e-6
        diff = np.abs(result.suppressed_image.astype(np.float32) - image.astype(np.float32)) / 255.0
        assert float(np.max(diff)) <= max_correction + quantization_margin


# ---------------------------------------------------------------------------
# Test E - NaN/Inf Guard
# ---------------------------------------------------------------------------

class _NonFiniteStubModel:
    def predict(self, image_bgr):
        h, w = image_bgr.shape[:2]
        return np.full((h, w, 3), np.nan, dtype=np.float64), np.full((h, w), 0.5, dtype=np.float64)


def test_non_finite_model_output_falls_back_to_original():
    rng = np.random.default_rng(2)
    image = (_random_clean_image(rng) * 255.0).astype(np.uint8)

    result = suppress_reflection(image, _NonFiniteStubModel(), SuppressionRuntimeConfig())

    assert result.success is False
    assert result.fell_back_to_original is True
    assert np.array_equal(result.suppressed_image, image)


class _RaisingStubModel:
    def predict(self, image_bgr):
        raise RuntimeError("simulated inference crash")


def test_model_exception_falls_back_to_original_and_reports_error():
    rng = np.random.default_rng(3)
    image = (_random_clean_image(rng) * 255.0).astype(np.uint8)

    result = suppress_reflection(image, _RaisingStubModel(), SuppressionRuntimeConfig())

    assert result.success is False
    assert result.fell_back_to_original is True
    assert np.array_equal(result.suppressed_image, image)
    assert result.error_message is not None


def test_low_reflection_guard_skips_suppression_when_alpha_is_low():
    """Test L - mean/P95/coverage가 전부 threshold 아래일 때만 skip한다."""
    model = _zero_bias_model(alpha_bias=-20.0, reflection_bias=20.0)  # alpha~0이지만 reflection~1
    rng = np.random.default_rng(4)
    image = (_random_clean_image(rng) * 255.0).astype(np.uint8)

    result = suppress_reflection(image, model, SuppressionRuntimeConfig())

    assert result.success
    assert result.skipped_due_to_low_reflection is True
    assert np.array_equal(result.suppressed_image, image)
    assert result.warning_message is not None


def test_low_reflection_guard_does_not_skip_strong_local_reflection():
    """Test M(안정화 라운드 항목 8) - 화면의 2%에만 강한(alpha=0.9) 반사가
    있는 경우, global mean alpha는 threshold보다 낮지만(예전 방식이면 잘못
    skip했을 상황) coverage는 여전히 threshold를 넘으므로 skip하면 안 된다.
    Guard 로직(`_should_skip_due_to_low_reflection`)을 직접, 결정론적으로
    검증한다(신경망의 spatial 표현력에 의존하지 않는 순수 단위 테스트)."""
    from calibration.windshield.reflection_suppression.runtime import _should_skip_due_to_low_reflection

    cfg = SuppressionRuntimeConfig()
    h = w = 100
    flat = np.zeros(h * w, dtype=np.float32)
    strong_pixel_count = int(0.02 * flat.size)  # 화면의 2%에만 강한 반사
    flat[:strong_pixel_count] = 0.9
    alpha_map = flat.reshape(h, w)

    mean_alpha = float(np.mean(alpha_map))
    alpha_p95 = float(np.percentile(alpha_map, 95.0))
    alpha_coverage = float(np.mean(alpha_map > cfg.alpha_presence_threshold))

    # 이 케이스가 실제로 "global mean은 낮지만 coverage는 반사 존재를
    # 시사한다"는 시나리오를 만드는지 먼저 확인한다(테스트 전제 조건).
    assert cfg.min_mean_alpha is not None and mean_alpha < cfg.min_mean_alpha
    assert cfg.min_alpha_coverage is not None and alpha_coverage >= cfg.min_alpha_coverage

    assert _should_skip_due_to_low_reflection(mean_alpha, alpha_p95, alpha_coverage, cfg) is False


# ---------------------------------------------------------------------------
# Test B/C - 짧게 학습한 모델로 구조적 개선/localization 확인(exact recovery
# 를 요구하지 않는다 - flaky한 tight threshold를 피한다).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_model_and_sample():
    rng = np.random.default_rng(7)
    samples: list[TrainingSample] = []
    held_out = None
    for i in range(10):
        clean = _random_clean_image(rng)
        interior = _random_clean_image(rng)
        s = make_synthetic_reflection_sample(clean, interior, rng, max_alpha=0.6)
        sample = TrainingSample(s.observed, s.clean, s.reflection, s.alpha)
        if i == 0:
            held_out = (s, sample)
        else:
            samples.append(sample)
        identity = make_identity_sample(clean)
        samples.append(TrainingSample(identity.observed, identity.clean, identity.reflection, identity.alpha))

    val = samples[:4]
    train = samples[4:]
    outcome = train_suppression_model(train, val, max_epochs=120, patience=30, seed=5)
    model = ReflectionSuppressionModel(outcome.state_dict)
    return model, held_out[0]


def test_trained_model_suppression_moves_closer_to_clean(trained_model_and_sample):
    model, sample = trained_model_and_sample
    observed_u8 = np.clip(sample.observed * 255.0, 0, 255).astype(np.uint8)
    clean_u8 = np.clip(sample.clean * 255.0, 0, 255).astype(np.uint8)

    result = suppress_reflection(observed_u8, model, _no_guard_config())
    assert result.success

    before_dist = float(np.mean(np.abs(observed_u8.astype(np.float32) - clean_u8.astype(np.float32))))
    after_dist = float(np.mean(np.abs(result.suppressed_image.astype(np.float32) - clean_u8.astype(np.float32))))
    assert after_dist <= before_dist + 1e-3  # 최소한 나빠지지는 않아야 한다(느슨한 구조적 검증)


def test_trained_model_learns_bottom_biased_alpha(trained_model_and_sample):
    """Synthetic generator의 local patch가 항상 아래쪽에 치우쳐 있으므로
    (synthetic.py의 `_random_alpha_map`), 학습된 모델의 alpha 예측도 평균적
    으로 위쪽보다 아래쪽에서 더 커야 한다(사용자 스펙 67-C번)."""
    model, sample = trained_model_and_sample
    observed_u8 = np.clip(sample.observed * 255.0, 0, 255).astype(np.uint8)

    result = suppress_reflection(observed_u8, model, _no_guard_config())
    assert result.success
    h = result.alpha_map.shape[0]
    top_mean = float(np.mean(result.alpha_map[: h // 3]))
    bottom_mean = float(np.mean(result.alpha_map[2 * h // 3:]))
    assert bottom_mean >= top_mean - 1e-3


def test_deterministic_known_reflection_case_improves_after_training():
    """Test G(안정화 라운드 항목 6) - random generator에만 의존하는 대신,
    고정된 checkerboard clean 이미지 + 알려진 bottom Gaussian alpha map +
    알려진 reflection color로 구성한 완전히 결정론적인 GT case를 만들어,
    학습 후 실제로 clean error가 유의미하게(margin 5%) 개선되는지 확인한다.
    완벽한 GT 복원(flaky)은 요구하지 않는다."""
    res = 48
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32)
    # 부드러운 gradient clean image - sharp periodic checkerboard는 3단
    # downsample U-Net에 aliasing을 일으켜 이 테스트가 검증하려는 것(known
    # reflection이 실제로 줄어드는가)과 무관한 이유로 학습을 어렵게 만들 수
    # 있다. 실제 windshield 뒤 scene도 sharp periodic pattern보다는 완만한
    # gradient에 더 가깝다.
    grad = (xx + yy) / (2.0 * (res - 1))
    clean = np.stack([grad, 1.0 - grad, np.full_like(grad, 0.5)], axis=-1).astype(np.float32)

    cy, cx = res * 0.85, res * 0.5  # 알려진 bottom-biased 위치(결정론적)
    alpha_gt = (
        np.exp(-(((yy - cy) / (res * 0.25)) ** 2 + ((xx - cx) / (res * 0.35)) ** 2)).astype(np.float32) * 0.6
    )
    reflection_gt = np.stack(
        [np.full((res, res), 0.9, np.float32), np.full((res, res), 0.2, np.float32), np.full((res, res), 0.2, np.float32)],
        axis=-1,
    )
    observed = np.clip(clean + alpha_gt[..., None] * reflection_gt, 0.0, 1.0).astype(np.float32)

    # 이 테스트의 목적은 "이 알려진 GT case를 배울 수 있는가"이지 미학습
    # scale/변형에 대한 일반화 성능이 아니다(item 6은 known reflection이
    # 실제로 줄어드는지만 요구한다) - 그래서 평가도 학습에 쓴 바로 그
    # (observed, clean) 쌍에 대해 수행한다(known_sample을 여러 번 반복 +
    # identity 샘플을 섞어 alpha=0 identity 제약도 함께 학습시킨다).
    known_sample = TrainingSample(observed, clean, reflection_gt, alpha_gt)
    identity_raw = make_identity_sample(clean)
    identity_sample = TrainingSample(identity_raw.observed, identity_raw.clean, identity_raw.reflection, identity_raw.alpha)
    train = [known_sample, identity_sample] * 4
    val = [known_sample, identity_sample]
    outcome = train_suppression_model(train, val, max_epochs=200, patience=40, seed=41)
    model = ReflectionSuppressionModel(outcome.state_dict)

    observed_u8 = np.clip(observed * 255.0, 0, 255).astype(np.uint8)
    clean_u8 = np.clip(clean * 255.0, 0, 255).astype(np.uint8)
    result = suppress_reflection(observed_u8, model, _no_guard_config())
    assert result.success

    before_mae = float(np.mean(np.abs(observed_u8.astype(np.float32) - clean_u8.astype(np.float32))))
    after_mae = float(np.mean(np.abs(result.suppressed_image.astype(np.float32) - clean_u8.astype(np.float32))))
    assert after_mae < before_mae * 0.95

    # 추가 metric(사용자 스펙 6번 "1~2개 추가") - 예측 alpha와 GT alpha의
    # 상관관계가 최소한 양(+)이어야 한다(완벽한 복원은 요구하지 않는다).
    predicted_alpha = result.alpha_map
    if np.std(predicted_alpha) > 1e-6 and np.std(alpha_gt) > 1e-6:
        correlation = float(np.corrcoef(predicted_alpha.reshape(-1), alpha_gt.reshape(-1))[0, 1])
        assert correlation > 0.0


# ---------------------------------------------------------------------------
# Hallucination Guard Test - 고대비 edge 위치가 크게 이동하지 않아야 한다.
# ---------------------------------------------------------------------------

def test_zero_alpha_model_does_not_move_high_contrast_edge():
    image = np.zeros((_RES, _RES, 3), dtype=np.uint8)
    image[:, _RES // 2 :, :] = 255  # 수직 edge(도로 표지/차선 비유)
    model = _zero_bias_model(alpha_bias=-20.0)

    result = suppress_reflection(image, model, _no_guard_config())
    assert result.success

    def _edge_column(img):
        gray = img.mean(axis=2)
        grad = np.abs(np.diff(gray.mean(axis=0)))
        return int(np.argmax(grad))

    assert abs(_edge_column(result.suppressed_image) - _edge_column(image)) <= 1


# ---------------------------------------------------------------------------
# Evaluation tests (STEP 6 evaluator 재사용)
# ---------------------------------------------------------------------------

def test_evaluate_suppression_reports_before_after_using_step6_evaluator(trained_model_and_sample):
    model, sample = trained_model_and_sample
    observed_u8 = np.clip(sample.observed * 255.0, 0, 255).astype(np.uint8)
    clean_u8 = np.clip(sample.clean * 255.0, 0, 255).astype(np.uint8)

    result = suppress_reflection(observed_u8, model, _no_guard_config())
    evaln = evaluate_suppression(observed_u8, result, reference_image=clean_u8)

    assert evaln.success, evaln.error_message
    assert evaln.before.mode == "reference"
    assert evaln.after.mode == "reference"
    assert evaln.reflection_mean_reduction is not None
    assert evaln.edge_retention_after is not None
    assert evaln.contrast_retention_after is not None


def test_evaluate_suppression_handles_failed_suppression_gracefully():
    rng = np.random.default_rng(9)
    image = (_random_clean_image(rng) * 255.0).astype(np.uint8)
    failed_result = suppress_reflection(image, _RaisingStubModel(), SuppressionRuntimeConfig())

    evaln = evaluate_suppression(image, failed_result, reference_image=image)
    assert evaln.success is False
    assert evaln.error_message is not None


def test_evaluate_suppression_no_reference_uses_likelihood_terminology_only():
    """Test D(CRITICAL, 안정화 라운드 항목 2) - Reference가 없으면 Reflection
    Mean/P95/Coverage Reduction은 항상 None이어야 하고, 대신 Reflection
    Likelihood Before/After/Reduction만 채워져야 한다(No-reference 결과를
    실제 reflection 측정값처럼 취급하지 않는다)."""
    model = _zero_bias_model(alpha_bias=-1.0, reflection_bias=0.5)
    rng = np.random.default_rng(22)
    image = (_random_clean_image(rng) * 255.0).astype(np.uint8)

    result = suppress_reflection(image, model, _no_guard_config())
    evaln = evaluate_suppression(image, result, reference_image=None)

    assert evaln.success
    assert evaln.before.mode == "no_reference"
    assert evaln.after.mode == "no_reference"
    assert evaln.reflection_mean_reduction is None
    assert evaln.reflection_p95_reduction is None
    assert evaln.coverage_reduction is None
    assert evaln.reflection_likelihood_before is not None
    assert evaln.reflection_likelihood_after is not None
    assert evaln.reflection_likelihood_reduction is not None


# ---------------------------------------------------------------------------
# Over-suppression metric / Clean identity test
# ---------------------------------------------------------------------------

def test_over_suppression_score_is_near_zero_for_clean_identity_model():
    model = _zero_bias_model(alpha_bias=-20.0)
    rng = np.random.default_rng(10)
    image = (_random_clean_image(rng) * 255.0).astype(np.uint8)
    clean_roi_mask = np.ones(image.shape[:2], dtype=bool)

    result = suppress_reflection(image, model, _no_guard_config())
    evaln = evaluate_suppression(image, result, reference_image=image, clean_roi_mask=clean_roi_mask)

    # 이론적으로는 alpha~0이므로 correction~0이어야 하지만, uint8<->float
    # 왕복 변환 자체의 반올림 잡음이 몇 픽셀에서 1/255 정도 차이를 만들 수
    # 있다 - "거의 변하지 않았다"를 확인하는 것이 목적이므로 1e-5처럼 과도하게
    # 엄격한 기준 대신 1/255보다 훨씬 작은(그러나 반올림 잡음은 허용하는)
    # 기준을 쓴다.
    assert evaln.over_suppression_score is not None
    assert evaln.over_suppression_score < 1.0 / 255.0


# ---------------------------------------------------------------------------
# Reference Leakage / Scene-level split / Alignment quality gate
# ---------------------------------------------------------------------------

def test_scene_level_split_keeps_same_scene_in_one_split_only():
    pairs = [
        SuppressionPair("n1.png", "r1.png", pair_id="p1", scene_id="sceneA"),
        SuppressionPair("n2.png", "r2.png", pair_id="p2", scene_id="sceneA"),
        SuppressionPair("n3.png", "r3.png", pair_id="p3", scene_id="sceneB"),
        SuppressionPair("n4.png", "r4.png", pair_id="p4", scene_id="sceneC"),
    ]
    train, val, test = scene_level_split(pairs, val_scene_ids={"sceneB"}, test_scene_ids={"sceneC"})

    assert {p.scene_id for p in train} == {"sceneA"}
    assert {p.scene_id for p in val} == {"sceneB"}
    assert {p.scene_id for p in test} == {"sceneC"}
    assert len(train) + len(val) + len(test) == len(pairs)


def test_scene_level_split_rejects_overlapping_val_test_scene_ids():
    pairs = [SuppressionPair("n1.png", "r1.png", scene_id="sceneA")]
    with pytest.raises(ValueError):
        scene_level_split(pairs, val_scene_ids={"sceneA"}, test_scene_ids={"sceneA"})


def test_load_manifest_reads_pairs_and_split_scenes(tmp_path):
    """안정화 라운드 항목 3 - 공식 real paired training entrypoint가 읽는
    YAML manifest 스키마(pairs + validation_scenes + test_scenes)를 직접
    검증한다. 상대경로는 manifest 파일 기준으로 해석돼야 한다."""
    import yaml as _yaml

    from calibration.windshield.reflection_suppression.dataset import load_manifest

    (tmp_path / "images").mkdir()
    manifest = {
        "pairs": [
            {"pair_id": "p001", "scene_id": "scene01", "normal": "images/n1.png", "reference": "images/r1.png"},
            {"pair_id": "p002", "scene_id": "scene02", "normal": "images/n2.png", "reference": "images/r2.png"},
        ],
        "validation_scenes": ["scene02"],
        "test_scenes": ["scene03"],
    }
    manifest_path = tmp_path / "manifest.yaml"
    with open(manifest_path, "w", encoding="utf-8") as f:
        _yaml.safe_dump(manifest, f)

    pairs, val_scene_ids, test_scene_ids = load_manifest(str(manifest_path))

    assert len(pairs) == 2
    assert pairs[0].pair_id == "p001" and pairs[0].scene_id == "scene01"
    assert pairs[0].normal_image_path == str(tmp_path / "images" / "n1.png")
    assert val_scene_ids == {"scene02"}
    assert test_scene_ids == {"scene03"}


def test_prepare_pair_excludes_misaligned_pairs(tmp_path):
    import cv2

    rng = np.random.default_rng(11)
    normal = (rng.uniform(0, 255, size=(80, 80, 3))).astype(np.uint8)
    unrelated_reference = (rng.uniform(0, 255, size=(80, 80, 3))).astype(np.uint8)
    normal_path = str(tmp_path / "normal.png")
    reference_path = str(tmp_path / "reference.png")
    cv2.imwrite(normal_path, normal)
    cv2.imwrite(reference_path, unrelated_reference)

    pair = SuppressionPair(normal_path, reference_path, pair_id="bad_pair", scene_id="sceneX")
    prepared = prepare_pair(pair)

    assert prepared is None  # 정렬 불가능한 pair는 학습 후보에서 제외돼야 한다


def test_prepare_pair_accepts_well_aligned_identical_pair(tmp_path):
    import cv2

    rng = np.random.default_rng(12)
    base = (rng.uniform(0, 255, size=(80, 80, 3))).astype(np.uint8)
    normal_path = str(tmp_path / "normal.png")
    reference_path = str(tmp_path / "reference.png")
    cv2.imwrite(normal_path, base)
    cv2.imwrite(reference_path, base)

    pair = SuppressionPair(normal_path, reference_path, pair_id="good_pair", scene_id="sceneY")
    prepared = prepare_pair(pair)

    assert prepared is not None
    assert prepared.alignment_status == "good"
    assert prepared.target_bgr.shape == base.shape


# ---------------------------------------------------------------------------
# Test F - Outer Test leakage regression(CRITICAL, 안정화 라운드 항목 4/5) -
# train_suppression_from_dataset()의 high-level entrypoint 레벨에서, Outer
# Test 이미지를 극단적으로 바꿔도 학습된 state_dict가 완전히 동일해야 한다.
# ---------------------------------------------------------------------------

def _make_dataset_training_pairs(base_dir: Path, corrupt_test: bool) -> list[SuppressionPair]:
    import cv2

    rng = np.random.default_rng(30)  # 항상 같은 seed로 새로 시작 - train/val 이미지가 두 데이터셋에서 완전히 동일해진다
    pairs = []
    for scene_id in ("scene_train_a", "scene_train_b", "scene_val", "scene_test"):
        for i in range(2):
            base = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
            normal = base.copy()
            reference = base.copy()
            if corrupt_test and scene_id == "scene_test":
                # +intensity, random noise, channel inversion을 모두 적용해
                # "극단적으로 변경"한다(사용자 스펙 5번 예시).
                noise = rng.integers(-100, 100, size=normal.shape)
                normal = np.clip(255 - (normal.astype(np.int16) + 50 + noise), 0, 255).astype(np.uint8)
            normal_path = str(base_dir / f"{scene_id}_{i}_normal.png")
            reference_path = str(base_dir / f"{scene_id}_{i}_reference.png")
            cv2.imwrite(normal_path, normal)
            cv2.imwrite(reference_path, reference)
            pairs.append(SuppressionPair(normal_path, reference_path, pair_id=f"{scene_id}_{i}", scene_id=scene_id))
    return pairs


def test_outer_test_change_does_not_affect_trained_state_dict(tmp_path):
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    pairs_a = _make_dataset_training_pairs(dir_a, corrupt_test=False)
    pairs_b = _make_dataset_training_pairs(dir_b, corrupt_test=True)

    common_kwargs = dict(val_scene_ids={"scene_val"}, test_scene_ids={"scene_test"}, max_epochs=8, patience=8, seed=42)
    result_a = train_suppression_from_dataset(pairs_a, **common_kwargs)
    result_b = train_suppression_from_dataset(pairs_b, **common_kwargs)

    # Training 결과(state_dict/best_epoch/validation score)는 완전히 같아야
    # 한다 - Outer Test는 optimizer update/early stopping 어디에도 영향을
    #주지 않는다.
    assert result_a.outcome.best_epoch == result_b.outcome.best_epoch
    assert result_a.outcome.best_val_total_loss == pytest.approx(result_b.outcome.best_val_total_loss, abs=1e-6)
    assert result_a.train_pair_count == result_b.train_pair_count == 4
    assert result_a.validation_pair_count == result_b.validation_pair_count == 2

    state_a, state_b = result_a.outcome.state_dict, result_b.outcome.state_dict
    assert state_a.keys() == state_b.keys()
    for key in state_a:
        torch.testing.assert_close(state_a[key], state_b[key], rtol=0, atol=1e-6)

    # Outer Test 평가 자체는 손상되지 않은 데이터셋(A)에서는 정상적으로
    # 수행된다(Outer Test 평가 경로 자체가 살아있다는 것을 확인한다). 극단적
    # 으로 손상된 데이터셋(B)에서는 STEP 6 alignment quality gate가 그
    # 손상된 pair 자체를 정당하게 걸러낼 수 있다(이것도 올바른 동작이다 -
    # "정렬이 안 되는 pair는 애초에 쓰지 않는다"는 정책이 Outer Test
    # 평가에도 그대로 적용된 것뿐, leakage와는 별개다) - 그래서 B의
    # test_evaluations 개수를 강제하지 않는다.
    assert len(result_a.test_evaluations) == 2
    assert len(result_b.test_evaluations) <= 2


def test_train_suppression_from_dataset_rejects_overlapping_scene_ids(tmp_path):
    pairs = _make_dataset_training_pairs(tmp_path, corrupt_test=False)
    with pytest.raises(ValueError):
        train_suppression_from_dataset(pairs, val_scene_ids={"scene_val"}, test_scene_ids={"scene_val"}, max_epochs=1)


# ---------------------------------------------------------------------------
# Save/Load round-trip
# ---------------------------------------------------------------------------

def test_save_and_load_model_round_trips(tmp_path):
    model = _zero_bias_model(alpha_bias=-3.0, reflection_bias=1.0)
    metadata = SuppressionModelMetadata(best_epoch=5, best_val_loss=0.1234)
    path = str(tmp_path / "model.yml")

    saved_path = save_suppression_model(model, metadata, path)
    assert Path(saved_path).exists()
    assert (tmp_path / "model.pt").exists()

    reloaded = load_suppression_model(path)
    rng = np.random.default_rng(13)
    image = (_random_clean_image(rng) * 255.0).astype(np.uint8)
    r1 = suppress_reflection(image, model, _no_guard_config())
    r2 = suppress_reflection(image, reloaded, _no_guard_config())
    assert np.array_equal(r1.suppressed_image, r2.suppressed_image)


def test_load_model_fails_clearly_if_sibling_pt_file_is_missing(tmp_path):
    model = _zero_bias_model(alpha_bias=-3.0)
    metadata = SuppressionModelMetadata()
    path = str(tmp_path / "model.yml")
    save_suppression_model(model, metadata, path)
    (tmp_path / "model.pt").unlink()

    with pytest.raises(OSError):
        load_suppression_model(path)

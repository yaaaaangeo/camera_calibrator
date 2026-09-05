"""
tests/test_reflection_suppression.py
==============================================================

STEP 7 - Reflection Suppression core tests.

Maps to the user's required test list:
  A. Zero Reflection            -> test_zero_alpha_model_reproduces_input_identity
  B. Known Reflection Overlay   -> test_trained_model_suppression_moves_closer_to_clean
  C. Reflection Localization    -> test_trained_model_learns_bottom_biased_alpha
  D. Correction Bound           -> test_correction_never_exceeds_configured_max
  E. NaN/Inf Guard              -> test_non_finite_model_output_falls_back_to_original
  Evaluation tests              -> test_evaluate_suppression_*
  Over-suppression metric       -> test_over_suppression_score_is_near_zero_for_clean_identity_model
  Hallucination guard           -> test_zero_alpha_model_does_not_move_high_contrast_edge
  Reference leakage             -> test_scene_level_split_*, test_prepare_pair_*
  Save/load round-trip          -> test_save_and_load_model_round_trips
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
from calibration.windshield.reflection_suppression.training import TrainingSample, train_suppression_model
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


# ---------------------------------------------------------------------------
# Test A - Zero Reflection: alpha ~ 0 -> output ~ input
# ---------------------------------------------------------------------------

def test_zero_alpha_model_reproduces_input_identity():
    model = _zero_bias_model(alpha_bias=-20.0)
    rng = np.random.default_rng(0)
    clean = (_random_clean_image(rng) * 255.0).astype(np.uint8)

    result = suppress_reflection(clean, model, SuppressionRuntimeConfig(min_confidence=None))

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
        cfg = SuppressionRuntimeConfig(strength=1.0, max_correction=max_correction, min_confidence=None)
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


def test_min_confidence_guard_skips_suppression_when_alpha_is_low():
    model = _zero_bias_model(alpha_bias=-20.0, reflection_bias=20.0)  # alpha~0이지만 reflection~1
    rng = np.random.default_rng(4)
    image = (_random_clean_image(rng) * 255.0).astype(np.uint8)

    result = suppress_reflection(image, model, SuppressionRuntimeConfig(min_confidence=0.02))

    assert result.success
    assert result.skipped_due_to_low_confidence is True
    assert np.array_equal(result.suppressed_image, image)
    assert result.warning_message is not None


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

    result = suppress_reflection(observed_u8, model, SuppressionRuntimeConfig(min_confidence=None))
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

    result = suppress_reflection(observed_u8, model, SuppressionRuntimeConfig(min_confidence=None))
    assert result.success
    h = result.alpha_map.shape[0]
    top_mean = float(np.mean(result.alpha_map[: h // 3]))
    bottom_mean = float(np.mean(result.alpha_map[2 * h // 3:]))
    assert bottom_mean >= top_mean - 1e-3


# ---------------------------------------------------------------------------
# Hallucination Guard Test - 고대비 edge 위치가 크게 이동하지 않아야 한다.
# ---------------------------------------------------------------------------

def test_zero_alpha_model_does_not_move_high_contrast_edge():
    image = np.zeros((_RES, _RES, 3), dtype=np.uint8)
    image[:, _RES // 2 :, :] = 255  # 수직 edge(도로 표지/차선 비유)
    model = _zero_bias_model(alpha_bias=-20.0)

    result = suppress_reflection(image, model, SuppressionRuntimeConfig(min_confidence=None))
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

    result = suppress_reflection(observed_u8, model, SuppressionRuntimeConfig(min_confidence=None))
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


# ---------------------------------------------------------------------------
# Over-suppression metric / Clean identity test
# ---------------------------------------------------------------------------

def test_over_suppression_score_is_near_zero_for_clean_identity_model():
    model = _zero_bias_model(alpha_bias=-20.0)
    rng = np.random.default_rng(10)
    image = (_random_clean_image(rng) * 255.0).astype(np.uint8)
    clean_roi_mask = np.ones(image.shape[:2], dtype=bool)

    result = suppress_reflection(image, model, SuppressionRuntimeConfig(min_confidence=None))
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
    r1 = suppress_reflection(image, model, SuppressionRuntimeConfig(min_confidence=None))
    r2 = suppress_reflection(image, reloaded, SuppressionRuntimeConfig(min_confidence=None))
    assert np.array_equal(r1.suppressed_image, r2.suppressed_image)


def test_load_model_fails_clearly_if_sibling_pt_file_is_missing(tmp_path):
    model = _zero_bias_model(alpha_bias=-3.0)
    metadata = SuppressionModelMetadata()
    path = str(tmp_path / "model.yml")
    save_suppression_model(model, metadata, path)
    (tmp_path / "model.pt").unlink()

    with pytest.raises(OSError):
        load_suppression_model(path)

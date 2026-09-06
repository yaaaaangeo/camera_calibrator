"""
calibration.windshield.reflection_suppression.training
==============================================================

STEP 7 - Training loop + CLI for Reflection Suppression.

    python -m calibration.windshield.reflection_suppression.training \\
        --synthetic-only --num-synthetic 200 --output model.yml

Training/Runtime UI 분리(사용자 스펙 59번): 이번 첫 버전은 GUI 안에 training
루프를 넣지 않는다 - CLI/스크립트로 학습하고, GUI는 학습된 model.yml을
불러와 inference/evaluation만 한다.

Scene-level train/val/test split은 이 모듈의 책임이 아니다(사용자 스펙 18번)
- `dataset.py::scene_level_split()`으로 이미 나뉜 pair 리스트를 호출부가
넘긴다. 이 모듈은 "이미 나뉜" 샘플만 다루므로 학습 코드 자체가 실수로 split을
다시 섞을 방법이 없다.

Loss(사용자 스펙 26-33번):

    L = λ_clean·L_clean + λ_reflection·L_reflection + λ_alpha·L_alpha
      + λ_edge·L_edge + λ_smooth·L_smooth + λ_sparse·L_sparse

Identity Loss(사용자 스펙 31번)는 별도 항이 아니라, alpha_gt=0/reflection_gt=0/
clean=observed인 "identity 샘플"을 위 손실식에 그대로 흘려보내는 것으로
구현한다 - identity 샘플에서는 L_clean이 곧 "출력이 입력과 같아야 한다"는
identity 제약이고 L_alpha가 "alpha가 0이어야 한다"는 제약이라, 별도 코드
경로 없이 정확히 같은 효과를 낸다(진단용으로 identity 샘플만의 clean loss
평균을 `identity_loss`로 따로 기록한다).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from calibration.windshield.reflection_suppression.config import (
    CHARBONNIER_EPS,
    DEFAULT_ACTIVATION,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DECODER_CHANNELS,
    DEFAULT_ENCODER_CHANNELS,
    DEFAULT_LAMBDA_ALPHA,
    DEFAULT_LAMBDA_CLEAN,
    DEFAULT_LAMBDA_EDGE,
    DEFAULT_LAMBDA_REFLECTION,
    DEFAULT_LAMBDA_SMOOTH,
    DEFAULT_LAMBDA_SPARSE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
)
from calibration.windshield.reflection_suppression.model import _require_torch, build_model


@dataclass
class TrainingSample:
    """하나의 학습 샘플. `reflection_gt`/`alpha_gt`는 synthetic(및 identity)
    샘플에서만 채워진다 - 실제 pair는 근사 pseudo reflection만 계산 가능하므로
    `None`으로 둔다(사용자 스펙 28/29번)."""
    observed: np.ndarray             # float32 HxWx3 [0,1]
    target_clean: np.ndarray         # float32 HxWx3 [0,1]
    reflection_gt: Optional[np.ndarray] = None   # float32 HxWx3 [0,1]
    alpha_gt: Optional[np.ndarray] = None        # float32 HxW [0,1]


def _set_all_seeds(seed: int) -> None:
    _require_torch()
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_tensor_batch(images: list[np.ndarray]):
    import torch

    arr = np.stack(images, axis=0).astype(np.float32)  # N,H,W,3
    return torch.from_numpy(arr.transpose(0, 3, 1, 2)).float()


def _to_alpha_tensor_batch(maps: list[np.ndarray]):
    import torch

    arr = np.stack(maps, axis=0).astype(np.float32)  # N,H,W
    return torch.from_numpy(arr).unsqueeze(1).float()


def _charbonnier(pred, target, eps: float = CHARBONNIER_EPS):
    diff = pred - target
    return (diff * diff + eps * eps).sqrt().mean()


def _gradient_magnitude(img):
    """단순 finite-difference gradient magnitude(Sobel 없이도 충분 -
    autograd로 미분 가능해야 하므로 torch 연산만 사용한다)."""
    import torch

    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    dx = torch.nn.functional.pad(dx, (0, 1, 0, 0))
    dy = torch.nn.functional.pad(dy, (0, 0, 0, 1))
    return torch.sqrt(dx * dx + dy * dy + 1e-8)


@dataclass
class LossWeights:
    lambda_clean: float = DEFAULT_LAMBDA_CLEAN
    lambda_reflection: float = DEFAULT_LAMBDA_REFLECTION
    lambda_alpha: float = DEFAULT_LAMBDA_ALPHA
    lambda_edge: float = DEFAULT_LAMBDA_EDGE
    lambda_smooth: float = DEFAULT_LAMBDA_SMOOTH
    lambda_sparse: float = DEFAULT_LAMBDA_SPARSE


def compute_batch_losses(net, batch: list[TrainingSample], weights: LossWeights) -> dict:
    """한 배치의 loss 성분을 전부 계산해 dict로 반환한다("total" 키가 최종
    학습에 쓰이는 scalar).

    안정화 라운드 항목 1(CRITICAL) - image formation model은
    `I = T + alpha*R`이므로 `I - T ≈ alpha*R`(predicted_correction)이지
    `I - T = R`이 아니다. Synthetic GT가 있는 샘플만 `reflection_hat`/
    `alpha_hat`을 각각 `R_GT`/`alpha_GT`와 직접 비교하고(`loss_reflection_gt`/
    `loss_alpha_gt`), 정확한 GT가 없는 real pair 샘플은 `alpha_hat*reflection_hat`
    (predicted correction) 전체를 `pseudo_correction = max(I_observed -
    T_reference, 0)`과 비교한다(`loss_real_correction`) - `reflection_hat`
    단독을 pseudo target에 맞추지 않는다."""
    import torch

    observed_t = _to_tensor_batch([s.observed for s in batch])
    target_clean_t = _to_tensor_batch([s.target_clean for s in batch])

    reflection_hat, alpha_hat = net(observed_t)
    predicted_correction = alpha_hat * reflection_hat
    pred_clean = torch.clamp(observed_t - predicted_correction, 0.0, 1.0)

    loss_clean = _charbonnier(pred_clean, target_clean_t)
    loss_edge = _charbonnier(_gradient_magnitude(pred_clean), _gradient_magnitude(target_clean_t))
    loss_smooth = (
        (alpha_hat[:, :, :, 1:] - alpha_hat[:, :, :, :-1]).abs().mean()
        + (alpha_hat[:, :, 1:, :] - alpha_hat[:, :, :-1, :]).abs().mean()
    )
    loss_sparse = alpha_hat.abs().mean()

    reflection_gt_terms, alpha_gt_terms, real_correction_terms = [], [], []
    for i, sample in enumerate(batch):
        has_synthetic_gt = sample.reflection_gt is not None and sample.alpha_gt is not None
        if has_synthetic_gt:
            r_gt = torch.from_numpy(sample.reflection_gt.astype(np.float32)).permute(2, 0, 1)
            a_gt = torch.from_numpy(sample.alpha_gt.astype(np.float32)).unsqueeze(0)
            reflection_gt_terms.append(_charbonnier(reflection_hat[i], r_gt))
            alpha_gt_terms.append(_charbonnier(alpha_hat[i], a_gt))
        else:
            # Real pair: 정확한 R/alpha GT가 없다(사용자 스펙 28/29번) -
            # I - T는 R이 아니라 alpha*R(correction)에 대한 근사치이므로,
            # predicted_correction 전체를 pseudo_correction과 비교한다.
            pseudo_correction = torch.clamp(observed_t[i] - target_clean_t[i], min=0.0)
            real_correction_terms.append(_charbonnier(predicted_correction[i], pseudo_correction))

    loss_reflection_gt = torch.stack(reflection_gt_terms).mean() if reflection_gt_terms else torch.tensor(0.0)
    loss_alpha_gt = torch.stack(alpha_gt_terms).mean() if alpha_gt_terms else torch.tensor(0.0)
    loss_real_correction = torch.stack(real_correction_terms).mean() if real_correction_terms else torch.tensor(0.0)

    total = (
        weights.lambda_clean * loss_clean
        + weights.lambda_reflection * (loss_reflection_gt + loss_real_correction)
        + weights.lambda_alpha * loss_alpha_gt
        + weights.lambda_edge * loss_edge
        + weights.lambda_smooth * loss_smooth
        + weights.lambda_sparse * loss_sparse
    )

    # Identity Loss(사용자 스펙 10/31번)는 별도 lambda 항이 아니다 - alpha_gt가
    # 전부 0인 synthetic identity 샘플이 위 loss들(특히 loss_clean/loss_alpha_gt)
    # 을 그대로 통과하는 것 자체가 identity 제약이다. 여기서는 진단용으로
    # identity 샘플만의 clean reconstruction error/평균 alpha/평균 correction을
    # 따로 기록한다.
    identity_indices = [i for i, s in enumerate(batch) if s.alpha_gt is not None and float(np.max(s.alpha_gt)) == 0.0]
    identity_clean_loss = (
        _charbonnier(pred_clean[identity_indices], observed_t[identity_indices]).item()
        if identity_indices else None
    )
    identity_mean_alpha = float(alpha_hat[identity_indices].mean().item()) if identity_indices else None
    identity_mean_correction = (
        float(predicted_correction[identity_indices].abs().mean().item()) if identity_indices else None
    )

    return {
        "total": total,
        "clean": float(loss_clean.item()),
        "reflection_gt": float(loss_reflection_gt.item()) if reflection_gt_terms else None,
        "alpha_gt": float(loss_alpha_gt.item()) if alpha_gt_terms else None,
        "real_correction": float(loss_real_correction.item()) if real_correction_terms else None,
        "edge": float(loss_edge.item()),
        "smooth": float(loss_smooth.item()),
        "sparse": float(loss_sparse.item()),
        "identity_clean": identity_clean_loss,
        "identity_mean_alpha": identity_mean_alpha,
        "identity_mean_correction": identity_mean_correction,
    }


@dataclass
class TrainingOutcome:
    state_dict: dict
    best_epoch: int
    best_val_total_loss: float
    stopped_early: bool


def train_suppression_model(
    train_samples: list[TrainingSample],
    val_samples: list[TrainingSample],
    *,
    encoder_channels: Optional[list[int]] = None,
    decoder_channels: Optional[list[int]] = None,
    activation: str = DEFAULT_ACTIVATION,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    seed: int = DEFAULT_SEED,
    weights: Optional[LossWeights] = None,
) -> TrainingOutcome:
    """Outer Test는 절대 이 함수에 넘기지 않는다(호출부 책임, 사용자 스펙
    19/71번) - 이 함수는 자신이 받은 train/val을 그대로 쓸 뿐이다. Best
    validation total loss 기준 early stopping + best checkpoint restore
    (STEP 5 Neural Residual과 동일한 정책)."""
    _require_torch()
    import torch

    if not train_samples:
        raise ValueError("train_samples must not be empty")

    _set_all_seeds(seed)
    net = build_model(encoder_channels, decoder_channels, activation)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_weights = weights or LossWeights()

    eval_samples = val_samples if val_samples else train_samples
    best_val = float("inf")
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    rng = np.random.default_rng(seed)

    n_train = len(train_samples)
    eff_batch = max(1, min(batch_size, n_train))

    for epoch in range(max_epochs):
        net.train()
        perm = rng.permutation(n_train)
        for start in range(0, n_train, eff_batch):
            idx = perm[start:start + eff_batch]
            batch = [train_samples[i] for i in idx]
            optimizer.zero_grad()
            losses = compute_batch_losses(net, batch, loss_weights)
            losses["total"].backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            val_losses = compute_batch_losses(net, eval_samples, loss_weights)
        val_total = float(val_losses["total"].item())

        if val_total < best_val - 1e-9:
            best_val = val_total
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                stopped_early = True
                break

    return TrainingOutcome(state_dict=best_state, best_epoch=best_epoch, best_val_total_loss=best_val, stopped_early=stopped_early)


@dataclass
class DatasetTrainingResult:
    outcome: TrainingOutcome
    metadata: object  # SuppressionModelMetadata (지연 import로 순환 참조 회피)
    test_evaluations: list
    train_pair_count: int
    validation_pair_count: int
    test_pair_count: int


def train_suppression_from_dataset(
    all_pairs: list,
    *,
    val_scene_ids: set,
    test_scene_ids: set,
    extra_synthetic_samples: Optional[list[TrainingSample]] = None,
    alignment_model: str = "translation",
    allow_warning_alignment: bool = False,
    encoder_channels: Optional[list[int]] = None,
    decoder_channels: Optional[list[int]] = None,
    activation: str = DEFAULT_ACTIVATION,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    seed: int = DEFAULT_SEED,
    weights: Optional[LossWeights] = None,
    max_correction: Optional[float] = None,
    default_strength: Optional[float] = None,
) -> DatasetTrainingResult:
    """실제 paired dataset을 위한 공식 high-level 학습 entrypoint(안정화
    라운드 항목 3/4, CRITICAL). Outer Test는 이 함수 안에서 **구조적으로**
    학습 경로에서 격리된다 - `scene_level_split()`으로 나뉜 `test_pairs`는

        optimizer update, early stopping, hyperparameter selection,
        threshold tuning, suppression strength selection, architecture selection

    그 무엇에도 절대 쓰이지 않는다. Test pair는 오직 마지막에 완전히
    고정(freeze)된 모델을 평가하는 데만 등장한다(순서: Scene Split ->
    Train/Val preprocessing -> Training -> Best checkpoint restore ->
    Model freeze -> Outer Test evaluation)."""
    from calibration.windshield.reflection_suppression.config import DEFAULT_ENCODER_CHANNELS as _DEC
    from calibration.windshield.reflection_suppression.config import DEFAULT_DECODER_CHANNELS as _DDC
    from calibration.windshield.reflection_suppression.config import DEFAULT_MAX_CORRECTION, DEFAULT_SUPPRESSION_STRENGTH
    from calibration.windshield.reflection_suppression.dataset import prepare_pair, scene_level_split
    from calibration.windshield.reflection_suppression.evaluation import evaluate_suppression
    from calibration.windshield.reflection_suppression.runtime import ReflectionSuppressionModel, SuppressionRuntimeConfig, suppress_reflection
    from calibration.windshield.reflection_suppression.types import SuppressionModelMetadata

    resolved_max_correction = DEFAULT_MAX_CORRECTION if max_correction is None else max_correction
    resolved_strength = DEFAULT_SUPPRESSION_STRENGTH if default_strength is None else default_strength

    # 1. Scene Split - 같은 scene_id는 절대 두 split에 동시에 존재할 수 없다
    #    (scene_level_split이 강제, ValueError로 overlap을 막는다).
    train_pairs, val_pairs, test_pairs = scene_level_split(
        all_pairs, val_scene_ids=val_scene_ids, test_scene_ids=test_scene_ids,
    )

    def _to_samples(pairs: list) -> list[TrainingSample]:
        samples = []
        for pair in pairs:
            prepared = prepare_pair(pair, alignment_model=alignment_model, allow_warning_alignment=allow_warning_alignment)
            if prepared is None:
                continue  # alignment quality gate 탈락 - 학습 후보에서 제외
            observed = prepared.input_bgr.astype(np.float32) / 255.0
            target = prepared.target_bgr.astype(np.float32) / 255.0
            samples.append(TrainingSample(observed=observed, target_clean=target))
        return samples

    # 2/3. Train/Validation preprocessing - Test pair는 여기 등장하지 않는다.
    train_samples = _to_samples(train_pairs)
    val_samples = _to_samples(val_pairs)
    if extra_synthetic_samples:
        train_samples = list(train_samples) + list(extra_synthetic_samples)

    if not train_samples:
        raise ValueError(
            "no usable training pairs remained after the alignment quality gate - "
            "check the manifest paths and alignment quality of the real pairs"
        )

    # 4/5. Training + best checkpoint restore(train_suppression_model 내부).
    outcome = train_suppression_model(
        train_samples, val_samples,
        encoder_channels=encoder_channels, decoder_channels=decoder_channels, activation=activation,
        learning_rate=learning_rate, weight_decay=weight_decay, batch_size=batch_size,
        max_epochs=max_epochs, patience=patience, seed=seed, weights=weights,
    )

    # 6. Model freeze - 이 시점 이후 state_dict는 절대 바뀌지 않는다.
    frozen_model = ReflectionSuppressionModel(outcome.state_dict, encoder_channels, decoder_channels, activation)
    runtime_cfg = SuppressionRuntimeConfig(strength=resolved_strength, max_correction=resolved_max_correction)

    # 7. Outer Test evaluation - 오직 평가만, 학습에 절대 재사용되지 않는다.
    test_evaluations = []
    for pair in test_pairs:
        prepared = prepare_pair(pair, alignment_model=alignment_model, allow_warning_alignment=allow_warning_alignment)
        if prepared is None:
            continue
        result = suppress_reflection(prepared.input_bgr, frozen_model, runtime_cfg)
        test_evaluations.append(evaluate_suppression(prepared.input_bgr, result, reference_image=prepared.target_bgr))

    metadata = SuppressionModelMetadata(
        encoder_channels=list(encoder_channels or _DEC),
        decoder_channels=list(decoder_channels or _DDC),
        activation=activation,
        training_seed=seed,
        max_correction=resolved_max_correction,
        default_strength=resolved_strength,
        best_epoch=outcome.best_epoch,
        best_val_loss=outcome.best_val_total_loss,
        dataset_num_real_pairs=len(train_pairs) + len(val_pairs) + len(test_pairs),
        dataset_num_synthetic_pairs=len(extra_synthetic_samples) if extra_synthetic_samples else 0,
        train_sample_count=len(train_samples),
        validation_sample_count=len(val_samples),
        test_sample_count=len(test_evaluations),
    )

    return DatasetTrainingResult(
        outcome=outcome,
        metadata=metadata,
        test_evaluations=test_evaluations,
        train_pair_count=len(train_pairs),
        validation_pair_count=len(val_pairs),
        test_pair_count=len(test_pairs),
    )


def _cli() -> None:  # pragma: no cover - thin CLI wrapper, exercised manually
    import argparse

    from calibration.windshield.reflection_suppression.config import DEFAULT_MAX_CORRECTION, DEFAULT_SUPPRESSION_STRENGTH, DEFAULT_TRAIN_RESOLUTION
    from calibration.windshield.reflection_suppression.runtime import ReflectionSuppressionModel, save_suppression_model
    from calibration.windshield.reflection_suppression.synthetic import make_identity_sample, make_synthetic_reflection_sample
    from calibration.windshield.reflection_suppression.types import SuppressionModelMetadata

    parser = argparse.ArgumentParser(description="Train a Reflection Suppression model (STEP 7).")
    parser.add_argument("--synthetic-only", action="store_true", help="Use only the synthetic reflection generator (no real paired dataset).")
    parser.add_argument("--manifest", type=str, default=None, help="YAML manifest (pairs + validation_scenes + test_scenes) for real paired dataset training.")
    parser.add_argument("--allow-warning-alignment", action="store_true", help="Also accept STEP 6 alignment status=='warning' pairs (default: only 'good').")
    parser.add_argument("--num-synthetic", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=DEFAULT_TRAIN_RESOLUTION)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if args.manifest:
        # 실제 paired dataset 공식 entrypoint(안정화 라운드 항목 3) - scene-
        # level split + Outer Test 완전 격리는 train_suppression_from_dataset()
        # 안에서 구조적으로 강제된다.
        from calibration.windshield.reflection_suppression.dataset import load_manifest

        pairs, val_scene_ids, test_scene_ids = load_manifest(args.manifest)
        result = train_suppression_from_dataset(
            pairs,
            val_scene_ids=val_scene_ids,
            test_scene_ids=test_scene_ids,
            allow_warning_alignment=args.allow_warning_alignment,
            max_epochs=args.max_epochs,
            patience=args.patience,
            seed=args.seed,
        )
        model = ReflectionSuppressionModel(result.outcome.state_dict)
        path = save_suppression_model(model, result.metadata, args.output)
        print(
            f"Saved model to {path} (best_epoch={result.outcome.best_epoch}, "
            f"train_pairs={result.train_pair_count}, val_pairs={result.validation_pair_count}, "
            f"test_pairs={result.test_pair_count})"
        )
        reductions = [
            e.reflection_mean_reduction for e in result.test_evaluations if e.reflection_mean_reduction is not None
        ]
        if reductions:
            print(f"Outer Test mean reflection-mean reduction: {sum(reductions) / len(reductions):.3f}")
        return

    if not args.synthetic_only:
        raise NotImplementedError(
            "Provide either --manifest (real paired dataset) or --synthetic-only."
        )

    rng = np.random.default_rng(args.seed)
    samples: list[TrainingSample] = []
    for _ in range(args.num_synthetic):
        clean = rng.uniform(0.0, 1.0, size=(args.resolution, args.resolution, 3)).astype(np.float32)
        interior = rng.uniform(0.0, 1.0, size=(args.resolution, args.resolution, 3)).astype(np.float32)
        sample = make_synthetic_reflection_sample(clean, interior, rng)
        samples.append(TrainingSample(sample.observed, sample.clean, sample.reflection, sample.alpha))
        if rng.random() < 0.3:
            identity = make_identity_sample(clean)
            samples.append(TrainingSample(identity.observed, identity.clean, identity.reflection, identity.alpha))

    n_val = max(1, int(0.2 * len(samples)))
    val_samples, train_samples = samples[:n_val], samples[n_val:]

    outcome = train_suppression_model(train_samples, val_samples, max_epochs=args.max_epochs, patience=args.patience, seed=args.seed)
    model = ReflectionSuppressionModel(outcome.state_dict)
    metadata = SuppressionModelMetadata(
        training_resolution=args.resolution,
        training_seed=args.seed,
        max_correction=DEFAULT_MAX_CORRECTION,
        default_strength=DEFAULT_SUPPRESSION_STRENGTH,
        best_epoch=outcome.best_epoch,
        best_val_loss=outcome.best_val_total_loss,
        dataset_num_synthetic_pairs=len(samples),
    )
    path = save_suppression_model(model, metadata, args.output)
    print(f"Saved model to {path} (best_epoch={outcome.best_epoch}, best_val_loss={outcome.best_val_total_loss:.6f})")


if __name__ == "__main__":  # pragma: no cover
    _cli()

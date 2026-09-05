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
    학습에 쓰이는 scalar). Synthetic GT가 없는 real pair 샘플은 reflection/
    alpha loss 계산에서 자동으로 제외된다(pseudo target으로 대체)."""
    import torch

    observed_t = _to_tensor_batch([s.observed for s in batch])
    target_clean_t = _to_tensor_batch([s.target_clean for s in batch])

    reflection_hat, alpha_hat = net(observed_t)
    pred_clean = torch.clamp(observed_t - alpha_hat * reflection_hat, 0.0, 1.0)

    loss_clean = _charbonnier(pred_clean, target_clean_t)
    loss_edge = _charbonnier(_gradient_magnitude(pred_clean), _gradient_magnitude(target_clean_t))
    loss_smooth = (
        (alpha_hat[:, :, :, 1:] - alpha_hat[:, :, :, :-1]).abs().mean()
        + (alpha_hat[:, :, 1:, :] - alpha_hat[:, :, :-1, :]).abs().mean()
    )
    loss_sparse = alpha_hat.abs().mean()

    reflection_terms, alpha_terms = [], []
    for i, sample in enumerate(batch):
        if sample.reflection_gt is not None:
            r_gt = torch.from_numpy(sample.reflection_gt.astype(np.float32)).permute(2, 0, 1)
            reflection_terms.append(_charbonnier(reflection_hat[i], r_gt))
        if sample.alpha_gt is not None:
            a_gt = torch.from_numpy(sample.alpha_gt.astype(np.float32)).unsqueeze(0)
            alpha_terms.append(_charbonnier(alpha_hat[i], a_gt))
        else:
            # Real pair: exact alpha GT가 없으므로 pseudo reflection(사용자
            # 스펙 28번, R_pseudo = max(I_normal - I_reference, 0))만
            # reflection loss의 weak target으로 쓴다.
            pseudo_reflection = torch.clamp(observed_t[i] - target_clean_t[i], min=0.0)
            reflection_terms.append(_charbonnier(reflection_hat[i], pseudo_reflection))

    loss_reflection = torch.stack(reflection_terms).mean() if reflection_terms else torch.tensor(0.0)
    loss_alpha = torch.stack(alpha_terms).mean() if alpha_terms else torch.tensor(0.0)

    total = (
        weights.lambda_clean * loss_clean
        + weights.lambda_reflection * loss_reflection
        + weights.lambda_alpha * loss_alpha
        + weights.lambda_edge * loss_edge
        + weights.lambda_smooth * loss_smooth
        + weights.lambda_sparse * loss_sparse
    )

    identity_indices = [i for i, s in enumerate(batch) if s.alpha_gt is not None and float(np.max(s.alpha_gt)) == 0.0]
    identity_loss = (
        _charbonnier(pred_clean[identity_indices], observed_t[identity_indices]).item()
        if identity_indices else None
    )

    return {
        "total": total,
        "clean": float(loss_clean.item()),
        "reflection": float(loss_reflection.item()) if reflection_terms else None,
        "alpha": float(loss_alpha.item()) if alpha_terms else None,
        "edge": float(loss_edge.item()),
        "smooth": float(loss_smooth.item()),
        "sparse": float(loss_sparse.item()),
        "identity": identity_loss,
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


def _cli() -> None:  # pragma: no cover - thin CLI wrapper, exercised manually
    import argparse

    from calibration.windshield.reflection_suppression.config import DEFAULT_MAX_CORRECTION, DEFAULT_SUPPRESSION_STRENGTH, DEFAULT_TRAIN_RESOLUTION
    from calibration.windshield.reflection_suppression.runtime import ReflectionSuppressionModel, save_suppression_model
    from calibration.windshield.reflection_suppression.synthetic import make_identity_sample, make_synthetic_reflection_sample
    from calibration.windshield.reflection_suppression.types import SuppressionModelMetadata

    parser = argparse.ArgumentParser(description="Train a Reflection Suppression model (STEP 7).")
    parser.add_argument("--synthetic-only", action="store_true", help="Use only the synthetic reflection generator (no real paired dataset).")
    parser.add_argument("--num-synthetic", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=DEFAULT_TRAIN_RESOLUTION)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if not args.synthetic_only:
        raise NotImplementedError(
            "Real paired dataset CLI training is not wired up yet in this first version - "
            "use --synthetic-only, or call train_suppression_model()/dataset.prepare_pair() "
            "directly from a script to mix in real pairs."
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

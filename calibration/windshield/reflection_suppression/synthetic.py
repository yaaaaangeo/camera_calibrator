"""
calibration.windshield.reflection_suppression.synthetic
==============================================================

STEP 7 - Synthetic reflection compositing(torch-free, numpy/OpenCV only).

    I = clip(T + alpha * R, 0, 1)

Ground truth(Clean T, Reflection R, Alpha α, Observed I)를 전부 알고 있으므로
Clean reconstruction/Reflection layer/Alpha mask 셋 다 직접 supervise할 수
있다(사용자 스펙 23번). 랜덤화 항목(사용자 스펙 21번): reflection
blur(defocus)/brightness/color shift/gamma, spatial alpha gradient + local
patch, reflection translation/scale.

**Ghost(동일 exterior scene의 double image)는 절대 만들지 않는다**(사용자
스펙 22번) - `interior`는 항상 `clean`과 다른, 독립적인 실내/장식용 이미지여야
한다(호출자 책임 - 이 함수 자체는 두 입력이 다른 scene이라고 강제하지 않지만,
테스트/데이터셋 구성 단계에서 절대 같은 이미지를 넘기지 않는다).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SyntheticReflectionSample:
    observed: np.ndarray     # I, float32 HxWx3, [0,1] (BGR 채널 순서, OpenCV 관례)
    clean: np.ndarray        # T, float32 HxWx3, [0,1]
    reflection: np.ndarray   # R, float32 HxWx3, [0,1] (alpha 적용 전 raw reflection layer)
    alpha: np.ndarray        # α, float32 HxW, [0,1]


def _to_unit_float(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return np.clip(image.astype(np.float32), 0.0, 1.0)


def _random_alpha_map(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """[0,1] 선형 gradient(임의 방향) + 60% 확률로 국소 patch(예: 계기판
    글레어 스팟, 아래쪽에 치우치게)를 합성한다."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy /= max(h - 1, 1)
    xx /= max(w - 1, 1)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    grad = xx * np.cos(angle) + yy * np.sin(angle)
    grad_range = grad.max() - grad.min()
    grad = (grad - grad.min()) / grad_range if grad_range > 1e-6 else np.zeros_like(grad)
    base = rng.uniform(0.2, 0.8) * grad + rng.uniform(0.0, 0.2)

    if rng.random() < 0.6:
        cy = rng.uniform(0.5, 1.0) * (h - 1)
        cx = rng.uniform(0.0, 1.0) * (w - 1)
        ry = max(rng.uniform(0.15, 0.4) * h, 1.0)
        rx = max(rng.uniform(0.15, 0.4) * w, 1.0)
        patch = np.exp(-(((yy * (h - 1) - cy) / ry) ** 2 + ((xx * (w - 1) - cx) / rx) ** 2))
        base = base + rng.uniform(0.3, 0.7) * patch

    return np.clip(base, 0.0, 1.0).astype(np.float32)


def make_synthetic_reflection_sample(
    clean: np.ndarray,
    interior: np.ndarray,
    rng: np.random.Generator,
    *,
    max_alpha: float = 0.5,
) -> SyntheticReflectionSample:
    """clean(외부 scene, GT target T)과 interior(반사원이 되는 별개의 실내
    이미지)로부터 하나의 synthetic 샘플을 만든다."""
    clean_f = _to_unit_float(clean)
    h, w = clean_f.shape[:2]
    interior_resized = cv2.resize(interior, (w, h), interpolation=cv2.INTER_LINEAR)
    interior_f = _to_unit_float(interior_resized)

    # Reflection translation/scale(사용자 스펙 21번) - Ghost와 다르게 clean
    # 자체가 아니라 interior 텍스처에만 적용된다.
    tx = rng.uniform(-0.05, 0.05) * w
    ty = rng.uniform(-0.05, 0.05) * h
    scale = rng.uniform(0.9, 1.15)
    warp = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, scale)
    warp[0, 2] += tx
    warp[1, 2] += ty
    interior_f = cv2.warpAffine(interior_f, warp, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Defocus blur
    sigma = float(rng.choice([0.0, 2.0, 4.0, 6.0]))
    if sigma > 0:
        interior_f = cv2.GaussianBlur(interior_f, (0, 0), sigmaX=sigma)

    # Brightness / color shift / gamma - reflection layer 자체에 적용해서
    # 합성식 I = clip(T + alpha*R, 0, 1)을 정확히 유지한다(exposure를
    # composite 전체에 별도로 적용하면 이 식이 깨진다).
    brightness = rng.uniform(0.6, 1.3)
    color_shift = rng.uniform(-0.08, 0.08, size=3).astype(np.float32)
    gamma = rng.uniform(0.85, 1.15)
    reflection = np.clip(interior_f, 1e-4, 1.0) ** gamma
    reflection = np.clip(reflection * brightness + color_shift[None, None, :], 0.0, 1.0).astype(np.float32)

    alpha = (_random_alpha_map(h, w, rng) * rng.uniform(0.15, max_alpha)).astype(np.float32)

    observed = np.clip(clean_f + alpha[..., None] * reflection, 0.0, 1.0).astype(np.float32)

    return SyntheticReflectionSample(observed=observed, clean=clean_f, reflection=reflection, alpha=alpha)


def make_identity_sample(clean: np.ndarray) -> SyntheticReflectionSample:
    """Reflection이 전혀 없는 clean identity 샘플(사용자 스펙 25번) -
    observed == clean, alpha == 0, reflection == 0. Identity Loss/Test용."""
    clean_f = _to_unit_float(clean)
    h, w = clean_f.shape[:2]
    return SyntheticReflectionSample(
        observed=clean_f.copy(),
        clean=clean_f.copy(),
        reflection=np.zeros_like(clean_f),
        alpha=np.zeros((h, w), dtype=np.float32),
    )

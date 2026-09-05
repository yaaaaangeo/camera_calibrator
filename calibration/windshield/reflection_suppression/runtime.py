"""
calibration.windshield.reflection_suppression.runtime
==============================================================

STEP 7 - Runtime Reflection Suppression API.

    suppress_reflection(image_bgr, model, config) -> ReflectionSuppressionResult

Final equation(사용자 스펙 3/6/36번):

    correction = clip(strength * alpha_hat * reflection_hat, 0, max_correction)
    I_out      = clip(I - correction, 0, 1)

Runtime은 Reference image를 요구하지 않는다(사용자 스펙 53번, No-Reference
Runtime) - Reference는 Training/Validation/Offline Test(evaluation.py)에서만
쓰인다. Backend는 이 함수 시그니처 뒤에 완전히 숨겨진다(사용자 스펙 65번) -
지금은 PyTorch backend 하나뿐이지만, 향후 ONNX/TensorRT backend가 추가돼도
호출자는 이 API만 알면 된다.
"""

from __future__ import annotations

import base64
import dataclasses
import io
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from calibration.windshield.reflection_suppression.config import (
    DEFAULT_ACTIVATION,
    DEFAULT_DECODER_CHANNELS,
    DEFAULT_ENCODER_CHANNELS,
    DEFAULT_MAX_CORRECTION,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_SUPPRESSION_STRENGTH,
)
from calibration.windshield.reflection_suppression.model import _require_torch, build_model
from calibration.windshield.reflection_suppression.types import ReflectionSuppressionResult, SuppressionModelMetadata


@dataclasses.dataclass
class SuppressionRuntimeConfig:
    strength: float = DEFAULT_SUPPRESSION_STRENGTH
    max_correction: float = DEFAULT_MAX_CORRECTION
    min_confidence: Optional[float] = DEFAULT_MIN_CONFIDENCE


class ReflectionSuppressionModel:
    """torch가 로드된 뒤에만 실제로 구성 가능한 runtime wrapper - UI/worker/
    테스트가 알아야 하는 유일한 클래스(내부 `nn.Module`은 노출하지 않는다)."""

    def __init__(
        self,
        state_dict: dict,
        encoder_channels: Optional[list[int]] = None,
        decoder_channels: Optional[list[int]] = None,
        activation: str = DEFAULT_ACTIVATION,
    ):
        _require_torch()
        self._net = build_model(encoder_channels, decoder_channels, activation)
        self._net.load_state_dict(state_dict)
        self._net.eval()
        self._encoder_channels = list(encoder_channels or DEFAULT_ENCODER_CHANNELS)
        self._decoder_channels = list(decoder_channels or DEFAULT_DECODER_CHANNELS)
        self._activation = activation

    def predict(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(reflection_layer, alpha)를 float64 [0,1] array로 반환한다.
        BGR 채널 순서를 끝까지 그대로 유지한다(OpenCV 관례 - synthetic
        generator/training data 준비 모두 동일 관례를 쓴다)."""
        import torch

        img = image_bgr.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        with torch.no_grad():
            reflection, alpha = self._net(tensor)
        reflection_np = reflection[0].permute(1, 2, 0).numpy().astype(np.float64)
        alpha_np = alpha[0, 0].numpy().astype(np.float64)
        return reflection_np, alpha_np

    def state_dict(self) -> dict:
        return self._net.state_dict()


def _encode_state_dict(state_dict: dict) -> str:
    _require_torch()
    import torch

    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_state_dict(state_dict_b64: str) -> dict:
    _require_torch()
    import torch

    raw = base64.b64decode(state_dict_b64.encode("ascii"))
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)


def suppress_reflection(
    image_bgr: np.ndarray,
    model: ReflectionSuppressionModel,
    config: Optional[SuppressionRuntimeConfig] = None,
) -> ReflectionSuppressionResult:
    """단일 이미지에 Reflection Suppression을 적용한다. 실패해도 절대
    corrupt/None 이미지를 반환하지 않는다 - 항상 원본을 fallback으로
    반환한다(사용자 스펙 38번, "자동차 perception pipeline에서 corrupt
    image를 반환하는 것보다 원본 유지가 안전하다")."""
    cfg = config or SuppressionRuntimeConfig()
    try:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("input image is empty")
        if not np.all(np.isfinite(image_bgr.astype(np.float64))):
            raise ValueError("input image contains non-finite values")

        img_float = image_bgr.astype(np.float32) / 255.0
        reflection, alpha = model.predict(image_bgr)

        if not np.all(np.isfinite(reflection)) or not np.all(np.isfinite(alpha)):
            return ReflectionSuppressionResult(
                success=False,
                suppressed_image=image_bgr.copy(),
                fell_back_to_original=True,
                suppression_strength=cfg.strength,
                error_message="Model produced non-finite reflection/alpha output; returning original image.",
            )

        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float64)
        reflection = np.clip(reflection, 0.0, 1.0).astype(np.float64)
        mean_alpha = float(np.mean(alpha))
        max_alpha = float(np.max(alpha))
        confidence = mean_alpha

        if cfg.min_confidence is not None and mean_alpha < cfg.min_confidence:
            return ReflectionSuppressionResult(
                success=True,
                suppressed_image=image_bgr.copy(),
                reflection_layer=(reflection * 255.0).astype(np.uint8),
                alpha_map=alpha.astype(np.float32),
                suppression_strength=cfg.strength,
                mean_alpha=mean_alpha,
                max_alpha=max_alpha,
                mean_correction=0.0,
                max_correction=0.0,
                confidence=confidence,
                skipped_due_to_low_confidence=True,
                warning_message=(
                    "Mean reflection confidence is below min_confidence - suppression skipped, "
                    "original image returned unchanged."
                ),
            )

        raw_correction = cfg.strength * alpha[..., None] * reflection
        correction = np.clip(raw_correction, 0.0, max(cfg.max_correction, 0.0))
        mean_correction = float(np.mean(correction))
        max_correction_actual = float(np.max(correction))

        suppressed_float = np.clip(img_float - correction, 0.0, 1.0)
        suppressed_image = np.clip(suppressed_float * 255.0, 0, 255).astype(np.uint8)

        return ReflectionSuppressionResult(
            success=True,
            suppressed_image=suppressed_image,
            reflection_layer=(reflection * 255.0).astype(np.uint8),
            alpha_map=alpha.astype(np.float32),
            suppression_strength=cfg.strength,
            mean_alpha=mean_alpha,
            max_alpha=max_alpha,
            mean_correction=mean_correction,
            max_correction=max_correction_actual,
            confidence=confidence,
        )
    except Exception as e:  # noqa: BLE001 - 사용자 스펙 38번, 항상 안전하게 fallback
        return ReflectionSuppressionResult(
            success=False,
            suppressed_image=(image_bgr.copy() if image_bgr is not None and image_bgr.size else None),
            fell_back_to_original=image_bgr is not None and image_bgr.size > 0,
            suppression_strength=cfg.strength,
            error_message=f"Reflection suppression failed, returning original image: {e}",
        )


def save_suppression_model(model: ReflectionSuppressionModel, metadata: SuppressionModelMetadata, path: str) -> str:
    """metadata YAML + sibling `<stem>.pt`(state_dict)로 저장한다(사용자
    스펙 61번, "PyTorch entire model pickle 금지")."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pt_path = p.with_suffix(".pt")

    _require_torch()
    import torch

    torch.save(model.state_dict(), str(pt_path))

    data = dataclasses.asdict(metadata)
    data["state_dict_file"] = pt_path.name
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return str(p)


def load_suppression_model(path: str) -> ReflectionSuppressionModel:
    """`save_suppression_model()`이 만든 YAML+`.pt` 쌍을 다시 읽어 실행
    가능한 모델로 되돌린다. sibling `.pt`가 없으면 조용히 넘어가지 않고
    `FileNotFoundError`를 그대로 낸다(silent fallback 금지, 사용자 스펙 5-F/
    STEP5 안정화 원칙과 동일)."""
    _require_torch()
    import torch

    p = Path(path)
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    pt_path = p.parent / data["state_dict_file"]
    state_dict = torch.load(str(pt_path), map_location="cpu", weights_only=True)

    metadata_fields = {f.name for f in dataclasses.fields(SuppressionModelMetadata)}
    metadata = SuppressionModelMetadata(**{k: v for k, v in data.items() if k in metadata_fields})
    return ReflectionSuppressionModel(state_dict, metadata.encoder_channels, metadata.decoder_channels, metadata.activation)

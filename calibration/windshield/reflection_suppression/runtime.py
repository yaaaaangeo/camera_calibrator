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
    DEFAULT_ALPHA_PRESENCE_THRESHOLD,
    DEFAULT_DECODER_CHANNELS,
    DEFAULT_EDGE_RETENTION_SAFETY_THRESHOLD,
    DEFAULT_ENCODER_CHANNELS,
    DEFAULT_MAX_CORRECTION,
    DEFAULT_MIN_ALPHA_COVERAGE,
    DEFAULT_MIN_ALPHA_P95,
    DEFAULT_MIN_MEAN_ALPHA,
    DEFAULT_SUPPRESSION_STRENGTH,
)
from calibration.windshield.reflection_suppression.model import _require_torch, build_model
from calibration.windshield.reflection_suppression.types import ReflectionSuppressionResult, SuppressionModelMetadata


@dataclasses.dataclass
class SuppressionRuntimeConfig:
    strength: float = DEFAULT_SUPPRESSION_STRENGTH
    max_correction: float = DEFAULT_MAX_CORRECTION

    # Low Reflection Guard(안정화 라운드 항목 8) - mean/P95/coverage 중
    # **전부**가 각자의 threshold보다 낮을 때만 suppression을 skip한다(하나
    # 라도 반사 존재를 시사하면 skip하지 않는다 - "화면의 3%에만 강한
    # 반사가 있는" 경우 global mean만으로 skip하면 그걸 놓친다). 개별
    # threshold를 None으로 두면 그 기준은 판단에서 제외된다.
    min_mean_alpha: Optional[float] = DEFAULT_MIN_MEAN_ALPHA
    min_alpha_p95: Optional[float] = DEFAULT_MIN_ALPHA_P95
    min_alpha_coverage: Optional[float] = DEFAULT_MIN_ALPHA_COVERAGE
    alpha_presence_threshold: float = DEFAULT_ALPHA_PRESENCE_THRESHOLD

    # No-reference edge preservation 진단(안정화 라운드 항목 9) - 미달 시
    # reject가 아니라 warning만 기록한다(ground truth가 아닌 heuristic).
    # None이면 계산하지 않는다.
    edge_retention_safety_threshold: Optional[float] = DEFAULT_EDGE_RETENTION_SAFETY_THRESHOLD


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


def _edge_energy(image_bgr: np.ndarray) -> float:
    """Reference 없이도 계산 가능한 gradient 에너지(Sobel) - No-reference
    edge preservation 진단 전용(사용자 스펙 9번). Ground truth가 아니라
    runtime safety heuristic이다."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


def _should_skip_due_to_low_reflection(
    mean_alpha: float, alpha_p95: float, alpha_coverage: float, cfg: SuppressionRuntimeConfig,
) -> bool:
    """mean/P95/coverage 중 활성화된(threshold가 None이 아닌) 기준이 **전부**
    "반사 없음"을 가리킬 때만 skip한다 - 안정화 라운드 항목 8("global mean만
    낮다고 잘못 skip하지 않는다")."""
    checks = []
    if cfg.min_mean_alpha is not None:
        checks.append(mean_alpha < cfg.min_mean_alpha)
    if cfg.min_alpha_p95 is not None:
        checks.append(alpha_p95 < cfg.min_alpha_p95)
    if cfg.min_alpha_coverage is not None:
        checks.append(alpha_coverage < cfg.min_alpha_coverage)
    if not checks:
        return False
    return all(checks)


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
        alpha_p95 = float(np.percentile(alpha, 95.0))
        max_alpha = float(np.max(alpha))
        alpha_coverage = float(np.mean(alpha > cfg.alpha_presence_threshold))

        if _should_skip_due_to_low_reflection(mean_alpha, alpha_p95, alpha_coverage, cfg):
            return ReflectionSuppressionResult(
                success=True,
                suppressed_image=image_bgr.copy(),
                reflection_layer=(reflection * 255.0).astype(np.uint8),
                alpha_map=alpha.astype(np.float32),
                suppression_strength=cfg.strength,
                mean_alpha=mean_alpha,
                alpha_p95=alpha_p95,
                max_alpha=max_alpha,
                alpha_coverage=alpha_coverage,
                mean_correction=0.0,
                max_correction=0.0,
                confidence=mean_alpha,  # deprecated alias
                skipped_due_to_low_reflection=True,
                warning_message=(
                    "Mean/P95/coverage of predicted alpha are all below their thresholds - "
                    "suppression skipped, original image returned unchanged."
                ),
            )

        raw_correction = cfg.strength * alpha[..., None] * reflection
        correction = np.clip(raw_correction, 0.0, max(cfg.max_correction, 0.0))
        mean_correction = float(np.mean(correction))
        max_correction_actual = float(np.max(correction))

        suppressed_float = np.clip(img_float - correction, 0.0, 1.0)
        suppressed_image = np.clip(suppressed_float * 255.0, 0, 255).astype(np.uint8)

        edge_retention_estimate = None
        warning_message = None
        if cfg.edge_retention_safety_threshold is not None:
            input_edge_energy = _edge_energy(image_bgr)
            output_edge_energy = _edge_energy(suppressed_image)
            edge_retention_estimate = output_edge_energy / (input_edge_energy + 1e-6)
            if edge_retention_estimate < cfg.edge_retention_safety_threshold:
                warning_message = (
                    f"No-reference edge retention estimate ({edge_retention_estimate:.2f}) is below the "
                    f"safety threshold ({cfg.edge_retention_safety_threshold:.2f}) - suppression may be "
                    "removing scene detail (heuristic, not ground truth; output is still returned)."
                )

        return ReflectionSuppressionResult(
            success=True,
            suppressed_image=suppressed_image,
            reflection_layer=(reflection * 255.0).astype(np.uint8),
            alpha_map=alpha.astype(np.float32),
            suppression_strength=cfg.strength,
            mean_alpha=mean_alpha,
            alpha_p95=alpha_p95,
            max_alpha=max_alpha,
            alpha_coverage=alpha_coverage,
            mean_correction=mean_correction,
            max_correction=max_correction_actual,
            edge_retention_estimate=edge_retention_estimate,
            confidence=mean_alpha,  # deprecated alias
            warning_message=warning_message,
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

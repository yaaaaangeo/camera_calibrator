"""
calibration.windshield.reflection_suppression.types
==============================================================

STEP 7 - Reflection Suppression 결과 타입(torch-free).

Reflection Evaluation(STEP 6)의 결과 타입을 재사용/합성하되, Suppression
고유의 photometric restoration 결과는 별도 타입으로 둔다(Evaluation
package에 억지로 끼워 넣지 않는다 - 사용자 스펙 0/57번).

큰 이미지 array(suppressed_image/reflection_layer/alpha_map)는
`.ccproj` 프로젝트 JSON에 직접 저장하지 않는다(사용자 스펙 55번) - UI가
표시/미리보기 용도로만 메모리에 들고 있고, 영구 저장이 필요하면 별도
이미지 파일로 내보낸다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from calibration.windshield.reflection.types import ReflectionEvaluationResult


@dataclass
class ReflectionSuppressionResult:
    success: bool

    suppressed_image: Optional[np.ndarray] = None      # uint8 BGR, HxWx3
    reflection_layer: Optional[np.ndarray] = None       # uint8 BGR, HxWx3 (예측된 reflection layer, alpha 적용 전)
    alpha_map: Optional[np.ndarray] = None              # float32, HxW, [0,1]

    suppression_strength: float = 0.0
    mean_alpha: float = 0.0
    max_alpha: float = 0.0
    mean_correction: float = 0.0
    max_correction: float = 0.0
    confidence: Optional[float] = None

    skipped_due_to_low_confidence: bool = False
    fell_back_to_original: bool = False

    warning_message: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class ReflectionSuppressionEvaluation:
    """STEP 6 evaluator를 suppression 전/후에 동일하게 적용한 결과(사용자
    스펙 41/56번) - before/after 둘 다 `ReflectionEvaluationResult`이고,
    이 타입은 그 둘의 차이(reduction/retention)만 추가로 요약한다."""
    before: ReflectionEvaluationResult
    after: ReflectionEvaluationResult

    reflection_mean_reduction: Optional[float] = None
    reflection_p95_reduction: Optional[float] = None
    coverage_reduction: Optional[float] = None

    edge_retention_after: Optional[float] = None
    contrast_retention_after: Optional[float] = None

    # Clean(=reflection이 거의 없다고 알려진) ROI에서 suppression이 원본을
    # 얼마나 불필요하게 바꿨는지(사용자 스펙 47번) - 낮을수록 좋다.
    over_suppression_score: Optional[float] = None

    success: bool = True
    warning_message: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class SuppressionModelMetadata:
    """state_dict와 별도로 저장되는 재구성 메타데이터(사용자 스펙 55/61/62번)
    - SciPy/PyTorch 내부 객체가 아니라 재구성에 필요한 public 값만 담는다."""
    model_version: int = 1
    architecture: str = "small_unet"
    input_channels: int = 3
    output_channels: int = 4
    encoder_channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    decoder_channels: list[int] = field(default_factory=lambda: [128, 64, 32])
    activation: str = "silu"
    normalization: str = "divide_by_255"
    training_resolution: int = 256
    training_seed: int = 42
    max_correction: float = 0.30
    default_strength: float = 0.8
    best_epoch: Optional[int] = None
    best_val_loss: Optional[float] = None
    dataset_num_real_pairs: int = 0
    dataset_num_synthetic_pairs: int = 0
    dataset_num_identity_pairs: int = 0

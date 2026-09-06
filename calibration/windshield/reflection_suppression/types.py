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

    # Alpha는 "reflection contribution/mask strength"이지 신경망의 epistemic
    # confidence가 아니다(안정화 라운드 항목 7) - mean 하나만으로는 화면
    # 일부에만 있는 강한 반사를 놓칠 수 있어 robust statistic을 함께
    # 기록한다(항목 8).
    mean_alpha: float = 0.0
    alpha_p95: float = 0.0
    max_alpha: float = 0.0
    alpha_coverage: float = 0.0  # alpha > alpha_presence_threshold인 픽셀 비율

    mean_correction: float = 0.0
    max_correction: float = 0.0

    # No-reference 상황에서도 계산 가능한 lightweight 진단(항목 9) - reject
    # 기준이 아니라 warning 트리거용 heuristic이다(ground truth가 아니다).
    edge_retention_estimate: Optional[float] = None

    # Deprecated(하위 호환용 별칭, mean_alpha와 동일한 값) - "confidence"라는
    # 이름이 신경망의 예측 확신도로 오해될 수 있어 새 코드에서는 쓰지 않는다.
    # UI/새 guard 로직은 이 필드를 참조하지 않는다.
    confidence: Optional[float] = None

    skipped_due_to_low_reflection: bool = False
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

    # Reference Mode 전용(안정화 라운드 항목 2) - Reference가 없으면(No-Reference
    # 모드) 이 셋은 항상 None으로 남는다. STEP 6의 "No-reference는 heuristic
    # likelihood일 뿐 ground-truth reflection measurement가 아니다"라는 원칙을
    # 여기서도 깨뜨리지 않기 위함이다 - No-Reference 결과의 mean_strength/
    # p95_strength/coverage를 이 필드들로 잘못 재활용하지 않는다.
    reflection_mean_reduction: Optional[float] = None
    reflection_p95_reduction: Optional[float] = None
    coverage_reduction: Optional[float] = None

    # No-Reference Mode 전용 - Reference Mode에서는 항상 None으로 남는다.
    # 이름 자체가 "heuristic likelihood"임을 분명히 한다(실제 reflection
    # 양이라고 주장하지 않는다).
    reflection_likelihood_before: Optional[float] = None
    reflection_likelihood_after: Optional[float] = None
    reflection_likelihood_reduction: Optional[float] = None

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
    # 안정화 라운드 항목 3/12 - 실제 paired dataset 학습 entrypoint
    # (training.train_suppression_from_dataset)가 채우는 split별 표본 수.
    # Synthetic-only CLI 경로에서는 0으로 남는다.
    train_sample_count: int = 0
    validation_sample_count: int = 0
    test_sample_count: int = 0

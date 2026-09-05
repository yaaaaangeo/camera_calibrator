"""
calibration.windshield.reflection_suppression.config
==============================================================

STEP 7 - Reflection Suppression 전용 순수 Python 상수/런타임 설정.

이 모듈은 PyTorch를 import하지 않는다(calibration.windshield.neural_config와
동일한 원칙, STEP 5에서 확립됨) - UI가 이 값들을 표시하는 것만으로 PyTorch가
로드되면 안 된다. `model.py`/`runtime.py`/`training.py`만 실제 사용 시점에
torch를 lazy import한다.

임의로 고른 threshold(예: max_correction=0.30)는 "과학적으로 검증된 값"이
아니라 initial default일 뿐이다 - 실제 데이터로 조정해야 한다(사용자 스펙
7/37번, "절대 scientific truth라고 주장하지 않는다").
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Architecture(고정 - 이번 STEP 7 첫 버전에서는 architecture search를 하지
# 않는다. CNN이 아니라 "얼마나 작은 U-Net으로 되는지"를 검증하는 단계다).
# ---------------------------------------------------------------------------
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 4  # 3 (reflection layer, BGR) + 1 (alpha)
DEFAULT_ENCODER_CHANNELS: list[int] = [32, 64, 128, 256]
DEFAULT_DECODER_CHANNELS: list[int] = [128, 64, 32]
DEFAULT_ACTIVATION = "silu"

ACTIVATION_TO_CODE: dict[str, float] = {"silu": 0.0, "leaky_relu": 1.0}
CODE_TO_ACTIVATION: dict[float, str] = {0.0: "silu", 1.0: "leaky_relu"}

# ---------------------------------------------------------------------------
# Runtime safety guard 기본값(사용자 스펙 7/38/39/40번)
# ---------------------------------------------------------------------------
DEFAULT_MAX_CORRECTION = 0.30
DEFAULT_MIN_CONFIDENCE = 0.02
DEFAULT_EDGE_RETENTION_SAFETY_THRESHOLD = 0.75

# Suppression Strength 프리셋(사용자 스펙 36/37번) - 전부 실데이터로 조정될
# empirical config다.
SUPPRESSION_STRENGTH_CONSERVATIVE = 0.6
SUPPRESSION_STRENGTH_STANDARD = 0.8
SUPPRESSION_STRENGTH_STRONG = 1.0
DEFAULT_SUPPRESSION_STRENGTH = SUPPRESSION_STRENGTH_STANDARD

# ---------------------------------------------------------------------------
# Training 기본값
# ---------------------------------------------------------------------------
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_EPOCHS = 200
DEFAULT_PATIENCE = 20
DEFAULT_SEED = 42
DEFAULT_TRAIN_RESOLUTION = 256
DEFAULT_VALIDATION_RATIO = 0.2  # synthetic pretrain 내부 train/val 분리에만 사용

# Loss weight 기본값(사용자 스펙 26-33번)
DEFAULT_LAMBDA_CLEAN = 1.0
DEFAULT_LAMBDA_REFLECTION = 0.5
DEFAULT_LAMBDA_ALPHA = 0.2
DEFAULT_LAMBDA_EDGE = 0.5
DEFAULT_LAMBDA_IDENTITY = 1.0
DEFAULT_LAMBDA_SMOOTH = 0.05
DEFAULT_LAMBDA_SPARSE = 0.01
CHARBONNIER_EPS = 1e-3

# 실제 paired dataset 학습에 허용하는 STEP 6 alignment status(사용자 스펙
# 17번, "GOOD -> Train 사용, WARNING -> configurable, INVALID -> 제외").
ALIGNMENT_STATUSES_ALWAYS_ALLOWED: tuple[str, ...] = ("good",)
ALIGNMENT_STATUSES_ALLOWED_WITH_WARNING: tuple[str, ...] = ("good", "warning")


def allowed_alignment_statuses(allow_warning: bool = False) -> tuple[str, ...]:
    return ALIGNMENT_STATUSES_ALLOWED_WITH_WARNING if allow_warning else ALIGNMENT_STATUSES_ALWAYS_ALLOWED

"""
calibration.windshield.reflection_suppression
==============================================================

STEP 7 - Windshield Reflection Suppression (photometric restoration).

이 패키지는 `calibration.windshield.reflection`(STEP 6, Reflection
Evaluation)과 완전히 독립적이다 - Evaluation은 PyTorch 없이 NumPy/OpenCV만
사용하고, 이 패키지의 `model.py`/`runtime.py`/`training.py`만 실제 사용
시점에 PyTorch를 lazy import한다(`config.py`/`types.py`/`synthetic.py`/
`dataset.py`는 torch-free).

Public API:

    suppress_reflection(image_bgr, model, config) -> ReflectionSuppressionResult
    evaluate_suppression(normal_image, suppression_result, reference_image, ...)
        -> ReflectionSuppressionEvaluation
"""

from __future__ import annotations

from calibration.windshield.reflection_suppression.evaluation import evaluate_suppression
from calibration.windshield.reflection_suppression.runtime import (
    ReflectionSuppressionModel,
    SuppressionRuntimeConfig,
    load_suppression_model,
    save_suppression_model,
    suppress_reflection,
)
from calibration.windshield.reflection_suppression.types import (
    ReflectionSuppressionEvaluation,
    ReflectionSuppressionResult,
    SuppressionModelMetadata,
)

__all__ = [
    "suppress_reflection",
    "evaluate_suppression",
    "ReflectionSuppressionModel",
    "SuppressionRuntimeConfig",
    "ReflectionSuppressionResult",
    "ReflectionSuppressionEvaluation",
    "SuppressionModelMetadata",
    "save_suppression_model",
    "load_suppression_model",
]

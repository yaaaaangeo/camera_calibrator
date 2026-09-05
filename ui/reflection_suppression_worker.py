"""
camera_calibrator.ui.reflection_suppression_worker
==============================================================

STEP 7 - Reflection Suppression을 위한 QThread worker.

`ui/reflection_worker.py`(STEP 6, Evaluation 전용)와 별도로 둔다(사용자
스펙 0/58번, Evaluation != Suppression) - 이 worker는 오직 inference만
수행한다(사용자 스펙 59번, "Training까지 GUI에 넣으면 복잡도가 크게
증가한다" - Training은 CLI/스크립트 전용, GUI는 이미 학습된 model.yml을
불러와 inference/evaluation만 한다).

`calibration.windshield.reflection_suppression`은 PyTorch를 필요로 하지만,
이 파일 자체는 그 패키지를 함수 본문 안에서만 import한다 - 다른 모든 lazy
import 규칙과 동일하게, Qt main thread에서 neural inference를 절대 돌리지
않는다(worker.run()이 QThread 안에서 실행됨)."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Signal


class ReflectionSuppressionWorker(QObject):
    progress = Signal(str)
    result_ready = Signal(object)   # ReflectionSuppressionResult
    error = Signal(str)
    finished = Signal()

    def __init__(self, model_path: str, image_bgr: np.ndarray, strength: float, max_correction: float):
        super().__init__()
        self._model_path = model_path
        self._image_bgr = image_bgr
        self._strength = strength
        self._max_correction = max_correction

    def run(self) -> None:
        try:
            self.progress.emit("Loading reflection suppression model...")
            from calibration.windshield.reflection_suppression.runtime import (
                SuppressionRuntimeConfig,
                load_suppression_model,
                suppress_reflection,
            )

            model = load_suppression_model(self._model_path)
            self.progress.emit("Running reflection suppression...")
            cfg = SuppressionRuntimeConfig(strength=self._strength, max_correction=self._max_correction)
            result = suppress_reflection(self._image_bgr, model, cfg)
            self.result_ready.emit(result)
        except ImportError as e:
            self.error.emit(str(e))
        except Exception as e:  # noqa: BLE001 - shown directly in the UI
            self.error.emit(f"Reflection suppression failed: {e}")
        finally:
            self.finished.emit()

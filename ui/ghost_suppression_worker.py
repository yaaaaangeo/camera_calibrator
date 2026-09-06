"""
camera_calibrator.ui.ghost_suppression_worker
==============================================================

STEP 8B - Ghost Suppression을 위한 QThread worker.

`ui/reflection_suppression_worker.py`와 완전히 독립된 worker다 - Ghost
Suppression은 Reflection Suppression 모델을 절대 재사용하지 않는다(사용자
스펙 37번). Deterministic iterative reconstruction이라 PyTorch가 필요 없고
가볍지만, Before/After 비교를 위해 evaluator를 두 번 돌리는 비용이 있어
Qt main thread 밖에서 실행한다.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Signal

from calibration.windshield.ghost.evaluator import evaluate_ghost_point_source
from calibration.windshield.ghost.suppression import load_ghost_model, suppress_ghost
from calibration.windshield.ghost.types import GhostEvaluationConfig, GhostField, GhostSuppressionEvaluation


class GhostSuppressionWorker(QObject):
    progress = Signal(str)
    result_ready = Signal(object)  # (GhostSuppressionResult, GhostSuppressionEvaluation) tuple
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        model_path: str | None,
        image_bgr: np.ndarray,
        eval_config: GhostEvaluationConfig,
        *,
        ghost_field: GhostField | None = None,
        iterations: int | None = None,
        max_correction: float | None = None,
    ):
        super().__init__()
        self._model_path = model_path
        self._ghost_field = ghost_field
        self._image_bgr = image_bgr
        self._eval_config = eval_config
        self._iterations = iterations
        self._max_correction = max_correction

    def run(self) -> None:
        try:
            if self._ghost_field is not None:
                ghost_field = self._ghost_field
            else:
                self.progress.emit("Loading ghost model...")
                ghost_field = load_ghost_model(self._model_path)

            kwargs = {}
            if self._iterations is not None:
                kwargs["iterations"] = self._iterations
            if self._max_correction is not None:
                kwargs["max_correction"] = self._max_correction

            self.progress.emit("Evaluating ghost before suppression...")
            before = evaluate_ghost_point_source(self._image_bgr, self._eval_config)

            self.progress.emit("Running ghost suppression...")
            supp = suppress_ghost(self._image_bgr, ghost_field, **kwargs)

            if supp.success and supp.suppressed_image is not None:
                self.progress.emit("Evaluating ghost after suppression...")
                after = evaluate_ghost_point_source(supp.suppressed_image, self._eval_config)
            else:
                after = before

            strength_reduction = None
            detection_reduction = None
            if before.mean_strength_ratio is not None and after.mean_strength_ratio is not None:
                strength_reduction = before.mean_strength_ratio - after.mean_strength_ratio
            if before.detection_count is not None and after.detection_count is not None:
                detection_reduction = before.detection_count - after.detection_count

            result = GhostSuppressionEvaluation(
                before=before,
                after=after,
                strength_reduction=strength_reduction,
                detection_reduction=detection_reduction,
                success=supp.success,
                warning_message=supp.warning_message,
                error_message=supp.error_message,
            )
            self.result_ready.emit((supp, result))
        except FileNotFoundError as e:
            self.error.emit(f"Ghost model file not found: {e}")
        except Exception as e:  # noqa: BLE001 - shown directly in the UI
            self.error.emit(f"Ghost suppression failed: {e}")
        finally:
            self.finished.emit()

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from calibration.windshield.reflection import (
    ReflectionDatasetResult,
    ReflectionEvaluationConfig,
    ReflectionImagePair,
    evaluate_reflection_dataset,
)


class ReflectionEvaluationWorker(QObject):
    progress = Signal(str)
    result_ready = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, pairs: list[ReflectionImagePair], config: ReflectionEvaluationConfig):
        super().__init__()
        self._pairs = pairs
        self._config = config

    def run(self) -> None:
        try:
            self.progress.emit(f"Evaluating reflection ({len(self._pairs)} pair(s))...")
            result: ReflectionDatasetResult = evaluate_reflection_dataset(self._pairs, self._config)
            self.result_ready.emit(result)
        except Exception as e:  # noqa: BLE001 - shown directly in the UI
            self.error.emit(f"Reflection evaluation failed: {e}")
        finally:
            self.finished.emit()

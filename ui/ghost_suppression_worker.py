"""
camera_calibrator.ui.ghost_suppression_worker
==============================================================

STEP 8B - Ghost Suppression을 위한 QThread worker.

`ui/reflection_suppression_worker.py`와 완전히 독립된 worker다 - Ghost
Suppression은 Reflection Suppression 모델을 절대 재사용하지 않는다(사용자
스펙 37번). Deterministic iterative reconstruction이라 PyTorch가 필요 없고
가볍지만, Before/After 비교를 위해 evaluator를 두 번 돌리는 비용이 있어
Qt main thread 밖에서 실행한다.

STEP 8 stabilization 8번 - Before/After 평가를 `evaluate_ghost_point_source`
로 하드코딩하지 않고, Evaluation Worker와 동일한 공용 dispatcher
(`evaluate_ghost_image`)를 쓴다 - Point Source/Edge Target/General
Likelihood 어떤 모드로 Evaluation을 돌렸든 Suppression Before/After도
같은 mode-aware evaluator로 비교한다.

STEP 8 semantic fix 2번 - Suppression summary는 mode에 따라 의미가 다른
필드를 채운다: Point Source/Edge Target은 `strength_reduction`(+
`detection_reduction`)을 primary metric으로 쓰고, General(No-Reference)
Likelihood는 `likelihood_reduction`(= before.ghost_likelihood -
after.ghost_likelihood)을 쓴다. 두 그룹을 절대 같은 필드에 섞지 않는다 -
이 mode 분기 로직 자체는 `suppression.py::build_suppression_evaluation()`
(Qt 비의존 순수 함수)에 있고, 이 worker는 그 함수를 호출하기만 한다 -
PySide6 없이도 pytest로 mode별 분기를 직접 검증할 수 있다.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Signal

from calibration.windshield.ghost.evaluator import evaluate_ghost_image
from calibration.windshield.ghost.suppression import build_suppression_evaluation, load_ghost_model, suppress_ghost
from calibration.windshield.ghost.types import GhostEvaluationConfig, GhostField


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
            before = evaluate_ghost_image(self._image_bgr, self._eval_config)

            self.progress.emit("Running ghost suppression...")
            supp = suppress_ghost(self._image_bgr, ghost_field, **kwargs)

            if supp.success and supp.suppressed_image is not None:
                self.progress.emit("Evaluating ghost after suppression...")
                after = evaluate_ghost_image(supp.suppressed_image, self._eval_config)
            else:
                after = before

            result = build_suppression_evaluation(before, after, supp)
            self.result_ready.emit((supp, result))
        except FileNotFoundError as e:
            self.error.emit(f"Ghost model file not found: {e}")
        except Exception as e:  # noqa: BLE001 - shown directly in the UI
            self.error.emit(f"Ghost suppression failed: {e}")
        finally:
            self.finished.emit()

"""
camera_calibrator.ui.ghost_evaluation_worker
==============================================================

STEP 8A - Ghost Evaluation을 위한 QThread worker.

`ui/reflection_worker.py`(Reflection Evaluation)와 완전히 독립된 worker다
- Ghost는 Reflection과 다른 image formation model을 갖는 별도 기능이므로
(사용자 스펙 1번), worker 레벨에서도 절대 섞지 않는다. Point-source/edge/
general-likelihood 평가는 모두 순수 NumPy/OpenCV라 가볍지만, 여러 프레임을
한 번에 돌리는 dataset 평가는 Qt main thread를 막을 수 있으므로 QThread
안에서 실행한다.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, Signal

from calibration.types import CameraModelType
from calibration.windshield.ghost.evaluator import (
    evaluate_ghost_dataset,
    evaluate_ghost_general_likelihood,
    evaluate_ghost_point_source,
)
from calibration.windshield.ghost.types import GhostEvaluationConfig


class GhostEvaluationWorker(QObject):
    progress = Signal(str)
    result_ready = Signal(object)  # GhostDatasetResult
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        images_bgr: list[np.ndarray],
        config: GhostEvaluationConfig,
        *,
        camera_matrix: Optional[np.ndarray] = None,
        distortion: Optional[np.ndarray] = None,
        camera_model: Optional[CameraModelType] = None,
        frame_ids: Optional[list[str]] = None,
    ):
        super().__init__()
        self._images = images_bgr
        self._config = config
        self._camera_matrix = camera_matrix
        self._distortion = distortion
        self._camera_model = camera_model
        self._frame_ids = frame_ids or [str(i) for i in range(len(images_bgr))]

    def run(self) -> None:
        try:
            self.progress.emit(f"Evaluating ghost ({len(self._images)} frame(s))...")
            per_frame = []
            for image, frame_id in zip(self._images, self._frame_ids):
                if self._config.mode == "general_likelihood":
                    res = evaluate_ghost_general_likelihood(image, self._config, pair_id=frame_id)
                else:
                    res = evaluate_ghost_point_source(
                        image,
                        self._config,
                        camera_matrix=self._camera_matrix,
                        distortion=self._distortion,
                        camera_model=self._camera_model,
                        pair_id=frame_id,
                    )
                per_frame.append(res)
            dataset_result = evaluate_ghost_dataset(per_frame, mode=self._config.mode)
            self.result_ready.emit(dataset_result)
        except Exception as e:  # noqa: BLE001 - shown directly in the UI
            self.error.emit(f"Ghost evaluation failed: {e}")
        finally:
            self.finished.emit()

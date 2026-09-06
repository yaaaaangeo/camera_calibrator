"""
camera_calibrator.ui.ghost_evaluation_worker
==============================================================

STEP 8A - Ghost Evaluation을 위한 QThread worker.

`ui/reflection_worker.py`(Reflection Evaluation)와 완전히 독립된 worker다
- Ghost는 Reflection과 다른 image formation model을 갖는 별도 기능이므로
(사용자 스펙 1번), worker 레벨에서도 절대 섞지 않는다.

STEP 8 stabilization 2/3번 - Point-source/Edge-target/General-likelihood
세 모드 모두 공용 dispatcher(`evaluate_ghost_image`)를 통해 실행한다(중복
if/else 제거, 사용자 스펙 8번).

이 worker 자체는 아주 얇다 - 파일 로딩 + resolution consistency gate(STEP
8 semantic/safety fix 4번, 서로 다른 해상도 프레임을 섞지 않는다) + 평가 +
집계 전체를 `evaluate_ghost_dataset_from_paths()`(Qt 비의존 순수 함수)에
위임한다. Qt main thread에는 파일 선택/설정 읽기/결과 표시만 남긴다(사용자
스펙 2-E번).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, Signal

from calibration.types import CameraModelType
from calibration.windshield.ghost.evaluator import evaluate_ghost_dataset_from_paths
from calibration.windshield.ghost.types import GhostEvaluationConfig


class GhostEvaluationWorker(QObject):
    progress = Signal(str)
    result_ready = Signal(object)  # GhostDatasetResult
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        image_paths: list[str],
        config: GhostEvaluationConfig,
        *,
        camera_matrix: Optional[np.ndarray] = None,
        distortion: Optional[np.ndarray] = None,
        camera_model: Optional[CameraModelType] = None,
        frame_ids: Optional[list[str]] = None,
    ):
        super().__init__()
        self._image_paths = image_paths
        self._config = config
        self._camera_matrix = camera_matrix
        self._distortion = distortion
        self._camera_model = camera_model
        self._frame_ids = frame_ids or [str(i) for i in range(len(image_paths))]

    def run(self) -> None:
        try:
            dataset_result = evaluate_ghost_dataset_from_paths(
                self._image_paths, self._config,
                frame_ids=self._frame_ids,
                camera_matrix=self._camera_matrix, distortion=self._distortion, camera_model=self._camera_model,
                progress_callback=self.progress.emit,
            )
            self.result_ready.emit(dataset_result)
        except Exception as e:  # noqa: BLE001 - shown directly in the UI (includes mixed-resolution gate messages)
            self.error.emit(f"Ghost evaluation failed: {e}")
        finally:
            self.finished.emit()

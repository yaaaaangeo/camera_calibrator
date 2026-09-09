"""
camera_calibrator.ui.kfold_worker
==============================================================

Repeated K-Fold(논문용 Multi-Metric raw statistics)를 위한 QThread worker.

`ui/worker.py`의 CrossDatasetValidationWorker와 같은 얇은 패턴을 따른다 -
실제 계산은 전부 `calibration/kfold.py::run_repeated_kfold_all_models()`
(Qt 비의존 순수 함수)에 위임하고, 이 worker는 QThread 경계를 넘기기 위한
signal 배선만 담당한다. K=5 x Repeats=5(기본값)만 해도 fold 25개 x 3모델 =
75번의 cv2.calibrateCamera/solvePnP 호출이라 GUI 스레드를 막지 않도록
반드시 별도 스레드에서 실행해야 한다(ui/worker.py 상단 docstring이 설명하는
GIL 관련 이유와 동일 - 다만 이 worker는 기존 파이프라인 worker처럼 별도
OS 프로세스까지는 쓰지 않는다: Repeated K-Fold는 사용자가 필요할 때만
선택적으로 실행하는 부가 분석이고, run_repeated_kfold_all_models 내부의
ThreadPoolExecutor(n_jobs)로도 이미 충분히 빨라질 수 있어 우선 QThread로
충분한지 검증한 뒤 필요해지면 프로세스 분리를 검토한다).
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from calibration.kfold import run_repeated_kfold_all_models
from calibration.types import CameraConfig, CameraModelType, Dataset, PatternConfig, RepeatedKFoldResult


class RepeatedKFoldWorker(QObject):
    progress = Signal(str)
    results_ready = Signal(dict)  # dict[CameraModelType, RepeatedKFoldResult]
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        dataset: Dataset,
        camera_config: CameraConfig,
        pattern_config: PatternConfig,
        k: int,
        n_repeats: int,
        base_seed: int = 42,
        n_jobs: int = 1,
    ):
        super().__init__()
        self._dataset = dataset
        self._camera_config = camera_config
        self._pattern_config = pattern_config
        self._k = k
        self._n_repeats = n_repeats
        self._base_seed = base_seed
        self._n_jobs = n_jobs

    def run(self) -> None:
        try:
            self.progress.emit(
                f"Repeated {self._k}-Fold x {self._n_repeats} 실행 중 "
                f"(Brown-Conrady/Rational/Fisheye, 총 {self._k * self._n_repeats * 3} fold 평가)..."
            )
            results: dict[CameraModelType, RepeatedKFoldResult] = run_repeated_kfold_all_models(
                self._dataset, self._camera_config, self._pattern_config,
                k=self._k, n_repeats=self._n_repeats, base_seed=self._base_seed, n_jobs=self._n_jobs,
            )
            self.results_ready.emit(results)
        except Exception as e:  # noqa: BLE001 - UI에 그대로 표시
            self.error.emit(f"Repeated K-Fold failed: {e}")
        finally:
            self.finished.emit()

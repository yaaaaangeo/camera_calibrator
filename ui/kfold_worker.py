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

진행률 보고(progress reporting): 75 fold가 끝날 때까지 UI가 완전히 비어
있으면 사용자가 "멈춘 건지 계산 중인지" 알 수 없다는 문제를 해결하기 위해,
run_repeated_kfold_all_models()의 optional progress_callback을 이 worker의
run() 안에서 넘긴다. 그 콜백은 (fold split/seed/model fairness/metric 계산
방식과는 완전히 무관하게) 이미 계산된 fold 하나가 끝날 때마다 호출되는
Qt-독립적인 calibration.kfold.KFoldProgressEvent를 받는데, 이 worker는 그걸
그대로 `progress_event` signal(object 하나)에 실어 GUI 스레드로 넘긴다 -
Qt QueuedConnection이 스레드 경계를 알아서 처리하므로 여기서 추가 동기화가
필요 없다(기존 progress(str)/results_ready(dict)와 동일한 패턴).
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from calibration.kfold import KFoldProgressEvent, run_repeated_kfold_all_models
from calibration.types import CameraConfig, CameraModelType, Dataset, PatternConfig, RepeatedKFoldResult


class RepeatedKFoldWorker(QObject):
    progress = Signal(str)  # 상태바용 사람이 읽는 한 줄 요약(기존과 동일하게 유지)
    progress_event = Signal(object)  # calibration.kfold.KFoldProgressEvent - 구조화된 진행 정보
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

    def _on_progress(self, event: KFoldProgressEvent) -> None:
        # calibration/kfold.py가 이미 이 콜백 호출 자체를 try/except로 감싸므로
        # (progress callback이 계산을 절대 실패시키지 않는다는 요구사항) 여기서
        # 다시 감쌀 필요는 없지만, Qt emit 자체가 예외를 낼 가능성까지 방어한다.
        try:
            self.progress_event.emit(event)
            self.progress.emit(_format_progress_text(event, self._k, self._n_repeats))
        except Exception:  # noqa: BLE001 - progress 표시 실패가 계산을 막으면 안 됨
            pass

    def run(self) -> None:
        try:
            self.progress.emit(
                f"Repeated {self._k}-Fold x {self._n_repeats} 실행 중 "
                f"(Brown-Conrady/Rational/Fisheye, 총 {self._k * self._n_repeats * 3} fold 평가)..."
            )
            results: dict[CameraModelType, RepeatedKFoldResult] = run_repeated_kfold_all_models(
                self._dataset, self._camera_config, self._pattern_config,
                k=self._k, n_repeats=self._n_repeats, base_seed=self._base_seed, n_jobs=self._n_jobs,
                progress_callback=self._on_progress,
            )
            self.results_ready.emit(results)
        except Exception as e:  # noqa: BLE001 - UI에 그대로 표시
            self.error.emit(f"Repeated K-Fold failed: {e}")
        finally:
            self.finished.emit()


_MODEL_TEXT_LABELS = {
    CameraModelType.BROWN_CONRADY: "Brown-Conrady",
    CameraModelType.EXTENDED_PINHOLE: "Rational",
    CameraModelType.FISHEYE: "Fisheye",
}


def _format_progress_text(event: KFoldProgressEvent, k: int, n_repeats: int) -> str:
    """progress(str) 상태바용 한 줄 요약. 실시간 표는 progress_event(구조화된
    KFoldProgressEvent)를 직접 구독하는 result_view가 그리므로, 이 문자열은
    "멈춘 게 아니라 지금 여기까지 진행 중이다"를 빠르게 알려주는 보조 텍스트다."""
    label = _MODEL_TEXT_LABELS.get(event.model, event.model.value if event.model else "")
    pct = (event.completed_folds / event.total_folds * 100.0) if event.total_folds else 0.0
    if event.stage == "model_started":
        return f"Repeated {k}-Fold x {n_repeats}: {label} 시작 ({event.completed_folds}/{event.total_folds} folds)"
    if event.stage == "fold_completed":
        return (
            f"Repeated {k}-Fold x {n_repeats}: {event.completed_folds}/{event.total_folds} folds "
            f"completed ({pct:.0f}%) - {label} {event.model_completed_folds}/{event.model_total_folds}"
        )
    if event.stage == "model_completed":
        return f"Repeated {k}-Fold x {n_repeats}: {label} 완료 ({event.model_completed_folds}/{event.model_total_folds})"
    if event.stage == "all_completed":
        return f"Repeated {k}-Fold x {n_repeats} 완료: {event.completed_folds}/{event.total_folds} folds"
    return f"Repeated {k}-Fold x {n_repeats}: {event.completed_folds}/{event.total_folds} folds"

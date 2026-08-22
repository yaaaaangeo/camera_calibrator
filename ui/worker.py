"""
camera_calibrator.ui.worker
===============================

캘리브레이션 파이프라인(검출 -> 품질분석 -> 3모델 학습 -> Hold-out -> 추천)은
이미지 수십 장 기준으로 수 초가 걸릴 수 있어, UI 스레드를 막지 않도록
QThread 워커로 분리한다.

이 파일은 순전히 "언제 무엇을 호출하고 어떤 신호로 알릴지"만 담당하고,
실제 계산 로직은 전부 calibration/*.py의 기존 함수를 그대로 호출한다.
UI 계층이 계산 로직을 재구현하지 않는다는 원칙을 지키기 위함.

--- QThread로 충분하지 않은 부분 (실제 사용자 버그) ---

검출(detect_dataset)은 이미 ProcessPoolExecutor로 병렬화돼 있어 문제 없다.
그런데 그 다음 3모델 계산(run_all_models)/Hold-out 검증(validate_all_models)은
cv2.calibrateCamera/cv2.fisheye.calibrate 같은 C++ 확장 함수를 오래 호출하는데,
이 함수들이 계산 도중 파이썬 GIL을 놓아준다는 보장이 없다. QThread는 같은
프로세스 안의 스레드일 뿐이라 GIL은 프로세스 전체에 하나 - 계산 스레드가
GIL을 계속 붙들면 GUI 스레드는 Qt 이벤트를 처리할 파이썬 코드를 돌릴 GIL을
못 얻어 그대로 멈춘다. 이미지 수백 장 + Rational model처럼 계산이 몇 초~
몇십 초 걸리면 OS가 이걸 "python3 is not responding"으로 표시한다.

그래서 이 두 계산은 calibration/pipeline_process.py를 통해 완전히 별도의
OS 프로세스에서 실행한다 - 그 프로세스는 자기만의 GIL을 가지므로 부모
프로세스(GUI)의 GIL은 계산 시간과 무관하게 항상 비어 있다.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

from PySide6.QtCore import QObject, QThread, Signal

from calibration.types import (
    CameraConfig,
    CameraModelType,
    Dataset,
    PatternConfig,
)
from calibration.detector import detect_dataset
from calibration.quality import analyze_dataset_quality
from calibration.frame_quality import compute_frame_quality_scores
from calibration.models.common import infer_image_size
from calibration.recommender import compute_model_scores, build_recommendation_message
from calibration.self_check import run_all_self_checks
from calibration.rosbag_reader import extract_images_from_bag
from calibration.pipeline_process import (
    run_models_and_validation,
    run_outlier_pruning_and_validation,
)

# 위 두 계산(3모델 + Hold-out 진행 상황)은 자식 프로세스 안에서 일어나므로
# 세부 진행률 문자열을 실시간으로 받을 수 없다 - future.result()를 이 간격
# 으로 짧게 타임아웃 걸어 반복 폴링하면서 "아직 죽지 않았다"는 하트비트만
# 상태바에 남긴다 (실제로 몇 초~몇십 초 걸릴 수 있는 계산이라, 아무 표시도
# 없으면 사용자가 또 "멈췄나?"하고 오해하기 쉽다).
_HEARTBEAT_SEC = 2.0


def _wait_with_heartbeat(future, progress_signal, label: str):
    """future.result()를 기다리는 동안 주기적으로 progress_signal에 경과
    시간을 emit한다. 계산 자체는 자식 프로세스에서 계속 진행 중이다.
    """
    start = time.monotonic()
    while True:
        try:
            return future.result(timeout=_HEARTBEAT_SEC)
        except FutureTimeoutError:
            elapsed = time.monotonic() - start
            progress_signal.emit(f"{label} ({elapsed:.0f}초 경과 - 계산 중입니다)")



class PipelineWorker(QObject):
    """전체 파이프라인(1차 실행)을 담당. run()이 끝나면 finished를 emit한다."""

    progress = Signal(str)
    dataset_ready = Signal(object)          # Dataset
    quality_ready = Signal(list)            # 경고 문구 리스트
    models_ready = Signal(dict)             # dict[CameraModelType, CalibrationResult]
    validation_ready = Signal(dict)         # dict[CameraModelType, ValidationResult]
    recommendation_ready = Signal(list, str)  # list[ModelScore], 추천 문구
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        image_paths: list[str],
        pattern_config: PatternConfig,
        camera_config: CameraConfig,
        test_ratio: float = 0.25,
        use_rational_model: bool = False,
    ):
        super().__init__()
        self.image_paths = image_paths
        self.pattern_config = pattern_config
        self.camera_config = camera_config
        self.test_ratio = test_ratio
        self.use_rational_model = use_rational_model
        self.dataset: Dataset | None = None

    def run(self) -> None:
        try:
            self.progress.emit(f"{len(self.image_paths)}장 이미지에서 ChArUco 코너 검출 중...")
            # 이미 QThread(백그라운드 스레드) 안이라 프로세스 풀을 더 띄워도 UI가
            # 멈추지 않는다. 이미지가 충분히 많을 때만 병렬화 이득이 프로세스 생성
            # 비용을 넘어서므로, 적은 장수(<= 8)에서는 그냥 순차로 둔다.
            use_parallel = len(self.image_paths) > 8
            dataset = detect_dataset(self.image_paths, self.pattern_config, parallel=use_parallel)
            self.dataset = dataset
            self.dataset_ready.emit(dataset)

            self.progress.emit("Coverage Map / 데이터셋 품질 분석 중...")
            warnings = analyze_dataset_quality(dataset, self.camera_config)
            self.quality_ready.emit(warnings)

            self.progress.emit("프레임별 품질 점수 계산 중...")
            image_size = infer_image_size(dataset, self.camera_config)
            compute_frame_quality_scores(
                dataset, self.pattern_config, image_size, use_reprojection=False
            )
            self.dataset_ready.emit(dataset)  # 1차 점수(재투영 오차 제외) 반영해서 테이블 갱신

            self.progress.emit("Pinhole / Extended Pinhole / Fisheye 3개 모델 계산 중...")
            # cv2.calibrateCamera 등은 GIL을 놓아준다는 보장이 없어, 완전히
            # 별도 프로세스로 돌려서 GUI 스레드가 절대 막히지 않게 한다
            # (자세한 이유는 파일 상단 docstring 참고).
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    run_models_and_validation,
                    dataset, self.camera_config, self.pattern_config,
                    self.test_ratio, self.use_rational_model,
                )
                calibration_results, validation_results = _wait_with_heartbeat(
                    future, self.progress,
                    "Pinhole / Extended Pinhole / Fisheye 3개 모델 + Hold-out Validation 계산 중...",
                )

            self.progress.emit("재투영 오차를 반영해 프레임 품질 점수 갱신 중...")
            compute_frame_quality_scores(
                dataset, self.pattern_config, image_size, use_reprojection=True
            )
            self.models_ready.emit(calibration_results)
            self.validation_ready.emit(validation_results)

            self.progress.emit("Model Score 계산 및 추천 생성 중...")
            scores = compute_model_scores(
                calibration_results, validation_results,
                use_rational_model=self.use_rational_model,
            )
            message = build_recommendation_message(scores, calibration_results, validation_results)
            self.recommendation_ready.emit(scores, message)

            self.progress.emit("완료.")
        except Exception as e:  # noqa: BLE001 - UI에 원인을 그대로 보여주기 위해 광범위하게 캐치
            self.error.emit(f"파이프라인 실행 중 오류: {e}")
        finally:
            self.finished.emit()


class OutlierPruneWorker(QObject):
    """이상치 제거 -> 데이터셋 전체 재계산.

    설계 문서 9번 원칙: 자동 삭제가 아니라 '추천을 사용자가 승인'한 뒤에만
    호출되어야 한다. 이 워커는 이미 승인된 상태에서 실행을 담당할 뿐이다.
    한 프레임이 나쁘면 세 모델 모두에게 나쁘다고 보고, 하나의 공유 Dataset에서
    비활성화 -> 세 모델 전체 재계산 -> Hold-out 재검증까지 한 번에 수행한다.
    """

    progress = Signal(str)
    dataset_updated = Signal(object)        # Dataset (프레임 status/오차가 갱신됨)
    quality_ready = Signal(list)            # 경고 문구 리스트 (재계산 후 coverage 갱신)
    outlier_ready = Signal(object, object)  # CalibrationResult(reference model), OutlierResult
    models_ready = Signal(dict)
    validation_ready = Signal(dict)
    recommendation_ready = Signal(list, str)
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        dataset: Dataset,
        camera_config: CameraConfig,
        pattern_config: PatternConfig,
        reference_model: CameraModelType,
        max_iterations: int = 3,
        test_ratio: float = 0.25,
        use_rational_model: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.camera_config = camera_config
        self.pattern_config = pattern_config
        self.reference_model = reference_model
        self.max_iterations = max_iterations
        self.test_ratio = test_ratio
        self.use_rational_model = use_rational_model

    def run(self) -> None:
        try:
            self.progress.emit(f"{self.reference_model.value} 기준으로 이상치 탐지 및 반복 재계산 중...")
            # 이상치 반복 재계산 + 3모델 재계산 + Hold-out 재검증까지 전부
            # 별도 프로세스에서 실행한다 (이유는 파일 상단 docstring 참고).
            # recalibrate_with_outlier_pruning()이 dataset을 in-place로
            # 바꾸는 부수효과가 있어서, 자식 프로세스가 반환한 dataset으로
            # self.dataset을 통째로 교체해야 한다 - 그래야 이상치 제외 상태가
            # 실제로 반영된다.
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    run_outlier_pruning_and_validation,
                    self.dataset, self.camera_config, self.pattern_config, self.reference_model,
                    self.max_iterations, self.test_ratio, self.use_rational_model,
                )
                (
                    updated_dataset, ref_result, outlier_result, warnings,
                    calibration_results, validation_results,
                ) = _wait_with_heartbeat(
                    future, self.progress,
                    f"{self.reference_model.value} 기준 이상치 재계산 + 3모델 + Hold-out 계산 중...",
                )

            self.dataset = updated_dataset

            self.outlier_ready.emit(ref_result, outlier_result)
            self.dataset_updated.emit(self.dataset)  # 프레임 status가 바뀌었으므로 즉시 반영

            self.progress.emit("Coverage Map 재분석 중...")
            self.quality_ready.emit(warnings)
            self.dataset_updated.emit(self.dataset)

            self.models_ready.emit(calibration_results)
            self.validation_ready.emit(validation_results)

            scores = compute_model_scores(
                calibration_results, validation_results,
                use_rational_model=self.use_rational_model,
            )
            message = build_recommendation_message(scores, calibration_results, validation_results)
            self.recommendation_ready.emit(scores, message)

            self.progress.emit("완료.")
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"이상치 재계산 중 오류: {e}")
        finally:
            self.finished.emit()


class SelfCheckWorker(QObject):
    """설계 문서 원칙과 무관하게, "이 툴이 실제로 정확한 결과를 내는지"를
    이미지 없이 즉석에서 확인하고 싶다는 요청으로 추가된 워커.

    알려진 정답 카메라 파라미터로 합성 ChArUco 이미지를 만들고, 실제
    calibrate_pinhole/calibrate_extended_pinhole을 그 데이터에 돌려서
    복원된 fx/fy/cx/cy가 정답에 얼마나 가까운지 검증한다
    (calibration/self_check.py, tests/test_calibration_accuracy.py와 동일한
    로직 재사용 - 계산 로직 중복 금지 원칙).

    사용자가 불러온 실제 이미지/데이터셋과는 무관하게 항상 같은 합성
    데이터로 돌아가므로, 실행 중인 캘리브레이션 세션 상태를 전혀 건드리지
    않는다 - 아무 때나 눌러도 안전하다.
    """

    progress = Signal(str)
    result_ready = Signal(list)   # list[SelfCheckResult]
    error = Signal(str)
    finished = Signal()

    def run(self) -> None:
        try:
            self.progress.emit(
                "합성 데이터(정답을 미리 아는 가짜 카메라)로 Pinhole/Extended Pinhole/"
                "Rational model 정확도를 확인하는 중... (수십 초 정도 걸릴 수 있습니다)"
            )
            results = run_all_self_checks()
            self.result_ready.emit(results)
            self.progress.emit("자체 진단 완료.")
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"자체 진단 중 오류: {e}")
        finally:
            self.finished.emit()


class BagExtractionWorker(QObject):
    """rosbag에서 이미지를 뽑아 .jpg로 저장하는 무거운 작업(수백~수천 개
    메시지를 디코딩+디스크 기록)을 GUI 스레드 밖으로 분리한다.

    실제 사용자 버그: main_window.py의 _on_load_from_bag()가
    calibration.rosbag_reader.extract_images_from_bag()을 GUI 스레드에서
    직접(동기) 호출하고 있었다. 큰 bag(수백 MB, 메시지 수천 개)에서는 이게
    수십 초~몇 분씩 걸리는데, 그동안 Qt 이벤트 루프가 완전히 멈춰서 OS가
    "python3 is not responding" 창을 띄운다. 다른 무거운 계산(검출/3모델
    학습/Hold-out)은 이미 PipelineWorker로 QThread에 분리돼 있었는데 이
    경로만 빠져 있었다.
    """

    progress = Signal(str)
    finished_extraction = Signal(list)  # list[str] - 저장된 이미지 경로
    error = Signal(str)
    finished = Signal()

    def __init__(self, bag_path: str, topic: str, output_dir: str, min_interval_sec: float):
        super().__init__()
        self.bag_path = bag_path
        self.topic = topic
        self.output_dir = output_dir
        self.min_interval_sec = min_interval_sec
        self._cancelled = False

    def request_cancel(self) -> None:
        """다른 스레드(GUI)에서 호출됨 - 진행률 다이얼로그의 취소 버튼용.
        불리언 플래그 하나만 건드리므로 락 없이도 안전하다.
        """
        self._cancelled = True

    def run(self) -> None:
        def _on_progress(done: int, total: int, saved: int) -> None:
            if total:
                pct = done / total * 100
                self.progress.emit(
                    f"bag에서 이미지 추출 중... {pct:.0f}% ({done}/{total} 메시지, {saved}장 저장됨)"
                )
            else:
                self.progress.emit(f"bag에서 이미지 추출 중... ({done}개 메시지 처리, {saved}장 저장됨)")

        try:
            extracted = extract_images_from_bag(
                self.bag_path,
                self.topic,
                self.output_dir,
                min_interval_sec=self.min_interval_sec,
                progress_callback=_on_progress,
                cancel_check=lambda: self._cancelled,
            )
            self.finished_extraction.emit(extracted)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"bag 이미지 추출 중 오류: {e}")
        finally:
            self.finished.emit()


def run_worker_in_thread(worker: QObject, parent: QObject) -> QThread:
    """QThread 보일러플레이트를 한 곳에 모아둔 헬퍼.
    호출부는 `thread = run_worker_in_thread(worker, self)` 후 필요한 signal에
    connect하고 thread.start()만 하면 된다.
    """
    thread = QThread(parent)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    return thread

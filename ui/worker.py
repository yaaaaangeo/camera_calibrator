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
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

from PySide6.QtCore import QObject, QMetaObject, QThread, Qt, Signal, Slot

from calibration.types import (
    CameraConfig,
    CameraModelType,
    Dataset,
    PatternConfig,
)
from calibration.detector import build_detect_fn, detect_dataset, summarize_dataset
from calibration.latest_frame import FrameBufferStats, LatestFrameBuffer
from calibration.quality import analyze_dataset_quality
from calibration.frame_quality import compute_frame_quality_scores, compute_dataset_quality_score
from calibration.image_quality import evaluate_dataset_image_quality
from calibration.quality import coverage_percentage
from calibration.models.common import infer_image_size
from calibration.recommender import compute_model_scores, build_recommendation_message
from calibration.self_check import run_all_self_checks
from calibration.library import save_calibration_run
from calibration.rosbag_reader import (
    extract_images_from_bag,
    extract_image_near_timestamp,
    extract_pointcloud_near_timestamp,
    list_image_topics,
)
from camera_lidar.multi_scene import calibrate_multi_scene, compare_strict_vs_flexible
from camera_lidar.types import ImageFrame, PointCloudFrame
from calibration.validation import validate_cross_datasets
from calibration.external_compare import compare_with_external_params
from calibration.model_refitting import refit_extended_pinhole_to_pinhole
from calibration.stereo_controller import StereoController
from calibration.camera_lidar_controller import CameraLidarController
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


class LiveDetectionWorker(QObject):
    """Latest-only pattern detection worker for the live preview.

    ``submit_frame`` is intentionally safe to call from the GUI thread even
    after this object has moved to a QThread: it only replaces a single-slot
    buffer and schedules at most one queued worker invocation.  While OpenCV
    is busy, any number of incoming frames therefore collapse into one latest
    frame instead of growing a Qt signal queue.
    """

    result_ready = Signal(object)  # DetectionResult
    frame_result_ready = Signal(object, object)  # exact source frame, DetectionResult
    error = Signal(str)

    def __init__(self, pattern_config: PatternConfig):
        super().__init__()
        self.pattern_config = pattern_config
        self._frame_buffer = LatestFrameBuffer()
        self._schedule_lock = threading.Lock()
        self._scheduled = False
        self._stopping = False
        self._detect_fn = None
        self._frame_index = 0

    @Slot()
    def initialize(self) -> None:
        """Build OpenCV detector objects in the thread that will use them."""
        try:
            self._detect_fn = build_detect_fn(self.pattern_config)
        except Exception as exc:  # noqa: BLE001 - keep raw preview alive
            self.error.emit(f"Live detection 초기화 실패: {exc}")

    def submit_frame(self, frame: object, timestamp_sec: float) -> None:
        """Replace the pending frame and schedule no more than one work item."""
        if not hasattr(frame, "shape"):
            return
        with self._schedule_lock:
            if self._stopping:
                return
            self._frame_buffer.put(frame, timestamp_sec)
            if self._scheduled:
                return
            self._scheduled = True
            QMetaObject.invokeMethod(self, "_process_latest", Qt.QueuedConnection)

    @Slot()
    def _process_latest(self) -> None:
        pending = self._frame_buffer.take()
        if pending is not None and not self._stopping and self._detect_fn is not None:
            frame, _timestamp_sec = pending
            image_id = f"live_preview_{self._frame_index:08d}"
            self._frame_index += 1
            try:
                result = self._detect_fn(frame, image_id)
            except Exception as exc:  # noqa: BLE001 - a bad frame must not stop preview
                self.error.emit(f"Live detection 오류: {exc}")
            else:
                if not self._stopping:
                    self.result_ready.emit(result)
                    self.frame_result_ready.emit(frame, result)

        # Detection may have taken longer than the camera interval.  If frames
        # arrived meanwhile, schedule exactly one more invocation for the most
        # recent slot; otherwise release the scheduling flag atomically.
        with self._schedule_lock:
            if self._stopping:
                self._scheduled = False
                self._frame_buffer.clear()
            elif self._frame_buffer.has_pending():
                QMetaObject.invokeMethod(self, "_process_latest", Qt.QueuedConnection)
            else:
                self._scheduled = False

    def request_stop(self) -> None:
        """Thread-safe stop request; an in-flight OpenCV call may finish first."""
        with self._schedule_lock:
            self._stopping = True
            self._frame_buffer.clear()

    def buffer_stats(self) -> FrameBufferStats:
        return self._frame_buffer.stats()



class PipelineWorker(QObject):
    """전체 파이프라인(1차 실행)을 담당. run()이 끝나면 finished를 emit한다."""

    progress = Signal(str)
    # total > 0이면 실제 완료 비율, total == 0이면 내부 반복 횟수를 알 수 없는
    # OpenCV 최적화가 실행 중이라는 busy indicator로 UI가 표시한다.
    progress_value = Signal(int, int)
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
            self.progress_value.emit(0, len(self.image_paths))
            # 이미 QThread(백그라운드 스레드) 안이라 프로세스 풀을 더 띄워도 UI가
            # 멈추지 않는다. 이미지가 충분히 많을 때만 병렬화 이득이 프로세스 생성
            # 비용을 넘어서므로, 적은 장수(<= 8)에서는 그냥 순차로 둔다.
            use_parallel = len(self.image_paths) > 8
            def _on_detection_progress(done: int, total: int) -> None:
                self.progress.emit(f"코너 검출 중... {done}/{total}장 ({done / total * 100:.0f}%)")
                self.progress_value.emit(done, total)

            dataset = detect_dataset(
                self.image_paths,
                self.pattern_config,
                parallel=use_parallel,
                progress_callback=_on_detection_progress,
            )
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
            # cv2의 optimizer는 iteration callback을 제공하지 않는다. 가짜 퍼센트를
            # 만들지 않고 이 구간만 실제 실행 여부를 나타내는 busy bar로 전환한다.
            self.progress_value.emit(0, 0)
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
            # 설계 문서 4번 - Overall Dataset Score. 개별 프레임 점수(방금 갱신됨)
            # + coverage + 다양성 + 중복 이미지 비율을 하나로 요약해 Dataset에 저장한다.
            _, duplicate_groups = evaluate_dataset_image_quality(dataset)
            dup_ratio = (
                sum(len(g.image_ids) for g in duplicate_groups) / dataset.num_total
                if dataset.num_total > 0 else 0.0
            )
            dataset.quality_score = compute_dataset_quality_score(
                dataset,
                coverage_pct=coverage_percentage(dataset.coverage_grid) if dataset.coverage_grid else None,
                duplicate_ratio=dup_ratio,
            )
            self.dataset_ready.emit(dataset)
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
            self.progress_value.emit(1, 1)
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
            # 설계 문서 4번 - 이상치 제거로 데이터셋이 바뀌었으니 Dataset Score도 재계산.
            _, duplicate_groups = evaluate_dataset_image_quality(self.dataset)
            dup_ratio = (
                sum(len(g.image_ids) for g in duplicate_groups) / self.dataset.num_total
                if self.dataset.num_total > 0 else 0.0
            )
            self.dataset.quality_score = compute_dataset_quality_score(
                self.dataset,
                coverage_pct=coverage_percentage(self.dataset.coverage_grid) if self.dataset.coverage_grid else None,
                duplicate_ratio=dup_ratio,
            )
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


class CrossDatasetValidationWorker(QObject):
    """외부 Dataset B/C/...를 검출하고 현재 calibration으로 generalization을 평가한다."""

    progress = Signal(str)
    results_ready = Signal(list)
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        target_image_paths: dict[str, list[str]],
        calibration_results: dict[CameraModelType, object],
        camera_config: CameraConfig,
        pattern_config: PatternConfig,
        source_dataset_id: str = "Dataset A",
    ):
        super().__init__()
        self.target_image_paths = target_image_paths
        self.calibration_results = calibration_results
        self.camera_config = camera_config
        self.pattern_config = pattern_config
        self.source_dataset_id = source_dataset_id

    def run(self) -> None:
        try:
            target_datasets: dict[str, Dataset] = {}
            for dataset_id, paths in self.target_image_paths.items():
                self.progress.emit(f"Cross-dataset target '{dataset_id}' 검출 중... ({len(paths)}장)")
                target = detect_dataset(
                    paths,
                    self.pattern_config,
                    parallel=len(paths) > 8,
                )
                self.progress.emit(f"{dataset_id}: {summarize_dataset(target)}")
                if target.num_detected > 0:
                    analyze_dataset_quality(target, self.camera_config)
                target_datasets[dataset_id] = target

            self.progress.emit("Cross-dataset validation 계산 중...")
            results = validate_cross_datasets(
                self.calibration_results,
                target_datasets,
                self.camera_config,
                self.pattern_config,
                source_dataset_id=self.source_dataset_id,
            )
            self.results_ready.emit(results)
            self.progress.emit("Cross-dataset validation 완료.")
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"Cross-dataset validation 중 오류: {e}")
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


class BagTopicDiscoveryWorker(QObject):
    """큰 rosbag의 메타데이터/인덱스 검색을 GUI 스레드 밖에서 수행한다.

    ROS1 bag은 파일을 여는 것만으로도 큰 인덱스를 읽을 수 있다. 이 단계를
    GUI에서 직접 실행하면 실제 이미지 디코딩을 시작하기도 전에 운영체제가
    창을 '응답 없음'으로 판정한다.
    """

    progress = Signal(str)
    # bag_path까지 signal payload로 보내고 MainWindow의 bound method에 직접
    # 연결한다. Python lambda를 중간에 두면 PySide가 그 lambda를 sender의
    # worker thread에서 호출할 수 있어, 그 안에서 dialog를 만들면 즉시
    # cross-thread QObject parent 오류/segfault가 난다.
    topics_ready = Signal(object, str)
    error = Signal(str)
    finished = Signal()

    def __init__(self, bag_path: str, list_fn=None, label: str = "이미지"):
        super().__init__()
        self.bag_path = bag_path
        # list_fn=None(기본값)이면 run()에서 모듈 전역 이름 list_image_topics를
        # "그 자리에서" 다시 찾는다 - list_fn의 기본값 자체를 list_image_topics로
        # 박아두면(이른 바인딩) 기존 회귀 테스트가 하던
        # monkeypatch.setattr("ui.worker.list_image_topics", ...) 가 이 워커에는
        # 더 이상 적용되지 않는 문제가 생긴다(그 몽키패치는 모듈 전역 이름을
        # 바꾸는 것이지, 이미 만들어진 클래스의 기본 인자 값을 바꾸지는 못한다).
        # camera_lidar의 Bag 소스 섹션은 이 워커를 list_fn=list_pointcloud_topics로
        # 두 번째로 재사용해서 LiDAR 토픽도 검색한다.
        self.list_fn = list_fn
        self.label = label

    def run(self) -> None:
        try:
            self.progress.emit(f"bag 인덱스에서 {self.label} 토픽을 검색 중...")
            list_fn = self.list_fn if self.list_fn is not None else list_image_topics
            self.topics_ready.emit(list_fn(self.bag_path), self.bag_path)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"bag 읽기 실패: {e}")
        finally:
            self.finished.emit()


class ExternalComparisonWorker(QObject):
    """External Compare의 재학습/K-fold/bootstrap을 별도 프로세스에서 실행."""

    progress = Signal(str)
    result_ready = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        dataset,
        camera_config,
        pattern_config,
        my_model,
        my_validation,
        external,
        use_rational_model: bool,
    ):
        super().__init__()
        self.dataset = dataset
        self.camera_config = camera_config
        self.pattern_config = pattern_config
        self.my_model = my_model
        self.my_validation = my_validation
        self.external = external
        self.use_rational_model = use_rational_model

    def run(self) -> None:
        try:
            self.progress.emit("External Compare 계산 중... (재학습/K-fold/bootstrap 포함)")
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    compare_with_external_params,
                    self.dataset,
                    self.camera_config,
                    self.pattern_config,
                    self.my_model,
                    self.my_validation,
                    self.external,
                    self.use_rational_model,
                )
                result = _wait_with_heartbeat(
                    future, self.progress, "External Compare 계산 중...",
                )
            self.result_ready.emit(result)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"External Compare 계산 중 오류: {e}")
        finally:
            self.finished.emit()


class ModelRefittingWorker(QObject):
    """8계수 Rational Pinhole -> 5계수 Pinhole 근사 최적화를 GUI 밖에서 수행."""

    progress = Signal(str)
    result_ready = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        camera_matrix,
        distortion,
        image_size: tuple[int, int],
        options: dict,
    ):
        super().__init__()
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.image_size = image_size
        self.options = options

    def run(self) -> None:
        try:
            self.progress.emit("Model Refitting 계산 중...")
            result = refit_extended_pinhole_to_pinhole(
                self.camera_matrix,
                self.distortion,
                self.image_size,
                mode=self.options.get("mode", "full"),
                grid_size=self.options.get("grid_size", (80, 50)),
                edge_weighting=bool(self.options.get("edge_weighting", False)),
                loss=self.options.get("loss", "linear"),
            )
            self.result_ready.emit(result)
        except Exception as e:  # noqa: BLE001
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class StereoPairDetectionWorker(QObject):
    """Camera 1/2 image folders를 검출하고 common ChArUco pair를 만든다."""

    progress = Signal(str)
    pairs_ready = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, camera1_paths: list[str], camera2_paths: list[str], pattern_config):
        super().__init__()
        self.camera1_paths = camera1_paths
        self.camera2_paths = camera2_paths
        self.pattern_config = pattern_config

    def run(self) -> None:
        try:
            self.progress.emit(f"Camera 1 pair 이미지 검출 중... ({len(self.camera1_paths)}장)")
            self.progress.emit(f"Camera 2 pair 이미지 검출 중... ({len(self.camera2_paths)}장)")
            self.progress.emit("Common ChArUco ID matching 중...")
            pairs, ds1, ds2 = StereoController().detect_pairs(
                self.camera1_paths,
                self.camera2_paths,
                self.pattern_config,
            )
            self.pairs_ready.emit((pairs, ds1, ds2))
            self.progress.emit(f"Stereo pair detection 완료: {len(pairs)} usable pairs")
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"Stereo pair detection 중 오류: {e}")
        finally:
            self.finished.emit()


class BenchmarkDetectionWorker(QObject):
    """External Compare 탭의 Independent Benchmark 이미지 검출을 GUI 밖에서 수행.

    detect_dataset() 자체는 8장 초과일 때 ProcessPoolExecutor로 병렬화되지만,
    호출부가 GUI 스레드에서 직접 그 결과를 기다리면 병렬 처리 여부와 무관하게
    GUI 스레드가 블로킹돼 "python3 is not responding"이 뜬다. 다른 탭의 검출
    (PipelineWorker, StereoPairDetectionWorker)은 이미 QThread로 분리돼 있었지만
    이 경로만 빠져 있었다.
    """

    progress = Signal(str)
    dataset_ready = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, image_paths: list[str], pattern_config: PatternConfig, camera_config: CameraConfig):
        super().__init__()
        self.image_paths = image_paths
        self.pattern_config = pattern_config
        self.camera_config = camera_config

    def run(self) -> None:
        try:
            total = len(self.image_paths)
            self.progress.emit(f"Benchmark 이미지 검출 중... ({total}장)")

            def _on_progress(done: int, total_count: int) -> None:
                self.progress.emit(f"Benchmark 이미지 검출 중... {done}/{total_count}장")

            dataset = detect_dataset(
                self.image_paths,
                self.pattern_config,
                parallel=total > 8,
                progress_callback=_on_progress,
            )
            if dataset.num_detected > 0:
                analyze_dataset_quality(dataset, self.camera_config)
            self.dataset_ready.emit(dataset)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"Benchmark 불러오기 실패: {e}")
        finally:
            self.finished.emit()


class LibrarySaveWorker(QObject):
    """계산 결과(이미지 사본 포함)를 Library 폴더에 저장하는 무거운 파일 I/O를
    GUI 밖에서 수행한다. dataset을 deepcopy하고 이미지 수백 장을 복사하는
    작업이라, 이미지가 많을 때 GUI 스레드에서 직접 하면 잠깐이라도 멈춘다.
    """

    saved = Signal(str)  # run_dir
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        dataset,
        camera_config: CameraConfig,
        pattern_config: PatternConfig,
        calibration_results,
        validation_results,
        model_scores=None,
    ):
        super().__init__()
        self.dataset = dataset
        self.camera_config = camera_config
        self.pattern_config = pattern_config
        self.calibration_results = calibration_results
        self.validation_results = validation_results
        self.model_scores = model_scores

    def run(self) -> None:
        try:
            run_dir = save_calibration_run(
                self.dataset,
                self.camera_config,
                self.pattern_config,
                self.calibration_results,
                self.validation_results,
                model_scores=self.model_scores,
            )
            self.saved.emit(str(run_dir))
        except Exception:  # noqa: BLE001 - Library 저장 실패로 사용자 작업을 막으면 안 됨
            self.error.emit(f"Library 저장 실패:\n{traceback.format_exc()}")
        finally:
            self.finished.emit()


class StereoCalibrationWorker(QObject):
    """Stereo calibration/rectification/validation을 GUI 스레드 밖에서 수행."""

    progress = Signal(str)
    result_ready = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, pairs, camera1, camera2, image_size: tuple[int, int], audit_mode: str = "full"):
        super().__init__()
        self.pairs = pairs
        self.camera1 = camera1
        self.camera2 = camera2
        self.image_size = image_size
        self.audit_mode = audit_mode

    def run(self) -> None:
        try:
            audit_text = "Full audit/bootstrap" if self.audit_mode == "full" else "Fast audit"
            self.progress.emit(f"Stereo calibration 계산 중... (K1/D1/K2/D2 고정, R/T 최적화, {audit_text})")
            result = StereoController().calibrate(
                self.pairs,
                self.camera1,
                self.camera2,
                self.image_size,
                audit_mode=self.audit_mode,
            )
            self.result_ready.emit(result)
            self.progress.emit("Stereo calibration 완료.")
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"Stereo calibration 중 오류: {e}")
        finally:
            self.finished.emit()


class CameraLidarCalibrationWorker(QObject):
    """단일 CalibrationScene에 대해 FAST-Calib(camera_lidar/*)를 GUI
    스레드 밖에서 수행한다. 다른 워커와 마찬가지로 실제 계산 로직은
    calibration.camera_lidar_controller에 그대로 있고, 여기서는 호출/신호
    변환만 담당한다."""

    progress = Signal(str)
    result_ready = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, scene, roi_mode: str = "manual"):
        super().__init__()
        self.scene = scene
        self.roi_mode = roi_mode

    def run(self) -> None:
        try:
            self.progress.emit("FAST-Calib 계산 중... (marker/plane 검출 -> correspondence -> R,t 계산)")
            result = CameraLidarController().calibrate(self.scene, roi_mode=self.roi_mode)
            self.result_ready.emit(result)
            if result.success:
                self.progress.emit(f"FAST-Calib 완료 (residual RMSE {result.residual_rmse_m * 1000:.2f} mm).")
            else:
                self.progress.emit(f"FAST-Calib 실패: {result.error_message}")
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"FAST-Calib 계산 중 오류: {e}")
        finally:
            self.finished.emit()


class SceneExtractionWorker(QObject):
    """bag의 camera topic 전체를 스캔해 Stable Scene Segment candidate들을
    찾는(MARKER EXTRACTION) 무거운 작업(프레임마다 ArUco 검출)을 GUI
    스레드 밖에서 수행한다. 실제 계산은
    calibration.camera_lidar_controller.CameraLidarController.
    extract_scene_candidates()에 있고, 여기서는 호출/신호 변환만 담당한다."""

    progress = Signal(str)
    # (done, total) -- total==0이면 total_frames를 알아내지 못한 경우로, UI가
    # BagExtractionWorker.progress_value와 동일하게 busy indicator로 표시한다.
    progress_value = Signal(int, int)
    candidates_ready = Signal(object)  # list[SceneCandidate]
    summary_ready = Signal(object)     # ExtractionDiagnosticSummary -- Marker Extraction Diagnostic funnel
    error = Signal(str)
    finished = Signal()

    def __init__(self, bag_path: str, camera_topic: str, lidar_topic: str, intrinsics, target):
        super().__init__()
        self.bag_path = bag_path
        self.camera_topic = camera_topic
        self.lidar_topic = lidar_topic
        self.intrinsics = intrinsics
        self.target = target
        self._cancelled = False

    def request_cancel(self) -> None:
        """다른 스레드(GUI)에서 호출됨 -- BagExtractionWorker.request_cancel과
        동일한 패턴. 불리언 플래그 하나만 건드리므로 락 없이도 안전하다."""
        self._cancelled = True

    def run(self) -> None:
        try:
            self.progress.emit(f"Marker extraction 시작: {self.camera_topic} 전체 프레임을 스캔합니다...")
            candidates, summary = CameraLidarController().extract_scene_candidates(
                self.bag_path, self.camera_topic, self.lidar_topic,
                self.intrinsics, self.target,
                progress_callback=self.progress.emit,
                frame_progress_callback=self.progress_value.emit,
                cancel_check=lambda: self._cancelled,
            )
            self.candidates_ready.emit(candidates)
            self.summary_ready.emit(summary)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"Marker extraction 중 오류: {e}")
        finally:
            self.finished.emit()


class BagPreviewWorker(QObject):
    """Bag timeline을 스크럽할 때 t 근처의 이미지+PointCloud2 프레임 하나씩을
    GUI 스레드 밖에서 읽어온다 (extract_images_from_bag의 "전체를 시간 간격
    샘플링해 디스크에 저장"과 달리, 단발성 조회라 훨씬 가볍지만 그래도 bag
    인덱스를 다시 여는 I/O라 스레드 분리 원칙은 동일하게 적용한다)."""

    progress = Signal(str)
    preview_ready = Signal(object, object)  # (ImageFrame, PointCloudFrame)
    error = Signal(str)
    finished = Signal()

    def __init__(self, bag_path: str, camera_topic: str, lidar_topic: str, t_sec: float):
        super().__init__()
        self.bag_path = bag_path
        self.camera_topic = camera_topic
        self.lidar_topic = lidar_topic
        self.t_sec = t_sec

    def run(self) -> None:
        try:
            self.progress.emit(f"t={self.t_sec:.2f}s 근처 프레임을 불러오는 중...")
            image, image_ts, image_frame_id = extract_image_near_timestamp(
                self.bag_path, self.camera_topic, self.t_sec
            )
            points, points_ts, points_frame_id = extract_pointcloud_near_timestamp(
                self.bag_path, self.lidar_topic, self.t_sec
            )
            image_frame = ImageFrame(
                timestamp=image_ts, image=image, frame_id=image_frame_id,
                source_metadata={"bag_path": self.bag_path, "topic": self.camera_topic},
            )
            cloud_frame = PointCloudFrame(
                timestamp=points_ts,
                points=points[:, :3],
                frame_id=points_frame_id,
                intensity=points[:, 3] if points.shape[1] > 3 else None,
                source_metadata={"bag_path": self.bag_path, "topic": self.lidar_topic},
            )
            self.preview_ready.emit(image_frame, cloud_frame)
            self.progress.emit(
                f"Preview 로드 완료 (Δt camera-lidar = {(image_ts - points_ts) * 1000:.1f} ms)."
            )
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"Bag preview 로드 실패: {e}")
        finally:
            self.finished.emit()


class MultiSceneCalibrationWorker(QObject):
    """여러 CapturedScene을 모아 Multi-Scene FAST-Calib(joint solve)를 GUI
    스레드 밖에서 수행한다. 실제 계산은 camera_lidar.multi_scene에 있고,
    여기서는 CameraLidarCalibrationWorker와 동일하게 호출/신호 변환만 담당한다."""

    progress = Signal(str)
    result_ready = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, captured_scenes: list, policy: str = "strict"):
        super().__init__()
        self.captured_scenes = captured_scenes
        self.policy = policy

    def run(self) -> None:
        try:
            self.progress.emit(
                f"Multi-Scene FAST-Calib 계산 중... ({self.policy.upper()} policy, "
                f"{len(self.captured_scenes)}개 scene 중 included된 scene만 사용)"
            )
            result = calibrate_multi_scene(self.captured_scenes, policy=self.policy)
            self.result_ready.emit(result)
            if result.success:
                self.progress.emit(
                    f"Multi-Scene FAST-Calib 완료 ({result.scene_count} scenes, "
                    f"residual RMSE {result.residual_rmse_m * 1000:.2f} mm)."
                )
            else:
                self.progress.emit(f"Multi-Scene FAST-Calib 실패: {result.error_message}")
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"Multi-Scene FAST-Calib 계산 중 오류: {e}")
        finally:
            self.finished.emit()


class PolicyComparisonWorker(QObject):
    """COMPARE BOTH: STRICT/FLEXIBLE Multi-Scene calibration을 각각 실행하고
    두 결과 차이를 계산한다(camera_lidar.multi_scene.compare_strict_vs_flexible).
    같은 solver를 두 번(다른 scene 부분집합으로) 돌리는 것뿐, 별도 solver를
    구현하지 않는다는 원칙은 그대로 지킨다."""

    progress = Signal(str)
    result_ready = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, captured_scenes: list):
        super().__init__()
        self.captured_scenes = captured_scenes

    def run(self) -> None:
        try:
            self.progress.emit("STRICT vs FLEXIBLE 비교 계산 중... (같은 solver를 두 scene 부분집합에 각각 적용)")
            result = compare_strict_vs_flexible(self.captured_scenes)
            self.result_ready.emit(result)
            if result.strict_result.success and result.flexible_result.success:
                self.progress.emit(f"비교 완료 (impact: {result.impact}).")
            else:
                self.progress.emit("비교 완료 (STRICT 또는 FLEXIBLE 중 하나 이상 실패).")
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"STRICT vs FLEXIBLE 비교 중 오류: {e}")
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
    # 실사용자 버그: 진행 다이얼로그가 setRange(0,0)짜리 "무한 반복 바"(busy
    # indicator)라서 실제 진행률과 무관하게 그냥 색 막대가 왔다갔다 하는
    # 것처럼 보였다(고정폭 안에서 채워지는 게 아니라 막대 자체가 움직임).
    # done/total은 이미 알고 있으므로 별도 숫자 시그널로 내보내
    # QProgressDialog.setMaximum()/setValue()에 직접 연결한다.
    progress_value = Signal(int, int)  # (done, total)
    finished_extraction = Signal(object, str)  # (list[str], bag_path)
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
            self.progress_value.emit(done, total)

        try:
            extracted = extract_images_from_bag(
                self.bag_path,
                self.topic,
                self.output_dir,
                min_interval_sec=self.min_interval_sec,
                progress_callback=_on_progress,
                cancel_check=lambda: self._cancelled,
            )
            self.finished_extraction.emit(extracted, self.bag_path)
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

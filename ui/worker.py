"""
camera_calibrator.ui.worker
===============================

캘리브레이션 파이프라인(검출 -> 품질분석 -> 3모델 학습 -> Hold-out -> 추천)은
이미지 수십 장 기준으로 수 초가 걸릴 수 있어, UI 스레드를 막지 않도록
QThread 워커로 분리한다.

이 파일은 순전히 "언제 무엇을 호출하고 어떤 신호로 알릴지"만 담당하고,
실제 계산 로직은 전부 calibration/*.py의 기존 함수를 그대로 호출한다.
UI 계층이 계산 로직을 재구현하지 않는다는 원칙을 지키기 위함.
"""

from __future__ import annotations

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
from calibration.compare import run_all_models
from calibration.validation import validate_all_models
from calibration.recommender import compute_model_scores, build_recommendation_message
from calibration.outlier import recalibrate_with_outlier_pruning


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
    ):
        super().__init__()
        self.image_paths = image_paths
        self.pattern_config = pattern_config
        self.camera_config = camera_config
        self.test_ratio = test_ratio
        self.dataset: Dataset | None = None

    def run(self) -> None:
        try:
            self.progress.emit(f"{len(self.image_paths)}장 이미지에서 ChArUco 코너 검출 중...")
            dataset = detect_dataset(self.image_paths, self.pattern_config)
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
            results_list = run_all_models(dataset, self.camera_config)
            calibration_results = {r.model_name: r for r in results_list}

            self.progress.emit("재투영 오차를 반영해 프레임 품질 점수 갱신 중...")
            compute_frame_quality_scores(
                dataset, self.pattern_config, image_size, use_reprojection=True
            )
            self.models_ready.emit(calibration_results)

            self.progress.emit("Hold-out Validation (Train/Test 분할 검증) 중...")
            validation_results = validate_all_models(
                dataset, self.camera_config, self.pattern_config, test_ratio=self.test_ratio
            )
            self.validation_ready.emit(validation_results)

            self.progress.emit("Model Score 계산 및 추천 생성 중...")
            scores = compute_model_scores(calibration_results, validation_results)
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
    ):
        super().__init__()
        self.dataset = dataset
        self.camera_config = camera_config
        self.pattern_config = pattern_config
        self.reference_model = reference_model
        self.max_iterations = max_iterations
        self.test_ratio = test_ratio

    def run(self) -> None:
        try:
            self.progress.emit(f"{self.reference_model.value} 기준으로 이상치 탐지 및 반복 재계산 중...")
            ref_result, outlier_result = recalibrate_with_outlier_pruning(
                self.dataset, self.camera_config, self.reference_model,
                max_iterations=self.max_iterations,
            )
            self.outlier_ready.emit(ref_result, outlier_result)
            self.dataset_updated.emit(self.dataset)  # 프레임 status가 바뀌었으므로 즉시 반영

            self.progress.emit("Coverage Map 재분석 중...")
            warnings = analyze_dataset_quality(self.dataset, self.camera_config)
            self.quality_ready.emit(warnings)

            image_size = infer_image_size(self.dataset, self.camera_config)
            compute_frame_quality_scores(
                self.dataset, self.pattern_config, image_size, use_reprojection=False
            )
            self.dataset_updated.emit(self.dataset)

            self.progress.emit("정제된 데이터셋으로 3개 모델 재계산 중...")
            results_list = run_all_models(self.dataset, self.camera_config)
            calibration_results = {r.model_name: r for r in results_list}

            compute_frame_quality_scores(
                self.dataset, self.pattern_config, image_size, use_reprojection=True
            )
            self.models_ready.emit(calibration_results)

            self.progress.emit("Hold-out Validation 재실행 중...")
            validation_results = validate_all_models(
                self.dataset, self.camera_config, self.pattern_config, test_ratio=self.test_ratio
            )
            self.validation_ready.emit(validation_results)

            scores = compute_model_scores(calibration_results, validation_results)
            message = build_recommendation_message(scores, calibration_results, validation_results)
            self.recommendation_ready.emit(scores, message)

            self.progress.emit("완료.")
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"이상치 재계산 중 오류: {e}")
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

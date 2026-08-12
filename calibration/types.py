"""
camera_calibrator.calibration.types
====================================

프로젝트 전체가 공유하는 핵심 데이터 구조.

설계 원칙 (설계 문서 18번 참고):
- UI(Qt / Web)와 완전히 무관하게 독립적으로 정의한다.
- 모든 백엔드 모듈(detector, models, optimizer, outlier, validation,
  recommender, export)은 이 파일의 타입을 주고받는다.
- 여기서 필드가 바뀌면 전체 파이프라인이 영향을 받으므로,
  가능한 한 초반에 구조를 확정하고 이후에는 "추가"만 하는 방향으로 간다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CameraModelType(str, Enum):
    """설계 문서 1번 - 카메라 모델 정의"""
    PINHOLE = "pinhole"
    EXTENDED_PINHOLE = "extended_pinhole"
    FISHEYE = "fisheye"


class PatternType(str, Enum):
    """설계 문서 2번 - ChArUco / AprilGrid 지원"""
    CHESSBOARD = "chessboard"
    CHARUCO = "charuco"
    APRILGRID = "apriltag_grid"


class FrameStatus(str, Enum):
    """개별 프레임(이미지)의 파이프라인 상태.

    검출 실패/이상치여도 파일이나 레코드를 삭제하지 않고
    상태값만 바꾼다 (설계 문서 9번, 17번 Step2 원칙).
    """
    PENDING = "pending"                 # 아직 검출 전
    DETECTED = "detected"                # 검출 성공, 캘리브레이션에 사용 가능
    DETECTION_FAILED = "detection_failed"  # 코너 검출 자체가 실패
    DISABLED_OUTLIER = "disabled_outlier"  # 이상치로 판단되어 자동 비활성화
    DISABLED_MANUAL = "disabled_manual"    # 사용자가 수동으로 제외


class QualityGrade(str, Enum):
    """설계 문서 3.1번 RMS 등급 + 12번 종합 등급에 공용으로 사용"""
    EXCELLENT = "excellent"
    VERY_GOOD = "very_good"
    GOOD = "good"
    WARNING = "warning"
    POOR = "poor"
    REJECT = "reject"


class ExportFormat(str, Enum):
    OPENCV_YAML = "opencv_yaml"
    ROS_CAMERA_INFO = "ros_camera_info"
    JSON = "json"
    CSV = "csv"
    HTML_REPORT = "html_report"


# ---------------------------------------------------------------------------
# Config (촬영 조건 / 패턴 정의)
# ---------------------------------------------------------------------------

@dataclass
class PatternConfig:
    """설계 문서 10번 - 패턴 메타정보. 결과와 함께 저장되어야 재현 가능."""
    type: PatternType
    squares_x: int
    squares_y: int
    square_size: float          # meter 단위 권장
    marker_size: Optional[float] = None   # ChArUco / AprilGrid 전용
    dictionary: Optional[str] = None      # 예: "DICT_6X6_250"

    @property
    def rows(self) -> int:
        return self.squares_y

    @property
    def cols(self) -> int:
        return self.squares_x


@dataclass
class CameraConfig:
    """설계 문서 18번 CameraConfig"""
    width: int
    height: int
    fps: Optional[float] = None
    model: Optional[CameraModelType] = None   # 최종 확정 모델 (미확정 시 None)
    sensor_name: Optional[str] = None
    hfov_deg: Optional[float] = None
    vfov_deg: Optional[float] = None


# ---------------------------------------------------------------------------
# 이미지 단위 정보 (설계 문서 17번 Step1)
# ---------------------------------------------------------------------------

@dataclass
class ImageInfo:
    image_id: str
    path: str
    width: int
    height: int
    sharpness: Optional[float] = None     # cv2.Laplacian(...).var()
    brightness: Optional[float] = None
    exposure: Optional[float] = None


# ---------------------------------------------------------------------------
# 검출 결과 (설계 문서 17번 Step2)
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    image_id: str
    success: bool
    corners: Optional[np.ndarray] = None       # (N, 1, 2) 2D 코너 좌표
    object_points: Optional[np.ndarray] = None  # (N, 1, 3) 대응 3D 좌표
    ids: Optional[np.ndarray] = None            # ChArUco/ArUco marker id
    num_corners: int = 0
    board_area_ratio: Optional[float] = None    # 이미지 대비 보드 면적 비율
    board_center_px: Optional[tuple[float, float]] = None
    board_tilt_deg: Optional[float] = None
    failure_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# 프레임 품질 (설계 문서 6번 - Frame Quality Score)
# ---------------------------------------------------------------------------

@dataclass
class FrameQuality:
    detection_score: float = 0.0   # 코너 수, confidence, blur, exposure 등 종합
    geometric_score: float = 0.0   # 중심과의 거리, 기울기, 중복도 등 종합
    overall_score: float = 0.0     # 0~100
    grade: QualityGrade = QualityGrade.POOR


# ---------------------------------------------------------------------------
# Frame = ImageInfo + DetectionResult + Quality + 상태를 하나로 묶은 단위
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    """Dataset.Frame[] 의 원소. 파이프라인 전체에서 '사진 한 장'을 대표하는 단위."""
    image_info: ImageInfo
    detection: Optional[DetectionResult] = None
    quality: Optional[FrameQuality] = None
    status: FrameStatus = FrameStatus.PENDING
    disabled_reason: Optional[str] = None       # 예: "high_reprojection_error"
    reprojection_error: Optional[float] = None  # 최종 캘리브레이션 기준 프레임별 오차

    @property
    def enabled(self) -> bool:
        return self.status in (FrameStatus.PENDING, FrameStatus.DETECTED)

    def disable(self, reason: str, outlier: bool = True) -> None:
        self.status = FrameStatus.DISABLED_OUTLIER if outlier else FrameStatus.DISABLED_MANUAL
        self.disabled_reason = reason


# ---------------------------------------------------------------------------
# Dataset (설계 문서 5, 7번 - Coverage / Diversity)
# ---------------------------------------------------------------------------

@dataclass
class CoverageCell:
    row: int
    col: int
    corner_count: int = 0
    coverage_score: float = 0.0   # 0~1


@dataclass
class DiversityScores:
    """설계 문서 7번 - 사진 개수보다 자세(Pose) 다양성"""
    position_coverage: float = 0.0   # 0~1
    distance_diversity: float = 0.0
    rotation_diversity: float = 0.0
    edge_coverage: float = 0.0

    @property
    def overall(self) -> float:
        values = [
            self.position_coverage,
            self.distance_diversity,
            self.rotation_diversity,
            self.edge_coverage,
        ]
        return sum(values) / len(values) if values else 0.0


@dataclass
class Dataset:
    frames: list[Frame] = field(default_factory=list)
    coverage_grid: list[CoverageCell] = field(default_factory=list)  # 예: 4x4 = 16개
    diversity: Optional[DiversityScores] = None

    @property
    def enabled_frames(self) -> list[Frame]:
        return [f for f in self.frames if f.enabled]

    @property
    def num_total(self) -> int:
        return len(self.frames)

    @property
    def num_enabled(self) -> int:
        return len(self.enabled_frames)

    @property
    def num_detected(self) -> int:
        return len([f for f in self.frames if f.detection and f.detection.success])


# ---------------------------------------------------------------------------
# 캘리브레이션 결과 (설계 문서 17번 Step4 CalibrationResult 확장)
# ---------------------------------------------------------------------------

@dataclass
class ParameterUncertainty:
    """설계 문서 3.2번 - calibrateCameraExtended()의 표준편차. V2 우선순위."""
    fx_std: Optional[float] = None
    fy_std: Optional[float] = None
    cx_std: Optional[float] = None
    cy_std: Optional[float] = None

    def is_within_threshold(self, fx: float, fy: float, ratio: float = 0.01) -> bool:
        """fx, fy 표준편차가 추정값의 1% 이내인지 (기본 threshold)"""
        if self.fx_std is None or self.fy_std is None:
            return False
        return (self.fx_std <= fx * ratio) and (self.fy_std <= fy * ratio)


@dataclass
class RegionalError:
    """설계 문서 4번 - 영역별 오차 분석"""
    center: Optional[float] = None
    left: Optional[float] = None
    right: Optional[float] = None
    top: Optional[float] = None
    bottom: Optional[float] = None
    corner: Optional[float] = None   # 네 귀퉁이 평균


@dataclass
class RadialBin:
    """설계 문서 4번 - Radial Error Profile의 구간 하나.
    이미지 중심으로부터의 반지름 구간별 평균 재투영 오차.
    """
    radius_min: float
    radius_max: float
    mean_error: Optional[float] = None
    num_points: int = 0   # 이 구간에 걸린 코너 포인트 개수 (프레임 수가 아님)

    @property
    def radius_center(self) -> float:
        return (self.radius_min + self.radius_max) / 2.0


@dataclass
class RadialErrorProfile:
    """설계 문서 4번 - "렌즈 외곽에서 모델이 잘 동작하는지" 확인용 그래프 데이터.
    코너 포인트 단위(프레임 단위 아님)로 집계해야 화각 전역의 경향을 정확히 반영한다.
    """
    bins: list[RadialBin] = field(default_factory=list)
    max_radius: float = 0.0   # 정규화(반지름 -> 0~1)에 사용할 수 있는 기준값 (이미지 대각선의 절반)


@dataclass
class CalibrationResult:
    """설계 문서 17번 Step4 CalibrationResult 그대로 + 영역별/반경별 오차 확장"""
    model_name: CameraModelType
    camera_matrix: Optional[np.ndarray] = None       # 3x3
    distortion: Optional[np.ndarray] = None           # 모델별로 길이 다름 (k1~k6, p1, p2 등)
    rvecs: list[np.ndarray] = field(default_factory=list)
    tvecs: list[np.ndarray] = field(default_factory=list)
    rms_error: Optional[float] = None
    per_frame_error: dict[str, float] = field(default_factory=dict)  # frame_id -> px error
    regional_error: Optional[RegionalError] = None
    radial_profile: Optional[RadialErrorProfile] = None  # 설계 문서 4번 V2 - Radial Error Profile
    param_uncertainty: Optional[ParameterUncertainty] = None
    success: bool = False
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Outlier (설계 문서 9번)
# ---------------------------------------------------------------------------

@dataclass
class OutlierResult:
    threshold_used: float                       # median(error) + 3*MAD, 또는 사용자 지정
    removed_frame_ids: list[str] = field(default_factory=list)
    rms_before: Optional[float] = None
    rms_after: Optional[float] = None
    iterations: int = 0
    max_iterations: int = 3


# ---------------------------------------------------------------------------
# Validation (설계 문서 3.3, 3.4번)
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Hold-out + Straightness 를 함께 담는다."""
    train_frame_ids: list[str] = field(default_factory=list)
    test_frame_ids: list[str] = field(default_factory=list)
    train_rms: Optional[float] = None
    test_rms: Optional[float] = None            # test intrinsic 재최적화 금지 원칙 준수
    edge_rms: Optional[float] = None
    straightness_residual: Optional[float] = None  # V2, 없으면 None
    # 5단계 추가: 학습 자체가 실패했거나(프레임 부족 등) 개별 test 프레임에서
    # solvePnP가 실패한 경우를 파이프라인이 죽지 않고 기록할 수 있도록.
    success: bool = True
    error_message: Optional[str] = None
    failed_test_frame_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 모델 추천 (설계 문서 8번 - Model Score)
# ---------------------------------------------------------------------------

@dataclass
class ModelScoreWeights:
    """Score = w1*E_train + w2*E_test + w3*E_edge + w4*E_line + w5*P"""
    w_train: float = 0.15
    w_test: float = 0.35
    w_edge: float = 0.25
    w_line: float = 0.10
    w_complexity: float = 0.15


@dataclass
class ModelScore:
    model_name: CameraModelType
    score: float
    components: dict[str, float] = field(default_factory=dict)  # 항목별 기여도 (디버깅/설명용)
    is_recommended: bool = False


# ---------------------------------------------------------------------------
# 최종 결과 (설계 문서 12번 - 종합 리포트)
# ---------------------------------------------------------------------------

@dataclass
class FinalResult:
    chosen_model: CameraModelType          # 추천이 아니라 "사용자가 최종 선택"한 모델
    calibration: CalibrationResult
    validation: Optional[ValidationResult] = None
    outlier: Optional[OutlierResult] = None
    dataset_coverage_pct: Optional[float] = None
    overall_grade: QualityGrade = QualityGrade.WARNING
    model_scores: list[ModelScore] = field(default_factory=list)  # 참고용 3개 모델 비교 스냅샷


# ---------------------------------------------------------------------------
# 프로젝트 루트 (설계 문서 18번)
# ---------------------------------------------------------------------------

@dataclass
class CalibrationProject:
    """설계 문서 13, 18번 - 전체를 감싸는 최상위 구조.
    LiDAR-Camera Extrinsic 등 향후 확장 필드는 별도 dataclass로 추가하되
    이 클래스에 Optional 필드로만 얹는 방식으로 확장한다 (V3, 아직 미구현).
    """
    project_name: str
    camera_config: CameraConfig
    pattern_config: PatternConfig
    dataset: Dataset = field(default_factory=Dataset)
    calibration_results: list[CalibrationResult] = field(default_factory=list)  # 3개 모델 동시 계산 결과
    outlier_result: Optional[OutlierResult] = None
    validation_result: Optional[ValidationResult] = None
    final_result: Optional[FinalResult] = None
    created_at: datetime = field(default_factory=datetime.now)
    project_dir: Optional[Path] = None
    export_paths: dict[ExportFormat, str] = field(default_factory=dict)

    def get_result_by_model(self, model: CameraModelType) -> Optional[CalibrationResult]:
        for r in self.calibration_results:
            if r.model_name == model:
                return r
        return None

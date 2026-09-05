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
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CameraModelType(str, Enum):
    """설계 문서 1번 - 카메라 모델 정의"""
    PINHOLE = "pinhole"
    BROWN_CONRADY = "brown_conrady"
    EXTENDED_PINHOLE = "extended_pinhole"
    FISHEYE = "fisheye"


class PatternType(str, Enum):
    """설계 문서 2번 - ChArUco / AprilGrid 지원"""
    CHESSBOARD = "chessboard"
    CHARUCO = "charuco"
    CIRCLE_GRID = "circle_grid"
    APRILGRID = "apriltag_grid"


class CircleGridType(str, Enum):
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


class AprilGridVariant(str, Enum):
    OPENCV_APRILTAG3 = "opencv_apriltag3"
    KALIBR = "kalibr"


class CalibrationMethod(str, Enum):
    STANDARD = "standard"
    OBJECT_RELEASING = "object_releasing"


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


class DiagnosisSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Config (촬영 조건 / 패턴 정의)
# ---------------------------------------------------------------------------

@dataclass
class PatternConfig:
    """설계 문서 10번 - 패턴 메타정보. 결과와 함께 저장되어야 재현 가능."""
    type: PatternType
    squares_x: int
    squares_y: int
    square_size: float          # meter 단위 권장. AprilGrid에서는 checker square가 아니라 tag pitch.
    marker_size: Optional[float] = None   # ChArUco / AprilGrid 전용
    dictionary: Optional[str] = None      # 예: "DICT_6X6_250"
    circle_grid_type: CircleGridType = CircleGridType.SYMMETRIC
    aprilgrid_variant: AprilGridVariant = AprilGridVariant.OPENCV_APRILTAG3

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
    # 설계 문서 3-1번 - 이미지 품질 검사 확장 필드 (전부 additive, 기존 필드는 안 건드림)
    contrast: Optional[float] = None            # grayscale std (낮을수록 흐릿/뿌연 이미지)
    saturation: Optional[float] = None          # 명부/암부 clipping 픽셀 비율 (0~1, 높을수록 나쁨)
    motion_blur_score: Optional[float] = None   # 방향성 블러 의심도 (1.0=등방성, 클수록 한쪽 방향으로만 블러)
    phash: Optional[str] = None                 # 중복/near-duplicate 판단용 perceptual hash (16진수 문자열)


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
    # 설계 문서 3-2번 - Calibration Target 품질 검사 확장 필드
    corner_confidence: Optional[float] = None   # 검출된 코너 수 / 보드가 이론상 가질 수 있는 최대 코너 수 (0~1)
    min_edge_margin_px: Optional[float] = None  # 코너 중 이미지 경계에 가장 가까운 코너까지의 거리(px)
    likely_cut_off: Optional[bool] = None       # min_edge_margin_px가 매우 작아 보드가 프레임 밖으로 잘렸을 가능성
    # 설계 문서 16번 - corner-level outlier. 프레임 전체는 정상인데 그 안의
    # 코너 몇 개만 유독 튄다면, 프레임을 통째로 버리는 대신 그 코너들만
    # calibration 입력에서 제외한다 (models/common.collect_calibration_inputs
    # 참고). corners/object_points/ids 배열 자체는 건드리지 않는다 - 원본 검출
    # 결과는 항상 보존하고, "이번 계산에서 몇 번 인덱스를 뺄지"만 별도로 기록한다.
    excluded_corner_indices: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 프레임 품질 (설계 문서 6번 - Frame Quality Score)
# ---------------------------------------------------------------------------

@dataclass
class FrameQuality:
    detection_score: float = 0.0   # 코너 수, confidence, blur, exposure 등 종합
    geometric_score: float = 0.0   # 중심과의 거리, 기울기, 중복도 등 종합
    overall_score: float = 0.0     # 0~100
    grade: QualityGrade = QualityGrade.POOR
    # 설계 문서 4번 - Dataset Quality Score의 개별 구성요소를 그대로 노출한다.
    # (0~100, "예시" 표의 Blur/Exposure/Corner Quality/Board Area/
    #  Edge Coverage/Pose Diversity 항목과 1:1 대응). detection_score/
    # geometric_score/overall_score는 기존 코드(project_io, dataset_view 등)
    # 호환을 위해 그대로 유지하고, 아래는 "왜 그 점수가 나왔는지" 보여주는
    # 세부 분해값이다 - 없으면(구버전 프로젝트 파일 로드 등) None으로 둔다.
    blur_score: Optional[float] = None
    exposure_score: Optional[float] = None
    corner_quality_score: Optional[float] = None
    board_area_score: Optional[float] = None
    edge_coverage_score: Optional[float] = None
    pose_diversity_score: Optional[float] = None


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
class DistributionStat:
    """설계 문서 6번 - 분포 하나(X위치/Y위치/면적/yaw/pitch/roll/거리)를
    mean/std/variance + 0~1 coverage 점수로 요약한 공용 그릇.
    coverage_score 계산 방식은 quality._normalized_spread()와 동일한 철학
    (표준편차가 넓게 퍼질수록 다양한 자세를 촬영했다는 뜻이므로 점수가 높다).
    """
    mean: Optional[float] = None
    std: Optional[float] = None
    variance: Optional[float] = None
    coverage_score: float = 0.0
    sample_count: int = 0


@dataclass
class PoseDistributionStats:
    """설계 문서 6번 - Pose Diversity 평가 확장. 사진 개수가 아니라
    "자세가 얼마나 다양했는가"를 7개 축으로 각각 분해해서 보여준다.

    yaw/pitch는 detector.py가 주는 board_tilt_deg(2D minAreaRect 각도, 사실상
    roll 근사치)만으로는 볼 수 없는 진짜 3D 회전이다 - 아직 카메라 파라미터가
    없는 단계이므로 정확한 값은 아니고, 거친 초기 추정(K를 f=max(w,h)로
    가정한 solvePnP)에 기반한 "다양성 진단용" 근사치임을 분명히 한다
    (quality.py의 _estimate_rough_pose_angles docstring 참고).
    """
    x_position: DistributionStat = field(default_factory=DistributionStat)
    y_position: DistributionStat = field(default_factory=DistributionStat)
    board_area: DistributionStat = field(default_factory=DistributionStat)
    yaw: DistributionStat = field(default_factory=DistributionStat)
    pitch: DistributionStat = field(default_factory=DistributionStat)
    roll: DistributionStat = field(default_factory=DistributionStat)
    distance: DistributionStat = field(default_factory=DistributionStat)


@dataclass
class DatasetQualityScore:
    """설계 문서 4번 - "Overall Dataset Score". 개별 프레임 점수(FrameQuality)와는
    다른 층위 - "이 데이터셋 전체가 캘리브레이션을 하기에 충분히 좋은가"를
    하나의 숫자와 근거로 요약한다.
    """
    avg_frame_quality: float = 0.0       # 활성화된 프레임들의 FrameQuality.overall_score 평균
    detection_success_rate: float = 0.0  # 검출 성공 / 전체 (0~100)
    coverage_score: float = 0.0          # quality.coverage_percentage 재사용 (0~100)
    diversity_score: float = 0.0         # DiversityScores.overall * 100
    duplicate_penalty: float = 0.0       # (거의)중복 이미지 비율에 비례한 감점 (0~100)
    overall: float = 0.0                 # 위 항목들을 가중합산한 최종 점수 (0~100)
    grade: QualityGrade = QualityGrade.POOR


@dataclass
class Dataset:
    frames: list[Frame] = field(default_factory=list)
    coverage_grid: list[CoverageCell] = field(default_factory=list)  # 예: 4x4 = 16개
    diversity: Optional[DiversityScores] = None
    quality_score: Optional[DatasetQualityScore] = None  # 설계 문서 4번

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
class DistortionCoeffStat:
    """설계 문서 20/21번 - distortion 계수(k1,k2,k3,...) 하나에 대한 bootstrap 통계.

    개수가 모델마다 다르므로(Pinhole 0개, Extended 5~8개, Fisheye 4개)
    ParameterUncertainty에 k1_std 식으로 필드를 늘어놓지 않고 리스트로 둔다.
    """
    index: int
    label: str = ""  # 예: "k1", "p1" - models/common.distortion_coeff_labels() 참고
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    stability_score: Optional[float] = None  # 0~100, 설계 문서 23번


@dataclass
class ParameterUncertainty:
    """설계 문서 3.2번 - calibrateCameraExtended()의 표준편차. V2 우선순위.

    설계 문서 21/22/23번 - Parameter Stability / Confidence Interval / Stability
    Score 확장. method가 "covariance"면 fx_std 등은 OpenCV의
    stdDeviationsIntrinsics에서, "bootstrap"이면 재표본화 반복 결과의
    표준편차/percentile에서 나온다 - 출처가 다르면 CI 계산 방식도 다르다
    (covariance는 정규근사 mean±1.96*std, bootstrap은 2.5/97.5 percentile 그대로
    사용 - 코드 내 compute_parameter_bootstrap docstring 참고). median/min/max는
    실제 재표본 분포가 있어야 의미가 있으므로 method="bootstrap"일 때만 채워진다.
    """
    fx_std: Optional[float] = None
    fy_std: Optional[float] = None
    cx_std: Optional[float] = None
    cy_std: Optional[float] = None
    method: str = "covariance"  # "covariance" | "bootstrap"
    n_bootstrap_success: Optional[int] = None  # method="bootstrap"일 때만 의미 있음
    fx_ci_low: Optional[float] = None
    fx_ci_high: Optional[float] = None
    fy_ci_low: Optional[float] = None
    fy_ci_high: Optional[float] = None
    cx_ci_low: Optional[float] = None
    cx_ci_high: Optional[float] = None
    cy_ci_low: Optional[float] = None
    cy_ci_high: Optional[float] = None
    # 설계 문서 22번 - "fx = 812.3 ± 2.1" 형태로 보여주려면 평균값 자체도 있어야
    # 한다. covariance 방식은 호출부가 실제 fit 결과(camera_matrix)의 값을 그대로
    # 채워 넣고, bootstrap 방식은 재표본들의 평균을 쓴다 - 후자는 원본 전체
    # 데이터 fit 값과 미세하게 다를 수 있다(재표본 평균이므로).
    fx_mean: Optional[float] = None
    fy_mean: Optional[float] = None
    cx_mean: Optional[float] = None
    cy_mean: Optional[float] = None
    # 설계 문서 21번 - Median/Min/Max (method="bootstrap"일 때만 의미 있음)
    fx_median: Optional[float] = None
    fy_median: Optional[float] = None
    cx_median: Optional[float] = None
    cy_median: Optional[float] = None
    fx_min: Optional[float] = None
    fx_max: Optional[float] = None
    fy_min: Optional[float] = None
    fy_max: Optional[float] = None
    cx_min: Optional[float] = None
    cx_max: Optional[float] = None
    cy_min: Optional[float] = None
    cy_max: Optional[float] = None
    # 설계 문서 23번 - Parameter Stability Score (0~100, 변동계수 기반 -
    # repeatability.py의 CV->점수 변환과 동일한 공식: 100*(1-CV), 0~100으로 clip)
    fx_stability: Optional[float] = None
    fy_stability: Optional[float] = None
    cx_stability: Optional[float] = None
    cy_stability: Optional[float] = None
    overall_stability: Optional[float] = None  # 위 항목(+distortion) 전부의 평균
    # 설계 문서 20/21번 - distortion 계수(k1,k2,...)별 bootstrap 통계
    distortion_stats: list["DistortionCoeffStat"] = field(default_factory=list)

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
    """설계 문서 4번/14번 - Radial Error Profile의 구간 하나.
    이미지 중심으로부터의 반지름 구간별 재투영 오차 통계.
    """
    radius_min: float
    radius_max: float
    mean_error: Optional[float] = None
    num_points: int = 0   # 이 구간에 걸린 코너 포인트 개수 (프레임 수가 아님)
    # 설계 문서 14번 - "Mean/Median/RMS/P95/Max" 전부. mean_error는 기존 호환용으로 남긴다.
    median_error: Optional[float] = None
    rms_error: Optional[float] = None
    p95_error: Optional[float] = None
    max_error: Optional[float] = None
    label: Optional[str] = None  # 예: "Center", "Inner", "Middle", "Outer", "Edge", "Corner"

    @property
    def radius_center(self) -> float:
        return (self.radius_min + self.radius_max) / 2.0


@dataclass
class SpatialErrorCell:
    """설계 문서 13번 - Spatial Error Map 한 칸.

    quality.CoverageCell(코너 "개수"만 셈)과 달리, 이 셀은 그 안에 찍힌
    코너들의 재투영 "오차"를 요약한다 - 특히 dx/dy 평균(=이 영역이 어느
    방향으로 치우쳐 틀리는가)이 CoverageCell엔 없는 정보다.
    """
    row: int
    col: int
    num_points: int = 0
    rms: Optional[float] = None          # 이 칸에 찍힌 포인트들의 재투영 오차 RMS
    p95: Optional[float] = None          # 같은 칸의 P95 (문서 13번 "image grid별 P95")
    mean_dx: Optional[float] = None      # 부호 있는 x방향 평균 오차 (양수=오른쪽으로 치우침)
    mean_dy: Optional[float] = None      # 부호 있는 y방향 평균 오차 (양수=아래쪽으로 치우침)
    direction_deg: Optional[float] = None  # atan2(mean_dy, mean_dx), 0=+x(오른쪽), 90=+y(아래쪽)


@dataclass
class SpatialErrorMap:
    """설계 문서 13번 - "residual direction(X/Y 방향)" heatmap.

    체계적인 방향 패턴(예: 모든 칸의 화살표가 바깥쪽을 향함)이 보이면
    카메라 모델이 데이터를 충분히 설명하지 못하고 있다는 신호다 - 이 구조
    자체가 "잔차가 무작위(방향 없음)인가, 편향(방향 있음)인가"를 진단하는
    목적이라 rows x cols 그리드 + 셀별 (rms, p95, mean_dx, mean_dy)를 그대로 둔다.
    """
    cells: list[SpatialErrorCell] = field(default_factory=list)
    rows: int = 4
    cols: int = 4


@dataclass
class RadialErrorProfile:
    """설계 문서 4번 - "렌즈 외곽에서 모델이 잘 동작하는지" 확인용 그래프 데이터.
    코너 포인트 단위(프레임 단위 아님)로 집계해야 화각 전역의 경향을 정확히 반영한다.
    """
    bins: list[RadialBin] = field(default_factory=list)
    max_radius: float = 0.0   # 정규화(반지름 -> 0~1)에 사용할 수 있는 기준값 (이미지 대각선의 절반)


@dataclass
class ResidualStats:
    """설계 문서 11번/12번 - Reprojection Error 지표 확장 + Residual Distribution.

    per_frame_error(프레임당 RMS 하나)보다 훨씬 세밀하다 - 모든 코너 포인트
    각각의 재투영 오차(Euclidean 거리)를 모아 표준 통계량과 histogram/CDF/
    박스플롯에 필요한 값을 전부 계산해둔다. RMS 하나만 보고 "좋다/나쁘다"를
    판단하지 않는다는 이 프로젝트의 핵심 철학을 오차 분포 층위에도 그대로
    적용한 것.

    histogram_bin_edges/histogram_counts: 원본 포인트 배열 전체를 저장하는
    대신(수천 개일 수 있어 프로젝트 파일이 불필요하게 커짐) 미리 집계한
    히스토그램만 저장한다 - CDF는 이 카운트의 누적합으로 그대로 그릴 수 있고,
    박스플롯은 min/q1/median/q3/max만 있으면 충분하다.
    outlier_count: outlier.py와 동일한 기준(median + 3*MAD)으로 판단한 이상치
    포인트 개수 - 앱 전체에서 "이상치"의 정의가 하나로 통일되도록 재사용한다.
    """
    n: int = 0
    rmse: Optional[float] = None
    mae: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    p90: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None
    max: Optional[float] = None
    outlier_count: int = 0
    histogram_bin_edges: list[float] = field(default_factory=list)
    histogram_counts: list[int] = field(default_factory=list)
    # 설계 문서 12번 "corner별 residual" - 전체 코너 포인트 원본 배열은 수천
    # 개일 수 있어 그대로 저장하면 프로젝트 파일이 불필요하게 커지므로,
    # 대표성 있는 무작위 표본(최대 _MAX_SAMPLE_RESIDUALS개)만 남긴다.
    # histogram으로 전체 분포 형태는 이미 알 수 있고, 이 표본은 산점도/strip
    # plot처럼 "개별 포인트"를 보여주고 싶을 때만 보조적으로 쓰인다.
    sample_residuals: list[float] = field(default_factory=list)


@dataclass
class ParameterCorrelation:
    """Observability 진단에서 강하게 얽힌 파라미터 쌍."""
    param_a: str
    param_b: str
    correlation: float


@dataclass
class ObservabilityReport:
    """Jacobian 기반 관측가능성 진단 요약.

    Jacobian 원본은 커질 수 있으므로 저장하지 않고, SVD/condition/correlation
    요약만 CalibrationResult에 붙인다.
    """
    parameter_labels: list[str] = field(default_factory=list)
    jacobian_rows: int = 0
    jacobian_cols: int = 0
    num_points: int = 0
    singular_values: list[float] = field(default_factory=list)
    rank: int = 0
    condition_number: Optional[float] = None
    min_singular_value: Optional[float] = None
    max_singular_value: Optional[float] = None
    max_abs_correlation: Optional[float] = None
    correlation_matrix: list[list[float]] = field(default_factory=list)
    observability_score: Optional[float] = None  # 0~100, 높을수록 좋음
    observability_grade: Optional[str] = None  # "GOOD" | "WARNING" | "POOR"
    top_correlations: list[ParameterCorrelation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class UndistortionQualityReport:
    """Undistort 결과에서 실제로 쓸 수 있는 영상 영역 품질 요약."""
    image_width: int = 0
    image_height: int = 0
    valid_pixel_ratio: float = 0.0       # 0~1, remap 결과가 원본 이미지 안을 참조하는 비율
    black_border_ratio: float = 0.0      # 0~1, remap 기준 border로 채워질 픽셀 비율
    roi_loss_ratio: float = 0.0          # 0~1, all-valid ROI로 crop할 때 잃는 면적 비율
    valid_roi: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    undistorted_black_pixel_ratio: Optional[float] = None  # 실제 샘플 undistort 이미지 기반
    sample_frame_id: Optional[str] = None
    quality_score: float = 0.0           # 0~100, 높을수록 좋음
    quality_grade: QualityGrade = QualityGrade.WARNING
    warnings: list[str] = field(default_factory=list)


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
    radial_bands: Optional[RadialErrorProfile] = None    # 설계 문서 14번 - Center~Corner 6단계 명명 대역
    spatial_error_map: Optional[SpatialErrorMap] = None  # 설계 문서 13번 - X/Y 방향 heatmap
    param_uncertainty: Optional[ParameterUncertainty] = None
    param_uncertainty_bootstrap: Optional[ParameterUncertainty] = None  # 설계 문서 20번 - 전 모델 공통 bootstrap
    residual_stats: Optional[ResidualStats] = None  # 설계 문서 11/12번 - 코너 포인트 단위 오차 분포
    observability: Optional[ObservabilityReport] = None  # Jacobian/SVD/condition/correlation 진단
    undistortion_quality: Optional[UndistortionQualityReport] = None  # valid pixel/black border/ROI loss
    calibration_method: CalibrationMethod = CalibrationMethod.STANDARD
    refined_object_points: Optional[np.ndarray] = None
    target_geometry_refinement: Optional[dict[str, float]] = None
    object_releasing_diagnostics: list[dict] = field(default_factory=list)
    success: bool = False
    error_message: Optional[str] = None
    # success=True인데도 사용자에게 알려야 할 게 있을 때(예: Fisheye가 특정
    # 프레임을 캘리브레이션에서 자동 제외했을 때) 쓰는 비-치명적 경고.
    # error_message와 분리한 이유: error_message는 "실패"를 의미하는 필드라
    # success=True와 함께 쓰면 UI 로직이 헷갈린다.
    warning_message: Optional[str] = None


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
    # 설계 문서 17번 - "Outlier 제거 전후 효과 측정"을 RMS 하나가 아니라
    # P95와 파라미터(fx/fy/distortion)까지 확장. camera_matrix/distortion은
    # 원본 배열(3x3, Nx1)을 그대로 스냅샷해둔다 - fx/fy만 따로 안 뽑는 이유는
    # 모델마다 distortion 길이가 다르고(Pinhole 0개, Fisheye 4개 등) 배열째로
    # 보관해야 어떤 모델이든 동일한 코드로 비교/표시할 수 있기 때문이다.
    p95_before: Optional[float] = None
    p95_after: Optional[float] = None
    camera_matrix_before: Optional[np.ndarray] = None
    camera_matrix_after: Optional[np.ndarray] = None
    distortion_before: Optional[np.ndarray] = None
    distortion_after: Optional[np.ndarray] = None


@dataclass
class CornerOutlierResult:
    """설계 문서 16번 - corner-level outlier 버전의 OutlierResult.
    프레임을 통째로 세는 대신 "몇 개 프레임에서 몇 개의 코너를 뺐는지"를 기록한다.
    """
    threshold_used: float = 0.0
    removed_corners: dict[str, list[int]] = field(default_factory=dict)  # frame_id -> 제외된 코너 인덱스들
    rms_before: Optional[float] = None
    rms_after: Optional[float] = None
    iterations: int = 0
    max_iterations: int = 3
    # 설계 문서 17번 - "RMSE/P95/parameter 변화"를 corner-level 제거에도 동일하게
    # 기록한다 (OutlierResult의 프레임 단위 버전과 필드 구성을 맞춰서, 두 결과
    # 타입 모두 outlier.format_outlier_before_after류 함수 하나로 다룰 수 있게 함).
    p95_before: Optional[float] = None
    p95_after: Optional[float] = None
    camera_matrix_before: Optional[np.ndarray] = None
    camera_matrix_after: Optional[np.ndarray] = None
    distortion_before: Optional[np.ndarray] = None
    distortion_after: Optional[np.ndarray] = None

    @property
    def total_corners_removed(self) -> int:
        return sum(len(v) for v in self.removed_corners.values())


# ---------------------------------------------------------------------------
# Validation (설계 문서 3.3, 3.4번)
# ---------------------------------------------------------------------------

@dataclass
class StraightnessBreakdown:
    """설계 문서 15번 - Line Straightness 평가 강화.

    기존 straightness_residual(스칼라 하나)은 "전체 평균"만 보여줬다 - 이걸
    방향(수평/수직/대각선)과 보드 내 위치(중앙/가장자리)로 쪼갠다. 예를 들어
    edge_line_error가 center_line_error보다 훨씬 크면 방사 왜곡 보정이 외곽에서
    덜 되고 있다는 뜻이고, diagonal_error만 유독 크면 접선(tangential) 왜곡
    쪽 문제일 가능성을 시사한다.
    """
    horizontal_error: Optional[float] = None
    vertical_error: Optional[float] = None
    diagonal_error: Optional[float] = None
    center_line_error: Optional[float] = None
    edge_line_error: Optional[float] = None
    corner_line_error: Optional[float] = None
    overall_error: Optional[float] = None
    num_lines: int = 0


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
    # 설계 문서 15번 - straightness_residual(스칼라 하나) 대신 방향/위치별로
    # 쪼갠 값. straightness_residual == straightness_breakdown.overall_error다
    # (하위 호환을 위해 둘 다 채운다).
    straightness_breakdown: Optional[StraightnessBreakdown] = None
    # 설계 문서 10번 - Hold-out Validation 강화. train_rms/test_rms(RMS 하나)만
    # 보지 않고 MAE/Median/Std/P90/P95/P99/Max까지 Train/Test 양쪽에 각각 둔다.
    # train_residual_stats는 이 모델을 학습시킨 CalibrationResult.residual_stats를
    # 그대로 참조하고, test_residual_stats는 _evaluate_on_test()가 test 프레임의
    # 코너 포인트 오차로 새로 계산한다.
    train_residual_stats: Optional[ResidualStats] = None
    test_residual_stats: Optional[ResidualStats] = None
    # 5단계 추가: 학습 자체가 실패했거나(프레임 부족 등) 개별 test 프레임에서
    # solvePnP가 실패한 경우를 파이프라인이 죽지 않고 기록할 수 있도록.
    success: bool = True
    error_message: Optional[str] = None
    failed_test_frame_ids: list[str] = field(default_factory=list)


@dataclass
class SceneQualityEntry:
    """Initial calibration을 기준으로 계산한 scene 한 장의 ranking 정보."""
    frame_id: str
    rank: int = 0
    quality_score: float = 0.0
    reprojection_error: Optional[float] = None
    detection_ratio: float = 0.0
    sharpness: Optional[float] = None
    reprojection_score: float = 50.0
    detection_score: float = 0.0
    sharpness_score: float = 50.0


@dataclass
class SceneQualityAnalysis:
    """Ranking이 어느 원본 camera model을 기준으로 했는지까지 보존."""
    model_name: CameraModelType
    scenes: list[SceneQualityEntry] = field(default_factory=list)


@dataclass
class SubsetCalibrationResult:
    """Original을 덮어쓰지 않는 사용자 선택 subset 재캘리브레이션 결과."""
    model_name: CameraModelType
    selected_frame_ids: list[str] = field(default_factory=list)
    calibration_result: Optional[CalibrationResult] = None
    validation_result: Optional[ValidationResult] = None
    original_validation_result: Optional[ValidationResult] = None
    coverage_grid: list[CoverageCell] = field(default_factory=list)
    diversity: Optional[DiversityScores] = None
    coverage_percentage: float = 0.0
    original_coverage_percentage: float = 0.0
    original_diversity: Optional[DiversityScores] = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class ObjectReleasingValidationResult:
    """Object-Releasing 전용 Hold-out 결과.

    ValidationResult과 분리한 이유: Object-Releasing은 Train에서 K/D뿐 아니라
    Refined Target Geometry까지 함께 확정하고, Test에서는 이 셋을 전부 고정한 채
    pose(solvePnP)만 다시 구한다 - Standard Hold-out(_evaluate_on_test)의 test
    프레임은 각자의 nominal object_points를 쓰지만, 여기서는 반드시 Train에서
    나온 refined_object_points를 재사용해야 하므로 계산 경로 자체가 다르다.
    별도 타입으로 두면 Standard 쪽 ValidationResult/직렬화에 영향을 주지 않고
    이 계약을 명시적으로 강제할 수 있다.
    """
    success: bool = True
    error_message: Optional[str] = None
    train_frame_ids: list[str] = field(default_factory=list)
    test_frame_ids: list[str] = field(default_factory=list)
    # Full-board가 아니어서애초에 eligible pool에도 못 들어간 프레임들
    # (collect_object_releasing_inputs의 diagnostics에서 그대로 가져온다 -
    # "이유 없이 조용히 skip"하지 않기 위함).
    excluded_frame_ids: list[str] = field(default_factory=list)
    excluded_reasons: dict[str, str] = field(default_factory=dict)
    # Test 시점에 방어적으로 재검증했을 때 실패한 프레임 (정상 흐름에서는
    # 비어있어야 함 - eligible pool 단계에서 이미 걸러졌으므로).
    failed_test_frame_ids: list[str] = field(default_factory=list)
    failed_test_reasons: dict[str, str] = field(default_factory=dict)
    train_rms: Optional[float] = None
    test_rms: Optional[float] = None
    test_residual_stats: Optional[ResidualStats] = None
    target_geometry_refinement: Optional[dict[str, float]] = None


@dataclass
class StandardVsObjectReleasingComparison:
    """Standard Brown-Conrady와 Object-Releasing Brown-Conrady의 공정 비교.

    두 arm이 반드시 "같은 eligible full-board 데이터셋" + "같은 train/test
    분할"을 쓰도록 강제하는 것이 이 타입의 존재 이유 - train_frame_ids/
    test_frame_ids가 두 arm 모두에 공통으로 딱 하나씩만 존재한다.
    """
    success: bool = True
    error_message: Optional[str] = None
    eligible_frame_ids: list[str] = field(default_factory=list)
    train_frame_ids: list[str] = field(default_factory=list)
    test_frame_ids: list[str] = field(default_factory=list)
    standard_result: Optional[CalibrationResult] = None
    standard_validation: Optional[ValidationResult] = None
    object_releasing_result: Optional[CalibrationResult] = None
    object_releasing_validation: Optional[ObjectReleasingValidationResult] = None
    # ro - standard, 키: fx/fy/cx/cy/k1/k2/p1/p2/k3
    intrinsics_delta: dict[str, float] = field(default_factory=dict)
    # 사실만 기술하는 경고 (예: "train은 좋아졌는데 hold-out은 그대로") -
    # "RO가 더 정확하다" 같은 자동 판정 문구는 절대 넣지 않는다.
    warnings: list[str] = field(default_factory=list)


@dataclass
class CrossDatasetValidationResult:
    """Dataset A에서 학습한 calibration을 Dataset B/C에 고정 평가한 결과."""
    source_dataset_id: str
    target_dataset_id: str
    model_name: CameraModelType
    train_rms: Optional[float] = None
    test_rms: Optional[float] = None
    test_p95: Optional[float] = None
    edge_rms: Optional[float] = None
    straightness_residual: Optional[float] = None
    generalization_gap: Optional[float] = None
    num_test_frames: int = 0
    failed_test_frame_ids: list[str] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# 설계 문서 18/19번 - K-Fold / Repeated K-Fold Cross Validation
# ---------------------------------------------------------------------------

@dataclass
class KFoldResult:
    """단일 K-Fold(예: 5-fold) 결과. Hold-out(1회 분할)의 한계 - "운 좋게/나쁘게
    뽑힌 test set"에 좌우될 수 있음 - 을, 데이터를 k개로 나눠 각자 한 번씩
    test가 되게 함으로써 완화한다.
    """
    k: int = 5
    fold_validation_results: list[ValidationResult] = field(default_factory=list)
    mean_test_rms: Optional[float] = None
    std_test_rms: Optional[float] = None
    min_test_rms: Optional[float] = None
    max_test_rms: Optional[float] = None
    mean_test_p95: Optional[float] = None
    std_test_p95: Optional[float] = None
    n_successful_folds: int = 0


@dataclass
class RepeatedKFoldResult:
    """설계 문서 19번 - K-Fold를 여러 번(예: 5-fold x 5회) 반복해서, fold 나누는
    방식 자체가 우연히 좋거나 나쁘게 뽑히는 효과까지 평균으로 눌러준다.
    """
    k: int = 5
    n_repeats: int = 5
    kfold_results: list[KFoldResult] = field(default_factory=list)
    mean_test_rms: Optional[float] = None
    std_test_rms: Optional[float] = None
    min_test_rms: Optional[float] = None
    max_test_rms: Optional[float] = None
    mean_test_p95: Optional[float] = None
    std_test_p95: Optional[float] = None
    n_successful_runs: int = 0


# ---------------------------------------------------------------------------
# 설계 문서 40번 - Calibration Repeatability
# ---------------------------------------------------------------------------

@dataclass
class RepeatabilityResult:
    """같은 데이터셋으로 여러 번 캘리브레이션해도 결과가 일관되는지 측정.

    cv2의 최적화는 결정론적이라(초기값도 선형근사로 고정 계산됨) 프레임 순서를
    바꿔도 실질적으로 항상 같은 해에 수렴하는 게 "정상"이다 - 그래서 대부분의
    경우 repeatability_pct가 매우 높게(거의 100%) 나오는 게 자연스러운 결과이지,
    계산이 잘못됐다는 신호가 아니다. 이 지표가 실제로 의미 있게 갈리는 경우는
    데이터가 부실해서(프레임 부족, 극단적 outlier) 최적화가 매번 다른 국소해에
    걸릴 위험이 있을 때다.
    """
    n_runs: int = 5
    n_successful: int = 0
    order_runs: int = 0
    order_successful: int = 0
    initial_condition_runs: int = 0
    initial_condition_successful: int = 0
    initial_condition_perturbation: float = 0.0
    fx_cv: Optional[float] = None  # 변동계수(표준편차/평균) - 작을수록 안정적
    fy_cv: Optional[float] = None
    cx_cv: Optional[float] = None
    cy_cv: Optional[float] = None
    rms_std: Optional[float] = None
    repeatability_pct: Optional[float] = None  # 100 * (1 - 평균 CV), 0~100


# ---------------------------------------------------------------------------
# 모델 추천 (설계 문서 8번 - Model Score)
# ---------------------------------------------------------------------------

@dataclass
class ModelScoreWeights:
    """Model selection score weights. 낮을수록 좋은 weighted penalty.

    합성 데이터로 튜닝을 시도해봤지만(scripts/tune_model_score_weights.py),
    held-out 검증에서 과적합 위험이 커서 Test/P95/Edge 중심의 보수적
    기본값을 둔다. AIC/BIC/Stability/Observability는 tie-breaker 성격의
    보조 신호로 실제 score에 포함한다.
    """
    w_train: float = 0.08
    w_test: float = 0.22
    w_edge: float = 0.16
    w_line: float = 0.06
    w_complexity: float = 0.08
    w_p95: float = 0.14
    w_radial: float = 0.08
    w_aic: float = 0.04
    w_bic: float = 0.04
    w_stability: float = 0.05
    w_observability: float = 0.05


@dataclass
class ModelScore:
    model_name: CameraModelType
    score: float
    components: dict[str, float] = field(default_factory=dict)  # 항목별 기여도 (디버깅/설명용)
    is_recommended: bool = False
    # 설계 문서 24/25번 - AIC/BIC. score 공식에는 아직 직접 섞지 않고
    # 모델 비교/리포트가 원본 값을 그대로 보여줄 수 있게 함께 보관한다.
    parameter_count: int = 0
    residual_sum_squares: Optional[float] = None
    num_observations: int = 0
    aic: Optional[float] = None
    bic: Optional[float] = None
    # 설계 문서 28/29번 - 추천 자체의 신뢰도. 1위와 2위가 score/test/P95/edge
    # 기준으로 거의 같으면 LOW로 낮춰 "모델 차이가 작다"는 경고를 함께 보여준다.
    selection_confidence: Optional[float] = None  # 0~100, 추천 모델에 주로 채움
    selection_confidence_level: Optional[str] = None  # "HIGH" | "MEDIUM" | "LOW"
    selection_confidence_reason: Optional[str] = None
    selection_reasons: list[str] = field(default_factory=list)


@dataclass
class FailurePattern:
    """Metric 조합을 사람이 읽을 수 있는 failure pattern으로 바꾼 결과."""
    code: str
    severity: DiagnosisSeverity
    title: str
    evidence: list[str] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class CaptureRecommendation:
    """다음 촬영에서 어떤 보드를 어떻게 추가하면 좋은지에 대한 실행 항목."""
    code: str
    priority: str
    title: str
    action: str
    reason: str = ""


@dataclass
class DiagnosisReport:
    """최종 모델 기준 diagnosis/recommendation 묶음."""
    model_name: CameraModelType
    patterns: list[FailurePattern] = field(default_factory=list)
    capture_recommendations: list[CaptureRecommendation] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        return any(p.severity in (DiagnosisSeverity.WARNING, DiagnosisSeverity.ERROR) for p in self.patterns)


@dataclass
class CalibrationConfidenceReport:
    """최종 calibration 신뢰도를 0~100 점수와 근거 분해로 표현."""
    score: float = 0.0
    level: str = "LOW"  # "HIGH" | "MEDIUM" | "LOW" | "REJECT"
    components: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 최종 결과 (설계 문서 12번 - 종합 리포트)
# ---------------------------------------------------------------------------

@dataclass
class FinalResult:
    chosen_model: CameraModelType          # 추천이 아니라 "사용자가 최종 선택"한 모델
    calibration: CalibrationResult
    validation: Optional[ValidationResult] = None
    outlier: Optional[OutlierResult] = None
    corner_outlier: Optional[CornerOutlierResult] = None  # 설계 문서 16번 - corner-level 제거 결과
    dataset_coverage_pct: Optional[float] = None
    overall_grade: QualityGrade = QualityGrade.WARNING
    confidence: Optional[CalibrationConfidenceReport] = None
    model_scores: list[ModelScore] = field(default_factory=list)  # 참고용 Standard 4모델 비교 스냅샷
    diagnosis: Optional[DiagnosisReport] = None


# ---------------------------------------------------------------------------
# 프로젝트 루트 (설계 문서 18번)
# ---------------------------------------------------------------------------

@dataclass
class CalibrationProject:
    """설계 문서 13, 18번 - 전체를 감싸는 최상위 구조. 저장/불러오기(.ccproj)의
    최상위 컨테이너 - calibration/project_io.py가 이 클래스를 JSON으로
    직렬화/역직렬화한다.

    필드 형태를 실제 UI(ui/main_window.py)와 CLI(app/cli.py)가 런타임에
    들고 다니는 형태(dict[CameraModelType, ...])에 맞췄다 - 원래 설계
    문서 18번 초안은 list/단일값이었지만, Standard 4모델을 항상 함께 다루는
    실제 파이프라인 구조상 모델별 dict가 훨씬 자연스럽고 실수를 줄인다.
    Object-Releasing(Advanced)은 이 dict에 섞이지 않고 별도 필드
    (object_releasing_result/object_releasing_validation_result/
    standard_vs_object_releasing_comparison)로 분리해서 담는다.
    """
    project_name: str
    camera_config: CameraConfig
    pattern_config: PatternConfig
    dataset: Dataset = field(default_factory=Dataset)
    calibration_results: dict[CameraModelType, CalibrationResult] = field(default_factory=dict)
    object_releasing_result: Optional[CalibrationResult] = None
    object_releasing_validation_result: Optional[ObjectReleasingValidationResult] = None
    standard_vs_object_releasing_comparison: Optional[StandardVsObjectReleasingComparison] = None
    validation_results: dict[CameraModelType, ValidationResult] = field(default_factory=dict)
    cross_dataset_results: list[CrossDatasetValidationResult] = field(default_factory=list)
    model_scores: list[ModelScore] = field(default_factory=list)
    outlier_result: Optional[OutlierResult] = None
    scene_quality_analysis: Optional[SceneQualityAnalysis] = None
    subset_calibration_result: Optional[SubsetCalibrationResult] = None
    final_result: Optional[FinalResult] = None
    # Windshield Refraction Calibration (calibration/windshield/) - Object-Releasing과
    # 같은 패턴으로 별도 필드에 담는다. calibration.types는 calibration.windshield를
    # import하지 않으므로(순환 참조 방지) 타입은 문자열 forward-reference로 둔다 -
    # from __future__ import annotations(이 파일 상단)로 dataclass 정의 자체는
    # 문제 없이 동작하고, 실제 클래스는 project_io.py에서만 import한다.
    windshield_config: Optional["WindshieldConfig"] = None
    windshield_dataset: Optional[Dataset] = None
    windshield_results: dict["WindshieldModelType", "WindshieldCalibrationResult"] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    export_paths: dict[ExportFormat, str] = field(default_factory=dict)

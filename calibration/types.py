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
    # 3D pose 근사치(models.common.estimate_rough_pose, solvePnP + rough K).
    # board_tilt_deg(2D minAreaRect 각도)는 in-plane rotation 근사치일 뿐 진짜
    # yaw/pitch를 담지 못한다 - 이 4개 필드가 그 대체다. detect_charuco/
    # detect_chessboard/detect_circle_grid/detect_aprilgrid가 검출 직후 채운다.
    # 계산 불가(포인트 부족/solvePnP 실패)면 전부 None으로 남는다(추가 필드라
    # 구버전 프로젝트 로드 시에도 자동으로 None) - 소비처는 반드시 None
    # fallback 경로를 가져야 한다.
    yaw_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    distance_m: Optional[float] = None
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
    # Paper Evidence 단계 추가 - "왜" stability_score가 낮은지 진단할 수 있게
    # reference(전체 데이터 fit 값)와 순수 CV(=std/|bootstrap mean|, reference와
    # 섞지 않은 값 - 논문 수식 CV_p = sigma_p/mu_p 그대로)를 별도로 남긴다.
    # stability_score 자체(위 필드)의 계산식은 절대 바꾸지 않는다 - 그 값을
    # 소비하는 recommender/UI가 조용히 달라지면 안 되기 때문이다.
    reference: Optional[float] = None
    relative_cv: Optional[float] = None
    # reference(전체 데이터 fit 값)가 0에 가까우면 std가 아주 작아도 상대
    # CV가 폭발할 수 있다 - "이 계수가 실제로 불안정해서"가 아니라 "0 근처라
    # 상대 지표 자체가 통계적으로 의미가 약해서" stability가 낮게 보일 수
    # 있다는 것을 명시적으로 표시한다(값 자체를 보정하지 않는다).
    near_zero_reference: bool = False
    diagnostic: Optional[str] = None  # 예: "near-zero coefficient; relative CV unstable"


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
    n_bootstrap_total: Optional[int] = None  # 시도한 전체 재표본 수(compute_parameter_bootstrap의 n_bootstrap) - "18/20"처럼 보여주기 위함
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
    # "All-Parameter Stability" - fx/fy/cx/cy stability + distortion_stats의
    # stability_score까지 전부 평균한 값(설계 문서 23번 원래 정의 그대로,
    # 값 계산식은 절대 바꾸지 않는다). recommender.py의 모델 선택 점수가
    # 이미 이 필드를 소비하고 있어 backward compatibility가 필요하다 -
    # 이름을 paper metric으로 바꾸지 않고 "all-parameter"라는 의미를
    # docstring/UI/report에 명확히 남기는 방식으로만 구분한다.
    overall_stability: Optional[float] = None
    # 설계 문서 20/21번 - distortion 계수(k1,k2,...)별 bootstrap 통계
    distortion_stats: list["DistortionCoeffStat"] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Paper Evidence 단계 추가 필드 (모두 additive - 위 필드들은 값/의미
    # 그대로 유지, recommender.py는 여전히 overall_stability만 본다).
    #
    # 논문이 실제로 쓰는 Stability 정의: CV_p = sigma_p / mu_p, p in
    # {fx, fy, cx, cy}; Stability = 100 * (1 - mean(CV_p)). distortion
    # coefficient는 이 정의에 들어가지 않는다 - 특히 k3/k4처럼 reference가
    # 0에 가까운 계수는 std가 작아도 상대 CV가 폭발해 overall_stability를
    # 크게 끌어내릴 수 있다("Fisheye Stability 62%"가 fx/fy/cx/cy 자체의
    # 불안정 때문인지, 이 near-zero distortion 분모 문제 때문인지는
    # paper_intrinsic_stability vs overall_stability를 나란히 봐야 구분된다).
    # ------------------------------------------------------------------
    paper_intrinsic_stability: Optional[float] = None  # fx/fy/cx/cy stability 4개만의 평균
    # distortion_stats의 stability_score만 모은 요약(참고용 diagnostic -
    # 이 값 자체도 recommender에는 쓰이지 않는다. 논문 본문에 쓸 metric이
    # 아니라 "distortion 쪽이 얼마나 불안정한지" 확인용).
    distortion_stability_summary: Optional[float] = None
    # fx/fy/cx/cy 각각의 reference(전체 데이터 fit 값)와 순수 CV(=std/|mean|,
    # reference와 섞지 않음 - 논문 수식 그대로) - fx_stability 등 기존
    # 필드의 계산식(reference와 blending된 scale)은 바꾸지 않고, 이 값들은
    # 진단/투명성 목적의 추가 정보다.
    fx_reference: Optional[float] = None
    fy_reference: Optional[float] = None
    cx_reference: Optional[float] = None
    cy_reference: Optional[float] = None
    fx_relative_cv: Optional[float] = None
    fy_relative_cv: Optional[float] = None
    cx_relative_cv: Optional[float] = None
    cy_relative_cv: Optional[float] = None
    # "어느 파라미터 때문에 stability가 낮은가"를 UI가 바로 보여줄 수 있게 -
    # fx/fy/cx/cy + 모든 distortion coefficient label 중 stability_score가
    # 가장 낮은 것의 이름(예: "k4"). 동점/데이터 없음이면 None.
    lowest_stability_parameter: Optional[str] = None
    lowest_stability_value: Optional[float] = None

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
    raw_condition_number: Optional[float] = None
    normalized_condition_number: Optional[float] = None
    normalization_scales: dict[str, float] = field(default_factory=dict)
    raw_singular_values: list[float] = field(default_factory=list)
    min_singular_value: Optional[float] = None
    max_singular_value: Optional[float] = None
    max_abs_correlation: Optional[float] = None
    correlation_matrix: list[list[float]] = field(default_factory=list)
    # 새로 계산되는 리포트는 항상 이 값("covariance_from_normalized_jacobian",
    # normalized Jacobian의 SVD에서 유도한 covariance-like matrix)을 쓴다.
    # 구버전 .ccproj에는 이 필드 자체가 없었는데, 그 시절 correlation_matrix는
    # np.corrcoef(raw Jacobian, rowvar=False) 기반이었다("legacy_jacobian_column_correlation").
    # 필드가 없는 legacy 프로젝트를 로드할 때 이 기본값으로 잘못 라벨링하면 안 되므로,
    # 복원 로직(calibration/project_codecs/intrinsic.py:_observability_report_from_dict)은
    # 이 dataclass 기본값을 쓰지 않고 명시적으로 legacy 라벨을 채운다.
    correlation_method: str = "covariance_from_normalized_jacobian"
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
    # Fisheye의 robust fallback(_robust_fisheye_calibrate)처럼 fit에 실제로
    # 쓰인 프레임 집합이 입력보다 작을 수 있는 모델을 위한 구조화된 회계.
    # Pinhole/Brown-Conrady/Rational은 프레임을 제외하지 않으므로
    # input_frame_count == used_frame_count, excluded_frame_ids == [].
    # 이전에는 warning_message 자유 텍스트 하나뿐이라 recommender.py/UI/export가
    # "이 모델이 몇 장으로 학습됐는지"를 기계적으로 읽을 수 없었다 - 모델 간
    # Train RMS 비교가 서로 다른 데이터셋 크기 위에서 이뤄질 수 있다는 사실을
    # 명시하기 위해 추가.
    input_frame_count: int = 0
    used_frame_count: int = 0
    excluded_frame_ids: list[str] = field(default_factory=list)
    exclusion_reason: Optional[str] = None


@dataclass
class OptimizerSettings:
    """User-facing optimizer controls.  Defaults are deliberately conservative."""
    backend: str = "scipy"
    multi_start: bool = True
    num_starts: int = 4
    robust_loss: str = "huber"
    huber_delta: float = 1.0
    staged: bool = True
    max_iterations_per_stage: int = 100
    final_joint_iterations: int = 200
    optimize_principal_point: bool = True
    optimize_focal_length: bool = True
    optimize_distortion: bool = True
    optimize_extrinsics: bool = True


@dataclass
class OptimizerMetricSet:
    train_rms: Optional[float] = None
    test_rms: Optional[float] = None
    test_p95: Optional[float] = None
    test_p99: Optional[float] = None
    edge_rms: Optional[float] = None
    stability: Optional[float] = None
    observability: Optional[float] = None


@dataclass
class OptimizerStartResult:
    name: str = ""
    converged: bool = False
    objective: Optional[float] = None
    message: str = ""
    selected: bool = False


@dataclass
class OptimizerStageResult:
    name: str = ""
    active_parameters: list[str] = field(default_factory=list)
    iterations: int = 0
    rms_before: Optional[float] = None
    rms_after: Optional[float] = None
    termination: str = ""
    success: bool = False


@dataclass
class OptimizerResult:
    model_name: CameraModelType
    settings: OptimizerSettings = field(default_factory=OptimizerSettings)
    train_frame_ids: list[str] = field(default_factory=list)
    holdout_frame_ids: list[str] = field(default_factory=list)
    original_calibration: Optional[CalibrationResult] = None
    optimized_calibration: Optional[CalibrationResult] = None
    # Snapshot of the UI/deployment result replaced by Apply.  This can be a
    # full-data OpenCV fit, whereas original_calibration above is deliberately
    # the leak-safe train-only Before fit.
    pre_apply_calibration: Optional[CalibrationResult] = None
    before_metrics: OptimizerMetricSet = field(default_factory=OptimizerMetricSet)
    after_metrics: OptimizerMetricSet = field(default_factory=OptimizerMetricSet)
    starts: list[OptimizerStartResult] = field(default_factory=list)
    stages: list[OptimizerStageResult] = field(default_factory=list)
    pipeline_status: dict[str, str] = field(default_factory=dict)
    recommendation: str = "Neutral / Marginal"
    reasons: list[str] = field(default_factory=list)
    applied: bool = False
    success: bool = False
    cancelled: bool = False
    error_message: Optional[str] = None
    input_training_frame_ids: list[str] = field(default_factory=list)
    used_training_frame_ids: list[str] = field(default_factory=list)
    excluded_training_frame_ids: list[str] = field(default_factory=list)
    exclusion_reason: Optional[str] = None


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


@dataclass
class HoldoutEvidenceGate:
    """Phase B-6 안정화 - Hold-out Evidence Gate.

    Hold-out RMS(test_rms) 숫자 자체는 그대로 두고, 그 숫자를 뒷받침할
    근거(test 프레임 수/코너 수/공간 coverage/자세 다양성)가 충분한지를
    별도로 판단한다. "낮은 RMS를 틀렸다고 바꾸는" 것이 아니라, 근거가
    부족하면 `status="insufficient_evidence"`로 명시적으로 구분해
    "VALID(낮은 RMS + 충분한 근거)"와 "우연히 낮게 나온 숫자"를 UI/보고서가
    서로 다르게 취급할 수 있게 한다. 계산은 `calibration/holdout_evidence.py`
    에 있다(순환 import를 피하기 위해 dataclass만 여기 둔다 - ResidualStats/
    residual_stats.py와 동일한 패턴).
    """
    status: str = "not_evaluated"  # "sufficient" | "insufficient_evidence" | "not_evaluated"
    test_frame_count: int = 0
    test_corner_count: int = 0
    test_coverage_pct: float = 0.0
    test_pose_diversity: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def validity(self):
        """Phase D-3 - 공통 `ResultValidity`로 읽는 read-only 뷰. `status`
        문자열 자체(직렬화 대상)는 바뀌지 않는다 - 지연 import로 순환
        의존을 피한다(calibration.result_validity는 calibration.types를
        참조하지 않으므로 실제로는 순환이 아니지만, 이 파일 최상단에
        서브모듈을 추가로 import하지 않는 기존 스타일을 유지한다)."""
        from calibration.result_validity import validity_from_legacy_status
        return validity_from_legacy_status(self.status)


# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Hold-out + Straightness 를 함께 담는다."""
    train_frame_ids: list[str] = field(default_factory=list)
    test_frame_ids: list[str] = field(default_factory=list)
    train_rms: Optional[float] = None
    test_rms: Optional[float] = None            # test intrinsic 재최적화 금지 원칙 준수
    # Pooled(전체 test corner point 기준, sqrt(mean(point_error**2))) 정의로
    # 통일했다 - train_rms(=CalibrationResult.rms_error, OpenCV
    # calibrateCamera*의 반환값도 pooled)와 동일한 observation weighting이어야
    # 공정하게 비교할 수 있다(ChArUco partial detection으로 frame마다 코너
    # 수가 크게 다르면, frame-equal 평균은 코너가 적은 frame에 과도한 가중치를
    # 준다). test_residual_stats.rmse와 항상 같은 값이다(그 필드에서 그대로
    # 가져옴) - 별도 필드로 존재하는 이유는 하위 호환(export/UI가 test_rms
    # 이름으로 이미 참조 중)과, residual_stats가 없는 예외 경로에서도 이
    # 필드만은 채워져 있길 기대하는 기존 코드가 있어서다.
    # 이전 정의(frame-equal, sqrt(mean(per_frame_rms**2)))는
    # test_macro_rms로 보존했다 - 정보 가치가 있어 버리지 않는다.
    test_macro_rms: Optional[float] = None
    edge_rms: Optional[float] = None
    straightness_residual: Optional[float] = None  # V2, 없으면 None
    straightness_source: Optional[str] = None  # "test" | "train_fallback" | None
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
    # frame_id -> 사람이 읽을 수 있는 실패 사유 요약(예: "fisheye.solvePnP:
    # returned False", "undistort+ITERATIVE: returned False", "IPPE: no
    # positive-depth solution"). Fisheye Hold-out test pose 추정이 solvePnP
    # 실패 한 번으로 프레임 전체를 버리던 문제를 디버깅하기 위해 추가 -
    # solve_pnp_for_model_robust()의 fallback 단계별 사유를 그대로 문자열로만
    # 담는다(원본 예외 객체/스택은 저장하지 않아 결과가 무거워지지 않는다).
    failed_test_frame_reasons: dict[str, str] = field(default_factory=dict)
    # Phase A-8 안정화 - 이 모델의 hold-out test 평가에서 나온 frame별 RMS
    # (frame_id -> rms_px). 이전에는 원본 Dataset의 Frame.reprojection_error를
    # 직접 덮어썼는데, Pinhole/Brown-Conrady/Rational/Fisheye를 순차적으로
    # hold-out 검증하면 같은 Frame 객체가 매번 다시 mutate되어 마지막에
    # 평가된 모델의 값만 남았다(어느 모델의 값인지 알 수 없는 채로). 이제는
    # model별로 이미 분리되어 있는 이 ValidationResult 안에만 값을 보관한다 -
    # 원본 Dataset은 건드리지 않는다.
    per_frame_error: dict[str, float] = field(default_factory=dict)
    # Phase B-6 안정화 - Hold-out RMS 값 자체와는 독립적으로, 그 값을
    # 신뢰할 근거가 충분한지 별도로 기록한다. None이면 아직 평가되지
    # 않은 경우(예: test 프레임이 아예 없어 애초에 hold-out 자체를
    # 수행하지 못한 경우)다.
    evidence_gate: Optional[HoldoutEvidenceGate] = None


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
    test_rms: Optional[float] = None  # pooled 정의 - ValidationResult.test_rms와 동일한 원칙
    test_macro_rms: Optional[float] = None  # 이전 frame-equal 정의 (ValidationResult 참고)
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
    # Paper Evidence 단계 추가 - 이 KFoldResult를 만든 split_k_folds() seed.
    # fold-level raw export(model/repeat_index/fold_index/seed row)에
    # 필요하고, "같은 seed면 항상 같은 fold partition"이라는 재현성 주장을
    # 검증하려면 어떤 seed였는지가 결과에 같이 남아 있어야 한다.
    seed: Optional[int] = None
    fold_validation_results: list[ValidationResult] = field(default_factory=list)
    mean_test_rms: Optional[float] = None
    std_test_rms: Optional[float] = None
    min_test_rms: Optional[float] = None
    max_test_rms: Optional[float] = None
    mean_test_p95: Optional[float] = None
    std_test_p95: Optional[float] = None
    n_successful_folds: int = 0
    # 논문용 Multi-Metric 확장 - Test RMS/P95 외에 Hold-out Test Edge RMS와
    # Test Straightness도 fold 간 mean/std/min/max로 aggregate한다. 반드시
    # ValidationResult.edge_rms(= Hold-out **test** edge에서 계산된 값)만
    # 쓴다 - Train regional_error fallback은 여기 절대 섞이지 않는다(그런
    # fallback 자체가 없다 - edge_rms는 애초에 test 프레임으로만 계산됨).
    mean_edge_rms: Optional[float] = None
    std_edge_rms: Optional[float] = None
    min_edge_rms: Optional[float] = None
    max_edge_rms: Optional[float] = None
    # straightness_source == "test"인 fold만 aggregate에 포함한다.
    # "train_fallback"(test 프레임에서 직선을 못 뽑아 train으로 대체한 값)은
    # 절대 섞지 않는다 - 그런 fold는 diagnostic으로만 남기고 straightness
    # aggregate에서는 missing으로 취급한다(임의의 숫자로 채우거나 최악
    # penalty를 주지 않는다).
    mean_test_straightness: Optional[float] = None
    std_test_straightness: Optional[float] = None
    min_test_straightness: Optional[float] = None
    max_test_straightness: Optional[float] = None
    # straightness aggregate에 실제로 포함된 fold 수(위 4개 값의 표본 크기) -
    # n_successful_folds보다 작을 수 있다(Test Straightness를 계산 못 한 fold가
    # 있으면). 0이면 straightness aggregate 전체가 missing이라는 뜻.
    n_straightness_folds: int = 0


@dataclass
class RepeatedKFoldResult:
    """설계 문서 19번 - K-Fold를 여러 번(예: 5-fold x 5회) 반복해서, fold 나누는
    방식 자체가 우연히 좋거나 나쁘게 뽑히는 효과까지 평균으로 눌러준다.
    """
    k: int = 5
    n_repeats: int = 5
    # Paper Evidence 단계 추가 - repeat r의 seed는 base_seed + r
    # (compute_repeated_kfold와 동일한 공식). fold-level raw export의
    # "seed" 컬럼과 split_manifest가 이 값을 그대로 사용한다.
    base_seed: int = 42
    kfold_results: list[KFoldResult] = field(default_factory=list)
    mean_test_rms: Optional[float] = None
    std_test_rms: Optional[float] = None
    min_test_rms: Optional[float] = None
    max_test_rms: Optional[float] = None
    mean_test_p95: Optional[float] = None
    std_test_p95: Optional[float] = None
    n_successful_runs: int = 0
    # KFoldResult와 같은 이유로 - Test Edge RMS/Test Straightness도
    # k*n_repeats개 fold 전체(성공한 fold만)를 모아 mean/std/min/max로
    # aggregate한다. straightness는 straightness_source == "test"인 fold만
    # 포함(train_fallback은 diagnostic 전용, 절대 섞지 않음).
    mean_edge_rms: Optional[float] = None
    std_edge_rms: Optional[float] = None
    min_edge_rms: Optional[float] = None
    max_edge_rms: Optional[float] = None
    mean_test_straightness: Optional[float] = None
    std_test_straightness: Optional[float] = None
    min_test_straightness: Optional[float] = None
    max_test_straightness: Optional[float] = None
    n_straightness_folds: int = 0
    # 논문용 fold 상태 diagnostic - "25 folds 중 몇 개가 완전히 정상 평가됐는지"를
    # 바로 확인할 수 있게 한다(특히 Fisheye가 일부 test frame에서 pose 추정에
    # 실패하는 경우를 조용히 숨기지 않기 위함).
    #   total_folds = k * n_repeats (설계상 실행되어야 했던 fold 수)
    #   fully_successful_folds = 성공 + 실패한 test frame이 0개인 fold
    #   partial_success_folds = 성공했지만 일부 test frame의 pose 추정은 실패한 fold
    #   failed_folds = validate_holdout 자체가 실패했거나(success=False),
    #       train 프레임 부족 등으로 애초에 fold가 실행조차 안 된 경우까지 포함
    #       (fully_successful_folds + partial_success_folds + failed_folds == total_folds)
    total_folds: int = 0
    fully_successful_folds: int = 0
    partial_success_folds: int = 0
    failed_folds: int = 0


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
    evidence_components: list[str] = field(default_factory=list)
    is_recommended: bool = False
    is_selection_eligible: bool = True
    selection_status: str = "ELIGIBLE"
    selection_ineligibility_reason: Optional[str] = None
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
    optimizer_results: dict[CameraModelType, OptimizerResult] = field(default_factory=dict)
    # Windshield Refraction Calibration (calibration/windshield/) - Object-Releasing과
    # 같은 패턴으로 별도 필드에 담는다. calibration.types는 calibration.windshield를
    # import하지 않으므로(순환 참조 방지) 타입은 문자열 forward-reference로 둔다 -
    # from __future__ import annotations(이 파일 상단)로 dataclass 정의 자체는
    # 문제 없이 동작하고, 실제 클래스는 project_io.py에서만 import한다.
    windshield_config: Optional["WindshieldConfig"] = None
    windshield_dataset: Optional[Dataset] = None
    windshield_results: dict["WindshieldResultKey", "WindshieldCalibrationResult"] = field(default_factory=dict)
    reflection_results: dict[str, "ReflectionDatasetResult"] = field(default_factory=dict)
    # Ghost / Double Image(STEP 8) - Reflection과 완전히 별도 필드다(사용자
    # 스펙 "ghost_results, ghost_models는 windshield_results/
    # reflection_results와 섞이지 않는다"). 같은 forward-reference 패턴을
    # 그대로 따른다(circular import 방지).
    ghost_results: dict[str, "GhostDatasetResult"] = field(default_factory=dict)
    ghost_models: dict[str, "GhostField"] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    export_paths: dict[ExportFormat, str] = field(default_factory=dict)

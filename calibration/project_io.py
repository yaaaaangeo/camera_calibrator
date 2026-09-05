"""
camera_calibrator.calibration.project_io
=============================================

설계 문서 16번 - `.ccproj` 프로젝트 저장/불러오기.

pickle 대신 JSON을 쓴다 - `.ccproj` 파일은 나중에 동료와 공유하거나
버전관리에 올릴 수도 있는 파일인데, pickle.load()는 신뢰할 수 없는 파일을
열 때 임의 코드 실행 위험이 있다. JSON은 느리고 코드가 더 필요하지만 안전하고,
이 프로젝트가 이미 YAML/JSON을 export 포맷으로 쓰는 철학과도 맞는다.

이미지 파일 자체는 프로젝트 파일 안에 복사해 넣지 않는다 (설계 문서 9번의
"파일을 삭제/복제하지 않고 메타데이터만 다룬다" 원칙과 같은 이유) - 경로만
저장하고, 불러올 때 존재 여부를 확인해 없어진 파일은 경고로만 알려준다
(크래시하지 않음).

직렬화 전략:
    save: dataclasses.asdict()로 중첩 dataclass를 재귀적으로 dict화한 뒤,
          _json_safe()로 numpy 배열/Enum/datetime/tuple을 JSON이 이해하는
          타입으로 변환.
    load: 일반 dict에서 다시 정확한 dataclass 인스턴스로 복원 - 이 방향은
          타입 정보가 없어져서 필드별로 명시적으로 재구성해야 한다
          (필드마다 numpy dtype, Enum 클래스가 다르므로 자동화하지 않음).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from calibration.json_utils import json_safe
from calibration.types import (
    CalibrationProject,
    CalibrationResult,
    CalibrationConfidenceReport,
    AprilGridVariant,
    CameraConfig,
    CameraModelType,
    CalibrationMethod,
    CaptureRecommendation,
    CircleGridType,
    CoverageCell,
    CornerOutlierResult,
    CrossDatasetValidationResult,
    Dataset,
    DatasetQualityScore,
    DetectionResult,
    DiagnosisReport,
    DiagnosisSeverity,
    DiversityScores,
    ExportFormat,
    FinalResult,
    FailurePattern,
    Frame,
    FrameQuality,
    FrameStatus,
    ImageInfo,
    ModelScore,
    ObjectReleasingValidationResult,
    ObservabilityReport,
    OutlierResult,
    ParameterUncertainty,
    ParameterCorrelation,
    PatternConfig,
    PatternType,
    QualityGrade,
    RadialBin,
    RadialErrorProfile,
    RegionalError,
    ResidualStats,
    SceneQualityAnalysis,
    SceneQualityEntry,
    SpatialErrorCell,
    SpatialErrorMap,
    StandardVsObjectReleasingComparison,
    SubsetCalibrationResult,
    StraightnessBreakdown,
    UndistortionQualityReport,
    ValidationResult,
)
from calibration.windshield.base import (
    WindshieldCalibrationResult,
    WindshieldConfig,
    WindshieldModelType,
    windshield_result_key_from_storage,
    windshield_result_key_to_storage,
)

PROJECT_FORMAT_VERSION = 2
PROJECT_EXTENSION = ".ccproj"

logger = logging.getLogger(__name__)


def project_to_dict(project: CalibrationProject) -> dict:
    raw = dataclasses.asdict(project)
    if raw.get("windshield_results"):
        raw["windshield_results"] = {
            windshield_result_key_to_storage(k): v
            for k, v in raw["windshield_results"].items()
        }
    safe = json_safe(raw)
    return {"format_version": PROJECT_FORMAT_VERSION, "project": safe}


def save_project(project: CalibrationProject, path: str) -> str:
    project.updated_at = datetime.now()
    payload = project_to_dict(project)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _arr(d, dtype) -> np.ndarray | None:
    """_json_safe가 만든 {"__ndarray__": True, "data": [...]} 구조 -> np.ndarray.
    구버전 파일 호환을 위해 그냥 리스트로 저장된 경우도 받아준다.
    """
    if d is None:
        return None
    if isinstance(d, dict) and d.get("__ndarray__"):
        return np.array(d["data"], dtype=dtype)
    return np.array(d, dtype=dtype)


def _dt(d) -> datetime:
    if d is None:
        return datetime.now()
    if isinstance(d, dict) and d.get("__datetime__"):
        return datetime.fromisoformat(d["iso"])
    return datetime.fromisoformat(d)


def _pattern_config_from_dict(d: dict) -> PatternConfig:
    return PatternConfig(
        type=PatternType(d["type"]),
        squares_x=d["squares_x"],
        squares_y=d["squares_y"],
        square_size=d["square_size"],
        marker_size=d.get("marker_size"),
        dictionary=d.get("dictionary"),
        circle_grid_type=CircleGridType(d.get("circle_grid_type", "symmetric")),
        aprilgrid_variant=AprilGridVariant(d.get("aprilgrid_variant", "opencv_apriltag3")),
    )


def _camera_config_from_dict(d: dict) -> CameraConfig:
    model_value = d.get("model")
    return CameraConfig(
        width=d["width"],
        height=d["height"],
        fps=d.get("fps"),
        model=CameraModelType(model_value) if model_value else None,
        sensor_name=d.get("sensor_name"),
        hfov_deg=d.get("hfov_deg"),
        vfov_deg=d.get("vfov_deg"),
    )


def _image_info_from_dict(d: dict) -> ImageInfo:
    return ImageInfo(
        image_id=d["image_id"], path=d["path"], width=d["width"], height=d["height"],
        sharpness=d.get("sharpness"), brightness=d.get("brightness"), exposure=d.get("exposure"),
        contrast=d.get("contrast"), saturation=d.get("saturation"),
        motion_blur_score=d.get("motion_blur_score"), phash=d.get("phash"),
    )


def _detection_result_from_dict(d) -> DetectionResult | None:
    if d is None:
        return None
    return DetectionResult(
        image_id=d["image_id"], success=d["success"],
        corners=_arr(d.get("corners"), np.float32),
        object_points=_arr(d.get("object_points"), np.float32),
        ids=_arr(d.get("ids"), np.int32),
        num_corners=d.get("num_corners", 0),
        board_area_ratio=d.get("board_area_ratio"),
        board_center_px=tuple(d["board_center_px"]) if d.get("board_center_px") else None,
        board_tilt_deg=d.get("board_tilt_deg"),
        failure_reason=d.get("failure_reason"),
        corner_confidence=d.get("corner_confidence"),
        min_edge_margin_px=d.get("min_edge_margin_px"),
        likely_cut_off=d.get("likely_cut_off"),
        excluded_corner_indices=d.get("excluded_corner_indices", []),
    )


def _frame_quality_from_dict(d) -> FrameQuality | None:
    if d is None:
        return None
    return FrameQuality(
        detection_score=d.get("detection_score", 0.0),
        geometric_score=d.get("geometric_score", 0.0),
        overall_score=d.get("overall_score", 0.0),
        grade=QualityGrade(d.get("grade", "poor")),
        blur_score=d.get("blur_score"),
        exposure_score=d.get("exposure_score"),
        corner_quality_score=d.get("corner_quality_score"),
        board_area_score=d.get("board_area_score"),
        edge_coverage_score=d.get("edge_coverage_score"),
        pose_diversity_score=d.get("pose_diversity_score"),
    )


def _frame_from_dict(d: dict) -> Frame:
    return Frame(
        image_info=_image_info_from_dict(d["image_info"]),
        detection=_detection_result_from_dict(d.get("detection")),
        quality=_frame_quality_from_dict(d.get("quality")),
        status=FrameStatus(d.get("status", "pending")),
        disabled_reason=d.get("disabled_reason"),
        reprojection_error=d.get("reprojection_error"),
    )


def _coverage_cell_from_dict(d: dict) -> CoverageCell:
    return CoverageCell(row=d["row"], col=d["col"], corner_count=d.get("corner_count", 0),
                         coverage_score=d.get("coverage_score", 0.0))


def _diversity_scores_from_dict(d) -> DiversityScores | None:
    if d is None:
        return None
    return DiversityScores(
        position_coverage=d.get("position_coverage", 0.0),
        distance_diversity=d.get("distance_diversity", 0.0),
        rotation_diversity=d.get("rotation_diversity", 0.0),
        edge_coverage=d.get("edge_coverage", 0.0),
    )


def _dataset_quality_score_from_dict(d) -> "DatasetQualityScore | None":
    if d is None:
        return None
    return DatasetQualityScore(
        avg_frame_quality=d.get("avg_frame_quality", 0.0),
        detection_success_rate=d.get("detection_success_rate", 0.0),
        coverage_score=d.get("coverage_score", 0.0),
        diversity_score=d.get("diversity_score", 0.0),
        duplicate_penalty=d.get("duplicate_penalty", 0.0),
        overall=d.get("overall", 0.0),
        grade=QualityGrade(d.get("grade", "poor")),
    )


def _dataset_from_dict(d: dict) -> Dataset:
    return Dataset(
        frames=[_frame_from_dict(f) for f in d.get("frames", [])],
        coverage_grid=[_coverage_cell_from_dict(c) for c in d.get("coverage_grid", [])],
        diversity=_diversity_scores_from_dict(d.get("diversity")),
        quality_score=_dataset_quality_score_from_dict(d.get("quality_score")),
    )


def _param_uncertainty_from_dict(d) -> ParameterUncertainty | None:
    if d is None:
        return None
    return ParameterUncertainty(
        fx_std=d.get("fx_std"), fy_std=d.get("fy_std"),
        cx_std=d.get("cx_std"), cy_std=d.get("cy_std"),
        method=d.get("method", "covariance"),
        n_bootstrap_success=d.get("n_bootstrap_success"),
        fx_ci_low=d.get("fx_ci_low"), fx_ci_high=d.get("fx_ci_high"),
        fy_ci_low=d.get("fy_ci_low"), fy_ci_high=d.get("fy_ci_high"),
        cx_ci_low=d.get("cx_ci_low"), cx_ci_high=d.get("cx_ci_high"),
        cy_ci_low=d.get("cy_ci_low"), cy_ci_high=d.get("cy_ci_high"),
    )


def _regional_error_from_dict(d) -> RegionalError | None:
    if d is None:
        return None
    return RegionalError(
        center=d.get("center"), left=d.get("left"), right=d.get("right"),
        top=d.get("top"), bottom=d.get("bottom"), corner=d.get("corner"),
    )


def _radial_profile_from_dict(d) -> RadialErrorProfile | None:
    if d is None:
        return None
    bins = [
        RadialBin(
            radius_min=b["radius_min"], radius_max=b["radius_max"],
            mean_error=b.get("mean_error"), num_points=b.get("num_points", 0),
            median_error=b.get("median_error"), rms_error=b.get("rms_error"),
            p95_error=b.get("p95_error"), max_error=b.get("max_error"), label=b.get("label"),
        )
        for b in d.get("bins", [])
    ]
    return RadialErrorProfile(bins=bins, max_radius=d.get("max_radius", 0.0))


def _residual_stats_from_dict(d) -> ResidualStats | None:
    if d is None:
        return None
    return ResidualStats(
        n=d.get("n", 0),
        rmse=d.get("rmse"), mae=d.get("mae"), median=d.get("median"), std=d.get("std"),
        min=d.get("min"), q1=d.get("q1"), q3=d.get("q3"),
        p90=d.get("p90"), p95=d.get("p95"), p99=d.get("p99"), max=d.get("max"),
        outlier_count=d.get("outlier_count", 0),
        histogram_bin_edges=d.get("histogram_bin_edges", []),
        histogram_counts=d.get("histogram_counts", []),
        sample_residuals=d.get("sample_residuals", []),
    )


def _spatial_error_map_from_dict(d) -> SpatialErrorMap | None:
    if d is None:
        return None
    cells = [
        SpatialErrorCell(
            row=c["row"], col=c["col"], num_points=c.get("num_points", 0),
            rms=c.get("rms"), p95=c.get("p95"),
            mean_dx=c.get("mean_dx"), mean_dy=c.get("mean_dy"),
            direction_deg=c.get("direction_deg"),
        )
        for c in d.get("cells", [])
    ]
    return SpatialErrorMap(cells=cells, rows=d.get("rows", 4), cols=d.get("cols", 4))


def _observability_report_from_dict(d) -> ObservabilityReport | None:
    if d is None:
        return None
    return ObservabilityReport(
        parameter_labels=d.get("parameter_labels", []),
        jacobian_rows=d.get("jacobian_rows", 0),
        jacobian_cols=d.get("jacobian_cols", 0),
        num_points=d.get("num_points", 0),
        singular_values=d.get("singular_values", []),
        rank=d.get("rank", 0),
        condition_number=d.get("condition_number"),
        min_singular_value=d.get("min_singular_value"),
        max_singular_value=d.get("max_singular_value"),
        max_abs_correlation=d.get("max_abs_correlation"),
        correlation_matrix=d.get("correlation_matrix", []),
        observability_score=d.get("observability_score"),
        observability_grade=d.get("observability_grade"),
        top_correlations=[
            ParameterCorrelation(
                param_a=c["param_a"],
                param_b=c["param_b"],
                correlation=c["correlation"],
            )
            for c in d.get("top_correlations", [])
        ],
        warnings=d.get("warnings", []),
    )


def _undistortion_quality_from_dict(d) -> UndistortionQualityReport | None:
    if d is None:
        return None
    return UndistortionQualityReport(
        image_width=d.get("image_width", 0),
        image_height=d.get("image_height", 0),
        valid_pixel_ratio=d.get("valid_pixel_ratio", 0.0),
        black_border_ratio=d.get("black_border_ratio", 0.0),
        roi_loss_ratio=d.get("roi_loss_ratio", 0.0),
        valid_roi=tuple(d.get("valid_roi", (0, 0, 0, 0))),
        undistorted_black_pixel_ratio=d.get("undistorted_black_pixel_ratio"),
        sample_frame_id=d.get("sample_frame_id"),
        quality_score=d.get("quality_score", 0.0),
        quality_grade=QualityGrade(d.get("quality_grade", "warning")),
        warnings=d.get("warnings", []),
    )


def _calibration_result_from_dict(d: dict) -> CalibrationResult:
    model_value = d["model_name"]
    return CalibrationResult(
        model_name=CameraModelType(model_value),
        camera_matrix=_arr(d.get("camera_matrix"), np.float64),
        distortion=_arr(d.get("distortion"), np.float64),
        rvecs=[_arr(r, np.float64) for r in d.get("rvecs", [])],
        tvecs=[_arr(t, np.float64) for t in d.get("tvecs", [])],
        rms_error=d.get("rms_error"),
        per_frame_error=d.get("per_frame_error", {}),
        regional_error=_regional_error_from_dict(d.get("regional_error")),
        radial_profile=_radial_profile_from_dict(d.get("radial_profile")),
        radial_bands=_radial_profile_from_dict(d.get("radial_bands")),
        spatial_error_map=_spatial_error_map_from_dict(d.get("spatial_error_map")),
        param_uncertainty=_param_uncertainty_from_dict(d.get("param_uncertainty")),
        param_uncertainty_bootstrap=_param_uncertainty_from_dict(d.get("param_uncertainty_bootstrap")),
        residual_stats=_residual_stats_from_dict(d.get("residual_stats")),
        observability=_observability_report_from_dict(d.get("observability")),
        undistortion_quality=_undistortion_quality_from_dict(d.get("undistortion_quality")),
        calibration_method=CalibrationMethod(d.get("calibration_method", "standard")),
        refined_object_points=_arr(d.get("refined_object_points"), np.float32),
        target_geometry_refinement=d.get("target_geometry_refinement"),
        object_releasing_diagnostics=d.get("object_releasing_diagnostics", []),
        success=d.get("success", False),
        error_message=d.get("error_message"),
        warning_message=d.get("warning_message"),
    )


def _outlier_result_from_dict(d) -> OutlierResult | None:
    if d is None:
        return None
    return OutlierResult(
        threshold_used=d["threshold_used"],
        removed_frame_ids=d.get("removed_frame_ids", []),
        rms_before=d.get("rms_before"), rms_after=d.get("rms_after"),
        iterations=d.get("iterations", 0), max_iterations=d.get("max_iterations", 3),
        p95_before=d.get("p95_before"), p95_after=d.get("p95_after"),
        camera_matrix_before=_arr(d.get("camera_matrix_before"), np.float64),
        camera_matrix_after=_arr(d.get("camera_matrix_after"), np.float64),
        distortion_before=_arr(d.get("distortion_before"), np.float64),
        distortion_after=_arr(d.get("distortion_after"), np.float64),
    )


def _straightness_breakdown_from_dict(d) -> StraightnessBreakdown | None:
    if d is None:
        return None
    return StraightnessBreakdown(
        horizontal_error=d.get("horizontal_error"), vertical_error=d.get("vertical_error"),
        diagonal_error=d.get("diagonal_error"), center_line_error=d.get("center_line_error"),
        edge_line_error=d.get("edge_line_error"), corner_line_error=d.get("corner_line_error"),
        overall_error=d.get("overall_error"), num_lines=d.get("num_lines", 0),
    )


def _validation_result_from_dict(d: dict) -> ValidationResult:
    return ValidationResult(
        train_frame_ids=d.get("train_frame_ids", []),
        test_frame_ids=d.get("test_frame_ids", []),
        train_rms=d.get("train_rms"), test_rms=d.get("test_rms"), edge_rms=d.get("edge_rms"),
        straightness_residual=d.get("straightness_residual"),
        straightness_breakdown=_straightness_breakdown_from_dict(d.get("straightness_breakdown")),
        train_residual_stats=_residual_stats_from_dict(d.get("train_residual_stats")),
        test_residual_stats=_residual_stats_from_dict(d.get("test_residual_stats")),
        success=d.get("success", True), error_message=d.get("error_message"),
        failed_test_frame_ids=d.get("failed_test_frame_ids", []),
    )


def _object_releasing_validation_result_from_dict(d) -> ObjectReleasingValidationResult | None:
    if d is None:
        return None
    return ObjectReleasingValidationResult(
        success=d.get("success", True),
        error_message=d.get("error_message"),
        train_frame_ids=d.get("train_frame_ids", []),
        test_frame_ids=d.get("test_frame_ids", []),
        excluded_frame_ids=d.get("excluded_frame_ids", []),
        excluded_reasons=d.get("excluded_reasons", {}),
        failed_test_frame_ids=d.get("failed_test_frame_ids", []),
        failed_test_reasons=d.get("failed_test_reasons", {}),
        train_rms=d.get("train_rms"),
        test_rms=d.get("test_rms"),
        test_residual_stats=_residual_stats_from_dict(d.get("test_residual_stats")),
        target_geometry_refinement=d.get("target_geometry_refinement"),
    )


def _standard_vs_object_releasing_comparison_from_dict(d) -> StandardVsObjectReleasingComparison | None:
    if d is None:
        return None
    return StandardVsObjectReleasingComparison(
        success=d.get("success", True),
        error_message=d.get("error_message"),
        eligible_frame_ids=d.get("eligible_frame_ids", []),
        train_frame_ids=d.get("train_frame_ids", []),
        test_frame_ids=d.get("test_frame_ids", []),
        standard_result=(
            _calibration_result_from_dict(d["standard_result"]) if d.get("standard_result") else None
        ),
        standard_validation=(
            _validation_result_from_dict(d["standard_validation"]) if d.get("standard_validation") else None
        ),
        object_releasing_result=(
            _calibration_result_from_dict(d["object_releasing_result"])
            if d.get("object_releasing_result") else None
        ),
        object_releasing_validation=_object_releasing_validation_result_from_dict(
            d.get("object_releasing_validation")
        ),
        intrinsics_delta=d.get("intrinsics_delta", {}),
        warnings=d.get("warnings", []),
    )


def _cross_dataset_result_from_dict(d: dict) -> CrossDatasetValidationResult:
    return CrossDatasetValidationResult(
        source_dataset_id=d.get("source_dataset_id", "A"),
        target_dataset_id=d.get("target_dataset_id", "B"),
        model_name=CameraModelType(d["model_name"]),
        train_rms=d.get("train_rms"),
        test_rms=d.get("test_rms"),
        test_p95=d.get("test_p95"),
        edge_rms=d.get("edge_rms"),
        straightness_residual=d.get("straightness_residual"),
        generalization_gap=d.get("generalization_gap"),
        num_test_frames=d.get("num_test_frames", 0),
        failed_test_frame_ids=d.get("failed_test_frame_ids", []),
        success=d.get("success", True),
        error_message=d.get("error_message"),
    )


def _scene_quality_analysis_from_dict(d) -> SceneQualityAnalysis | None:
    if d is None:
        return None
    return SceneQualityAnalysis(
        model_name=CameraModelType(d["model_name"]),
        scenes=[SceneQualityEntry(**scene) for scene in d.get("scenes", [])],
    )


def _subset_calibration_result_from_dict(d) -> SubsetCalibrationResult | None:
    if d is None:
        return None
    return SubsetCalibrationResult(
        model_name=CameraModelType(d["model_name"]),
        selected_frame_ids=d.get("selected_frame_ids", []),
        calibration_result=(
            _calibration_result_from_dict(d["calibration_result"])
            if d.get("calibration_result") else None
        ),
        validation_result=(
            _validation_result_from_dict(d["validation_result"])
            if d.get("validation_result") else None
        ),
        original_validation_result=(
            _validation_result_from_dict(d["original_validation_result"])
            if d.get("original_validation_result") else None
        ),
        coverage_grid=[_coverage_cell_from_dict(c) for c in d.get("coverage_grid", [])],
        diversity=_diversity_scores_from_dict(d.get("diversity")),
        coverage_percentage=d.get("coverage_percentage", 0.0),
        original_coverage_percentage=d.get("original_coverage_percentage", 0.0),
        original_diversity=_diversity_scores_from_dict(d.get("original_diversity")),
        warnings=d.get("warnings", []),
    )


def _model_score_from_dict(d: dict) -> ModelScore:
    return ModelScore(
        model_name=CameraModelType(d["model_name"]), score=d["score"],
        components=d.get("components", {}), is_recommended=d.get("is_recommended", False),
        parameter_count=d.get("parameter_count", 0),
        residual_sum_squares=d.get("residual_sum_squares"),
        num_observations=d.get("num_observations", 0),
        aic=d.get("aic"),
        bic=d.get("bic"),
        selection_confidence=d.get("selection_confidence"),
        selection_confidence_level=d.get("selection_confidence_level"),
        selection_confidence_reason=d.get("selection_confidence_reason"),
        selection_reasons=d.get("selection_reasons", []),
    )


def _diagnosis_report_from_dict(d) -> DiagnosisReport | None:
    if d is None:
        return None
    return DiagnosisReport(
        model_name=CameraModelType(d["model_name"]),
        patterns=[
            FailurePattern(
                code=p["code"],
                severity=DiagnosisSeverity(p.get("severity", "warning")),
                title=p["title"],
                evidence=p.get("evidence", []),
                recommendation=p.get("recommendation", ""),
            )
            for p in d.get("patterns", [])
        ],
        capture_recommendations=[
            CaptureRecommendation(
                code=r["code"],
                priority=r.get("priority", "medium"),
                title=r["title"],
                action=r["action"],
                reason=r.get("reason", ""),
            )
            for r in d.get("capture_recommendations", [])
        ],
    )


def _confidence_report_from_dict(d) -> CalibrationConfidenceReport | None:
    if d is None:
        return None
    return CalibrationConfidenceReport(
        score=d.get("score", 0.0),
        level=d.get("level", "LOW"),
        components=d.get("components", {}),
        reasons=d.get("reasons", []),
        warnings=d.get("warnings", []),
    )


def _corner_outlier_result_from_dict(d) -> CornerOutlierResult | None:
    if d is None:
        return None
    return CornerOutlierResult(
        threshold_used=d.get("threshold_used", 0.0),
        removed_corners=d.get("removed_corners", {}),
        rms_before=d.get("rms_before"), rms_after=d.get("rms_after"),
        iterations=d.get("iterations", 0), max_iterations=d.get("max_iterations", 3),
        p95_before=d.get("p95_before"), p95_after=d.get("p95_after"),
        camera_matrix_before=_arr(d.get("camera_matrix_before"), np.float64),
        camera_matrix_after=_arr(d.get("camera_matrix_after"), np.float64),
        distortion_before=_arr(d.get("distortion_before"), np.float64),
        distortion_after=_arr(d.get("distortion_after"), np.float64),
    )


def _final_result_from_dict(d) -> FinalResult | None:
    if d is None:
        return None
    return FinalResult(
        chosen_model=CameraModelType(d["chosen_model"]),
        calibration=_calibration_result_from_dict(d["calibration"]),
        validation=_validation_result_from_dict(d["validation"]) if d.get("validation") else None,
        outlier=_outlier_result_from_dict(d.get("outlier")),
        corner_outlier=_corner_outlier_result_from_dict(d.get("corner_outlier")),
        dataset_coverage_pct=d.get("dataset_coverage_pct"),
        overall_grade=QualityGrade(d.get("overall_grade", "warning")),
        confidence=_confidence_report_from_dict(d.get("confidence")),
        model_scores=[_model_score_from_dict(s) for s in d.get("model_scores", [])],
        diagnosis=_diagnosis_report_from_dict(d.get("diagnosis")),
    )


def _windshield_config_from_dict(d) -> WindshieldConfig | None:
    if d is None:
        return None
    return WindshieldConfig(
        base_model_name=CameraModelType(d["base_model_name"]),
        base_camera_matrix=_arr(d.get("base_camera_matrix"), np.float64),
        base_distortion=_arr(d.get("base_distortion"), np.float64),
        windshield_model=WindshieldModelType(d.get("windshield_model", "baseline")),
        test_ratio=d.get("test_ratio", 0.25),
        split_seed=d.get("split_seed", 42),
        glass_refractive_index=d.get("glass_refractive_index"),
        glass_thickness_m=d.get("glass_thickness_m"),
        windshield_position_hint=d.get("windshield_position_hint"),
    )


def _windshield_calibration_result_from_dict(d) -> WindshieldCalibrationResult | None:
    if d is None:
        return None
    return WindshieldCalibrationResult(
        windshield_model=WindshieldModelType(d["windshield_model"]),
        base_model_name=CameraModelType(d["base_model_name"]),
        base_camera_matrix=_arr(d.get("base_camera_matrix"), np.float64),
        base_distortion=_arr(d.get("base_distortion"), np.float64),
        train_frame_ids=d.get("train_frame_ids", []),
        test_frame_ids=d.get("test_frame_ids", []),
        failed_frame_ids=d.get("failed_frame_ids", []),
        per_frame_error=d.get("per_frame_error", {}),
        residual_stats=_residual_stats_from_dict(d.get("residual_stats")),
        test_residual_stats=_residual_stats_from_dict(d.get("test_residual_stats")),
        regional_error=_regional_error_from_dict(d.get("regional_error")),
        radial_profile=_radial_profile_from_dict(d.get("radial_profile")),
        radial_bands=_radial_profile_from_dict(d.get("radial_bands")),
        spatial_error_map=_spatial_error_map_from_dict(d.get("spatial_error_map")),
        mean_dx=d.get("mean_dx"),
        mean_dy=d.get("mean_dy"),
        test_regional_error=_regional_error_from_dict(d.get("test_regional_error")),
        test_radial_profile=_radial_profile_from_dict(d.get("test_radial_profile")),
        test_radial_bands=_radial_profile_from_dict(d.get("test_radial_bands")),
        test_spatial_error_map=_spatial_error_map_from_dict(d.get("test_spatial_error_map")),
        test_mean_dx=d.get("test_mean_dx"),
        test_mean_dy=d.get("test_mean_dy"),
        ray_angular_error_deg=d.get("ray_angular_error_deg"),
        test_ray_angular_error_deg=d.get("test_ray_angular_error_deg"),
        fitted_params=d.get("fitted_params", {}),
        success=d.get("success", False),
        error_message=d.get("error_message"),
        warning_message=d.get("warning_message"),
    )


def _raw_array_len(d) -> int | None:
    """migrate_v1_to_v2용 - 아직 dataclass로 복원하지 않은 raw JSON 값에서
    distortion 배열 길이만 알고 싶을 때. _arr()과 같은 언랩 규칙(_json_safe가
    만드는 {"__ndarray__": True, "data": [...]} 포맷, 구버전의 순수 list 둘 다)을
    따르되 dtype 변환은 하지 않는다 - 여기서는 길이만 필요하다.
    """
    if d is None:
        return None
    if isinstance(d, dict) and d.get("__ndarray__"):
        d = d.get("data")
    if not isinstance(d, list):
        return None
    try:
        return int(np.asarray(d).reshape(-1).size)
    except (TypeError, ValueError):
        return None


def _migrate_model_name_refs(container, field_name: str, should_rename: bool) -> None:
    """container[field_name]이 문자열 "extended_pinhole"이면(그리고
    should_rename이면) "brown_conrady"로 바꾼다. dict 하나에 대한 최소 단위
    연산 - migrate_v1_to_v2가 여러 중첩 위치(ModelScore/CrossDatasetValidation
    Result/DiagnosisReport 등)에 동일하게 적용하기 위한 헬퍼.
    """
    if not should_rename or container is None:
        return
    if container.get(field_name) == "extended_pinhole":
        container[field_name] = "brown_conrady"


def _apply_model_map_to_field(container, field_name: str, legacy_model_map: dict[str, str]) -> None:
    if not isinstance(container, dict):
        return
    value = container.get(field_name)
    if value in legacy_model_map:
        container[field_name] = legacy_model_map[value]


def _apply_model_map_to_keyed_dict(container: dict, legacy_model_map: dict[str, str]) -> None:
    for old, new in list(legacy_model_map.items()):
        if old == new or old not in container:
            continue
        if new in container:
            logger.warning(
                "Legacy project contains both %s and %s entries; keeping existing %s entry.",
                old, new, new,
            )
            continue
        container[new] = container.pop(old)


def migrate_v1_to_v2(payload: dict) -> dict:
    """v1 -> v2 마이그레이션.

    v1 시절 "extended_pinhole"은 실제로는 두 가지 다른 의미로 쓰였을 수 있다:
      - distortion 5계수(k1,k2,p1,p2,k3) -> 지금의 Brown-Conrady 역할이었음
      - distortion 8계수 이상(k1~k6,p1,p2) -> 지금의 extended_pinhole(Rational)과 동일

    문자열만 보고 바꾸지 않는다 - calibration_results["extended_pinhole"]의
    distortion 벡터 길이로 프로젝트 전체에 딱 한 번 실제 의미를 판별한
    (should_rename_extended_to_brown) 뒤, 그 판정을 프로젝트 안의 모든 model
    reference(calibration_results/validation_results/model_scores/
    cross_dataset_results/final_result 하위 전부)에 동일하게 적용한다 -
    한 프로젝트 안에서 "extended_pinhole"이 어떤 곳에서는 Brown, 다른 곳에서는
    Rational을 가리키는 모순된 상태가 생기지 않도록.

    ModelScore/CrossDatasetValidationResult/DiagnosisReport 자체에는
    distortion vector가 없으므로 개별적으로 재판별하지 않는다 - 위에서 정한
    프로젝트 단위 판정을 그대로 물려받는다(사용자 스펙 6번 "Model Score
    migration 주의" 항목과 동일한 원칙).

    "pinhole"/"fisheye"는 v1과 v2에서 의미가 같으므로 손대지 않는다.
    object_releasing_result가 v1 payload에 아예 없는 경우는 project_from_dict의
    기존 .get(...) 처리로 이미 None으로 정상 로드되므로 여기서 손댈 필요 없다.
    """
    project = payload.get("project", {})
    calibration_results: dict = project.get("calibration_results", {}) or {}
    validation_results: dict = project.get("validation_results", {}) or {}

    legacy_model_map: dict[str, str] = {}

    legacy_entry = calibration_results.get("extended_pinhole")
    if legacy_entry is not None:
        dist_len = _raw_array_len(legacy_entry.get("distortion"))
        if dist_len == 5:
            legacy_model_map["extended_pinhole"] = "brown_conrady"
            logger.info(
                "Migrated legacy project model: extended_pinhole (5 coeffs) -> brown_conrady"
            )
        elif dist_len is not None and dist_len >= 8:
            legacy_model_map["extended_pinhole"] = "extended_pinhole"
            logger.info(
                "Legacy project model extended_pinhole (%d coeffs) already matches current "
                "Rational meaning - kept as extended_pinhole.", dist_len
            )
        elif dist_len is not None:
            logger.warning(
                "Legacy project model extended_pinhole has an unexpected distortion length "
                "(%d) - left unchanged, please verify manually.", dist_len
            )
        else:
            logger.warning(
                "Legacy project model extended_pinhole has no readable distortion vector - "
                "left unchanged, please verify manually."
            )

    _apply_model_map_to_field(project.get("camera_config"), "model", legacy_model_map)

    _apply_model_map_to_keyed_dict(calibration_results, legacy_model_map)
    for result in calibration_results.values():
        _apply_model_map_to_field(result, "model_name", legacy_model_map)

    _apply_model_map_to_keyed_dict(validation_results, legacy_model_map)
    for result in validation_results.values():
        _apply_model_map_to_field(result, "model_name", legacy_model_map)

    # 최상위 model_scores / cross_dataset_results 리스트.
    for score in project.get("model_scores", []) or []:
        _apply_model_map_to_field(score, "model_name", legacy_model_map)
    for cross_result in project.get("cross_dataset_results", []) or []:
        _apply_model_map_to_field(cross_result, "model_name", legacy_model_map)
    _apply_model_map_to_field(project.get("object_releasing_result"), "model_name", legacy_model_map)

    object_releasing_comparison = project.get("standard_vs_object_releasing_comparison")
    if object_releasing_comparison:
        _apply_model_map_to_field(
            object_releasing_comparison.get("standard_result"),
            "model_name",
            legacy_model_map,
        )
        _apply_model_map_to_field(
            object_releasing_comparison.get("object_releasing_result"),
            "model_name",
            legacy_model_map,
        )

    final_result = project.get("final_result")
    if final_result:
        _apply_model_map_to_field(final_result, "chosen_model", legacy_model_map)
        _apply_model_map_to_field(final_result.get("calibration"), "model_name", legacy_model_map)
        _apply_model_map_to_field(final_result.get("validation"), "model_name", legacy_model_map)
        for score in final_result.get("model_scores", []) or []:
            _apply_model_map_to_field(score, "model_name", legacy_model_map)
        _apply_model_map_to_field(final_result.get("diagnosis"), "model_name", legacy_model_map)

    if legacy_model_map.get("extended_pinhole") == "brown_conrady":
        logger.info(
            "Migrated legacy project model references (model_scores/cross_dataset_results/"
            "final_result): extended_pinhole -> brown_conrady"
        )

    payload["format_version"] = PROJECT_FORMAT_VERSION
    return payload


def project_from_dict(payload: dict) -> CalibrationProject:
    version = payload.get("format_version")
    if version == 1:
        payload = migrate_v1_to_v2(payload)
        version = payload["format_version"]
    if version != PROJECT_FORMAT_VERSION:
        raise ValueError(
            f"지원하지 않는 프로젝트 파일 버전입니다: {version} "
            f"(이 버전의 툴은 {PROJECT_FORMAT_VERSION}만 지원)"
        )
    d = payload["project"]

    calibration_results = {
        CameraModelType(k): _calibration_result_from_dict(v)
        for k, v in d.get("calibration_results", {}).items()
    }
    validation_results = {
        CameraModelType(k): _validation_result_from_dict(v)
        for k, v in d.get("validation_results", {}).items()
    }
    export_paths = {ExportFormat(k): v for k, v in d.get("export_paths", {}).items()}

    return CalibrationProject(
        project_name=d["project_name"],
        camera_config=_camera_config_from_dict(d["camera_config"]),
        pattern_config=_pattern_config_from_dict(d["pattern_config"]),
        dataset=_dataset_from_dict(d.get("dataset", {})),
        calibration_results=calibration_results,
        object_releasing_result=(
            _calibration_result_from_dict(d["object_releasing_result"])
            if d.get("object_releasing_result") else None
        ),
        object_releasing_validation_result=_object_releasing_validation_result_from_dict(
            d.get("object_releasing_validation_result")
        ),
        standard_vs_object_releasing_comparison=_standard_vs_object_releasing_comparison_from_dict(
            d.get("standard_vs_object_releasing_comparison")
        ),
        validation_results=validation_results,
        cross_dataset_results=[
            _cross_dataset_result_from_dict(r)
            for r in d.get("cross_dataset_results", [])
        ],
        model_scores=[_model_score_from_dict(s) for s in d.get("model_scores", [])],
        outlier_result=_outlier_result_from_dict(d.get("outlier_result")),
        scene_quality_analysis=_scene_quality_analysis_from_dict(d.get("scene_quality_analysis")),
        subset_calibration_result=_subset_calibration_result_from_dict(d.get("subset_calibration_result")),
        final_result=_final_result_from_dict(d.get("final_result")),
        windshield_config=_windshield_config_from_dict(d.get("windshield_config")),
        windshield_dataset=_dataset_from_dict(d["windshield_dataset"]) if d.get("windshield_dataset") else None,
        windshield_results={
            windshield_result_key_from_storage(k): _windshield_calibration_result_from_dict(v)
            for k, v in d.get("windshield_results", {}).items()
        },
        created_at=_dt(d.get("created_at")),
        updated_at=_dt(d.get("updated_at")),
        export_paths=export_paths,
    )


def load_project(path: str) -> tuple[CalibrationProject, list[str]]:
    """.ccproj 파일을 읽어 CalibrationProject로 복원.

    Returns:
        (project, missing_image_paths) - 프로젝트 저장 이후 원본 이미지
        파일이 옮겨지거나 지워졌으면 missing_image_paths에 담겨서 반환된다
        (예외로 죽지 않고, 호출부가 사용자에게 경고할 수 있게 함 - 설계
        문서 9번과 같은 "파일 없어도 메타데이터는 살아있다" 원칙).
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    project = project_from_dict(payload)

    missing = [
        f.image_info.path
        for f in project.dataset.frames
        if not Path(f.image_info.path).exists()
    ]
    return project, missing

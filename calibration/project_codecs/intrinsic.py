"""
camera_calibrator.calibration.project_codecs.intrinsic
==============================================================

Phase D-2 안정화 - Camera Intrinsic Calibration/Validation/Dataset 관련
dict -> dataclass 디코더(`_pattern_config_from_dict` ~ `_final_result_from_dict`).
`calibration/project_io.py`에서 로직 변경 없이 그대로 옮겨왔다(순수 파일
이동 - 어떤 함수 본문도 수정하지 않았다).
"""

from __future__ import annotations

import numpy as np
import math

from calibration.project_codecs.common import _arr
from calibration.types import (
    AprilGridVariant,
    CalibrationConfidenceReport,
    CalibrationMethod,
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    CaptureRecommendation,
    CircleGridType,
    CornerOutlierResult,
    CoverageCell,
    CrossDatasetValidationResult,
    Dataset,
    DatasetQualityScore,
    DetectionResult,
    DiagnosisReport,
    DiagnosisSeverity,
    DiversityScores,
    FailurePattern,
    FinalResult,
    Frame,
    FrameQuality,
    FrameStatus,
    HoldoutEvidenceGate,
    ImageInfo,
    ModelScore,
    ObjectReleasingValidationResult,
    ObservabilityReport,
    OutlierResult,
    ParameterCorrelation,
    ParameterUncertainty,
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
    StraightnessBreakdown,
    SubsetCalibrationResult,
    UndistortionQualityReport,
    ValidationResult,
)

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


# correlation_method 필드가 저장되지 않은 구버전 .ccproj를 로드할 때 쓰는 기본값.
#
# 이 필드가 추가되기 전까지 이 코드베이스의 parameter correlation 계산은 항상
# np.corrcoef(raw Jacobian, rowvar=False) 기반 구현 하나뿐이었다(git 히스토리로
# 확인됨 - 다른 구현이 존재한 적이 없다). 따라서 필드가 없다는 것 자체가 "그
# corrcoef 기반 legacy 구현으로 계산됐다"는 것을 신뢰성 있게 나타내는 신호이며,
# 새 covariance_from_normalized_jacobian 방식으로 계산된 것처럼 라벨링하면 안 된다.
_LEGACY_CORRELATION_METHOD = "legacy_jacobian_column_correlation"


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
        raw_condition_number=d.get("raw_condition_number"),
        normalized_condition_number=d.get("normalized_condition_number"),
        normalization_scales=d.get("normalization_scales", {}),
        raw_singular_values=d.get("raw_singular_values", []),
        min_singular_value=d.get("min_singular_value"),
        max_singular_value=d.get("max_singular_value"),
        max_abs_correlation=d.get("max_abs_correlation"),
        correlation_matrix=d.get("correlation_matrix", []),
        # 필드가 저장돼 있으면 그 값을 그대로 신뢰한다(새 프로젝트라면
        # "covariance_from_normalized_jacobian", 이미 legacy로 라벨링된
        # 값이라면 그것도 그대로 보존). 필드 자체가 없는 구버전 프로젝트만
        # legacy 기본값으로 복원한다 - 새 방식으로 계산됐다고 잘못 표시하지 않는다.
        correlation_method=d.get("correlation_method", _LEGACY_CORRELATION_METHOD),
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


def _holdout_evidence_gate_from_dict(d) -> HoldoutEvidenceGate | None:
    if d is None:
        return None
    return HoldoutEvidenceGate(
        status=d.get("status", "not_evaluated"),
        test_frame_count=d.get("test_frame_count", 0),
        test_corner_count=d.get("test_corner_count", 0),
        test_coverage_pct=d.get("test_coverage_pct", 0.0),
        test_pose_diversity=d.get("test_pose_diversity", 0.0),
        reasons=d.get("reasons", []),
    )


def _validation_result_from_dict(d: dict) -> ValidationResult:
    return ValidationResult(
        train_frame_ids=d.get("train_frame_ids", []),
        test_frame_ids=d.get("test_frame_ids", []),
        train_rms=d.get("train_rms"), test_rms=d.get("test_rms"), edge_rms=d.get("edge_rms"),
        straightness_residual=d.get("straightness_residual"),
        straightness_source=d.get("straightness_source"),
        straightness_breakdown=_straightness_breakdown_from_dict(d.get("straightness_breakdown")),
        train_residual_stats=_residual_stats_from_dict(d.get("train_residual_stats")),
        test_residual_stats=_residual_stats_from_dict(d.get("test_residual_stats")),
        success=d.get("success", True), error_message=d.get("error_message"),
        failed_test_frame_ids=d.get("failed_test_frame_ids", []),
        failed_test_frame_reasons=d.get("failed_test_frame_reasons", {}),
        per_frame_error=d.get("per_frame_error", {}),
        evidence_gate=_holdout_evidence_gate_from_dict(d.get("evidence_gate")),
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
    score = d.get("score")
    if score is None and not d.get("is_selection_eligible", True):
        score = math.inf
    return ModelScore(
        model_name=CameraModelType(d["model_name"]), score=score,
        components=d.get("components", {}), is_recommended=d.get("is_recommended", False),
        evidence_components=d.get("evidence_components", []),
        is_selection_eligible=d.get("is_selection_eligible", True),
        selection_status=d.get("selection_status", "ELIGIBLE"),
        selection_ineligibility_reason=d.get("selection_ineligibility_reason"),
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



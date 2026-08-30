"""
camera_calibrator.app.cli
==============================

헤드리스(UI 없는) CLI 진입점. CI/서버/배치 처리용.

    python -m app.cli --images ./photos --squares-x 7 --squares-y 5 \
        --square-size 0.04 --marker-size 0.03 --output-dir ./out

전체 흐름은 ui/worker.py의 PipelineWorker와 동일한 순서(Detection -> Quality
-> Standard 4모델 -> Validation -> 추천 -> [옵션] Outlier 제거 -> Final Result ->
Export)를 따르되, Qt 시그널 대신 stdout에 진행상황을 찍고 예외는 그대로
위로 던지지 않고 종료 코드로 변환한다. `--calibration-method object_releasing`이면
Standard 4모델과 별도로 Object-Releasing(Advanced) 결과 + 전용 Hold-out +
Standard Brown-Conrady와의 비교도 함께 계산한다.

종료 코드:
    0 = 성공 (export 파일까지 다 만들어짐)
    1 = 입력 문제 (이미지 없음, 인자 오류 등)
    2 = 검출은 됐지만 모든 모델의 캘리브레이션이 실패함 (또는 Object-Releasing
        계산 자체가 실패함)
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import glob as globmod
import html
import json
import logging
import sys
from pathlib import Path

import cv2

from calibration.compare import run_all_models, format_comparison_table
from calibration.detector import detect_dataset, summarize_dataset
from calibration.frame_quality import compute_frame_quality_scores
from calibration.frame_quality import compute_dataset_quality_score, format_dataset_quality_score
from calibration.image_quality import evaluate_dataset_image_quality, format_image_quality_summary, find_duplicate_groups
from calibration.log_utils import setup_logging
from calibration.models.common import infer_image_size, collect_calibration_inputs
from calibration.kfold import compute_kfold_validation, compute_repeated_kfold, format_kfold_result, format_repeated_kfold_result
from calibration.repeatability import compute_repeatability, format_repeatability
from calibration.bootstrap import compute_parameter_bootstrap, format_parameter_uncertainty
from calibration.calibration_io import load_standard_calibration
from calibration.external_compare import ExternalComparisonResult, compare_reference_candidate_calibrations
from calibration.json_utils import json_safe
from calibration.models.pinhole import calibrate_pinhole
from calibration.models.object_releasing import (
    calibrate_object_releasing_brown_conrady,
    is_object_releasing_supported_pattern,
)
from calibration.object_releasing_validation import (
    compare_standard_vs_object_releasing_brown,
    format_standard_vs_object_releasing_table,
    validate_object_releasing_holdout,
)
from calibration.outlier import recalibrate_with_outlier_pruning, format_outlier_before_after
from calibration.outlier import recalibrate_with_corner_outlier_pruning, format_corner_outlier_before_after
from calibration.quality import (
    analyze_dataset_quality,
    coverage_percentage,
    compute_pose_distribution_stats,
    format_pose_distribution_stats,
)
from calibration.target_quality import evaluate_dataset_target_quality, format_target_quality_summary
from calibration.sanity_check import run_sanity_checks, format_sanity_checks
from calibration.residual_stats import format_residual_stats, format_residual_boxplot, format_cdf
from calibration.spatial_error_map import format_spatial_error_map
from calibration.radial_profile import format_radial_bands
from calibration.recommender import (
    build_recommendation_message,
    compare_model_rankings,
    compute_final_result,
    compute_model_scores,
)
from calibration.types import (
    AprilGridVariant,
    CalibrationMethod,
    CalibrationProject,
    CameraConfig,
    CameraModelType,
    CircleGridType,
    PatternConfig,
    PatternType,
)
from calibration.project_io import load_project, save_project
from calibration.validation import (
    format_cross_dataset_validation_table,
    format_validation_table,
    format_train_test_residual_comparison,
    format_straightness_comparison,
    validate_all_models,
    validate_cross_datasets,
    validate_holdout,
    split_train_test,
    _subset_dataset,
    recalibrate_train_with_outlier_pruning,
    recalibrate_train_with_corner_outlier_pruning,
)
from export.opencv import export_opencv_yaml
from export.report import export_html_report
from export.ros import export_ros_camera_info
from export.json_export import export_json
from export.csv_export import export_csv
from export.kalibr import build_kalibr_camera_calibration_command, export_kalibr_target_yaml

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

_MODEL_BY_NAME = {
    "pinhole": CameraModelType.PINHOLE,
    "ideal_pinhole": CameraModelType.PINHOLE,
    "brown_conrady": CameraModelType.BROWN_CONRADY,
    "brown-conrady": CameraModelType.BROWN_CONRADY,
    "extended_pinhole": CameraModelType.EXTENDED_PINHOLE,
    "extended": CameraModelType.EXTENDED_PINHOLE,
    "fisheye": CameraModelType.FISHEYE,
}
_MODEL_CLI_CHOICES = sorted(_MODEL_BY_NAME)
_PATTERN_BY_NAME = {
    "charuco": PatternType.CHARUCO,
    "chessboard": PatternType.CHESSBOARD,
    "circle_grid": PatternType.CIRCLE_GRID,
    "circle-grid": PatternType.CIRCLE_GRID,
    "circles": PatternType.CIRCLE_GRID,
    "apriltag_grid": PatternType.APRILGRID,
    "aprilgrid": PatternType.APRILGRID,
    "april_grid": PatternType.APRILGRID,
}

_DEFAULT_EXPORTS = ["opencv", "ros", "report"]
_DIAGNOSTIC_EXPORTS = ["opencv", "ros", "report", "json", "csv"]
_DIAGNOSTIC_DEFAULT_KFOLD = 5
_DIAGNOSTIC_DEFAULT_BOOTSTRAP = 100


class CliError(Exception):
    """인자/입력 문제 - argparse 밖에서 나는 사용자 오류. 종료 코드 1로 변환됨."""


def _load_config_file(path: str, known_dests: set[str]) -> dict:
    """--config로 지정된 .yaml/.yml/.json 파일을 읽어 argparse dest 이름 ->
    값 딕셔너리로 반환한다.

    형식: 최상위가 key: value인 평범한 객체. 키는 각 옵션의 argparse dest와
    동일해야 한다 (예: --square-size -> square_size, --images -> images).
    하이픈(-)으로 써도 관대하게 받아준다 (예: square-size도 허용).

        # camera.yaml 예시
        squares_x: 7
        squares_y: 5
        square_size: 0.04
        marker_size: 0.03
        dictionary: DICT_5X5_100
        sensor_name: front_camera
        output_dir: ./out
        export: [opencv, ros, report, json]

    실행 시 --config camera.yaml --square-size 0.05 처럼 같은 옵션을 커맨드
    라인에 또 주면, 커맨드라인 쪽이 항상 이긴다 (build_arg_parser()가 반환한
    parser에 이 함수의 결과를 parser.set_defaults(**config)로 얹은 뒤 실제
    파싱을 하기 때문 - argparse의 표준 동작: 명시적으로 준 옵션은 set_defaults()
    보다 항상 우선한다).
    """
    p = Path(path)
    if not p.exists():
        raise CliError(f"--config 파일을 찾을 수 없습니다: {path}")

    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            import yaml
            data = yaml.safe_load(text)
        elif suffix == ".json":
            data = json.loads(text)
        else:
            raise CliError(
                f"--config는 .yaml/.yml/.json만 지원합니다 (입력: {p.suffix or '(확장자 없음)'})"
            )
    except CliError:
        raise
    except Exception as e:  # noqa: BLE001 - yaml/json 파서 예외를 CliError로 통일
        raise CliError(f"--config 파일을 읽는 중 오류: {path}\n  {e}") from e

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise CliError(
            f"--config 파일의 최상위는 key: value 형태의 객체여야 합니다 (입력: {path})"
        )

    # 하이픈 표기를 언더스코어로 정규화 (예: square-size -> square_size)
    normalized = {str(k).replace("-", "_"): v for k, v in data.items()}

    unknown = set(normalized) - known_dests
    if unknown:
        raise CliError(
            f"--config 파일에 알 수 없는 키가 있습니다: {', '.join(sorted(unknown))}\n"
            f"  (--help로 사용 가능한 옵션 이름을 확인하세요. 하이픈은 언더스코어로 "
            f"바꿔 쓰세요 - 예: square-size -> square_size)"
        )
    return normalized


def _log(quiet: bool, msg: str) -> None:
    if not quiet:
        print(msg)


def _resolve_image_path_items(items: list[str]) -> list[str]:
    """파일/glob/디렉토리 혼용 입력을 이미지 경로 리스트로 확장한다."""
    paths: list[str] = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            for ext in _IMAGE_EXTENSIONS:
                paths.extend(sorted(str(x) for x in p.glob(ext)))
        elif any(ch in item for ch in "*?[]"):
            paths.extend(sorted(globmod.glob(item)))
        elif p.is_file():
            paths.append(str(p))
        else:
            raise CliError(f"이미지 경로를 찾을 수 없습니다: {item}")

    # 중복 제거(디렉토리+glob이 겹칠 수 있음), 순서는 유지
    seen = set()
    unique = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    if not unique:
        raise CliError("주어진 경로에서 이미지를 찾지 못했습니다 (jpg/jpeg/png/bmp).")
    return unique


def _resolve_image_paths(args) -> list[str]:
    """--images(파일/glob/디렉토리 혼용)와 --bag 중 하나로 최종 이미지 경로 리스트를 만든다."""
    if args.bag:
        return _extract_from_bag(args)

    if not args.images:
        raise CliError("--images 또는 --bag 중 하나는 반드시 지정해야 합니다.")
    return _resolve_image_path_items(args.images)


def _parse_cross_dataset_spec(spec: str, index: int) -> tuple[str, list[str]]:
    """NAME=PATH 또는 PATH 형식의 cross-dataset CLI 입력을 정규화한다."""
    if "=" in spec:
        name, raw_paths = spec.split("=", 1)
        dataset_id = name.strip()
        path_spec = raw_paths.strip()
        if not dataset_id:
            raise CliError(f"--cross-dataset #{index}: NAME=PATH에서 NAME이 비어 있습니다.")
        if not path_spec:
            raise CliError(f"--cross-dataset {dataset_id}: PATH가 비어 있습니다.")
    else:
        path_spec = spec.strip()
        cleaned = path_spec.rstrip("\\/")
        dataset_id = Path(cleaned).stem or f"target_{index}"

    return dataset_id, [path_spec]


def _load_cross_dataset_targets(args, pattern_config, camera_config):
    """CLI로 지정된 Dataset B/C/... 이미지를 검출해 Dataset 객체로 만든다."""
    targets = {}
    specs = args.cross_datasets or []
    used_ids: set[str] = set()
    for i, spec in enumerate(specs, start=1):
        dataset_id, image_specs = _parse_cross_dataset_spec(spec, i)
        base_id = dataset_id
        suffix = 2
        while dataset_id in used_ids:
            dataset_id = f"{base_id}_{suffix}"
            suffix += 1
        used_ids.add(dataset_id)

        image_paths = _resolve_image_path_items(image_specs)
        _log(args.quiet, f"Cross-dataset target '{dataset_id}' 검출 중... ({len(image_paths)}장)")
        target = detect_dataset(
            image_paths, pattern_config,
            parallel=args.jobs != 1,
            max_workers=None if args.jobs in (0, 1) else args.jobs,
        )
        _log(args.quiet, f"  {summarize_dataset(target)}")
        if target.num_detected == 0:
            _log(args.quiet, f"  ⚠ '{dataset_id}'에서 검출된 프레임이 없어 실패 결과로 기록됩니다.")
        else:
            analyze_dataset_quality(target, camera_config)
        targets[dataset_id] = target
    return targets


def _extract_from_bag(args) -> list[str]:
    if not args.topic:
        raise CliError("--bag을 쓰려면 --topic도 함께 지정해야 합니다 (--list-topics로 목록 확인 가능).")

    from calibration.rosbag_reader import extract_images_from_bag

    out_dir = str(Path(args.output_dir) / "bag_extracted")
    try:
        paths = extract_images_from_bag(
            args.bag, args.topic, out_dir, min_interval_sec=args.bag_interval
        )
    except ImportError as e:
        raise CliError(str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise CliError(f"bag에서 이미지 추출 실패: {e}") from e

    if not paths:
        raise CliError(f"토픽 '{args.topic}'에서 추출된 이미지가 없습니다.")
    return paths


def _list_bag_topics(bag_path: str) -> int:
    try:
        from calibration.rosbag_reader import list_image_topics
        topics = list_image_topics(bag_path)
    except ImportError as e:
        print(f"오류: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"bag을 여는 데 실패했습니다: {e}", file=sys.stderr)
        return 1

    if not topics:
        print("이미지 토픽을 찾지 못했습니다.")
        return 0
    for t in topics:
        print(f"{t.name}\t{t.msg_type}\t{t.count}개")
    return 0


def _build_pattern_config(args) -> PatternConfig:
    pattern_name = args.pattern.value if isinstance(args.pattern, PatternType) else str(args.pattern)
    try:
        pattern_type = _PATTERN_BY_NAME[pattern_name.lower()]
    except KeyError:
        raise CliError(f"알 수 없는 패턴 타입: {args.pattern}")

    if args.squares_x < 3 or args.squares_y < 3:
        raise CliError("--squares-x/--squares-y는 각각 3 이상이어야 합니다.")
    if args.square_size <= 0:
        raise CliError("--square-size는 0보다 커야 합니다.")
    uses_marker_dictionary = pattern_type in (PatternType.CHARUCO, PatternType.APRILGRID)
    if uses_marker_dictionary and args.marker_size is None:
        raise CliError(f"{pattern_type.value} 패턴은 --marker-size가 반드시 필요합니다.")
    if uses_marker_dictionary:
        if args.marker_size <= 0:
            raise CliError("--marker-size는 0보다 커야 합니다.")
        if args.marker_size >= args.square_size:
            raise CliError("--marker-size는 --square-size보다 작아야 합니다.")
        if pattern_type == PatternType.APRILGRID and not str(args.dictionary).startswith("DICT_APRILTAG_"):
            raise CliError(
                "AprilGrid 패턴은 --dictionary에 OpenCV AprilTag dictionary를 지정해야 합니다 "
                "(예: DICT_APRILTAG_36h11)."
            )

    circle_grid_type = getattr(args, "circle_grid_type", CircleGridType.SYMMETRIC)
    if not isinstance(circle_grid_type, CircleGridType):
        circle_grid_type = CircleGridType(str(circle_grid_type))
    aprilgrid_variant = getattr(args, "aprilgrid_variant", AprilGridVariant.OPENCV_APRILTAG3)
    if not isinstance(aprilgrid_variant, AprilGridVariant):
        aprilgrid_variant = AprilGridVariant(str(aprilgrid_variant))

    return PatternConfig(
        type=pattern_type,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_size=args.square_size,
        marker_size=args.marker_size if uses_marker_dictionary else None,
        dictionary=args.dictionary if uses_marker_dictionary else None,
        circle_grid_type=circle_grid_type,
        aprilgrid_variant=aprilgrid_variant,
    )


def _infer_resolution(first_image_path: str) -> tuple[int, int]:
    img = cv2.imread(first_image_path)
    if img is None:
        raise CliError(f"해상도를 유추하려고 첫 이미지를 열었는데 실패했습니다: {first_image_path}")
    h, w = img.shape[:2]
    return w, h


def _write_json_summary(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _benchmark_result_payload(
    result: ExternalComparisonResult,
    *,
    reference_path: str,
    candidate_path: str,
    validation_inputs: list[str],
    dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
) -> dict:
    payload = json_safe(dataclasses.asdict(result), ndarray_wrapper=False)
    payload.update({
        "benchmark_format_version": 1,
        "reference_file": reference_path,
        "candidate_file": candidate_path,
        "validation_inputs": validation_inputs,
        "dataset": {
            "num_total": dataset.num_total,
            "num_detected": dataset.num_detected,
        },
        "camera": {
            "width": camera_config.width,
            "height": camera_config.height,
            "sensor_name": camera_config.sensor_name,
        },
        "pattern": json_safe(dataclasses.asdict(pattern_config), ndarray_wrapper=False),
    })
    return payload


def _benchmark_report_html(payload: dict) -> str:
    decision = payload.get("winner_decision") or {}
    final_rows = payload.get("final_benchmark_rows") or []
    caveats = payload.get("caveats") or []
    evidence = decision.get("evidence") or []
    warnings = decision.get("warnings") or []

    def esc(value) -> str:
        return html.escape("" if value is None else str(value))

    rows_html = "".join(
        "<tr>"
        f"<td>{esc(row.get('metric'))}</td>"
        f"<td>{esc(row.get('reference'))}</td>"
        f"<td>{esc(row.get('candidate'))}</td>"
        f"<td>{esc(row.get('improvement'))}</td>"
        f"<td>{esc(row.get('winner'))}</td>"
        "</tr>"
        for row in final_rows
    )
    evidence_html = "".join(f"<li>{esc(item)}</li>" for item in evidence)
    warning_html = "".join(f"<li>{esc(item)}</li>" for item in warnings)
    caveat_html = "".join(f"<li>{esc(item)}</li>" for item in caveats)
    verdict = payload.get("verdict") or decision.get("status") or "N/A"

    def row_by_metric(metric: str) -> dict:
        return next((row for row in final_rows if row.get("metric") == metric), {})

    def metric_summary(metric: str) -> str:
        row = row_by_metric(metric)
        if not row:
            return f"{metric}: N/A"
        return (
            f"{metric}: Reference {row.get('reference', 'N/A')}, "
            f"Candidate {row.get('candidate', 'N/A')}, "
            f"Improvement {row.get('improvement', 'N/A')}, Winner {row.get('winner', 'N/A')}"
        )

    stat_bits = []
    for test in payload.get("statistical_tests") or []:
        stat_bits.append(
            f"{test.get('test_name', 'test')} p={test.get('p_value', 'N/A')}, "
            f"effect={test.get('effect_size', 'N/A')} {test.get('effect_size_name', '')}".strip()
        )
    bootstrap = payload.get("bootstrap_comparison") or {}
    if bootstrap.get("probability_candidate_better") is not None:
        stat_bits.append(
            "Bootstrap P(Candidate < Reference)="
            f"{float(bootstrap['probability_candidate_better']) * 100.0:.2f}%"
        )

    visual_summary = "; ".join([
        metric_summary("Edge RMS"),
        metric_summary("Radial P95"),
        metric_summary("Straightness"),
        f"Residual heatmaps: {', '.join((payload.get('residual_heatmaps') or {}).keys()) or 'N/A'}",
        f"Worst-case rows: {len(payload.get('worst_case_rows') or [])}",
    ])

    param_bits = []
    for diag in (payload.get("parameter_diagnostics") or {}).values():
        weak = ", ".join(diag.get("weak_parameters") or []) or "none"
        param_bits.append(
            f"{diag.get('side_label', 'side')}: rank {diag.get('rank', 'N/A')}/"
            f"{diag.get('jacobian_cols', 'N/A')}, condition {diag.get('condition_number', 'N/A')}, "
            f"max |corr| {diag.get('max_abs_correlation', 'N/A')}, weak {weak}"
        )

    section_rows = [
        ("Performance Comparison", " / ".join([
            metric_summary("RMSE"),
            metric_summary("P95"),
            metric_summary("Frame wins"),
        ])),
        ("Statistical Evidence", "; ".join(stat_bits) or "N/A"),
        ("Visual Evidence", visual_summary),
        ("Parameter Analysis", "; ".join(param_bits) or "N/A"),
        ("Model Analysis", "N/A for standalone Reference/Candidate benchmark CLI."),
        ("FINAL VERDICT", decision.get("status") or "N/A"),
        ("One-line diagnosis", verdict),
    ]
    sections_html = "".join(
        f"<tr><th>{esc(name)}</th><td>{esc(summary)}</td></tr>"
        for name, summary in section_rows
    )

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>Calibration Benchmark Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #202124; }}
    h1, h2 {{ margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #d0d7de; padding: 8px; text-align: left; }}
    th {{ background: #f6f8fa; }}
    .verdict {{ font-size: 18px; font-weight: 700; margin: 12px 0; }}
    .muted {{ color: #5f6368; }}
  </style>
</head>
<body>
  <h1>Calibration Benchmark Report</h1>
  <p class="muted">Reference: {esc(payload.get('reference_file'))}<br>
  Candidate: {esc(payload.get('candidate_file'))}</p>
  <div class="verdict">{esc(verdict)}</div>
  <h2>Final Report Summary</h2>
  <table>
    <tbody>{sections_html}</tbody>
  </table>
  <h2>Final Benchmark Table</h2>
  <table>
    <thead><tr><th>Metric</th><th>Reference</th><th>Candidate</th><th>Improvement</th><th>Winner</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
  <h2>Evidence</h2>
  <ul>{evidence_html or '<li>N/A</li>'}</ul>
  <h2>Warnings</h2>
  <ul>{warning_html or '<li>N/A</li>'}</ul>
  <h2>Caveats</h2>
  <ul>{caveat_html or '<li>N/A</li>'}</ul>
</body>
</html>
"""


def _run_benchmark_cli(args) -> int:
    quiet = args.quiet
    try:
        pattern_config = _build_pattern_config(args)
        validation_inputs = args.validation_dataset or []
        image_paths = _resolve_image_path_items(validation_inputs)
        reference = load_standard_calibration(args.reference, camera_key=args.reference_camera_key)
        candidate = load_standard_calibration(args.candidate, camera_key=args.candidate_camera_key)
    except (CliError, ValueError, OSError) as e:
        print(f"오류: {e}", file=sys.stderr)
        return 1

    if args.width and args.height:
        width, height = args.width, args.height
    else:
        width = reference.width or candidate.width
        height = reference.height or candidate.height
        if not width or not height:
            try:
                width, height = _infer_resolution(image_paths[0])
            except CliError as e:
                print(f"오류: {e}", file=sys.stderr)
                return 1
    camera_config = CameraConfig(width=int(width), height=int(height), sensor_name=args.sensor_name)

    _log(quiet, f"Benchmark validation dataset 검출 중... ({len(image_paths)}장, {width}x{height})")
    try:
        dataset = detect_dataset(
            image_paths, pattern_config,
            parallel=args.jobs != 1,
            max_workers=None if args.jobs in (0, 1) else args.jobs,
        )
    except ValueError as e:
        print(f"오류: {e}", file=sys.stderr)
        return 1
    _log(quiet, summarize_dataset(dataset))
    if dataset.num_detected == 0:
        print("오류: validation dataset에서 calibration target이 검출되지 않았습니다.", file=sys.stderr)
        return 1

    test_ids = [f.image_info.image_id for f in dataset.enabled_frames if f.detection and f.detection.success]
    kfold = args.kfold or 5
    n_bootstrap = args.bootstrap if args.bootstrap is not None else args.n_bootstrap
    result = compare_reference_candidate_calibrations(
        dataset,
        camera_config,
        pattern_config,
        reference,
        candidate,
        test_ids,
        benchmark_kfold=kfold,
        benchmark_bootstrap=n_bootstrap,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = _benchmark_result_payload(
        result,
        reference_path=args.reference,
        candidate_path=args.candidate,
        validation_inputs=validation_inputs,
        dataset=dataset,
        camera_config=camera_config,
        pattern_config=pattern_config,
    )

    json_path = out_dir / "benchmark_result.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(quiet, f"저장: {json_path}")

    if args.benchmark_report:
        report_path = out_dir / "benchmark_report.html"
        report_path.write_text(_benchmark_report_html(payload), encoding="utf-8")
        _log(quiet, f"저장: {report_path}")

    if args.json_summary:
        _write_json_summary(args.json_summary, {
            "success": result.external.success and result.mine.success,
            "mode": "benchmark",
            "winner_status": result.winner_decision.status,
            "num_validation_images": dataset.num_total,
            "num_validation_detected": dataset.num_detected,
            "benchmark_result_json": str(json_path),
            "benchmark_report_html": str(out_dir / "benchmark_report.html") if args.benchmark_report else None,
        })

    print(result.verdict)
    return 0 if result.external.success and result.mine.success else 2


def _normalize_cli_args(args, parser: argparse.ArgumentParser):
    """Turn friendly CLI aliases/presets into the internal option names.

    This keeps older flags working while supporting the document-style UX:
    `--diagnostic --cross-validation 5 --bootstrap 100`.
    """
    if args.jobs < 0:
        parser.error("--jobs는 0 이상이어야 합니다 (0=자동, 1=순차, N=N개 worker).")
    if args.width is not None and args.width <= 0:
        parser.error("--width는 0보다 커야 합니다.")
    if args.height is not None and args.height <= 0:
        parser.error("--height는 0보다 커야 합니다.")

    if args.cross_validation is not None:
        if args.cross_validation < 2:
            parser.error("--cross-validation은 2 이상이어야 합니다.")
        if args.kfold is not None and args.kfold != args.cross_validation:
            parser.error("--kfold와 --cross-validation은 같은 기능입니다. 둘 중 하나만 쓰거나 같은 값을 쓰세요.")
        args.kfold = args.cross_validation

    if args.kfold is not None and args.kfold < 2:
        parser.error("--kfold는 2 이상이어야 합니다.")
    if args.kfold_repeats < 1:
        parser.error("--kfold-repeats는 1 이상이어야 합니다.")

    if args.bootstrap is not None:
        if args.bootstrap < 1:
            parser.error("--bootstrap은 1 이상이어야 합니다.")
        args.bootstrap_ci = True
        args.n_bootstrap = args.bootstrap
    if args.n_bootstrap < 1:
        parser.error("--n-bootstrap은 1 이상이어야 합니다.")

    args.benchmark_mode = bool(args.reference or args.candidate or args.validation_dataset)
    if args.benchmark_mode:
        if not args.reference or not args.candidate:
            parser.error("benchmark mode는 --reference와 --candidate를 둘 다 지정해야 합니다.")
        if not args.validation_dataset:
            parser.error("benchmark mode는 --validation-dataset이 필요합니다.")
        if args.images or args.bag or args.load_project:
            parser.error("benchmark mode에서는 --images/--input, --bag, --load-project를 함께 쓸 수 없습니다.")
        if args.benchmark_report:
            args.report = True
    elif args.benchmark_report:
        parser.error("--benchmark-report는 --reference/--candidate benchmark mode에서만 사용할 수 있습니다.")

    if args.export is None:
        args.export = list(_DEFAULT_EXPORTS)
    if args.report and "report" not in args.export:
        args.export.append("report")

    if args.diagnostic:
        if args.kfold is None:
            args.kfold = _DIAGNOSTIC_DEFAULT_KFOLD
        if args.bootstrap is None and not args.bootstrap_ci:
            args.bootstrap_ci = True
            args.n_bootstrap = _DIAGNOSTIC_DEFAULT_BOOTSTRAP
        for target in ("report", "json", "csv"):
            if target not in args.export:
                args.export.append(target)

    if args.models:
        seen = set()
        args.models = [m for m in args.models if not (m in seen or seen.add(m))]
        if args.model and args.model not in args.models:
            parser.error("--model은 --models 목록 안의 모델이어야 합니다.")
        if args.model is None and len(args.models) == 1:
            args.model = args.models[0]

    if not isinstance(args.calibration_method, CalibrationMethod):
        args.calibration_method = CalibrationMethod(str(args.calibration_method))

    return args


def _selected_model_types(args) -> list[CameraModelType]:
    names = args.models or ["pinhole", "brown_conrady", "extended_pinhole", "fisheye"]
    return [_MODEL_BY_NAME[name] for name in names]


def run_pipeline(args) -> int:
    quiet = args.quiet

    if args.load_project:
        return _run_from_loaded_project(args)

    # --- 1. 입력 이미지 확보 (파일/디렉토리/glob 또는 rosbag) ---
    try:
        image_paths = _resolve_image_paths(args)
        pattern_config = _build_pattern_config(args)
    except CliError as e:
        print(f"오류: {e}", file=sys.stderr)
        return 1

    if args.width and args.height:
        width, height = args.width, args.height
    else:
        try:
            width, height = _infer_resolution(image_paths[0])
        except CliError as e:
            print(f"오류: {e}", file=sys.stderr)
            return 1
    camera_config = CameraConfig(width=width, height=height, sensor_name=args.sensor_name)
    if args.calibration_method == CalibrationMethod.OBJECT_RELEASING and not is_object_releasing_supported_pattern(pattern_config):
        pattern_label = pattern_config.type.value if hasattr(pattern_config.type, "value") else str(pattern_config.type)
        print(
            f"오류: Object-Releasing is disabled for {pattern_label}. "
            "Supported targets: chessboard, circle_grid. Use Standard calibration for ChArUco/AprilGrid.",
            file=sys.stderr,
        )
        return 2

    _log(quiet, f"이미지 {len(image_paths)}장, 해상도 {width}x{height}")

    # --- 2. Detection ---
    pattern_label = pattern_config.type.value if hasattr(pattern_config.type, "value") else str(pattern_config.type)
    _log(quiet, f"{pattern_label} 패턴 검출 중...")
    try:
        dataset = detect_dataset(
            image_paths, pattern_config,
            parallel=args.jobs != 1,
            max_workers=None if args.jobs in (0, 1) else args.jobs,
        )
    except ValueError as e:
        print(f"오류: {e}", file=sys.stderr)
        return 1
    _log(quiet, summarize_dataset(dataset))
    if dataset.num_detected == 0:
        print("오류: 어떤 이미지에서도 ChArUco 패턴이 검출되지 않았습니다.", file=sys.stderr)
        return 1

    # --- 3. Quality Gate (이미지 품질 / Target 품질 / Coverage / Diversity) ---
    # 설계 문서 3-1/3-2번 - 캘리브레이션 계산 전에 데이터 자체가 괜찮은지 먼저 본다.
    image_reports, duplicate_groups = evaluate_dataset_image_quality(dataset)
    if not quiet:
        summary = format_image_quality_summary(image_reports, duplicate_groups)
        if summary.strip() and "발견되지 않았습니다" not in summary.split("\n")[0]:
            print(summary)

    target_reports = evaluate_dataset_target_quality(dataset.frames, pattern_config)
    if not quiet:
        target_summary = format_target_quality_summary(target_reports)
        if "경고 없음" not in target_summary:
            print(target_summary)

    _log(quiet, "Coverage Map / 데이터셋 품질 분석 중...")
    warnings = analyze_dataset_quality(dataset, camera_config)
    for w in warnings:
        _log(quiet, f"  \u26a0 {w}")

    image_size = infer_image_size(dataset, camera_config)

    # 설계 문서 6번 - Pose Diversity 확장 (X/Y/면적/yaw/pitch/roll/거리 분포)
    if not quiet:
        pose_stats = compute_pose_distribution_stats(dataset, camera_config)
        print(format_pose_distribution_stats(pose_stats))
    compute_frame_quality_scores(dataset, pattern_config, image_size, use_reprojection=False)

    # --- 4. Standard camera model 계산 ---
    _log(quiet, "Ideal Pinhole / Brown-Conrady / Rational / Fisheye 계산 중...")
    results = run_all_models(
        dataset, camera_config, use_rational_model=args.rational,
        bootstrap_jobs=args.jobs, model_jobs=args.jobs,
        persistent_cache_dir=args.cache_dir,
        models=_selected_model_types(args),
    )
    calibration_results = {r.model_name: r for r in results}
    object_releasing_result = None
    object_releasing_validation_result = None
    standard_vs_object_releasing_comparison = None
    if args.calibration_method == CalibrationMethod.OBJECT_RELEASING:
        object_releasing_result = calibrate_object_releasing_brown_conrady(dataset, camera_config, pattern_config)
        if not object_releasing_result.success:
            print(f"오류: Object-Releasing calibration failed: {object_releasing_result.error_message}", file=sys.stderr)
            return 2
        object_releasing_validation_result = validate_object_releasing_holdout(
            dataset, camera_config, pattern_config, test_ratio=args.test_ratio, seed=args.seed,
        )
        standard_vs_object_releasing_comparison = compare_standard_vs_object_releasing_brown(
            dataset, camera_config, pattern_config, test_ratio=args.test_ratio, seed=args.seed,
        )
        if not quiet:
            print("Object-Releasing calibration: Brown-Conrady result stored as an advanced result.")
            if object_releasing_result.warning_message:
                print(object_releasing_result.warning_message)
            if object_releasing_validation_result.success:
                print(f"Object-Releasing Hold-out RMSE: {object_releasing_validation_result.test_rms:.3f} px")
            else:
                print(f"Object-Releasing Hold-out: {object_releasing_validation_result.error_message}")
            print(format_standard_vs_object_releasing_table(standard_vs_object_releasing_comparison))
    compute_frame_quality_scores(dataset, pattern_config, image_size, use_reprojection=True)

    # 설계 문서 4번 - Overall Dataset Score. 개별 프레임 점수(방금 갱신됨) +
    # coverage(quality.analyze_dataset_quality가 이미 채워둔 dataset.coverage_grid/
    # diversity) + 중복 이미지 비율을 한 번에 요약한다.
    dup_ratio = (
        sum(len(g.image_ids) for g in duplicate_groups) / dataset.num_total
        if dataset.num_total > 0 else 0.0
    )
    dataset.quality_score = compute_dataset_quality_score(
        dataset,
        coverage_pct=coverage_percentage(dataset.coverage_grid) if dataset.coverage_grid else None,
        duplicate_ratio=dup_ratio,
    )
    if not quiet:
        print(format_dataset_quality_score(dataset.quality_score))

    if not any(r.success for r in results):
        print(f"오류: 선택된 모델 {len(results)}개 모두 캘리브레이션에 실패했습니다.", file=sys.stderr)
        for r in results:
            if r.error_message:
                print(f"  {r.model_name.value}: {r.error_message}", file=sys.stderr)
        return 2

    if not quiet:
        print(format_comparison_table(results))
        # 설계 문서 11/12번 - Reprojection Error 지표 확장 / Residual Distribution.
        # Train RMS 하나만 보지 않는다는 원칙을 코너 포인트 단위 오차에도 적용.
        for r in results:
            if r.success and r.residual_stats and r.residual_stats.n > 0:
                print()
                print(f"[{r.model_name.value}]")
                print(format_residual_stats(r.residual_stats))
                print(format_residual_boxplot(r.residual_stats))
                print(format_cdf(r.residual_stats))
                if r.spatial_error_map:
                    print()
                    print(format_spatial_error_map(r.spatial_error_map))
                if r.radial_bands and r.radial_bands.bins:
                    print()
                    print(format_radial_bands(r.radial_bands))
        # 설계 문서 8번 - Calibration 결과 sanity check. RMS가 낮다고 결과가
        # 물리적으로 정상인 건 아니므로, 이 자리에서 항상 함께 보여준다.
        sanity_checks = run_sanity_checks(results, camera_config, image_size)
        if any(c.issues for c in sanity_checks):
            print()
            print(format_sanity_checks(sanity_checks))

    return _validate_choose_and_export(
        args,
        dataset,
        camera_config,
        pattern_config,
        calibration_results,
        object_releasing_result=object_releasing_result,
        object_releasing_validation_result=object_releasing_validation_result,
        standard_vs_object_releasing_comparison=standard_vs_object_releasing_comparison,
    )


def _run_from_loaded_project(args) -> int:
    """--load-project로 저장된 상태를 이어서 쓴다. 원본 이미지 파일이 없어도
    (옮겨지거나 지워졌어도) 재계산/재-export가 가능하다 - 코너 좌표 등
    calibrateCamera에 필요한 데이터는 전부 프로젝트 파일 안에 이미 들어있고,
    export/report는 이미지 바이트를 다시 읽지 않기 때문이다.
    """
    quiet = args.quiet
    try:
        project, missing = load_project(args.load_project)
    except (OSError, ValueError, KeyError) as e:
        print(f"오류: 프로젝트 파일을 여는 데 실패했습니다: {e}", file=sys.stderr)
        return 1

    _log(quiet, f"프로젝트 불러옴: {project.project_name} ({project.dataset.num_total}장, "
                f"검출 {project.dataset.num_detected}장)")
    if missing:
        _log(quiet, f"  \u26a0 원본 이미지 {len(missing)}개를 찾을 수 없습니다 (경로가 바뀌었거나 삭제됨) - "
                     f"재계산/재-export는 이미지 없이도 가능합니다:")
        for m in missing[:5]:
            _log(quiet, f"    {m}")
        if len(missing) > 5:
            _log(quiet, f"    ... 외 {len(missing) - 5}개")

    if not project.calibration_results or not any(r.success for r in project.calibration_results.values()):
        print("오류: 불러온 프로젝트에 성공한 캘리브레이션 결과가 없습니다.", file=sys.stderr)
        return 2

    return _validate_choose_and_export(
        args, project.dataset, project.camera_config, project.pattern_config,
        project.calibration_results,
        object_releasing_result=project.object_releasing_result,
        object_releasing_validation_result=project.object_releasing_validation_result,
        standard_vs_object_releasing_comparison=project.standard_vs_object_releasing_comparison,
        precomputed_validation=project.validation_results,
        precomputed_scores=project.model_scores,
        precomputed_cross_dataset_results=project.cross_dataset_results,
    )


def _validate_choose_and_export(
    args, dataset, camera_config, pattern_config, calibration_results,
    object_releasing_result=None,
    object_releasing_validation_result=None,
    standard_vs_object_releasing_comparison=None,
    precomputed_validation=None,
    precomputed_scores=None,
    precomputed_cross_dataset_results=None,
) -> int:
    """설계 문서 15번 파이프라인의 5~9단계(Validation -> 추천 -> Outlier ->
    Final Result -> Export -> [옵션] 프로젝트 저장). 새로 계산한 경우와
    --load-project로 불러온 경우 둘 다 이 함수로 수렴한다.

    precomputed_validation/scores가 주어지면(불러온 프로젝트) 다시 계산하지
    않고 그대로 쓴다 - --outlier로 데이터셋이 바뀌면 그때는 다시 계산한다.
    """
    quiet = args.quiet

    if precomputed_validation is not None:
        # 이상치 제거를 하더라도, "이상치 제거를 어느 모델 기준으로 할지" 고르는
        # 초기 선택에는 저장된 값으로 충분하다 - 이상치 제거 이후 재검증은
        # 아래 --outlier 블록에서 어차피 다시 하므로, 여기서 미리 다시
        # 계산하는 건 중복 작업이다.
        validation_results = precomputed_validation
        scores = precomputed_scores or compute_model_scores(
            calibration_results, validation_results, use_rational_model=args.rational
        )
        _log(quiet, "저장된 Hold-out Validation/추천 결과를 재사용합니다 (재계산 안 함).")
    else:
        # --- 5. Hold-out Validation ---
        _log(quiet, "Hold-out Validation 중...")
        validation_results = validate_all_models(
            dataset, camera_config, pattern_config, test_ratio=args.test_ratio,
            seed=args.seed, use_rational_model=args.rational,
        )
        selected_models = set(_selected_model_types(args))
        validation_results = {
            m: v for m, v in validation_results.items() if m in selected_models
        }
        if not quiet:
            print(format_validation_table(validation_results))
            print()
            print(format_train_test_residual_comparison(validation_results))
            print()
            print(format_straightness_comparison(validation_results))

        # --- 6. 추천 ---
        scores = compute_model_scores(calibration_results, validation_results, use_rational_model=args.rational)
        _log(quiet, build_recommendation_message(scores, calibration_results, validation_results))

    selected_model_list = _selected_model_types(args)
    selected_model_set = set(selected_model_list)
    calibration_results = {
        m: r for m, r in calibration_results.items() if m in selected_model_set
    }
    validation_results = {
        m: v for m, v in validation_results.items() if m in selected_model_set
    }
    scores = [s for s in scores if s.model_name in selected_model_set]

    if args.model:
        chosen_model = _MODEL_BY_NAME[args.model]
        if chosen_model not in calibration_results or not calibration_results[chosen_model].success:
            print(f"오류: --model {args.model}을 지정했지만 해당 모델 캘리브레이션이 실패했습니다.", file=sys.stderr)
            return 2
    else:
        recommended = next((s for s in scores if s.is_recommended), None)
        if recommended is None:
            print("오류: 추천할 수 있는 모델이 없습니다 (모든 모델 실패).", file=sys.stderr)
            return 2
        chosen_model = recommended.model_name

    outlier_result = None
    corner_outlier_result = None
    scores_before_outlier = list(scores)

    # --- 7. (옵션) Outlier 제거 + 재계산 (프레임 단위/코너 단위, 둘 다 가능) ---
    if args.outlier or args.corner_outlier:
        _log(quiet, f"{chosen_model.value} 기준 이상치 탐지 및 재계산 중...")

        # 설계 문서 9번 - 아래 Hold-out Validation은 leak-safe해야 한다. 이
        # 시점의 dataset은 아직 어떤 outlier 판단도 반영 안 된 상태이므로,
        # 지금 복사본을 떠 두고 그 복사본 안에서만 "split -> train-only
        # outlier -> test 평가" 순서를 밟는다. 바로 아래에서 하는 전체
        # 데이터 기준 outlier 제거(최종 배포용 계산, 이 프로젝트의 의도적
        # 설계: 배포용 파라미터는 train/test 구분 없이 좋은 데이터를 전부
        # 쓴다)와는 완전히 분리해야 서로의 상태 변경이 섞이지 않는다.
        validation_dataset = copy.deepcopy(dataset)

        # --- 7-1. 최종 배포용 계산: 전체 데이터 기준 이상치 제거 ---
        ref_result = calibration_results[chosen_model]

        if args.outlier:
            ref_result, outlier_result = recalibrate_with_outlier_pruning(
                dataset, camera_config, chosen_model,
                max_iterations=args.max_iterations, use_rational_model=args.rational,
            )
            if outlier_result.removed_frame_ids:
                _log(quiet, f"  제외된 프레임: {outlier_result.removed_frame_ids}")
                if not quiet:
                    print(format_outlier_before_after(outlier_result))

        if args.corner_outlier:
            # 설계 문서 16번 - 프레임 전체가 아니라 문제 코너만 골라 뺀다.
            # --outlier와 함께 쓰면 이미 프레임 단위로 정리된 dataset 위에서
            # 한 번 더(코너 단위로 더 세밀하게) 정리한다.
            ref_result, corner_outlier_result = recalibrate_with_corner_outlier_pruning(
                dataset, camera_config, chosen_model,
                max_iterations=args.max_iterations, use_rational_model=args.rational,
            )
            if corner_outlier_result.removed_corners:
                _log(quiet, f"  제외된 코너: {corner_outlier_result.removed_corners}")
                if not quiet:
                    print(format_corner_outlier_before_after(corner_outlier_result))

        calibration_results[chosen_model] = ref_result
        # 나머지 두 모델도 정제된 데이터셋 기준으로 재계산해야 비교표/리포트가 일관됨
        results = run_all_models(
            dataset, camera_config, use_rational_model=args.rational,
            bootstrap_jobs=args.jobs, model_jobs=args.jobs,
            persistent_cache_dir=args.cache_dir,
            models=selected_model_list,
        )
        calibration_results = {r.model_name: r for r in results}
        calibration_results[chosen_model] = ref_result

        # --- 7-2. Leak-safe Hold-out Validation (평가 전용, 복사본에서만) ---
        _log(quiet, "Leak-safe Hold-out Validation 재계산 중 (Train-only Outlier Detection)...")
        train_ids, test_ids = split_train_test(
            validation_dataset, camera_config, args.test_ratio, args.seed
        )

        ref_validation = None
        if args.outlier:
            _, ref_val_outlier, ref_validation = recalibrate_train_with_outlier_pruning(
                validation_dataset, camera_config, pattern_config, chosen_model, train_ids, test_ids,
                max_iterations=args.max_iterations, use_rational_model=args.rational,
            )
            if ref_val_outlier.removed_frame_ids:
                _log(
                    quiet,
                    f"  [Validation 전용] Train에서만 제외된 프레임: {ref_val_outlier.removed_frame_ids} "
                    "(Test는 절대 건드리지 않음)",
                )

        if args.corner_outlier:
            _, ref_val_corner_outlier, ref_validation = recalibrate_train_with_corner_outlier_pruning(
                validation_dataset, camera_config, pattern_config, chosen_model, train_ids, test_ids,
                max_iterations=args.max_iterations, use_rational_model=args.rational,
            )
            if ref_val_corner_outlier.removed_corners:
                _log(
                    quiet,
                    f"  [Validation 전용] Train에서만 제외된 코너: {ref_val_corner_outlier.removed_corners} "
                    "(Test는 절대 건드리지 않음)",
                )

        if ref_validation is None:
            # 이 분기는 이론상 도달하지 않는다(바깥 if에서 outlier/corner_outlier
            # 둘 중 하나는 참이어야만 여기로 옴) - 그래도 방어적으로 처리.
            ref_validation = validate_holdout(
                validation_dataset, camera_config, pattern_config, chosen_model, train_ids, test_ids,
                use_rational_model=args.rational,
            )

        validation_results = {chosen_model: ref_validation}

        # chosen_model이 이미 위에서 이상치를 제거했으므로, dataset의 frame
        # status가 공유되어(같은 validation_dataset 객체) 나머지 두 모델도
        # 그 결과를 그대로 물려받는다 - 여기서 이상치 탐지를 또 하지 않는다.
        train_subset = _subset_dataset(validation_dataset, train_ids)
        pinhole_init = calibrate_pinhole(train_subset, camera_config)
        for m in selected_model_list:
            if m == chosen_model:
                continue
            fisheye_guess = pinhole_init if m == CameraModelType.FISHEYE else None
            validation_results[m] = validate_holdout(
                validation_dataset, camera_config, pattern_config, m, train_ids, test_ids,
                use_rational_model=args.rational, fisheye_initial_guess=fisheye_guess,
            )

        if not quiet:
            print(format_validation_table(validation_results))
            print()
            print(format_train_test_residual_comparison(validation_results))
            print()
            print(format_straightness_comparison(validation_results))

        scores = compute_model_scores(calibration_results, validation_results, use_rational_model=args.rational)

        # 설계 문서 17번 - "model ranking 변화" 기록. outlier 제거 전(scores_before_outlier)과
        # 후(scores) 두 순위표를 비교해 추천 모델이 바뀌었는지 보여준다.
        if not quiet:
            print()
            print(compare_model_rankings(scores_before_outlier, scores))

    # --- 8. Final Result ---
    coverage_pct = coverage_percentage(dataset.coverage_grid) if dataset.coverage_grid else None
    final_result = compute_final_result(
        chosen_model, calibration_results, validation_results,
        dataset_coverage_pct=coverage_pct, outlier_result=outlier_result,
        corner_outlier_result=corner_outlier_result, scores=scores,
        coverage_grid=dataset.coverage_grid,
        dataset_diversity=dataset.diversity,
    )
    _log(quiet, f"\n선택된 모델: {chosen_model.value}  종합 등급: {final_result.overall_grade.value}")

    # 설계 문서 8번 - 최종 선택된 모델은 outlier 제거 등으로 값이 바뀌었을 수
    # 있으므로 export 직전에 한 번 더 sanity check를 실행해 보여준다.
    image_size = infer_image_size(dataset, camera_config)
    final_sanity = run_sanity_checks([calibration_results[chosen_model]], camera_config, image_size)[0]
    if final_sanity.issues and not quiet:
        print()
        print(final_sanity.format())

    kfold_result = None
    repeated_kfold_result = None
    uncertainty_to_show = None
    cross_dataset_results = list(precomputed_cross_dataset_results or [])

    # --- 8-0. (옵션) Cross-Dataset Validation ---
    # Dataset A에서 얻은 intrinsic/distortion은 고정하고, Dataset B/C/...에서는
    # 각 프레임 pose만 다시 풀어 generalization을 확인한다.
    if args.cross_datasets:
        source_dataset_id = args.source_dataset_id or args.sensor_name or "Dataset A"
        try:
            target_datasets = _load_cross_dataset_targets(args, pattern_config, camera_config)
        except CliError as e:
            print(f"오류: {e}", file=sys.stderr)
            return 1
        new_cross_results = validate_cross_datasets(
            calibration_results, target_datasets, camera_config, pattern_config,
            source_dataset_id=source_dataset_id,
        )
        cross_dataset_results.extend(new_cross_results)
        if not quiet:
            print()
            print(format_cross_dataset_validation_table(new_cross_results))

    # --- 8-1. (옵션) K-Fold / Repeated K-Fold Cross Validation ---
    # 설계 문서 18/19번. Hold-out(1회 분할)보다 데이터를 더 알뜰하게 쓰고
    # "운 좋은/나쁜 분할" 효과를 줄인 검증이 필요할 때만 켠다 - 비용이 크므로
    # (모델 하나당 k번 또는 k*repeats번 전체 재계산) 기본은 꺼져 있다.
    if args.kfold:
        if args.kfold_repeats > 1:
            _log(quiet, f"Repeated {args.kfold}-Fold x {args.kfold_repeats} 계산 중 ({chosen_model.value})...")
            repeated_kfold_result = compute_repeated_kfold(
                dataset, camera_config, pattern_config, chosen_model,
                k=args.kfold, n_repeats=args.kfold_repeats, base_seed=args.seed,
                use_rational_model=args.rational, n_jobs=args.jobs,
            )
            if not quiet:
                print()
                print(format_repeated_kfold_result(repeated_kfold_result))
        else:
            _log(quiet, f"{args.kfold}-Fold Cross Validation 계산 중 ({chosen_model.value})...")
            kfold_result = compute_kfold_validation(
                dataset, camera_config, pattern_config, chosen_model,
                k=args.kfold, seed=args.seed, use_rational_model=args.rational, n_jobs=args.jobs,
            )
            if not quiet:
                print()
                print(format_kfold_result(kfold_result))

    # --- 8-2. (옵션) Repeatability 측정 ---
    # 설계 문서 40번. 프레임 순서를 바꿔가며 최종 모델을 반복 재계산해
    # fx/fy/cx/cy가 얼마나 일관되게 나오는지 확인한다.
    if args.repeatability:
        _log(quiet, f"Repeatability 측정 중 ({chosen_model.value}, {args.repeatability}회)...")
        repeatability_result = compute_repeatability(
            dataset, camera_config, chosen_model,
            n_runs=args.repeatability, seed=args.seed, use_rational_model=args.rational, n_jobs=args.jobs,
        )
        if not quiet:
            print()
            print(format_repeatability(repeatability_result))

    # --- 8-3. (옵션) Bootstrap 기반 Parameter 95% CI ---
    # 설계 문서 20/21/22번. Pinhole/Extended는 covariance 기반 std가 이미
    # param_uncertainty에 있으므로, 이 옵션은 그것과 교차검증할 독립적인
    # bootstrap 추정치를 추가로 계산한다. Fisheye는 애초에 bootstrap이
    # 유일한 방법이라(calibrate_fisheye의 estimate_uncertainty로 이미 계산됨)
    # 여기서 다시 계산하지 않고 기존 param_uncertainty를 그대로 보여준다.
    if args.bootstrap_ci:
        chosen_result = calibration_results[chosen_model]
        if chosen_model == CameraModelType.FISHEYE:
            uncertainty_to_show = chosen_result.param_uncertainty
        else:
            _log(quiet, f"Bootstrap 기반 Parameter CI 계산 중 ({chosen_model.value}, {args.n_bootstrap}회)...")
            frames_for_bootstrap, obj_pts, img_pts = collect_calibration_inputs(dataset)
            uncertainty_to_show = compute_parameter_bootstrap(
                obj_pts, img_pts, image_size, chosen_model,
                chosen_result.camera_matrix, chosen_result.distortion,
                flags=(cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
                       if chosen_model == CameraModelType.PINHOLE else 0) | cv2.CALIB_USE_INTRINSIC_GUESS,
                n_bootstrap=args.n_bootstrap, rng_seed=args.seed, n_jobs=args.jobs,
            )
            # 계산 결과를 CalibrationResult에도 남겨서 export/report에 그대로 반영되게 한다.
            chosen_result.param_uncertainty_bootstrap = uncertainty_to_show
        if not quiet:
            print()
            print(format_parameter_uncertainty(uncertainty_to_show))

    # --- 9. Export ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, str] = {}

    export_targets = set(args.export)
    if "opencv" in export_targets:
        p = export_opencv_yaml(calibration_results[chosen_model], camera_config, pattern_config, str(out_dir / "camera.yaml"))
        exported["opencv_yaml"] = p
        _log(quiet, f"저장: {p}")
    if "ros" in export_targets:
        p = export_ros_camera_info(calibration_results[chosen_model], camera_config, str(out_dir / "camera_info.yaml"))
        exported["ros_yaml"] = p
        _log(quiet, f"저장: {p}")
    if "report" in export_targets:
        p = export_html_report(
            args.sensor_name or "camera_calibrator", camera_config, pattern_config, dataset,
            calibration_results, validation_results, final_result, str(out_dir / "report.html"),
            cross_dataset_results=cross_dataset_results,
            kfold_result=kfold_result, repeated_kfold_result=repeated_kfold_result,
        )
        exported["report_html"] = p
        _log(quiet, f"저장: {p}")
    if "json" in export_targets:
        p = export_json(
            camera_config, pattern_config, dataset, calibration_results, validation_results,
            chosen_model, str(out_dir / "calibration.json"),
            final_result=final_result, model_scores=scores,
            cross_dataset_results=cross_dataset_results,
            kfold_result=kfold_result, repeated_kfold_result=repeated_kfold_result,
        )
        exported["json"] = p
        _log(quiet, f"저장: {p}")
    if "csv" in export_targets:
        p = export_csv(dataset, str(out_dir / "dataset.csv"))
        exported["csv"] = p

    if "kalibr" in export_targets:
        try:
            p = export_kalibr_target_yaml(pattern_config, str(out_dir / "kalibr_aprilgrid.yaml"))
        except ValueError as e:
            raise CliError(str(e)) from e
        exported["kalibr_target_yaml"] = p
        if args.bag and args.topic:
            command = build_kalibr_camera_calibration_command(
                bag_path=args.bag,
                topic=args.topic,
                target_yaml_path=p,
                camera_model=args.kalibr_camera_model,
            )
            command_path = out_dir / "kalibr_camera_calibration_command.txt"
            command_path.write_text(command + "\n", encoding="utf-8")
            exported["kalibr_command"] = str(command_path)
        _log(quiet, f"저장: {p}")

    if args.save_project:
        project = CalibrationProject(
            project_name=args.sensor_name or "camera_calibrator",
            camera_config=camera_config, pattern_config=pattern_config, dataset=dataset,
            calibration_results=calibration_results, object_releasing_result=object_releasing_result,
            object_releasing_validation_result=object_releasing_validation_result,
            standard_vs_object_releasing_comparison=standard_vs_object_releasing_comparison,
            validation_results=validation_results,
            cross_dataset_results=cross_dataset_results,
            model_scores=scores, outlier_result=outlier_result, final_result=final_result,
        )
        saved_path = save_project(project, args.save_project)
        exported["project_ccproj"] = saved_path
        _log(quiet, f"저장: {saved_path} (나중에 --load-project로 이어서 쓸 수 있음)")

    if args.json_summary:
        cal = calibration_results[chosen_model]
        summary = {
            "chosen_model": chosen_model.value,
            "overall_grade": final_result.overall_grade.value,
            "success": cal.success,
            "rms_error": cal.rms_error,
            "test_rms": (validation_results.get(chosen_model).test_rms if validation_results.get(chosen_model) else None),
            "num_images_total": dataset.num_total,
            "num_images_detected": dataset.num_detected,
            "num_images_used": dataset.num_enabled,
            "coverage_pct": coverage_pct,
            "diagnostic": bool(args.diagnostic),
            "cross_validation_k": args.kfold,
            "kfold_mean_test_rms": (
                kfold_result.mean_test_rms if kfold_result else
                repeated_kfold_result.mean_test_rms if repeated_kfold_result else None
            ),
            "bootstrap_requested": args.n_bootstrap if args.bootstrap_ci else None,
            "bootstrap_success": (
                uncertainty_to_show.n_bootstrap_success if uncertainty_to_show else None
            ),
            "cross_dataset_validation": [
                {
                    "source_dataset_id": r.source_dataset_id,
                    "target_dataset_id": r.target_dataset_id,
                    "model": r.model_name.value,
                    "success": r.success,
                    "train_rms": r.train_rms,
                    "test_rms": r.test_rms,
                    "test_p95": r.test_p95,
                    "generalization_gap": r.generalization_gap,
                    "num_test_frames": r.num_test_frames,
                    "error_message": r.error_message,
                }
                for r in cross_dataset_results
            ],
            "outlier_removed": outlier_result.removed_frame_ids if outlier_result else [],
            "corner_outlier_removed": corner_outlier_result.removed_corners if corner_outlier_result else {},
            "exported_files": exported,
        }
        _write_json_summary(args.json_summary, summary)
        _log(quiet, f"저장: {args.json_summary}")

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="camera_calibrator",
        description="헤드리스 카메라 캘리브레이션 CLI (UI 없이 배치/CI 환경에서 실행).",
    )

    src = p.add_argument_group("입력 (이미지 또는 rosbag 또는 저장된 프로젝트 중 하나)")
    src.add_argument(
        "--config", metavar="PATH",
        help="패턴/카메라/파이프라인 옵션을 담은 .yaml/.yml/.json 파일. 같은 옵션을 "
             "커맨드라인에서 또 주면 커맨드라인 쪽이 우선한다 (반복 실행하는 카메라 "
             "설정을 파일로 고정해두고, 그때그때 바뀌는 값만 커맨드라인으로 덮어쓰는 "
             "용도). 예: --config camera.yaml --images ./photos",
    )
    src.add_argument("--images", nargs="+", help="이미지 파일/디렉토리/glob 패턴 (여러 개 가능)")
    src.add_argument(
        "--input", dest="images", nargs="+",
        help="문서 예시 호환 alias: --images와 동일. 예: --input ./photos",
    )
    src.add_argument("--bag", help="ROS1(.bag)/ROS2(.db3, .mcap) 파일 경로 (--images 대신 사용)")
    src.add_argument("--topic", help="--bag 사용 시 추출할 이미지 토픽")
    src.add_argument("--bag-interval", type=float, default=0.5, help="bag 프레임 샘플링 최소 간격(초), 기본 0.5")
    src.add_argument("--list-topics", metavar="BAG_PATH", help="이 bag의 이미지 토픽 목록만 출력하고 종료")
    src.add_argument(
        "--load-project", metavar="CCPROJ_PATH",
        help="저장된 .ccproj 프로젝트를 불러와 이어서 진행 (검출/모델 계산을 다시 안 함, "
             "--images/--bag/패턴 옵션과 함께 쓸 수 없음). 원본 이미지가 없어져도 동작함.",
    )
    src.add_argument(
        "--reference", metavar="CALIBRATION_PATH",
        help="Benchmark 전용: 기존/기준 calibration 파일(JSON/OpenCV YAML/ROS CameraInfo/Kalibr camchain).",
    )
    src.add_argument(
        "--candidate", metavar="CALIBRATION_PATH",
        help="Benchmark 전용: 비교할 새 calibration 파일(JSON/OpenCV YAML/ROS CameraInfo/Kalibr camchain).",
    )
    src.add_argument(
        "--validation-dataset", nargs="+", metavar="IMAGE_OR_DIR",
        help="Benchmark 전용: Reference/Candidate를 동일하게 평가할 validation 이미지/디렉토리/glob.",
    )
    src.add_argument(
        "--reference-camera-key", metavar="CAM_KEY",
        help="Benchmark 전용: Kalibr camchain에서 Reference로 읽을 cam key(cam0 등).",
    )
    src.add_argument(
        "--candidate-camera-key", metavar="CAM_KEY",
        help="Benchmark 전용: Kalibr camchain에서 Candidate로 읽을 cam key(cam0 등).",
    )

    pat = p.add_argument_group("패턴")
    pat.add_argument(
        "--pattern", default="charuco",
        help="패턴 타입: charuco, chessboard, circle_grid/circles, apriltag_grid/aprilgrid "
             "(기본값 charuco, AprilGrid는 DICT_APRILTAG_* dictionary 필요)",
    )
    pat.add_argument("--squares-x", type=int, help="가로 사각형 개수")
    pat.add_argument("--squares-y", type=int, help="세로 사각형 개수")
    pat.add_argument("--square-size", type=float, help="사각형 한 칸 크기 (미터)")
    pat.add_argument("--marker-size", type=float, help="마커 크기 (미터, ChArUco/AprilGrid 필수)")
    pat.add_argument("--dictionary", default="DICT_5X5_100", help="ArUco dictionary 이름, 기본 DICT_5X5_100")
    pat.add_argument(
        "--circle-grid-type",
        type=CircleGridType,
        choices=list(CircleGridType),
        default=CircleGridType.SYMMETRIC,
        help="Circle Grid 종류: symmetric 또는 asymmetric",
    )
    pat.add_argument(
        "--aprilgrid-variant",
        type=AprilGridVariant,
        choices=list(AprilGridVariant),
        default=AprilGridVariant.OPENCV_APRILTAG3,
        help="AprilGrid variant: opencv_apriltag3 또는 kalibr",
    )

    cam = p.add_argument_group("카메라")
    cam.add_argument("--width", type=int, help="이미지 가로 해상도 (생략 시 첫 이미지에서 자동 유추)")
    cam.add_argument("--height", type=int, help="이미지 세로 해상도 (생략 시 첫 이미지에서 자동 유추)")
    cam.add_argument("--sensor-name", help="리포트/카메라 이름표에 쓸 라벨")

    pipe = p.add_argument_group("파이프라인")
    pipe.add_argument(
        "--diagnostic", action="store_true",
        help="종합 진단 preset. 별도 지정이 없으면 --cross-validation 5, --bootstrap 100을 켜고 "
             "report/json/csv export를 포함한다.",
    )
    pipe.add_argument("--test-ratio", type=float, default=0.25, help="Hold-out validation test 비율, 기본 0.25")
    pipe.add_argument("--seed", type=int, default=42, help="train/test 분할 시드, 기본 42")
    pipe.add_argument("--rational", action="store_true", help="Rational 모델에서 k1~k6,p1,p2 8계수 사용")
    pipe.add_argument(
        "--calibration-method",
        type=CalibrationMethod,
        choices=list(CalibrationMethod),
        default=CalibrationMethod.STANDARD,
        help="Calibration method: standard 또는 object_releasing",
    )
    pipe.add_argument("--outlier", action="store_true", help="추천된 모델 기준으로 이상치 탐지+재계산까지 수행 (프레임 단위)")
    pipe.add_argument(
        "--corner-outlier", action="store_true",
        help="프레임 전체가 아니라 개별 코너 단위로 이상치를 탐지+제거 (--outlier와 함께 쓸 수 있음, 그 뒤에 적용됨)",
    )
    pipe.add_argument("--max-iterations", type=int, default=3, help="이상치 제거 최대 반복 횟수, 기본 3")
    pipe.add_argument(
        "--model", choices=_MODEL_CLI_CHOICES, default=None,
        help="자동 추천 대신 이 모델을 강제로 최종 선택 (export/report 기준)",
    )
    pipe.add_argument(
        "--models", nargs="+", choices=_MODEL_CLI_CHOICES, default=None,
        help="문서 예시 호환: 계산/검증할 모델 목록. 예: --models pinhole extended fisheye",
    )
    pipe.add_argument(
        "--validate", action="store_true",
        help="문서 예시 호환 alias. 이 CLI는 기본적으로 Hold-out validation을 항상 수행하므로 no-op.",
    )
    pipe.add_argument(
        "--kfold", type=int, default=None, metavar="K",
        help="설계 문서 18번 - Hold-out 대신(추가로) K-Fold Cross Validation 수행 (예: --kfold 5)",
    )
    pipe.add_argument(
        "--cross-validation", type=int, default=None, metavar="K",
        help="문서형 alias: K-Fold Cross Validation 수행 (예: --cross-validation 5). --kfold와 동일.",
    )
    pipe.add_argument(
        "--cross-dataset", "--test-dataset",
        dest="cross_datasets",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Dataset A에서 학습한 calibration을 외부 Dataset B/C/...에 고정 평가한다. "
             "여러 번 지정 가능하며 NAME=PATH 또는 PATH를 받는다. 예: "
             "--cross-dataset B=./dataset_b --cross-dataset C=./dataset_c",
    )
    pipe.add_argument(
        "--source-dataset-id",
        help="Cross-dataset validation 표/report에 표시할 source Dataset A 라벨 "
             "(기본: --sensor-name 또는 'Dataset A').",
    )
    pipe.add_argument(
        "--kfold-repeats", type=int, default=1, metavar="N",
        help="설계 문서 19번 - K-Fold를 N번 반복(다른 분할로) - --kfold와 함께 사용, 기본 1(반복 없음)",
    )
    pipe.add_argument(
        "--repeatability", type=int, default=None, metavar="N",
        help="설계 문서 40번 - 최종 선택된 모델로 데이터 순서를 N번 바꿔가며 재계산해 반복 재현성 측정",
    )
    pipe.add_argument(
        "--bootstrap-ci", action="store_true",
        help="설계 문서 20/22번 - 최종 선택된 모델의 fx/fy/cx/cy에 대해 bootstrap 기반 95%% CI를 추가로 계산 (비용이 큼)",
    )
    pipe.add_argument(
        "--bootstrap", type=int, default=None, metavar="N",
        help="문서형 alias: bootstrap 기반 Parameter CI를 N회 재표본으로 계산 (예: --bootstrap 100).",
    )
    pipe.add_argument(
        "--n-bootstrap", type=int, default=20, metavar="N",
        help="--bootstrap-ci 또는 Fisheye 기본 불확실성 추정에 쓸 재표본 횟수, 기본 20",
    )

    out = p.add_argument_group("출력")
    out.add_argument("--output-dir", default="./calibration_output", help="결과 저장 폴더, 기본 ./calibration_output")
    out.add_argument(
        "--export", nargs="+", default=None,
        choices=["opencv", "ros", "report", "json", "csv", "kalibr"],
        help="내보낼 형식 (기본: opencv/ros/report - json/csv/kalibr는 명시해야 포함됨)",
    )
    out.add_argument(
        "--kalibr-camera-model", default="pinhole-radtan",
        choices=["pinhole-radtan", "pinhole-equi", "omni-radtan", "ds-none", "eucm-none"],
        help="--export kalibr가 command hint를 만들 때 사용할 Kalibr camera model",
    )
    out.add_argument("--report", action="store_true", help="문서 예시 호환 alias: --export report를 추가한다.")
    out.add_argument(
        "--benchmark-report", action="store_true",
        help="Benchmark 전용: output-dir에 benchmark_report.html을 함께 저장한다 "
             "(benchmark_result.json은 항상 저장).",
    )
    out.add_argument("--json-summary", help="기계가 읽기 좋은 JSON 요약을 이 경로에 저장 (CI 스크립팅용)")
    out.add_argument(
        "--save-project", metavar="CCPROJ_PATH",
        help="계산이 끝난 뒤 전체 상태(데이터셋, Standard 4모델 결과, 검증, 추천, "
             "Object-Releasing 결과/검증/비교)를 .ccproj 파일로 저장. "
             "나중에 --load-project로 다시 불러와 이어서 쓸 수 있음.",
    )
    out.add_argument(
        "--jobs", type=int, default=1, metavar="N",
        help="이미지 검출, 모델 계산 일부, optional heavy analysis(K-fold/repeatability/bootstrap)를 N개 worker로 병렬화 "
             "(기본 1=순차 처리, 기존과 동일 동작). 0을 주면 CPU 코어 수만큼 자동 사용.",
    )
    out.add_argument(
        "--cache-dir", metavar="PATH",
        help="Persistent result cache 폴더. 지정하면 같은 dataset/config/model 옵션의 모델 계산 결과를 "
             "디스크에 저장하고 다음 실행에서 재사용한다.",
    )
    out.add_argument("--quiet", action="store_true", help="진행상황 출력을 최소화 (콘솔 로그도 ERROR만 표시)")
    out.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="진단 로그 상세도를 높임. -v: INFO, -vv: DEBUG (여러 번 줄 수 있음). "
             "실시간 ROS 구독처럼 사용자 환경에서만 재현되는 문제를 진단할 때 유용.",
    )
    out.add_argument(
        "--log-file", metavar="PATH",
        help="진단 로그(DEBUG 레벨 전체)를 이 파일에 남김. --quiet/--verbose와 무관하게 "
             "항상 전체 상세도로 기록됨 - 버그 재현 후 이 파일을 첨부하면 원인 파악이 빨라짐.",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()

    # --config는 2단계로 처리한다: 먼저 (다른 옵션은 몰라도 되니) --config 값만
    # 알아내고, 그 파일 내용을 parser의 기본값으로 얹은 뒤 실제 파싱을 한다.
    # 이러면 "커맨드라인에서 명시적으로 준 옵션이 항상 이긴다"는 argparse의
    # 표준 동작이 config 파일에도 자연스럽게 적용된다 (set_defaults()는 기본값만
    # 바꿀 뿐, 이미 명시적으로 준 값을 덮어쓰지 않음).
    pre_args, _ = parser.parse_known_args(argv)
    if pre_args.config:
        known_dests = {action.dest for action in parser._actions}
        try:
            config_values = _load_config_file(pre_args.config, known_dests)
        except CliError as e:
            print(f"오류: {e}", file=sys.stderr)
            return 1
        parser.set_defaults(**config_values)

    args = _normalize_cli_args(parser.parse_args(argv), parser)

    setup_logging(verbosity=args.verbose, quiet=args.quiet, log_file=args.log_file)
    logger.debug("CLI 시작: argv=%s", argv if argv is not None else sys.argv[1:])
    if pre_args.config:
        logger.info("--config 적용됨: %s", pre_args.config)

    if args.list_topics:
        return _list_bag_topics(args.list_topics)

    if args.benchmark_mode:
        for required in ("squares_x", "squares_y", "square_size"):
            if getattr(args, required) is None:
                parser.error(f"--{required.replace('_', '-')}는 필수입니다.")
    elif args.load_project:
        if args.images or args.bag:
            parser.error("--load-project는 --images/--bag와 함께 쓸 수 없습니다.")
        # 패턴/카메라 옵션은 불러온 프로젝트에 이미 저장돼 있으므로 필요 없음.
    else:
        if not args.images and not args.bag:
            parser.error("--images, --bag, --load-project 중 하나는 반드시 지정해야 합니다.")
        if args.images and args.bag:
            parser.error("--images와 --bag는 동시에 쓸 수 없습니다.")
        for required in ("squares_x", "squares_y", "square_size"):
            if getattr(args, required) is None:
                parser.error(f"--{required.replace('_', '-')}는 필수입니다.")

    try:
        if args.benchmark_mode:
            return _run_benchmark_cli(args)
        return run_pipeline(args)
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨 (KeyboardInterrupt)")
        print("\n중단됨.", file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001
        # logger.debug(..., exc_info=True): DEBUG 레벨이라 콘솔에는(별도로 -vv를
        # 주지 않는 한) 안 뜨지만, --log-file을 지정했다면 파일 핸들러는 항상
        # DEBUG 레벨이라 전체 스택트레이스가 거기엔 남는다. 콘솔 쪽 traceback
        # 표시 여부는 기존과 동일하게 --quiet로 결정한다.
        logger.debug("파이프라인 실행 중 처리되지 않은 예외 발생", exc_info=True)
        print(f"예상하지 못한 오류: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())

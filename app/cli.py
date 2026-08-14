"""
camera_calibrator.app.cli
==============================

헤드리스(UI 없는) CLI 진입점. CI/서버/배치 처리용.

    python -m app.cli --images ./photos --squares-x 7 --squares-y 5 \
        --square-size 0.04 --marker-size 0.03 --output-dir ./out

전체 흐름은 ui/worker.py의 PipelineWorker/OutlierPruneWorker와 동일한 순서
(Detection -> Quality -> 3모델 -> Validation -> 추천 -> [옵션] Outlier 제거
-> Final Result -> Export)를 따르되, Qt 시그널 대신 stdout에 진행상황을
찍고 예외는 그대로 위로 던지지 않고 종료 코드로 변환한다.

종료 코드:
    0 = 성공 (export 파일까지 다 만들어짐)
    1 = 입력 문제 (이미지 없음, 인자 오류 등)
    2 = 검출은 됐지만 모든 모델의 캘리브레이션이 실패함
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import logging
import sys
from pathlib import Path

import cv2

from calibration.compare import run_all_models, format_comparison_table
from calibration.detector import detect_dataset, summarize_dataset
from calibration.frame_quality import compute_frame_quality_scores
from calibration.log_utils import setup_logging
from calibration.models.common import infer_image_size
from calibration.outlier import recalibrate_with_outlier_pruning
from calibration.quality import analyze_dataset_quality, coverage_percentage
from calibration.recommender import (
    build_recommendation_message,
    compute_final_result,
    compute_model_scores,
)
from calibration.types import (
    CalibrationProject,
    CameraConfig,
    CameraModelType,
    PatternConfig,
    PatternType,
)
from calibration.project_io import load_project, save_project
from calibration.validation import format_validation_table, validate_all_models
from export.opencv import export_opencv_yaml
from export.report import export_html_report
from export.ros import export_ros_camera_info
from export.json_export import export_json
from export.csv_export import export_csv

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

_MODEL_BY_NAME = {
    "pinhole": CameraModelType.PINHOLE,
    "extended_pinhole": CameraModelType.EXTENDED_PINHOLE,
    "fisheye": CameraModelType.FISHEYE,
}


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


def _resolve_image_paths(args) -> list[str]:
    """--images(파일/glob/디렉토리 혼용)와 --bag 중 하나로 최종 이미지 경로 리스트를 만든다."""
    if args.bag:
        return _extract_from_bag(args)

    if not args.images:
        raise CliError("--images 또는 --bag 중 하나는 반드시 지정해야 합니다.")

    paths: list[str] = []
    for item in args.images:
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
        raise CliError("주어진 --images 경로에서 이미지를 찾지 못했습니다 (jpg/jpeg/png/bmp).")
    return unique


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
    try:
        pattern_type = PatternType(args.pattern)
    except ValueError:
        raise CliError(f"알 수 없는 패턴 타입: {args.pattern}")

    if pattern_type not in (PatternType.CHARUCO, PatternType.CHESSBOARD):
        raise CliError(
            f"현재는 charuco, chessboard 패턴만 실제로 구현되어 있습니다 (입력: {args.pattern})."
        )
    if pattern_type == PatternType.CHARUCO and args.marker_size is None:
        raise CliError("ChArUco 패턴은 --marker-size가 반드시 필요합니다.")

    return PatternConfig(
        type=pattern_type,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_size=args.square_size,
        marker_size=args.marker_size if pattern_type == PatternType.CHARUCO else None,
        dictionary=args.dictionary if pattern_type == PatternType.CHARUCO else None,
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

    _log(quiet, f"이미지 {len(image_paths)}장, 해상도 {width}x{height}")

    # --- 2. Detection ---
    _log(quiet, f"{pattern_config.type.value} 패턴 검출 중...")
    dataset = detect_dataset(
        image_paths, pattern_config,
        parallel=args.jobs != 1,
        max_workers=None if args.jobs in (0, 1) else args.jobs,
    )
    _log(quiet, summarize_dataset(dataset))
    if dataset.num_detected == 0:
        print("오류: 어떤 이미지에서도 ChArUco 패턴이 검출되지 않았습니다.", file=sys.stderr)
        return 1

    # --- 3. Quality Gate (Coverage/Diversity) ---
    _log(quiet, "Coverage Map / 데이터셋 품질 분석 중...")
    warnings = analyze_dataset_quality(dataset, camera_config)
    for w in warnings:
        _log(quiet, f"  \u26a0 {w}")

    image_size = infer_image_size(dataset, camera_config)
    compute_frame_quality_scores(dataset, pattern_config, image_size, use_reprojection=False)

    # --- 4. 3모델 계산 ---
    _log(quiet, "Pinhole / Extended Pinhole / Fisheye 계산 중...")
    results = run_all_models(dataset, camera_config, use_rational_model=args.rational)
    calibration_results = {r.model_name: r for r in results}
    compute_frame_quality_scores(dataset, pattern_config, image_size, use_reprojection=True)

    if not any(r.success for r in results):
        print("오류: 3개 모델 모두 캘리브레이션에 실패했습니다.", file=sys.stderr)
        for r in results:
            if r.error_message:
                print(f"  {r.model_name.value}: {r.error_message}", file=sys.stderr)
        return 2

    if not quiet:
        print(format_comparison_table(results))

    return _validate_choose_and_export(
        args, dataset, camera_config, pattern_config, calibration_results
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
        precomputed_validation=project.validation_results,
        precomputed_scores=project.model_scores,
    )


def _validate_choose_and_export(
    args, dataset, camera_config, pattern_config, calibration_results,
    precomputed_validation=None, precomputed_scores=None,
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
        if not quiet:
            print(format_validation_table(validation_results))

        # --- 6. 추천 ---
        scores = compute_model_scores(calibration_results, validation_results, use_rational_model=args.rational)
        _log(quiet, build_recommendation_message(scores, calibration_results, validation_results))

    if args.model:
        chosen_model = _MODEL_BY_NAME[args.model]
        if not calibration_results[chosen_model].success:
            print(f"오류: --model {args.model}을 지정했지만 해당 모델 캘리브레이션이 실패했습니다.", file=sys.stderr)
            return 2
    else:
        recommended = next((s for s in scores if s.is_recommended), None)
        if recommended is None:
            print("오류: 추천할 수 있는 모델이 없습니다 (모든 모델 실패).", file=sys.stderr)
            return 2
        chosen_model = recommended.model_name

    outlier_result = None

    # --- 7. (옵션) Outlier 제거 + 재계산 ---
    if args.outlier:
        _log(quiet, f"{chosen_model.value} 기준 이상치 탐지 및 재계산 중...")
        ref_result, outlier_result = recalibrate_with_outlier_pruning(
            dataset, camera_config, chosen_model,
            max_iterations=args.max_iterations, use_rational_model=args.rational,
        )
        calibration_results[chosen_model] = ref_result
        if outlier_result.removed_frame_ids:
            _log(quiet, f"  제외된 프레임: {outlier_result.removed_frame_ids}")
        # 나머지 두 모델도 정제된 데이터셋 기준으로 재계산해야 비교표/리포트가 일관됨
        results = run_all_models(dataset, camera_config, use_rational_model=args.rational)
        calibration_results = {r.model_name: r for r in results}
        calibration_results[chosen_model] = ref_result
        validation_results = validate_all_models(
            dataset, camera_config, pattern_config, test_ratio=args.test_ratio,
            seed=args.seed, use_rational_model=args.rational,
        )
        scores = compute_model_scores(calibration_results, validation_results, use_rational_model=args.rational)

    # --- 8. Final Result ---
    coverage_pct = coverage_percentage(dataset.coverage_grid) if dataset.coverage_grid else None
    final_result = compute_final_result(
        chosen_model, calibration_results, validation_results,
        dataset_coverage_pct=coverage_pct, outlier_result=outlier_result, scores=scores,
    )
    _log(quiet, f"\n선택된 모델: {chosen_model.value}  종합 등급: {final_result.overall_grade.value}")

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
        )
        exported["report_html"] = p
        _log(quiet, f"저장: {p}")
    if "json" in export_targets:
        p = export_json(
            camera_config, pattern_config, dataset, calibration_results, validation_results,
            chosen_model, str(out_dir / "calibration.json"),
            final_result=final_result, model_scores=scores,
        )
        exported["json"] = p
        _log(quiet, f"저장: {p}")
    if "csv" in export_targets:
        p = export_csv(dataset, str(out_dir / "dataset.csv"))
        exported["csv"] = p
        _log(quiet, f"저장: {p}")

    if args.save_project:
        project = CalibrationProject(
            project_name=args.sensor_name or "camera_calibrator",
            camera_config=camera_config, pattern_config=pattern_config, dataset=dataset,
            calibration_results=calibration_results, validation_results=validation_results,
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
            "outlier_removed": outlier_result.removed_frame_ids if outlier_result else [],
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
    src.add_argument("--bag", help="ROS1(.bag)/ROS2(.db3, .mcap) 파일 경로 (--images 대신 사용)")
    src.add_argument("--topic", help="--bag 사용 시 추출할 이미지 토픽")
    src.add_argument("--bag-interval", type=float, default=0.5, help="bag 프레임 샘플링 최소 간격(초), 기본 0.5")
    src.add_argument("--list-topics", metavar="BAG_PATH", help="이 bag의 이미지 토픽 목록만 출력하고 종료")
    src.add_argument(
        "--load-project", metavar="CCPROJ_PATH",
        help="저장된 .ccproj 프로젝트를 불러와 이어서 진행 (검출/3모델 계산을 다시 안 함, "
             "--images/--bag/패턴 옵션과 함께 쓸 수 없음). 원본 이미지가 없어져도 동작함.",
    )

    pat = p.add_argument_group("패턴")
    pat.add_argument(
        "--pattern", default="charuco",
        help="패턴 타입: charuco 또는 chessboard (기본값 charuco, ChArUco를 권장 - "
             "chessboard는 보드 전체가 다 보여야 하고 대칭이라 방향 모호성 위험이 있음)",
    )
    pat.add_argument("--squares-x", type=int, help="가로 사각형 개수")
    pat.add_argument("--squares-y", type=int, help="세로 사각형 개수")
    pat.add_argument("--square-size", type=float, help="사각형 한 칸 크기 (미터)")
    pat.add_argument("--marker-size", type=float, help="ArUco 마커 크기 (미터, ChArUco 필수)")
    pat.add_argument("--dictionary", default="DICT_5X5_100", help="ArUco dictionary 이름, 기본 DICT_5X5_100")

    cam = p.add_argument_group("카메라")
    cam.add_argument("--width", type=int, help="이미지 가로 해상도 (생략 시 첫 이미지에서 자동 유추)")
    cam.add_argument("--height", type=int, help="이미지 세로 해상도 (생략 시 첫 이미지에서 자동 유추)")
    cam.add_argument("--sensor-name", help="리포트/카메라 이름표에 쓸 라벨")

    pipe = p.add_argument_group("파이프라인")
    pipe.add_argument("--test-ratio", type=float, default=0.25, help="Hold-out validation test 비율, 기본 0.25")
    pipe.add_argument("--seed", type=int, default=42, help="train/test 분할 시드, 기본 42")
    pipe.add_argument("--rational", action="store_true", help="Extended Pinhole에 rational model(k1~k6, 8계수) 사용")
    pipe.add_argument("--outlier", action="store_true", help="추천된 모델 기준으로 이상치 탐지+재계산까지 수행")
    pipe.add_argument("--max-iterations", type=int, default=3, help="이상치 제거 최대 반복 횟수, 기본 3")
    pipe.add_argument(
        "--model", choices=sorted(_MODEL_BY_NAME), default=None,
        help="자동 추천 대신 이 모델을 강제로 최종 선택 (export/report 기준)",
    )

    out = p.add_argument_group("출력")
    out.add_argument("--output-dir", default="./calibration_output", help="결과 저장 폴더, 기본 ./calibration_output")
    out.add_argument(
        "--export", nargs="+", default=["opencv", "ros", "report"],
        choices=["opencv", "ros", "report", "json", "csv"], help="내보낼 형식 (기본: opencv/ros/report - json/csv는 명시해야 포함됨)",
    )
    out.add_argument("--json-summary", help="기계가 읽기 좋은 JSON 요약을 이 경로에 저장 (CI 스크립팅용)")
    out.add_argument(
        "--save-project", metavar="CCPROJ_PATH",
        help="계산이 끝난 뒤 전체 상태(데이터셋, 3모델 결과, 검증, 추천)를 .ccproj 파일로 저장. "
             "나중에 --load-project로 다시 불러와 이어서 쓸 수 있음.",
    )
    out.add_argument(
        "--jobs", type=int, default=1, metavar="N",
        help="이미지 검출을 N개 프로세스로 병렬화 (기본 1=순차 처리, 기존과 동일 동작). "
             "0을 주면 CPU 코어 수만큼 자동 사용. 이미지 수백 장 이상일 때 효과가 큼 "
             "(코어당 이미지 몇 십 장 미만이면 프로세스 생성 비용 때문에 오히려 느려질 수 있음).",
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

    args = parser.parse_args(argv)

    setup_logging(verbosity=args.verbose, quiet=args.quiet, log_file=args.log_file)
    logger.debug("CLI 시작: argv=%s", argv if argv is not None else sys.argv[1:])
    if pre_args.config:
        logger.info("--config 적용됨: %s", pre_args.config)

    if args.jobs < 0:
        parser.error("--jobs는 0 이상이어야 합니다 (0=자동, 1=순차, N=N개 프로세스).")

    if args.list_topics:
        return _list_bag_topics(args.list_topics)

    if args.load_project:
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

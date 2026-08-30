"""
tests/test_cli.py
======================

app/cli.py - 헤드리스 CLI 진입점. 실제로 subprocess처럼 main()을 호출해서
종료 코드/출력 파일까지 확인한다 (단위 테스트 수준을 넘어 "CLI로서 실제로
동작하는가"를 검증하는 것이 목적).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from calibration.calibration_io import StandardCalibration, export_standard_json
from calibration.types import CameraModelType
from app.cli import _normalize_cli_args, build_arg_parser, main

pytestmark = pytest.mark.slow

IMG_W, IMG_H = 1920, 1080
TRUE_K = np.array([[1100.0, 0.0, IMG_W / 2], [0.0, 1100.0, IMG_H / 2], [0.0, 0.0, 1.0]])
TRUE_D = np.array([-0.28, 0.10, 0.0, 0.0, 0.0])


def _base_pattern_args() -> list[str]:
    return [
        "--squares-x", "7", "--squares-y", "5",
        "--square-size", "0.04", "--marker-size", "0.03",
    ]


def _render_and_save_chessboard_images(out_dir, n=8, seed=1):
    import cv2
    import numpy as np

    squares_x, squares_y, square_px = 7, 5, 100
    base = np.full((squares_y * square_px, squares_x * square_px), 255, dtype=np.uint8)
    for r in range(squares_y):
        for c in range(squares_x):
            if (r + c) % 2 == 0:
                base[r * square_px:(r + 1) * square_px, c * square_px:(c + 1) * square_px] = 0
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    W, H = 640, 480
    rng = np.random.default_rng(seed)
    for i in range(n):
        scale = 0.5 + rng.random() * 0.2
        bw, bh = int(700 * scale), int(500 * scale)
        small = cv2.resize(base, (min(bw, W - 10), min(bh, H - 10)))
        canvas = np.full((H, W, 3), 200, dtype=np.uint8)
        bh2, bw2 = small.shape[:2]
        x0 = rng.integers(0, max(W - bw2, 1))
        y0 = rng.integers(0, max(H - bh2, 1))
        canvas[y0:y0 + bh2, x0:x0 + bw2] = small
        cv2.imwrite(str(out_dir / f"cb_{i:02d}.jpg"), canvas)


def test_cli_chessboard_pattern_full_pipeline(tmp_path):
    """--pattern chessboard로도 CLI 전체 파이프라인이 돌아야 한다
    (marker_size/dictionary 없이도 정상 동작 확인).
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _render_and_save_chessboard_images(images_dir)

    out_dir = tmp_path / "out"
    exit_code = main([
        "--images", str(images_dir),
        "--pattern", "chessboard",
        "--squares-x", "7", "--squares-y", "5", "--square-size", "0.04",
        "--output-dir", str(out_dir),
        "--quiet",
    ])
    assert exit_code == 0
    assert (out_dir / "camera.yaml").exists()


def test_cli_chessboard_does_not_require_marker_size(tmp_path):
    """chessboard는 --marker-size 없이도 성공해야 하고, 반대로 charuco는
    --marker-size 없으면 (argparse 필수 인자는 통과했더라도) 명확한 에러로
    종료 코드 1을 내야 한다 - 패턴 타입별로 필수 인자가 다르게 검증되는지 확인.
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _render_and_save_chessboard_images(images_dir, n=4)

    # chessboard: marker-size 없이도 성공
    exit_code_chessboard = main([
        "--images", str(images_dir), "--pattern", "chessboard",
        "--squares-x", "7", "--squares-y", "5", "--square-size", "0.04",
        "--output-dir", str(tmp_path / "out_cb"), "--quiet",
    ])
    assert exit_code_chessboard == 0

    # charuco: marker-size 없으면 실패해야 함 (chessboard 이미지라 검출은 실패하겠지만,
    # 그 전에 marker-size 누락으로 먼저 걸려야 함 - CliError -> exit 1)
    exit_code_charuco = main([
        "--images", str(images_dir), "--pattern", "charuco",
        "--squares-x", "7", "--squares-y", "5", "--square-size", "0.04",
        "--output-dir", str(tmp_path / "out_ch"), "--quiet",
    ])
    assert exit_code_charuco == 1


def test_cli_full_pipeline_succeeds(synthetic_distorted_dataset_dir, tmp_path):
    out_dir = tmp_path / "out"
    summary_path = tmp_path / "summary.json"

    exit_code = main([
        "--images", synthetic_distorted_dataset_dir,
        *_base_pattern_args(),
        "--output-dir", str(out_dir),
        "--json-summary", str(summary_path),
        "--quiet",
    ])

    assert exit_code == 0
    assert (out_dir / "camera.yaml").exists()
    assert (out_dir / "camera_info.yaml").exists()
    assert (out_dir / "report.html").exists()
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["success"] is True
    assert summary["chosen_model"] in ("pinhole", "brown_conrady", "extended_pinhole", "fisheye")
    assert summary["num_images_total"] == 16
    assert summary["num_images_detected"] >= 10


def test_cli_model_override_is_respected(synthetic_distorted_dataset_dir, tmp_path):
    out_dir = tmp_path / "out"
    summary_path = tmp_path / "summary.json"

    exit_code = main([
        "--images", synthetic_distorted_dataset_dir,
        *_base_pattern_args(),
        "--model", "extended_pinhole",
        "--output-dir", str(out_dir),
        "--json-summary", str(summary_path),
        "--quiet",
    ])

    assert exit_code == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["chosen_model"] == "extended_pinhole"


def test_cli_outlier_flag_runs_without_crashing(synthetic_distorted_dataset_dir, tmp_path):
    out_dir = tmp_path / "out"
    exit_code = main([
        "--images", synthetic_distorted_dataset_dir,
        *_base_pattern_args(),
        "--outlier", "--max-iterations", "2",
        "--output-dir", str(out_dir),
        "--quiet",
    ])
    assert exit_code == 0
    assert (out_dir / "camera.yaml").exists()


def test_cli_save_and_load_project_round_trip(synthetic_distorted_dataset_dir, tmp_path):
    """--save-project로 저장하고, 원본 이미지를 지운 뒤 --load-project로만
    이어서 export까지 되는지 - 이 기능의 핵심 시나리오.
    """
    import shutil

    out_dir1 = tmp_path / "out1"
    project_path = out_dir1 / "session.ccproj"

    exit_code = main([
        "--images", synthetic_distorted_dataset_dir,
        *_base_pattern_args(),
        "--output-dir", str(out_dir1),
        "--save-project", str(project_path),
        "--quiet",
    ])
    assert exit_code == 0
    assert project_path.exists()

    # 원본 이미지를 통째로 지워서 "이미지 없이도 이어서 되는지" 시나리오 구성
    images_backup = tmp_path / "images_backup"
    shutil.move(synthetic_distorted_dataset_dir, images_backup)

    out_dir2 = tmp_path / "out2"
    exit_code = main([
        "--load-project", str(project_path),
        "--output-dir", str(out_dir2),
        "--export", "report",
        "--quiet",
    ])
    assert exit_code == 0
    assert (out_dir2 / "report.html").exists()

    # 원상복구 (다른 테스트가 이 fixture 디렉토리를 재사용할 수 있으므로)
    shutil.move(images_backup, synthetic_distorted_dataset_dir)


def test_cli_load_project_with_outlier_continues_pipeline(synthetic_distorted_dataset_dir, tmp_path):
    out_dir1 = tmp_path / "out1"
    project_path = out_dir1 / "session.ccproj"

    exit_code = main([
        "--images", synthetic_distorted_dataset_dir,
        *_base_pattern_args(),
        "--output-dir", str(out_dir1),
        "--save-project", str(project_path),
        "--quiet",
    ])
    assert exit_code == 0

    out_dir2 = tmp_path / "out2"
    exit_code = main([
        "--load-project", str(project_path),
        "--outlier",
        "--output-dir", str(out_dir2),
        "--quiet",
    ])
    assert exit_code == 0
    assert (out_dir2 / "camera.yaml").exists()


def test_cli_load_project_conflicts_with_images():
    with pytest.raises(SystemExit) as exc_info:
        main(["--load-project", "some.ccproj", "--images", "some_dir"])
    assert exc_info.value.code != 0


def test_cli_rational_flag_no_longer_exists():
    """P0-1 회귀 방지: --rational boolean 옵션이 완전히 제거됐는지 확인.
    Rational은 이제 --model rational (또는 extended_pinhole/extended)로만
    선택한다 - 별도 on/off 플래그가 없다.
    """
    parser = build_arg_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--images", "some_dir", *_base_pattern_args(), "--rational"])
    assert exc_info.value.code != 0


def test_cli_model_rational_alias_maps_to_extended_pinhole():
    """--model rational이 --model extended_pinhole과 완전히 같은 모델을
    가리키는지 확인 (README/CLI 문서에 약속한 alias)."""
    from app.cli import _MODEL_BY_NAME

    parser = build_arg_parser()
    args = parser.parse_args([
        "--images", "some_dir", *_base_pattern_args(), "--model", "rational",
    ])
    args = _normalize_cli_args(args, parser)
    assert args.model == "rational"
    assert _MODEL_BY_NAME[args.model] == CameraModelType.EXTENDED_PINHOLE


def test_cli_load_nonexistent_project_returns_exit_code_1():
    exit_code = main(["--load-project", "/nonexistent/path.ccproj"])
    assert exit_code == 1


def test_cli_export_subset_only_writes_requested_files(synthetic_distorted_dataset_dir, tmp_path):
    out_dir = tmp_path / "out"
    exit_code = main([
        "--images", synthetic_distorted_dataset_dir,
        *_base_pattern_args(),
        "--export", "report",
        "--output-dir", str(out_dir),
        "--quiet",
    ])
    assert exit_code == 0
    assert (out_dir / "report.html").exists()
    assert not (out_dir / "camera.yaml").exists()
    assert not (out_dir / "camera_info.yaml").exists()


def test_cli_export_json_and_csv(synthetic_distorted_dataset_dir, tmp_path):
    import json

    out_dir = tmp_path / "out"
    exit_code = main([
        "--images", synthetic_distorted_dataset_dir,
        *_base_pattern_args(),
        "--export", "json", "csv",
        "--output-dir", str(out_dir),
        "--quiet",
    ])
    assert exit_code == 0
    assert (out_dir / "calibration.json").exists()
    assert (out_dir / "dataset.csv").exists()
    # 다른 export는 명시 안 했으니 안 만들어져야 함 (opt-in 확인)
    assert not (out_dir / "camera.yaml").exists()
    assert not (out_dir / "report.html").exists()

    with open(out_dir / "calibration.json", encoding="utf-8") as f:
        data = json.load(f)  # 파싱되면 유효한 JSON
    assert "chosen_model" in data
    assert "models" in data

    csv_content = (out_dir / "dataset.csv").read_text(encoding="utf-8")
    assert "image_id" in csv_content.splitlines()[0]
    assert len(csv_content.splitlines()) == 17  # 헤더 1줄 + 16장


def test_cli_cross_dataset_validation_workflow(synthetic_distorted_dataset_dir, tmp_path):
    out_dir = tmp_path / "out"
    summary_path = tmp_path / "summary.json"
    project_path = tmp_path / "session.ccproj"

    exit_code = main([
        "--images", synthetic_distorted_dataset_dir,
        *_base_pattern_args(),
        "--source-dataset-id", "DatasetA",
        "--cross-dataset", f"DatasetB={synthetic_distorted_dataset_dir}",
        "--export", "json", "report",
        "--json-summary", str(summary_path),
        "--save-project", str(project_path),
        "--output-dir", str(out_dir),
        "--quiet",
    ])

    assert exit_code == 0

    payload = json.loads((out_dir / "calibration.json").read_text(encoding="utf-8"))
    assert payload["cross_dataset_validation"]
    assert {r["target_dataset_id"] for r in payload["cross_dataset_validation"]} == {"DatasetB"}
    assert {r["source_dataset_id"] for r in payload["cross_dataset_validation"]} == {"DatasetA"}

    html = (out_dir / "report.html").read_text(encoding="utf-8")
    assert "Cross-Dataset Generalization" in html
    assert "DatasetB" in html

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["cross_dataset_validation"]
    assert summary["cross_dataset_validation"][0]["target_dataset_id"] == "DatasetB"

    saved = json.loads(project_path.read_text(encoding="utf-8"))
    assert saved["project"]["cross_dataset_results"]


def test_cli_diagnostic_preset_enables_document_style_defaults():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--images", "imgs",
        *_base_pattern_args(),
        "--diagnostic",
    ])
    args = _normalize_cli_args(args, parser)

    assert args.kfold == 5
    assert args.bootstrap_ci is True
    assert args.n_bootstrap == 100
    assert {"report", "json", "csv"}.issubset(set(args.export))


def test_cli_document_style_aliases_map_to_internal_options():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--input", "imgs",
        *_base_pattern_args(),
        "--models", "pinhole", "extended", "fisheye",
        "--validate",
        "--report",
        "--cross-validation", "4",
        "--bootstrap", "12",
    ])
    args = _normalize_cli_args(args, parser)

    assert args.images == ["imgs"]
    assert args.models == ["pinhole", "extended", "fisheye"]
    assert args.validate is True
    assert args.kfold == 4
    assert args.bootstrap_ci is True
    assert args.n_bootstrap == 12
    assert "report" in args.export


def test_cli_benchmark_aliases_activate_reference_candidate_mode():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--reference", "ref.yaml",
        "--candidate", "cand.yaml",
        "--validation-dataset", "validation",
        *_base_pattern_args(),
        "--benchmark-report",
        "--bootstrap", "12",
    ])
    args = _normalize_cli_args(args, parser)

    assert args.benchmark_mode is True
    assert args.reference == "ref.yaml"
    assert args.candidate == "cand.yaml"
    assert args.validation_dataset == ["validation"]
    assert args.benchmark_report is True
    assert args.bootstrap_ci is True
    assert args.n_bootstrap == 12


def test_cli_benchmark_reference_candidate_workflow_writes_outputs(
    synthetic_distorted_dataset_dir, tmp_path,
):
    ref_path = tmp_path / "reference.json"
    cand_path = tmp_path / "candidate.json"
    out_dir = tmp_path / "benchmark_out"
    summary_path = tmp_path / "benchmark_summary.json"

    export_standard_json(
        StandardCalibration(
            label="Reference",
            camera_matrix=TRUE_K.copy(),
            distortion=TRUE_D.copy() * 0.0,
            model_name=CameraModelType.EXTENDED_PINHOLE,
            distortion_model="plumb_bob",
            width=IMG_W,
            height=IMG_H,
        ),
        str(ref_path),
    )
    export_standard_json(
        StandardCalibration(
            label="Candidate",
            camera_matrix=TRUE_K.copy(),
            distortion=TRUE_D.copy(),
            model_name=CameraModelType.EXTENDED_PINHOLE,
            distortion_model="plumb_bob",
            width=IMG_W,
            height=IMG_H,
        ),
        str(cand_path),
    )

    exit_code = main([
        "--reference", str(ref_path),
        "--candidate", str(cand_path),
        "--validation-dataset", synthetic_distorted_dataset_dir,
        *_base_pattern_args(),
        "--output-dir", str(out_dir),
        "--benchmark-report",
        "--bootstrap", "10",
        "--json-summary", str(summary_path),
        "--quiet",
    ])

    assert exit_code == 0
    result_path = out_dir / "benchmark_result.json"
    report_path = out_dir / "benchmark_report.html"
    assert result_path.exists()
    assert report_path.exists()
    assert summary_path.exists()

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["benchmark_format_version"] == 1
    assert payload["external"]["label"] == "Reference"
    assert payload["mine"]["label"] == "Candidate"
    assert payload["dataset"]["num_detected"] >= 10
    assert payload["final_benchmark_rows"]
    assert payload["winner_decision"]["status"] in {
        "Candidate Preferred",
        "Reference Preferred",
        "Inconclusive",
        "Insufficient Evidence",
    }

    html = report_path.read_text(encoding="utf-8")
    assert "Calibration Benchmark Report" in html
    assert "Final Report Summary" in html
    assert "Performance Comparison" in html
    assert "Statistical Evidence" in html
    assert "Visual Evidence" in html
    assert "Parameter Analysis" in html
    assert "Model Analysis" in html
    assert "FINAL VERDICT" in html
    assert "One-line diagnosis" in html
    assert "Final Benchmark Table" in html

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["mode"] == "benchmark"
    assert summary["benchmark_result_json"] == str(result_path)


def test_cli_models_single_alias_becomes_final_model():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--input", "imgs",
        *_base_pattern_args(),
        "--models", "extended",
    ])
    args = _normalize_cli_args(args, parser)

    assert args.model == "extended"


def test_cli_model_must_be_inside_models_list():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--images", "imgs",
        *_base_pattern_args(),
        "--models", "pinhole",
        "--model", "fisheye",
    ])

    with pytest.raises(SystemExit):
        _normalize_cli_args(args, parser)


def test_cli_document_example_aliases_run_pipeline(synthetic_distorted_dataset_dir, tmp_path):
    out_dir = tmp_path / "out"
    exit_code = main([
        "--input", synthetic_distorted_dataset_dir,
        *_base_pattern_args(),
        "--models", "pinhole", "extended", "fisheye",
        "--validate",
        "--report",
        "--output-dir", str(out_dir),
        "--quiet",
    ])

    assert exit_code == 0
    assert (out_dir / "report.html").exists()


def test_cli_cross_validation_alias_conflicts_with_different_kfold():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--images", "imgs",
        *_base_pattern_args(),
        "--kfold", "3",
        "--cross-validation", "5",
    ])

    with pytest.raises(SystemExit):
        _normalize_cli_args(args, parser)


def test_cli_glob_pattern_resolves_images(synthetic_distorted_dataset_dir, tmp_path):
    out_dir = tmp_path / "out"
    exit_code = main([
        "--images", f"{synthetic_distorted_dataset_dir}/*.jpg",
        *_base_pattern_args(),
        "--output-dir", str(out_dir),
        "--quiet",
    ])
    assert exit_code == 0


def test_cli_empty_image_dir_returns_exit_code_1(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    exit_code = main(["--images", str(empty_dir), *_base_pattern_args()])
    assert exit_code == 1


def test_cli_missing_required_args_exits_nonzero():
    with pytest.raises(SystemExit) as exc_info:
        main(["--images", "some_dir"])  # squares-x 등 누락
    assert exc_info.value.code != 0


def test_cli_conflicting_images_and_bag_exits_nonzero():
    with pytest.raises(SystemExit) as exc_info:
        main(["--images", "a", "--bag", "b.bag", *_base_pattern_args()])
    assert exc_info.value.code != 0


def test_cli_missing_marker_size_returns_exit_code_1(synthetic_distorted_dataset_dir):
    exit_code = main([
        "--images", synthetic_distorted_dataset_dir,
        "--squares-x", "7", "--squares-y", "5", "--square-size", "0.04",
        # --marker-size 없음
    ])
    assert exit_code == 1


@pytest.mark.xfail(
    reason=(
        "알려진 이슈: ChArUco로 렌더링된 이미지에 --pattern chessboard를 지정하면 "
        "체스보드 코너 검출기가 우연히 격자 무늬를 찾아 '검출 성공'으로 통과해버린다. "
        "Pinhole/Extended Pinhole은 그 잘못된 대응관계로도 cv2 관점에서 수치적으로는 "
        "'성공'(RMS는 비정상적으로 높지만 발산은 아님)해서 exit code 0으로 끝난다 - "
        "Fisheye만 수학적으로 발산해 예외로 실패한다. sanity_check.py도 이 경우를 못 잡는다 "
        "(fx/fy가 유한하고 양수라 ERROR가 아니라 WARNING만 뜸). 근본적으로 고치려면 "
        "'검출된 패턴이 지정한 패턴과 실제로 일치하는지' 검증하는 별도 기능이 필요하다 "
        "(현재 범위 밖 - 3~8번 작업과 무관하게 이전부터 있던 문제)."
    ),
    strict=True,
)
def test_cli_unsupported_pattern_type_returns_exit_code_1(synthetic_distorted_dataset_dir):
    exit_code = main([
        "--images", synthetic_distorted_dataset_dir,
        "--pattern", "chessboard",
        *_base_pattern_args(),
    ])
    assert exit_code == 1


def test_cli_bag_without_topic_returns_exit_code_1(tmp_path):
    fake_bag = tmp_path / "fake.bag"
    fake_bag.write_text("not a real bag")
    exit_code = main(["--bag", str(fake_bag), *_base_pattern_args()])
    assert exit_code == 1


def test_cli_list_topics_on_real_bag(tmp_path):
    pytest.importorskip("rosbags", reason="rosbags 미설치")
    import numpy as np
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import get_typestore, Stores
    from rosbags.typesys.stores.ros1_noetic import sensor_msgs__msg__Image as RbImage
    from rosbags.typesys.stores.ros1_noetic import std_msgs__msg__Header as RbHeader
    from rosbags.typesys.stores.ros1_noetic import builtin_interfaces__msg__Time as RbTime

    ts = get_typestore(Stores.ROS1_NOETIC)
    bag_path = str(tmp_path / "t.bag")
    with Writer(bag_path) as writer:
        conn = writer.add_connection("/cam/image_raw", RbImage.__msgtype__, typestore=ts)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        msg = RbImage(
            header=RbHeader(seq=0, stamp=RbTime(sec=0, nanosec=0), frame_id="c"),
            height=10, width=10, encoding="bgr8", is_bigendian=0, step=30, data=img.reshape(-1),
        )
        writer.write(conn, 0, ts.serialize_ros1(msg, RbImage.__msgtype__))

    exit_code = main(["--list-topics", bag_path])
    assert exit_code == 0

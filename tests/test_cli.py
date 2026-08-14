"""
tests/test_cli.py
======================

app/cli.py - 헤드리스 CLI 진입점. 실제로 subprocess처럼 main()을 호출해서
종료 코드/출력 파일까지 확인한다 (단위 테스트 수준을 넘어 "CLI로서 실제로
동작하는가"를 검증하는 것이 목적).
"""

from __future__ import annotations

import json

import pytest

from app.cli import main

pytestmark = pytest.mark.slow


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
    assert summary["chosen_model"] in ("pinhole", "extended_pinhole", "fisheye")
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

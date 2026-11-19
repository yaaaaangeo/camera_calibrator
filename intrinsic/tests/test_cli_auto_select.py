"""CLI frame auto-selection (rosbag inputs).

Why: Perspective fits are cubic in frame count (measured 960x540: 20 views
3.5ms/iteration, 146 views 894ms/iteration). A 3-model CLI run on a rosbag
(150-330 frames) took 15 min to 2h+; with auto-selection it takes ~1-5 min.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import app.cli as cli
from calibration.image_selection import ImageSelectionConfig, select_calibration_images
from calibration.types import CameraConfig, Dataset, DetectionResult, Frame, FrameStatus, ImageInfo
from tests.test_image_selection import _frame


def _args(**kw):
    base = dict(bag=None, auto_select=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_bag_input_auto_selects_by_default():
    assert cli._effective_auto_select_count(_args(bag="x.db3")) == cli.DEFAULT_BAG_AUTO_SELECT_COUNT


def test_image_input_keeps_all_frames_by_default():
    assert cli._effective_auto_select_count(_args(images=["a"])) == 0


def test_explicit_count_and_zero_disable():
    assert cli._effective_auto_select_count(_args(bag="x.db3", auto_select=0)) == 0
    assert cli._effective_auto_select_count(_args(auto_select=30)) == 30
    assert cli._effective_auto_select_count(_args(auto_select=-5)) == 0


def test_parser_accepts_auto_select():
    args = cli.build_arg_parser().parse_args(["--bag", "x.db3", "--topic", "/t", "--auto-select", "0"])
    assert args.auto_select == 0
    assert cli.build_arg_parser().parse_args(["--bag", "x.db3", "--topic", "/t"]).auto_select is None


def _few_corner_frame(image_id: str, corners: int) -> Frame:
    frame = _frame(image_id)
    frame.detection.num_corners = corners
    frame.detection.corners = np.zeros((corners, 1, 2), dtype=np.float32)
    return frame


def test_min_corners_rejects_sparse_detections_and_default_keeps_them():
    frames = [_few_corner_frame("f1", 4), _frame("f2"), _frame("f3", center=(200.0, 540.0))]
    kept = select_calibration_images(Dataset(frames=[_few_corner_frame("f1", 4)]), ImageSelectionConfig(target_count=5))
    assert kept.selected_image_ids == ["f1"]  # default min_corners=0: unchanged behavior

    result = select_calibration_images(Dataset(frames=frames), ImageSelectionConfig(target_count=5, min_corners=12))
    assert "f1" not in result.selected_image_ids
    assert result.counts["too_few_corners"] == 1
    record = next(r for r in result.records if r.image_id == "f1")
    assert record.exclusion_reason == "too_few_corners"
    assert frames[0].status != FrameStatus.DETECTED
    assert "too few corners 1" in result.summary_text()


def test_apply_auto_selection_keeps_only_selected_frames(monkeypatch):
    frames = [_frame(f"f{i}", center=(200.0 + 150 * (i % 10), 150.0 + 80 * (i // 10)), area=0.05 + 0.01 * i)
              for i in range(1, 31)]
    dataset = Dataset(frames=frames)
    monkeypatch.setattr(cli, "analyze_dataset_quality", lambda *a, **k: [])
    monkeypatch.setattr(cli, "compute_frame_quality_scores", lambda *a, **k: None)
    summary = cli._apply_auto_selection(dataset, None, CameraConfig(1920, 1080), 10, quiet=True)
    assert summary["applied"] and summary["num_input"] == 30
    assert summary["selected_count"] == len(dataset.frames) <= 10
    assert [f.image_info.image_id for f in dataset.frames] == summary["selected_image_ids"]


def test_apply_auto_selection_skipped_when_already_small():
    dataset = Dataset(frames=[_frame("f1"), _frame("f2")])
    summary = cli._apply_auto_selection(dataset, None, CameraConfig(1920, 1080), 10, quiet=True)
    assert summary["applied"] is False and len(dataset.frames) == 2

from datetime import datetime, timezone
import json

from export.output_manager import OutputManager, SESSION_DIRECTORIES


FIXED_TIME = datetime(2026, 9, 11, 14, 5, 9, tzinfo=timezone.utc)


def test_session_tree_and_collision_suffix(tmp_path):
    first = OutputManager(tmp_path, camera_name="front cam", now=lambda: FIXED_TIME)
    first_dir = first.ensure_session("front cam")
    second = OutputManager(tmp_path, now=lambda: FIXED_TIME)
    second_dir = second.ensure_session("front cam")

    assert first_dir.name == "2026-09-11_140509_front_cam"
    assert second_dir.name == "2026-09-11_140509_front_cam_02"
    assert all((first_dir / relative).is_dir() for relative in SESSION_DIRECTORIES)


def test_meaningful_paths_and_manifest_are_session_relative(tmp_path):
    manager = OutputManager(tmp_path, now=lambda: FIXED_TIME)
    session = manager.ensure_session()
    artifact = manager.intrinsic_path("extended_pinhole")
    artifact.write_text("yaml", encoding="utf-8")

    manager.update_context(
        camera={"name": "front", "resolution": [1920, 1080]},
        pattern={"type": "charuco", "squares": [10, 7]},
        calibration={"available_models": ["pinhole", "extended_pinhole"]},
    )
    manager.record_export("intrinsic.rational", artifact, metadata={"selected": True})

    assert artifact == session / "intrinsic/camera_rational.yaml"
    manifest = json.loads(manager.manifest_path.read_text(encoding="utf-8"))
    assert manifest["camera"]["name"] == "front"
    assert manifest["calibration"]["available_models"] == [
        "pinhole",
        "extended_pinhole",
    ]
    assert manifest["exports"]["intrinsic.rational"]["file"] == (
        "intrinsic/camera_rational.yaml"
    )


def test_windshield_and_comparison_paths(tmp_path):
    manager = OutputManager(tmp_path, now=lambda: FIXED_TIME)
    assert manager.windshield_geometry_path("fisheye", "residual rbf").name == "residual_rbf.yaml"
    assert manager.reflection_path().parent.name == "reflection"
    assert manager.ghost_path().parent.name == "ghost"
    assert manager.validation_directory().name == "subset_comparison"


def test_unavailable_and_failure_do_not_block_later_export(tmp_path):
    manager = OutputManager(tmp_path, now=lambda: FIXED_TIME)
    good_path = manager.report_path("good.txt")

    skipped = manager.execute_export("missing", None)
    failed = manager.execute_export("broken", lambda: (_ for _ in ()).throw(ValueError("boom")))
    succeeded = manager.execute_export(
        "good", lambda: (good_path.write_text("ok", encoding="utf-8"), good_path)[1]
    )

    assert skipped["status"] == "skipped"
    assert failed["status"] == "failed"
    assert "boom" in failed["error"]
    assert succeeded["status"] == "exported"
    assert good_path.read_text(encoding="utf-8") == "ok"

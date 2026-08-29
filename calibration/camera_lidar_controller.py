"""
Workflow-facing camera-lidar (FAST-Calib) calibration service.

Mirrors calibration/stereo_controller.py: kept intentionally thin so UI/
worker code depends on this controller instead of importing
camera_lidar.pipeline internals directly.
"""

from __future__ import annotations

from typing import Callable, Optional

from calibration.calibration_io import StandardCalibration
from calibration.rosbag_reader import extract_pointcloud_near_timestamp, iterate_images, list_image_topics
from camera_lidar.extraction_diagnostics import ExtractionDiagnosticSummary
from camera_lidar.pipeline import calibrate_single_scene
from camera_lidar.scene_extraction import build_scene_candidates
from camera_lidar.target_config import TargetConfig
from camera_lidar.types import CalibrationScene, CameraLidarCalibrationResult, SceneCandidate


class CameraLidarController:
    def calibrate(
        self,
        scene: CalibrationScene,
        roi_mode: str = "manual",
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> CameraLidarCalibrationResult:
        return calibrate_single_scene(scene, roi_mode=roi_mode, cancel_check=cancel_check)

    def extract_scene_candidates(
        self,
        bag_path: str,
        camera_topic: str,
        lidar_topic: str,
        intrinsics: StandardCalibration,
        target: TargetConfig,
        progress_callback: Optional[Callable[[str], None]] = None,
        frame_progress_callback: Optional[Callable[[int, int], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        pair_lidar: bool = True,
        detector_workers: int = 1,
    ) -> tuple[list[SceneCandidate], ExtractionDiagnosticSummary]:
        """Bag-wide MARKER EXTRACTION scan (camera_lidar.scene_extraction is
        ROS-independent and never imports calibration.rosbag_reader itself --
        this controller is the adapter glue that supplies it a bag frame
        stream + a LiDAR-cloud lookup closure)."""

        def cloud_lookup(t_sec: float):
            try:
                points, cloud_ts, _frame_id = extract_pointcloud_near_timestamp(bag_path, lidar_topic, t_sec)
                return points, cloud_ts
            except Exception:  # noqa: BLE001 -- a missing/undecodable LiDAR frame just means no cloud pairing for this candidate
                return None

        # Cheap metadata-only re-read (same bag index list_image_topics already
        # used for topic discovery) purely so progress can show a real
        # percentage/(done, total) instead of an unbounded, unfamiliar count --
        # a long silent scan otherwise looks identical to a hung one.
        total_frames: Optional[int] = None
        try:
            for topic in list_image_topics(bag_path):
                if topic.name == camera_topic:
                    total_frames = topic.count
                    break
        except Exception:  # noqa: BLE001 -- progress is a nicety, never block extraction over it
            total_frames = None

        return build_scene_candidates(
            frames_factory=lambda: iterate_images(bag_path, camera_topic),
            camera_topic=camera_topic,
            lidar_topic=lidar_topic,
            intrinsics=intrinsics,
            target=target,
            cloud_lookup=cloud_lookup,
            total_frames=total_frames,
            progress_callback=progress_callback,
            frame_progress_callback=frame_progress_callback,
            cancel_check=cancel_check,
            pair_lidar=pair_lidar,
            detector_workers=detector_workers,
        )

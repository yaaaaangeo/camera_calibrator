"""
camera_calibrator.camera_lidar.target_config
================================================

FAST-Calib circular-hole target geometry configuration.

Field names mirror upstream hku-mars/FAST-Calib's config/qr_params.yaml
(marker_size, delta_width_qr_center, delta_height_qr_center,
delta_width_circles, delta_height_circles, circle_radius) so a value can be
copied over directly, but this loader parses its own smaller YAML schema --
upstream's qr_params.yaml also embeds ROS launch args (topics, bag paths)
that don't belong in a ROS-independent config.

The board has 4 ArUco markers (one per corner) and 4 circular holes, each
inset from its corner marker. `marker_ids` fixes which physical corner each
ArUco ID sits at -- this is the piece of information that lets
camera_detector.py assign a real-world "top_left/top_right/bottom_right/
bottom_left" label to each detected circle center; the LiDAR side has no
such absolute reference (see camera_lidar/correspondence.py for how that
ambiguity is resolved).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import yaml

# A cyclic traversal of the rectangle's 4 corners (adjacent entries share an
# edge). camera_lidar.correspondence relies on this being a true cycle.
CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")


@dataclass
class TargetConfig:
    marker_size: float = 0.20                # meters, ArUco marker side length
    delta_width_qr_center: float = 0.55      # meters, horizontal spacing between marker centers
    delta_height_qr_center: float = 0.35     # meters, vertical spacing between marker centers
    delta_width_circles: float = 0.5         # meters, horizontal spacing between circle centers
    delta_height_circles: float = 0.4        # meters, vertical spacing between circle centers
    circle_radius: float = 0.12              # meters
    aruco_dictionary: str = "DICT_4X4_50"
    marker_ids: dict[str, int] = field(
        default_factory=lambda: {"top_left": 1, "top_right": 2, "bottom_right": 3, "bottom_left": 4}
    )

    def _corner_offsets(self, delta_width: float, delta_height: float) -> dict[str, tuple[float, float]]:
        hw, hh = delta_width / 2.0, delta_height / 2.0
        return {
            "top_left": (-hw, hh),
            "top_right": (hw, hh),
            "bottom_right": (hw, -hh),
            "bottom_left": (-hw, -hh),
        }

    def circle_centers_board_frame(self) -> np.ndarray:
        """4x3 circle centers in the board's own local frame (Z=0 plane,
        board center at origin), ordered per CORNER_ORDER."""
        offsets = self._corner_offsets(self.delta_width_circles, self.delta_height_circles)
        return np.array([[offsets[c][0], offsets[c][1], 0.0] for c in CORNER_ORDER], dtype=np.float64)

    def marker_centers_board_frame(self) -> dict[int, np.ndarray]:
        """marker_id -> 3-vector marker center in the board's local frame."""
        offsets = self._corner_offsets(self.delta_width_qr_center, self.delta_height_qr_center)
        return {
            self.marker_ids[corner]: np.array([offsets[corner][0], offsets[corner][1], 0.0])
            for corner in CORNER_ORDER
        }


def load_target_config(path: str) -> TargetConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    field_names = set(TargetConfig.__dataclass_fields__.keys())
    kwargs = {k: v for k, v in data.items() if k in field_names}
    return TargetConfig(**kwargs)


def save_target_config(target: TargetConfig, path: str) -> None:
    data = {
        "marker_size": target.marker_size,
        "delta_width_qr_center": target.delta_width_qr_center,
        "delta_height_qr_center": target.delta_height_qr_center,
        "delta_width_circles": target.delta_width_circles,
        "delta_height_circles": target.delta_height_circles,
        "circle_radius": target.circle_radius,
        "aruco_dictionary": target.aruco_dictionary,
        "marker_ids": target.marker_ids,
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

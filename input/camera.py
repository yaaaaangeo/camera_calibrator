"""
input/camera.py

Camera model + camera data loader, per the Input Loader Spec (v0.1) in
evaluation_metric_spec.md.

Responsibility: PARSE ONLY. This module turns raw camera config + files into
a standardized CameraModel + list of Frame objects. It does not validate
calibration correctness (that's input/extrinsic.py's verify_extrinsic) and
does not compute any evaluation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import glob
import os

import numpy as np
import cv2

from geometry.projection import intrinsics_matrix, plumb_bob_dist_coeffs, fisheye_dist_coeffs


SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def as_matrix(self) -> np.ndarray:
        return intrinsics_matrix(self.fx, self.fy, self.cx, self.cy)


@dataclass
class CameraDistortion:
    model: Literal["plumb_bob", "fisheye_equidistant", "none"]
    coeffs: dict = field(default_factory=dict)

    def as_array(self) -> Optional[np.ndarray]:
        if self.model == "none":
            return None
        if self.model == "plumb_bob":
            return plumb_bob_dist_coeffs(self.coeffs)
        if self.model == "fisheye_equidistant":
            return fisheye_dist_coeffs(self.coeffs)
        raise ValueError(f"Unknown distortion model: {self.model!r}")


@dataclass
class CameraSource:
    kind: Literal["image_dir", "video", "rosbag", "ros_topic"]
    path: str
    topic: Optional[str] = None
    timestamp_source: Literal["filename", "embedded", "topic_header"] = "filename"


@dataclass
class CameraModel:
    width: int
    height: int
    model: Literal["pinhole", "fisheye"]
    intrinsics: CameraIntrinsics
    distortion: CameraDistortion
    source: CameraSource

    # floor(Z) Term 3 -- optional, see quality/noise_floor.py
    edge_localization_floor_px: Optional[float] = None

    def K(self) -> np.ndarray:
        return self.intrinsics.as_matrix()

    def dist_coeffs(self) -> Optional[np.ndarray]:
        return self.distortion.as_array()

    def projection_model_name(self) -> str:
        """Maps the high-level 'pinhole'/'fisheye' model field to the
        projection function selector used by geometry/projection.py."""
        return self.model

    def verify_image_shape(self, image: np.ndarray) -> None:
        """
        Raise a clear ValueError if `image`'s actual pixel dimensions
        don't match this CameraModel's declared width/height.

        Nothing in the loading path actually opens an image file to check
        it against the config's width/height (loading is intentionally
        lazy -- see CameraFrame.load()), so a typo'd width/height, or an
        image_dir that mixes differently-sized files, previously surfaced
        as a downstream IndexError wherever a consumer happened to index
        into the image array using the declared dimensions (several
        visualization functions sample image[py, px] directly; others,
        like Canny-based edge detection, use the image's own shape and
        don't reveal a mismatch at all, but would still be measuring
        alignment against the wrong assumed field of view). This is the
        one place callers can check that instead, with a message that
        says what's actually wrong.
        """
        if image.ndim < 2:
            raise ValueError(f"Expected a 2D/3D image array, got shape {image.shape}")
        actual_height, actual_width = image.shape[:2]
        if actual_width != self.width or actual_height != self.height:
            raise ValueError(
                f"Camera config declares width={self.width}, height={self.height}, "
                f"but the loaded image is {actual_width}x{actual_height}. Check the "
                f"camera config's width/height against the actual image file, or "
                f"whether image_dir mixes differently-sized images."
            )


@dataclass
class CameraFrame:
    timestamp: float
    path: Optional[str] = None
    image: Optional[np.ndarray] = None  # lazily loaded if only `path` is set

    def load(self) -> np.ndarray:
        """Return the image array, loading from disk on first access if
        needed. Cached on the frame object after first load."""
        if self.image is not None:
            return self.image
        if self.path is None:
            raise ValueError("CameraFrame has neither `image` nor `path` set.")
        img = cv2.imread(self.path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Failed to read image at {self.path}")
        self.image = img
        return img


class CameraLoadWarning(RuntimeWarning):
    pass


@dataclass
class CameraLoadResult:
    camera: CameraModel
    frames: list[CameraFrame]
    warnings: list[str] = field(default_factory=list)


def _timestamp_from_filename(path: str) -> float:
    """
    Extract a timestamp from a filename stem. Supports:
      - pure numeric stems (e.g. '1699999999.123456.png' or '000123.png')
      - falls back to file index order (returned as float) with a caller-side
        warning if the stem isn't numeric -- handled by the caller so it can
        aggregate a single warning instead of one per file.
    """
    stem = Path(path).stem
    try:
        return float(stem)
    except ValueError:
        return float("nan")


def load_camera_from_image_dir(
    path: str,
    width: int,
    height: int,
    model: Literal["pinhole", "fisheye"],
    intrinsics: CameraIntrinsics,
    distortion: CameraDistortion,
    timestamp_source: Literal["filename", "embedded"] = "filename",
    edge_localization_floor_px: Optional[float] = None,
    lazy: bool = True,
) -> CameraLoadResult:
    """
    Load a CameraModel + sorted list of CameraFrame from a directory of
    image files.

    timestamp_source:
      - 'filename': parse timestamp from the numeric filename stem. If
        filenames aren't numeric, falls back to sequential indices
        (0, 1, 2, ...) and records a warning -- downstream sync (dataset.py)
        will then be unable to do real timestamp matching, which the
        warning makes explicit rather than failing silently.
      - 'embedded': not implemented in this pass (would require per-format
        metadata extraction, e.g. EXIF); raises NotImplementedError.
    """
    warnings: list[str] = []

    files = sorted(
        f for f in glob.glob(os.path.join(path, "*"))
        if os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No supported image files found in {path!r} "
                                 f"(looked for {SUPPORTED_IMAGE_EXTENSIONS})")

    if timestamp_source == "embedded":
        raise NotImplementedError(
            "timestamp_source='embedded' is not implemented yet; use 'filename'."
        )
    if timestamp_source != "filename":
        raise ValueError(f"Unsupported timestamp_source for image_dir: {timestamp_source!r}")

    raw_timestamps = [_timestamp_from_filename(f) for f in files]
    if any(np.isnan(t) for t in raw_timestamps):
        warnings.append(
            "One or more image filenames were not numeric; falling back to "
            "sequential integer timestamps (0, 1, 2, ...). Timestamp-based "
            "sync with LiDAR frames will not reflect real capture time."
        )
        raw_timestamps = [float(i) for i in range(len(files))]

    frames = [
        CameraFrame(timestamp=ts, path=f, image=None)
        for ts, f in zip(raw_timestamps, files)
    ]

    if not lazy:
        for fr in frames:
            fr.load()

    source = CameraSource(kind="image_dir", path=path, timestamp_source=timestamp_source)
    camera = CameraModel(
        width=width, height=height, model=model,
        intrinsics=intrinsics, distortion=distortion, source=source,
        edge_localization_floor_px=edge_localization_floor_px,
    )

    return CameraLoadResult(camera=camera, frames=frames, warnings=warnings)


def load_camera_from_video(*args, **kwargs) -> CameraLoadResult:
    raise NotImplementedError(
        "Video source loading is not implemented in this pass. "
        "Use load_camera_from_image_dir with pre-extracted frames, "
        "or extend this function (cv2.VideoCapture) as a follow-up."
    )


# ---------------------------------------------------------------------------
# rosbag reader (rosbag1 .bag and rosbag2 directories, via the pure-Python
# `rosbags` package -- no ROS/rclpy installation required)
# ---------------------------------------------------------------------------

_IMAGE_MSGTYPES = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")


def _require_rosbags():
    """Import the optional `rosbags` dependency, raising a clear,
    actionable error (not a bare ImportError) if it's missing -- mirrors
    input/lidar.py's _require_rosbags(); kept as a separate copy here
    rather than a cross-module import since input/camera.py and
    input/lidar.py are otherwise independent leaf modules and this avoids
    creating a dependency between them just for a shared one-liner."""
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as e:
        raise ImportError(
            "rosbag loading requires the optional 'rosbags' package "
            "(a pure-Python rosbag1/rosbag2 reader -- no ROS/rclpy "
            "installation needed). Install it with:\n"
            "    pip install \"cam-lidar-eval[rosbag]\"\n"
            "or directly: pip install rosbags rosbags-image"
        ) from e
    return AnyReader


def _stamp_to_seconds(header, fallback_ns: int) -> tuple[float, bool]:
    """Prefer the message's own header.stamp; fall back to the bag's
    recv-time timestamp if unstamped (sec=0, nanosec=0). Mirrors
    input/lidar.py's _stamp_to_seconds -- see that copy's docstring for
    why this fallback exists; not shared cross-module for the same reason
    as _require_rosbags above."""
    if header is not None and (header.stamp.sec != 0 or header.stamp.nanosec != 0):
        return header.stamp.sec + header.stamp.nanosec * 1e-9, False
    return fallback_ns * 1e-9, True


def load_camera_from_rosbag(
    path: str,
    width: int,
    height: int,
    model: Literal["pinhole", "fisheye"],
    intrinsics: CameraIntrinsics,
    distortion: CameraDistortion,
    topic: Optional[str] = None,
    edge_localization_floor_px: Optional[float] = None,
) -> CameraLoadResult:
    """
    Load a CameraModel + list of CameraFrame from a rosbag by reading every
    sensor_msgs/msg/Image or sensor_msgs/msg/CompressedImage message on
    `topic` (rosbag1 .bag file or rosbag2 directory, auto-detected -- both
    handled by the `rosbags` package's AnyReader), converting each to a
    BGR OpenCV array via the `rosbags-image` companion package.

    topic: which image topic to read. If None and the bag has exactly one
    Image/CompressedImage topic, that one is used automatically; if it has
    more than one, a ValueError lists the available topics so the caller
    can pick.

    Unlike load_camera_from_image_dir, frames here are NOT lazily loaded
    from a per-frame file path -- images are decoded eagerly as the bag is
    read (there's no per-frame file to defer loading to), so this holds
    every decoded frame in memory at once. For very large bags this means
    higher peak memory than the image_dir path.

    Requires the optional `rosbags` + `rosbags-image` dependencies (`pip
    install "cam-lidar-eval[rosbag]"`) -- pure-Python bag/image readers,
    not an actual ROS/rclpy installation. Live `ros_topic` (subscribing to
    a running ROS node) remains NOT implemented: it needs an active ROS2
    middleware/DDS connection, categorically different from parsing an
    already-recorded bag file.
    """
    AnyReader = _require_rosbags()
    try:
        from rosbags.image import message_to_cvimage
    except ImportError as e:
        raise ImportError(
            "rosbag camera loading also requires the 'rosbags-image' package "
            "(pure-Python ROS Image -> OpenCV conversion). Install it with:\n"
            "    pip install \"cam-lidar-eval[rosbag]\"\n"
            "or directly: pip install rosbags-image"
        ) from e

    warnings: list[str] = []

    bag_path = Path(path)
    if not bag_path.exists():
        raise FileNotFoundError(f"rosbag path does not exist: {path!r}")

    frames: list[CameraFrame] = []
    used_stamp_fallback = False

    with AnyReader([bag_path]) as reader:
        img_conns = [c for c in reader.connections if c.msgtype in _IMAGE_MSGTYPES]
        if not img_conns:
            raise ValueError(
                f"No Image/CompressedImage topics found in {path!r}. "
                f"Available topics: {sorted({c.topic for c in reader.connections})}"
            )
        available_topics = sorted({c.topic for c in img_conns})
        if topic is None:
            if len(available_topics) > 1:
                raise ValueError(
                    f"Multiple image topics found in {path!r}: {available_topics}. "
                    f"Specify `topic` to pick one."
                )
            topic = available_topics[0]
        elif topic not in available_topics:
            raise ValueError(
                f"Topic {topic!r} not found (or not an image topic) in {path!r}. "
                f"Available: {available_topics}"
            )

        selected = [c for c in img_conns if c.topic == topic]
        for conn, bag_timestamp_ns, rawdata in reader.messages(connections=selected):
            msg = reader.deserialize(rawdata, conn.msgtype)
            cv_img = message_to_cvimage(msg, "bgr8")
            ts_sec, used_fallback = _stamp_to_seconds(getattr(msg, "header", None), bag_timestamp_ns)
            used_stamp_fallback = used_stamp_fallback or used_fallback
            frames.append(CameraFrame(timestamp=ts_sec, path=None, image=cv_img))

    if not frames:
        raise ValueError(f"Topic {topic!r} in {path!r} has no messages.")
    if used_stamp_fallback:
        warnings.append(
            f"Some messages on topic {topic!r} had an unstamped header (sec=0, "
            f"nanosec=0); their bag record-time was used as the frame timestamp "
            f"instead. This is usually fine, but if the publisher applies "
            f"significant latency before publishing, sync with LiDAR frames "
            f"may be slightly off."
        )

    frames.sort(key=lambda fr: fr.timestamp)

    source = CameraSource(kind="rosbag", path=path, topic=topic, timestamp_source="topic_header")
    camera = CameraModel(
        width=width, height=height, model=model,
        intrinsics=intrinsics, distortion=distortion, source=source,
        edge_localization_floor_px=edge_localization_floor_px,
    )

    return CameraLoadResult(camera=camera, frames=frames, warnings=warnings)

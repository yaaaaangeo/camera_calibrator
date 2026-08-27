"""
input/lidar.py

LiDAR sensor spec + point cloud sequence loader, per the Input Loader Spec
(v0.1) in evaluation_metric_spec.md.

Supports PCD (ASCII and binary, the common subset) and PLY (ASCII) point
cloud directories. rosbag/ros_topic sources are stubbed (NotImplementedError)
since ROS deserialization deps aren't available in this environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import glob
import os

import numpy as np

from quality.noise_floor import LidarSensorSpecForFloor


SUPPORTED_POINTCLOUD_EXTENSIONS = (".pcd", ".ply")


@dataclass
class LidarSensorSpec(LidarSensorSpecForFloor):
    """
    Extends the minimal floor(Z)-relevant spec (LidarSensorSpecForFloor)
    with the remaining fields from the input loader spec. Kept as a subclass
    so this can be passed directly into quality.noise_floor.resolve_floor_inputs
    without translation.
    """
    min_range_m: float = 0.0
    max_range_m: float = 200.0


@dataclass
class LidarSource:
    kind: Literal["pcd_dir", "ply_dir", "rosbag", "ros_topic"]
    path: str
    topic: Optional[str] = None
    point_fields: list[str] = field(default_factory=lambda: ["x", "y", "z"])


@dataclass
class LidarModel:
    source: LidarSource
    sensor_spec: LidarSensorSpec


@dataclass
class LidarFrame:
    timestamp: float
    path: Optional[str] = None
    points: Optional[np.ndarray] = None  # (N, 3) or (N, 4) with intensity; lazily loaded

    def load(self) -> np.ndarray:
        if self.points is not None:
            return self.points
        if self.path is None:
            raise ValueError("LidarFrame has neither `points` nor `path` set.")
        ext = os.path.splitext(self.path)[1].lower()
        if ext == ".pcd":
            pts = read_pcd(self.path)
        elif ext == ".ply":
            pts = read_ply_ascii(self.path)
        else:
            raise ValueError(f"Unsupported point cloud extension: {ext!r}")
        self.points = pts
        return pts


@dataclass
class LidarLoadResult:
    lidar: LidarModel
    frames: list[LidarFrame]
    warnings: list[str] = field(default_factory=list)


def _timestamp_from_filename(path: str) -> float:
    stem = Path(path).stem
    try:
        return float(stem)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# PCD reader (ASCII + binary "DATA binary" variants of the common PCD subset)
# ---------------------------------------------------------------------------

_PCD_TYPE_MAP = {
    ("F", 4): np.float32,
    ("F", 8): np.float64,
    ("U", 1): np.uint8,
    ("U", 2): np.uint16,
    ("U", 4): np.uint32,
    ("I", 1): np.int8,
    ("I", 2): np.int16,
    ("I", 4): np.int32,
}


def read_pcd(path: str) -> np.ndarray:
    """
    Minimal PCD (Point Cloud Data) reader supporting ASCII and binary
    DATA sections for the common FIELDS subset (x y z [intensity ...]).
    Does not support 'binary_compressed'.

    Returns an (N, 3) array if only x,y,z are present, or (N, 4) if an
    'intensity' field is also present (appended as the 4th column).
    """
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while parsing PCD header: {path}")
            line_str = line.decode("ascii", errors="strict").strip()
            if line_str.startswith("#") or line_str == "":
                continue
            key, _, rest = line_str.partition(" ")
            header[key] = rest.strip()
            if key == "DATA":
                break

        fields = header["FIELDS"].split()
        sizes = [int(s) for s in header["SIZE"].split()]
        types = header["TYPE"].split()
        counts = [int(c) for c in header["COUNT"].split()]
        n_points = int(header["POINTS"])
        data_kind = header["DATA"]

        try:
            x_idx = fields.index("x")
            y_idx = fields.index("y")
            z_idx = fields.index("z")
        except ValueError as e:
            raise ValueError(f"PCD file {path} missing required x/y/z field: {e}")
        intensity_idx = fields.index("intensity") if "intensity" in fields else None

        if data_kind == "ascii":
            data = np.loadtxt(f, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            xyz = data[:, [x_idx, y_idx, z_idx]]
            if intensity_idx is not None:
                intensity = data[:, [intensity_idx]]
                return np.hstack([xyz, intensity]).astype(np.float32)
            return xyz.astype(np.float32)

        elif data_kind == "binary":
            dtype_fields = []
            for name, size, typ, count in zip(fields, sizes, types, counts):
                np_type = _PCD_TYPE_MAP.get((typ, size))
                if np_type is None:
                    raise ValueError(f"Unsupported PCD field type {typ}{size} for field {name!r}")
                if count == 1:
                    dtype_fields.append((name, np_type))
                else:
                    dtype_fields.append((name, np_type, (count,)))
            struct_dtype = np.dtype(dtype_fields)
            raw = f.read(n_points * struct_dtype.itemsize)
            arr = np.frombuffer(raw, dtype=struct_dtype, count=n_points)
            xyz = np.stack([arr["x"].astype(np.float32),
                             arr["y"].astype(np.float32),
                             arr["z"].astype(np.float32)], axis=1)
            if intensity_idx is not None:
                intensity = arr["intensity"].astype(np.float32).reshape(-1, 1)
                return np.hstack([xyz, intensity])
            return xyz
        else:
            raise NotImplementedError(
                f"PCD DATA kind {data_kind!r} not supported (only 'ascii' and 'binary'). "
                f"'binary_compressed' would need LZF decompression, not implemented."
            )


def read_ply_ascii(path: str) -> np.ndarray:
    """Minimal ASCII PLY reader for point clouds with x,y,z (+ optional
    intensity) vertex properties."""
    with open(path, "r") as f:
        line = f.readline().strip()
        if line != "ply":
            raise ValueError(f"Not a PLY file: {path}")

        fmt = None
        n_vertices = None
        properties = []
        in_vertex_element = False

        while True:
            line = f.readline().strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
                in_vertex_element = True
            elif line.startswith("element") and not line.startswith("element vertex"):
                in_vertex_element = False
            elif line.startswith("property") and in_vertex_element:
                properties.append(line.split()[-1])
            elif line == "end_header":
                break

        if fmt != "ascii":
            raise NotImplementedError(
                f"Only ASCII PLY is supported in this pass, got format {fmt!r}."
            )
        if n_vertices is None:
            raise ValueError(f"PLY file {path} has no 'element vertex' declaration.")

        try:
            xi, yi, zi = properties.index("x"), properties.index("y"), properties.index("z")
        except ValueError as e:
            raise ValueError(f"PLY file {path} missing x/y/z property: {e}")
        ii = properties.index("intensity") if "intensity" in properties else None

        rows = []
        for _ in range(n_vertices):
            vals = f.readline().split()
            rows.append([float(vals[xi]), float(vals[yi]), float(vals[zi])] +
                        ([float(vals[ii])] if ii is not None else []))

        return np.array(rows, dtype=np.float32)


def load_lidar_from_pcd_dir(
    path: str,
    sensor_spec: LidarSensorSpec,
    timestamp_source: Literal["filename"] = "filename",
    lazy: bool = True,
) -> LidarLoadResult:
    """Load a LidarModel + sorted list of LidarFrame from a directory of
    .pcd or .ply files. File format is auto-detected per-file by extension."""
    warnings: list[str] = []

    files = sorted(
        f for f in glob.glob(os.path.join(path, "*"))
        if os.path.splitext(f)[1].lower() in SUPPORTED_POINTCLOUD_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No supported point cloud files found in {path!r} "
                                 f"(looked for {SUPPORTED_POINTCLOUD_EXTENSIONS})")

    raw_timestamps = [_timestamp_from_filename(f) for f in files]
    if any(np.isnan(t) for t in raw_timestamps):
        warnings.append(
            "One or more point cloud filenames were not numeric; falling back "
            "to sequential integer timestamps (0, 1, 2, ...). Timestamp-based "
            "sync with camera frames will not reflect real capture time."
        )
        raw_timestamps = [float(i) for i in range(len(files))]

    frames = [LidarFrame(timestamp=ts, path=f, points=None) for ts, f in zip(raw_timestamps, files)]

    if not lazy:
        for fr in frames:
            fr.load()

    ext_kind = "pcd_dir" if files[0].lower().endswith(".pcd") else "ply_dir"
    source = LidarSource(kind=ext_kind, path=path)
    lidar = LidarModel(source=source, sensor_spec=sensor_spec)

    # Sensor-spec fallback warnings surface here too (mirrors what
    # quality.noise_floor.resolve_floor_inputs will do later), so users see
    # them at load time rather than only deep in a metric report.
    warnings.extend(_sensor_spec_fallback_warnings(sensor_spec))

    return LidarLoadResult(lidar=lidar, frames=frames, warnings=warnings)


# ---------------------------------------------------------------------------
# rosbag reader (rosbag1 .bag and rosbag2 directories, via the pure-Python
# `rosbags` package -- no ROS/rclpy installation required)
# ---------------------------------------------------------------------------

_POINTCLOUD2_MSGTYPE = "sensor_msgs/msg/PointCloud2"

# PointField.datatype -> numpy dtype (see sensor_msgs/msg/PointField.msg)
_POINTFIELD_DATATYPE_TO_NUMPY = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


def _require_rosbags():
    """Import the optional `rosbags` dependency, raising a clear,
    actionable error (not a bare ImportError) if it's missing -- rosbag
    support is opt-in (`pip install "cam-lidar-eval[rosbag]"`) since most
    users only need image_dir/pcd_dir and shouldn't have to pull in an
    extra dependency for a source kind they don't use."""
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as e:
        raise ImportError(
            "rosbag loading requires the optional 'rosbags' package "
            "(a pure-Python rosbag1/rosbag2 reader -- no ROS/rclpy "
            "installation needed). Install it with:\n"
            "    pip install \"cam-lidar-eval[rosbag]\"\n"
            "or directly: pip install rosbags"
        ) from e
    return AnyReader


def _pointcloud2_to_array(msg) -> np.ndarray:
    """
    Parse a deserialized sensor_msgs/msg/PointCloud2 message's raw `data`
    bytes into an (N, 3) or (N, 4) numpy array (x, y, z, [intensity]),
    using the message's own `fields` (name/offset/datatype/count) to build
    a matching structured dtype -- the same manual-parsing approach
    read_pcd() above already uses for PCD's binary DATA section, so no
    ROS-specific point cloud library (sensor_msgs_py) is needed.
    """
    try:
        x_field = next(f for f in msg.fields if f.name == "x")
        y_field = next(f for f in msg.fields if f.name == "y")
        z_field = next(f for f in msg.fields if f.name == "z")
    except StopIteration:
        raise ValueError(
            "PointCloud2 message is missing a required x/y/z field "
            f"(fields present: {[f.name for f in msg.fields]})"
        )
    intensity_field = next((f for f in msg.fields if f.name == "intensity"), None)

    n_points = msg.height * msg.width
    raw = msg.data.tobytes() if hasattr(msg.data, "tobytes") else bytes(msg.data)
    endian = ">" if msg.is_bigendian else "<"

    def _read_field(field) -> np.ndarray:
        np_type = _POINTFIELD_DATATYPE_TO_NUMPY.get(field.datatype)
        if np_type is None:
            raise ValueError(f"Unsupported PointField datatype code: {field.datatype}")
        dtype = np.dtype(f"{endian}{np.dtype(np_type).kind}{np.dtype(np_type).itemsize}")
        # Each point occupies point_step bytes; the field's own value sits
        # at a fixed byte offset within each point -- a strided view over
        # the raw buffer avoids building a full structured dtype for
        # fields we don't care about (there may be many: ring, time,
        # reflectivity, etc, none of which we need here).
        return np.ndarray(
            shape=(n_points,), dtype=dtype, buffer=raw,
            offset=field.offset, strides=(msg.point_step,),
        )

    x = _read_field(x_field).astype(np.float32)
    y = _read_field(y_field).astype(np.float32)
    z = _read_field(z_field).astype(np.float32)
    if intensity_field is not None:
        intensity = _read_field(intensity_field).astype(np.float32)
        return np.stack([x, y, z, intensity], axis=1)
    return np.stack([x, y, z], axis=1)


def _stamp_to_seconds(header, fallback_ns: int) -> tuple[float, bool]:
    """Prefer the message's own header.stamp (sec + nanosec) as the
    timestamp; if it's zero (unstamped, common in bags recorded without
    a synced clock), fall back to the bag's own recv-time timestamp and
    report that a fallback was used."""
    if header is not None and (header.stamp.sec != 0 or header.stamp.nanosec != 0):
        return header.stamp.sec + header.stamp.nanosec * 1e-9, False
    return fallback_ns * 1e-9, True


def _sensor_spec_fallback_warnings(sensor_spec: LidarSensorSpec) -> list[str]:
    """Shared between load_lidar_from_pcd_dir and load_lidar_from_rosbag --
    surfaces at load time the same fallbacks quality.noise_floor.
    resolve_floor_inputs would otherwise only report deep in a metric
    report."""
    warnings: list[str] = []
    if sensor_spec.horizontal_resolution_deg is None and sensor_spec.vertical_resolution_deg is None:
        if not (sensor_spec.channels and sensor_spec.vertical_fov_deg):
            warnings.append(
                "No angular resolution info in sensor_spec (horizontal/vertical_"
                "resolution_deg or channels+vertical_fov_deg). floor(Z) will use "
                "a default value; thresholds derived from it will be unreliable."
            )
    if sensor_spec.range_accuracy_m is None:
        warnings.append(
            "No range_accuracy_m in sensor_spec. floor(Z) will use a default "
            "(2cm) range-noise assumption."
        )
    return warnings


def load_lidar_from_rosbag(
    path: str,
    sensor_spec: LidarSensorSpec,
    topic: Optional[str] = None,
) -> LidarLoadResult:
    """
    Load a LidarModel + list of LidarFrame from a rosbag by reading every
    sensor_msgs/msg/PointCloud2 message on `topic` (rosbag1 .bag file or
    rosbag2 directory, auto-detected -- both handled by the `rosbags`
    package's AnyReader).

    topic: which PointCloud2 topic to read. If None and the bag has
    exactly one PointCloud2 topic, that one is used automatically; if it
    has more than one, a ValueError lists the available topics so the
    caller can pick.

    Unlike load_lidar_from_pcd_dir, frames here are NOT lazily loaded from
    a per-frame file path -- a bag has to be read as a single sequential
    pass to demux messages by topic, so every frame's points are already
    materialized in memory by the time this function returns. For very
    large bags this means higher peak memory than the pcd_dir path; there
    is no `lazy=` parameter for that reason.

    Requires the optional `rosbags` dependency (`pip install
    "cam-lidar-eval[rosbag]"`) -- a pure-Python rosbag1/rosbag2 reader,
    not an actual ROS/rclpy installation. Live `ros_topic` (subscribing to
    a running ROS node) is a different, NOT-implemented source kind: that
    needs an active ROS2 middleware/DDS connection, which is categorically
    different from parsing an already-recorded bag file and is out of
    scope even with `rosbags` installed.
    """
    AnyReader = _require_rosbags()
    warnings: list[str] = []

    bag_path = Path(path)
    if not bag_path.exists():
        raise FileNotFoundError(f"rosbag path does not exist: {path!r}")

    frames: list[LidarFrame] = []
    used_stamp_fallback = False

    with AnyReader([bag_path]) as reader:
        pc2_conns = [c for c in reader.connections if c.msgtype == _POINTCLOUD2_MSGTYPE]
        if not pc2_conns:
            raise ValueError(
                f"No {_POINTCLOUD2_MSGTYPE} topics found in {path!r}. "
                f"Available topics: {sorted({c.topic for c in reader.connections})}"
            )
        available_topics = sorted({c.topic for c in pc2_conns})
        if topic is None:
            if len(available_topics) > 1:
                raise ValueError(
                    f"Multiple {_POINTCLOUD2_MSGTYPE} topics found in {path!r}: "
                    f"{available_topics}. Specify `topic` to pick one."
                )
            topic = available_topics[0]
        elif topic not in available_topics:
            raise ValueError(
                f"Topic {topic!r} not found (or not {_POINTCLOUD2_MSGTYPE}) in {path!r}. "
                f"Available: {available_topics}"
            )

        selected = [c for c in pc2_conns if c.topic == topic]
        for conn, bag_timestamp_ns, rawdata in reader.messages(connections=selected):
            msg = reader.deserialize(rawdata, conn.msgtype)
            points = _pointcloud2_to_array(msg)
            ts_sec, used_fallback = _stamp_to_seconds(getattr(msg, "header", None), bag_timestamp_ns)
            used_stamp_fallback = used_stamp_fallback or used_fallback
            frames.append(LidarFrame(timestamp=ts_sec, path=None, points=points))

    if not frames:
        raise ValueError(f"Topic {topic!r} in {path!r} has no messages.")
    if used_stamp_fallback:
        warnings.append(
            f"Some messages on topic {topic!r} had an unstamped header (sec=0, "
            f"nanosec=0); their bag record-time was used as the frame timestamp "
            f"instead. This is usually fine, but if the publisher applies "
            f"significant latency before publishing, sync with camera frames "
            f"may be slightly off."
        )

    frames.sort(key=lambda fr: fr.timestamp)

    source = LidarSource(kind="rosbag", path=path, topic=topic)
    lidar = LidarModel(source=source, sensor_spec=sensor_spec)
    warnings.extend(_sensor_spec_fallback_warnings(sensor_spec))

    return LidarLoadResult(lidar=lidar, frames=frames, warnings=warnings)

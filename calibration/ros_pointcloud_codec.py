"""
camera_calibrator.calibration.ros_pointcloud_codec
======================================================

sensor_msgs/PointCloud2 -> (N,3) 또는 (N,4, xyz+intensity) np.ndarray 디코딩 로직.

ros_image_codec.py와 같은 이유로 존재한다: bag 리더(rosbags로 역직렬화된
메시지)와 향후 live 구독(진짜 rospy/rclpy 메시지) 둘 다 같은 필드 이름
(height, width, point_step, is_bigendian, fields, data)을 쓰므로, 메시지
객체를 duck-typing으로 받아 하나의 구현만 유지한다.

PointField 기반 파싱 로직은 input/lidar.py의 `_pointcloud2_to_array`와
동일한 접근(필드 offset/datatype을 메시지 자체에서 읽어 strided view를
만드는 방식, ROS 전용 라이브러리인 sensor_msgs_py 불필요)을 이 모듈에
독립적으로 다시 구현한 것이다 - input/lidar.py는 별개의(camera_lidar와
무관한) 평가 서브시스템이라 그쪽에 의존성을 만들지 않기 위해서다.
"""

from __future__ import annotations

import numpy as np

# PointField.datatype -> numpy dtype (sensor_msgs/msg/PointField.msg 참고)
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


def decode_pointcloud2_message(msg) -> np.ndarray:
    """sensor_msgs/PointCloud2 (duck-typed: .height .width .fields .data
    .point_step .is_bigendian) -> (N,3) 또는 intensity 필드가 있으면 (N,4)
    np.ndarray. x/y/z는 필수, intensity는 optional로 존재하면 4번째 열에
    포함시킨다. 필드 순서/개수는 임의로 와도 되고(ring, time 등은 무시),
    메시지 자체의 fields(name/offset/datatype)를 읽어 처리한다.
    """
    try:
        x_field = next(f for f in msg.fields if f.name == "x")
        y_field = next(f for f in msg.fields if f.name == "y")
        z_field = next(f for f in msg.fields if f.name == "z")
    except StopIteration:
        raise ValueError(
            "PointCloud2 메시지에 필수 x/y/z field가 없습니다 "
            f"(존재하는 field: {[f.name for f in msg.fields]})"
        )
    intensity_field = next((f for f in msg.fields if f.name == "intensity"), None)

    n_points = msg.height * msg.width
    raw = msg.data.tobytes() if hasattr(msg.data, "tobytes") else bytes(msg.data)
    endian = ">" if msg.is_bigendian else "<"

    def _read_field(field) -> np.ndarray:
        np_type = _POINTFIELD_DATATYPE_TO_NUMPY.get(field.datatype)
        if np_type is None:
            raise ValueError(f"지원하지 않는 PointField datatype 코드: {field.datatype}")
        dtype = np.dtype(f"{endian}{np.dtype(np_type).kind}{np.dtype(np_type).itemsize}")
        # 포인트 하나가 point_step 바이트를 차지하고, 관심 있는 field는 그
        # 안의 고정 offset에 있다 - strided view로 필요 없는 다른 field
        # (ring, time, reflectivity 등)를 위한 structured dtype을 만들지
        # 않아도 된다.
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

"""ROS 2 message layouts as plain DDS types, for subscribing without ROS installed.

A ROS 2 topic is an ordinary DDS topic: the name gets an `rt` prefix and the
type a `pkg::msg::dds_::Name_` spelling, and the payload is CDR. So any DDS
implementation can read one, given the struct layout -- which is all this file
is. Only the three message types the calibrator reads are declared; their
layout has not changed from Foxy to Jazzy.

Kept apart from live.py on purpose: cyclonedds reads these annotations at class
creation, so they must be real types rather than the strings that
`from __future__ import annotations` would turn them into.
"""

from dataclasses import dataclass

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import int32, sequence, uint8, uint32


@dataclass
class Time_(IdlStruct, typename="builtin_interfaces::msg::dds_::Time_"):
    sec: int32
    nanosec: uint32


@dataclass
class Header_(IdlStruct, typename="std_msgs::msg::dds_::Header_"):
    stamp: Time_
    frame_id: str


@dataclass
class PointField_(IdlStruct, typename="sensor_msgs::msg::dds_::PointField_"):
    name: str
    offset: uint32
    datatype: uint8
    count: uint32


@dataclass
class PointCloud2_(IdlStruct, typename="sensor_msgs::msg::dds_::PointCloud2_"):
    header: Header_
    height: uint32
    width: uint32
    fields: sequence[PointField_]
    is_bigendian: bool
    point_step: uint32
    row_step: uint32
    data: sequence[uint8]
    is_dense: bool


@dataclass
class Image_(IdlStruct, typename="sensor_msgs::msg::dds_::Image_"):
    header: Header_
    height: uint32
    width: uint32
    encoding: str
    is_bigendian: uint8
    step: uint32
    data: sequence[uint8]


@dataclass
class CompressedImage_(IdlStruct, typename="sensor_msgs::msg::dds_::CompressedImage_"):
    header: Header_
    format: str
    data: sequence[uint8]


@dataclass
class PointCloud2Bytes(IdlStruct, typename="sensor_msgs::msg::dds_::PointCloud2_"):
    header: Header_
    height: uint32
    width: uint32
    fields: sequence[PointField_]
    is_bigendian: bool
    point_step: uint32
    row_step: uint32
    data: bytes
    is_dense: bool


@dataclass
class ImageBytes(IdlStruct, typename="sensor_msgs::msg::dds_::Image_"):
    header: Header_
    height: uint32
    width: uint32
    encoding: str
    is_bigendian: uint8
    step: uint32
    data: bytes


@dataclass
class CompressedImageBytes(IdlStruct, typename="sensor_msgs::msg::dds_::CompressedImage_"):
    header: Header_
    format: str
    data: bytes



# ROS name -> (wire struct for the topic, bytes twin for re-serializing)
TYPES = {
    "sensor_msgs/msg/PointCloud2": (PointCloud2_, PointCloud2Bytes),
    "sensor_msgs/msg/Image": (Image_, ImageBytes),
    "sensor_msgs/msg/CompressedImage": (CompressedImage_, CompressedImageBytes),
}


def to_cdr(sample, fast_cls) -> bytes:
    """CDR of a received sample, via its bytes twin."""
    import dataclasses

    kw = {f.name: getattr(sample, f.name) for f in dataclasses.fields(fast_cls)}
    kw["data"] = bytes(kw["data"])
    return fast_cls(**kw).serialize()

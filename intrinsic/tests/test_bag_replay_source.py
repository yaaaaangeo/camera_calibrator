"""automation live 의 bag 재생 소스(BagReplay): 시각·FPS·ROS 2 저장 형식·디코딩 오류."""

import numpy as np
import pytest

from automation.live.sources import BagReplay
from tests.test_rosbag_reader import synthetic_bag_path  # noqa: F401 - pytest fixture

def test_ros1_timestamp_fps_and_backward_compatibility(synthetic_bag_path):
    source=BagReplay(synthetic_bag_path,'/camera/image_raw',extract_fps=2)
    source.start();times=[]
    try:
        while not source.exhausted:
            frame=source.poll()
            if frame is not None:times.append(frame[1])
    finally:source.stop()
    np.testing.assert_allclose(times,[0,.5,1,1.5,2,2.5],rtol=0,atol=1e-9)

@pytest.mark.parametrize('fps',[0,-1,float('nan'),float('inf'),True,'2'])
def test_invalid_fps_rejected(tmp_path,fps):
    with pytest.raises(ValueError):BagReplay(tmp_path/'not-read.bag','/camera',extract_fps=fps)

def write_ros2(path,plugin):
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores,get_typestore
    store=get_typestore(Stores.ROS2_HUMBLE);t=store.types
    Image=t['sensor_msgs/msg/Image'];Header=t['std_msgs/msg/Header'];Time=t['builtin_interfaces/msg/Time']
    pixels=np.full((48,64,3),100,np.uint8)
    with Writer(path,version=8,storage_plugin=plugin) as writer:
        connection=writer.add_connection('/camera',Image.__msgtype__,typestore=store)
        for i in range(6):
            msg=Image(header=Header(stamp=Time(sec=0,nanosec=i*100000000),frame_id='c'),height=48,width=64,
                encoding='bgr8',is_bigendian=0,step=192,data=pixels.ravel())
            writer.write(connection,i*100000000,store.serialize_cdr(msg,Image.__msgtype__))

@pytest.mark.parametrize('kind',['db3','mcap'])
def test_ros2_storage_formats_topic_fps_and_native_resolution(tmp_path,kind):
    from rosbags.rosbag2.enums import StoragePlugin
    folder=tmp_path/'recording';write_ros2(folder,StoragePlugin.SQLITE3 if kind=='db3' else StoragePlugin.MCAP)
    path=folder if kind=='db3' else next(folder.glob('*.mcap'))
    source=BagReplay(path,'/camera',extract_fps=2);source.start();frames=[]
    try:
        while not source.exhausted:
            f=source.poll()
            if f is not None:frames.append(f)
    finally:source.stop()
    assert len(frames)==2 and all(f[0].shape==(48,64,3) for f in frames)

def test_selected_decode_error_is_not_silently_dropped(tmp_path,monkeypatch):
    from types import SimpleNamespace
    source=BagReplay(tmp_path/'unused.bag','/camera')
    connection=SimpleNamespace(msgtype='sensor_msgs/msg/Image')
    source.messages=iter([(connection,0,b'bad')])
    source.reader=SimpleNamespace(deserialize=lambda *a:None)
    monkeypatch.setattr('calibration.ros_image_codec.decode_image_message',lambda *a:None)
    with pytest.raises(ValueError,match='not silently skipped'):source.poll()

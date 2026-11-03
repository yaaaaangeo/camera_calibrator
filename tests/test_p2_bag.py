from tests.test_rosbag_reader import synthetic_bag_path


def test_bag_replay_decodes_existing_ros1_fixture(synthetic_bag_path):
    from automation.live.sources import BagReplay
    source=BagReplay(synthetic_bag_path,'/camera/image_raw'); source.start()
    frames=[]
    try:
        while not source.exhausted:
            result=source.poll()
            if result is not None: frames.append(result)
    finally: source.stop()
    assert len(frames)==30 and frames[0][0].shape==(48,64,3)


def test_bag_replay_decodes_real_ros2_recording_format(tmp_path):
    import numpy as np
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import get_typestore,Stores
    from rosbags.typesys.stores.ros2_humble import sensor_msgs__msg__Image as Image
    from rosbags.typesys.stores.ros2_humble import std_msgs__msg__Header as Header
    from rosbags.typesys.stores.ros2_humble import builtin_interfaces__msg__Time as Time
    from automation.live.sources import BagReplay
    store=get_typestore(Stores.ROS2_HUMBLE); path=tmp_path/'ros2'
    pixels=np.full((10,10,3),77,np.uint8)
    with Writer(path,version=8) as writer:
        connection=writer.add_connection('/camera',Image.__msgtype__,typestore=store)
        message=Image(header=Header(stamp=Time(sec=0,nanosec=0),frame_id='c'),height=10,width=10,
                      encoding='bgr8',is_bigendian=0,step=30,data=pixels.reshape(-1))
        writer.write(connection,0,store.serialize_cdr(message,Image.__msgtype__))
    source=BagReplay(path,'/camera'); source.start()
    try:
        actual,_=source.poll(); assert np.array_equal(actual,pixels)
        assert source.poll() is None and source.exhausted
    finally: source.stop()

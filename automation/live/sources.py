"""ROS1/ROS2 latest-only input and deterministic saved-image/rosbag replay."""
from pathlib import Path
from calibration.latest_frame import LatestFrameBuffer


class ImageReplay:
    def __init__(self, directory):
        self.paths = sorted(p for p in Path(directory).expanduser().resolve().iterdir()
                            if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'})
        if not self.paths:
            raise ValueError('No replay images')
        self.index = 0
        self.exhausted = False

    def start(self):
        pass

    def poll(self):
        import cv2
        if self.index >= len(self.paths):
            self.exhausted = True
            return None
        path = self.paths[self.index]
        self.index += 1
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Unreadable replay image: {path}')
        return image, float(self.index)

    def clear(self):
        pass

    def stop(self):
        pass


class BagReplay:
    def __init__(self, path, topic, *, extract_fps=None):
        self.path, self.topic = str(Path(path).expanduser().resolve()), topic
        if extract_fps is not None:
            import math
            if isinstance(extract_fps,bool) or not isinstance(extract_fps,(int,float)) or not math.isfinite(extract_fps) or extract_fps<=0:
                raise ValueError('extract_fps must be finite and positive')
        self.extract_fps=extract_fps
        self.last_sample_ns=None
        self.exhausted = False
        self.reader = None

    def start(self):
        from calibration.rosbag_reader import _open_reader, _require_rosbags
        _require_rosbags()
        self.exhausted=False
        self.last_sample_ns=None
        self.reader = _open_reader(self.path)
        self.reader.open()
        self.connections = [c for c in self.reader.connections if c.topic == self.topic
                            and c.msgtype in {'sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage'}]
        if not self.connections:
            self.stop()
            raise ValueError('No image messages on bag topic')
        self.messages = self.reader.messages(connections=self.connections)

    def poll(self):
        from calibration.ros_image_codec import decode_image_message
        for connection, timestamp, raw in self.messages:
            if self.extract_fps is not None and self.last_sample_ns is not None and timestamp-self.last_sample_ns < 1e9/self.extract_fps:
                continue
            image = decode_image_message(self.reader.deserialize(raw, connection.msgtype), connection.msgtype)
            if image is None:
                raise ValueError(f'Cannot decode image message on {self.topic} at {timestamp}; input not silently skipped')
            self.last_sample_ns=timestamp
            return image, timestamp / 1e9
        self.exhausted = True
        return None

    def clear(self):
        pass

    def stop(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None


class RosSource:
    exhausted = False

    def __init__(self, topic, msg_type):
        self.topic, self.msg_type = topic, msg_type
        self.buffer = LatestFrameBuffer()
        self.subscriber = None
        from collections import deque
        self.errors = deque(maxlen=16)

    def start(self):
        from calibration.ros_live import LiveTopicSubscriber
        self.subscriber = LiveTopicSubscriber()
        self.subscriber.start(self.topic, self.msg_type,
                              on_frame=lambda frame, stamp: self.buffer.put(frame.copy(), stamp),
                              on_error=self.errors.append)

    def poll(self):
        return self.buffer.take()

    def clear(self):
        self.buffer.clear()

    def stop(self):
        if self.subscriber is not None:
            self.subscriber.stop()
            self.subscriber = None
        self.clear()

"""
camera_calibrator.camera_lidar
=================================

ROS-independent FAST-Calib camera-LiDAR extrinsic calibration core.

Nothing under this package imports rospy, rclpy, rosbags, or
calibration.ros_live/calibration.rosbag_reader. Input adapters (bag
readers, live ROS subscribers, file loaders) normalize their data into
the Common Data Model in camera_lidar.types (ImageFrame, PointCloudFrame,
CalibrationScene) before calling camera_lidar.pipeline -- see that
module's docstring for the enforced dependency direction.
"""

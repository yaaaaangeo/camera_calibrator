"""
camera_calibrator.integrations
==================================

External-program adapters. This is the ONLY place in the app allowed to
know about ROS commands/environment -- camera_lidar/, geometry/, and
evaluation/ never import anything here and never gain a ROS dependency.
"""

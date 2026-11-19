"""CAMERA–LiDAR calibration room.

Parts shared with the Camera Intrinsic tool (ROS I/O, camera model files, Qt
helpers, theme, Git upload) live in ../intrinsic -- made importable here, so
every entry point (room_calib.py, the main app, tests) gets them.
"""

import os as _os
import sys as _sys

_INTRINSIC = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "intrinsic")
if _os.path.isdir(_INTRINSIC) and _INTRINSIC not in _sys.path:
    _sys.path.append(_INTRINSIC)

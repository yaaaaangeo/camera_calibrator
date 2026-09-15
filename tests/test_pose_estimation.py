"""Test 1 (계획 문서 11번) - estimate_rough_pose의 yaw/pitch/roll 부호 판별.

board_tilt_deg(detector.py의 minAreaRect 각도)는 2D in-plane 근사치일 뿐
진짜 yaw(좌우)/pitch(상하) 회전을 구분하지 못한다. estimate_rough_pose
(calibration/models/common.py)가 이를 올바르게 구분하는지, 그리고
image_selection.classify_tilt가 더 이상 board_tilt_deg만으로 좌우를
판정하지 않는지 확인한다.

순수 축 회전(Y축만=yaw, X축만=pitch, Z축만=roll)은 표준 R->Euler(XYZ)
분해에서 rvec=(x,0,0)/(0,y,0)/(0,0,z)가 각각 정확히 pitch=x, yaw=y, roll=z
로 복원된다(다른 두 축은 정확히 0) - 아래 테스트가 그 수학적 사실에
기반한다.
"""

from __future__ import annotations

import cv2
import numpy as np

from calibration.image_selection import classify_tilt
from calibration.models.common import estimate_rough_pose, rough_camera_matrix
from calibration.types import DetectionResult, Frame, FrameStatus, ImageInfo

IMG_W, IMG_H = 1920, 1080


def _board_object_points(n: int = 6) -> np.ndarray:
    xs, ys = np.meshgrid(np.linspace(-0.1, 0.1, n), np.linspace(-0.1, 0.1, n))
    pts = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    return pts.astype(np.float64)


def _project(rvec_deg_xyz: tuple[float, float, float], distance: float = 1.0):
    rvec = np.radians(np.array(rvec_deg_xyz, dtype=np.float64)).reshape(3, 1)
    tvec = np.array([[0.0], [0.0], [distance]])
    K = rough_camera_matrix((IMG_W, IMG_H))
    obj = _board_object_points()
    img, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    return obj, img.reshape(-1, 1, 2)


def test_pure_axis_rotations_recover_correct_yaw_pitch_roll_signs():
    obj, img = _project((0.0, 0.0, 0.0))
    front = estimate_rough_pose(obj, img, (IMG_W, IMG_H))
    assert front is not None
    yaw, pitch, roll, distance = front
    assert abs(yaw) < 1.0
    assert abs(pitch) < 1.0
    assert abs(roll) < 1.0
    assert abs(distance - 1.0) < 0.05

    obj, img = _project((0.0, 25.0, 0.0))
    yaw_right = estimate_rough_pose(obj, img, (IMG_W, IMG_H))[0]
    obj, img = _project((0.0, -25.0, 0.0))
    yaw_left = estimate_rough_pose(obj, img, (IMG_W, IMG_H))[0]
    assert yaw_right > 15.0
    assert yaw_left < -15.0

    obj, img = _project((25.0, 0.0, 0.0))
    pitch_a = estimate_rough_pose(obj, img, (IMG_W, IMG_H))[1]
    obj, img = _project((-25.0, 0.0, 0.0))
    pitch_b = estimate_rough_pose(obj, img, (IMG_W, IMG_H))[1]
    assert pitch_a > 15.0
    assert pitch_b < -15.0
    # pitch != yaw axis - a pure-pitch rotation must not register as yaw.
    yaw_component_of_pitch_a = estimate_rough_pose(*_project((25.0, 0.0, 0.0)), (IMG_W, IMG_H))
    assert abs(yaw_component_of_pitch_a[0]) < 2.0

    obj, img = _project((0.0, 0.0, 25.0))
    roll_cw = estimate_rough_pose(obj, img, (IMG_W, IMG_H))[2]
    obj, img = _project((0.0, 0.0, -25.0))
    roll_ccw = estimate_rough_pose(obj, img, (IMG_W, IMG_H))[2]
    assert roll_cw > 15.0
    assert roll_ccw < -15.0
    # roll (in-plane rotation) must not leak into yaw/pitch.
    yaw_pitch_of_roll = estimate_rough_pose(*_project((0.0, 0.0, 25.0)), (IMG_W, IMG_H))
    assert abs(yaw_pitch_of_roll[0]) < 2.0
    assert abs(yaw_pitch_of_roll[1]) < 2.0


def _frame_with_pose(image_id: str, *, yaw_deg: float, board_tilt_deg: float) -> Frame:
    """board_tilt_deg를 의도적으로 yaw_deg와 무관한 값(0)으로 고정해, classify_tilt가
    실제로 yaw_deg를 쓰는지(옛날처럼 board_tilt_deg만 보는 게 아닌지) 확인한다.
    """
    info = ImageInfo(image_id=image_id, path=f"/x/{image_id}.png", width=IMG_W, height=IMG_H)
    detection = DetectionResult(
        image_id=image_id, success=True,
        corners=np.zeros((24, 1, 2), dtype=np.float32), num_corners=24,
        board_center_px=(960.0, 540.0), board_area_ratio=0.2,
        board_tilt_deg=board_tilt_deg, yaw_deg=yaw_deg, pitch_deg=0.0, roll_deg=0.0,
    )
    return Frame(image_info=info, detection=detection, status=FrameStatus.DETECTED)


def test_classify_tilt_uses_yaw_not_minarearect_angle():
    # Same board_tilt_deg (0.0) for both - a minAreaRect-only classifier would
    # call both "front". yaw_deg differs and must be what drives the result.
    left = _frame_with_pose("left", yaw_deg=-30.0, board_tilt_deg=0.0)
    right = _frame_with_pose("right", yaw_deg=30.0, board_tilt_deg=0.0)
    front = _frame_with_pose("front", yaw_deg=0.0, board_tilt_deg=0.0)

    assert classify_tilt(left) == "left_tilt"
    assert classify_tilt(right) == "right_tilt"
    assert classify_tilt(front) == "front"


def test_classify_tilt_falls_back_to_board_tilt_deg_when_yaw_unavailable():
    info = ImageInfo(image_id="legacy", path="/x/legacy.png", width=IMG_W, height=IMG_H)
    detection = DetectionResult(
        image_id="legacy", success=True,
        corners=np.zeros((24, 1, 2), dtype=np.float32), num_corners=24,
        board_center_px=(960.0, 540.0), board_area_ratio=0.2,
        board_tilt_deg=20.0, yaw_deg=None, pitch_deg=None, roll_deg=None,
    )
    frame = Frame(image_info=info, detection=detection, status=FrameStatus.DETECTED)
    assert classify_tilt(frame) == "right_tilt"

"""
tests/test_ros_live.py
===========================

실시간 ROS 구독 모듈의 "ROS가 설치 안 된 환경에서도 앱이 안 죽는지"를
검증한다. 실제 rospy/rclpy 콜백 동작 자체는 이 CI 환경에 ROS를 설치할 수
없어 테스트하지 못한다 (README 5.2절에 명시된 알려진 한계) - 대신 백엔드
감지, graceful degradation, 에러 메시지가 정확한지를 확인한다.

실제 ROS 환경에서의 검증은 별도로 필요하다.
"""

from __future__ import annotations

import pytest

import calibration.ros_live as ros_live


def test_module_import_never_fails_regardless_of_ros_installation():
    """이 모듈은 ROS 미설치 환경에서도 import 자체는 항상 성공해야 한다 -
    지연 임포트(try/except)로 감싸져 있기 때문. 이미 이 파일 상단의
    import로 검증되지만, 명시적으로도 확인한다.
    """
    import importlib
    importlib.reload(ros_live)
    assert ros_live.ROS_LIVE_BACKEND in (None, "ros1", "ros2")


def test_subscriber_raises_clear_importerror_when_no_backend():
    """이 CI 환경에는 ROS가 없으므로 ROS_LIVE_BACKEND는 None이어야 하고,
    LiveTopicSubscriber() 생성 시 명확한 안내가 담긴 ImportError가 나야 한다
    (조용히 죽거나 이해 안 되는 스택트레이스를 던지면 안 됨).
    """
    if ros_live.ROS_LIVE_BACKEND is not None:
        pytest.skip("이 환경에는 실제로 ROS가 설치되어 있어 이 케이스를 재현할 수 없음")

    with pytest.raises(ImportError, match="ROS1 또는 ROS2"):
        ros_live.LiveTopicSubscriber()


def test_require_backend_message_mentions_pip_is_not_the_fix():
    """rosbags(선택적, pip 설치 가능)와 달리 rospy/rclpy는 pip로 안 된다는 걸
    사용자가 오해하지 않도록 에러 메시지에 명시되어 있어야 한다.
    """
    if ros_live.ROS_LIVE_BACKEND is not None:
        pytest.skip("이 환경에는 실제로 ROS가 설치되어 있어 이 케이스를 재현할 수 없음")

    with pytest.raises(ImportError) as exc_info:
        ros_live._require_backend()
    assert "pip" in str(exc_info.value)

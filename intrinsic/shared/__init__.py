"""Camera Intrinsic 과 CAMERA–LiDAR 가 같이 쓰는 부품.

    shared.naming     파일·폴더 이름 정리
    shared.qt         Qt 공용: 백그라운드 작업, BGR 이미지 -> QPixmap
    shared.ros        ROS 입출력: bag 읽기, 메시지 디코딩, 실시간 토픽 수신 (ROS 1 / ROS 2 / DDS)
    shared.camera     카메라 내부 파라미터 한 벌(K, D, 모델)과 OpenCV YAML 읽기/쓰기
    shared.aruco      ArUco 검출기 생성

CAMERA–LiDAR(extrinsic/gui)는 이 폴더(intrinsic/)를 sys.path 에 넣고 import 한다.
"""

# Jetson AGX Orin 64GB 설치 및 실시간 캘리브레이션

지원 기준은 JetPack 6.2.1, Ubuntu 22.04, aarch64, Python 3.10, ROS 2
Humble입니다. 일반 `requirements.txt` 대신 Jetson 전용 고정 버전을 사용합니다.

## 1. ROS 2 준비

ROS 2 Humble과 사용하는 카메라 드라이버를 먼저 설치합니다. 새 터미널마다 다음
환경을 적용해야 앱에서 `rclpy`와 카메라 토픽을 찾을 수 있습니다.

```bash
source /opt/ros/humble/setup.bash
ros2 topic list -t
```

카메라 토픽 타입은 `sensor_msgs/msg/Image` 또는
`sensor_msgs/msg/CompressedImage`여야 합니다.

## 2. 자동 설치

저장소 루트에서 실행합니다. 스크립트는 ARM64/Ubuntu/ROS 버전을 먼저 검사하고,
Qt 런타임 라이브러리와 `.venv-jetson`을 설치한 뒤 preflight까지 실행합니다.

```bash
source /opt/ros/humble/setup.bash
chmod +x scripts/install_jetson.sh
./scripts/install_jetson.sh
```

설치 후 실행:

```bash
source /opt/ros/humble/setup.bash
source .venv-jetson/bin/activate
camera-calibrator --verbose
```

`--system-site-packages`로 가상환경을 만드는 이유는 ROS apt 패키지의 `rclpy`와
`sensor_msgs`를 가상환경에서도 사용하기 위해서입니다. `pip install rclpy`로
대체하지 마십시오.

## 3. 고정한 ARM64 패키지

- `PySide6==6.7.3`: ARM64 wheel이 `manylinux_2_31`이므로 JetPack 6의 glibc
  2.35에서 동작합니다.
- `opencv-contrib-python-headless==4.10.0.84`: ARM64 manylinux2014 wheel이며
  ChArUco/ArUco를 포함합니다. OpenCV Qt5를 제외해 PySide6 Qt6와의 plugin
  충돌 및 프로세스 abort 가능성을 낮춥니다.
- `numpy==1.26.4`: ROS Humble/Jammy 바이너리 확장과 안정적으로 조합되는
  NumPy 1.x를 유지합니다.

일반 `pip install -e .`를 다시 실행하면 GUI 포함 OpenCV가 추가될 수 있습니다.
Jetson에서는 반드시 다음처럼 dependency 설치를 차단합니다.

```bash
python -m pip install --no-deps -e .
```

## 4. 실시간 토픽 권장 설정

- 캘리브레이션을 실제 사용할 해상도와 crop/ISP 모드로 발행합니다.
- 캘리브레이션 촬영에는 고 FPS가 필요 없으므로 5~15 FPS를 권장합니다.
- 앱 프리뷰는 최대 10 FPS이며, 표시보다 빨리 도착한 프레임은 최신 한 장만
  남기고 폐기합니다. 화면 아래에서 수신/표시/폐기 수를 확인할 수 있습니다.
- 지원 인코딩: BGR/RGB/mono/Bayer/YUV422/NV12/NV21 및 JPEG/PNG compressed.
- ROS 2 subscriber는 sensor-data QoS(Best Effort)를 사용합니다.

## 5. 현장 확인

```bash
source /opt/ros/humble/setup.bash
source .venv-jetson/bin/activate
python scripts/jetson_preflight.py
ros2 topic list -t
ros2 topic hz /camera/image_raw
camera-calibrator --verbose --log-file jetson-live.log
```

앱에서 `실시간 카메라 구독`을 열고 토픽을 고른 뒤 다음을 확인합니다.

1. 프리뷰가 30초 이상 멈추지 않는다.
2. 폐기 수가 증가해도 UI 입력과 창 이동이 즉시 반응한다.
3. 수동 캡처 20장 이상이 저장되고 Coverage 안내가 갱신된다.
4. 완료 후 Standard 4모델 캘리브레이션과 결과 탭이 생성된다.

실제 Jetson 장치와 카메라 드라이버 조합은 이 저장소의 일반 CI에서 재현할 수
없으므로, 위 현장 확인을 배포 승인 조건으로 사용합니다.

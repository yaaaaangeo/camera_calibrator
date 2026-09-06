# Camera Calibration Tool

[![Tests](https://github.com/yaaaaangeo/camera_calibrator/actions/workflows/tests.yml/badge.svg)](https://github.com/yaaaaangeo/camera_calibrator/actions/workflows/tests.yml)

**Standard Calibration**으로 Ideal Pinhole / Brown-Conrady / Rational
(`extended_pinhole`) / Fisheye(Kannala-Brandt) 네 모델을 ChArUco·Chessboard·Circle
Grid·AprilGrid 중 원하는 패턴으로 동시에 캘리브레이션하고, Hold-out 검증 +
Model Score(AIC/BIC) 기반으로 근거 있는 추천을 해주는 도구. Checkerboard/Circle
Grid에는 **Advanced Calibration(Object-Releasing)** — 카메라 파라미터와 함께
타겟 형상 자체도 함께 보정하는 `cv2.calibrateCameraRO` 기반 고급 모드 — 도
별도로 제공한다 (4번 섹션 참고).

`windshield` 브랜치에는 이 위에 얹는 **Windshield Refraction Calibration**
계층(Camera Intrinsic은 고정한 채 windshield 굴절만 별도로 모델링 + 반사/
유령상 photometric 평가)도 있다 - 12번 섹션 참고.

## 1. 요구 사항

- **Python 3.10 이상** (3.11 권장)
- OS: Windows / macOS / Linux 모두 가능 (PySide6가 크로스플랫폼)
- **OpenCV**: 4.7 이상, **5.0.0도 지원** (`opencv-contrib-python==5.0.0.93`으로
  실제 검증됨). 4.x와 5.x 둘 다에서 전체 테스트 스위트가 통과한다 - OpenCV
  5.0에서 `cv2.fisheye.CALIB_*` 플래그 위치가 바뀌고 `cv2.fisheye.calibrate()`의
  요구 shape이 엄격해진 것에 대응하는 코드가 `calibration/models/fisheye.py`에
  있다 (자세한 내용은 11번 섹션).

## 2. 설치

**방법 A - requirements.txt (기존 방식)**

```bash
# 1) 이 폴더(camera_calibrator/)로 이동
cd camera_calibrator

# 2) (권장) 가상환경 생성
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3) 의존성 설치
pip install -r requirements.txt
```

`requirements.txt`가 설치하는 것:

| 패키지 | 용도 |
|---|---|
| `opencv-contrib-python` | ChArUco 검출, 캘리브레이션 계산 (cv2.aruco는 contrib에만 있음) |
| `numpy` | 행렬/배열 연산 |
| `PyYAML` | ROS CameraInfo YAML export |
| `PySide6` | 데스크톱 UI (Qt6) |
| `rosbags` | (선택) rosbag(.bag/.db3/.mcap)에서 이미지 직접 불러오기. 순수 Python이라 ROS 설치 불필요 |

> ⚠️ `opencv-python`과 `opencv-contrib-python`을 **동시에 설치하면 안 됩니다**
> (둘 다 `cv2`라는 이름을 써서 충돌합니다). 이미 `opencv-python`이 깔려있다면
> `pip uninstall opencv-python opencv-python-headless` 먼저 실행하세요.

**방법 B - pyproject.toml (패키지로 설치, `camera-calibrator` 커맨드 사용 가능)**

```bash
pip install -e .            # 기본 설치
pip install -e ".[ros]"     # rosbag 기능까지 포함
pip install -e ".[dev]"     # 테스트(pytest)까지 포함
```

이렇게 설치하면 레포 폴더 밖 어디서든 아래 커맨드로 바로 실행할 수 있습니다
(3번 섹션 참고). `requirements.txt` 방식과 설치되는 의존성은 동일하며, 어느
쪽을 쓰든 상관없습니다 - 개발/기여 목적이면 방법 B, 그냥 써보는 목적이면
방법 A가 조금 더 단순합니다.

**방법 C - Docker (설치 없이 바로, CI/배치 처리에 특히 유용)**

이 섹션이 스스로 경고하는 함정들(opencv-python과 opencv-contrib-python 동시
설치 금지, PySide6의 시스템 라이브러리 의존성 등)을 매번 새로 겪고 싶지 않다면
Docker가 제일 간단합니다:

```bash
docker build -t camera-calibrator .

docker run --rm \
  -v "$(pwd)/photos:/data/photos:ro" \
  -v "$(pwd)/out:/data/out" \
  camera-calibrator \
  --images /data/photos --squares-x 7 --squares-y 5 \
  --square-size 0.04 --marker-size 0.03 --output-dir /data/out
```

기본 이미지는 헤드리스 CLI 전용입니다 (컨테이너 안에 디스플레이가 없어
`QT_QPA_PLATFORM=offscreen`이 기본값). GUI를 실제로 띄우거나, VS Code에서 이
프로젝트를 곧바로 개발 컨테이너로 열고 싶다면 `Dockerfile` 상단 주석과
`.devcontainer/devcontainer.json`을 참고하세요 (X11 포워딩 방법 포함).

**방법 D - JetPack 6.2.1 / Jetson AGX Orin**

JetPack의 Ubuntu 22.04 ARM64에서는 일반 requirements 대신 호환 wheel을 고정한
자동 설치 스크립트를 사용합니다.

```bash
source /opt/ros/humble/setup.bash
./scripts/install_jetson.sh
```

자세한 전제 조건, 실행법과 카메라 현장 검증은 [JETSON.md](JETSON.md)에 있습니다.

**방법 E - JetPack 5.1.2 / Jetson AGX Orin Developer Kit 64GB**

JetPack 5.1.2는 Jetson Linux R35.4.1 계열, Ubuntu 20.04, aarch64 환경입니다.
이 프로젝트의 JetPack 5.1.2 지원 기준 Python은 3.10입니다. Ubuntu 20.04 기본
Python 3.8을 대상으로 dependency를 낮추지는 않습니다.

```bash
python3.10 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-jetson-jp512.txt
python -m pip install -e . --no-deps

python scripts/jetson_jp512_preflight.py
python -m app.main
```

이 수동 절차를 자동화한 `scripts/install_jetson_jp512.sh`도 있다(aarch64/
Ubuntu 20.04/`python3.10` 존재 여부를 먼저 확인하고, 위 단계를 그대로
수행한다). JetPack 6 설치 스크립트(`install_jetson.sh`)와 달리
**`--system-site-packages`를 쓰지 않는다** - ROS1 Noetic의 apt 패키지
(`rospy`/`sensor_msgs`/`cv_bridge`)는 Ubuntu 20.04 기본 Python 3.8용으로
빌드되어 있어 이 프로젝트가 요구하는 Python 3.10 venv와 ABI가 맞지 않는다
(특히 `cv_bridge`는 컴파일된 확장이라 3.8용 바이너리를 3.10 인터프리터가
아예 로드할 수 없다):

```bash
./scripts/install_jetson_jp512.sh
```

JetPack 5.1.2 profile은 GUI를 PySide6로 띄우기 때문에 OpenCV는
`opencv-contrib-python-headless`를 사용합니다. 이 프로젝트는 ChArUco/ArUco,
AprilGrid, Circle Grid, fisheye, `calibrateCameraRO`를 쓰므로 contrib build가
필요하지만, OpenCV가 번들한 Qt와 PySide6의 Qt runtime을 섞지 않기 위해 headless
wheel을 우선합니다. JetPack system OpenCV를 같은 venv 안에 섞어 쓰지 마세요.

PySide6 GUI가 시작되지 않으면 Qt/XCB runtime library를 먼저 확인하세요.

```bash
sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libegl1 libgl1
```

ROS1 Noetic live topic 사용은 pip dependency가 아닙니다. `rospy`, `sensor_msgs`,
`cv_bridge`는 ROS/apt 환경에서 관리하고, 필요할 때 ROS 환경을 source한 뒤 실행합니다.
오프라인 bag 파일 읽기용 `rosbags`도 이 JetPack 5.1.2 core+GUI profile에는 넣지
않았습니다.

## 3. 실행

```bash
python -m app.main
```

`pyproject.toml`로 설치했다면(2번 방법 B) 이 커맨드로도 동일하게 실행됩니다:

```bash
camera-calibrator
```

창이 뜨면:
1. 상단에서 해상도(Width/Height)와 패턴 정보를 입력. **Pattern type**을
   ChArUco(기본, 권장), Chessboard(일반 체스보드), Circle Grid(Symmetric/
   Asymmetric), AprilGrid 중 고를 수 있다. Chessboard/Circle Grid를 고르면
   Marker size/Dictionary 입력칸이 자동으로 숨겨진다 (그 패턴엔 필요 없으므로).
   ChArUco/AprilGrid는 사각형(또는 태그) 개수/한 칸 크기(mm)/마커 크기(mm)/
   dictionary가 필요하고, Chessboard/Circle Grid는 사각형(또는 원) 개수/
   한 칸 크기(mm)만 있으면 된다 (Circle Grid는 추가로 Grid type을
   Symmetric/Asymmetric 중에서 고른다). AprilGrid는 Kalibr과 같은 row-major ID
   배치(왼쪽 위 0번부터 오른쪽으로 증가, 다음 줄로 이동)를 가정하고, **AprilGrid
   variant**를 OpenCV/AprilTag3 style(기본) 또는 Kalibr style(experimental) 중
   고를 수 있다 — 오늘 시점 둘 다 같은 OpenCV AprilTag detector 경로를 쓰고
   variant는 로그 문구만 다르다 (4번 섹션의 AprilGrid 표 참고). 앱 내부에서는
   OpenCV `DICT_APRILTAG_*` dictionary로 기본 검출을 수행할 수 있고,
   `--export kalibr`로 Kalibr 공식 `aprilgrid` target YAML도 만들 수 있다.

   > ⚠️ **Chessboard를 쓸 때 주의**: 일반 체스보드는 대칭 패턴이라 ChArUco와
   > 달리 (1) 보드 전체가 이미지 안에 다 보여야 검출되고, (2) "어느 쪽이
   > 진짜 첫 번째 코너인지"를 원리적으로 구분할 방법이 없다 - 같은 보드를
   > 정방향으로 찍든 180도 돌려서 찍든 육안으로는 구분도 안 되는데, 촬영
   > 방향이 데이터셋 안에서 뒤섞이면 캘리브레이션이 심하게 틀어질 수 있다.
   > 이건 이 프로젝트가 만든 문제가 아니라 OpenCV 표준 체스보드 캘리브레이션
   > 자체의 잘 알려진 한계다 (설계 문서 2번이 애초에 ChArUco를 우선한 이유이기도
   > 하다). 가능하면 ChArUco를 쓰고, 꼭 체스보드를 써야 한다면 촬영 내내
   > 보드 방향을 일관되게 유지할 것.
2. 프로그램 실행 후 **Calibration Home**에서 작업 종류 선택:
   - **Camera Intrinsic**: 기존 K/D 캘리브레이션 workspace로 진입
3. Camera Setup 패널은 왼쪽/중앙/오른쪽 3열로 구성된다.
   - **왼쪽(Camera)**: Camera Name → **INPUT**(한 줄에 **[실시간]**·**[rosbag]**·
     **[이미지]** 3버튼 - 실시간은 ROS1/ROS2 이미지 토픽을 실시간 구독해 라이브
     프리뷰를 보며 직접 캡처(실제 ROS1/ROS2 설치 필요), rosbag은 ROS1 .bag/ROS2
     .db3·.mcap 로그에서 이미지 토픽을 뽑아 자동 추출(rospy/rclpy 설치 불필요),
     이미지는 jpg/png/bmp 파일을 직접 선택) → **[해상도 확인]**(JPEG/PNG 이미지
     한 장을 실제로 디코딩해 크기를 확인하고 Width/Height에 자동 반영) →
     Width/Height(한 줄)
   - **중앙(Pattern)**: **Calibration method**(Standard/Object-Releasing, 아래
     4번 섹션 참고) → Pattern type/Squares X·Y/Square size/Marker size/
     Dictionary/Grid type/AprilGrid variant. Object-Releasing을 고르면 지원
     대상(Checkerboard/Circle Grid)이 아닌 패턴에서는 안내 문구가 뜨고 실행이
     막힌다.
   - **오른쪽(Actions)**: **[캘리브레이션 실행]** → **[Export]**(계산된 모델을
     골라 OpenCV YAML로 저장 - 예전 Export 탭과 동일한 기능) → **[취소]**
     (코너 검출/모델 계산이 진행 중일 때 즉시 중단하고, 원하는 데이터/설정으로
     다시 실행할 수 있는 상태로 되돌림). Rational은 별도 Standard Camera Model이며
     항상 8계수로 계산된다. Standard 계산은 Ideal Pinhole/Brown-Conrady/
     Rational/Fisheye 네 모델을 함께 계산한다 (4번 섹션 참고).
4. **[캘리브레이션 실행]** 클릭 → 검출 → Standard 4모델 계산(+ Object-Releasing을
   골랐다면 Advanced 결과도 함께) → Hold-out → 추천까지 자동 진행
5. 탭을 넘기며 결과 확인:
   - **① Dataset**: Coverage Map(4×4 커버리지 맵) → Dataset Diversity(다양성 점수 +
     Overall Dataset Score) → Batch(이미지별 상태/코너 수/재투영 오차/
     **품질 점수(Frame Quality Score)·등급**) 순서로 한 화면에 모아 보여준다.
   - **② Preview**: 이미지 하나를 골라 원본과 (왜곡 보정 후 + Straightness
     오버레이)를 나란히 보여준다 - 옛 Undistort Preview와 Straightness Map을 한
     화면으로 합치고 Edge Error Map은 제거했다. 이미지 아래에는
     "Line Straightness — 보정 전 Xpx → 보정 후 Ypx (개선율 N%)" 한 줄만 표시된다.
   - **③ Model Comparison**: Standard 4모델 비교표(Train/Test/P95/Edge/Radial/
     AIC/BIC/Stability/Observability) + 추천 이유. Object-Releasing을 골랐다면
     그 아래 **Advanced Calibration** 패널에 Object-Releasing 결과 + 전용
     Hold-out Validation(Train/Test Frames, RMSE/Median/P95/P99/Max) +
     Standard Brown-Conrady와의 공정 비교표(같은 full-board 데이터셋/같은
     train-test 분할, 4번 섹션 참고)가 함께 표시된다. Standard Hold-out/AIC/BIC와는
     완전히 분리된 결과다.
6. 상단 메뉴 **파일 → 프로젝트 저장(Ctrl+S)** 으로 지금까지의 전체 상태(데이터셋,
   Standard 4모델 결과, Object-Releasing 결과/검증/비교, 추천)를 `.ccproj`
   파일로 저장할 수 있다. **파일 → 프로젝트 불러오기(Ctrl+O)** 로 나중에 이어서
   작업 가능 — 원본 이미지 파일이 없어져도 재계산/export는 그대로 된다
   (자세한 내용은 5번 폴더 구조의 `project_io.py` 설명 참고).

## 4. Camera Models / Calibration Targets / Calibration Method

### Camera Models (Standard Calibration, 4개)

| 모델 | 계수 | 설명 |
|---|---|---|
| **Ideal Pinhole** | `fx fy cx cy`, D=0 | 왜곡 없음을 가정하는 특수 목적 모델. |
| **Brown-Conrady** | `fx fy cx cy` + `k1 k2 p1 p2 k3` (항상 5계수 - runtime toggle 없음) | 일반적인 raw camera image에 쓰는 기본 모델. |
| **Rational** (내부 식별자 `extended_pinhole`) | `fx fy cx cy` + `k1 k2 p1 p2 k3` + `k4 k5 k6` (항상 8계수 - runtime toggle 없음) | 강한 방사 왜곡을 더 정밀하게 잡는 모델. |
| **Fisheye** (Kannala-Brandt) | `fx fy cx cy` + `k1 k2 k3 k4` | 초광각 렌즈용 `cv2.fisheye` 모델. |

네 모델을 항상 같은 데이터셋/같은 train-test 분할로 동시에 계산하고 비교한다
(`calibration/compare.py::run_all_models`).

### Calibration Targets

| 타겟 | 상태 |
|---|---|
| Checkerboard(일반 체스보드) | 지원. 방향 모호성 있음 (3번 섹션 주의사항 참고). |
| ChArUco | 지원, 기본값/권장. |
| Circle Grid | 지원, Symmetric/Asymmetric 둘 다. |
| AprilGrid | 지원. Variant: **OpenCV / AprilTag3 style**(기본) 또는 **Kalibr style
  [Experimental]** — 오늘 시점 두 variant는 동일한 OpenCV AprilTag detector
  경로를 쓰고 로그 문구만 다르다 (`calibration/detector.py::detect_aprilgrid`).
  Kalibr 스타일로 생성된 실제 타겟에 대한 회귀 테스트가
  `tests/test_kalibr_aprilgrid_fixture.py` + `tests/assets/aprilgrid/`에
  있으며(fixture 출처/한계는 그 폴더의 README 참고), 그 검증이 실제 Kalibr
  산출물로 재확인되기 전까지는 "Kalibr Compatible"로 승격하지 않고
  Experimental 표시를 유지한다. |

### Calibration Method: Standard vs Object-Releasing (Advanced)

- **Standard**: 위 4개 모델을 계산하고, Hold-out Validation + AIC/BIC 기반
  추천까지 수행한다. 모든 타겟에서 지원된다.
- **Object-Releasing** (Advanced): `cv2.calibrateCameraRO`로 카메라 파라미터와
  **캘리브레이션 타겟 형상(target geometry)** 을 함께 추정한다. 타겟 인쇄/부착
  오차까지 보정하고 싶은 고정밀 캘리브레이션용 모드다.
  - **지원 타겟**: Checkerboard, Circle Grid만. ChArUco/AprilGrid는 지원하지
    않는다 (동일 개수의 포인트를 프레임마다 정확히 같은 순서로 대응시켜야
    하는데, ChArUco/AprilGrid는 부분 검출이 흔해 이 전제를 보장하기 어렵기
    때문 - 지원하지 않는 패턴을 고르면 UI/CLI 모두 명확한 오류로 막는다).
  - **Full-board 검출 필수**: 타겟 전체가 빠짐없이, 매 프레임 동일한 포인트
    ID·순서로 검출된 프레임만 쓴다. 일부만 보이는 프레임은 이유와 함께
    제외된다.
  - **Standard 결과와 분리**: Object-Releasing 결과/Hold-out
    Validation/Standard-vs-Object-Releasing 비교는 전부 Standard 4모델의
    Hold-out/AIC/BIC와 별도로 계산·표시된다 - 서로 섞이지 않는다
    (`calibration/object_releasing_validation.py`).
  - **Object-Releasing 전용 Hold-out**: Train 프레임만으로 K/D/Refined Target
    Geometry를 확정하고, Test 프레임에는 그 셋을 전부 고정한 채 pose(solvePnP)만
    다시 구해 RMSE/Median/P95/P99/Max를 계산한다 - Test 데이터로
    `calibrateCameraRO`를 다시 부르는 일은 없다.
  - **Standard Brown-Conrady와의 비교**: 반드시 같은 full-board eligible
    데이터셋 + 같은 train/test 분할로만 비교한다 (그렇지 않은 비교는 UI/CLI에
    표시하지 않는다). "RO가 더 정확하다" 같은 자동 판정은 하지 않고, RMSE
    변화율/파라미터 delta 같은 사실만 보여준다. Target geometry refinement가
    과도하게 크면(명목 간격 대비) 경고를 표시한다.

## 5. 폴더 구조

```
camera_calibrator/
├── app/
│   ├── main.py                # GUI 실행 진입점 (python -m app.main)
│   └── cli.py                 # 헤드리스 CLI 진입점 (python -m app.cli), CI/배치용
├── calibration/              # 순수 계산 로직 (UI 의존성 없음)
│   ├── types.py              # 전체가 공유하는 데이터 구조
│   ├── detector.py           # ChArUco/Chessboard/Circle Grid/AprilGrid 검출
│   ├── models/                # pinhole / brown_conrady / extended_pinhole /
│   │                          #   fisheye / object_releasing (Advanced)
│   ├── compare.py            # Standard 4모델(Ideal Pinhole/Brown-Conrady/
│   │                          #   Rational/Fisheye) 동시 실행 + 비교표
│   ├── validation.py         # Standard Hold-out 검증 (+ Line Straightness)
│   ├── object_releasing_validation.py  # Object-Releasing 전용 Hold-out +
│   │                          #   Standard Brown-Conrady와의 공정 비교
│   ├── outlier.py            # 이상치 탐지/제거
│   ├── quality.py            # Coverage Map / 데이터셋 다양성
│   ├── frame_quality.py      # 프레임별 품질 점수 (Detection + Geometric)
│   ├── radial_profile.py     # Edge Error Map (반지름별 재투영 오차)
│   ├── straightness.py       # Line Straightness Residual (ChArUco 격자 재활용,
│   │                          #   compute_frame_straightness_lines()가 행/열별 상세 제공)
│   ├── rosbag_reader.py      # ROS1(.bag)/ROS2(.db3, .mcap)에서 이미지 추출
│   ├── ros_live.py           # 실시간 ROS1(rospy)/ROS2(rclpy) 토픽 구독 (자동 감지)
│   ├── ros_image_codec.py    # sensor_msgs/Image·CompressedImage 디코딩 (위 둘이 공유)
│   ├── calibration_io.py     # 외부 calibration 포맷을 benchmark용 표준 schema로 정규화
│   ├── benchmark_compatibility.py # Reference/Candidate 비교 전 compatibility 검사
│   ├── project_io.py         # .ccproj 프로젝트 저장/불러오기 (JSON, pickle 미사용)
│   ├── json_utils.py         # dataclass/numpy -> JSON 안전 변환 (project_io.py, export/json_export.py 공유)
│   └── recommender.py        # Model Score 기반 추천 + 최종 결과(FinalResult) 조립
├── export/                   # OpenCV YAML / ROS CameraInfo YAML / HTML 리포트 / JSON / CSV
│   ├── opencv.py
│   ├── ros.py
│   ├── report.py             # 종합 HTML 리포트 (브라우저 인쇄로 PDF 변환 가능)
│   ├── json_export.py        # 구조화된 JSON (카메라 행렬·오차 지표·최종 등급, 외부 도구 연동용)
│   └── csv_export.py         # 이미지별 상세 데이터 CSV (스프레드시트 분석용)
├── ui/                       # PySide6 화면 (계산 로직 없음, calibration/*만 호출)
│   ├── undistort_straightness_view.py  # ② Preview 탭 (원본 | 보정 후+Straightness 오버레이)
│   └── live_capture_dialog.py   # 실시간 구독 + 라이브 프리뷰 + 수동/자동 캡처 + 구역별 다양성 코칭
├── .github/workflows/tests.yml  # GitHub Actions CI (push/PR마다 Python 3.10/3.11/3.12 자동 테스트 + 커버리지)
├── .devcontainer/devcontainer.json  # VS Code/Codespaces 개발 컨테이너 (아래 Dockerfile 재사용)
├── Dockerfile                    # 헤드리스 CLI 실행 / 개발 컨테이너 베이스
├── .dockerignore
├── .gitignore                   # __pycache__, venv, .ccproj 등 로컬 산출물 제외
├── pyproject.toml                # 패키징 메타데이터 (pip install -e ., camera-calibrator 콘솔 커맨드)
├── requirements.txt
└── requirements-dev.txt          # 테스트(pytest, pytest-cov) 실행용 추가 의존성
```

`calibration/`은 UI와 완전히 독립적이라, CLI 스크립트나 다른 프론트엔드에서도
그대로 재사용할 수 있습니다.

## 6. ROS 연동

두 단계로 나뉩니다.

### 5.1 rosbag에서 이미지 불러오기 (`[rosbag에서 불러오기]` 버튼, ROS 설치 불필요)

순수 Python 라이브러리 `rosbags`로 ROS1(.bag)/ROS2(.db3, .mcap)를 직접 읽습니다.
ROS가 설치 안 된 컴퓨터에서도 동작합니다.

> **알려진 이슈 (해결됨)**: `ros2 bag record`로 녹화한 bag은 메시지 타입 정의가
> bag 안에 통째로 안 담기는 경우가 흔한데, 그런 bag을 열면
> `Bag contains no type definitions. Instantiate AnyReader with a
> default_typestore argument.` 에러가 났었습니다. `AnyReader`에
> `default_typestore=get_typestore(Stores.LATEST)`를 넘기도록 고쳤습니다
> (`calibration/rosbag_reader.py`) - sensor_msgs/Image, CompressedImage는
> ROS2 배포판이 달라도 정의가 동일해 어떤 배포판의 bag이든 문제없습니다.

지원하는 이미지 인코딩(`calibration/ros_image_codec.py`):
`mono8`, `mono16`, `bgr8`, `rgb8`, `bgra8`, `rgba8`,
`bayer_rggb8/bggr8/gbrg8/grbg8`(+16비트 버전), `yuv422`/`yuv422_yuy2`/`yuyv`/`uyvy`/`yuy2`
(YUYV·UYVY 계열), `NV12`/`NV21`(Jetson CSI/GStreamer 계열),
CompressedImage(jpeg/png).
지원 안 하는 인코딩을 만나면 에러 메시지에 실제 인코딩 이름이 나옵니다
(예: "발견된 인코딩: 32FC1") - 필요하면 이슈로 알려주시면 추가하겠습니다.

### 5.2 실시간 토픽 구독 (`[실시간 카메라 구독]` 버튼, **ROS1 또는 ROS2 설치 필요**)

이건 다릅니다 - `rospy`/`rclpy`는 pip로 설치되지 않고, 실제 ROS1(noetic 등) 또는
ROS2(humble 등)가 컴퓨터에 설치되고 환경이 source 되어 있어야만 동작합니다
(예: `source /opt/ros/noetic/setup.bash`). ROS1/ROS2 어느 쪽이 설치돼 있는지는
`calibration/ros_live.py`가 자동으로 감지합니다 (`rospy` 먼저 시도 -> 없으면
`rclpy` 시도 -> 둘 다 없으면 버튼을 눌러도 안내 메시지만 뜨고 앱은 정상 동작).

동작 방식: 토픽을 구독하면서 라이브 프리뷰를 보여주고, `[📸 캡처]` 버튼을 누른
시점의 프레임을 저장합니다 (자동 전체 녹화가 아니라 원하는 자세에서 직접
캡처하는 방식 - 설계 문서 7번, 장수보다 자세 다양성이 중요하기 때문). 편의를
위해 "N초마다 자동 캡처" 옵션도 있습니다.

고속/4K 토픽은 Qt 이벤트 큐에 프레임을 계속 쌓지 않습니다. 스레드 안전한 단일
슬롯에 **최신 프레임 한 장만** 유지하고 프리뷰를 최대 10 FPS로 제한합니다.
원본 해상도의 최신 프레임은 그대로 캡처하며, 화면 아래에서 수신/표시/폐기 수를
확인할 수 있습니다. Jetson CSI/GStreamer 토픽을 위해 NV12/NV21도 지원합니다.

JetPack 6.2.1 + ROS 2 Humble 설치는 일반 requirements 대신 ARM64 호환 버전을
고정해야 합니다. 자동 설치와 현장 검증 절차는 [JETSON.md](JETSON.md)를 따르세요.

> **알려진 이슈 (부분 해결)**: 환경이 감지되고 토픽도 맞게 골랐는데 "프레임
> 수신 대기 중..."에서 멈추는 경우가 있었습니다. 원인 중 하나를 찾아 고쳤습니다:
> 카메라가 당시 지원하지 않던 인코딩으로 발행하면 프레임이 실제로 도착해도
> 디코딩에 실패해 **조용히 버려지고 있었습니다** - 이제 디코딩 실패 시 화면에
> "⚠ 프레임은 도착했지만 디코딩에 실패했습니다 (encoding='...')" 라고 표시됩니다
> (3초 rate-limit). 그래도 계속 멈춰있다면 다른 원인(토픽에 실제로 발행이
>없거나, ROS 네트워크 설정 문제 등)일 수 있습니다.
>
> ⚠️ `calibration/ros_live.py`의 rospy/rclpy 경로는 실제 ROS 런타임(roscore 또는
> ROS2 데몬)이 있어야만 끝까지 검증할 수 있어, 개발 과정에서 end-to-end로
> 테스트하지 못했습니다. 표준 API 기준으로 작성했지만, 실제 ROS 환경에서
> 한 번 확인해보시는 걸 권장합니다. 문제가 있으면 이슈로 알려주세요.

## 7. 프로젝트 저장/불러오기 (`.ccproj`)

데이터셋이 크거나 캘리브레이션에 시간이 걸릴 때, 앱을 껐다 켜도(또는 CLI를
여러 번 나눠 실행해도) 이어서 작업할 수 있다.

- **저장되는 것**: 카메라/패턴 설정, 데이터셋(이미지별 검출 결과·품질 점수·상태),
  Standard 4모델 캘리브레이션 결과, Hold-out Validation 결과, 추천 점수, 이상치
  제거 이력, 최종 결과, (Object-Releasing을 썼다면) Advanced 결과 + 전용
  Hold-out + Standard와의 비교 결과까지 - 사실상 화면에 보이는 모든 것.
- **저장 안 되는 것**: 원본 이미지 파일 자체(바이트 복사 안 함, 경로만 저장) -
  설계 문서 9번의 "파일을 삭제/복제하지 않는다" 원칙과 같은 이유. 그래서
  **원본 이미지가 없어지거나 옮겨져도** 불러오기 자체는 되고, 재계산(이상치 제거
  등)이나 export도 그대로 된다 - 다만 UI의 "Undistort Preview" 탭만 해당
  이미지를 못 보여준다.
- **포맷**: JSON (pickle 아님 - `.ccproj`는 나중에 공유하거나 버전관리에 올릴 수도
  있는 파일이라, pickle의 "불러올 때 임의 코드 실행 위험"을 피하려고 일부러
  더 번거로운 JSON 직렬화를 택했다).

```bash
# CLI
python -m app.cli --images ./photos --squares-x 7 --squares-y 5 \
  --square-size 0.04 --marker-size 0.03 --save-project ./session.ccproj

python -m app.cli --load-project ./session.ccproj --outlier --output-dir ./out
```

UI에서는 메뉴 **파일 → 프로젝트 저장/불러오기** (`Ctrl+S` / `Ctrl+O`).

## 8. UI 없이 쓰기 (CLI / Python API)

### 8.1 CLI (`app/cli.py`) — CI/서버/배치 처리용

UI를 안 띄우고 헤드리스로 전체 파이프라인(검출→Standard 4모델→검증→추천→export,
`--calibration-method object_releasing`이면 Advanced 결과/Hold-out/비교도 포함)을
한 번에 돌린다. 종료 코드로 성공/실패를 판단할 수 있어 CI 파이프라인에
바로 끼워 넣기 좋다 (0=성공, 1=입력 문제, 2=전 모델 캘리브레이션 실패 또는
Object-Releasing 계산 실패).

```bash
python -m app.cli \
  --images ./photos \
  --squares-x 7 --squares-y 5 --square-size 0.04 --marker-size 0.03 \
  --output-dir ./out \
  --json-summary ./out/summary.json
```

종합 진단 리포트가 필요하면 문서형 옵션을 그대로 쓰면 된다:

```bash
python -m app.cli \
  --images ./photos \
  --squares-x 7 --squares-y 5 --square-size 0.04 --marker-size 0.03 \
  --diagnostic \
  --cross-validation 5 \
  --bootstrap 100 \
  --jobs 0 \
  --output-dir ./out
```

`--diagnostic`은 별도 지정이 없으면 5-fold cross validation과 100회 bootstrap을
켜고, `report/json/csv` 산출물을 함께 만든다.

주요 옵션:

| 옵션 | 설명 |
|---|---|
| `--config PATH` | 패턴/카메라/파이프라인 옵션을 담은 `.yaml`/`.yml`/`.json` 파일 (아래 예시 참고). 같은 옵션을 커맨드라인에 또 주면 커맨드라인이 우선 |
| `--images` | 이미지 파일/디렉토리/glob 패턴 (여러 개 가능) |
| `--pattern {charuco,chessboard,circle_grid,apriltag_grid}` | 패턴 타입 (기본 charuco). `circle-grid`/`circles`, `aprilgrid`/`april_grid` alias도 허용. AprilGrid는 `DICT_APRILTAG_*` dictionary와 Kalibr-compatible row-major marker ID 배치를 사용 |
| `--circle-grid-type {symmetric,asymmetric}` | Circle Grid일 때 격자 종류 (기본 symmetric) |
| `--aprilgrid-variant {opencv_apriltag3,kalibr}` | AprilGrid detector variant (기본 opencv_apriltag3). 오늘 시점 둘은 동일한 검출 경로를 쓰고 로그 문구만 다름 (4번 섹션 참고) |
| `--calibration-method {standard,object_releasing}` | Standard 4모델(기본) 또는 Object-Releasing(Advanced, Checkerboard/Circle Grid만) |
| `--bag`, `--topic`, `--bag-interval` | rosbag에서 이미지 추출 (`--images` 대신) |
| `--list-topics BAG_PATH` | bag의 이미지 토픽 목록만 보고 종료 |
| `--model {pinhole,brown_conrady,extended_pinhole,fisheye,rational,...}` | 자동 추천 대신 강제로 이 모델 선택. `ideal_pinhole`/`brown-conrady`/`extended`/`rational` alias도 허용 - `rational`은 `extended_pinhole`과 완전히 동일한 모델을 가리키는 정식 이름이다 (Rational은 항상 8계수, runtime toggle 없음) |
| `--models MODEL [MODEL ...]` | 계산/검증할 모델 목록 (여러 개). 예: `--models pinhole brown_conrady rational` |
| `--outlier` | 이상치 탐지 + 재계산까지 수행 |
| `--diagnostic` | 종합 진단 preset. 기본 5-fold CV + 100회 bootstrap + report/json/csv export |
| `--cross-validation K` | K-Fold Cross Validation 수행 (`--kfold K`와 동일) |
| `--bootstrap N` | 최종 선택 모델의 bootstrap 기반 Parameter CI를 N회 재표본으로 계산 |
| `--repeatability N` | 최종 선택 모델을 N회 반복 재계산해 파라미터 안정성 측정 |
| `--jobs N` | 이미지 검출과 heavy analysis(K-fold/repeatability/bootstrap)를 N개 worker로 병렬화 (기본 1=순차, 0=자동) |
| `--export {opencv,ros,report,json,csv,kalibr}` | 내보낼 형식 선택 (기본: opencv/ros/report - json/csv/kalibr는 명시해야 포함됨). `kalibr`는 AprilGrid target YAML을 생성 |
| `--kalibr-camera-model MODEL` | `--export kalibr`에서 command hint를 만들 때 쓸 Kalibr camera model. 기본 `pinhole-radtan` |
| `--json-summary PATH` | 기계가 읽는 JSON 요약 저장 (CI 스크립팅용) |
| `--quiet` | 진행상황 출력 최소화 |
| `-v`/`--verbose`, `--log-file PATH` | 진단 로그 상세도/파일 저장 (버그 재현 시 유용) |

**`--config` 사용 예시** - 같은 카메라로 반복 실행하는 운영 환경(예: 여러 로봇에
같은 카메라 모듈)에서 패턴/카메라 설정을 매번 타이핑하는 대신 파일로 고정해두면
실수를 줄일 수 있다:

```yaml
# camera.yaml
squares_x: 7
squares_y: 5
square_size: 0.04
marker_size: 0.03
dictionary: DICT_5X5_100
sensor_name: front_camera
output_dir: ./out
export: [opencv, ros, report, json]
```

```bash
# 패턴/카메라 설정은 파일에서, 이미지 경로만 그때그때 커맨드라인으로
python -m app.cli --config camera.yaml --images ./photos_2026_08_14

# 특정 값만 그날 잠깐 다르게 쓰고 싶으면 커맨드라인이 파일 값을 덮어쓴다
python -m app.cli --config camera.yaml --images ./photos --square-size 0.05
```

키 이름은 각 옵션의 `--long-name`에서 하이픈을 언더스코어로 바꾼 것과 동일하다
(`--square-size` → `square_size`). 알 수 없는 키가 있으면 즉시 에러로 알려준다
(오타 방지). JSON도 동일한 키로 그대로 쓸 수 있다.

전체 옵션은 `python -m app.cli --help` 참고. rosbag 예시:

```bash
python -m app.cli --list-topics drive.bag   # 토픽 확인
python -m app.cli --bag drive.bag --topic /camera/image_raw --bag-interval 0.5 \
  --squares-x 7 --squares-y 5 --square-size 0.04 --marker-size 0.03 \
  --output-dir ./out
```

### 8.2 Python API 직접 사용

CLI보다 세밀하게 제어하고 싶다면 `calibration/`, `export/` 모듈을 직접 호출한다:

```python
from calibration.types import PatternConfig, PatternType, CameraConfig
from calibration.detector import detect_dataset
from calibration.compare import run_all_models
from calibration.validation import validate_all_models
from calibration.recommender import compute_model_scores, build_recommendation_message, compute_final_result
from export.report import export_html_report

pattern = PatternConfig(type=PatternType.CHARUCO, squares_x=7, squares_y=5,
                         square_size=0.04, marker_size=0.03, dictionary="DICT_5X5_100")
camera_config = CameraConfig(width=1920, height=1080)

dataset = detect_dataset(["img001.jpg", "img002.jpg", ...], pattern)
calibration_results = {r.model_name: r for r in run_all_models(dataset, camera_config)}
validation_results = validate_all_models(dataset, camera_config, pattern)
scores = compute_model_scores(calibration_results, validation_results)
print(build_recommendation_message(scores, calibration_results, validation_results))

# 종합 HTML 리포트 (Train/Test/Edge RMS, Straightness, Edge Error Map, 등급까지 포함)
recommended = next(s.model_name for s in scores if s.is_recommended)
final_result = compute_final_result(recommended, calibration_results, validation_results, scores=scores)
export_html_report("my_camera", camera_config, pattern, dataset,
                    calibration_results, validation_results, final_result, "report.html")
```

JSON/CSV로도 export할 수 있다:

```python
from export.json_export import export_json
from export.csv_export import export_csv

# 구조화된 전체 결과(카메라 행렬, 오차 지표, 등급) - 다른 스크립트/도구가 읽기 좋음
export_json(camera_config, pattern, dataset, calibration_results, validation_results,
            recommended, "calibration.json", final_result=final_result, model_scores=scores)

# 이미지별 상세 데이터(코너 수, 선명도, 재투영 오차, 품질 점수/등급) - 스프레드시트 분석용
export_csv(dataset, "dataset.csv")
```

rosbag에서 이미지를 뽑고 싶다면:

```python
from calibration.rosbag_reader import list_image_topics, extract_images_from_bag

for t in list_image_topics("drive.bag"):
    print(t.name, t.msg_type, t.count)

paths = extract_images_from_bag("drive.bag", "/camera/image_raw", "extracted/", min_interval_sec=0.5)
dataset = detect_dataset(paths, pattern)  # 이후 흐름은 위 예시와 동일
```

## 9. 테스트

`tests/` 폴더에 pytest 스위트가 있다 (수백 개 단위, 빠른 것만 돌리면 수 초 -
전체는 몇 분 정도, OpenCV 4.x/5.x 둘 다에서 통과 확인됨. 정확한 개수는 계속
늘어나므로 `pytest --collect-only -q`로 직접 확인하는 걸 권장). 코드를 고치다가
뭔가 깨지면 이게 잡아준다 - 예전엔 검증할 때마다 스크립트를 즉석으로 짰다가
끝나면 지웠는데, 그러면 다음에 같은 곳이 또 깨져도 아무도 모른다.

```bash
pip install -r requirements-dev.txt
pytest              # 전체 실행 (~1분 40초)
pytest -m "not slow"   # 느린 통합 테스트 빼고 빠르게만 (~7초, 스모크 테스트 포함)
pytest tests/test_straightness.py -v   # 특정 파일만
```

같은 명령을 매번 타이핑하지 않도록 `./run_tests.sh`(전체 `pytest -q`)와
`./run_tests_fast.sh`(`pytest -q -m "not slow"`) 래퍼 스크립트도 있다.
과거 버전의 `run_tests.sh`는 각 테스트 파일을 `python3 file.py`로 개별
실행했는데, 이 저장소의 테스트 파일들은 전부 pytest 함수/fixture 기반이라
`if __name__ == "__main__":` 블록이 없다 - 즉 그 방식은 매번 "0 passed"를
조용히 성공으로 보고할 뿐 테스트를 단 하나도 실행하지 않고 있었다(Phase
A-2에서 발견/수정). 지금은 두 스크립트 모두 실제 pytest 실행기를 그대로
호출한다.

**두 단계로 나뉜다:**
- **빠른 티어** (마커 없음, `not slow`로 걸러짐): 단위 테스트 대부분 +
  `test_smoke_pipeline.py` - 3D->2D 직접 사영으로 이미지 렌더링/검출 없이,
  Ideal Pinhole+Rational 2모델만(Fisheye 생략, Standard 4모델 중 가장 느리고
  발산 위험도 큼) 작은 데이터셋(8~10장)으로 도는 가벼운 파이프라인 스모크
  테스트. Object-Releasing 전용 검증(`test_object_releasing_validation.py`,
  `test_project_migration.py`, `test_kalibr_aprilgrid_fixture.py`)도 합성
  데이터/생성된 fixture만 써서 이 빠른 티어에 포함된다. "핵심 배선이 안
  끊어졌는지"를 몇 초 안에 확인하는 용도.
- **느린 티어** (`@pytest.mark.slow`): 실제 ChArUco 이미지 렌더링+검출,
  Standard 4모델 전부, Hold-out validation, export까지 포함하는 진짜 통합
  테스트 (`test_pipeline_integration.py` 등). 정확성 자체를 검증하는 최종 보루.

**fixture 캐싱**: 무거운 계산(Standard 4모델+검증+이상치 제거 전체 파이프라인)을
쓰는 파일들(`test_project_io.py`, `test_ui_project_io.py`)은 그 계산을
`module` 스코프 fixture로 한 번만 돌리고 파일 안의 여러 테스트가 공유한다 -
전에는 테스트마다 매번 새로 계산해서 낭비가 컸다(전체 스위트가 이 변경
하나로 127초 -> 102초). 세션 전체가 공유하는 `synthetic_dataset`
fixture(`conftest.py`)를 건드리는 fixture는 `copy.deepcopy()`로 복사본을
써서 다른 테스트 파일에 상태가 새지 않게 한다.

`tests/conftest.py`가 왜곡이 실제로 적용된 합성 ChArUco 이미지 데이터셋을
세션당 한 번만 만들어서 여러 테스트가 공유한다. `test_pipeline_integration.py`가
가장 중요한 파일 - Detection부터 Export(OpenCV/ROS/HTML)까지 전체 파이프라인을
실제로 이어붙여서 돈다.

`rosbags`가 설치 안 돼 있으면 `test_rosbag_reader.py`는 자동으로 스킵된다
(선택적 의존성이라 앱도, 테스트도 없어도 동작해야 하므로).

`.github/workflows/tests.yml`로 GitHub Actions CI가 붙어있어 push/PR마다
Python 3.10/3.11/3.12에서 자동으로 돌아간다 (`test` 잡). 같은 워크플로우 안에
OpenCV 4.x/5.x 호환성 매트릭스(`opencv-compat`)와 빠른 티어만 도는
`smoke` 잡도 함께 있다.

**커버리지**: Python 3.11 잡에서 `pytest-cov`로 커버리지를 측정한다
(`calibration/`, `app/`, `export/`, `ui/` 대상). 매 실행마다:
- Actions 실행 결과 페이지 상단 "Summary"에 모듈별 커버리지 표가 바로 보인다
  (파일을 따로 열 필요 없음)
- HTML 상세 리포트(`htmlcov/`)와 원본 `coverage.xml`을 Artifacts로 다운로드할 수
  있다 (14일 보관)

README 배지는 지금은 통과/실패 여부만 보여준다 - 커버리지 %까지 배지로
고정하려면 Codecov 같은 외부 서비스 연동이 필요한데, 그건 저장소 소유자가
직접 Codecov 계정을 만들어 토큰을 등록해야 해서 여기서는 붙이지 않았다
(원하면 `.github/workflows/tests.yml`의 coverage 단계 뒤에 `codecov/codecov-action`
스텝만 추가하면 된다).

## 10. Model Score 가중치 튜닝

`scripts/tune_model_score_weights.py`로 실제 카메라 데이터셋 없이도 가중치를
"정답을 아는" 합성 시나리오로 검증해봤다. 진짜 Pinhole/Rational(Extended
Pinhole)/Fisheye 카메라를 합성으로 만들어(픽셀 노이즈, 저데이터 경계 케이스
포함, Brown-Conrady 시나리오는 아직 없음) "정답 모델을 골랐는가"를 채점하고,
Dirichlet 무작위 탐색으로 더 나은
가중치를 찾아봤다.

```bash
# 시드 하나씩 캐시를 만들어 pickle로 저장 (계산이 오래 걸려 나눠서 실행 가능)
python scripts/tune_model_score_weights.py --build-cache 1 --out /tmp/c1.pkl
python scripts/tune_model_score_weights.py --build-cache 2 --out /tmp/c2.pkl

# 캐시로 가중치 탐색 + held-out 검증
python scripts/tune_model_score_weights.py --search /tmp/c1.pkl /tmp/c2.pkl --holdout /tmp/c11.pkl
```

**결론: 기본 가중치(`ModelScoreWeights()`)는 바꾸지 않았다.** 탐색용
데이터에서는 튜닝된 가중치가 이겨 보였지만(정답률 66.7%→75.0%), **탐색에
안 쓴 held-out 시드로 재확인하니 기본 가중치와 완전히 동률(50.0%=50.0%)이었다**
- 2회 독립 반복 모두 같은 패턴. 즉 "개선"처럼 보였던 건 8~12개짜리 작은
탐색 세트에 대한 과적합이었다. 자세한 수치와 방법론은
[`scripts/TUNING_RESULTS.md`](scripts/TUNING_RESULTS.md) 참고.

가중치와 무관하게 발견한 진짜 한계도 있다: 화각이 넓지 않은 데이터에서는
Rational과 Fisheye가 통계적으로 구분하기 어려워질 수 있다 (근본적인
모델 식별성 문제, 가중치 튜닝으로 해결 안 됨). `tests/test_recommender_accuracy.py`에
이 한계를 `xfail`로 정직하게 기록해뒀다.

## 11. 개발 진행 상황

설계 문서 기준 V1(필수 기능)은 완료됐고, V2(완성도) 항목도 대부분 구현됐습니다.

| V2 항목 | 상태 |
|---|---|
| Dataset Diversity Score | ✅ |
| Undistortion Preview | ✅ |
| Automatic Model Recommendation | ✅ |
| Parameter Uncertainty (Pinhole/Brown-Conrady/Rational/Fisheye) | ✅ Pinhole/Brown-Conrady/Rational은 `calibrateCameraExtended()`의 stdDeviations를 그대로 사용. Fisheye는 OpenCV가 covariance를 안 줘서 bootstrap resampling으로 별도 추정 (`calibration/models/fisheye.py`의 `_bootstrap_fisheye_uncertainty()`, `estimate_uncertainty=True`일 때만 - 1차 실행 결과에서 기본 활성화) |
| **Frame Quality Score** | ✅ `calibration/frame_quality.py` |
| **Edge Error Map (Radial Error Profile)** | ✅ 계산은 유지되어 Model Comparison 표의 Radial Edge 열/HTML 리포트에 반영됨 (`calibration/radial_profile.py`). 전용 그래프 탭은 UI 단순화로 제거됨 |
| **Line Straightness Residual** | ✅ `calibration/straightness.py` |
| **HTML Report** | ✅ `export/report.py` |
| **ROS 연동 확장 (rosbag 이미지 직접 불러오기)** | ✅ `calibration/rosbag_reader.py` (ROS1/ROS2 둘 다 지원, 순수 Python, ROS 설치 불필요) |
| **ROS 연동 확장 (실시간 토픽 구독)** | ✅ `calibration/ros_live.py` + `ui/live_capture_dialog.py` (ROS1/ROS2 자동 감지, ⚠️ 실제 ROS 환경에서 최종 검증 필요 — 아래 6번 참고) |
| **CLI 진입점 (헤드리스 실행)** | ✅ `app/cli.py` (CI/배치 처리용, JSON 요약, 종료 코드 설계) |
| **프로젝트 저장/불러오기** | ✅ `calibration/project_io.py` (`.ccproj`, JSON, 원본 이미지 없이도 이어서 작업 가능) |
| **Straightness Residual 시각화** | ✅ `ui/undistort_straightness_view.py` (② Preview 탭, Undistort Preview와 한 화면으로 합쳐져 행/열 라인을 보정 후 이미지 위에 초록~빨강으로 오버레이, 아래 텍스트는 보정 전/후 수치 한 줄만 표시) |
| **JSON/CSV export** | ✅ `export/json_export.py`, `export/csv_export.py` (구조화된 전체 결과 / 이미지별 상세 데이터, UI·CLI 둘 다 지원) |
| **Chessboard(일반 체스보드) 패턴 지원** | ✅ `calibration/detector.py` (UI/CLI 둘 다, ChArUco와 동일한 파이프라인 재사용 - straightness.py 등 기존 모듈 변경 없음. ⚠️ 대칭 패턴이라 방향 모호성 있음, README 3번 주의사항 참고) |

### 실사용 중 발견/수정된 버그 (2026-08)

실제 사용자 환경(ROS2 humble, OpenCV 5.0.0)에서 나온 리포트를 그 환경을
직접 재현해서 고쳤습니다:

- **OpenCV 5.0.0에서 Fisheye 캘리브레이션 실패**: `cv2.fisheye.CALIB_*` 플래그가
  5.0부터 최상위 `cv2.CALIB_*`로 옮겨간 것과, `cv2.fisheye.calibrate()`가
  `(N,1,3)`이 아니라 `(1,N,3)` shape을 요구하게 된 것(4.x에서는 관대했음) 두
  가지가 원인이었습니다. `opencv-contrib-python==5.0.0.93`을 직접 설치해
  재현하고 수정 후 4.13.0/5.0.0 둘 다에서 전체 테스트 통과를 확인했습니다.
- **rosbag 읽기 실패("Bag contains no type definitions")**: 위 6번 섹션 참고.
- **실시간 구독이 "프레임 수신 대기 중"에서 안 멈춤**: 지원 안 하는 인코딩으로
  프레임이 오면 조용히 버려지던 것을 고쳐 화면에 표시되게 함. YUV422 계열
  인코딩(`yuv422`/`yuyv`/`uyvy` 등, 흔한 카메라 드라이버가 씀) 지원 추가.
- **Dataset 탭 UI**: 검출 실패 이유 표시, "상태" 컬럼 폭/줄바꿈 개선,
  Coverage 탭 막대그래프 정렬, Square/Marker size mm 입력, Complexity 행 제거,
  모델 선택 콤보 위치 및 실패 모델 상태 표시 개선.

## 12. Windshield Refraction Calibration (`windshield` 브랜치)

Camera Intrinsic(위 1~11번 섹션) 위에 얹는 **별도 계층**이다. 실제 차량은
카메라 앞에 windshield(자동차 유리)가 있고, 그 유리가 굴절을 일으켜 순수
K/D 모델만으로는 설명 안 되는 잔차가 남는다 - 이 계층은 그 잔차를
모델링한다.

### 12.1 아키텍처 (절대 원칙)

```
Camera Intrinsic Calibration (1~11번 섹션)
        │  (fx, fy, cx, cy, distortion, base camera model)
        ▼
   Fixed Base K, D  ──────────────────────────────────────
        │                                                  │  (재최적화 절대 금지)
        ▼                                                  │
Windshield Geometry Correction                             │
  (Baseline/Spherical/Residual Grid/Residual RBF/           │
   Spline/Neural Residual 중 하나)                          │
        │                                                  │
        ▼                                                  │
  Corrected Projection (project_point / unproject_pixel) ◄──┘

── 완전히 독립된 두 번째 축(Geometry 점수에 절대 반영되지 않음) ──
Photometric Evaluation
  Reflection (Reference-mode 실측 vs No-Reference heuristic)
  Ghost / Double Image (Point-source 실측 vs General heuristic)
  → Glare / Saturation은 Reflection의 하위 지표
```

**두 가지 절대 원칙**(코드 전체에 강제됨, 테스트로 회귀 방지):

1. **Camera Intrinsic vs Windshield 분리**: Windshield 관련 어떤 함수도
   `fx/fy/cx/cy`/distortion/base camera model을 재추정하지 않는다.
   `WindshieldConfig.base_camera_matrix`/`base_distortion`은 항상 이미
   확정된 `CalibrationResult`의 스냅샷(`.copy()`)이다.
2. **Geometry vs Photometric 분리**: Reflection/Ghost 점수(강도, 검출률,
   likelihood)는 Windshield Geometry 모델의 "승자" 선택에 절대 영향을 주지
   않는다. Geometry는 Hold-out RMS/P95/P99/Regional/Edge/Stability로만
   비교하고, Photometric은 별도 축으로만 보고한다.

### 12.2 Windshield Geometry 모델

| 모델 | 설명 |
|---|---|
| **Baseline** | 보정 없음(항등) - Base K,D 그대로 투영/역투영. Windshield 효과가 실제로 얼마나 되는지 측정하는 기준선. |
| **Spherical** | Windshield를 단일 구(sphere)로 근사하고 Snell 굴절 법칙(공기→유리→공기, 굴절률/유리 두께 고정 파라미터)을 광선 단위로 실제 계산한다. |
| **Residual Ray (Grid)** | Base Ray(Baseline/Spherical 없이 순수 카메라 광선)에 이미지 위 성긴 grid에 저장된 3D 방향 보정을 bilinear interpolation으로 더한다 - closed-form 물리 모델이 아니라 관측 데이터 기반 보정. |
| **Residual Ray (RBF)** | 위와 같은 아이디어지만 grid 대신 `scipy.interpolate.RBFInterpolator`(thin plate spline)로 성긴 center 집합을 보간한다 - 불규칙한 코너 분포에서 Grid보다 유연할 수 있다. |
| **Spline** | Base Sphere(고정) 위에 bicubic B-spline으로 국소 표면 변형(surface deformation)을 추가하고, 그 변형된 표면에서 실제 Snell 굴절을 계산한다 - Spherical보다 표현력이 높지만 코너마다 광선-표면 교차를 두 번(안쪽+바깥쪽 표면) 풀어야 해서 계산이 훨씬 무겁다. |
| **Residual Ray (Neural)** | Grid/RBF와 같은 계열이지만 보정을 작은 MLP(PyTorch)가 학습한다 - CPU inference만으로 동작(GPU 불필요). 선택적 의존성(12.8번 참고). |

Grid/RBF/Neural은 전부 `WindshieldModelType.RESIDUAL_RAY` 하나의 enum
아래 variant(`residual_ray_hint["method"]`로 구분)다 - UI의 Comparison
표에는 "Residual Grid"/"Residual RBF"로 구분해서 보여준다.

모든 모델은 공통 런타임 API를 구현한다(`calibration/windshield/base.py`):
`project_point(x, y, z) -> (u, v)`, `unproject_pixel(u, v) -> (dx, dy, dz)`.
어떤 계산된 결과(`WindshieldCalibrationResult`)로도 항상
`build_projector(result)` 하나로 실행 가능한 모델 인스턴스를 얻는다
(`calibration/windshield/projection.py`) - 모델 종류를 몰라도 되는 단일
진입점.

### 12.3 Photometric Evaluation - Reflection

Windshield에 비치는 반사(하늘/차내 조명/대시보드 등)를 평가한다. **두
모드는 신뢰 수준이 완전히 다르다**:

- **Reference Mode** (실측): 같은 장면을 반사가 없는(또는 적은) 기준
  이미지와 비교해서 실제 반사 강도(gain/bias 보정 후 차이)를 측정한다.
  Mean/Median/P95/P99/Max/Coverage, 영역별(Center/Top/Bottom/Left/Right/
  Corners) 지표, 자동차 특화 하단 ROI 지표까지 낸다.
- **No-Reference Mode** (heuristic): 기준 이미지가 없을 때 이미지 자체의
  통계만으로 반사 가능성을 추정한다. 결과는 **`reflection_likelihood`에만
  채워지고 `is_likelihood=True`가 항상 함께 붙는다 - 절대 Ground Truth가
  아니다.**

Reflection과는 별개로 **Glare**(과도한 밝기+낮은 대비 영역)와
**Saturation**(픽셀값 clipping 비율)도 독립된 지표로 낸다 - 셋을 하나로
합치지 않는다.

**Reflection Alignment**(Reference mode 전용, Phase B-1/B-2 안정화):
Reference 이미지를 Normal 이미지에 맞추는 정렬(translation 또는 affine)이
`cv2.findTransformECC`(`cv2.MOTION_TRANSLATION`/`cv2.MOTION_AFFINE`) 기반으로
이뤄지고, `method` 값이 지원 목록(`"translation"`,
`"affine"`) 밖이면 조용히 fallback하지 않고 명시적으로 실패한다. 정렬로
생긴 인공적인 border(예: `cv2.BORDER_REFLECT`로 채워진 영역)는 **Valid
Warp Mask**로 계산돼 모든 Reflection 지표 계산에서 제외된다 - border
아티팩트가 반사 강도로 오인되지 않는다. Affine을 쓰면 translation/
rotation/scale/shear 진단값이 함께 기록되고, 비정상적으로 큰 변환은
`warning` 상태로 표시된다(정렬 실패 가능성).

### 12.4 Photometric Evaluation - Ghost / Double Image

Windshield의 이중 표면(안쪽/바깥쪽) 반사가 만드는 유령상(ghost, 원본보다
어둡고 살짝 어긋난 복제 이미지)을 평가한다.

- **Point-source Ghost Evaluation**: 밝은 광원(LED, 헤드라이트 등)의 Main
  피크와 Ghost 피크를 짝지어 displacement(dx, dy)와 강도 비율을 직접
  측정한다. 촘촘한 LED array에서 이웃 Main끼리 잘못 짝지어지는 문제를
  막기 위해 **2-pass spatial pairing**을 쓴다: PASS 1이 전체 이미지의
  dominant displacement vector로 대략적인 pairing을 하고, PASS 2가 그
  결과로 만든 성긴 spatial field에서 위치별 local vector를 다시 추정해
  재-pairing한다(windshield 곡률 등으로 위치마다 실제 ghost displacement가
  달라지는 경우 대응) - PASS 2가 실패하거나 근거가 부족하면 항상 PASS 1로
  안전하게 fallback한다.
- **General (No-Reference) Likelihood**: Point-source 타겟 없이 일반
  도로 영상에서 이미지의 row/column profile을 스캔해 double-edge 패턴
  비율을 heuristic하게 추정한다. **`ghost_likelihood`/`is_likelihood=True`
  로만 노출되고 절대 Ground Truth가 아니다** - `ghost_likelihood`와 실측
  강도 지표(`mean_strength`)를 같은 필드에 절대 섞지 않는다.
- **GhostField**: Dataset 전체의 point-source detection을 robust
  median/MAD로 집계해 만드는 성긴 2D displacement/strength field. 유효
  detection이 0개면 "ghost 없음"과 "ghost=0으로 측정됨"을 구분하기 위해
  명시적으로 fitting을 거부한다(조용히 0으로 채우지 않음).
- **Ghost Suppression**: `I = T + α·W_delta(T)`의 결정론적 역변환으로
  ghost를 제거한다(`suppress_ghost(image, ghost_field)`) - PyTorch 불필요.
  Strength 감소만으로는 성공이 아니므로 edge retention/clean-region
  변화까지 함께 본다(over-suppression score).

### 12.5 Validation Metrics 용어집

| 용어 | 의미 |
|---|---|
| Train RMS | 학습에 쓰인 프레임에서의 재투영 오차 RMS. |
| Hold-out(Test) RMS | 학습에 전혀 쓰이지 않은 프레임에서, intrinsic/geometry 파라미터를 고정한 채 pose만 재추정해 계산한 RMS - 진짜 generalization 지표. |
| Median / P90 / P95 / P99 / Max | 코너 포인트 단위 오차 분포의 요약 통계. RMS 하나로는 안 보이는 꼬리(worst-case) 거동을 드러낸다. |
| Regional / Edge Error | 이미지의 중앙/가장자리/코너 영역별 오차 - 가장자리에서만 오차가 크면 왜곡 보정이 그쪽에서 덜 됐다는 뜻. |
| Radial Profile | 이미지 중심으로부터의 거리(반지름)별 오차 분포. |
| Stability (Repeated Hold-out) | 서로 다른 train/test 분할(seed)로 여러 번 반복한 hold-out 결과의 흩어짐(std) - 낮을수록 특정 분할에 우연히 의존하지 않는 안정적인 모델. |
| **Hold-out Evidence Gate** (Phase B-6, 신규) | Hold-out RMS가 낮다는 사실 하나만으로 "검증됨"이라 말할 수 없다 - test 프레임 수/코너 수/공간 coverage/자세(pose) 다양성이 충분한지 별도로 판정해 `sufficient`/`insufficient_evidence`/`not_evaluated`로 명시한다(`calibration/holdout_evidence.py`). RMS 숫자 자체를 바꾸거나 무효화하지 않는다 - "낮은 RMS + 근거 부족"과 "낮은 RMS + 근거 충분"을 구분해서 보고할 뿐이다. |

### 12.6 Runtime 성능 - Exact vs Batch(Exact) vs LUT(근사)

`build_projector(result)`가 만드는 point-by-point 모델은 Calibration/
Evaluation에는 충분히 빠르지만, Camera-LiDAR 프로젝션처럼 프레임당
수만~수백만 포인트를 처리해야 하는 런타임에는 느릴 수 있다(특히
Spherical/Spline/Neural은 포인트마다 root-solve/surface-intersection이
필요). `calibration/windshield/runtime_projector.py`가 이를 위한 **별도
API**(`build_runtime_projector()` - 기존 `build_projector()`는 그대로
유지)를 추가한다:

- **Batch(Exact)**: `project_points_exact_batch()`/
  `unproject_pixels_exact_batch()` - 계산 자체는 point-by-point와 100%
  동일(근사 아님), 호출자가 (N,3)/(N,2) 배열 하나로 넘길 수 있는 껍데기다.
  Baseline은 실제로 벡터화된 `cv2.projectPoints` 경로를 타 진짜 빨라지고,
  나머지는 반복 호출 오버헤드만 줄인다.
- **LUT(근사)**: `RuntimeWindshieldProjector` - 이미지 전체를 성긴 grid로
  미리 샘플링해 둔 뒤 bilinear 보간(`unproject_pixels`)/k-최근접
  역거리가중평균(`project_points`)으로 조회한다. **정확도 tradeoff가
  있다**: `unproject_pixels`는 격자 해상도만큼의 보간 오차만 생기지만,
  `project_points`는 "포인트의 방향만" 쓰는 근사라 windshield에 아주
  가까운 포인트(수십 cm 이내)일수록 오차가 커질 수 있다(parallax) - LiDAR
  포인트처럼 windshield-camera 간격보다 훨씬 먼 포인트에서는 근사 오차가
  작을 것으로 기대되지만, **실측 없이 이 가정을 신뢰하지 않는다.**
  `validate_runtime_projector_vs_exact()`가 median/P95/P99/Max 픽셀 오차
  (project)와 각도 오차(unproject)를 실제로 측정해 알려준다.

성능 실측은 `scripts/benchmark_windshield_runtime.py`로 직접 확인한다
(1/100/1,000/10,000 포인트 스케일 x Scalar/Batch(Exact)/LUT 세 티어,
CPU-first, Jetson에서도 실행 가능):

```bash
python scripts/benchmark_windshield_runtime.py
```

**이 스크립트가 출력하는 숫자는 그 스크립트를 실행한 기기의 실측치일
뿐이다** - Jetson 성능을 알고 싶으면 실제 Jetson에서 이 스크립트를 직접
돌려야 한다. 이 저장소의 CI는 Jetson 하드웨어를 갖고 있지 않으므로 Jetson
숫자를 대신 만들어내지 않는다.

### 12.7 Real Vehicle Day/Night/Session Validation Framework

`calibration/vehicle_validation.py`가 실제 차량에서 촬영한 여러
세션(차량 x 시간대 x windshield 상태)을 비교하기 위한 메타데이터/집계/
비교 프레임워크를 제공한다 - Day/Night 비교, Geometry(Hold-out) 지표와
Photometric(Reflection/Ghost) 지표를 절대 하나로 섞지 않는 두 개의 독립된
요약(`SessionGeometrySummary`/`SessionPhotometricSummary`)으로 관리한다.

> ⚠️ **NOT YET VALIDATED ON REAL VEHICLE.** 이 저장소에는 실제 차량 캡처
> 데이터셋이 전혀 없다. 위 프레임워크는 그런 데이터가 생기면 곧바로 쓸 수
> 있는 틀만 제공할 뿐, 어떤 실차 검증 결과도 만들어내지 않는다. 모든
> 세션 메타데이터는 `data_provenance`("real_vehicle"/"synthetic"/
> "unknown")를 명시해야 하고, `real_vehicle_validation_disclaimer()`는
> real_vehicle 세션이 하나도 없으면 이 사실을 항상 그대로 보고한다.

### 12.8 Neural Residual (선택적 의존성)

Residual Ray의 Neural variant만 PyTorch가 필요하다 - 다른 모든 기능
(Baseline/Spherical/Residual Grid/Residual RBF/Spline, Reflection/Ghost
평가/Suppression 포함)은 PyTorch 없이 완전히 동작한다.

```bash
# Desktop
pip install -e ".[neural]"        # torch>=2.0 (일반 PyPI wheel)
```

**Jetson에서는 일반 PyPI `torch` wheel을 설치하지 않는다** - JetPack
버전에 맞는 NVIDIA 제공 PyTorch wheel을 따로 설치해야 한다(architecture가
aarch64 + CUDA라 PyPI의 표준 wheel과 호환되지 않음). `requirements-jetson*.txt`
에는 의도적으로 `torch`를 넣지 않았다.

Reflection Suppression도 별도의 학습된 PyTorch 모델(`calibration/windshield/
reflection_suppression/`)을 쓴다 - Ghost Suppression(12.4번, 결정론적,
PyTorch 불필요)과는 완전히 다른 접근이다.

### 12.9 Windshield 테스트

Camera Intrinsic과 같은 `pytest`로 함께 돌아간다 - Windshield 전용
테스트만 고르려면:

```bash
pytest -q -k windshield
pytest -q tests/test_windshield_ghost.py tests/test_windshield_ghost_suppression.py
pytest -q tests/test_windshield_reflection_evaluation.py
pytest -q tests/test_windshield_runtime_projector.py
```

기능별 GitHub Actions 워크플로우(`.github/workflows/ghost-tests.yml`,
`reflection-tests.yml`, `reflection-suppression-tests.yml`,
`neural-tests.yml`)가 각각 독립적으로 돌고, `.github/workflows/ci.yml`
(Python 3.10/3.11 매트릭스)이 `pytest -q -m "not slow"`로 이 저장소
전체(Windshield 포함)의 핵심 회귀를 한 번 더 빠르게 확인한다 - torch를
요구하지 않으므로 Neural 전용 회귀는 여전히 `neural-tests.yml`이
담당한다.

### 12.10 Current Validation Status

| 대상 | 상태 |
|---|---|
| Synthetic unit/integration tests (CI) | ✅ 검증됨 - 이 저장소의 pytest 스위트가 매 커밋마다 확인 |
| Desktop (Windows/macOS/Linux) 수동 실행 | ⚠️ 부분적 - 이 세션(Windows 샌드박스)에서는 PySide6 자체를 import할 수 없어 UI를 직접 띄워보지 못했다(DLL 로드 실패, 코드/데이터 레이어만 검증) |
| Jetson JetPack 6.2.1 | ⚠️ 부분적 - 설치 스크립트/preflight만 있고 실제 Jetson 기기에서 실행한 적은 없다 |
| Jetson JetPack 5.1.2 | ⚠️ 부분적 - 설치 스크립트/preflight만 있고 실제 Jetson 기기에서 실행한 적은 없다 |
| **실제 차량(Real Vehicle)** | ❌ **NOT YET VALIDATED** - 실차 캡처 데이터셋 자체가 이 저장소에 없다(12.7번 참고) |

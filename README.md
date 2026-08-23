# Camera Calibration Tool

[![Tests](https://github.com/yaaaaangeo/camera_calibrator/actions/workflows/tests.yml/badge.svg)](https://github.com/yaaaaangeo/camera_calibrator/actions/workflows/tests.yml)

Pinhole / Extended Pinhole(Brown-Conrady) / Fisheye(Kannala-Brandt) 세 모델을
ChArUco 패턴으로 동시에 캘리브레이션하고, Hold-out 검증 + Model Score 기반으로
근거 있는 추천을 해주는 도구.

## 1. 요구 사항

- **Python 3.10 이상** (3.11 권장)
- OS: Windows / macOS / Linux 모두 가능 (PySide6가 크로스플랫폼)
- **OpenCV**: 4.7 이상, **5.0.0도 지원** (`opencv-contrib-python==5.0.0.93`으로
  실제 검증됨). 4.x와 5.x 둘 다에서 전체 테스트 스위트가 통과한다 - OpenCV
  5.0에서 `cv2.fisheye.CALIB_*` 플래그 위치가 바뀌고 `cv2.fisheye.calibrate()`의
  요구 shape이 엄격해진 것에 대응하는 코드가 `calibration/models/fisheye.py`에
  있다 (자세한 내용은 10번 섹션).

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
   ChArUco(기본, 권장), Chessboard(일반 체스보드), AprilGrid 중 고를 수 있다.
   Chessboard를 고르면 Marker size/Dictionary 입력칸이 자동으로 숨겨진다
   (체스보드엔 필요 없으므로). ChArUco/AprilGrid는 사각형 개수/한 칸
   크기(mm)/마커 크기(mm)/dictionary가 필요하고, Chessboard는 사각형 개수/
   한 칸 크기(mm)만 있으면 된다. AprilGrid는 Kalibr과 같은 row-major ID
   배치(왼쪽 위 0번부터 오른쪽으로 증가, 다음 줄로 이동)를 가정한다.
   앱 내부에서는 OpenCV `DICT_APRILTAG_*` dictionary로 기본 검출을 수행할 수
   있고, `--export kalibr`로 Kalibr 공식 `aprilgrid` target YAML도 만들 수 있다.

   > ⚠️ **Chessboard를 쓸 때 주의**: 일반 체스보드는 대칭 패턴이라 ChArUco와
   > 달리 (1) 보드 전체가 이미지 안에 다 보여야 검출되고, (2) "어느 쪽이
   > 진짜 첫 번째 코너인지"를 원리적으로 구분할 방법이 없다 - 같은 보드를
   > 정방향으로 찍든 180도 돌려서 찍든 육안으로는 구분도 안 되는데, 촬영
   > 방향이 데이터셋 안에서 뒤섞이면 캘리브레이션이 심하게 틀어질 수 있다.
   > 이건 이 프로젝트가 만든 문제가 아니라 OpenCV 표준 체스보드 캘리브레이션
   > 자체의 잘 알려진 한계다 (설계 문서 2번이 애초에 ChArUco를 우선한 이유이기도
   > 하다). 가능하면 ChArUco를 쓰고, 꼭 체스보드를 써야 한다면 촬영 내내
   > 보드 방향을 일관되게 유지할 것.
2. **[이미지 불러오기]** 로 촬영한 사진 여러 장 선택 (jpg/png/bmp), 또는
   **[rosbag에서 불러오기]** 로 ROS1(.bag)/ROS2(.db3, .mcap) 로그에서 이미지 토픽을
   골라 자동 추출 (rospy/rclpy 설치 불필요, 순수 Python `rosbags` 라이브러리 사용), 또는
   **[실시간 카메라 구독]** 으로 ROS1/ROS2 이미지 토픽을 실시간 구독해서 라이브
   프리뷰를 보며 원하는 자세에서 직접 캡처 (이 기능은 실제 ROS1 또는 ROS2가
   컴퓨터에 설치되어 있어야 함 - rospy/rclpy는 pip로 설치되지 않음)
3. **[캘리브레이션 실행]** 클릭 → 검출 → 3모델 계산 → Hold-out → 추천까지 자동 진행
4. 탭을 넘기며 결과 확인:
   - **① Dataset**: 이미지별 상태/코너 수/재투영 오차/**품질 점수(Frame Quality Score)·등급**
   - **② Detection**: 검출 성공/실패와 실패 이유를 독립 탭에서 확인
   - **③ Coverage**: 4×4 커버리지 맵 + 데이터셋 다양성 점수 + 경고
   - **④ Calibration**: 선택 모델 기준 이상치 제거·재계산
   - **⑤ Validation**: sanity check + Dataset B/C cross-dataset validation
   - **⑥ Error Analysis**: Undistort Preview, Edge Error Map, Straightness Map, 외부 결과 비교
     (OpenCV YAML / ROS CameraInfo YAML / Kalibr camchain YAML / 표준 JSON import)
     + benchmark compatibility 검사(width/height, model, distortion model, 계수 개수,
     NaN/Inf, matrix shape, 파라미터 범위)
     + Reference/Candidate를 둘 다 파일로 로드하는 독립 benchmark 비교
   - **⑦ Stability**: bootstrap/repeatability 기반 파라미터 안정성
   - **⑧ Model Comparison**: 3모델 비교표(Train/Test/P95/Edge/Radial/AIC/BIC/Stability/Observability) + 추천 이유
   - **⑨ Diagnosis**: failure pattern, 원인 분석, 다음 촬영 추천
   - **⑩ Export**: OpenCV/ROS YAML, **HTML 종합 리포트**, **JSON**, **CSV** export
5. 상단 메뉴 **파일 → 프로젝트 저장(Ctrl+S)** 으로 지금까지의 전체 상태(데이터셋,
   3모델 결과, 검증, 추천)를 `.ccproj` 파일로 저장할 수 있다. **파일 → 프로젝트
   불러오기(Ctrl+O)** 로 나중에 이어서 작업 가능 — 원본 이미지 파일이 없어져도
   재계산/이상치 제거/export는 그대로 된다 (자세한 내용은 4번 폴더 구조의
   `project_io.py` 설명 참고).

## 4. 폴더 구조

```
camera_calibrator/
├── app/
│   ├── main.py                # GUI 실행 진입점 (python -m app.main)
│   └── cli.py                 # 헤드리스 CLI 진입점 (python -m app.cli), CI/배치용
├── calibration/              # 순수 계산 로직 (UI 의존성 없음)
│   ├── types.py              # 전체가 공유하는 데이터 구조
│   ├── detector.py           # ChArUco + Chessboard(일반 체스보드) 검출
│   ├── models/                # pinhole / extended_pinhole / fisheye
│   ├── compare.py            # 3모델 동시 실행 + 비교표
│   ├── validation.py         # Hold-out 검증 (+ Line Straightness)
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
│   ├── radial_profile_view.py   # Edge Error Map 그래프 (QPainter 커스텀 위젯)
│   ├── straightness_view.py     # Straightness Map (행/열 라인을 이미지 위에 색으로 오버레이)
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

## 5. ROS 연동

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
(YUYV·UYVY 계열, v4l2_camera/usb_cam 등 흔한 드라이버가 씀), CompressedImage(jpeg/png).
지원 안 하는 인코딩을 만나면 에러 메시지에 실제 인코딩 이름이 나옵니다
(예: "발견된 인코딩: nv12") - 필요하면 이슈로 알려주시면 추가하겠습니다.

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

> **알려진 이슈 (부분 해결)**: 환경이 감지되고 토픽도 맞게 골랐는데 "프레임
> 수신 대기 중..."에서 멈추는 경우가 있었습니다. 원인 중 하나를 찾아 고쳤습니다:
> 카메라가 지원 안 하는 인코딩(예: yuv422)으로 발행하면 프레임이 실제로 도착해도
> 디코딩에 실패해 **조용히 버려지고 있었습니다** - 이제 디코딩 실패 시 화면에
> "⚠ 프레임은 도착했지만 디코딩에 실패했습니다 (encoding='...')" 라고 표시됩니다
> (3초 rate-limit). 그래도 계속 멈춰있다면 다른 원인(토픽에 실제로 발행이
>없거나, ROS 네트워크 설정 문제 등)일 수 있습니다.
>
> ⚠️ `calibration/ros_live.py`의 rospy/rclpy 경로는 실제 ROS 런타임(roscore 또는
> ROS2 데몬)이 있어야만 끝까지 검증할 수 있어, 개발 과정에서 end-to-end로
> 테스트하지 못했습니다. 표준 API 기준으로 작성했지만, 실제 ROS 환경에서
> 한 번 확인해보시는 걸 권장합니다. 문제가 있으면 이슈로 알려주세요.

## 6. 프로젝트 저장/불러오기 (`.ccproj`)

데이터셋이 크거나 캘리브레이션에 시간이 걸릴 때, 앱을 껐다 켜도(또는 CLI를
여러 번 나눠 실행해도) 이어서 작업할 수 있다.

- **저장되는 것**: 카메라/패턴 설정, 데이터셋(이미지별 검출 결과·품질 점수·상태),
  3모델 캘리브레이션 결과, Hold-out Validation 결과, 추천 점수, 이상치 제거 이력,
  최종 결과 - 사실상 화면에 보이는 모든 것.
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

## 7. UI 없이 쓰기 (CLI / Python API)

### 6.1 CLI (`app/cli.py`) — CI/서버/배치 처리용

UI를 안 띄우고 헤드리스로 전체 파이프라인(검출→3모델→검증→추천→export)을
한 번에 돌린다. 종료 코드로 성공/실패를 판단할 수 있어 CI 파이프라인에
바로 끼워 넣기 좋다 (0=성공, 1=입력 문제, 2=전 모델 캘리브레이션 실패).

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
| `--pattern {charuco,chessboard,apriltag_grid}` | 패턴 타입 (기본 charuco). `aprilgrid` alias도 허용. AprilGrid는 `DICT_APRILTAG_*` dictionary와 Kalibr-compatible row-major marker ID 배치를 사용 |
| `--bag`, `--topic`, `--bag-interval` | rosbag에서 이미지 추출 (`--images` 대신) |
| `--list-topics BAG_PATH` | bag의 이미지 토픽 목록만 보고 종료 |
| `--model {pinhole,extended_pinhole,fisheye}` | 자동 추천 대신 강제로 이 모델 선택 |
| `--outlier` | 이상치 탐지 + 재계산까지 수행 |
| `--rational` | Extended Pinhole에 8계수(rational) 모델 사용 |
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

### 6.2 Python API 직접 사용

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

## 8. 테스트

`tests/` 폴더에 pytest 스위트가 있다 (161개, 빠른 것만 돌리면 ~11초 - 전체는
3~4분 정도, OpenCV 4.x/5.x 둘 다에서 통과 확인됨). 코드를 고치다가
뭔가 깨지면 이게 잡아준다 - 예전엔 검증할 때마다 스크립트를 즉석으로 짰다가
끝나면 지웠는데, 그러면 다음에 같은 곳이 또 깨져도 아무도 모른다.

```bash
pip install -r requirements-dev.txt
pytest              # 전체 실행 (~1분 40초)
pytest -m "not slow"   # 느린 통합 테스트 빼고 빠르게만 (~7초, 스모크 테스트 포함)
pytest tests/test_straightness.py -v   # 특정 파일만
```

**두 단계로 나뉜다:**
- **빠른 티어** (마커 없음, `not slow`로 걸러짐): 단위 테스트 대부분 +
  `test_smoke_pipeline.py` - 3D->2D 직접 사영으로 이미지 렌더링/검출 없이,
  Pinhole+Extended 2모델만(Fisheye 생략, 셋 중 가장 느리고 발산 위험도 큼)
  작은 데이터셋(8~10장)으로 도는 가벼운 파이프라인 스모크 테스트. "핵심
  배선이 안 끊어졌는지"를 몇 초 안에 확인하는 용도.
- **느린 티어** (`@pytest.mark.slow`): 실제 ChArUco 이미지 렌더링+검출,
  3모델 전부, Hold-out validation, export까지 포함하는 진짜 통합 테스트
  (`test_pipeline_integration.py` 등). 정확성 자체를 검증하는 최종 보루.

**fixture 캐싱**: 무거운 계산(3모델+검증+이상치 제거 전체 파이프라인)을
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

## 9. Model Score 가중치 튜닝

`scripts/tune_model_score_weights.py`로 실제 카메라 데이터셋 없이도 가중치를
"정답을 아는" 합성 시나리오로 검증해봤다. 진짜 Pinhole/Extended
Pinhole/Fisheye 카메라를 합성으로 만들어(픽셀 노이즈, 저데이터 경계 케이스
포함) "정답 모델을 골랐는가"를 채점하고, Dirichlet 무작위 탐색으로 더 나은
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
Extended Pinhole과 Fisheye가 통계적으로 구분하기 어려워질 수 있다 (근본적인
모델 식별성 문제, 가중치 튜닝으로 해결 안 됨). `tests/test_recommender_accuracy.py`에
이 한계를 `xfail`로 정직하게 기록해뒀다.

## 10. 개발 진행 상황

설계 문서 기준 V1(필수 기능)은 완료됐고, V2(완성도) 항목도 대부분 구현됐습니다.

| V2 항목 | 상태 |
|---|---|
| Dataset Diversity Score | ✅ |
| Undistortion Preview | ✅ |
| Automatic Model Recommendation | ✅ |
| Parameter Uncertainty (Pinhole/Extended/Fisheye) | ✅ Pinhole/Extended는 `calibrateCameraExtended()`의 stdDeviations를 그대로 사용. Fisheye는 OpenCV가 covariance를 안 줘서 bootstrap resampling으로 별도 추정 (`calibration/models/fisheye.py`의 `_bootstrap_fisheye_uncertainty()`, `estimate_uncertainty=True`일 때만 - 1차 실행 결과에서 기본 활성화) |
| **Frame Quality Score** | ✅ `calibration/frame_quality.py` |
| **Edge Error Map (Radial Error Profile)** | ✅ `calibration/radial_profile.py` |
| **Line Straightness Residual** | ✅ `calibration/straightness.py` |
| **HTML Report** | ✅ `export/report.py` |
| **ROS 연동 확장 (rosbag 이미지 직접 불러오기)** | ✅ `calibration/rosbag_reader.py` (ROS1/ROS2 둘 다 지원, 순수 Python, ROS 설치 불필요) |
| **ROS 연동 확장 (실시간 토픽 구독)** | ✅ `calibration/ros_live.py` + `ui/live_capture_dialog.py` (ROS1/ROS2 자동 감지, ⚠️ 실제 ROS 환경에서 최종 검증 필요 — 아래 5번 참고) |
| **CLI 진입점 (헤드리스 실행)** | ✅ `app/cli.py` (CI/배치 처리용, JSON 요약, 종료 코드 설계) |
| **프로젝트 저장/불러오기** | ✅ `calibration/project_io.py` (`.ccproj`, JSON, 원본 이미지 없이도 이어서 작업 가능) |
| **Straightness Residual 시각화** | ✅ `ui/straightness_view.py` (⑥ Straightness Map 탭, 행/열 라인을 이미지 위에 초록~빨강으로 오버레이) |
| **JSON/CSV export** | ✅ `export/json_export.py`, `export/csv_export.py` (구조화된 전체 결과 / 이미지별 상세 데이터, UI·CLI 둘 다 지원) |
| **Chessboard(일반 체스보드) 패턴 지원** | ✅ `calibration/detector.py` (UI/CLI 둘 다, ChArUco와 동일한 파이프라인 재사용 - straightness.py 등 기존 모듈 변경 없음. ⚠️ 대칭 패턴이라 방향 모호성 있음, README 3번 주의사항 참고) |

남은 것: V3(LiDAR-Camera Extrinsic, Multi-sensor Calibration Platform으로 확장).

### 실사용 중 발견/수정된 버그 (2026-08)

실제 사용자 환경(ROS2 humble, OpenCV 5.0.0)에서 나온 리포트를 그 환경을
직접 재현해서 고쳤습니다:

- **OpenCV 5.0.0에서 Fisheye 캘리브레이션 실패**: `cv2.fisheye.CALIB_*` 플래그가
  5.0부터 최상위 `cv2.CALIB_*`로 옮겨간 것과, `cv2.fisheye.calibrate()`가
  `(N,1,3)`이 아니라 `(1,N,3)` shape을 요구하게 된 것(4.x에서는 관대했음) 두
  가지가 원인이었습니다. `opencv-contrib-python==5.0.0.93`을 직접 설치해
  재현하고 수정 후 4.13.0/5.0.0 둘 다에서 전체 테스트 통과를 확인했습니다.
- **rosbag 읽기 실패("Bag contains no type definitions")**: 위 5번 섹션 참고.
- **실시간 구독이 "프레임 수신 대기 중"에서 안 멈춤**: 지원 안 하는 인코딩으로
  프레임이 오면 조용히 버려지던 것을 고쳐 화면에 표시되게 함. YUV422 계열
  인코딩(`yuv422`/`yuyv`/`uyvy` 등, 흔한 카메라 드라이버가 씀) 지원 추가.
- **Dataset 탭 UI**: 검출 실패 이유 표시, "상태" 컬럼 폭/줄바꿈 개선,
  Coverage 탭 막대그래프 정렬, Square/Marker size mm 입력, Complexity 행 제거,
  모델 선택 콤보 위치 및 실패 모델 상태 표시 개선.

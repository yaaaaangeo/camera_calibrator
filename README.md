# Camera Calibration Tool

Pinhole / Extended Pinhole(Brown-Conrady) / Fisheye(Kannala-Brandt) 세 모델을
ChArUco 패턴으로 동시에 캘리브레이션하고, Hold-out 검증 + Model Score 기반으로
근거 있는 추천을 해주는 도구.

## 1. 요구 사항

- **Python 3.10 이상** (3.11 권장)
- OS: Windows / macOS / Linux 모두 가능 (PySide6가 크로스플랫폼)

## 2. 설치

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

## 3. 실행

```bash
python -m app.main
```

창이 뜨면:
1. 상단에서 해상도(Width/Height)와 ChArUco 패턴 정보(사각형 개수, 한 칸 크기(m),
   마커 크기(m), dictionary)를 입력
2. **[이미지 불러오기]** 로 촬영한 사진 여러 장 선택 (jpg/png/bmp), 또는
   **[rosbag에서 불러오기]** 로 ROS1(.bag)/ROS2(.db3, .mcap) 로그에서 이미지 토픽을
   골라 자동 추출 (rospy/rclpy 설치 불필요, 순수 Python `rosbags` 라이브러리 사용), 또는
   **[실시간 카메라 구독]** 으로 ROS1/ROS2 이미지 토픽을 실시간 구독해서 라이브
   프리뷰를 보며 원하는 자세에서 직접 캡처 (이 기능은 실제 ROS1 또는 ROS2가
   컴퓨터에 설치되어 있어야 함 - rospy/rclpy는 pip로 설치되지 않음)
3. **[캘리브레이션 실행]** 클릭 → 검출 → 3모델 계산 → Hold-out → 추천까지 자동 진행
4. 탭을 넘기며 결과 확인:
   - **① Dataset**: 이미지별 검출 상태/코너 수/재투영 오차/**품질 점수(Frame Quality Score)·등급**
   - **② Coverage**: 4×4 커버리지 맵 + 데이터셋 다양성 점수 + 경고
   - **③ Model / Validation / Export**: 3모델 비교표(Train/Test/Edge RMS + **Line Straightness**) +
     추천 + 이상치 제거 + OpenCV/ROS YAML export + **HTML 종합 리포트 export**
   - **④ Undistort Preview**: 원본 vs 보정 이미지 비교
   - **⑤ Edge Error Map**: 이미지 중심으로부터의 거리(반지름)별 재투영 오차 막대그래프
     (Radial Error Profile) — 렌즈 외곽에서 모델이 잘 맞는지 한눈에 확인

## 4. 폴더 구조

```
camera_calibrator/
├── app/main.py              # 실행 진입점
├── calibration/              # 순수 계산 로직 (UI 의존성 없음)
│   ├── types.py              # 전체가 공유하는 데이터 구조
│   ├── detector.py           # ChArUco 검출
│   ├── models/                # pinhole / extended_pinhole / fisheye
│   ├── compare.py            # 3모델 동시 실행 + 비교표
│   ├── validation.py         # Hold-out 검증 (+ Line Straightness)
│   ├── outlier.py            # 이상치 탐지/제거
│   ├── quality.py            # Coverage Map / 데이터셋 다양성
│   ├── frame_quality.py      # 프레임별 품질 점수 (Detection + Geometric)
│   ├── radial_profile.py     # Edge Error Map (반지름별 재투영 오차)
│   ├── straightness.py       # Line Straightness Residual (ChArUco 격자 재활용)
│   ├── rosbag_reader.py      # ROS1(.bag)/ROS2(.db3, .mcap)에서 이미지 추출
│   ├── ros_live.py           # 실시간 ROS1(rospy)/ROS2(rclpy) 토픽 구독 (자동 감지)
│   ├── ros_image_codec.py    # sensor_msgs/Image·CompressedImage 디코딩 (위 둘이 공유)
│   └── recommender.py        # Model Score 기반 추천 + 최종 결과(FinalResult) 조립
├── export/                   # OpenCV YAML / ROS CameraInfo YAML / HTML 리포트
│   ├── opencv.py
│   ├── ros.py
│   └── report.py             # 종합 HTML 리포트 (브라우저 인쇄로 PDF 변환 가능)
├── ui/                       # PySide6 화면 (계산 로직 없음, calibration/*만 호출)
│   ├── radial_profile_view.py   # Edge Error Map 그래프 (QPainter 커스텀 위젯)
│   └── live_capture_dialog.py   # 실시간 구독 + 라이브 프리뷰 + 수동/자동 캡처 다이얼로그
└── requirements.txt
```

`calibration/`은 UI와 완전히 독립적이라, CLI 스크립트나 다른 프론트엔드에서도
그대로 재사용할 수 있습니다.

## 5. ROS 연동

두 단계로 나뉩니다.

### 7.1 rosbag에서 이미지 불러오기 (`[rosbag에서 불러오기]` 버튼, ROS 설치 불필요)

순수 Python 라이브러리 `rosbags`로 ROS1(.bag)/ROS2(.db3, .mcap)를 직접 읽습니다.
ROS가 설치 안 된 컴퓨터에서도 동작합니다.

### 7.2 실시간 토픽 구독 (`[실시간 카메라 구독]` 버튼, **ROS1 또는 ROS2 설치 필요**)

이건 다릅니다 - `rospy`/`rclpy`는 pip로 설치되지 않고, 실제 ROS1(noetic 등) 또는
ROS2(humble 등)가 컴퓨터에 설치되고 환경이 source 되어 있어야만 동작합니다
(예: `source /opt/ros/noetic/setup.bash`). ROS1/ROS2 어느 쪽이 설치돼 있는지는
`calibration/ros_live.py`가 자동으로 감지합니다 (`rospy` 먼저 시도 -> 없으면
`rclpy` 시도 -> 둘 다 없으면 버튼을 눌러도 안내 메시지만 뜨고 앱은 정상 동작).

동작 방식: 토픽을 구독하면서 라이브 프리뷰를 보여주고, `[📸 캡처]` 버튼을 누른
시점의 프레임을 저장합니다 (자동 전체 녹화가 아니라 원하는 자세에서 직접
캡처하는 방식 - 설계 문서 7번, 장수보다 자세 다양성이 중요하기 때문). 편의를
위해 "N초마다 자동 캡처" 옵션도 있습니다.

> ⚠️ `calibration/ros_live.py`의 rospy/rclpy 경로는 실제 ROS 런타임(roscore 또는
> ROS2 데몬)이 있어야만 끝까지 검증할 수 있어, 개발 과정에서 end-to-end로
> 테스트하지 못했습니다. 표준 API 기준으로 작성했지만, 실제 ROS 환경에서
> 한 번 확인해보시는 걸 권장합니다. 문제가 있으면 이슈로 알려주세요.

## 6. CLI로만 써보고 싶다면 (UI 없이)

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

rosbag에서 이미지를 뽑고 싶다면:

```python
from calibration.rosbag_reader import list_image_topics, extract_images_from_bag

for t in list_image_topics("drive.bag"):
    print(t.name, t.msg_type, t.count)

paths = extract_images_from_bag("drive.bag", "/camera/image_raw", "extracted/", min_interval_sec=0.5)
dataset = detect_dataset(paths, pattern)  # 이후 흐름은 위 예시와 동일
```

## 7. 테스트

`tests/` 폴더에 pytest 스위트가 있다 (64개, 빠른 것만 돌리면 수 초 - 전체는 3모델
계산이 포함된 시나리오 때문에 2분 정도). 코드를 고치다가
뭔가 깨지면 이게 잡아준다 - 예전엔 검증할 때마다 스크립트를 즉석으로 짰다가
끝나면 지웠는데, 그러면 다음에 같은 곳이 또 깨져도 아무도 모른다.

```bash
pip install -r requirements-dev.txt
pytest              # 전체 실행
pytest -m "not slow"   # 느린 통합 테스트(파이프라인 전체 e2e) 빼고 빠르게만
pytest tests/test_straightness.py -v   # 특정 파일만
```

`tests/conftest.py`가 왜곡이 실제로 적용된 합성 ChArUco 이미지 데이터셋을
세션당 한 번만 만들어서 여러 테스트가 공유한다. `test_pipeline_integration.py`가
가장 중요한 파일 - Detection부터 Export(OpenCV/ROS/HTML)까지 전체 파이프라인을
실제로 이어붙여서 돈다.

`rosbags`가 설치 안 돼 있으면 `test_rosbag_reader.py`는 자동으로 스킵된다
(선택적 의존성이라 앱도, 테스트도 없어도 동작해야 하므로).

`.github/workflows/tests.yml`로 GitHub Actions CI가 붙어있어 push/PR마다
Python 3.10/3.11/3.12에서 자동으로 돌아간다.

## 8. Model Score 가중치 튜닝

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

## 9. 개발 진행 상황

설계 문서 기준 V1(필수 기능)은 완료됐고, V2(완성도) 항목도 대부분 구현됐습니다.

| V2 항목 | 상태 |
|---|---|
| Dataset Diversity Score | ✅ |
| Undistortion Preview | ✅ |
| Automatic Model Recommendation | ✅ |
| Parameter Uncertainty (Pinhole/Extended) | ✅ (Fisheye는 OpenCV 미지원) |
| **Frame Quality Score** | ✅ `calibration/frame_quality.py` |
| **Edge Error Map (Radial Error Profile)** | ✅ `calibration/radial_profile.py` |
| **Line Straightness Residual** | ✅ `calibration/straightness.py` |
| **HTML Report** | ✅ `export/report.py` |
| **ROS 연동 확장 (rosbag 이미지 직접 불러오기)** | ✅ `calibration/rosbag_reader.py` (ROS1/ROS2 둘 다 지원, 순수 Python, ROS 설치 불필요) |
| **ROS 연동 확장 (실시간 토픽 구독)** | ✅ `calibration/ros_live.py` + `ui/live_capture_dialog.py` (ROS1/ROS2 자동 감지, ⚠️ 실제 ROS 환경에서 최종 검증 필요 — 아래 5번 참고) |

남은 것: V3(LiDAR-Camera Extrinsic, Multi-sensor Calibration Platform으로 확장).

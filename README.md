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
2. **[이미지 불러오기]** 로 촬영한 사진 여러 장 선택 (jpg/png/bmp)
3. **[캘리브레이션 실행]** 클릭 → 검출 → 3모델 계산 → Hold-out → 추천까지 자동 진행
4. 탭을 넘기며 결과 확인:
   - **① Dataset**: 이미지별 검출 상태/코너 수/재투영 오차
   - **② Coverage**: 4×4 커버리지 맵 + 데이터셋 다양성 점수 + 경고
   - **③ Model / Validation / Export**: 3모델 비교표 + 추천 + 이상치 제거 + YAML export
   - **④ Undistort Preview**: 원본 vs 보정 이미지 비교

## 4. 폴더 구조

```
camera_calibrator/
├── app/main.py              # 실행 진입점
├── calibration/              # 순수 계산 로직 (UI 의존성 없음)
│   ├── types.py              # 전체가 공유하는 데이터 구조
│   ├── detector.py           # ChArUco 검출
│   ├── models/                # pinhole / extended_pinhole / fisheye
│   ├── compare.py            # 3모델 동시 실행 + 비교표
│   ├── validation.py         # Hold-out 검증
│   ├── outlier.py            # 이상치 탐지/제거
│   ├── quality.py            # Coverage Map / 데이터셋 다양성
│   └── recommender.py        # Model Score 기반 추천
├── export/                   # OpenCV YAML / ROS CameraInfo YAML
├── ui/                       # PySide6 화면 (계산 로직 없음, calibration/*만 호출)
└── requirements.txt
```

`calibration/`은 UI와 완전히 독립적이라, CLI 스크립트나 다른 프론트엔드에서도
그대로 재사용할 수 있습니다.

## 5. CLI로만 써보고 싶다면 (UI 없이)

```python
from calibration.types import PatternConfig, PatternType, CameraConfig
from calibration.detector import detect_dataset
from calibration.compare import run_all_models
from calibration.validation import validate_all_models
from calibration.recommender import compute_model_scores, build_recommendation_message

pattern = PatternConfig(type=PatternType.CHARUCO, squares_x=7, squares_y=5,
                         square_size=0.04, marker_size=0.03, dictionary="DICT_5X5_100")
camera_config = CameraConfig(width=1920, height=1080)

dataset = detect_dataset(["img001.jpg", "img002.jpg", ...], pattern)
calibration_results = {r.model_name: r for r in run_all_models(dataset, camera_config)}
validation_results = validate_all_models(dataset, camera_config)
scores = compute_model_scores(calibration_results, validation_results)
print(build_recommendation_message(scores, calibration_results, validation_results))
```

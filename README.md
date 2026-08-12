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
│   └── recommender.py        # Model Score 기반 추천 + 최종 결과(FinalResult) 조립
├── export/                   # OpenCV YAML / ROS CameraInfo YAML / HTML 리포트
│   ├── opencv.py
│   ├── ros.py
│   └── report.py             # 종합 HTML 리포트 (브라우저 인쇄로 PDF 변환 가능)
├── ui/                       # PySide6 화면 (계산 로직 없음, calibration/*만 호출)
│   └── radial_profile_view.py   # Edge Error Map 그래프 (QPainter 커스텀 위젯)
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

## 6. 개발 진행 상황

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

남은 것: V3(LiDAR-Camera Extrinsic, Multi-sensor Calibration Platform으로 확장).

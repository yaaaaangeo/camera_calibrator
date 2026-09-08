# Camera Calibrator — Expert / Research Guide

이 문서는 이 프로젝트를 **개발/연구/디버깅** 목적으로 다루는 사람을 위한
기술 문서다. "처음 써보는 사람"을 위한 안내는
[README.md](../README.md)를 먼저 읽어라 - 거기서 설치/실행/결과 해석의
큰 흐름을 잡은 뒤 이 문서로 넘어오는 순서를 권장한다.

이 문서 전체에 걸쳐 각 기능은 다음 넷 중 하나로 명시적으로 표시한다:

- **Implemented** - 소스에 실제로 구현되어 있고 테스트가 있다.
- **Experimental** - 동작은 하지만 실사용 검증이 제한적이다.
- **Partial** - 일부만 구현되었거나 일부 환경에서만 검증됐다.
- **Not Yet Validated** - 코드/틀은 있지만 실제 데이터로 확인된 적이 없다.

"원래 이럴 예정"을 현재 기능처럼 적지 않는다 - 계획은 16번(Future Work)에만
적는다.

## 목차

1. [Architecture](#1-architecture)
2. [Camera Models](#2-camera-models)
3. [Calibration Targets](#3-calibration-targets)
4. [Validation](#4-validation)
5. [Train/Test Leakage Policy](#5-traintest-leakage-policy)
6. [Windshield Geometry](#6-windshield-geometry)
7. [Reflection](#7-reflection)
8. [Ghost / Double Image](#8-ghost--double-image)
9. [Project Persistence](#9-project-persistence)
10. [Dependencies](#10-dependencies)
11. [OpenCV Compatibility](#11-opencv-compatibility)
12. [GitHub Actions / Testing](#12-github-actions--testing)
13. [Jetson](#13-jetson)
14. [Runtime Performance](#14-runtime-performance)
15. [Real Vehicle Validation](#15-real-vehicle-validation)
16. [Future Work](#16-future-work)

## 1. Architecture

```
app/            GUI(main.py) / 헤드리스 CLI(cli.py) 진입점
calibration/    순수 계산 로직 (UI 의존성 없음, 재사용 가능)
  types.py                전체가 공유하는 데이터 구조
  detector.py              ChArUco/Chessboard/Circle Grid/AprilGrid 검출
  models/                  pinhole / brown_conrady / extended_pinhole /
                           fisheye / object_releasing
  compare.py               Standard 4모델 동시 실행 + 비교표
  validation.py            Standard Hold-out 검증
  object_releasing_validation.py  Object-Releasing 전용 Hold-out
  recommender.py           Model Score 기반 추천
  windshield/               Windshield Geometry(Baseline/Spherical/
                           Residual Grid/RBF/Neural/Spline) + Ghost +
                           Reflection + Reflection Suppression
  project_io.py            .ccproj 저장/불러오기 (JSON)
export/         OpenCV YAML / ROS CameraInfo YAML / HTML / JSON / CSV
ui/             PySide6 화면 (계산 로직 없음, calibration/*만 호출)
```

`calibration/`은 UI와 완전히 독립적이다 - CLI/다른 프론트엔드에서도 그대로
재사용 가능. Windshield 계층(`calibration/windshield/`)은 Camera Intrinsic
계층 위에 얹히는 **별도 계층**으로, 6번 섹션의 원칙에 따라 서로 침범하지
않는다.

## 2. Camera Models

Standard Calibration은 아래 4개 모델을 **항상 같은 데이터셋/같은
train-test 분할로 동시에** 계산하고 비교한다(`calibration/compare.py::
run_all_models`).

| 모델 | 내부 식별자 | 계수 | OpenCV API | 장단점 / 추천 상황 |
|---|---|---|---|---|
| Ideal Pinhole | `pinhole` | `fx fy cx cy`, D=0 | `cv2.calibrateCamera` (distortion 없음) | 왜곡이 사실상 없는 특수 렌즈/기준선용. 대부분의 실카메라에는 부적합 - 비교 기준선 역할이 큼. |
| Brown-Conrady | `brown_conrady` | `fx fy cx cy` + `k1 k2 p1 p2 k3` (5계수 고정, runtime toggle 없음) | `cv2.calibrateCameraExtended` | 일반적인 raw 카메라 이미지의 기본값. 대부분의 웹캠/산업카메라에 적합. |
| Rational | `extended_pinhole` | `fx fy cx cy` + `k1 k2 p1 p2 k3 k4 k5 k6` (8계수 고정) | `cv2.calibrateCameraExtended` (`CALIB_RATIONAL_MODEL`) | 강한 방사 왜곡(광각)을 더 정밀하게 잡는다. 화각이 좁으면 Brown-Conrady와 통계적으로 구분이 어려워질 수 있음(10번 섹션 튜닝 결과 참고). |
| Fisheye (Kannala-Brandt) | `fisheye` | `fx fy cx cy` + `k1 k2 k3 k4` | `cv2.fisheye.*` | 초광각(어안) 렌즈 전용. `cv2.fisheye` 네임스페이스는 `cv2.calibrateCameraExtended`에 대응하는 Extended 버전이 없어 `calibration/models/fisheye.py`가 프레임별 오차/파라미터 불확실성을 직접 계산한다(아래 참고). |

Fisheye 전용 구현 세부사항(`calibration/models/fisheye.py`):

1. **초기값 안전장치**: 피쉬아이는 초기 `fx, fy` 추정이 나쁘면 최적화가
   발산한다. Pinhole 캘리브레이션 결과의 `fx, fy, cx, cy`를
   `CALIB_USE_INTRINSIC_GUESS`로 넘겨 시작점을 잡는다
   (Pinhole → Fisheye initial K → Fisheye calibration).
2. **프레임별 재투영 오차 수동 계산**: `cv2.fisheye.projectPoints()`를
   프레임마다 호출해서 `perViewErrors`가 없는 공백을 메운다.
3. **파라미터 불확실성 - Bootstrap Resampling**: `cv2.fisheye`에는
   `stdDeviationsIntrinsics`가 없어서, `calibrate_fisheye(...,
   estimate_uncertainty=True)`가 bootstrap resampling으로 우회한다
   (`_bootstrap_fisheye_uncertainty()`). 데이터셋이 `_BOOTSTRAP_MAX_FRAMES`
   (150장)를 넘으면 비용 대비 한계효용이 낮아 건너뛴다.

Pinhole/Brown-Conrady/Rational은 `cv2.calibrateCameraExtended()`가
`stdDeviationsIntrinsics`를 무료로 주므로 그걸 그대로 쓴다.

## 3. Calibration Targets

| 타겟 | 상태 | 비고 |
|---|---|---|
| ChArUco | Implemented, 기본값/권장 | 부분 검출을 허용하는 유일한 타겟이라 Object-Releasing 이외 대부분의 워크플로우에 가장 안정적. |
| Checkerboard(일반 체스보드) | Implemented | `findChessboardCornersSB`(강건, 실패 시 고전 방식+`cornerSubPix`로 재시도)를 쓴다. 대칭 패턴이라 (1) 보드 전체가 이미지 안에 다 보여야 검출되고 (2) 180도 돌려 찍은 것과 원리적으로 구분이 안 되는 근본 한계가 있다(OpenCV 표준 체스보드의 잘 알려진 한계, 이 프로젝트가 만든 문제가 아님). |
| Circle Grid | Implemented, Symmetric/Asymmetric 둘 다 | `cv2.findCirclesGrid`. `build_circle_grid_blob_detector()`가 호출자가 blob_detector를 직접 넘기지 않으면 dark(blobColor=0)/bright(blobColor=255) 극성과 `CALIB_CB_CLUSTERING` on/off 네 조합을 순서대로 시도한다 - 실측 결과 OpenCV의 asymmetric grid 검출이 이 두 축 모두에 민감했다(11번 섹션 개정 이력 참고). 극단적으로 작은 격자(예: 4x3 asymmetric, 12점)는 이 fallback으로도 원리적으로 검출되지 않는다 - OpenCV `CALIB_CB_ASYMMETRIC_GRID` 알고리즘 자체의 한계이며, 실사용 asymmetric 보드는 이보다 크게 쓴다(OpenCV 공식 샘플도 4x11). |
| AprilGrid | Implemented | Kalibr과 같은 row-major ID 배치. Variant는 OpenCV/AprilTag3 style(기본)과 Kalibr style(**Experimental**) 두 가지인데, 오늘 시점 둘 다 같은 OpenCV AprilTag detector 경로를 쓰고 로그 문구만 다르다(`calibration/detector.py::detect_aprilgrid`). Kalibr 스타일 fixture 회귀가 `tests/test_kalibr_aprilgrid_fixture.py`에 있지만, 실제 Kalibr 산출물로 재확인되기 전까지 "Kalibr Compatible"로 승격하지 않는다. |

**Partial detection**: ChArUco/AprilGrid는 보드 일부만 보여도 검출된
포인트만으로 계산에 참여한다(마커 기반이라 포인트별 ID 대응이 가능하기
때문). Checkerboard/Circle Grid는 전부 아니면 실패(findChessboardCorners*/
findCirclesGrid 자체의 특성)다 - 그래서 Object-Releasing(4번 섹션의
"Calibration Method")이 이 둘만 지원한다.

## 4. Validation

`calibration/validation.py`가 Standard Hold-out을 담당한다.

| 지표 | 정의 |
|---|---|
| Train RMS | 학습에 쓰인 프레임에서의 재투영 오차 RMS. |
| Hold-out(Test) RMS | 학습에 전혀 쓰이지 않은 프레임에서, intrinsic 파라미터를 고정한 채 pose만 재추정해 계산한 RMS - 진짜 generalization 지표(5번 섹션). |
| Median / P90 / P95 / P99 / Max | 코너 포인트 단위 오차 분포 요약. RMS만으로 안 보이는 꼬리(worst-case) 거동을 드러낸다. |
| Regional / Edge Error | 이미지의 중앙/가장자리/코너 영역별 오차. |
| Radial Profile | 이미지 중심으로부터의 거리(반지름)별 오차 분포. |
| AIC / BIC | 모델 복잡도(파라미터 수) 대비 적합도 - 파라미터가 많을수록 Train RMS는 항상 좋아지므로, 그 페널티를 준 지표로 모델 간 공정 비교를 돕는다. |
| Bootstrap | 최종 선택 모델을 N회 재표본으로 반복 계산해 파라미터 신뢰구간(CI)을 추정. |
| Stability (Repeated Hold-out) | 서로 다른 train/test 분할(seed)로 여러 번 반복한 hold-out 결과의 흩어짐(std) - 낮을수록 특정 분할에 우연히 의존하지 않는 안정적인 모델. |
| Observability | 파라미터가 실제로 데이터에 의해 잘 제약되는지(강한 상관/약한 제약이 있으면 불확실성이 커짐). |
| **Hold-out Evidence Gate** | Hold-out RMS가 낮다는 사실 하나만으로 "검증됨"이라 말할 수 없다 - test 프레임 수/코너 수/공간 coverage/자세(pose) 다양성이 충분한지 별도로 판정해 `sufficient`/`insufficient_evidence`/`not_evaluated`로 명시한다(`calibration/holdout_evidence.py`). RMS 숫자 자체를 바꾸거나 무효화하지 않는다. |

**Sequential validation 독립성**: 여러 모델을 같은 `train_ids`/`test_ids`로
순차 검증해도(`validate_holdout()`을 Pinhole → Brown-Conrady → ... 순서로
호출), 각 `ValidationResult.per_frame_error`는 서로 다른 dict 객체로
독립적으로 보존된다 - 같은 test `Frame` 객체를 여러 모델이 순차적으로
평가해도 마지막 모델의 값만 남는 mutation 버그가 없다
(`tests/test_validation.py::
test_sequential_holdout_validation_across_models_keeps_independent_test_errors`,
`test_holdout_test_evaluation_does_not_mutate_test_frame_reprojection_error`).
이 두 회귀 테스트는 CI에서 항상 안정적으로 재현되는 Pinhole/Brown-Conrady
조합을 쓴다 - 이 테스트의 목적은 "mutation independence" 자체를 확인하는
것이지 특정 모델의 적합 품질이 아니기 때문이다. Fisheye(Kannala-Brandt)
전용 정확도 검증은 실제 `cv2.fisheye.projectPoints()`로 만든 GT 데이터로
`test_fisheye_holdout_validation_recovers_known_k_d`가 별도로 담당한다 -
공용 synthetic fixture는 pinhole/Brown 왜곡 기반이라 Fisheye 모델
자체(Kannala-Brandt)의 생성 과정과 맞지 않으므로, "여러 모델을 순차
검증해도 안 섞인다"는 배선 테스트와 "Fisheye가 실제로 K,D를 정확히
복원하는가"라는 정확도 테스트를 분리해뒀다.

## 5. Train/Test Leakage Policy

**핵심 원칙(코드 전체에 강제됨, 회귀 테스트로 보호):**

- **Train**: `camera_matrix`/`distortion`(K, D) fitting 가능.
- **Test**: K, D를 **절대 수정하지 않는다** → pose(`solvePnP`/
  `cv2.fisheye.solvePnP`)만 재추정 → 그 pose로 재투영 오차만 계산.

`_evaluate_on_test()`(`calibration/validation.py`)가 이 원칙을 지키는
유일한 진입점이다 - `validate_holdout()`과
`recalibrate_train_with_outlier_pruning()` 둘 다 이 함수를 재사용해서
"test 평가"의 정의가 코드 두 곳에서 갈라지는 사고를 막는다. 이상치 제거
로직(`recalibrate_train_with_outlier_pruning`)도 test 프레임의 오차를 아예
계산하지 않으므로, test 프레임은 이상치 판정 후보에 구조적으로 오를 수
없다(`tests/test_validation.py::
test_leak_safe_outlier_pruning_only_removes_train_frames`).

Object-Releasing(4번 섹션)도 동일 원칙을 따르되 범위가 하나 더 넓다 - Train
프레임만으로 K/D **뿐 아니라 Refined Target Geometry**까지 확정하고, Test
프레임에는 그 셋을 전부 고정한 채 pose만 다시 구한다
(`calibration/object_releasing_validation.py`).

## 6. Windshield Geometry

Camera Intrinsic(2번 섹션) 위에 얹는 **별도 계층**이다. 실제 차량은 카메라
앞에 windshield(자동차 유리)가 있고, 그 유리가 굴절을 일으켜 순수 K/D
모델만으로는 설명 안 되는 잔차가 남는다 - 이 계층은 그 잔차를 모델링한다.

### 6.1 아키텍처 (절대 원칙)

```
Camera Intrinsic Calibration (2번 섹션)
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

### 6.2 모델

| 모델 | 상태 | 설명 |
|---|---|---|
| Baseline | Implemented | 보정 없음(항등) - Base K,D 그대로 투영/역투영. Windshield 효과가 실제로 얼마나 되는지 측정하는 기준선. |
| Spherical | Implemented | Windshield를 단일 구(sphere)로 근사하고 Snell 굴절 법칙(공기→유리→공기, 굴절률/유리 두께 고정 파라미터)을 광선 단위로 실제 계산. |
| Residual Ray (Grid) | Implemented | Base Ray(순수 카메라 광선)에 이미지 위 성긴 grid에 저장된 3D 방향 보정을 bilinear interpolation으로 더한다 - closed-form 물리 모델이 아니라 관측 데이터 기반 보정. |
| Residual Ray (RBF) | Implemented | 같은 아이디어지만 grid 대신 `scipy.interpolate.RBFInterpolator`(thin plate spline)로 성긴 center 집합을 보간 - 불규칙한 코너 분포에서 Grid보다 유연할 수 있다. |
| Spline | Implemented | Base Sphere(고정) 위에 bicubic B-spline으로 국소 표면 변형을 추가하고, 그 변형된 표면에서 실제 Snell 굴절을 계산 - Spherical보다 표현력이 높지만 코너마다 광선-표면 교차를 두 번(안쪽+바깥쪽) 풀어야 해서 계산이 훨씬 무겁다. |
| Residual Ray (Neural) | Experimental, 선택적 의존성 | Grid/RBF와 같은 계열이지만 보정을 작은 MLP(PyTorch)가 학습한다 - CPU inference만으로 동작(GPU 불필요). 10번 섹션 참고. |

Grid/RBF/Neural은 전부 `WindshieldModelType.RESIDUAL_RAY` 하나의 enum
아래 variant(`residual_ray_hint["method"]`로 구분)다.

모든 모델은 공통 런타임 API를 구현한다(`calibration/windshield/base.py`):
`project_point(x, y, z) -> (u, v)`, `unproject_pixel(u, v) -> (dx, dy, dz)`.
어떤 계산된 결과(`WindshieldCalibrationResult`)로도 `build_projector(result)`
하나로 실행 가능한 모델 인스턴스를 얻는다(`calibration/windshield/
projection.py`).

## 7. Reflection

Windshield에 비치는 반사(하늘/차내 조명/대시보드 등)를 평가한다. **두
모드는 신뢰 수준이 완전히 다르다**:

- **Reference Mode**(실측, Implemented): 같은 장면을 반사가 없는(또는
  적은) 기준 이미지와 비교해서 실제 반사 강도(gain/bias 보정 후 차이)를
  측정한다. Mean/Median/P95/P99/Max/Coverage, 영역별(Center/Top/Bottom/
  Left/Right/Corners) 지표, 자동차 특화 하단 ROI 지표까지 낸다.
- **No-Reference Mode**(heuristic, Experimental): 기준 이미지가 없을 때
  이미지 자체의 통계만으로 반사 가능성을 추정한다. 결과는
  `reflection_likelihood`에만 채워지고 `is_likelihood=True`가 항상 함께
  붙는다 - **절대 Ground Truth가 아니다.**

Reflection과는 별개로 **Glare**(과도한 밝기+낮은 대비 영역)와
**Saturation**(픽셀값 clipping 비율)도 독립된 지표로 낸다 - 셋을 하나로
합치지 않는다.

**Reflection Alignment**(Reference mode 전용): Reference 이미지를 Normal
이미지에 맞추는 정렬(translation 또는 affine)이 `cv2.findTransformECC`
(`cv2.MOTION_TRANSLATION`/`cv2.MOTION_AFFINE`) 기반으로 이뤄지고, `method`
값이 지원 목록 밖이면 조용히 fallback하지 않고 명시적으로 실패한다. 정렬로
생긴 인공적인 border(예: `cv2.BORDER_REFLECT`로 채워진 영역)는 **Valid Warp
Mask**로 계산돼 모든 Reflection 지표 계산에서 제외된다. Affine을 쓰면
translation/rotation/scale/shear 진단값이 함께 기록되고, 비정상적으로 큰
변환은 `warning` 상태로 표시된다(정렬 실패 가능성).

**Reflection Suppression**(Experimental, 선택적 의존성): 별도로 학습된
PyTorch 모델(`calibration/windshield/reflection_suppression/`)을 쓴다 -
아래 Ghost Suppression(결정론적, PyTorch 불필요)과는 완전히 다른 접근이다.

## 8. Ghost / Double Image

Windshield의 이중 표면(안쪽/바깥쪽) 반사가 만드는 유령상(ghost, 원본보다
어둡고 살짝 어긋난 복제 이미지)을 평가한다. **Reflection과는 완전히 다른
현상**이다 - Reflection은 "밝기가 얼마나 겹쳐 보이는가"를, Ghost는 "같은
형체가 얼마나 어긋나서 두 번 보이는가(displacement)"를 다룬다.

- **Point-source Ghost Evaluation**(Implemented): 밝은 광원(LED, 헤드라이트
  등)의 Main 피크와 Ghost 피크를 짝지어 displacement(dx, dy)와 강도 비율을
  직접 측정한다. 촘촘한 LED array에서 이웃 Main끼리 잘못 짝지어지는 문제를
  막기 위해 **2-pass spatial pairing**을 쓴다: PASS 1이 전체 이미지의
  dominant displacement vector로 대략적인 pairing을 하고, PASS 2가 그
  결과로 만든 성긴 spatial field에서 위치별 local vector를 다시 추정해
  재-pairing한다(windshield 곡률 등으로 위치마다 실제 ghost displacement가
  달라지는 경우 대응) - PASS 2가 실패하거나 근거가 부족하면 항상 PASS 1로
  안전하게 fallback한다.
- **General (No-Reference) Likelihood**(Experimental): Point-source 타겟
  없이 일반 도로 영상에서 이미지의 row/column profile을 스캔해 double-edge
  패턴 비율을 heuristic하게 추정한다. `ghost_likelihood`/`is_likelihood=True`
  로만 노출되고 절대 Ground Truth가 아니다 - `ghost_likelihood`와 실측 강도
  지표(`mean_strength`)를 같은 필드에 절대 섞지 않는다.
- **GhostField**(Implemented): Dataset 전체의 point-source detection을
  robust median/MAD로 집계해 만드는 성긴 2D displacement/strength field.
  유효 detection이 0개면 "ghost 없음"과 "ghost=0으로 측정됨"을 구분하기
  위해 명시적으로 fitting을 거부한다(조용히 0으로 채우지 않음).
- **Ghost Suppression**(Implemented, PyTorch 불필요): `I = T +
  α·W_delta(T)`의 결정론적 역변환으로 ghost를 제거한다
  (`suppress_ghost(image, ghost_field)`). Strength 감소만으로는 성공이
  아니므로 edge retention/clean-region 변화까지 함께 본다
  (over-suppression score).

## 9. Project Persistence

`.ccproj` 파일(JSON, pickle 아님 - "불러올 때 임의 코드 실행 위험"을
피하려고 일부러 더 번거로운 JSON 직렬화를 택했다)에는:

- **저장되는 것**: 카메라/패턴 설정, 데이터셋(이미지별 검출 결과·품질
  점수·상태), Standard 4모델 캘리브레이션 결과, Hold-out Validation 결과,
  추천 점수, 이상치 제거 이력, 최종 결과, Object-Releasing 결과/Hold-out/
  비교, `windshield` 브랜치에서는 `windshield_results`/
  `reflection_results`/`ghost_models` 등 Windshield 계층 산출물까지 - 화면에
  보이는 사실상 모든 것.
- **저장 안 되는 것**: 원본 이미지 파일 자체(바이트 복사 안 함, 경로만
  저장) - "파일을 삭제/복제하지 않는다"는 원칙과 같은 이유. 원본 이미지가
  없어지거나 옮겨져도 불러오기/재계산/export는 그대로 되고, UI의 이미지
  미리보기만 해당 이미지를 못 보여준다.

`calibration/project_io.py`(저장/불러오기), `calibration/json_utils.py`
(dataclass/numpy → JSON 안전 변환, `export/json_export.py`와 공유)가 구현을
담당한다.

## 10. Dependencies

**Core (desktop) - `requirements.txt` == `pyproject.toml`
`[project] dependencies`** (둘이 다른 패키지/버전을 선언하면 `pip install -e
.`와 `pip install -r requirements.txt`가 다른 환경을 만들게 되므로 항상
동일하게 유지한다):

| 패키지 | 최소 버전 | 용도 |
|---|---|---|
| `opencv-contrib-python-headless` | `>=4.10` | ChArUco/AprilGrid/Circle Grid 검출, 캘리브레이션 계산. `cv2.aruco`는 contrib 빌드에만 있다. `>=4.10`인 이유는 11번 섹션. headless인 이유: GUI는 PySide6가 전담하고 이 프로젝트 어디서도 `cv2` 자체의 highgui(`cv2.imshow` 등)를 쓰지 않으므로, OpenCV가 번들하는 Qt5 plugin이 PySide6의 Qt6 platform plugin과 충돌할 여지를 원천 차단한다. |
| `numpy` | `>=1.24` | 행렬/배열 연산 |
| `scipy` | `>=1.11` | Windshield 모델 피팅(root-solve, RBF 보간 등) |
| `PyYAML` | `>=6.0` | ROS CameraInfo YAML export, 프로젝트/설정 파일 |
| `PySide6` | `>=6.5.3` | 데스크톱 UI (Qt6). JetPack 5.1.2의 aarch64 wheel 제약과 metadata를 맞추기 위해 core minimum도 6.5.3으로 둔다. |

> ⚠️ `opencv-python`/`opencv-python-headless`/`opencv-contrib-python`
> (비-headless)을 `opencv-contrib-python-headless`와 동시에 설치하지 마라 -
> 전부 `cv2`라는 이름을 써서 마지막에 설치된 것이 이긴다(원인 파악이 매우
> 어려운 silent breakage).

**Optional extras** (`pyproject.toml [project.optional-dependencies]`):

| Extra | 설치 커맨드 | 내용 |
|---|---|---|
| `ros` | `pip install -e ".[ros]"` | `rosbags>=0.9` - **ROS를 설치하지 않고도** `.bag`/`.db3`/`.mcap`에서 이미지를 오프라인으로 직접 읽는 순수 Python 라이브러리. ROS1 live topic 의존성(`rospy`/`rclpy`/`cv_bridge`)과는 다르다 - 그건 어떤 extra로도 설치되지 않고 실제 ROS 환경이 시스템에 설치/source되어 있어야 한다. |
| `neural` | `pip install -e ".[neural]"` | `torch>=2.0` - Neural Residual Windshield 모델(6.2번 섹션) 전용. Jetson은 일반 PyPI wheel 대신 JetPack에 맞는 NVIDIA 제공 PyTorch wheel을 별도 설치해야 한다 - `requirements-jetson*.txt`에는 의도적으로 `torch`를 넣지 않았다. |
| `dev` | `pip install -e ".[dev]"` | `pytest>=7.4` + `rosbags>=0.9`(테스트가 rosbag 코드도 검증하므로) - `requirements-dev.txt`와 동일 구성(`pytest-cov`는 로컬 커버리지 측정용으로 `requirements-dev.txt`에만 추가로 있다, 12번 섹션 참고). |

**Jetson**은 별도 플랫폼 정책이라 desktop과 억지로 동일하게 맞추지 않는다
(13번 섹션) - `requirements-jetson.txt`/`requirements-jetson-jp512.txt`가
JetPack별 호환 wheel을 고정한다.

## 11. OpenCV Compatibility

**최소 지원 버전: `opencv-contrib-python-headless>=4.10`.**

이유: `calibration/models/common.py::solve_pnp_for_model`이 Hold-out
test 평가(고정 K,D로 pose만 재추정)와 Windshield 계층 양쪽에서
`cv2.fisheye.solvePnP()`를 조건 없이 호출한다. 이 함수는 OpenCV
**4.10.0**에서 추가됐다([opencv/opencv#25028](https://github.com/opencv/opencv/pull/25028),
2024-02-16 merge, target 4.10.0) - 그 이전 버전에는 아예 없어
`AttributeError`로 즉시 실패한다. 예전에는 `requirements.txt`/
`pyproject.toml`이 `>=4.9`를 선언하고 있었는데(`requirements-jetson*.txt`는
이미 `==4.10.0.84`로 고정돼 있어 서로 모순됐다), 이번에 `>=4.10`으로
통일했다.

**OpenCV 5.0 대응**(`opencv-contrib-python==5.0.0.93`으로 직접 설치해
재현/검증):

1. OpenCV 5.0부터 `cv2.fisheye.CALIB_*` 플래그들이 최상위 `cv2.CALIB_*`로
   옮겨갔다(`cv2.fisheye` 네임스페이스엔 더 이상 없음) - `_fisheye_flag()`
   (`calibration/models/fisheye.py`)가 `cv2.fisheye`에서 못 찾으면 최상위
   `cv2`에서 다시 찾는다.
2. `cv2.fisheye.calibrate()`는 `(N,1,3)`/`(N,1,2)` shape(`cv2.calibrateCamera`
   관례)을 받지 않고 `(1,N,3)`/`(1,N,2)`를 요구한다 - OpenCV 4.13.0에서는
   그냥 넘어가던 게 5.0.0부터 엄격해졌다. `_to_fisheye_points()`가 항상
   `(1,N,·)`로 명시 변환한다.

회귀 테스트: `tests/test_fisheye_opencv5_compat.py`(4.x/5.x 둘 다 통과
확인).

**Circle Grid 검출 극성/CLUSTERING 민감도**: `cv2.findCirclesGrid`(특히
`CALIB_CB_ASYMMETRIC_GRID`)는 blob polarity(어두운 원 vs 밝은 원)와
`CALIB_CB_CLUSTERING` 플래그 사용 여부 양쪽에 검출 성공 여부가 좌우된다 -
실측(로컬 재현)으로 4x4/5x4 같은 실사용 크기에서도 어느 한 조합만으로는
안정적으로 못 잡는 경우를 확인했다. `build_circle_grid_blob_detector()`가
호출자로부터 명시적 `blob_detector`를 받지 않으면(기본 경로,
`build_detect_fn()`의 dispatch가 실제로 쓰는 경로), dark/bright 두 극성 x
CLUSTERING on/off 두 옵션 = 최대 4가지 조합을 순서대로 시도한다
(`calibration/detector.py::detect_circle_grid`). 호출자가 `blob_detector`를
직접 넘기면 그 detector를 그대로 존중하고 fallback을 타지 않는다(기존 API
계약 유지). 회귀 테스트: `tests/test_circle_grid.py`(symmetric/asymmetric x
dark/light 조합).

**Ghost Edge Target**: `cv2.HoughLinesP()` 반환 shape이 OpenCV 버전에 따라
`(N,1,4)`/`(N,4)`로 달라질 수 있어 양쪽을 모두 처리하도록
`calibration/windshield/ghost/edge_detector.py`에서 대응했다.

**"4.x/5.x 전체 테스트 스위트 통과"는 실제 CI 실행 결과로만 주장한다** -
12번 섹션의 CI 상태 표를 최신 상태 기준으로 참고할 것.

## 12. GitHub Actions / Testing

`tests/`에 pytest 스위트가 있다(정확한 개수는 계속 늘어나므로 `pytest
--collect-only -q`로 직접 확인 권장). `tests/conftest.py`가 왜곡이 실제로
적용된 합성 ChArUco 이미지 데이터셋을 세션당 한 번만 만들어 여러 테스트가
공유한다.

**두 티어:**

- **빠른 티어**(마커 없음, `pytest -m "not slow"`로 걸러짐): 단위 테스트
  대부분 + `test_smoke_pipeline.py`(3D→2D 직접 사영, 이미지 렌더링/검출
  없이 Ideal Pinhole+Rational 2모델만, 작은 데이터셋). Object-Releasing
  전용 검증도 합성 데이터/생성된 fixture만 써서 이 티어에 포함된다.
- **느린 티어**(`@pytest.mark.slow`): 실제 ChArUco 이미지 렌더링+검출,
  Standard 4모델 전부, Hold-out validation, export까지 포함하는 진짜 통합
  테스트(`test_pipeline_integration.py` 등). Spline Windshield 모델은
  코너마다 물리적으로 정확한 ray-surface intersection을 풀어야 해서 최소
  grid에서도 `calibrate_spline()` 1회 호출이 수 분 걸릴 수 있다 - 이 비용을
  가진 파일/테스트(`test_windshield_spline.py`,
  `test_windshield_spline_stabilization.py`,
  `test_windshield_export.py::test_export_windshield_yaml_round_trip_spline`,
  `test_windshield_project_io.py::test_spline_result_round_trips_through_project`)
  도 이 티어로 뺐다 - `pytest`(마커 없이 전체 실행)로는 여전히 돈다.

> **Spherical zero-refraction policy**: `n_air == n_glass`는 굴절이 없는
> 퇴화 케이스다. Projection은 Baseline과 같아야 하지만, sphere
> center/radius는 영상 관측으로 식별되지 않는다. 그래서
> `calibrate_spherical()`은 이 조건에서 성공한 geometry fit을 꾸며내지 않고
> `success=False`와 unobservable error를 반환한다. 회귀 테스트는 projection
> sanity와 calibration observability failure를 분리한다.

```bash
pip install -e ".[dev]"
pytest                  # 전체 실행
pytest -m "not slow"    # 느린 통합 테스트 빼고 빠르게만
pytest tests/test_straightness.py -v   # 특정 파일만
```

**GitHub Actions 워크플로우 5개** - 전부 push/PR마다 자동으로 돈다:

| 워크플로우 파일 | 이름 | 대상 |
|---|---|---|
| `.github/workflows/ci.yml` | General CI | `pytest -q -m "not slow"`(핵심 회귀) - Python 3.10/3.11 매트릭스, ubuntu-latest. `libegl1`/`libgl1`을 설치하고 `QT_QPA_PLATFORM=offscreen`으로 PySide6 의존 테스트까지 헤드리스로 돌린다(Xvfb 등 실제 디스플레이 서버는 안 씀). `torch`(neural extra)는 설치하지 않는다 - Neural 전용 회귀는 `neural-tests.yml`이 담당. |
| `.github/workflows/ghost-tests.yml` | Ghost Windshield Tests | Ghost/Double Image 평가·억제·UI-architecture·Project IO 회귀 |
| `.github/workflows/reflection-tests.yml` | Reflection Windshield Tests | Reflection 평가·UI-architecture·Project IO·Export 회귀 |
| `.github/workflows/reflection-suppression-tests.yml` | Reflection Suppression Tests | Reflection Suppression(PyTorch) 회귀 |
| `.github/workflows/neural-tests.yml` | Neural Windshield Tests | Neural Residual Windshield 모델(PyTorch) 회귀 |

**커버리지**: `requirements-dev.txt`의 `pytest-cov`로 로컬에서는 바로 쓸 수
있다(`pytest --cov=calibration --cov=app --cov=export --cov=ui`). 이
브랜치의 실제 GitHub Actions 워크플로우 중 커버리지를 측정/리포트하는 잡은
**현재 없다**.

**최신 CI 실행 결과**는 README.md 상단 배지와, 필요하면 저장소의 Actions
탭(`https://github.com/yaaaaangeo/camera_calibrator/actions`)에서 직접
확인해라 - 이 문서에 박제된 PASS/FAIL 문구보다 항상 그쪽이 최신이다.

## 13. Jetson

두 프로필이 있고, **Desktop과 억지로 동일하게 맞추지 않는다** - Python
ABI/apt 패키지 제약이 실제로 다르기 때문이다.

| 프로필 | JetPack | Python | 상태 |
|---|---|---|---|
| JetPack 6.2.1, Jetson AGX Orin | 6.2.1 | 시스템 기본 | `scripts/install_jetson.sh` - Ubuntu 22.04 ARM64용 호환 wheel 고정. 자세한 전제조건/실행법/현장 검증은 [JETSON.md](../JETSON.md). |
| JetPack 5.1.2, Jetson AGX Orin Developer Kit 64GB | 5.1.2(R35.4.1) | **3.10 venv** | `scripts/install_jetson_jp512.sh` - Ubuntu 20.04 기본 Python 3.8을 쓰지 않고 3.10 venv를 별도로 만든다(`--system-site-packages` 안 씀 - ROS1 Noetic `cv_bridge`가 3.8용으로 빌드돼 있어 3.10 인터프리터와 ABI가 안 맞기 때문). |

두 프로필 모두 `opencv-contrib-python-headless`를 쓴다(OpenCV가 번들한 Qt와
PySide6의 Qt runtime 충돌을 피하려는 이유는 Desktop과 동일 - 10번 섹션).
ROS1 Noetic live topic(`rospy`, `sensor_msgs`, `cv_bridge`)은 pip
dependency가 아니다 - ROS/apt 환경에서 관리하고 필요할 때 source한다.

JetPack 5.1.2 profile의 지원 범위(Python 3.8/3.10 ABI 불일치 때문에 실제로
이렇게 나뉜다):

| 대상 | 상태 |
|---|---|
| JetPack 5.1.2, Python 3.10 venv에서 GUI/core 캘리브레이션 워크플로우(이미지 파일/rosbag 오프라인 처리 포함) | Implemented |
| 같은 Python 3.10 venv 안에서 ROS1 Noetic `cv_bridge`로 실시간(live) 토픽 구독 | 지원 안 함(`cv_bridge`가 Python 3.8용으로 빌드되어 3.10 인터프리터에서 로드 자체 불가) |
| ROS1 Noetic 환경(별도 시스템 Python 3.8)에서 직접 실행 | 이 프로젝트가 다루는 범위 밖 |

이 저장소의 CI는 Jetson 하드웨어를 갖고 있지 않다 - Jetson 관련 수치(설치
성공 여부, 런타임 성능)는 실제 Jetson 기기에서 직접 확인해야 한다(14번,
15번 섹션과 동일한 원칙).

## 14. Runtime Performance

`build_projector(result)`가 만드는 point-by-point Windshield 모델은
Calibration/Evaluation에는 충분히 빠르지만, Camera-LiDAR 프로젝션처럼
프레임당 수만~수백만 포인트를 처리해야 하는 런타임에는 느릴 수 있다(특히
Spherical/Spline/Neural은 포인트마다 root-solve/surface-intersection이
필요). `calibration/windshield/runtime_projector.py`가 별도 API
(`build_runtime_projector()` - 기존 `build_projector()`는 그대로 유지)를
추가한다:

- **Batch(Exact)**: `project_points_exact_batch()`/
  `unproject_pixels_exact_batch()` - 계산 자체는 point-by-point와 100%
  동일(근사 아님), (N,3)/(N,2) 배열 하나로 넘길 수 있는 벡터화 껍데기.
  Baseline은 실제로 벡터화된 `cv2.projectPoints` 경로를 타 진짜 빨라지고,
  나머지는 반복 호출 오버헤드만 줄인다.
- **LUT(근사)**: `RuntimeWindshieldProjector` - 이미지 전체를 성긴 grid로
  미리 샘플링해 둔 뒤 bilinear 보간(`unproject_pixels`)/k-최근접
  역거리가중평균(`project_points`)으로 조회한다. `unproject_pixels`는 격자
  해상도만큼의 보간 오차만 생기지만, `project_points`는 "포인트의 방향만"
  쓰는 근사라 windshield에 아주 가까운 포인트(수십 cm 이내)일수록 오차가
  커질 수 있다(parallax) - LiDAR 포인트처럼 windshield-camera 간격보다
  훨씬 먼 포인트에서는 근사 오차가 작을 것으로 기대되지만, **실측 없이 이
  가정을 신뢰하지 않는다.** `validate_runtime_projector_vs_exact()`가
  median/P95/P99/Max 픽셀 오차(project)와 각도 오차(unproject)를 실제로
  측정한다.
  `project_points_with_mask()`는 NaN/Inf/zero-vector/behind-camera 입력을
  batch 전체 예외로 만들지 않고 해당 row만 `[nan, nan]`, `valid_mask=False`
  로 반환한다. Validation report는 측정 가능한 sample이 없으면 metric을
  `None`으로 둔다.

```bash
python scripts/benchmark_windshield_runtime.py
```

1/100/1,000/10,000 포인트 스케일 x Scalar/Batch(Exact)/LUT 세 티어를
CPU-first로 측정한다(Jetson에서도 실행 가능). **이 스크립트가 출력하는
숫자는 그 스크립트를 실행한 기기의 실측치일 뿐이다** - Jetson 성능은 실제
Jetson에서 직접 돌려야 하며, 이 저장소의 CI는 Jetson 하드웨어를 갖고 있지
않으므로 Jetson 숫자를 대신 만들어내지 않는다.

## 15. Real Vehicle Validation

`calibration/vehicle_validation.py`가 실제 차량에서 촬영한 여러
세션(차량 x 시간대 x windshield 상태)을 비교하기 위한 메타데이터/집계/
비교 프레임워크를 제공한다 - Day/Night 비교, Geometry(Hold-out) 지표와
Photometric(Reflection/Ghost) 지표를 절대 하나로 섞지 않는 두 개의 독립된
요약(`SessionGeometrySummary`/`SessionPhotometricSummary`)으로 관리한다.

> ⚠️ **NOT YET VALIDATED ON REAL VEHICLE.** 이 저장소에는 실제 차량 캡처
> 데이터셋이 전혀 없다. 위 프레임워크는 그런 데이터가 생기면 곧바로 쓸 수
> 있는 틀만 제공할 뿐, 어떤 실차 검증 결과도 만들어내지 않는다. 모든 세션
> 메타데이터는 `data_provenance`("real_vehicle"/"synthetic"/"unknown")를
> 명시해야 하고, `real_vehicle_validation_disclaimer()`는 real_vehicle
> 세션이 하나도 없으면 이 사실을 항상 그대로 보고한다.

같은 이유로, Desktop GUI 수동 실행과 Jetson 두 프로필도 "설치 스크립트/
코드는 있으나 실제 기기에서의 end-to-end 실행 확인은 제한적"인 상태다 -
정확한 최신 상태는 README.md의 "결과를 믿기 전에" 절 및 이 저장소의 최신
커밋/CI 기준으로 판단할 것(박제된 문구를 그대로 믿지 말 것).

## 16. Future Work

아래는 **현재 구현되어 있지 않다** - 계획/아이디어 수준이며, 위 섹션들의
"Implemented"와 절대 혼동하지 말 것.

- **CAD 기반 Windshield 형상**: 현재 Windshield Geometry 모델(6번 섹션)은
  전부 단일 구/성긴 보정장/스플라인 등 파라메트릭 근사다. 실제 windshield
  설계 형상(CAD/STEP)을 불러와 surface prior로 쓰거나, CAD 형상 + residual
  spline을 결합하는 접근은 아직 코드베이스에 없다.
- **Day/Night/Session 반복성(repeatability)의 실측 정량화**: 15번 섹션의
  프레임워크는 틀만 있고, 실제 여러 세션 데이터로 얻은 반복성 수치는 아직
  없다.
- **실차(Real Vehicle) 검증**: 15번 섹션 참고 - 실차 데이터셋 자체가
  없으므로 그 어떤 실차 기반 결론도 아직 낼 수 없다.
- **Jetson 두 프로필의 실기기 현장 검증**: 설치 스크립트/preflight
  체크만 있고, 실제 Jetson 기기에서의 반복 실행 확인은 제한적이다.

이 목록에 있는 항목을 실제로 구현하게 되면, 그 즉시 해당 섹션의 상태
표시를 "Future Work"에서 "Implemented"/"Experimental"로 옮기고 이 목록에서
지운다 - 계획과 구현 완료 상태가 이 문서 안에서 섞이지 않게 하는 것이
원칙이다.

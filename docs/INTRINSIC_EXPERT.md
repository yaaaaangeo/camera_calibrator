# 내부파라미터(Intrinsic) — Expert / Research Guide

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
6. [Windshield / Reflection / Ghost (제거됨)](#6-windshield--reflection--ghost-제거됨)
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
  project_io.py            .ccproj 저장/불러오기 (JSON)
export/         OpenCV YAML / ROS CameraInfo YAML / HTML / JSON / CSV
ui/             PySide6 화면 (계산 로직 없음, calibration/*만 호출)
```

`calibration/`은 UI와 완전히 독립적이다 - CLI/다른 프론트엔드에서도 그대로
재사용 가능.

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
findCirclesGrid 자체의 특성)다 - 그래서 Object-Releasing(라이브러리 전용,
아래 참고)이 이 둘만 지원한다.

## 4. Validation

`calibration/validation.py`가 Standard Hold-out을 담당한다.

> **GUI 범위**: 앱은 **Standard 캘리브레이션 고정**이다. Object-Releasing
> (`calibrateCameraRO`), Repeated K-Fold(Paper Metrics), Scene Ranking / Best
> Subset 재계산은 GUI 에서 제외됐고, 계산 코드(`calibration/object_releasing*.py`,
> `calibration/kfold.py`, `calibration/paper_evidence.py`, `calibration/scene_quality.py`,
> `calibration/subset_comparison.py`)와 그 테스트만 라이브러리로 남아 있다.
> 예전 프로젝트 파일(.ccproj)에 저장된 해당 결과도 그대로 읽힌다(화면에 표시하지 않을 뿐).

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

## 6. Windshield / Reflection / Ghost (제거됨)

Windshield Geometry(Baseline/Spherical/Residual Grid·RBF·Neural/Spline),
Reflection, Reflection Suppression, Ghost 평가·억제 기능과 관련 UI·Export·
프로젝트 저장 항목·CI 워크플로는 2026-11-10에 모두 제거했다.

이유: 실차 rosbag(앞유리 안쪽 econ 960×540, sensing 3840×2160, 대조군으로
유리 없는 우측·후방)으로 분석한 결과, 앞유리 안쪽에서 찍어 Camera Intrinsic을
계산하면 앞유리 굴절이 K·D에 함께 흡수됐다. 물리 계산(평판·구면 유리, 5mm,
60°)으로 유리가 만드는 이동 1.2~2.6px 중 97% 안팎이 흡수되고, 남는 오차는
먼 거리에서 0.11~0.16px(960×540)로 같은 카메라를 다시 캘리브레이션할 때의
차이보다 작았다. Spherical 모델은 Baseline보다 오히려 나빴다.

운용 원칙: 앞유리 안쪽 카메라는 **장착 상태 그대로 유리 너머로** Camera
Intrinsic을 계산한다. 유리를 교체하거나 카메라를 다시 장착하면 다시 계산한다.
예전 `.ccproj`에 저장된 `windshield_*`/`reflection_results`/`ghost_*` 항목은
불러올 때 무시된다. 기능이 다시 필요하면 git 기록(제거 이전 커밋)에서 복원할 수 있다.

## 9. Project Persistence

`.ccproj` 파일(JSON, pickle 아님 - "불러올 때 임의 코드 실행 위험"을
피하려고 일부러 더 번거로운 JSON 직렬화를 택했다)에는:

- **저장되는 것**: 카메라/패턴 설정, 데이터셋(이미지별 검출 결과·품질
  점수·상태), Standard 4모델 캘리브레이션 결과, Hold-out Validation 결과,
  추천 점수, 이상치 제거 이력, 최종 결과, Object-Releasing 결과/Hold-out/
  비교 - 화면에 보이는 사실상 모든 것.
- **저장 안 되는 것**: 원본 이미지 파일 자체(바이트 복사 안 함, 경로만
  저장) - "파일을 삭제/복제하지 않는다"는 원칙과 같은 이유. 원본 이미지가
  없어지거나 옮겨져도 불러오기/재계산/export는 그대로 되고, UI의 이미지
  미리보기만 해당 이미지를 못 보여준다.

`calibration/project_io.py`(저장/불러오기), `calibration/json_utils.py`
(dataclass/numpy → JSON 안전 변환, `export/json_export.py`와 공유)가 구현을
담당한다.

## 10. Dependencies

의존성은 저장소 루트의 `requirements.txt` 하나로 관리한다(내부파라미터 + 카메라·LiDAR 룸 공통).
`./install.sh`가 루트 `venv/`에 설치한다.

| 패키지 | 버전 | 용도 |
|---|---|---|
| `opencv-contrib-python-headless` | `>=4.10,<5` | ChArUco/AprilGrid/Circle Grid 검출, 캘리브레이션 계산. `cv2.aruco`는 contrib 빌드에만 있다. headless인 이유: GUI는 PySide6가 전담하므로 OpenCV가 번들하는 Qt5 plugin과의 충돌을 원천 차단. `<5`인 이유는 11번 섹션. |
| `numpy` | `>=1.24` | 행렬/배열 연산 |
| `scipy` | `>=1.11` | 보조 수치 계산 |
| `PyYAML` | `>=6.0` | ROS CameraInfo YAML export, 프로젝트/설정 파일 |
| `PySide6` | `>=6.5.3` | 데스크톱 UI (Qt6) |
| `rosbags` | `>=0.9` | ROS 설치 없이 `.bag`/`.db3`/`.mcap`에서 이미지·포인트클라우드를 읽는 순수 Python 라이브러리 |
| `pyqtgraph`, `PyOpenGL` | | 카메라·LiDAR 룸의 3D 포인트클라우드 뷰 |

> ⚠️ `opencv-python`/`opencv-python-headless`/`opencv-contrib-python`
> (비-headless)을 `opencv-contrib-python-headless`와 동시에 설치하지 마라 -
> 전부 `cv2`라는 이름을 써서 마지막에 설치된 것이 이긴다.

ROS live topic(`rospy`/`rclpy`/`cv_bridge`)은 pip로 설치하지 않는다 - 실제 ROS 환경을
source해야 한다. **Jetson**은 별도 플랫폼 정책이라 `intrinsic/requirements-jetson*.txt`가
JetPack별 호환 wheel을 고정한다(13번 섹션).

## 11. OpenCV Compatibility

**최소 지원 버전: `opencv-contrib-python-headless>=4.10`.**

이유: `calibration/models/common.py::solve_pnp_for_model`이 Hold-out
test 평가(고정 K,D로 pose만 재추정)에서
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

**OpenCV 5.x는 지원하지 않는다(`<5`).** 위 호환 처리에도 불구하고 5.1.0에서
`tests/test_smoke_pipeline.py`의 Rational 모델이 물리 왜곡 검증
(`rational_denominator_zero_near_zero_or_uncertified`)에서 실패하는 것을 확인했다.
4.10.0.84에서는 통과한다.

## 12. Testing

`intrinsic/tests/`에 pytest 스위트가 있다. `tests/conftest.py`가 왜곡이 실제로
적용된 합성 ChArUco 이미지 데이터셋을 세션당 한 번만 만들어 여러 테스트가 공유한다.

- **빠른 티어**(`-m "not slow"`): 단위 테스트 대부분 + `test_smoke_pipeline.py`.
- **느린 티어**(`@pytest.mark.slow`): 실제 ChArUco 이미지 렌더링+검출, 4모델 전부,
  Hold-out validation, export까지 포함하는 통합 테스트.

```bash
./run_tests.sh                 # 저장소 루트에서. 빠른 티어
./run_tests.sh --all           # 전체 (느린 티어 포함)
./run_tests.sh tests/test_straightness.py -v   # 특정 파일만
```

`run_tests.sh`는 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`로 실행한다 - ROS 환경이 source된
셸에서는 ROS의 pytest 플러그인(`launch_testing`)이 끼어들어 수집 단계에서 실패하기 때문.

## 13. Jetson

두 프로필이 있고, **Desktop과 억지로 동일하게 맞추지 않는다** - Python
ABI/apt 패키지 제약이 실제로 다르기 때문이다.

| 프로필 | JetPack | Python | 상태 |
|---|---|---|---|
| JetPack 6.2.1, Jetson AGX Orin | 6.2.1 | 시스템 기본 | `intrinsic/scripts/install_jetson.sh` - Ubuntu 22.04 ARM64용 호환 wheel 고정. 자세한 전제조건/실행법/현장 검증은 [JETSON.md](JETSON.md). |
| JetPack 5.1.2, Jetson AGX Orin Developer Kit 64GB | 5.1.2(R35.4.1) | **3.10 venv** | `intrinsic/scripts/install_jetson_jp512.sh` - Ubuntu 20.04 기본 Python 3.8을 쓰지 않고 3.10 venv를 별도로 만든다(`--system-site-packages` 안 씀 - ROS1 Noetic `cv_bridge`가 3.8용으로 빌드돼 있어 3.10 인터프리터와 ABI가 안 맞기 때문). |

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

Windshield runtime projector(LUT)와 benchmark 스크립트는 6번 섹션의 기능
제거와 함께 삭제됐다. Camera Intrinsic 결과(OpenCV/ROS YAML)는 표준
`cv2.projectPoints`/`cv2.fisheye.projectPoints`로 그대로 쓰면 된다.

## 15. Real Vehicle Validation

차량 세션 비교 프레임워크(`calibration/vehicle_validation.py`)는 Reflection/
Ghost 지표에 의존해서 6번 섹션의 기능 제거와 함께 삭제됐다. 실차 rosbag으로
한 앞유리 영향 분석 결과는 6번 섹션 요약을 참고한다.

Desktop GUI 수동 실행과 Jetson 두 프로필은 "설치 스크립트/코드는 있으나 실제
기기에서의 end-to-end 실행 확인은 제한적"인 상태다 - 정확한 최신 상태는
README.md의 "결과를 믿기 전에" 절 및 최신 커밋/CI 기준으로 판단할 것.

## 16. Future Work

아래는 **현재 구현되어 있지 않다** - 계획/아이디어 수준이며, 위 섹션들의
"Implemented"와 절대 혼동하지 말 것.

- **Day/Night/Session 반복성(repeatability)의 실측 정량화**: 여러 세션
  데이터로 얻은 반복성 수치는 아직 없다.
- **Jetson 두 프로필의 실기기 현장 검증**: 설치 스크립트/preflight
  체크만 있고, 실제 Jetson 기기에서의 반복 실행 확인은 제한적이다.

이 목록에 있는 항목을 실제로 구현하게 되면, 그 즉시 해당 섹션의 상태
표시를 "Future Work"에서 "Implemented"/"Experimental"로 옮기고 이 목록에서
지운다 - 계획과 구현 완료 상태가 이 문서 안에서 섞이지 않게 하는 것이
원칙이다.

# Camera Calibrator

[![General CI](https://github.com/yaaaaangeo/camera_calibrator/actions/workflows/ci.yml/badge.svg)](https://github.com/yaaaaangeo/camera_calibrator/actions/workflows/ci.yml)

카메라로 찍은 사진 여러 장을 넣으면, 그 카메라가 사진을 얼마나 "찌그러뜨려서"
찍는지를 계산해주는 프로그램입니다. 카메라 캘리브레이션을 처음 해보는
사람도 큰 흐름을 따라갈 수 있도록 이 문서를 썼습니다.

이미 카메라 모델 이름(Pinhole, Fisheye 등)에 익숙하고 더 깊은 내용이
필요하다면 → [전문가/연구자용 상세 문서](docs/README_EXPERT.md)로 바로
가도 됩니다.

## 1. 이 프로그램은 뭐예요?

카메라는 사진을 찍을 때 실제 세상을 조금 찌그러뜨려서 보여줄 수 있습니다.
특히 화각이 넓은 렌즈일수록 직선이 사진 속에서는 살짝 휘어 보입니다.

**Camera Calibration**은 그 찌그러짐을 숫자로 찾아내는 과정입니다. 한 번
찾아두면, 나중에 그 숫자로 사진을 "펴서" 실제 세상과 더 가깝게 되돌릴 수
있습니다. 자율주행차, 로봇, 3D 스캐너처럼 "카메라가 본 것을 실제 거리/
좌표로 바꿔야 하는" 모든 곳에서 이 계산이 필요합니다.

이 프로그램이 도와주는 흐름은 이렇습니다.

```
사진 여러 장
 ↓
캘리브레이션 보드 찾기       (사진 속 체크무늬/원 무늬 위치 찾기)
 ↓
카메라 특성 계산            (찌그러짐을 숫자로 표현)
 ↓
여러 모델 비교               (어떤 계산 방식이 이 카메라에 맞는지)
 ↓
가장 믿을 만한 결과 선택
```

## 2. 어떤 기능이 있나요?

- **Camera Intrinsic Calibration** - 카메라 하나의 찌그러짐(초점거리,
  중심점, 왜곡)을 계산하는 핵심 기능.
- **여러 종류의 캘리브레이션 보드** - ChArUco(권장), 일반 체스보드, Circle
  Grid, AprilGrid 중 갖고 있는 보드에 맞춰 고를 수 있습니다.
- **여러 계산 모델 동시 비교** - Ideal Pinhole / Brown-Conrady / Rational /
  Fisheye, 네 가지 방식으로 동시에 계산해서 이 카메라에 가장 잘 맞는 걸
  골라줍니다.
- **Train / Hold-out 검증** - 계산에 쓰지 않고 남겨둔 사진으로 "진짜
  맞는지" 확인합니다(6번 섹션에서 더 설명).
- **Windshield Calibration** *(windshield 브랜치)* - 자동차 앞유리를 통해
  보는 카메라를 위한 추가 보정(7번 섹션).
- **Reflection / Ghost(유령상) 평가** *(windshield 브랜치)* - 앞유리에
  비치는 반사나 겹쳐 보이는 상을 평가/완화하는 기능.

각 기능의 자세한 원리와 파라미터는
[전문가 문서](docs/README_EXPERT.md)에 있습니다 - 여기서는 "이런 게
있다" 정도만 알면 됩니다.

## 3. 5분 만에 시작하기

**필요한 것**: Python 3.10 이상, 그리고 캘리브레이션 보드를 찍은 사진
여러 장(없다면 4번 섹션 참고).

```bash
# 1) 이 폴더로 이동
cd camera_calibrator

# 2) 가상환경 만들기 (권장)
python -m venv venv
source venv/bin/activate        # Windows는: venv\Scripts\activate

# 3) 필요한 패키지 설치
pip install -r requirements.txt

# 4) 실행
python -m app.main
```

창이 뜨면 5번 섹션을 따라가면 됩니다. 설치 중 막히면 8번 섹션을 확인하세요.

## 4. 어떤 캘리브레이션 보드를 쓰면 되나요?

**ChArUco를 가장 추천합니다.**

체스보드처럼 검은/흰 사각형이 있지만, 그 위에 작은 마커(ArUco 태그)가 같이
찍혀 있는 보드입니다. 이게 좋은 이유:

- 보드 일부가 사진 밖으로 잘려도 보이는 부분만으로 계산에 쓸 수 있습니다
  (일반 체스보드는 보드 전체가 안 보이면 아예 못 씁니다).
- 마커 덕분에 "어느 쪽이 위인지"가 항상 명확합니다 - 일반 체스보드는
  좌우/상하가 뒤집혀도 사진만 봐서는 구분이 안 되는 근본적인 한계가 있고,
  이게 섞이면 계산이 크게 틀어질 수 있습니다.

인쇄/보정판 준비가 어렵다면 Circle Grid나 AprilGrid도 지원하니 갖고 있는
보드에 맞춰 쓰면 됩니다 - 다만 처음이라면 ChArUco로 시작하는 걸
권장합니다.

## 5. 기본 사용 순서

1. **사진 넣기** - 캘리브레이션 보드를 여러 각도/거리/위치에서 찍은 사진을
   불러옵니다. 한 각도에서만 찍은 사진 10장보다, 다양한 각도로 찍은 사진
   10장이 훨씬 좋은 결과를 줍니다.
2. **보드 설정** - 어떤 보드(ChArUco 등)를 썼는지, 칸 크기가 몇 mm인지
   입력합니다.
3. **Detection 확인** - 프로그램이 사진마다 보드를 잘 찾았는지 확인합니다.
   못 찾은 사진은 이유와 함께 표시됩니다.
4. **Calibration 실행** - [캘리브레이션 실행] 버튼 하나로 검출 → 여러 모델
   계산 → 검증 → 추천까지 자동으로 진행됩니다.
5. **Hold-out 확인** - 결과 탭에서 Train/Test 숫자를 함께 확인합니다(6번
   섹션 참고).
6. **결과 저장** - 계산된 카메라 값을 OpenCV YAML 등으로 내보내거나, 나중에
   이어서 작업할 수 있도록 프로젝트 파일로 저장합니다.

## 6. 숫자는 뭘 보면 돼요?

결과 화면에 여러 숫자가 나오는데, 처음에는 이 정도만 봐도 충분합니다.

| 숫자 | 뜻 |
|---|---|
| **Train RMS** | 계산에 실제로 쓴 사진들에서, 계산된 카메라 값으로 얼마나 정확히 맞는지(픽셀 단위 오차). |
| **Hold-out(Test) RMS** | 계산에는 **쓰지 않고 일부러 남겨둔** 사진으로 확인한 오차. |
| **Median / P95** | 오차들을 크기 순으로 줄 세웠을 때 중간값(Median) / 상위 5% 지점(P95). RMS 하나만 보면 안 보이는 "가끔 많이 틀리는 사진"이 있는지 보여줍니다. |
| **Edge RMS** | 사진 가장자리 영역 코너 포인트들의 RMS 오차. 가장자리에서만 유독 크면 왜곡 보정이 그쪽에서 덜 된 것입니다. |

가장 중요한 건 **Train RMS만 보지 말고 Hold-out RMS도 같이 보는 것**입니다.

> Train RMS만 좋고 Hold-out RMS가 나쁘면, 시험 범위를 외워서 그 문제만 잘
> 푼 것과 비슷합니다 - 계산에 쓴 사진에서는 잘 맞는데, 처음 보는 사진에는
> 안 맞을 수 있다는 뜻입니다. 두 숫자가 비슷하게 낮아야 진짜로 믿을 만한
> 결과입니다.

더 자세한 지표(P90/P99/AIC/BIC/Stability 등)의 정의는
[전문가 문서 4번 섹션](docs/README_EXPERT.md#4-validation)에 있습니다.

## 7. Windshield 기능은 뭐예요?

*(`windshield` 브랜치에만 있는 추가 기능입니다.)*

일반 카메라 캘리브레이션은 "카메라 자체"의 찌그러짐만 계산합니다. 그런데
자동차에 달린 카메라는 보통 앞유리(windshield) **너머로** 세상을 봅니다 -
그 유리도 빛을 살짝 굴절시키기 때문에, 카메라만 보정해서는 설명 안 되는
오차가 남습니다.

```
Camera         카메라 자체의 찌그러짐 (1~6번 섹션에서 계산)
 ↓
Windshield     앞유리를 통과하며 생기는 추가 굴절/반사
 ↓
Road           최종적으로 도로/세상이 보이는 모습
```

이 기능은 이 굴절만 따로 추가로 계산합니다 - 카메라 자체의 계산 값(1~6번
섹션 결과)은 건드리지 않고 그 위에 얹는 방식입니다. 크게 세 가지를
구분해서 다룹니다.

- **Geometry(형상) 보정** - 유리 때문에 빛이 얼마나 휘는지 계산.
- **Reflection(반사)** - 유리에 비치는 하늘/조명 등을 평가.
- **Ghost(유령상)** - 밝은 빛이 유리에 이중으로 반사되어 살짝 어긋난
  채로 두 번 보이는 현상을 평가.

세 가지는 서로 다른 문제라 점수도 따로 나옵니다 - 반사가 있다고 해서
형상 보정 결과가 나빠 보이지 않도록 분리되어 있습니다. 원리와 각 모델의
차이는 [전문가 문서 6~8번 섹션](docs/README_EXPERT.md#6-windshield-geometry)
에 있습니다.

Spherical 모델에서 유리 굴절률을 공기와 같게 두면 "굴절이 없음"은
확인할 수 있지만, 그 상태로는 구의 중심/반지름을 식별할 수 없습니다. 또한
실시간 Camera-LiDAR 투영처럼 포인트가 아주 많을 때는 exact projector와
빠른 LUT projector가 분리되어 있으며, LUT 정확도는 실제 사용 범위에서
검증해야 합니다. CAD/STEP 기반 windshield surface prior는 현재 필수 기능이
아니며, 향후 optional 기능으로만 다룹니다.

## 8. 설치에서 막히면

| 증상 | 원인/해결 |
|---|---|
| `python -m app.main` 실행 시 PySide6 관련 에러 | GUI 라이브러리(PySide6)가 설치 안 됐거나 시스템에 필요한 그래픽 라이브러리가 없는 경우입니다. `pip install -r requirements.txt`를 다시 확인하세요. Linux(특히 서버/컨테이너)에서는 `libegl1`, `libgl1`이 추가로 필요할 수 있습니다. |
| ChArUco/AprilGrid를 못 찾는다는 에러(`cv2.aruco` 관련) | 일반 `opencv-python`이 아니라 `opencv-contrib-python-headless`가 설치되어야 합니다 - `requirements.txt`가 이미 이 패키지를 지정하지만, 다른 `opencv-*` 패키지를 이미 설치해뒀다면 충돌합니다. `pip uninstall`로 다른 `opencv-*`를 먼저 지우세요. |
| ROS 관련 기능이 필요함 | `.bag`/`.db3`/`.mcap` 파일에서 이미지만 뽑는 거라면 ROS 설치 없이 `pip install -e ".[ros]"`로 충분합니다. 실시간으로 ROS 토픽을 구독하려면 실제 ROS1/ROS2 환경이 컴퓨터에 따로 설치되어 있어야 합니다. |
| Jetson 보드에서 설치하고 싶음 | 일반 설치 방법 대신 전용 스크립트가 있습니다. [전문가 문서 13번 섹션](docs/README_EXPERT.md#13-jetson)과 [JETSON.md](JETSON.md)를 참고하세요. |

## 9. 전문가인가요?

카메라 모델의 수식, 각 파라미터의 의미, 소스 코드 구조, 검증 지표의
정확한 정의, CI 구성, Windshield 알고리즘의 세부 수학까지 필요하다면 →

**[전문가/연구자용 상세 문서 → docs/README_EXPERT.md](docs/README_EXPERT.md)**

CLI 옵션 전체 목록, Python API로 직접 호출하는 방법, 프로젝트 파일
구조(`.ccproj`), 테스트 실행법 등 개발/기여자를 위한 내용도 그 문서에
있습니다.

## 결과를 믿기 전에 (Current Validation Status)

이 프로그램이 "어디까지 실제로 검증됐는지"를 정직하게 밝힙니다 - 전부 통과했다고
과장하지 않습니다.

| 대상 | 상태 |
|---|---|
| 자동화 테스트(pytest, 매 커밋마다 실행) | 최신 상태는 상단 배지와 [Actions 탭](https://github.com/yaaaaangeo/camera_calibrator/actions)이 source of truth입니다 |
| Dedicated Windshield 워크플로우(Ghost/Reflection/Reflection Suppression/Neural) | 각 워크플로우별 최신 실행 결과는 [Actions 탭](https://github.com/yaaaaangeo/camera_calibrator/actions)에서 직접 확인 |
| General CI(Python 3.10/3.11, 핵심 회귀) | 이 문서에 고정된 PASS/FAIL 스냅샷을 두지 않습니다. 최신 결과는 상단 배지나 Actions 탭에서 확인하세요 |
| 실제 차량(Real Vehicle)에서의 Windshield 보정 검증 | **아직 검증되지 않음(NOT YET VALIDATED)** - 이 저장소에는 실차 캡처 데이터셋이 없습니다 |
| Jetson 실기기에서의 반복 실행 | 코드/패키징만 준비됨, 실기기 검증 아님. JetPack 5.1.2의 Python 3.10 venv는 ROS1 Noetic Python 3.8 `cv_bridge` 기반 live topic을 직접 지원하지 않습니다 |

이 표가 실제 최신 상태와 다르다고 느껴지면, 문구를 그대로 믿지 말고 위
Actions 링크에서 최신 실행 결과를 직접 확인하세요. 더 자세한 상태 구분은
[전문가 문서 15번 섹션](docs/README_EXPERT.md#15-real-vehicle-validation)에
있습니다.

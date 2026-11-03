# P5 독립 촬영 검증 절차

최종 Export K/D를 **고정**하고 새 촬영의 각 프레임 자세만 추정한다. 초기값 선택·K/D 재학습·추천 갱신·게시·차량 적용은 수행하지 않는다. 실제 새 촬영 검증은 아직 미완료이며, 이 경로의 동작은 합성 대조군으로 검증했다.

## 1. 동결된 입력

- 후보: `.verification/p5final/intrinsic/camera_fisheye.yaml` — 기존 P5 최종 Export, Full34/1255코너로 학습. 해당 결과의 기존 Hold-out은 별도의 Train27 K/D에서 평가됐다.
- 비교 기준: `.verification/p5-followup/frozen/rational_common34.yaml` — 같은34장/1255코너의 기본 Rational 결과. 원본 P5 Rational35 K/D를 덮어쓰지 않았다.
- 개발 데이터 출처: `.verification/p5final/project/calibration.ccproj` — 기존35장 전체를 중복 금지 대상으로 사용한다.
- 추가 분석 데이터90장도 중복 금지하려면 아래 `--training-manifest`를 사용한다. 명령에서 경로를 자동으로 찾는다.

보드 설정은 위 프로젝트에 저장된 ChArUco10×7, DICT_5X5_100, Square100mm, **Marker75mm 미확인 가정**을 사용한다. 실물 marker/square/평탄도를 먼저 확인해야 물리적 정확도를 해석할 수 있다. 사양을 정정하면 기존 결과와 동일 보드 조건이 아니므로 별도 검증 작업으로 분리한다.

## 2. 새 촬영 구성

기존 `rosbag2_2026_10_23-15_38_30`와 다른 녹화·시간·보드 배치를 사용한다. 인접 프레임만 늘리지 않는다. 기존 Train/Test의 scene cluster 공유가 있었으므로 새 촬영은 session 단위로 분리한다.

- 중간/먼 거리에서 여러 yaw/pitch(양 방향)를 확보하고, 근접에서도 충분한 2차원 코너 분포가 보이도록 한다.
- 부분 보드를 일괄 금지하지 않는다. 7코너가 한 줄과 한 점에 몰린 frame5 유형은 별도 스트레스 그룹으로 기록한다.
- 보닛 위 하단 경계의 **관측 가능한 영역**, 좌우 가장자리와 상단을 촬영한다. 보닛 영역 자체를 촬영 목표로 삼지 않는다.
- 적어도10장의 패턴 검출 가능한 새 프레임이 필요하다. 이는 기존 독립 benchmark 경로의 최소치이며,10장만으로 충분한 품질·독립성이 보장되는 것은 아니다.
- 카메라/렌즈/초점/해상도/차량 설치를 고정한다. 3840×2160 원본 해상도를 유지한다.

이미지 토픽 점검:

```powershell
venv\Scripts\python.exe -m app.cli --list-topics 'C:\path\to\NEW_ROSBAG'
```

PNG 추출은 기존 `BagReplay`를 사용한다. 아래 Python 예시는 **계산 없이** 원본 픽셀을 저장하며 새 출력 디렉터리만 허용한다. 예제를 파일로 저장하여 실행하고 bag 경로·출력 경로를 새 촬영에 맞춘다.

```python
from pathlib import Path
import cv2
from automation.live.sources import BagReplay

output = Path('.verification/independent-new/images')
if output.exists():
    raise ValueError('새 출력 경로를 사용하세요')
output.mkdir(parents=True)
source = BagReplay(r'C:\path\to\NEW_ROSBAG', '/video0/image_raw', extract_fps=1.0)
source.start()
try:
    index = 0
    while not source.exhausted:
        item = source.poll()
        if item is None:
            continue
        pixels, timestamp = item
        if not cv2.imwrite(str(output / f'new_{index:05d}.png'), pixels):
            raise OSError('PNG 저장 실패')
        index += 1
finally:
    source.stop()
```

실행 도중 중단되면 부분 추출 디렉터리는 보존한다. 완전한 새 평가 집합인지 확인한 후 평가한다. 원본 bag과 기존 저장 이미지를 변경하지 않는다.

## 3. 고정 K/D 평가 명령

저장소 루트 PowerShell에서 실행한다. 입력 이미지·capture ID·출력 경로는 새 촬영 값을 지정한다.

```powershell
$historyManifest = (Get-ChildItem .verification/rb1023 -Recurse -Filter p2_manifest.json | Select-Object -First 1).FullName
venv\Scripts\python.exe -m scripts.evaluate_frozen_intrinsics `
  --source-project .verification/p5final/project/calibration.ccproj `
  --reference .verification/p5-followup/frozen/rational_common34.yaml `
  --candidate .verification/p5final/intrinsic/camera_fisheye.yaml `
  --images .verification/independent-new/images `
  --capture-id NEW_CAPTURE_UNIQUE_ID `
  --training-manifest $historyManifest `
  --output .verification/independent-new/evaluation
```

새 결과 파일: `independent_result.json`. Reference/Candidate K/D, 모델별 고정 파라미터 평가, 프레임/영역별 잔차, bootstrap 비교와 평가 입력 해시를 기록한다. 모든 결과의 `approval`은 `NOT_APPROVED`다.

검사 사항:

- source project·Reference·Candidate 파일 SHA256을 실행 전후 고정한다.
- 원본35장 및 추가 지정90장과 **디코딩된 픽셀 해시**가 같으면 거부한다. PNG→BMP처럼 저장 형식만 바꿔도 중복으로 판정한다.
- 해상도 불일치, 평가 집합 내 동일 픽셀 중복, 기존 capture ID, 새 이미지10장 미만 또는 검출10장 미만, 기존 출력 경로를 거부한다.
- 독립 benchmark 실패 시 기존 내부 Hold-out으로 돌아가서 결과를 대신 내지 않는다.
- 기존 `compare_reference_candidate_calibrations`의 고정 K/D 평가만 사용한다. 테스트에서 OpenCV intrinsic calibration 함수를 금지한 상태로 이 경로를 검증했다.

해시 분리는 **동일 픽셀 재사용 방지**다. 손실 압축·변형된 복사본이나 인접 장면의 독립성까지 증명하지는 못한다. 별도 녹화 여부·시간·장면/보드 배치는 사람이 검토하고 기록해야 한다. 카메라 하드웨어 serial 및 신규 촬영 메타데이터 확인도 남아 있다.

새 Test를 본 뒤 초기값·모델·촬영 선별 전략을 바꾸면 해당 집합은 더 이상 최종 독립 Test가 아니다. 개선 전략은 개발/Validation 집합에서 고정하고, 변경 후 평가는 또 다른 새 Test로 수행한다. 수치가 좋아도 자동 승인·차량 적용·외부 게시를 수행하지 않는다.

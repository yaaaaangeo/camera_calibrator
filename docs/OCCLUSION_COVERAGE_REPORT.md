# 카메라별 Occlusion Mask Coverage

## 구현과 평가 범위

기존 Full-frame `compute_coverage_grid`/`coverage_percentage`는 변경하지 않았다. `calibration/occlusion.py`가 관측 가능 영역의 공간 Coverage를 별도로 산출하며, 마스크가 설정된 P2 Readiness만 이 값을 사용한다. 마스크가 없으면 기존 Coverage·모서리·안내·프로파일 fingerprint 동작을 유지한다.

마스크는 촬영 준비도의 공간 평가용이다. 이미지를 자르거나 수정하지 않고, 패턴 검출·학습 코너·K/D·RMS·Hold-out·Observability·추천 가중치·결과 품질 임계값에는 전달하지 않는다. 마스크 내부 또는 외부의 캘리브레이션 정확도가 검증됐다는 의미가 아니다. Readiness=true는 계산 시작을 위한 데이터 준비도이며, 결과 PASS나 차량 적용 승인이 아니다.

## 설정

P2 YAML 최상위의 선택 필드 `coverage_mask`, 통합 experiment YAML의 `live.coverage_mask`에 같은 객체를 **인라인**으로 넣는다. 외부 파일 경로를 지정하는 필드는 추가하지 않았다. 실제 자료 설정은 [masks/rosbag1023-video0.yaml](masks/rosbag1023-video0.yaml)에 있다. 이 YAML 객체를 해당 위치에 복사하면 된다.

필수 필드:

|필드|의미|
|---|---|
|`schema_version: 1`|설정 형식 버전|
|`version`|카메라 장착 상태/마스크 개정 식별자|
|`source`|경계 판단 방법과 근거 자료|
|`camera: {model, serial}`|실행의 카메라 식별자와 정확히 일치해야 하는 바인딩|
|`width`, `height`|원본 이미지 해상도|
|`coordinate_system: native_distorted_pixel_edges`|원본 왜곡 영상의 픽셀 경계 좌표; 좌상단 (0,0), 오른쪽/아래 방향 +, 우하단 (width,height)|
|`polygons`|제외 영역의 다각형 목록. 사각형 ROI도 네 꼭짓점 Polygon으로 표현|

자체 교차·퇴화·화면 밖·NaN·전체 화면 제외·잘못된 좌표계는 거부한다. 각 Polygon은 convex여야 하며, 비볼록 모양은 여러 Polygon의 합집합으로 작성한다. 겹친 부분은 한 번만 제외된다. 암묵적인 해상도 변경·리사이즈·크롭·정규화 좌표 변환은 하지 않는다. 카메라 불일치는 실행 전 거부하고, 실제 해상도 불일치는 Preflight와 P2 캡처에서 저장 전에 차단한다.

`UNKNOWN / DIAG-video0`는 현재 녹화에만 사용하는 진단 식별자이며 실제 하드웨어 모델/SN이 아니다. 다른 카메라에 이 예시를 적용하면 안 된다. 실제 모델/SN과 장착 상태가 확인되면 프로파일과 마스크 바인딩을 함께 개정해야 한다.

## Grid 경계 처리와 수식

원본과 같은 4×4 Grid를 유지한다. Polygon 합집합을 원본 해상도에서 픽셀 중심으로 평가하여 관측 가능 픽셀을 구한다. 검출 코너는 해당 원본 픽셀의 마스크 값을 조회하며 Coverage 진단에서만 가려진 코너를 제외한다. 정밀도는 픽셀 raster 수준이다.

- 셀 i의 관측 가능 픽셀 수: A_i
- 전체 관측 가능 픽셀 수: A = ΣA_i
- 관측 가능 영역의 검출 코너 수: N, 해당 셀 코너 수 n_i
- 기대 코너 수: E_i = N × A_i / A
- 셀 점수: min(1, n_i/E_i), E_i=0이면 0
- 충분한 셀 판정: 기존 `coverage_percentage`의 기본 기준 0.3을 그대로 사용
- Observable Coverage = 100 × Σ(충분한 셀의 A_i) / A

**경계 교차 셀은 버리지 않는다.** 남은 관측 가능 면적에 비례해 기대 코너 수와 Coverage 가중치를 계산한다. 50% 등 임의 면적 cutoff는 없다. A_i=0인 셀만 분모와 촬영 부족 안내에서 제외한다. 관측 가능 영역에 코너가 전혀 없으면 Coverage=0이다.

마스크가 있을 때 필수 모서리 셀은 원래 네 Grid 모서리 위치에서 가장 가까운, 관측 가능 면적이 남은 셀로 대응시킨다(격자 좌표 제곱 거리; 동률이면 행/열 오름차순). 대응 셀의 관측 가능 부분에 검출 코너가 있어야 한다. 이번 자료의 필수 셀은 1기준 `(1,1), (1,4), (3,1), (3,4)`다. 하단을 통째로 무시하는 대신 보닛 위 경계 쪽 셀을 검사한다. 이 규칙은 복잡한 마스크의 모든 경계 점이 관측됐음을 보장하지 않는다.

거리·Tilt·중심 위치·기존 Edge 다양성 및 Hold-out 조건은 종전 그대로다. 마스크가 이러한 조건의 실패를 무시하거나 임계값을 낮추지는 않는다. Full-frame 기반 다른 품질 진단은 그대로 남는다.

## 실제 영상의 경계 설정 근거

`rosbag2_2026_10_23-15_38_30`의 `/video0/image_raw`, 3840×2160. 기존 시간 샘플 0/30/60/90/113초를 시각 확인했다. 고정 보닛/대시보드 실루엣을 보수적으로 수동 기록했으며, 움직이는 보드·사람은 영구 마스크로 추가하지 않았다. 명확한 차체 영역 쪽에 경계를 두었으므로 가장자리의 불확실한 띠를 과도하게 제외하지 않는다. 자동 segmentation이나 차량 표면 측정 결과는 아니다.

경계 x: `0,480,960,1440,1920,2400,2880,3360,3840`

경계 y: `1580,1560,1550,1555,1555,1555,1550,1550,1570`

인접 두 점과 이미지 아래쪽 경계로 8개 사다리꼴 Polygon을 작성했다. Grid 행 번호나 준비도 점수에서 좌표를 역산하지 않았다. 관측 가능 면적은 72.04861%, 제외 면적은 27.95139%다. 하단 행을 제외하는 효과 외에도 3행과 실제 경계가 교차하므로 면적 가중 평가가 적용된다.

근거 이미지 SHA-256·경계·설정 digest는 `.verification/occlusion/boundary-evidence.json`, 검토용 시각화는 `.verification/occlusion/boundary-qa.svg`와 `boundary-qa.png`에 기록했다. 시각화도 확인했다. 원본은 수정하지 않았다. 다른 장착 각도·크롭·해상도·차량에 대한 범용 마스크가 아니다.

## 실제 데이터 비교

기존 저장 90장과 검출 결과를 재사용했고 bag 재추출·중복 캘리브레이션은 하지 않았다. 선정은 앞선 Coverage 보존 개선 후의 동일한 35장을 사용했다.

|자료/설정|Full-frame Coverage|Observable Coverage|데이터 Readiness|
|---|---:|---:|---|
|전체 저장 90장, 마스크 공간 진단|75.00%|84.76%|전체 90장에 별도 Runner Readiness는 수행하지 않음|
|선정 35장, 마스크 없음|75.00%|미적용|false: 하단 모서리 미관측|
|동일 선정 35장, 마스크 있음|75.00%|92.37%|true: 공간/기타 데이터 조건 충족|

Observable Coverage의 분모는 관측 가능 면적이며 코너 분포는 면적별 fair-share로 정규화된다. 이미지가 많다고 값이 단조 증가하는 지표는 아니다. Full-frame 75%와 Observable 92.37%는 같은 분모의 백분율이 아니다.

마스크 후에도 3행 4열은 부족 셀이다. 관측 가능 면적 비율 87.963%, 코너 27개, 점수 0.25548로 기존 셀 기준 0.3보다 낮아 Observable Coverage 분자에서 제외된다. 셀을 임의로 통과시키지 않았다. 그래도 전체 Observable Coverage가 기존 Readiness 기준 75%를 초과하고 경계 모서리 셀에는 코너가 있어 데이터 준비도를 충족했다. 하단 완전 차폐 셀은 추가 촬영 대상으로 안내하지 않는다. 실패 시에는 관측 가능한 부족 셀과 선정 손실을 구분해 행/열로 안내한다.

선정 이미지 ID와 Train/Test 분할이 마스크 전후 동일함을 assertion으로 확인했다. 거리·Tilt·Edge 다양성, 중복 비율, Hold-out 값도 동일하다. Hold-out은 7장/309코너, Coverage 50%, 자세 다양성 0.70833, 구성 sufficient로 유지됐다. 저장 PNG 90장의 SHA-256을 기존 Manifest와 검증 전후 대조하여 모두 일치했다.

### 계산 결과와 승인 범위

이번 변경에서 실제 자료의 재계산, 최종 모델 추천, YAML 재발행, Git 게시를 수행하지 않았다. 동일한 선정·분할이며 마스크는 계산 함수에 전달되지 않으므로 기존 진단 K/D·RMS를 보존했다. 앞선 결과는 [ROSBAG_COVERAGE_PRESERVATION_REPORT.md](ROSBAG_COVERAGE_PRESERVATION_REPORT.md)에 있다. 마스크 적용 후 **전체 자동 실행의 Observability 및 최종 Acceptance는 이번 실제 녹화로 미검증**이다. Readiness 데이터 조건 true를 그 결과들까지 PASS로 확장하지 않는다.

기존 합성 ROSbag E2E 및 P2→P1/동일 입력 직접 CLI 비교 회귀는 통과했다. 실제 보드 Marker 75mm는 아직 미확인 가정이다(Square 100mm, 10×7, DICT_5X5_100은 사용자 확인). 따라서 마스크가 적용된 관측 가능 영역에서도 정확도 검증 완료를 주장하지 않는다. 실제 ROS/카메라 연결, 차량 적용, GitHub 인증/게시도 미검증이다.

## Manifest와 재시작

P2 `p2_manifest.json`에 해상도·좌표계·카메라 바인딩·Polygon·설정 출처·버전·SHA-256·평가 규칙 버전·`accuracy_validated:false`를 기록한다. Readiness에는 Full-frame과 Observable 값 및 셀별 면적·코너·점수·필수 모서리를 함께 기록한다. Experiment Manifest도 마스크 metadata를 저장한다. GUI 상태 표시에는 두 Coverage와 정확도 검증이 아님을 표시한다.

마스크는 LiveProfile fingerprint에 포함된다. 마스크/출처/버전을 바꾸고 기존 세션을 조용히 재개할 수 없다. 새 job/session을 사용해야 하며, 이전 실패 Manifest를 성공으로 덮어쓰지 않는다. 기존 P1 계산 Manifest와 결과는 수정하지 않았다.

## 실행 명령

현재 머신의 완성된 진단 프로파일: `.verification/occlusion/profile.yaml`. 외부 Publisher는 disabled, 원격 미설정, Factory 미설정이다. 확인된 보드 설정과 현재 녹화 경로를 포함한다.

```powershell
venv\Scripts\python.exe -m automation experiment preflight --profile .verification/occlusion/profile.yaml
venv\Scripts\python.exe -m automation experiment run --profile .verification/occlusion/profile.yaml --dry-run
# 실제 새 세션으로 자동 계산을 시작할 때:
venv\Scripts\python.exe -m automation experiment run --profile .verification/occlusion/profile.yaml
```

실제 자료 dry-run: 마스크 검사 PASS, 전체 WARN(물리 카메라 식별 미확인/Publisher 비활성/Factory 없음 등 기존 경고), `calibration_will_run=false`. 출력 세션 폴더 생성 없음. 이 작업에서는 마지막 실제 실행 명령은 수행하지 않았다.

다른 환경에서는 `docs/experiment.example.yaml`을 복사하고 실제 카메라/녹화/보드/출력을 설정한 후 `live.coverage_mask` 객체를 넣는다. 마스크 바인딩과 카메라 설정을 일치시키되, 실제 영상을 확인하지 않고 이 차량의 좌표를 다른 장착 상태에 재사용하지 않는다.

## 테스트와 변경 파일

고유 테스트 기준 **361 PASS / 0 FAIL / 1 SKIP / 0 XFAIL**. 신규 마스크 회귀 24개 모두 PASS. 기존 실제 과노출 이미지 fixture가 없는 1개 테스트는 SKIP이다. 중간 unit/regression/new-final JUnit의 중복 사례는 합산하지 않았다. 전수 전체 저장소 테스트 결과는 아니다.

범위: P1 프로파일/Acceptance/Runner, P2 Live/Readiness/복구/GUI/게이트/합성 E2E, P3 번들/CLI, P4 Factory/GUI, ROS1/ROS2/MCAP 통합·해상도·Dry-run·게시 실패 복구. 마스크 전용 회귀는 완전 차폐/부분 셀/픽셀 경계/Union/잘못된 설정/카메라·해상도 불일치/Manifest/변경 마스크 재개 거부/마스크 없는 동작/기존 Hold-out 실패 유지/GUI 두 값 표시를 검증했다.

최종 증거: `.verification/occlusion/test-ledger.json`, `unit.xml`, `regression.xml`, `new-final.xml`, 각 실행 로그, `real-result.json`, `dry-run.log`. 실제 데이터 검증 스크립트 결과는 pytest 개수에 합산하지 않았다. 이미 완료된 계산 결과는 재사용했다.

변경 파일:

- `calibration/occlusion.py`: 마스크 검증·영역 산출·Observable Coverage·metadata
- `automation/live/config.py`: 선택 마스크 설정 및 fingerprint
- `automation/live/readiness.py`: 별도 공간 평가·경계 모서리·차폐 영역 제외 안내
- `automation/live/controller.py`: 카메라/해상도 바인딩 및 P2 Manifest
- `automation/experiment/preflight.py`, `runner.py`: 마스크 사전 점검·Experiment Manifest
- `ui/automatic_calibration_dialog.py`: Full-frame/Observable 구분 표시
- `tests/test_occlusion_coverage.py`, `docs/masks/rosbag1023-video0.yaml`, 본 보고서

기존 미커밋 변경을 유지했다. 코드 저장소 HEAD는 `acb22f147b277a600649e9ece84729c66dee1fb1`로 동일하다. 코드 저장소 Commit·Push, 외부 Push, 배포는 하지 않았다.

**실행 제한 위반 기록:** 최초 기존 회귀 묶음에 포함한 `tests/test_experiment.py`의 기존 게시 테스트가 `.verification/omunit/` 내부 임시 로컬 bare 저장소에 테스트 Commit·Push를 수행했다. 요청의 Commit·Push 금지 범위를 충분히 지키지 못한 실수다. 외부/코드 저장소가 아닌 격리된 테스트 저장소이며 증거는 보존했다. 확인 후 후속 검증에서는 이 묶음을 제외했다. 이 항목을 'Commit·Push 전혀 없음'으로 보고하지 않는다.

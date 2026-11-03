# P5 후속 검증 — 동일 조건 비교·품질 원인·독립 촬영 준비

2026-11-03. 분석과 명확한 안내 버그의 최소 수정만 수행했다. 원본125 이미지의 SHA256, 기존 결과 및 미커밋 변경을 보존했다. 코드 저장소 HEAD는 `acb22f147b277a600649e9ece84729c66dee1fb1`로 유지했다. 차량 적용/승인, 외부 Push, 코드 저장소 Commit/Push, 배포를 수행하지 않았다. Git 회귀 테스트는 임시 **결과 전용 로컬 Bare 저장소**에서 수행했다.

## 결론과 이전 요약 정정

1. 부분 보드 frame5는 검출 성공/7코너이지만 네이티브 Extrinsic 초기화가 불안정하다. 같은 P5 K seed에서 `CALIB_CHECK_COND ... input array 5` 오류 후 해당1장만 제외하면 성공하는 것을 재현했다. 원본 데이터의 일반적인 불량 판정은 아니다.
2. 공통34장/1255코너, 동일 Train27/Test7로 비교하면 **Rational Full-fit1.532873px, Test1.611704px**, **Fisheye Full-fit1.538666px, Test1.613605px**다. Fisheye가 압도적으로 우수하다는 결론은 지지되지 않는다.
3. 원래 Rational35의 K/D·자세를 고정하고 frame5만 빼면 RMS는10.515849→**10.507178px**에 불과하다. 공통34장을 새로 최적화했을 때1.532873px로 내려간 것은 단순한7개 큰 잔차 제거 효과가 아니라 **최적화 초기화·수렴 경로 변화**와 관련된다.
4. 이전 P5 요약의 **62.8/LOW는 모델 선택 Confidence**다. 저장된 실제 **캘리브레이션 최종 Confidence는10.4/REJECT**다. 최종 grade WARN과 Confidence를 혼동하지 않도록 정정했다. 기존 수치·판정은 변경하지 않았다.
5. 보닛을 촬영 부족으로 안내하던 후단 Export/P2 Task 경로를 수정했다. **안내만** Observable 영역 기준으로 바꾸었고, 기존 Full-frame 품질 점수·진단 severity·추천 가중치·Confidence는 유지했다. 실제 재Export에서 K/D·RMS·모든 추천 점수·grade WARN·Confidence10.4/REJECT가 동일했다.
6. 고정 Export K/D를 새 데이터에서 평가하는 경로를 합성 데이터로 검증했다. **실제 새 독립 촬영은 아직 없으므로 실데이터 독립 정확도는 미검증**이다.

## 1. 제외 프레임의 실제 상태와 근거

- ID: `capture_000005_e8c1fca679b94372b393b8913ffcbff8`
- ChArUco 검출 성공, corner IDs **39,48,49,50,51,52,53**, finite 좌표, 원본 상태 DETECTED, corner exclusion 없음.
- 영상 확인: 보드가 매우 가깝고 상단·좌측이 화면 밖으로 잘렸다. 검출점6개는 한 행에 몰리고1개만 다른 행에 있다. 원본은 유지했고 시각화만 별도 생성했다: `.verification/p5-followup/excluded-frame.png`.
- 검출점 span1964.27×366.50px. object XY singular values0.480249/0.080645, image XY1846.03/288.24로 rank2다. **정확한 collinear는 아니지만 2차원 관측 제약이 약한 부분 보드**다.
- quality score: blur21.2, exposure66.8, corner13.0, geometry0.0, overall27.4/WARN. 이 점수들이 직접 native 제외 사유는 아니다. 검출 hull area ratio0.04337은 보드 전체의 실제 영상 점유율이 아니다.
- 기준 K/D로 추정한 optical-axis z≈0.158m, normal tilt≈15.22°. 절대 거리는 보드 치수 가정·모델에 의존하며 실측이 아니다. 2D 회전값−89.58°는 yaw/pitch tilt가 아니다.

실제 P5 Full-fit의 **같은 Train-only seed K**로 다시 실행한 trace:

1.35장, flags58720257, criteria(EPS+MAX_ITER,100,1e−6).
2.OpenCV5.0.0 `fisheye.cpp:806`, `CalibrateExtrinsics`: **Ill-conditioned matrix for input array5**.
3.그 인덱스가 정확히 frame5임을 확인했다.34장 재시도에서 **RMS1.5386657428960624**로 원본 P5와 정확히 일치했다.

즉 이번 제외는 이진 탐색의 임의 fallback이 아니라 **네이티브 indexed condition-check 오류에 근거**했다. 원본 Dataset 상태나 다른 모델의 프레임을 자동 삭제하지 않았다. 원본픽셀/코너/ID 및 native 호출 목록은 `excluded-frame.json`, `exclusion-trace.json`에 보관했다.

제외 프레임을 공통34장 K/D에 나중에 고정 적용해 자세만 추정한 진단에서는 Rational cold1.313px, Rational warm2565.019px, Fisheye569.327px였다. 이 프레임은 각34장 fit에서 빠졌지만 **이미 분석에 사용한 개발 프레임**이므로 독립 Test가 아니다. `solvePnP=True`라도 좋은 자세라는 보장은 없고, 작은 K/D 차이가 부분 보드의 잘못된 자세 해로 이어질 수 있다. 해당 진단으로 후보를 다시 선택하거나 임계값을 조정하지 않았다.

## 2. 공정 비교 설계

계산 전에 `.verification/p5-followup/frozen-plan.json`을 고정했다.

- Full 공통34장,1255코너.
- 기존 Outer Train28에서 frame5만 제외한 **공통 Train27장,952코너**.
- 원래 Outer Test7장,303코너는 그대로 유지했다. 삭제 후 `split_train_test`를 다시 호출하지 않았다.
- 모든 후보의 full/train frame IDs, object points·corner bytes SHA256, 실제 유지 프레임을 일치 확인했다. **추가 프레임 제외 없음**.
- Fisheye는 P5에서 이미 정한 `same_model_subset` 전략을 고정하여 공통 Train27에서 seed를 만들고, 그 K로 Full34를 재학습했다. 다른 모델의 D를 복사하지 않았다.
- Rational은 기존 cold 초기화를 기준으로 사용했다. Train27 K/D warm-start는 사전에 선언한 **진단 후보**로만 비교했고 기본 경로·추천·Export를 교체하지 않았다.
- Test7의 오차는 평가 후 기록만 했다. 초기값/전략/가중치 선택에 사용하지 않았다.

Full-fit RMS는 전체 학습 코너 가중 RMS, Hold-out Test RMS는 별도 Train-only K/D의 `sqrt(mean(frame_RMS²))`다. 서로 다른 정의나 K/D를 섞지 않았다.

| 후보 | Full34 RMS px | 별도 Train27 RMS px | Test7 RMS px | Test Edge RMS px | Test P95 px |
|---|---:|---:|---:|---:|---:|
| rational_cold | 1.532873 | 1.508076 | 1.611704 | 1.490901 | 3.441066 |
| rational_train_warm_diagnostic | 1.530513 | 1.508076 | 1.611704 | 1.490901 | 3.441066 |
| fisheye_locked_p5 | 1.538666 | 1.516407 | 1.613605 | 1.413608 | 3.582528 |

기존 P5의 nominal Train28은 Fisheye 내부에서 frame5를 제외한 실제27장으로 학습됐다. 이번 공통 비교는 처음부터 Train27을 입력했다. 고정된 subset 초기화 전략의 입력 순서가 달라져 기존 P5 Test1.6136043045px와 이번1.6136047483px에 약4.44e−7px 차이가 있다. 기존 Export/Train K/D는 교체하지 않았으며 두 출처를 Manifest에 별도 기록했다.

Test RMS 차이는0.001901px(약0.12%)다. Fisheye는 Edge 오차가 더 낮지만 P95는 더 높다. 근소한 학습 RMS나 하나의 지표만으로 모델 우열·정답을 단정할 수 없다. warm Rational은 Full RMS가0.00236px 낮아도 앞서 frame5 자세 추정에서 오히려 악화했다.

| 후보 | Full fx | Full fy | Full cx | Full cy | Rank | Condition | 최대 상관 | AIC | BIC |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| rational_cold | 1933.687869 | 1940.797678 | 1928.438886 | 1097.544968 | 12/12 | 442223.720 | 0.999997260 | 1096.130 | 1157.749 |
| rational_train_warm_diagnostic | 1936.300515 | 1943.290515 | 1928.567477 | 1097.070299 | 12/12 | 1831715.578 | 0.999998719 | 1092.264 | 1153.882 |
| fisheye_locked_p5 | 1936.611126 | 1944.674070 | 1921.084162 | 1098.078408 | 8/8 | 1436.474 | 0.992195560 | 1097.598 | 1138.677 |

AIC/BIC는 기존 함수와 파라미터 수(Rational12, Fisheye8)를 사용했다. 같은1255관측에서 Rational은AIC가 약간 낮고 Fisheye는BIC가 낮다. 서로 다른 자유도·condition 숫자만으로 정답을 고르지 않는다. 세 후보 모두 기존 Observability Score0/POOR다. full/Train의 전체 K/D, per-frame residual, point 통계, seed·flags·시간은 후보별 JSON/ccproj와 `FOLLOWUP_VERIFICATION_MANIFEST.json`에 분리했다.

이 비교는 기존 P5의4모델 추천 순위를 자동 갱신하는 작업이 아니다.4모델 min-max 정규화 점수를2모델만으로 다시 계산하면 의미가 달라지므로, 기존 추천과 수치를 보존하고 동일 관측의 비교 근거를 제공했다. 기존 전체 모델 추천 점수0.141634(Fisheye),0.270696(Rational)은 기존 서로 다른 full-fit 집합/수렴 결과를 포함한다는 제한도 명시한다.

## 3.34장 사용 전후 분포와 Observability

| 항목 |35장 | frame5 제외34장 |
|---|---:|---:|
| 코너 |1262|1255|
| Full-frame Coverage |68.75%|68.75%|
| Observable Coverage |77.028112%|77.028112%|
| 위치 다양성 |0.6875|0.6875|
| 기존 거리 다양성 |1.0|1.0|
| 기존 회전 다양성 |1.0|1.0|
| Edge coverage 다양성 |0.666667|0.666667|
| 기준 K/D의 z median |1.067371m|1.086883m|
| 기준 K/D의 z P10/P90 |0.326795/1.445691m|0.432218/1.449127m|
| 기준 K/D의 normal tilt median |23.8048°|26.3462°|
| 기준 K/D의 normal tilt P10/P90 |8.1288/47.6482°|8.0370/48.1564°|

거리/회전 요약점수1.0은 기존 정의의 포화이며 좋은3D 분포나 높은 식별가능성을 보장하지 않는다. 위 z/tilt는 **동일한 기준 Train28 Rational K/D**와 저장된 최적화 Train 자세/고정-K Test 자세를 사용한 진단 추정이다. 새 독립 성능이나 물리적 실측이 아니다. translation norm은 보드 원점까지의 거리여서 보드 중심 거리와 구분했다.

Observability를 통제해서 비교하기 위해 위 같은 K/D·각 프레임 자세를 고정하고 frame5만 제외했다. Rank12/12 유지, Condition154079.057→154075.493, 최대상관0.999807082→0.999806400, Score0→0이었다. 이 작은 condition 변화가 식별가능성 개선을 뜻하지 않는다.

**Fisheye35장의 native 성공 해는 없으므로, 성공한35장 해와34장 해의 Observability 비교는 불가능**하다. 기존 P5 Fisheye report도 dataset 객체는35장이지만 `per_frame_error` 유지 ID로 필터하여 실제34장/1255코너에서 계산된 값이다. 이를35장 관측가능성으로 잘못 표시하지 않았다.34장 Fisheye는 Rank8/8, Condition1436.474, 최대상관0.992196(k3/k4), Score0/POOR다.

공통34장 Rational의 Condition442223.720와 강한 분모/분자 계수 상관도 남아 있다. 모든 값은 **고정 자세의 local intrinsic Jacobian** 기준이며 per-frame Extrinsic coupling을 주변화한 전체 식별가능성 평가가 아니다. 모델별 차원도 다르므로 서로의 condition을 그대로 품질 승패로 해석하지 않는다.

## 4. WARN·LOW·REJECT의 정확한 원인

세 상태를 분리한다.

- **Overall grade WARN:** 기존 `_grade_for_value`에서 Full1.5387, Test1.6136, Edge1.4136px가1≤값<2px 구간에 있다. Straightness0.40033px는 더 좋은 구간이다. P95는 이 overall grade의 직접 입력이 아니라 Confidence 입력이다.
- **모델 선택 Confidence62.8/LOW:** 원래 Fisheye/Rational Test RMS1.6136/1.5920px가 기존3% 유사 지표 규칙에 걸린다. score gap≈0.129여도 `close metrics: Test RMS` 때문에LOW로 표시한다. 절대 정확도 신뢰도가 아니다.
- **최종 Calibration Confidence10.4/REJECT:** 기존 구성점수와 진단 감점을 합친 값이다. 이전 보고서에 정정 주석을 추가했다.

| Confidence 구성점수 | 값 /100 | 해석 |
|---|---:|---|
| Full-fit RMS |26.4|기존 우수/불량 기준0.25/2.0px 대비 높은 잔차|
| Hold-out Test RMS |22.7|기존0.30/2.0px 기준|
| Test P95 |0.0|3.5825px가 기존 poor3.0px 이상|
| Edge RMS |51.7|관측 가능한 가장자리 정밀도 개선 필요|
| Straightness |84.6|상대적으로 양호|
| Full-frame Coverage |68.8|기존 품질 평가는 전체 화면 기준|
| Parameter stability |69.0|고차 왜곡 계수 불확실성|
| Observability |0.0|최대상관0.9922와 식별가능성 부족|
| Undistortion |99.9|매핑 유효성 점수이며 실제 정확도 보증 아님|
| Model selection |62.8|상대 모델 선택 근거|

기존 가중 합산은 약40.4점이며, warning5개(Full-frame coverage, coverage 위치, edge residual, parameter stability, high correlation)가 각6점/최대30점 감점을 만들어 최종10.4점이 된다. 이 기준·가중치·감점은 변경하지 않았다.

Bootstrap20/20 성공은 샘플 계산 성공이지 정밀도 보증이 아니다. Fisheye fx/fy 상대 변동은 약1.49%/1.35%지만 k2/k3/k4 stability는27.48/25.13/25.16% 수준이다. 고차 계수가 강하게 결합하며 K가 비교적 안정적이어도 D가 식별되지 않을 수 있다.

개선 가능한 항목: 관측 가능한 edge의 선명한2차원 코너 분포, yaw/pitch·거리의 독립적인 변화, 보드 평탄도·정확한 치수 확인, 부분 보드의 자세 추정 스트레스 평가. 구조적인 보닛 occlusion은 추가 촬영으로 해결할 수 없다. Full-frame quality coverage는 기존 정의를 유지했으므로 이번 안내 수정만으로 해당 감점이나Confidence를 올리지 않았다.

## 5. 확인한 버그와 최소 수정

P2 Readiness는 Mask를 알지만 P1 최종 Export와 계산 후 P2 Task는 `diagnose_calibration`에 full grid만 전달하여 보닛 하단을 반복 촬영하라고 안내했다. 실제 저장된 원본 P5 JSON에서도 `capture_bottom`, `capture_bottom_left`, `capture_bottom_right`를 확인했다.

수정:

- P2 Task가 동일한 검증된 Mask를 P1의 `capture_guidance_mask`로 전달한다.
- CLI의 `--capture-guidance-mask PATH`는 **촬영 안내 전용**이다. 원본 픽셀, full/observable readiness 계산, K/D 계산, 품질 점수·threshold·추천을 바꾸지 않는다.
- 선택 모델의 audit에 canonical Mask를 보관하여 project 재Export에서도 같은 안내를 유지한다.
- fully occluded 셀은 촬영 대상에서 빼고, 부분 교차 셀은 기존 Observable 평가의 visible area·sufficient 상태로 안내한다. 안내는 unmasked 영역으로 한정한다.
- 기존 full-frame 진단 code/severity/evidence는 유지한다. 해당 warning 설명에는 구조적 occlusion을 촬영으로 해결하지 말라는 문구를 추가한다. 이로써 Confidence 계산이 몰래 좋아지지 않는다.

실제35장 재Export 결과: 안내는 **1-based grid(3,1),(3,2),(3,3)**의 보닛 위 가시 영역으로 바뀌었다. 완전히 가려진4행 셀은 안내하지 않는다. 원래Mask 버전·해상도·native 좌표계·경계를 그대로 사용했다. `guided-export/`에 별도 저장했고 기존 P5 Export는 덮어쓰지 않았다.

## 6. 독립 고정 K/D 검증 준비

실행 절차: `docs/P5_INDEPENDENT_VALIDATION.md`.

새 `scripts/evaluate_frozen_intrinsics.py`는 기존 고정-K benchmark API의 얇은 검증 경로다. Reference는 공통34장 Rational, Candidate는 **원래 P5 최종 Fisheye Export K/D**를 사용한다. 별도 Train-only K/D로 바꾸거나 재학습하지 않는다.

- 기존35장/선택적으로 분석90장과 검출 입력과 같은 canonical BGR 픽셀 해시 중복을 거부한다. 저장 형식이나 alpha 채널만 바꾼 복사도 거부한다.
- 새 capture ID, 원본 해상도, 새 이미지/검출 각10장 이상, 새로운 출력 경로를 요구한다.
- source project/Reference/Candidate 파일을 SHA256으로 고정한다.
- 기존 internal Hold-out으로 자동 fallback하지 않는다. 실제 intrinsic calibration 함수를 금지한 합성 테스트에서 평가가 완료됨을 확인했다.
- 최종 JSON의 `approval=NOT_APPROVED`, `intrinsics_refitted=false`를 기록한다. 해시 분리는 시간/장면 독립성을 증명하지 않으므로 새 녹화·배치의 검토가 필요하다.

현재 기존 Test7는 이미 개발 분석에서 반복 관찰했고 최종 Full-fit에 들어갔다. 따라서 **Export K/D의 독립 검증 집합이 아니다**. 새 촬영 없이 실데이터 독립 정확도나 차량 적용 가능성을 주장하지 않는다.

## 7. 회귀 테스트와 결과 보존

고유 테스트 기준 **319 PASS /0 FAIL /0 SKIP /0 XFAIL**. 전체 repository suite를 뜻하지 않는다. 배치별 JUnit, 중간 실패, 최종 상태 병합은 `.verification/p5-followup/test-ledger.json`에 보관했다.

- 안내/profile/Mask 회귀79개.
- P1/P2 E2E·P2 runtime·P4 Factory·P3 bundle/outbox·P5 Robust·프로젝트/Export 회귀154개.
- 독립 고정 파라미터 평가·호환성·외부 파라미터 비교·로컬 Git 회귀49개.
- ROSbag/P2 gate/readiness34개 및 Mask+Robust MCAP 완료 결과1개.
- 실제 공통 프레임/코너·잔차 재구성·indexed 예외·판정 보존·원본 해시 검증5개. 중복되는 단위 테스트는 한 번만 집계했다.

Mask+Robust 합성 MCAP는16장 선정 후4개 Full-fit이 성공했지만 **Rational Train-only는 물리 검사 실패**, Hold-out 미산출이다. 선택 모델은 유효한Brown이며 파이프라인COMPLETED/WARN, Factory reference 없음은SKIPPED, Publisher disabled/PENDING이다. 처음 분석용 harness가 모든 모델의 Hold-out 성공을 가정해 실패했으나, 실제 실패의 명시·선택 부적격·WARN 보존을 확인하도록 정정했다. **계산 결과를 재실행하지 않고 완료된 저장 결과를 검증**했으며 Rational Hold-out 실패를 성공으로 바꾸지 않았다. 수정 전 harness와 JUnit도 남겼다.

로컬 Git E2E는 합성 P1/P2 결과를 전용 Bare 저장소에 게시했고 두 이력을 확인했다:

- P1 `bc51d8c3164f872a1ed5b089e4ad6a75e9853c5e`
- P2 `35d0d8ce0e712991d79d4a98489c28a3d2bf6ea1`
- Push 직후 중단·복구 시험 `7b463482dbdcdb1d332b8daa90a420e78250af12`: retry는duplicate이며 Commit 수1 유지.

이는 임시 테스트용 **결과 저장소**의 Commit이다. 코드 저장소 Commit/Push나 외부 네트워크 게시가 아니다. 업로드 데이터에 원본 영상/bag이 포함되지 않는 bundle 정책·민감 정보 검출·결과 보존 회귀도 기존 테스트로 확인했다. 실제 외부 GitHub 인증/Push는 미검증이다.

과거 전체 suite의 알려진4실패·SKIP/XFAIL·장시간 Spline 미완료 항목은 이번 범위에서 재실행하지 않았고 성공으로 처리하지 않는다. 실제 ROS 하드웨어/카메라 연결 및 신규 독립 bag 검증도 미검증이다.

## 8. 다음 촬영과 개발 우선순위

1. **보드 실측**: Square100mm와 Marker75mm 비율/실제길이, 평탄도·인쇄를 확인한다. Marker75mm는 계속 UNCONFIRMED_ASSUMPTION으로 Manifest에 기록했다.
2. **가시 경계 촬영**: native4×4 grid의3행1~3열, 좌우·상단 가장자리에서 sharp/노출 적정 코너를 확보한다. 보닛 마스크 내부는 제외한다.
3. **자세와 거리 분리**: in-plane roll만 돌리지 말고 같은 거리에서 yaw/pitch 양방향을 바꾸고, 같은 자세에서 근접/중간/원거리도 바꾼다. 근접은 코너가 한 행에 몰리지 않게 한다. 부분 보드는 별도 스트레스 그룹으로 보존한다.
4. **전략 고정 후 새 Test**: 새 녹화의 timestamp/scene cluster를 기존 개발 집합과 분리하고 고정 Export K/D로 평가한다. 충분한 RMS/P95/edge/거리·tilt별 근거가 쌓이기 전 WARN/검토 상태를 유지한다.
5. **다음 개발**: frame5류 PnP가 native 성공을 반환해도 큰 잔차를 가질 수 있는 문제를 추가 분석한다. 이번에는 기존 PnP/Calibration 알고리즘·임계값을 바꾸지 않았다. 동일Test를 보고 seed·모델·검사를 조정하는 방식으로 성능을 올리지 않는다.

## 변경 및 증거

수정 Python: `calibration/diagnosis.py`, `calibration/recommender.py`(안내 context 전달만), `automation/profile.py`, `automation/live/task.py`, `app/cli.py`.
신규: `scripts/evaluate_frozen_intrinsics.py`, `tests/test_occlusion_capture_guidance.py`, `tests/test_frozen_intrinsics_evaluation.py`, 본 보고서와 독립 검증 runbook. 기존 `AUTOMATION_P5_REPORT.md`에는 두 Confidence의 구분 정정을 추가했다.

코어 optimizer, 모델 추천 산식/가중치, Observability 기준, Acceptance 기준은 유지했다. 숫자 정책 함수는 AST 비교, 코어 파일은 시작 시 hash 비교, 실제 Export/Confidence는 원본 프로젝트와 수치 비교로 확인했다.

증거 디렉터리: `.verification/p5-followup/`. frozen plan, 후보별 JSON/ccproj, distribution, frame 시각화·trace, 고정 비교 YAML, 재Export 결과, JUnit 및 `FOLLOWUP_VERIFICATION_MANIFEST.json`에 수치/입력/가정/미검증 상태를 기록했다.

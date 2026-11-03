# P5 Calibration 최적화 안정성 검증

> 후속 검증 정정: 62.8/LOW는 모델 선택 Confidence다. 저장된 캘리브레이션 최종 Confidence는 **10.4/REJECT**다. 두 지표를 분리한 후속 분석은 `P5_FAIR_COMPARISON_REPORT.md` 참조. 기존 계산 결과는 변경하지 않았다.

2026-11-03. 결과는 진단용이며 차량 적용·승인 대상이 아니다. 기존 기본 계산 경로와 추천 가중치, 품질 판정 기준은 유지했다. 외부 Push, 코드 저장소 Commit/Push, 배포를 수행하지 않았다.

## 결론

- 기본 경로의 35장 결과를 재실행하여 성공 3개 모델의 K/D·RMS 및 Fisheye 실패를 **정확히 재현**했다.
- Pinhole/Brown/Rational의 초기값 변경은 안정적인 개선 근거가 부족하다. **기본값 유지**. Rational warm Full-fit은 오히려 12.010536px로 악화되어 채택하지 않았다.
- 선택 가능한 `robust_optimization` 경로에서 Train 내부 Validation으로 초기값 전략을 비교하고 Fisheye 재시도 및 Train→Full 재학습을 수행한다. 실제 Fisheye Full-fit은 **1.538666px**, Train-only Hold-out은 **1.613604px**. 추천 모델은 기존 점수 산식으로 Fisheye이며 최종 등급은 **WARN**, 모델 선택 Confidence **62.8 / LOW**, 캘리브레이션 최종 Confidence **10.4 / REJECT**다. PASS나 자동 승인을 부여하지 않았다.
- Fisheye Full-fit은 35장 중 1장/7코너가 기존 재시도 로직에서 제외된 **34장/1,255코너** 계산이다. 다른 모델의 35장/1,262코너 Fit RMS와 동일 목적함수라고 주장하지 않는다. Hold-out 평가의 Test 7장은 모든 모델이 동일하다.
- **최종 Export K/D와 Hold-out 평가 K/D는 다르다.** 최종 전체 데이터 재학습에는 기존 Test 프레임도 포함된다. 따라서 Hold-out 점수는 Export K/D의 독립 검증값이 아니다. 새 독립 촬영 검증은 미완료다.

## 입력과 보존

원본 ROSbag `rosbag2_2026_10_23-15_38_30`, `/video0/image_raw`, 원본 3840×2160. 이번 P5는 기존 자동 선정 35장 및 저장된 검출 코너를 재사용했다. bag 추출·촬영을 다시 수행하지 않았다. 실제 ROS·카메라 연결 검증은 미검증이다.

보드: ChArUco 10×7, DICT_5X5_100, Square 100mm. **Marker 75mm는 미확인 가정**이다. 이전 영상 기반 비율 추정은 실측을 대신하지 않는다. 보드 치수·평탄도·인쇄 오차가 확인되지 않아 물리적 정확도와 절대 거리 정확도를 보장할 수 없다.

기존 Polygon `docs/masks/rosbag1023-video0.yaml`을 유지했다. 좌표 변환·resize·crop 없이 원본 픽셀과 K/D를 사용한다. 선정35장과 Fisheye 유지34장 모두 Full-frame Coverage **68.75%**, Observable Coverage **77.028112%**다. 마스크 영역의 정확도 검증을 뜻하지 않는다. 원본90장+현재35장 PNG **125파일 SHA256 일치**. 이전 결과·미커밋 변경을 보존했으며, 기존 image 디렉터리 삭제 상태도 변경하지 않았다.

## 1. 실제 원인과 수치 정의

Full-fit RMS는 해당 모델의 K/D 및 해당 학습 프레임 자세를 동시 최적화한 코너 가중 Euclidean RMS다. Hold-out은 **Train-only K/D를 고정**하고 각 Test 프레임 자세만 추정한 뒤 `sqrt(mean(frame_RMS²))`로 집계한다. 프레임별 코너 수가 달라 point-weighted Test RMS와 다르다. 어느 정의도 변경하지 않았다.

기존 Rational: Full35 K의 fx/fy는 5731.97/5447.52, Train28 K의 fx/fy는 1914.72/1923.13이다. 10.516px와 1.592px는 같은 K/D의 학습·검증 오차가 아니다. Train28의 최적화된 자세와 고정 K/D의 Test 자세를 결합하면 같은35장/1,262코너에서 **1.545861px**의 실현 가능한 잔차가 나온다. 이는 10.516px가 데이터가 강제하는 최저 오차가 아님을 보여준다. 전역 최적해·실제 정답 증명은 아니다.

동일한 좋은 K/D에서도 모든 자세를 새 PnP로 추정하면 부분 보드 frame5에서 2659.96px의 잘못된 자세가 가능했다(기존 최적화 자세 약0.941px). Train-only K/D를 초기값으로 넣어도 Full-fit은 12.010536px로 수렴했다. **K 초기값뿐 아니라 초기 Extrinsic과 비선형 수렴 경로의 민감성**이 확인된다. 네이티브 내부의 각 반복을 추적하지 않았으므로 정확히 어느 iteration에서 국소해로 진입했는지는 미확인이다.

기존 Full-fit Rational 잔차는 median6.376, P95 20.063, 최대69.349px였다. 최악2프레임 SSE 합계24.77%, 최악5프레임44.76%. 상위 프레임을 단순 제거해도 잔여 RMS는 9.269/8.528px이므로 하나의 이상치만으로 설명되지 않는다. 자세한 이전 근거: `docs/CALIBRATION_QUALITY_DIAGNOSIS.md`와 `.verification/quality-diagnosis/`.

## 2. 모델별 초기화 구조

| 모델 | 기존 경로 | 비교 가능한 추가 초기값 | 왜곡 전달 규칙 |
|---|---|---|---|
| Pinhole | OpenCV planar cold initialization, D=0 고정 | Pinhole K, Brown K, 같은 모델 subset | D는 항상0 |
| Brown | cold, 5계수 | Pinhole K, Brown K/D, subset | 같은 Brown5만 복사, 다른 모델은 K만 |
| Rational | cold, RATIONAL_MODEL 8계수 | Pinhole K, Brown→Rational, Rational subset | Brown5 + 분모3개0 또는 동일 Rational8 |
| Fisheye | 같은 데이터 Pinhole K + D4=0 | cold, Brown K, Fisheye subset의 K | **모든 seed는 K만**, Brown/Rational D를 복사하지 않음 |

기존 목적함수·native solver·종료조건 `EPS+MAX_ITER,100,1e-6` 유지. 새 seed 경로에만 `USE_INTRINSIC_GUESS`가 추가된다. 후보별 native flags, 실제 전달 D, seed 출처 IDs/K/D, 성공·실패 이유, 유지 프레임, 잔차 통계, 시간을 audit에 기록한다. OpenCV API는 실제 반복 횟수·최종 gradient·EPS/iteration-limit 중 종료 이유를 반환하지 않는다. 따라서 단순 native return을 수렴 보증으로 기록하지 않는다.

## 3. 후보 비교 및 채택 범위

고정 Outer Train28/Test7. Train28 내부에서 **Train21/Validation7**을 고정했다. 아래 Validation7은 Outer Test7과 다르다. 후보는 같은 Inner Train21 입력이며 seed 계산도 그 내부 데이터만 사용한다. 같은 초기화인 perspective cold=legacy, Fisheye pinhole_k=legacy는 중복 실행하지 않는다.

| 모델 | 초기화 전략 | Inner fit RMS px | Inner Validation RMS px | 기본 물리 검사 | 후보 시간 s |
|---|---|---:|---:|---|---:|
| pinhole | legacy | 20.739575 | 20.531205 | True | 0.093 |
| pinhole | pinhole_k | 20.472339 | 85.058428 | True | 0.188 |
| pinhole | brown_compatible | 20.472339 | 19.125019 | True | 0.172 |
| pinhole | same_model_subset | 20.472733 | 525.305790 | True | 0.125 |
| brown_conrady | legacy | 2.668883 | 5450138456892.924805 | True | 0.093 |
| brown_conrady | pinhole_k | 2.521989 | 515.308038 | True | 0.157 |
| brown_conrady | brown_compatible | 2.521989 | 515.307161 | True | 0.109 |
| brown_conrady | same_model_subset | 2.521989 | 515.308407 | True | 0.094 |
| extended_pinhole | legacy | 7.787831 | 416.363484 | True | 0.094 |
| extended_pinhole | pinhole_k | 1.415584 | 1280.054051 | True | 0.187 |
| extended_pinhole | brown_compatible | 1.415550 | 1.613411 | True | 0.157 |
| extended_pinhole | same_model_subset | 10.227056 | 8.057389 | True | 0.140 |
| fisheye | legacy | 1.441618 | 228.319666 | True | 0.641 |
| fisheye | cold | 1.441618 | 331.449668 | True | 0.265 |
| fisheye | brown_compatible | 1.441618 | 286.791460 | True | 0.547 |
| fisheye | same_model_subset | 1.441618 | 223.153809 | True | 0.422 |

Inner Validation의 수백~수조 px 값은 성공 반환만으로 좋은 자세·모델이 보장되지 않음을 보여준다. 특히 부분 보드 PnP 민감성 때문에 이 작은 내부 분할의 순위를 일반화하기 어렵다. 이를 통과 기준 완화로 해결하지 않았다.

- 전략 선택은 **같은 Validation 프레임 전부의 평가 성공 + frame-equal RMS**로 수행한다. Fit RMS 최소값만으로 고르지 않는다. Test 점수는 선택·초기값·하이퍼파라미터 결정에 사용하지 않는다.
- 유한 K/D, 양의 초점거리, 양의 관측점 depth, Rational 분모 pole(화면 K 정규화 영역에 대한 보수적 점검), Fisheye 전방 hemisphere 각도함수 fold를 검사한다. 이는 실측 기반 물리적 정답·전 화면 유효성 보증을 대신하지 않는다.
- perspective 모델은 후보 재학습 악화가 확인돼 **legacy 유지 정책**이다. 초기 실험 Brown은 선택 seed로 Train45.7308/Full40.7322px, Rational warm Full은12.010536px였다. `.verification/p5/{model}.json`, `final-extended_pinhole.json`은 **채택하지 않은 중간 실험**이며 최종 동작은 `p5final` 및 `current-real-verification.json`이다.
- Fisheye만 선택된 전략으로 Outer Train을 재학습하고, 그 K로 Full 재학습한다. 선택/재학습 실패 또는 물리 검사 실패는 기존 경로로 복귀하며 이유를 기록한다. 기존 fallback도 최종 물리 검사에 실패하면 성공으로 반환하지 않는다. 단순 성공 반환도 품질 승인이 되는 것은 아니다.
- 프로세스 내 development cache(최대16건)는 실제 corner/object-point bytes, 제외 코너, 프레임 상태, camera, model, seed, OpenCV version/thread 수에 묶는다. Full-fit과 Hold-out에서 같은 후보를 재실행하지 않는다. 기존 persistent result cache와 혼용하지 않는다.

## 4. 실제35장 최종 비교

| 모델 | 기본 Full-fit px | P5 Full-fit px | 별도 Train-only px | Hold-out Test px | 실제 Fit 장수 |
|---|---:|---:|---:|---:|---:|
| pinhole | 20.819605 | 20.819605 | 19.356177 | 12.939660 | 35 |
| brown_conrady | 12.293655 | 12.293655 | 2.584170 | 2.718053 | 35 |
| extended_pinhole | 10.515849 | 10.515849 | 1.500972 | 1.591968 | 35 |
| fisheye | FAIL | 1.538666 | 1.516407 | 1.613604 | 34 |

기본 perspective 모델 K/D와 위 수치는 모두 정확히 동일하다. Fisheye native bootstrap **20/20 성공**, 동일한 기존 불확실성 정의를 유지했다. 추천 가중치는 바꾸지 않았으며, 성공한 Fisheye를 포함하면서 추천 결과가 Rational→Fisheye로 달라졌다. 최종 grade WARN, 모델 선택 Confidence62.8/LOW, 캘리브레이션 최종 Confidence10.4/REJECT; 독립 검증/승인 아님.

### Full-fit K

| 모델 | fx | fy | cx | cy |
|---|---:|---:|---:|---:|
| pinhole | 4754.170188 | 4324.790773 | 1041.883787 | 608.654045 |
| brown_conrady | 6949.111133 | 6855.691687 | 1896.595986 | 858.553048 |
| extended_pinhole | 5731.973588 | 5447.518103 | 1949.690663 | 960.675513 |
| fisheye | 1936.611126 | 1944.674070 | 1921.084162 | 1098.078408 |

최종 Fisheye D(k1,k2,k3,k4): `[-0.04604940690803187, 0.02317061089358344, -0.026220970698268797, 0.009285811137334328]`.
각 모델 Full/Train의 전체 K/D 및 정확한 프레임 ID, 코너 데이터 근거는 `.verification/p5/P5_VERIFICATION_MANIFEST.json`, `.verification/p5final/project/calibration.ccproj`에 별도 보관했다.

### 반복성과 비용

같은 전략을 독립 재실행한 실제 데이터 실험에서 모든 모델의 K/D·RMS가 정확히 일치했다. 최종 코드의 물리 검사·audit 추가 후 실제4모델 Full-fit/Train/Test를 다시 확인하여 최종 CLI와도 정확히 일치했다. 해당 검사에서는 불필요한 bootstrap 반복 없이 K/D와 핵심 수치를 비교했다.

기존 CLI 기준선98.781초(이전 측정), 최종 robust CLI128.985초. 약30.204초/30.6% 증가지만 동시 회귀 테스트, 새로 성공한 Fisheye bootstrap20회 등 비용이 섞여 있어 순수 multi-start overhead의 엄밀한 대조 실험은 아니다. 각 후보 native fit/내부 검증 및 최종 refit 시간은 audit에 기록했다. 단일 실행의 전체 단계 시간과 후보 시간 합계를 혼동하지 않는다.

## 5. Fisheye 실패와 제외 영향

기존 `calibration/models/fisheye.py` → `_robust_fisheye_calibrate` → OpenCV `InitExtrinsics`의 homography 정규화/방향 벡터 초기화, 또는 `CalibrateExtrinsics`의 조건수 점검에서 실패한다. 원래 Full35 예외는 `fabs(norm_u1)>0`, Train28 Pinhole seed 경로는 `CALIB_CHECK_COND input array 0`였다.

Train28 Pinhole fx≈472의 seed는 정규화 반경이 최대4.407이며81.96% 코너가 π/2보다 크고, D=0 fisheye 역변환이 약4.24e16까지 발산했다. fx≈1915의 호환 K는 최대반경1.061이다. 이는 왜곡0 Pinhole의 낮은 초점거리가 fisheye 초기 자세에 부적절할 수 있음을 설명한다. 이전 trace에서 Pinhole seed는25번 native 실패 후 포기했지만, cold 및 Rational K seed는 frame5 하나 제외 후 성공했다. 모든 제거 프레임이 실제 불량이었다는 뜻은 아니다.

문제 frame: `capture_000005_e8c1fca679b94372b393b8913ffcbff8`,7코너, 근접·부분 보드. object/image point는 rank2로 단순 collinear는 아니다. 기존 wrapper의 indexed exception을 근거로 **그 계산에서만** 제외한다. 원본 Dataset 상태나 이미지·다른 모델의 입력은 변경하지 않는다. 잘못된 기존 경고문(원본 Dataset 탭 상태도 변경한다고 안내)을 실제 동작에 맞게 수정했다.

P5 Full Fisheye는34장/1255코너, Train Fisheye는27장으로 각각 frame5 제외. 검증 Test7는 삭제하지 않았고 전부 평가했다. Coverage는 선정35장과 유지34장에서 동일하지만 보드 경계 및 독립 촬영에 대한 정밀도는 여전히 미검증이다. 부분 보드를 전부 삭제하는 정책은 도입하지 않았다.

## 6. Export와 평가 출처

- `CalibrationResult`: Full-fit K/D + 실제 유지 프레임 및 optimization audit.
- `ValidationResult`: 별도 **Train-only K/D**, Train/Test IDs, 각 RMS와 audit. 과거 프로젝트에 해당 K/D가 없으면 None으로 읽으며 Export K/D로 채워 넣지 않는다.
- `.ccproj` round-trip과 JSON에 두 파라미터 집합을 보존한다. HTML 비교표의 기존 혼동되는 Train RMS는 Full-fit RMS로 표시하고 Hold-out의 출처 안내를 추가했다.
- OpenCV YAML에 parameter_scope 및 fit_frame_ids를 기록한다. 표준 ROS CameraInfo schema는 유지하며 같은 session의 JSON/project에서 학습 출처를 확인한다.
- CLI summary는 `rms_parameter_scope`, `test_rms_parameter_scope`, `num_fit_frames`, `fit_frame_ids`, Hold-out Train K/D를 별도로 제공한다. `num_images_used=35`는 upstream 선정 수이며 native Fisheye Fit 수34와 구분한다.
- Export 재생성은 기존 project를 사용하여 **재계산 없이** 검증했다. OpenCV/ROS YAML K/D는 프로젝트와 일치한다. ROS의 기존 소수8자리 직렬화 양자화 때문에 최대차 약3.74e-9가 발생할 수 있으며 기존 포맷을 바꾸지 않았다.

원본35장은 이미 최종 재학습에 들어갔으므로 그중7장에 Export K/D를 적용한 pose-only 잔차를 독립 Test 성능으로 제시하지 않는다. 학습되지 않은 새 촬영이 있어야 Export K/D의 독립 검증을 수행할 수 있다.

## 7. 사용 설정

기본값은 OFF. 기존 명령을 그대로 실행하면 기존 수치·native 경로가 유지된다. 새 계산용 기존 CLI 설정에서 다음을 켠다:

```yaml
robust_optimization: true
test_ratio: 0.25
seed: 42
jobs: 1
```

```powershell
venv\Scripts\python.exe -m app.cli --config <새 출력 경로를 지정한 CLI 설정.yaml> --robust-optimization
```

P1 프로파일에서는 `calibration.robust_optimization: true`, P2/실험 프로파일에서는 `live.p1.calibration.robust_optimization: true`로 전달한다. `execution.opencv_threads: 1`을 유지한다. 실제 검증에 사용한 설정은 `.verification/p5/final-cli.yaml`이며 기존 결과를 덮어쓰는 재실행은 불필요하다.

결과만 다시 내보내려면 `--load-project .verification/p5final/project/calibration.ccproj`와 새로운 `--session-dir`를 사용한다. 저장된 validation·추천을 재사용한다. robust 옵션은 저장된 결과를 소급 재최적화하지 않는다.

이번 robust 옵션은 **Standard calibration, outlier pruning OFF**에서 지원한다. Object-Releasing 또는 기존 frame/corner outlier pruning과 동시 지정하면 계산 전에 명확하게 거부한다. 해당 기능은 기본 경로에서 계속 사용 가능하다. GUI에는 새로운 robust 토글을 추가하지 않았으며 기존 GUI 기본 동작은 유지했다. Factory 알고리즘은 변경하지 않았다.

## 8. 검증과 한계

최종 고유 테스트 기준 **291 PASS / 0 FAIL / 0 SKIP / 0 XFAIL**. 전체 repository suite를 뜻하지 않는다. 중복 실행은 최종 상태로 병합했으며 `.verification/p5/test-ledger.json`에 각 테스트와 원본 JUnit을 기록했다.

- 신규 단위10개: 계수 호환성, invalid seed/pole, Outer Test 비접근, Train K/D 프로젝트 round-trip, 기본 초기화 동일성, 정답 K/D Brown/Fisheye 대조군2개, Fit RMS 대신 Validation 기반 선택, 물리적으로 부적합한 fallback의 성공 처리 방지.
- 합성 정답 대조군은 K 절대오차≤0.02px, D≤0.002, 고정 Test 각 프레임 RMS<0.01px를 만족했다. 이것은 실제 영상의 정확도 인증이 아니다.
- P1/P2 합성 E2E, 기존 CLI 동일 입력 비교, 반복성, Factory, P3 bundle/CLI의 오프라인 동작, Mask 회귀:125 PASS.
- opt-in 합성 MCAP→자동 선정→Readiness→4모델→YAML→게시 outbox:1 PASS. Factory 파일 없음은 SKIPPED/NO_FACTORY_REFERENCE, 게시 비활성화/PENDING, Git Push0회다.
- 모델 dispatch, JSON/ROS/HTML, 프로젝트 직렬화 등46 PASS; validation/profile/runner 회귀 및 OpenCV5 Fisheye 호환성도 통과했다. 위 집계는 겹치는 고유 테스트를 한 번만 센다.
- 실제 데이터의 최종 코드 수치/분할 일치, 재계산 없는 Export, 원본125파일 및 추천/판정 코드 보존:3 PASS.

중간 테스트 실패는 시험 코드의 project_name 누락, OpenCV 다중 thread bitwise 비교, float64 object point fixture, Windows 긴 임시 경로, ROS YAML 양자화의 과도한 exact assertion, Windows 경로 구분자 조회 문제였다. 필요한 시험 수정 또는 짧은 임시 경로로 해당 실패만 재실행하여 최종 PASS를 확인했다. 첫 CLI 호출의 잘못된 `pipeline` 인자도 계산 전에 거부됐으며 올바른 기존 CLI로 수정해 완료했다. 로그를 삭제하지 않았다.

과거 전체 테스트의 알려진4실패, SKIP/XFAIL, 장시간 Spline 미완료 항목은 이번에 재실행하지 않았다. 이번 결과로 해당 항목을 성공 처리하지 않는다. 실제 외부 GitHub 인증/Push 및 차량/ROS 하드웨어는 미검증이다. P3 실제 게시 재시험은 수행하지 않았고 오프라인 bundle/outbox만 확인했다.

## 9. 다음 독립 검증 우선순위

1. Marker 길이·square 길이·보드 평탄도를 실측한다. 확인 전75mm 가정과 모든 결과의 진단용 상태를 유지한다.
2. 최종 전략·K/D·판정 기준을 고정한 뒤 **새 ROSbag**을 촬영한다. 기존 Test7와 가까운 장면/시간 cluster를 분리하고 새로운 Test는 초기값 선택에 사용하지 않는다. 기존 Train28의17장이 기존 Test와 scene cluster를 공유했으므로 현재 Hold-out의 독립성은 제한적이다.
3. 근접 부분 보드 frame5 유형과 충분한 코너의 중간/먼 거리·좌우 tilt·상단/관측 가능한 측면 가장자리를 각각 확보한다. 구조적 보닛 영역을 추가 촬영 대상으로 안내하지 않는다. 단순 장수/coverage 증가만으로 품질을 보장하지 않는다.
4. Export K/D를 고정한 새 Test의 전체/영역별·거리/tilt별 잔차, 직선성, bootstrap 및 실측 기준을 평가한다. 왜곡 계수의 식별가능성과 uncertainty가 부족하면 WARN/검토를 유지한다.
5. Rational/Perspective 개선은 초기 Extrinsic을 안정적으로 전달할 수 있는 native 경로의 원인 조사 후 별도 검증한다. 현재 개선 근거 없이 기본 optimizer를 교체하거나 학습 RMS 최소 후보를 자동 채택하지 않는다.

## 변경 파일

아래는 P5 시작 시 source hash와 비교한 Python 파일 목록이다. 기존 누적 미커밋 전체 diff와 구분했다.

- `calibration/compare.py` (modified)
- `calibration/initialization.py` (new)
- `calibration/robust_optimization.py` (new)
- `calibration/types.py` (modified)
- `calibration/validation.py` (modified)
- `calibration/models/brown_conrady.py` (modified)
- `calibration/models/extended_pinhole.py` (modified)
- `calibration/models/fisheye.py` (modified)
- `calibration/models/pinhole.py` (modified)
- `calibration/project_codecs/intrinsic.py` (modified)
- `automation/profile.py` (modified)
- `automation/runner.py` (modified)
- `app/cli.py` (modified)
- `export/json_export.py` (modified)
- `export/opencv.py` (modified)
- `export/report.py` (modified)

추가: `tests/test_robust_optimization.py`, 본 보고서, `.verification/p5/`의 재현 스크립트·후보별 JSON·JUnit·로그·Manifest. 실행 결과는 `.verification/p5final/`, 재계산 없는 최신 Export 검증은 `.verification/p5export2/`에 보관했다. 원본 결과 경로는 덮어쓰지 않았다.

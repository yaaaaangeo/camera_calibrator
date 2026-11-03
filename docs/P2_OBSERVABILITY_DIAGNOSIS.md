# P2 실데이터 검증 및 Observability 진단

이 문서는 정책 변경 전의 분석 이력이다. 이후 구현된 계산 가능성/Acceptance 분리와 최신 실데이터 결과는 [게이트 개선 보고서](P2_GATE_IMPROVEMENT_REPORT.md)를 참조한다.

검증일: 2026-11-02. 원본 55장, 이전 P1/P2 결과와 미커밋 변경을 보존했다. 임계값·캘리브레이션 수식·추천 가중치는 바꾸지 않았다. 커밋·푸시·배포하지 않았다. 이번 실행 증거는 `.verification/p2-diagnosis/`에 분리했다.

## 1. 결론

현재 차단은 **계산 불가능이 아니라 Brown 사전 계산의 높은 파라미터 상관계수를 품질 게이트로 적용한 결과**다. 사전 fit은 성공했고 Rank 9/9, normalized Condition Number 100.011이다. 최대 상관계수 0.9833169(k2/k3)에 따른 감점이 점수 3.5를 만들어 기본 최소 50을 충족하지 못한다. 수량·coverage·거리/회전 다양성·hold-out 조건은 통과했다.

데이터에도 하단 관측 부족과 잔차/일반화 문제가 있다. 따라서 데이터가 충분히 좋다거나 운영 승인 가능하다는 결론도 아니다. 반면 알려진 K/D를 거의 정확히 복원하는 합성 대조군도 같은 게이트에서 차단됐다. **실데이터 부족만으로 원인을 설명할 수 없으며, 모델별 상관계수 점수의 승인 게이트 적합성도 검증되지 않았다.** 단순 임계값 완화로 해결하지 않았다.

구체적 하단 촬영 안내가 관측성 실패 후 유실되는 버그와, 모델별 프레임 제외 후 pose 매칭이 어긋나는 진단 버그만 회귀 재현 후 최소 수정했다. 후자는 이번 Brown 차단의 원인은 아니다.

## 2. 입력과 분석 방법

- 원본: `image/`의 3840×2160 이미지 55장. 최초/최종 집약 SHA256이 P1 감사 기록과 일치한다.
- 보드 가정: ChArUco **10×7 squares, DICT_5X5_100, square 0.04m, marker 0.03m**. 물리 보드의 확인값이 아닌 추정 사양이다. 치수/사전/비율 오류가 대응점·스케일·포즈·추정값에 영향을 줄 수 있어 절대 거리나 정밀도 승인 근거로 사용하지 않는다.
- 원본55장 재검출, 기존 P2 저장54장(중복 자세1장 제외), 동일 P2 선별35장을 비교했다. 기존 체크포인트는 읽기만 했다. 재선별된35장의 경로 집합이 이전 체크포인트와 같은지 assertion으로 확인했다.
- 모델 fit은 기존 함수를 그대로 호출했다. raw55와 selected35는 네 모델의 전체/train fit, accepted54는 Brown 전체/train fit을 수행했다. 총18개 작업마다 별도 JSON/log와120초 timeout을 사용했다. 완료 결과는 재실행하지 않는 스크립트다. bootstrap은 이 진단 fit에서 실행하지 않았다.
- 기존 P1/CLI 결과는 재사용하고, 별도로 현행 P1 Runner를 P2 저장54장에 **진단 목적으로 한 번** 실행했다. 자동 Readiness를 통과시킨 것이 아니다. 최종 품질 승인과 분리했다.

재현 명령:

```powershell
venv/Scripts/python.exe .verification/p2-diagnosis/analyze.py
venv/Scripts/python.exe .verification/p2-diagnosis/controls.py
venv/Scripts/python.exe .verification/p2-diagnosis/run_real_runner.py
```

마지막 스크립트는 완료 outcome이 존재하면 중복 계산을 거부한다. 실행별 JSON과 CLI 하위 로그가 남는다. 분석용 합성 대응점은 실사진/실센서 데이터에 섞거나 P2 운영 임계값 검증값으로 사용하지 않았다.

## 3. 실제 이미지 분포

| 지표 | 원본55 | 자동 저장54 | 선택35 |
|---|---:|---:|---:|
| coverage | 75% | 75% | 75% |
| 거리 다양성 점수 | 1.0 | 1.0 | 1.0 |
| 회전 다양성 점수 | 1.0 | 1.0 | 1.0 |
| edge coverage 점수 | 0.6667 | 0.6667 | 0.6667 |
| 보드 면적 비율 최소/중앙/최대 | .0096/.1029/.3739 | .0096/.1020/.3739 | .0113/.1340/.3739 |

보드 중심의 3×3 분포(화면 상단→하단, 좌→우):

| 행 | 원본55 (좌/중/우) | 저장54 | 선택35 |
|---|---|---|---|
| 상 | 8 / 6 / 6 | 8 / 6 / 6 | 2 / 4 / 4 |
| 중 | 9 / 14 / 11 | 9 / 13 / 11 | 8 / 9 / 7 |
| 하 | 0 / 1 / 0 | 0 / 1 / 0 | 0 / 1 / 0 |

선택35장의4×4 코너 coverage score:

| 행 | 좌1 | 중2 | 중3 | 우4 |
|---|---:|---:|---:|---:|
| 상1 | .449 | 1.000 | 1.000 | 1.000 |
| 중2 | 1.000 | 1.000 | 1.000 | 1.000 |
| 중3 | .840 | 1.000 | 1.000 | .898 |
| 하4 | .049 | .185 | .166 | .068 |

네 모서리 모두 코너가 존재한다는 조건은 만족하지만, 하단 좌/우는 각각5/7코너뿐이다. 단순 존재 여부와 충분한 관측 밀도는 다르다. coverage75%도 하단4칸이 부족한 상태를 허용한다. 이는 계산 오류가 아니라 현재 coarse readiness 기준의 한계다.

거리 점수는 보드 면적 비율의 CV를 포화시킨 대리값이다. 실제 거리1.0이나 충분한3D 깊이 다양성을 뜻하지 않는다. 선택35장 Brown fit의 추정 z는 약0.155..1.563m, board-normal angle은 약5.5..71.5도이지만 보드 가정과 모델에 의존하는 사후 추정값이다.

### Tilt 점수의 해석상 문제

`board_tilt_deg`는 minAreaRect 회전각이며 yaw/pitch가 아니다. 선택35장의 최소/중앙/최대는 약-89.935/-4.411/-0.480도다. 사각형의 각 표현이0도 부근에서-90도로 바뀌어 표준편차가 과장될 수 있다.

대조군에서 실제 yaw/pitch=0인 보드를 roll -0.2→0→+0.2도로 바꾸면 검출 tilt는 약-0.200→-90.000→-89.800도가 됐다. 따라서 rotation diversity1.0을 실제 out-of-plane 다양성 충족의 강한 증거로 해석하면 안 된다. 기존 공용 품질/검출 지표의 정의를 바꾸는 수정은 이번 범위에서 하지 않았다. 후속 설계에서는 각도의 주기성과 실제 법선 방향 다양성을 구분해야 한다.

## 4. Readiness 차단의 정확한 조건

최종 선별35장, 중심7/9칸, distance/rotation1.0, edge0.6667, 중복 후보 비율0.35185로 가벼운 조건을 통과했다. train27장/hold-out8장, hold-out332코너, coverage62.5%, pose diversity0.78125로 증거량도 통과했다.

Brown train-only 사전 결과:

| 항목 | 값 | 이번 차단 여부 |
|---|---:|---|
| fit | 성공, train RMS2.308475px | 계산 실패 아님 |
| Rank | 9/9 | 통과 |
| normalized Condition Number | 100.0112 | score condition 감점0 |
| raw Condition Number | 68374.52 | 점수에는 정규화값 사용 |
| 최대 절대 상관계수 | .9833169 | **주된 감점 원인** |
| strongest pairs | k2/k3=-.983317, k1/k2=-.970048 | radial 계수 coupling |
| score / 최소 기준 | 3.5 / 50 | **실패** |

기존 점수는 `100 - max(rank감점, condition감점, correlation감점)`이다. 상관 감점은 `clip((rho-.8)/.19*100,0,100)`이므로 이번 감점은 약96.48이다. 다른 감점이 없을 때 score≥50은 rho≤.895에 해당한다. 코드의 arithmetic 오류가 아니라 높은 다항식 계수 상관을 강하게 벌점화한 정책이다. rank가 충분하고 condition 감점이 없어도 이 한 항목만으로 차단된다.

기존55장 재생의 신규 촬영 누적에 따른 사전 점수는 다음과 같다. 과거 응답을 읽었으며 동일 계산을 반복하지 않았다.

| 저장 누적 | 35 | 40 | 45 | 50 | 54 |
|---|---:|---:|---:|---:|---:|
| 사전 Brown score | 4.0 | 3.2 | 3.7 | 3.5 | 3.5 |
| Rank | 9 | 9 | 9 | 9 | 9 |

추가된19장의 실제 기존 이미지로는 Readiness가 통과하지 않았다. 새 실물 촬영 데이터가 없으므로 특정 촬영 지시를 따르면 반드시 통과한다고 주장할 수 없다.

## 5. 선별 전후 및 수동/CLI 비교

Brown 비교:

| 데이터/fit | RMS(px) | Rank | norm condition | 최대상관 | score |
|---|---:|---:|---:|---:|---:|
| 원본55 전체 | 6.512080 | 9/9 | 106.310 | .979791 | 5.4 |
| 원본55 train43 | 2.996301 | 9/9 | 105.642 | .980031 | 5.2 |
| 저장54 전체 | 7.170810 | 9/9 | 123.343 | .980358 | 5.1 |
| 저장54 train42 | 6.714083 | 9/9 | 128.822 | .982592 | 3.9 |
| 선택35 전체 | 2.600584 | 9/9 | 97.031 | .982306 | 4.0 |
| 선택35 train27 | 2.308475 | 9/9 | 100.011 | .983317 | 3.5 |

선별 후 fit 잔차는 줄지만 관측성 score는 오히려 낮다. 선별이 이 상관계수 점수를 최적화하는 것은 아니다. 원본55와 선택35는 train/test 집합도 달라 잔차 개선을 동일 hold-out 성능 개선으로 단정할 수 없다. 원본55 test RMS4.2522px, 선택35 test RMS4.2712px였다. 전체 raw fit과 train fit의 focal/주점 차이도 존재하므로 작은 잔차만으로 파라미터 안정성을 승인할 수 없다.

별도 **진단용 P1 Runner**는 저장54장→선별35장→기존 CLI/4모델/hold-out/export를 정상 완료했다. 자동 Readiness를 변경하거나 우회 승인하는 기능은 추가하지 않았다.

| 기존 P1 경로 모델 | 전체35 fit RMS | hold-out train RMS | hold-out test RMS | 전체 fit 관측성 score |
|---|---:|---:|---:|---:|
| Pinhole | 33.852197 | 31.629171 | 44.229990 | 91.9 |
| Brown | 2.600584 | 2.308475 | 4.271152 | 4.0 |
| Rational | 7.863552 | 4.284946 | 6.221400 | 0.0 |
| Fisheye | 1.885608 | 1.292665 | 4.161213 | 0.0 |

이번 Runner와 기존 직접 CLI 아티팩트 간 네 모델 모두 **K/D 최대 차이0, fit/test/edge RMS 차이0**, 선택 모델은 동일한Brown이었다. train/test ID 문자열은 파일명이 달라 직접 동일하지 않지만 수치 결과가 일치했다. 판정은 **WARN(P1 exit3)**이며 기존 종합등급POOR, 신뢰도REJECT, 운영 허용값 미설정이 기록됐다. 계산 완료와 품질 승인/PASS는 다르다.

Fisheye의 standalone 진단 fit은 초기 K를 주지 않았고 P1 hold-out은 기존 pinhole-train 초기값을 사용한다. standalone train RMS1.400686/test3.990750과 위 P1값의 차이를 P2 불일치로 판정하지 않았다. 서로 다른 초기화/자동 제외 경로다. 실제 P1 실행에서는 불안정한 train 프레임12장 자동 제외 경고도 발생했다. 따라서 모든 입력 프레임이 모델마다 동일하게 학습됐다고 간주하지 않는다.

## 6. 모델별 관측성의 적절성

기존 구현은 모델별 projection의 수치 Jacobian을 사용한다. Pinhole4변수, Brown9변수, Rational12변수, Fisheye8변수로 차원이 다르다. Rational의 k4..k6 활성화와 Fisheye의 별도 투영식은 [OpenCV calibration 문서](https://docs.opencv.org/4.10.0/d9/d0c/group__calib3d.html)에 따른 모델 구분이다.

같은 score50을 전 모델의 동일 신뢰도로 해석하는 근거는 부족하다. radial 다항식 기저 간 상관은 모델 복잡도/샘플 반경 분포에 의존한다. 이번 Pinhole은 score91.9이지만 test RMS44px로 매우 나쁘다. 관측성은 모델 적합성/정밀도/hold-out을 대체하지 못하며, 높은 score만으로 모델을 바꾸거나 PASS 처리하면 안 된다.

관측성은 현재 **extrinsic을 고정한 국소 intrinsic 진단**이다. extrinsic과의 결합을 주변화하지 않아 full rank가 전체 calibration의 식별 가능성을 보장하지 않는다. column 정규화는 단위 영향을 줄이지만 모델 기저의 상관 자체를 제거하지 않는다. score는 오차 확률·신뢰구간·예측 오차의 보증값이 아니다.

### 알려진 K/D 합성 대조군

실사진을 흉내 낸 측정값이 아니라, 알려진 K(fx=fy750px)/Brown D와 보드/포즈에서 직접 생성한 **노이즈 없는 대응점** 실험이다. float32 양자화는 포함된다. seed214, 현재 calibrate_brown_conrady/observability 함수를 그대로 사용했다. 이미지 검출·선별/ROS/센서 시험을 대체하지 않는다.

| 단계 | 표본 수 | coverage | fit RMS(px) | K 최대복원오차(px) | 최대상관 | score |
|---|---:|---:|---:|---:|---:|---:|
| 상·중 영역 | 36 | 75% | .0000201 | .000195 | .980989 | 4.7 |
| 하단 추가 | 60 | 93.75% | .0000218 | .000167 | .983509 | 3.4 |
| 거리·기울기·가장자리 추가 | 96 | 100% | .0000214 | .000104 | .984004 | 3.2 |

모두 rank9/9, normalized condition 약489..529이며 D 최대 복원오차는3.8e-7 미만이다. 실제 식별값을 거의 정확하게 복원해도 score50을 넘지 않는다. 이 결과는 현 게이트의 false-block 가능성을 보여주며, 실데이터가 양질이라는 증거로 전용하지 않는다. 추가 촬영이 coverage/복원오차를 개선해도 해당 score는 단조 개선하지 않았다.

## 7. 촬영 가이드와 필요한 구분

현재 실제 데이터에 근거한 우선 가이드는 **하단 좌측/하단 우측/하단 중앙**이다. 보드 코너가 해당 하단 영역에 실제로 관측되도록 위치를 이동하고, 같은 위치의 연사 대신 가까운/중간/먼 거리에서 선명한 프레임을 확보해야 한다. 실제 보드 법선 방향을 바꾸는 yaw/pitch와 영상 내 roll을 구분해야 한다. 이미 포화된 거리/회전1.0만 보고 추가 변화가 불필요하다고 안내하면 안 된다.

기존 quality 모듈은 이러한 하단 위치 안내를 이미 생성했으나 관측성 실패 시 controller가 generic reason으로 덮었다. 이를 보존하도록 수정했다. 최종 안내에 보이는 구체적인 추가 장수는 기존 휴리스틱 권고이며 해당 수량이면 통과한다는 보장이 아니다. 상관계수 coupling에는 촬영 외에 모델 적합성·보드 확인·검출 오차·판정 정책 검토도 필요하다.

| 상태 | 의미 | 처리 원칙/이번 데이터 |
|---|---|---|
| 계산 불가능 | 입력/대응점 부족, 비유한 값, fit 실패 등 | ERROR와 원인 기록. 이번 Brown은 해당하지 않음 |
| 계산 가능·낮은 신뢰도 | fit 성공, 높은 결합/잔차/불안정 | 진단 결과를 보존하되 PASS 금지. 이번 데이터 해당 |
| 추가 촬영으로 개선 가능한 부분 | 하단 관측 부족 등 구체적 결손 | 정상 데이터 보존 후 새 표본으로 재평가. 개선 보장 금지 |
| 사용자 검토 필요 | 보드 사양 미확인, 모델 부적합, 미검증 승인 기준 | 검토 전 적용 금지. 이번 데이터 해당 |

진단 계산과 최종 승인을 분리하는 구조가 적절하다. 이번에는 기존 Runner를 명시적인 별도 진단으로 실행했고 P2 자동 게이트는 유지했다. 향후 자동 진단 경로를 추가한다면 계산 가능성 검사와 승인 검사를 분리하고, 낮은 신뢰도 결과에 명시적인 미승인 상태·한정된 재시도·전체 근거를 저장해야 한다. 자동 PASS/게이트 무시 모드는 이번에 추가하지 않았다.

## 8. 확인된 버그와 최소 수정

1. `automation/live/controller.py`: 사전 관측성 실패 후 위치별 coaching을 버리는 문제. 실패 사유와 기존 coaching을 중복 없이 합친다. readiness 수치/threshold/state transition은 바꾸지 않았다.
2. `calibration/observability.py`: 모델이 일부 프레임만 유지한 경우 원본 prefix와 rvec/tvec를 zip하는 문제. 기존 `per_frame_error`의 retained ID로 입력을 필터링해 원래 순서대로 매칭한다. ID 없이 subset 길이도 다르면 임의 매칭하지 않고 관측성 산출 불가로 처리한다. optimizer/K/D/추천 가중치/score 수식은 변경하지 않았다.

회귀 테스트는 `tests/test_p2_observability_diagnostics.py`의 안내 보존1개, `tests/test_observability_frame_alignment.py`의 매칭2개다. 전자 파일에는 실패→추가5장→재평가→다음 계산 요청을 확인하는 상태 통합 테스트1개도 추가했다. 해당 상태 검증은 준비도 함수를 mock한 테스트 전용 소규모 데이터이며 운영 임계값 통과 증거가 아니다. 수정 전 안내 유실/잘못된 residual 및 불명확 subset 허용을 재현했고 수정 후 통과했다. 후자의 수정은 이 실데이터의 Brown 원인과 별개다. 기존/신규 정상 전체-frame 아티팩트4개×4모델의 관측성 보고서16개를 재산출해 **기존값과 완전 일치**함을 확인했다. 이전 보고서/결과 파일은 수정하지 않았다.

Tilt의 각도 wrap 문제와 Brown 공통 게이트의 정책 적합성은 확인/검토 사항으로 남겼다. 이들을 고친다는 명목으로 공용 품질 정의나 임계값을 임의 변경하지 않았다.

## 9. 통합 검증과 테스트 집계

실행 환경은 Windows venv, GUI는 offscreen이다. 원장/집계는 `.verification/p2-diagnosis/test-result-ledger.json`, `test-summary.json`이다. 동일 ID의 최신 결과만 합산한다. 수정 전 의도적 실패 XML은 회귀 재현 근거로 별도 보존한다.

```powershell
venv/Scripts/python.exe -m pytest -q tests/test_p2_live.py tests/test_p2_readiness.py tests/test_p2_runtime.py tests/test_p2_bag.py tests/test_p2_gui.py tests/test_p2_e2e.py --basetemp=.verification/p2-diagnosis/p2-tests --junitxml=.verification/p2-diagnosis/p2-tests.xml
venv/Scripts/python.exe -m pytest -q tests/test_automation_acceptance.py tests/test_automation_profile.py tests/test_automation_runner.py tests/test_automation_e2e.py tests/test_cli_session_dir.py --basetemp=.verification/p2-diagnosis/p1-tests --junitxml=.verification/p2-diagnosis/p1-tests.xml
venv/Scripts/python.exe -m pytest -q tests/test_observability.py tests/test_observability_frame_alignment.py tests/test_quality_pose_distribution.py tests/test_diagnosis.py --basetemp=.verification/p2-diagnosis/metrics-after --junitxml=.verification/p2-diagnosis/metrics-after.xml
venv/Scripts/python.exe -m pytest -q tests/test_p2_observability_diagnostics.py --basetemp=.verification/p2-diagnosis/guide-integration-final --junitxml=.verification/p2-diagnosis/guide-integration-final.xml
venv/Scripts/python.exe -m pytest -q tests/test_p2_e2e.py --basetemp=.verification/p2-diagnosis/e2e-after --junitxml=.verification/p2-diagnosis/e2e-after.xml
```

- P2 기존 신규군61개를 재실행했다. 계산 연결/CLI 일치 E2E는 합성 이미지 기반으로, 연결 확인을 위한 테스트 전용 준비도/관측성 완화값을 사용한다. **기본 게이트가 실데이터/실카메라에서 통과했다는 증거가 아니다.** 두 버그 수정 뒤 E2E2개도58.71초에 통과했다.
- P1 회귀138개, 기존 관측성/품질/진단27개 및 새 매칭 회귀2개, 새 안내 회귀1개를 확인했다.
- 실패→추가 촬영 분기/최소 신규 프레임 조건, 취소/재시작/응답 재사용, 실제 소유 프로세스 트리 취소, GUI 시작/오류 복귀 및 수동 버튼 호환은 해당 단위/런타임/GUI 테스트 범위다. 실제 카메라로 추가 촬영하여 최종 승인까지 도달한 통합 시험은 아니다.
- 실제 ROS/카메라 하드웨어, 실기 자동/수동 전환, Linux/Jetson은 계속 미검증이다.
- 전체 기존1376개를 다시 실행한 것은 아니다. 기존 FAIL4개와 Spline 미완료6개를 해결/통과 처리하지 않았다. 과거 단일 프로세스 전체 실행의 비정상 종료와 이번 파일/그룹별 완료 결과는 별도다.

최종 고유 테스트는 **230개: PASS230 / FAIL0 / SKIP0 / XFAIL0 / 미완료0**이다. P2기존61＋신규P2검증2＋P1회귀138＋기존지표27＋신규매칭2로 구성된다. 중간 fixture assertion 수정 이력과 의도적인 수정 전 실패는 로그에 보존하고 최신 결과로 대체했다. 이전 전체 기준선의 기존FAIL4개가 이번에 해결됐다는 뜻은 아니다. P1회귀군은142.10초, 기존P2군은73.03초, 수정 후 지표군29개는1.67초에 완료했다. 오래 걸리는 작업은 전용 로그/작업 JSON을 남겼고 코드 변경으로 영향을 받는 E2E만 추가 재확인했다.

최종 원본 무변경과 모델/추천/임계값 파일12개의 이전 설치 스냅샷 대비 바이트 일치는 `final-integrity.json`으로 확인했다. 이번 제품 코드 수정은 위 두 파일뿐이며, 신규 테스트2파일/본 보고서/기존 P2보고서의 후속 링크 및 별도 검증 자료를 추가했다. `git diff --check`도 통과했다.

## 10. 다음 개발 진행 여부

오프라인 진단·촬영 안내 보완 및 실ROS 검증 준비는 진행 가능하다. **현재 기본 Observability 게이트를 실카메라 자동 완료/승인 정책으로 확정하거나 Factory/차량 적용 단계로 넘어가기는 이르다.** 실데이터 결손과 정책 부적합 가능성이 함께 남아 있다.

다음 선행 조건은 물리 보드/Serial 확인, 하단·다중 거리·실제 yaw/pitch를 보완한 새 실사진, 동일 독립 hold-out에서의 모델별 잔차/안정성 비교, 모델별 관측성의 해석/오승인·오차단 검증, extrinsic 결합을 고려한 독립 진단 설계 검토, 실ROS 취소/재연결/GUI 시험이다. 기준은 검증 자료로 승인해야 하며 이번 점수를 통과시키기 위해 낮추면 안 된다.

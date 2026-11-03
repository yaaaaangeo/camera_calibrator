# P4 Factory reference comparison

P4는 선택적인 비교 진단이다. 기존 Calibration, 추천 가중치, PASS/WARN/FAIL,
승인·배포 상태를 변경하지 않는다. 비교가 실패해도 완료된 Local 결과는 유지한다.

## 파일과 좌표 선언

`factory_reference.example.json`은 합성 예제이며 제조사 데이터가 아니다.
제조사 파일을 확보한 뒤 정확한 Camera Model, S/N, 해상도, 출처를 기록한다.
`projection_convention: opencv`는 해당 데이터의 실제 투영 수식과 계수 정의를
확인한 경우에만 선언한다. 출처 문자열이나 S/N 일치는 제조사 인증이 아니다.

지원하는 수학 모델은 Brown 5, Rational 8 (`extended_pinhole`), OpenCV Fisheye 4다.
Rational 순서는 `k1,k2,p1,p2,k3,k4,k5,k6`다. 미확인 vendor 순서를 자동 변환하지 않는다.
Camera Model 문자열은 특정 두 기종으로 제한하지 않아 추가 기종의 파일도 사용할 수 있다.
SG8-ISX028C의 SDK 지원을 의미하지는 않는다.

JSON에 K/D를 직접 넣거나, K/D 대신 `intrinsic_file: intrinsic.yaml`을 지정한다.
후자는 JSON과 같은 디렉터리 아래의 파일만 읽고 기존 OpenCV/표준 calibration
파서를 재사용한다. 원본 YAML에 없는 식별·좌표·계수 정의는 JSON에 반드시 명시한다.
메타데이터 없는 YAML을 제조사 파일로 추정하지 않는다.

현재는 명시적인 full-frame ROI, binning `[1,1]`, orientation `identity`, 동일 기준
해상도만 비교한다. crop/binning/flip/resize 자동 변환은 지원하지 않는다.
Session에도 같은 형식의 좌표 JSON을 별도로 제공한다. 과거 Session에 이 정보가
없으면 실제 촬영 설정을 확인하기 전에는 선언하지 않는다.

## CLI

```powershell
python -m automation factory lookup --serial SN123 --camera-model SG3S-ISX031C --factory reference.json
python -m automation factory compare --session <session> --factory reference.json --coordinates session-coordinates.json --seeded
```

기본 출력은 원본 Session 밖 `factory_comparisons/<unique-id>/`다.
`--output-root`로 별도 부모 폴더를 지정할 수 있다. 원본 Session 안으로 쓰기는 거부한다.
JSON에는 전체 수치, 프레임별 실패, 원본 바인딩 SHA256이 들어가고 HTML은 동일 JSON을
escape하여 표시한다. 비교 성공은 exit 0, 후보 실패로 INCOMPARABLE은 3,
입력·호환성·SDK 조회 오류는 1이다. 비교 명령에 P1의 품질 PASS 의미를 부여하지 않는다.

P1 `run`, P2 `live`에 `--factory reference.json --factory-coordinates session-coordinates.json
[--factory-seeded]`를 추가하면 Local 완료 후 비교한다. 실패 이벤트는 INCOMPARABLE이며
Local 종료 코드와 품질 판정을 유지한다. 옵션이 없으면 기존 경로 그대로다.

GUI External Comparison의 **P4 Factory 비교 보고서 보기**는 JSON을 읽기 전용으로
표시한다. 수동 K/D 입력이나 모델 선택을 바꾸지 않는다. 계산 실행은 CLI를 사용한다.

## 비교와 해석

기존 P1 chosen model의 기록된 Train/Test 분할을 사용한다. Local은 기존 함수로 Train만
재학습하고, Factory-fixed는 K/D를 고정하며, Seeded는 Factory 모델의 K와 D 둘 다
초기값으로 Train만 최적화한다. 기본 solver는 변경하지 않는다. 중복 ID, 교집합,
동일 이미지 bytes/동일 검출 관측의 Train/Test 중복을 거부한다. 검출 코너와 object points의
해시 및 개수를 기록한다. 제조사 자체 학습 데이터와의 독립성은 별도 확인이 필요하다.

평가는 기존 `_evaluate_side`를 공통 사용한다. 모델별 PnP로 자세만 추정한다.
Test RMS는 프레임 RMS의 제곱평균, `residual_stats.rmse`는 코너를 합친 RMS다.
Median/P95/P99, edge, straightness, spatial, K/D, nominal FOV, 모델 자유도와
기존 fixed-pose 파라미터 진단을 기록한다. 후자는 Test의 국소 sensitivity/covariance
근사이며 training observability 또는 반복 촬영 안정성 측정이 아니다.
다른 모델의 D 항목은 숫자로 직접 비교하지 않는다. 같은 모델은 기존 P3의 공통 광선
투영 비교도 재사용하며, 다른 모델은 해당 비교 제한 이유를 기록한다.

결론은 FACTORY_SUPERIOR / LOCAL_SUPERIOR / SEEDED_SUPERIOR / UNCLEAR / INCOMPARABLE다.
모든 Test 프레임·코너가 평가되어야 한다. 실패 후보를 숨기거나 공통 생존 프레임만으로
순위를 매기지 않는다. 우세 판정에는 최소 5 Test 프레임, 각 상대와의 frame bootstrap
95% 개선 CI 하한 > 0, edge/straightness 근거의 비열화, RMS 차이 > 0.001px가 필요하다.
이는 미세 수치 차이의 과장 방지를 위한 비교 규칙이며 기존 acceptance 임계값이 아니다.
여러 후보의 차이가 작거나 근거가 없으면 UNCLEAR다. Factory RMS는 비교 순위에서 제외한다.
한 번의 hold-out 탐색으로 일반적인 우월성이나 차량 적용 안전성을 확정하지 않는다.

## P3 연결

`publish --session <session> --factory-report <comparison.json>`으로 명시적 첨부가 가능하다.
또는 P1/P2의 Factory와 publish 옵션을 함께 사용하면 생성된 보고서를 전달한다.
실제 외부 게시 여부는 기존 별도 publish 설정을 따른다. 기본 disabled 설정은 로컬
Outbox만 작성한다. P4 개발 검증 중 외부 게시·Git commit/push는 실행하지 않는다.

보고서의 Session manifest/project SHA 및 식별자를 검증한 후 정제 JSON과 요약을 bundle과
history에 넣는다. 원본 HTML은 첨부하지 않는다. 비교 수치 변경은 내용 해시에 반영하지만
RunID/시간에 따라 달라지는 바인딩 SHA는 semantic hash에서 제외하여 기존 중복 규칙을 유지한다.
원본 파일 SHA는 감사 메타데이터에 남는다. 게시·비교 우세는 승인이나 배포가 아니다.

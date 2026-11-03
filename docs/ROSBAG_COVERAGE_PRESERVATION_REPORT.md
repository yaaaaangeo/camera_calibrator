# 실제 ROSbag Coverage 보존 개선 검증

대상: `rosbag2_2026_10_23-15_38_30`, `/video0/image_raw`, 3840×2160. 기존 저장 90장과 검출 결과를 재사용했다. bag 재추출·원본/저장 이미지 변경·삭제·Commit·Push·배포는 수행하지 않았다.

보드: 사용자 확인 ChArUco 10×7, DICT_5X5_100, Square 100mm. Marker 75mm는 미확인 가정이다. 검출 코너의 실제 위치 정확도 및 산출 K/D를 정답으로 간주하지 않는다. 카메라 모델/SN도 미확인 진단 식별자다.

## 결과

|자료|장수|Coverage|Readiness|
|---|---:|---:|---|
|전체 저장|90|75%|하단 모서리 미관측|
|기존 선정|35|68.75%|Coverage 및 모서리 미충족|
|개선 선정|35|75%|모서리 미충족 유지|

전체 90장에서 4행 전체 4개 셀이 비어 있다. 가능한 관측 셀은 최대 12/16=75%이며, 개선 선정이 이 상한을 달성했다. 이 데이터의 최대값에 대한 근거는 빈 셀 상한과 35장 구성의 실제 달성 결과다. 구현은 일반적인 전역 최적 조합 보장 알고리즘이 아니다.

사라졌던 셀은 **3행 4열**이다(위→아래/좌→우 1부터). 완전 무관측이 아니라 fair-share 정규화 기준 탈락이다. 기존 14코너/점수 0.15819 → 개선 27코너/0.31191로 기존 기준 0.3을 충족한다. 정규화 분모도 총 코너 수에 따라 바뀐다.

## Coverage Grid 코너 수

행은 화면 위→아래, 열은 좌→우이며, 점수/전체 ID는 증거 JSON에 함께 기록했다.

### all

```text
 342  402  410  345
 319  558  325  253
  97  110   73   66
   0    0    0    0
```

### before

```text
  76  170  233   55
 143  326  208   69
  33   49   40   14
   0    0    0    0
```

### after

```text
  76  152  224   47
 143  308  199   87
  33   49   40   27
   0    0    0    0
```

## 선정 변경과 제약

기존 위치 할당·품질 점수 선정 후, 품질 필터를 통과한 후보끼리 교체한다. 거리/Tilt 분류별 장수를 보존하고, 기존 위치 분류를 소멸시키지 않으며, 중복 클러스터 반복 수를 증가시키지 않는다. 이미 기준을 통과한 Coverage 셀을 잃지 않고, 기존 coverage_percentage 기준의 셀 수와 부족 셀 지원량을 우선 개선한다. 점수는 동률 선택에만 기존 값을 사용한다. 계산/추천 가중치와 품질 임계값은 바꾸지 않았다.

처음의 동일 위치 장수까지 고정한 방식은 75%를 달성하지 못했다. 최종 방식은 위치별 장수보다 관측 셀 보존을 우선하며, 위치 분류 자체와 거리/Tilt 장수는 유지한다.

제외된 55장 중 3행 4열에 기여하는 9장:

|이미지 ID|해당 셀 코너 수|기존 제외 이유|
|---|---:|---|
|`capture_000030_41ae01f5284d4345b8c580c9261531a2`|4|duplicate_scene|
|`capture_000032_82c04019854e497494afdad2cac6125c`|4|duplicate_scene|
|`capture_000033_85540b3114fb471788bb0396d587a511`|9|duplicate_scene|
|`capture_000034_0349ab72a85a4588a866dc82be8f7a4e`|10|duplicate_scene|
|`capture_000035_fd2fa70083354c8cbd9968bfcb97ee66`|8|duplicate_scene|
|`capture_000036_73cfe5da7ce64068b4a8b8874a025b02`|3|duplicate_scene|
|`capture_000037_66da09ff0d444feb886d05ebb37af0a1`|5|duplicate_scene|
|`capture_000038_14a144e5393c4b36b3d7ab6a7fd8db4b`|8|duplicate_scene|
|`capture_000082_14da318079c14a27abb86433dc3b06d8`|1|duplicate_scene|

추가 ID:
- `capture_000034_0349ab72a85a4588a866dc82be8f7a4e`
- `capture_000035_fd2fa70083354c8cbd9968bfcb97ee66`

제외 ID:
- `capture_000029_64aeddcdff4e40f5857862f7dd60d1c4`
- `capture_000057_51e2d17fa12243a9bb71a1554101aabf`

개선 후 전체 35장 ID:

- `capture_000000_dceaacf838f74596b9b11030d21e5718`
- `capture_000003_7abccedea68448c8930af0016c53a70b`
- `capture_000005_2e49b4a843df47a592378c3bae1b393f`
- `capture_000006_ba43de5226d44741838c498a09867342`
- `capture_000007_6508e2aa886e4d84b248b4f6be809e89`
- `capture_000008_fc365fe20b454c7c989eb983e40086f1`
- `capture_000009_249747c9efdc4836aed4fec540d33317`
- `capture_000011_d7c52fc9bae04f08ad636fd26f4582b6`
- `capture_000012_336b1b7fb199403bb372066e98f8b909`
- `capture_000013_db832343617b40eda5d3b61536aefca8`
- `capture_000014_5421cae6c6964904b5582cf613810576`
- `capture_000015_6bb79b061a774e2785d85a2fb151efd4`
- `capture_000019_d99cc9404f8c4f4299efd3876304525d`
- `capture_000021_69fa55ed242e4b368f9d2db2dbe3e072`
- `capture_000022_d35c818865ef4a17826705f9376b3e1f`
- `capture_000023_475745059c8f4cccaf9eb3cf526cd211`
- `capture_000034_0349ab72a85a4588a866dc82be8f7a4e`
- `capture_000035_fd2fa70083354c8cbd9968bfcb97ee66`
- `capture_000041_4b7711b2b66d4c6eaa0cff00aea26d74`
- `capture_000042_5f7920ca2b7641a3b083f7c201e1deb6`
- `capture_000043_bef2f5ddb8e14dad886547d4209c6dd7`
- `capture_000044_02c29d3f4cbd4908801e50a517c3bb77`
- `capture_000045_a771707169344a0ab694c5b140e17c62`
- `capture_000049_24f78f28209d49f59ab290981083c986`
- `capture_000050_2c3e7d77557e4b01a7a428f0a2fec340`
- `capture_000055_aad5892cc6e140eba1372bf61fe017ba`
- `capture_000060_87b9c8c9ec404888acc3a55d2e9d4aa4`
- `capture_000061_9ee2be9722564c83a3288406815dbdec`
- `capture_000075_13975686e5764bdea3d1795038feb886`
- `capture_000077_18e6c7bfbfd84a05bbf78fec816d95e6`
- `capture_000080_390f3b7bbaf6427b951e894507eba7ca`
- `capture_000081_26ae9597170b469f8add52d200d7e7ef`
- `capture_000083_22f6d26af97f491a873384f08d1570ce`
- `capture_000087_90aab7c067174806818e307cf101c533`
- `capture_000089_08cff39c2edc42139d2d4f38e72660c1`

전체 선정 전후 ID·모든 55장별 기여도·각 품질 기록: `.verification/coverage-repair/comparison.json`.

## Readiness와 품질 보호

실제 check_readiness 전체 경로를 재평가했다. Coverage 부족 원인은 해소됐지만 하단 모서리 미관측으로 ready=false다. 사용자 안내에는 전체 입력에서 관측이 없는 셀과 선별 때문에 기준에서 탈락한 셀을 구분해 1기준 행/열을 표시한다.

- 거리 다양성 1.0→1.0, Tilt 다양성 1.0→1.0; 거리/Tilt 분류별 장수 동일.
- Edge coverage 0.58333→0.66667.
- 중복 클러스터 반복 수 7→7. 35개가 모두 독립 장면이라는 뜻은 아니다.
- Blur/노출/검출 필터 그대로, 이번 90장은 모두 해당 선정 필터 통과. 불량 후보를 보존 과정에서 재투입하지 않는다.
- 선택 장의 Blur 점수 범위는 전후 동일 39.4907~577.7740(기준 최소 15), 밝기 범위도 동일 78.5863~132.7434(기준 25~230)다. Clipping 최대 비율은 0.04305→0.04709로 소폭 증가하지만 기존 한도 0.15를 충족한다. 모든 개별 품질 수치가 동일하거나 개선됐다는 의미는 아니다.
- 개선 Hold-out 구성: {'status': 'sufficient', 'test_frame_count': 7, 'test_corner_count': 309, 'test_coverage_pct': 50.0, 'test_pose_diversity': 0.7083333333333334, 'reasons': []}.
- 저장 PNG 90장 SHA-256을 기존 p2_manifest와 대조하여 모두 일치(검증 전후).

## 진단 계산: 미승인

Readiness를 우회하는 자동화 옵션은 추가하지 않았다. 별도 오프라인 진단 스크립트에서 기존 4개 모델 계산 함수를 전/후 Train에 각각 한 번 호출했다(28 Train/7 Test, seed 42). 기존 Test 평가 함수를 사용해 재학습 없이 Hold-out 오차도 계산했다. 수치 계산 성공과 품질 승인은 분리하며, 모든 결과는 DIAGNOSTIC_ONLY / NOT_APPROVED다. 추천·YAML 최종 승인·게시·차량 적용은 수행하지 않았다.

|모델|전 Train RMS(px)|후 Train RMS(px)|전 Test RMS(px)|후 Test RMS(px)|최대 K 절대 차이|
|---|---:|---:|---:|---:|---:|
|pinhole|22.492976|24.039733|23.715092|16.197103|100.200038|
|brown|3.960559|5.341176|2.994257|2.617966|265.262963|
|rational|3.430354|3.414496|1.922631|1.606795|12.608557|
|fisheye|3.454416|3.457195|2.067299|1.656225|4.669108|

4개 모델 모두 전후 수치 계산 성공. K/D는 동일하지 않다. Pinhole D는 고정 0으로 동일하다. 입력 선정과 Train/Test 구성 자체가 달라졌으므로 수치 동일성을 기대하면 안 된다. 전후 Test 집합도 달라 위 수치는 동일 Test 집합 A/B 비교나 품질 개선 증명이 아니다. Brown Train 오차 증가와 K 변화 등 민감도가 있으며 하단 미관측 및 Marker 치수 불확실성도 남아 있으므로 PASS로 승인하지 않는다. Fisheye는 기존 알고리즘에 따라 불안정 Train 프레임 2장을 자동 제외했으며 로그에 기록했다.

과거 bag 자동 실행에서는 계산이 0회였으므로 대조할 기존 자동 계산 결과가 없다. 이번 진단은 기존 함수 직접 호출이며 전체 CLI와의 K/D bitwise 일치는 별도로 검증하지 않았다. 동일 입력으로 불필요한 재계산은 하지 않았다.

## 테스트 및 변경 파일

최종 고유 pytest 결과: **74 PASS / 0 FAIL / 1 SKIP / 0 XFAIL**. 신규 회귀 4개 모두 PASS. SKIP은 기존 실제 과노출 이미지 fixture가 없는 테스트다. 실제 데이터 검증 스크립트는 pytest 수에 합산하지 않고 별도로 PASS를 기록했다. 전체 P1~P4 전수 회귀를 수행한 결과는 아니다.

- `calibration/image_selection.py`: 기존 미커밋 변경을 유지하고 Coverage 교체 보존 추가.
- `automation/live/readiness.py`: 미관측/선정 손실 셀을 구분하는 안내 추가.
- `tests/test_selection_coverage_preservation.py`: 결정성·공간 보존·거리/Tilt/중복 보호·구체 안내 회귀.
- 본 보고서 및 `.verification/coverage-repair/` 진단 증거.

테스트 범위: image_selection, pattern exposure, P2 readiness/gate/coaching, P1 automation runner. 로그 `tests-final.log`, JUnit `tests-final.xml`; 실제 검증 `real-verification.json`; 각 모델 전후 K/D·RMS `diagnostic-*.json`; bag 원본은 다시 열거나 변경하지 않았다. 긴 실행은 개별 로그와 JSON으로 남겼고 완료 파일은 재사용하도록 진단 스크립트를 구성했다.

남은 검증: 실제 보드 Marker 75mm 확인, 하단 셀과 좌우 하단 모서리 추가 촬영, 동일 외부 Test 데이터 검증, 실제 ROS/카메라 연결과 GitHub 게시. 외부 Push는 비활성화 상태다.

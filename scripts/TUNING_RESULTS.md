# Model Score 가중치 튜닝 실험 결과

`scripts/tune_model_score_weights.py`로 진행한 실험의 기록. 결론부터: **기본
가중치(`ModelScoreWeights()`)를 바꾸지 않기로 했다** - 아래에 왜 그런지 정직하게
적는다.

## 방법론

"진짜 카메라 모델을 아는" 합성 시나리오 4개를 만들었다:

| 시나리오 | 진짜 모델 | 특징 |
|---|---|---|
| `true_pinhole_noisy` | Pinhole | 왜곡 없음 + 픽셀 노이즈(σ=0.4px) |
| `true_extended_noisy` | Extended Pinhole | 뚜렷한 방사+접선 왜곡 + 노이즈 |
| `true_extended_mild_lowdata` | Extended Pinhole | **약한** 왜곡(k1=-0.06) + 적은 프레임(9장) - 의도적으로 어려운 경계 케이스 |
| `true_fisheye_noisy` | Fisheye | cv2.fisheye 왜곡 모델 + 넓은 화각 + 노이즈 |

각 시나리오를 서로 다른 시드로 여러 번 반복해 "3모델 계산 + Hold-out
validation"까지 실제로 돌리고 결과를 캐싱했다. 그 다음 `recommender.
compute_model_scores()`를 그대로 재사용해(재구현 없음) 다양한 가중치
후보(Dirichlet 분포로 무작위 샘플링, 5차원)가 "정답 모델을 골라내는 비율"을
채점했다.

**탐색용 시드와 최종 평가용 시드를 분리했다** (이 프로젝트 자체가 강조하는
"test intrinsic 재최적화 금지"와 같은 이유 - 탐색에 쓴 노이즈로만 평가하면
과적합인지 알 수 없다).

## 결과 (2회 독립 반복)

| | 탐색용 세트 정답률 | held-out 세트 정답률 |
|---|---|---|
| 1차 (train seed 1,2 / holdout seed 11) | 기본 62.5% → 튜닝 75.0% | 기본 62.5% = 튜닝 62.5% |
| 2차 (train seed 1,2,3 / holdout seed 11,12,13) | 기본 66.7% → 튜닝 75.0% | 기본 50.0% = 튜닝 50.0% |

**두 번 모두 같은 패턴**: 탐색용 세트에서는 튜닝된 가중치가 이겼지만,
held-out에서는 기본 가중치와 완전히 동률이었다. 이건 탐색이 실제로 더 나은
가중치를 찾은 게 아니라 **탐색용 캐시의 특정 노이즈 실현값에 과적합**됐다는
뜻이다 (Dirichlet 무작위 탐색을 5차원 공간에서 8~12개 샘플로 한 것이므로,
사실 놀라운 결과는 아니다 - 있을 법한 일이다).

## 결론

**기본 가중치를 그대로 유지한다.** "탐색에서 이겼다"는 이유만으로 프로덕션
기본값을 바꾸면, 실제로는 아무 근거 없는 변경을 "튜닝했다"고 포장하는
꼴이 된다. 이건 이 프로젝트가 설계 문서 3.1/3.3번에서 반복해서 경고하는
바로 그 함정(테스트 데이터에 대한 과적합)이다.

## 발견한 진짜 문제 (가중치와 무관하게 유용한 정보)

`true_extended_mild_lowdata`(왜곡이 약하고 데이터가 적은 경계 케이스)에서는
**기본 가중치든 튜닝된 가중치든 상관없이** 자주 Fisheye를 잘못 고른다
(6번 중 5번 실패).

**더 중요한 발견**: `true_extended_noisy`처럼 겉보기엔 "뚜렷한 왜곡이 있는
쉬운 케이스"로 설계했던 시나리오조차, 노이즈가 있는 3개 시드 중 2개에서
Fisheye로 잘못 분류됐다 (`tests/test_recommender_accuracy.py`에서 재현·고정).

원인으로 추정되는 것: 이 프로젝트의 합성 카메라는 화각이 그렇게 넓지
않다(`angle_scale=0.5`) - Fisheye(Kannala-Brandt) 모델의 진짜 차별점은
**아주 넓은 화각**에서 원근 투영이 근본적으로 깨지는 지점에서 드러나는데,
제한된 화각에서는 Brown-Conrady(Extended Pinhole)의 방사왜곡 항들이
Kannala-Brandt 곡선을 상당히 잘 근사해버린다. 게다가 Fisheye는 자유도가
4개(Extended Pinhole은 5개)라 복잡도 페널티도 근소하게 유리하다.

**이건 가중치 문제가 아니라 근본적인 모델 식별성(identifiability) 문제다.**
화각이 넓은 데이터를 더 모으지 않는 한, 어떤 가중치 조합을 써도 이 모호함
자체는 해결되지 않는다.

## 사용자에게 전달할 실무 조언

- 데이터셋이 작고(10장 이하) 왜곡이 미묘하면, 추천 시스템의 모델 선택을
  곧이곧대로 믿기보다 Model Comparison 표를 직접 보고 판단하는 게 낫다.
- 진짜 fisheye 렌즈(120°+ 광각)가 아니라면, Extended Pinhole과 Fisheye
  중 애매하게 갈리는 결과가 나와도 놀랄 일이 아니다 - 오히려 화각이 좁아서
  구분이 원래 어려운 상황일 가능성이 높다. 이런 경우 보드를 화면 최외곽
  가까이까지 채우는 사진을 더 찍으면(Coverage Map 활용) 구분력이 좋아진다.

## 재현 방법

```bash
# 시드별로 나눠서 캐시 생성 (시간이 오래 걸려 한 번에 다 돌리기보다 나눠서 실행 권장)
python scripts/tune_model_score_weights.py --build-cache 1 --out /tmp/cache_train1.pkl
python scripts/tune_model_score_weights.py --build-cache 2 --out /tmp/cache_train2.pkl
python scripts/tune_model_score_weights.py --build-cache 11 --out /tmp/cache_holdout11.pkl

# 탐색 + held-out 검증
python scripts/tune_model_score_weights.py \
  --search /tmp/cache_train1.pkl /tmp/cache_train2.pkl \
  --holdout /tmp/cache_holdout11.pkl \
  --n-candidates 8000
```

더 많은 시나리오/시드/후보 개수로 다시 시도해보고 싶다면 `SCENARIOS`,
`SEEDS`, `HOLDOUT_SEEDS`를 수정하면 된다. 계산 비용 대부분은 캐시 빌드
단계(실제 3모델 캘리브레이션)에 있고, 탐색 자체(캐시 재사용)는 몇천 개
후보를 뽑아도 몇 초면 끝난다.

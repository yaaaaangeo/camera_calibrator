"""
camera_calibrator.calibration.kfold
=======================================

설계 문서 18/19번 - K-Fold / Repeated K-Fold Cross Validation.

기존 Hold-out(validate_holdout, 1회 분할)은 "운 좋게/나쁘게 뽑힌 test set"에
결과가 좌우될 수 있다는 근본적인 한계가 있다 - 특히 데이터셋이 작을수록
심하다. K-Fold는 데이터를 k개 조각으로 나눠 각 조각이 정확히 한 번씩
test가 되게 해서(나머지 k-1개는 train), k번의 독립적인 평가를 평균낸다.
Repeated K-Fold는 이 k-분할 자체를 여러 번(다른 seed로) 반복해서, "이번
분할이 우연히 좋았다/나빴다"는 효과까지 평균으로 눌러준다.

leakage 안전성: 각 fold의 test 조각은 그 fold의 train(나머지 k-1조각)과
절대 겹치지 않는다(split_k_folds가 프레임을 겹침 없이 분배) - 그리고 각
fold 평가는 validate_holdout()을 그대로 재사용하므로, 설계 문서 9번의
Train/Test leakage 방지 원칙(test로 파라미터를 수정하지 않음)이 여기서도
그대로 지켜진다.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import random

import numpy as np

from calibration.cache import (
    KFOLD_VALIDATION_CACHE,
    ValidationCache,
    camera_fingerprint,
    dataset_fingerprint,
    pattern_fingerprint,
)
from calibration.models.common import MIN_FRAMES_REQUIRED, infer_image_size
from calibration.performance import resolve_worker_count
from calibration.types import (
    CameraConfig,
    CameraModelType,
    Dataset,
    Frame,
    KFoldResult,
    PatternConfig,
    RepeatedKFoldResult,
)


def _fold_cache_key(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    model: CameraModelType,
    train_ids: list[str],
    test_ids: list[str],
) -> tuple:
    return (
        "kfold_validation",
        dataset_fingerprint(dataset),
        camera_fingerprint(camera_config),
        pattern_fingerprint(pattern_config),
        model.value,
        tuple(train_ids),
        tuple(test_ids),
    )


def _validate_fold(args: tuple):
    (
        dataset,
        camera_config,
        pattern_config,
        model,
        train_ids,
        test_ids,
        cache,
    ) = args
    from calibration.validation import validate_holdout  # 순환 참조 회피

    key = _fold_cache_key(
        dataset, camera_config, pattern_config, model, train_ids, test_ids
    )
    cached = cache.get(key) if cache is not None else None
    if cached is not None:
        return cached

    result = validate_holdout(
        dataset, camera_config, pattern_config, model, train_ids, test_ids,
    )
    if cache is not None:
        cache.set(key, result)
    return result


def split_k_folds(
    dataset: Dataset,
    camera_config: CameraConfig,
    k: int = 5,
    seed: int = 42,
) -> list[list[str]]:
    """전체 사용 가능 프레임을 겹침 없이 k개 폴드로 나눈다 (stratified).

    validation.split_train_test()와 같은 stratum 정의(_stratum_key: 위치 x
    거리)를 재사용해서, 폴드마다 "전부 중앙에서 가까이 찍은 사진만 모임"
    같은 편향이 생기지 않게 한다 - 각 stratum 안에서 셔플한 뒤 폴드에
    라운드로빈으로 나눠 담는다.
    """
    from calibration.validation import _stratum_key  # 순환 참조 회피 - 지연 import

    usable = [
        f for f in dataset.enabled_frames
        if f.detection and f.detection.success and f.detection.num_corners >= 4
    ]
    if not usable or k < 2:
        return [[] for _ in range(max(k, 1))]

    image_size = infer_image_size(dataset, camera_config)
    area_ratios = [
        f.detection.board_area_ratio for f in usable if f.detection.board_area_ratio is not None
    ]
    median_ratio = float(np.median(area_ratios)) if area_ratios else 0.0

    strata: dict[str, list[Frame]] = {}
    for f in usable:
        key = _stratum_key(f, image_size, median_ratio)
        strata.setdefault(key, []).append(f)

    rng = random.Random(seed)
    folds: list[list[str]] = [[] for _ in range(k)]
    fold_cursor = 0  # 여러 stratum에 걸쳐 이어지는 전역 라운드로빈 커서.
    # stratum마다 커서를 0부터 다시 시작하면, 작은 stratum이 많을 때 앞쪽
    # 폴드(특히 fold 0)에 나머지(leftover) 조각들이 쏠려서 폴드 크기가
    # 불균등해진다 - 전역 커서를 이어가야 폴드 전체 크기가 고르게 맞는다.
    for frames in strata.values():
        frames = frames[:]
        rng.shuffle(frames)
        for f in frames:
            folds[fold_cursor % k].append(f.image_info.image_id)
            fold_cursor += 1

    return folds


def _stat(values: list[float]) -> tuple[float | None, float | None, float | None, float | None]:
    """values 리스트 하나에서 (mean, std, min, max)를 뽑는 공용 헬퍼.

    표본이 없으면 전부 None("missing" - 0이나 임의의 숫자로 채우지 않는다),
    표본이 1개면 std는 ddof=1 표준편차를 정의할 수 없으므로 None.
    """
    if not values:
        return None, None, None, None
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else None
    return mean, std, float(np.min(values)), float(np.max(values))


def _collect_multi_metrics(
    fold_results: list,
) -> dict:
    """K-Fold/Repeated K-Fold 공용 - fold ValidationResult 목록에서 Test
    RMS/P95/Edge RMS/Test Straightness를 모아 mean/std/min/max로 aggregate.

    반드시 지켜야 하는 두 가지 fairness 규칙:
      - edge_rms는 ValidationResult.edge_rms(Hold-out **test** edge)만 쓴다.
        Train regional_error fallback은 애초에 이 필드에 들어가지 않으므로
        여기서 추가로 걸러낼 것도 없다 - 그대로 신뢰할 수 있다.
      - straightness는 straightness_source == "test"인 fold만 포함한다.
        "train_fallback"은 diagnostic으로만 남기고 aggregate에서는 제외
        (missing으로 취급 - 임의의 숫자/worst-penalty로 대체하지 않는다).
    """
    rmses = [vr.test_rms for vr in fold_results if vr.success and vr.test_rms is not None]
    p95s = [
        vr.test_residual_stats.p95 for vr in fold_results
        if vr.success and vr.test_residual_stats and vr.test_residual_stats.p95 is not None
    ]
    edge_rmses = [vr.edge_rms for vr in fold_results if vr.success and vr.edge_rms is not None]
    straightness = [
        vr.straightness_residual for vr in fold_results
        if vr.success and vr.straightness_source == "test" and vr.straightness_residual is not None
    ]

    rms_mean, rms_std, rms_min, rms_max = _stat(rmses)
    p95_mean, p95_std, _p95_min, _p95_max = _stat(p95s)
    edge_mean, edge_std, edge_min, edge_max = _stat(edge_rmses)
    straight_mean, straight_std, straight_min, straight_max = _stat(straightness)

    return {
        "mean_test_rms": rms_mean, "std_test_rms": rms_std,
        "min_test_rms": rms_min, "max_test_rms": rms_max,
        "mean_test_p95": p95_mean, "std_test_p95": p95_std,
        "mean_edge_rms": edge_mean, "std_edge_rms": edge_std,
        "min_edge_rms": edge_min, "max_edge_rms": edge_max,
        "mean_test_straightness": straight_mean, "std_test_straightness": straight_std,
        "min_test_straightness": straight_min, "max_test_straightness": straight_max,
        "n_straightness_folds": len(straightness),
        "n_successful": len(rmses),
    }


def _fold_status(vr) -> str:
    """fold 하나의 상태를 "fully_successful" / "partial_success" / "failed"로 분류.

    - fully_successful: validate_holdout 성공 + 실패한 test frame이 0개
      (Fisheye pose fallback을 포함해 모든 test frame이 정상 평가됨).
    - partial_success: 성공했지만 일부 test frame의 pose 추정은 끝내 실패함
      (해당 frame들은 failed_test_frame_ids/failed_test_frame_reasons에 남음).
    - failed: validate_holdout 자체가 실패(모든 test frame 실패, 또는 train
      프레임 부족 등).
    """
    if not vr.success:
        return "failed"
    if vr.failed_test_frame_ids:
        return "partial_success"
    return "fully_successful"


def compute_kfold_validation(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    model: CameraModelType,
    k: int = 5,
    seed: int = 42,
    n_jobs: int = 1,
    cache: ValidationCache | None = KFOLD_VALIDATION_CACHE,
) -> KFoldResult:
    """단일 K-Fold 실행. 각 fold를 정확히 한 번 test로 써서 validate_holdout()으로
    평가하고, fold별 test_rms/test_residual_stats.p95를 모아 mean/std/min/max로
    요약한다.
    """
    folds = split_k_folds(dataset, camera_config, k=k, seed=seed)
    tasks = []

    for i in range(k):
        test_ids = folds[i]
        train_ids = [fid for j in range(k) if j != i for fid in folds[j]]
        if len(train_ids) < MIN_FRAMES_REQUIRED or not test_ids:
            continue
        tasks.append(
            (
                dataset,
                camera_config,
                pattern_config,
                model,
                train_ids,
                test_ids,
                cache,
            )
        )

    workers = resolve_worker_count(n_jobs, len(tasks))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            fold_results = list(executor.map(_validate_fold, tasks))
    else:
        fold_results = [_validate_fold(task) for task in tasks]

    metrics = _collect_multi_metrics(fold_results)

    return KFoldResult(
        k=k,
        fold_validation_results=fold_results,
        mean_test_rms=metrics["mean_test_rms"], std_test_rms=metrics["std_test_rms"],
        min_test_rms=metrics["min_test_rms"], max_test_rms=metrics["max_test_rms"],
        mean_test_p95=metrics["mean_test_p95"], std_test_p95=metrics["std_test_p95"],
        mean_edge_rms=metrics["mean_edge_rms"], std_edge_rms=metrics["std_edge_rms"],
        min_edge_rms=metrics["min_edge_rms"], max_edge_rms=metrics["max_edge_rms"],
        mean_test_straightness=metrics["mean_test_straightness"],
        std_test_straightness=metrics["std_test_straightness"],
        min_test_straightness=metrics["min_test_straightness"],
        max_test_straightness=metrics["max_test_straightness"],
        n_straightness_folds=metrics["n_straightness_folds"],
        n_successful_folds=metrics["n_successful"],
    )


def compute_repeated_kfold(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    model: CameraModelType,
    k: int = 5,
    n_repeats: int = 5,
    base_seed: int = 42,
    n_jobs: int = 1,
    cache: ValidationCache | None = KFOLD_VALIDATION_CACHE,
) -> RepeatedKFoldResult:
    """설계 문서 19번 - K-Fold를 n_repeats번(각기 다른 seed로 다시 분할) 반복.

    "5-fold x 5회 반복" = fold 25개의 test_rms를 전부 모아 하나의 분포로 보고
    mean/std/min/max를 낸다 - 개별 KFoldResult(1회 분할 기준 평균)도 참고용으로
    보존한다.
    """
    def _run_repeat(repeat_index: int) -> KFoldResult:
        return compute_kfold_validation(
            dataset, camera_config, pattern_config, model,
            k=k, seed=base_seed + repeat_index,
            n_jobs=1, cache=cache,
        )

    repeat_indices = list(range(n_repeats))
    workers = resolve_worker_count(n_jobs, len(repeat_indices))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            kfold_results = list(executor.map(_run_repeat, repeat_indices))
    else:
        kfold_results = [_run_repeat(r) for r in repeat_indices]

    all_fold_results = [vr for kf in kfold_results for vr in kf.fold_validation_results]
    metrics = _collect_multi_metrics(all_fold_results)

    total_folds = k * n_repeats
    fully_successful = sum(1 for vr in all_fold_results if _fold_status(vr) == "fully_successful")
    partial_success = sum(1 for vr in all_fold_results if _fold_status(vr) == "partial_success")
    # "failed"에는 validate_holdout이 명시적으로 실패라고 보고한 fold뿐 아니라,
    # (train 프레임 부족 등으로) 애초에 실행조차 안 돼 fold_validation_results에
    # 아예 나타나지 않은 fold도 포함한다 - total_folds - (실제 실행된 fold 수)
    # 만큼을 여기에 더해야 fully_successful+partial+failed == total_folds가 된다
    # (조용히 사라지는 fold 없이, 25 folds 중 몇 개가 진짜로 정상 평가됐는지를
    # 그대로 볼 수 있게 하기 위함).
    executed_failed = sum(1 for vr in all_fold_results if _fold_status(vr) == "failed")
    missing_folds = max(0, total_folds - len(all_fold_results))
    failed_folds = executed_failed + missing_folds

    return RepeatedKFoldResult(
        k=k,
        n_repeats=n_repeats,
        kfold_results=kfold_results,
        mean_test_rms=metrics["mean_test_rms"], std_test_rms=metrics["std_test_rms"],
        min_test_rms=metrics["min_test_rms"], max_test_rms=metrics["max_test_rms"],
        mean_test_p95=metrics["mean_test_p95"], std_test_p95=metrics["std_test_p95"],
        mean_edge_rms=metrics["mean_edge_rms"], std_edge_rms=metrics["std_edge_rms"],
        min_edge_rms=metrics["min_edge_rms"], max_edge_rms=metrics["max_edge_rms"],
        mean_test_straightness=metrics["mean_test_straightness"],
        std_test_straightness=metrics["std_test_straightness"],
        min_test_straightness=metrics["min_test_straightness"],
        max_test_straightness=metrics["max_test_straightness"],
        n_straightness_folds=metrics["n_straightness_folds"],
        n_successful_runs=metrics["n_successful"],
        total_folds=total_folds,
        fully_successful_folds=fully_successful,
        partial_success_folds=partial_success,
        failed_folds=failed_folds,
    )


def format_kfold_result(result: KFoldResult) -> str:
    """설계 문서 18번 출력 형식(+ 논문용 Multi-Metric 확장: Edge RMS/Straightness).

        5-Fold Cross Validation (성공 4/5 fold)
        Test RMSE: 0.340px (mean) / 0.021px (std) / 0.310px~0.365px (min~max)
        Test P95:  0.612px (mean) / 0.045px (std)
        Test Edge RMS: 0.410px (mean) / 0.030px (std) / 0.370px~0.460px (min~max)
        Test Straightness: 0.055px (mean) / 0.008px (std) [test 프레임 기준 4/4 fold]
    """
    def fmt(v: float | None) -> str:
        return f"{v:.3f}px" if v is not None else "N/A"

    lines = [f"{result.k}-Fold Cross Validation (성공 {result.n_successful_folds}/{result.k} fold)"]
    lines.append(
        f"Test RMSE: {fmt(result.mean_test_rms)} (mean) / {fmt(result.std_test_rms)} (std) / "
        f"{fmt(result.min_test_rms)}~{fmt(result.max_test_rms)} (min~max)"
    )
    lines.append(f"Test P95:  {fmt(result.mean_test_p95)} (mean) / {fmt(result.std_test_p95)} (std)")
    lines.append(
        f"Test Edge RMS: {fmt(result.mean_edge_rms)} (mean) / {fmt(result.std_edge_rms)} (std) / "
        f"{fmt(result.min_edge_rms)}~{fmt(result.max_edge_rms)} (min~max)"
    )
    if result.n_straightness_folds > 0:
        lines.append(
            f"Test Straightness: {fmt(result.mean_test_straightness)} (mean) / "
            f"{fmt(result.std_test_straightness)} (std) "
            f"[test 프레임 기준 {result.n_straightness_folds}/{result.k} fold]"
        )
    else:
        # train_fallback 값을 여기 섞지 않는다 - 계산 불가 상태를 그대로 보여준다.
        lines.append("Test Straightness: N/A (test 프레임 기준으로 계산된 fold가 없음)")
    return "\n".join(lines)


def format_repeated_kfold_result(result: RepeatedKFoldResult) -> str:
    """설계 문서 19번 출력 형식(+ 논문용 Multi-Metric 확장).

        Repeated 5-Fold x 5
        Successful folds: 25/25 (fully 23, partial 2, failed 0)
        Test RMS: Mean = 0.345px, Std = 0.024px, Min = 0.301px, Max = 0.392px
        Test P95: Mean = 0.612px, Std = 0.045px
        Test Edge RMS: Mean = 0.410px, Std = 0.030px, Min = 0.370px, Max = 0.460px
        Test Straightness: Mean = 0.055px, Std = 0.008px [test 프레임 기준 20/25 fold]
    """
    def fmt(v: float | None) -> str:
        return f"{v:.3f}px" if v is not None else "N/A"

    lines = [
        f"Repeated {result.k}-Fold x {result.n_repeats}",
        f"Successful folds: {result.n_successful_runs}/{result.total_folds} "
        f"(fully {result.fully_successful_folds}, partial {result.partial_success_folds}, "
        f"failed {result.failed_folds})",
        f"Test RMS: Mean = {fmt(result.mean_test_rms)}, Std = {fmt(result.std_test_rms)}, "
        f"Min = {fmt(result.min_test_rms)}, Max = {fmt(result.max_test_rms)}",
        f"Test P95: Mean = {fmt(result.mean_test_p95)}, Std = {fmt(result.std_test_p95)}",
        f"Test Edge RMS: Mean = {fmt(result.mean_edge_rms)}, Std = {fmt(result.std_edge_rms)}, "
        f"Min = {fmt(result.min_edge_rms)}, Max = {fmt(result.max_edge_rms)}",
    ]
    if result.n_straightness_folds > 0:
        lines.append(
            f"Test Straightness: Mean = {fmt(result.mean_test_straightness)}, "
            f"Std = {fmt(result.std_test_straightness)} "
            f"[test 프레임 기준 {result.n_straightness_folds}/{result.total_folds} fold]"
        )
    else:
        lines.append("Test Straightness: N/A (test 프레임 기준으로 계산된 fold가 없음)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 논문용 - Brown/Rational/Fisheye를 같은 fold partition으로 비교
# ---------------------------------------------------------------------------

def run_repeated_kfold_all_models(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    models: list[CameraModelType] | None = None,
    k: int = 5,
    n_repeats: int = 5,
    base_seed: int = 42,
    n_jobs: int = 1,
    cache: ValidationCache | None = KFOLD_VALIDATION_CACHE,
) -> dict[CameraModelType, RepeatedKFoldResult]:
    """Brown-Conrady/Rational(Extended Pinhole)/Fisheye 세 모델을 정확히 같은
    fold partition으로 비교하기 위한 편의 함수.

    "같은 fold partition"이 보장되는 이유(모델별로 별도 random split을 만들지
    않는다는 원칙): split_k_folds(dataset, camera_config, k, seed)는 model을
    인자로 받지 않는다 - 즉 어떤 모델을 평가하든 같은 dataset/camera_config/
    k/seed 조합이면 항상 정확히 같은 프레임 id 목록으로 분할된다. 이 함수는
    세 모델 각각에 대해 compute_repeated_kfold를 같은 (dataset, camera_config,
    k, base_seed) 인자로 호출할 뿐이라, 모델 간 fold 배정 fairness는 이미
    split_k_folds의 결정론적 동작으로 구조적으로 보장된다(별도 조율 로직이
    필요 없다) - test_repeated_kfold_all_models_share_fold_partition()이
    이 사실을 회귀 테스트로 고정한다.
    """
    target_models = models or [
        CameraModelType.BROWN_CONRADY,
        CameraModelType.EXTENDED_PINHOLE,
        CameraModelType.FISHEYE,
    ]
    return {
        model: compute_repeated_kfold(
            dataset, camera_config, pattern_config, model,
            k=k, n_repeats=n_repeats, base_seed=base_seed, n_jobs=n_jobs, cache=cache,
        )
        for model in target_models
    }


_MODEL_COMPARISON_LABELS = {
    CameraModelType.PINHOLE: "Pinhole",
    CameraModelType.BROWN_CONRADY: "Brown-Conrady",
    CameraModelType.EXTENDED_PINHOLE: "Rational",
    CameraModelType.FISHEYE: "Fisheye",
}


def format_model_comparison_table(results: dict[CameraModelType, RepeatedKFoldResult]) -> str:
    """Brown/Rational/Fisheye Repeated K-Fold 결과를 한 표로 요약.

    threshold/weight로 "이긴 모델"을 판정하지 않는다 - raw mean ± std를
    모델별로 나란히 보여줄 뿐이다(어느 모델이 반복 unseen-data에서 더
    일관되게 좋은지는 이 표를 보는 사람이 직접 비교 판단하게 둔다).

        Model          Test RMS         Test P95         Edge RMS         Straightness      Folds
        Brown-Conrady  0.345 ± 0.024px  0.612 ± 0.045px  0.410 ± 0.030px  0.055 ± 0.008px   25/25
        Rational       0.360 ± 0.041px  0.655 ± 0.060px  0.430 ± 0.038px  0.061 ± 0.011px   25/25
        Fisheye        0.352 ± 0.029px  0.630 ± 0.050px  0.415 ± 0.033px  0.058 ± 0.009px   25/25
    """
    def pm(mean: float | None, std: float | None) -> str:
        if mean is None:
            return "N/A"
        if std is None:
            return f"{mean:.3f}px"
        return f"{mean:.3f} ± {std:.3f}px"

    header = f"{'Model':<15}{'Test RMS':<18}{'Test P95':<18}{'Edge RMS':<18}{'Straightness':<20}{'Folds':<10}"
    lines = [header]
    for model, result in results.items():
        label = _MODEL_COMPARISON_LABELS.get(model, model.value)
        straight = (
            pm(result.mean_test_straightness, result.std_test_straightness)
            if result.n_straightness_folds > 0 else "N/A"
        )
        lines.append(
            f"{label:<15}"
            f"{pm(result.mean_test_rms, result.std_test_rms):<18}"
            f"{pm(result.mean_test_p95, result.std_test_p95):<18}"
            f"{pm(result.mean_edge_rms, result.std_edge_rms):<18}"
            f"{straight:<20}"
            f"{f'{result.n_successful_runs}/{result.total_folds}':<10}"
        )
    return "\n".join(lines)


def model_comparison_rows(results: dict[CameraModelType, RepeatedKFoldResult]) -> list[dict]:
    """export/report/UI가 표를 직접 그릴 때 쓰기 좋은 구조화된 형태 -
    format_model_comparison_table()과 같은 raw 값을 dict 리스트로 반환한다
    (문자열 파싱 없이 JSON/HTML/Qt 테이블에 바로 꽂아 쓸 수 있게).
    """
    rows = []
    for model, result in results.items():
        rows.append({
            "model": model.value,
            "mean_test_rms": result.mean_test_rms, "std_test_rms": result.std_test_rms,
            "mean_test_p95": result.mean_test_p95, "std_test_p95": result.std_test_p95,
            "mean_edge_rms": result.mean_edge_rms, "std_edge_rms": result.std_edge_rms,
            "mean_test_straightness": (
                result.mean_test_straightness if result.n_straightness_folds > 0 else None
            ),
            "std_test_straightness": (
                result.std_test_straightness if result.n_straightness_folds > 0 else None
            ),
            "n_straightness_folds": result.n_straightness_folds,
            "successful_folds": result.n_successful_runs,
            "total_folds": result.total_folds,
            "fully_successful_folds": result.fully_successful_folds,
            "partial_success_folds": result.partial_success_folds,
            "failed_folds": result.failed_folds,
        })
    return rows

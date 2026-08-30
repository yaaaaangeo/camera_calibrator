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

    test_rmses = [vr.test_rms for vr in fold_results if vr.success and vr.test_rms is not None]
    test_p95s = [
        vr.test_residual_stats.p95 for vr in fold_results
        if vr.success and vr.test_residual_stats and vr.test_residual_stats.p95 is not None
    ]

    return KFoldResult(
        k=k,
        fold_validation_results=fold_results,
        mean_test_rms=float(np.mean(test_rmses)) if test_rmses else None,
        std_test_rms=float(np.std(test_rmses, ddof=1)) if len(test_rmses) > 1 else None,
        min_test_rms=float(np.min(test_rmses)) if test_rmses else None,
        max_test_rms=float(np.max(test_rmses)) if test_rmses else None,
        mean_test_p95=float(np.mean(test_p95s)) if test_p95s else None,
        std_test_p95=float(np.std(test_p95s, ddof=1)) if len(test_p95s) > 1 else None,
        n_successful_folds=len(test_rmses),
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

    all_rmses: list[float] = []
    all_p95s: list[float] = []

    for kf in kfold_results:
        for vr in kf.fold_validation_results:
            if vr.success and vr.test_rms is not None:
                all_rmses.append(vr.test_rms)
            if vr.success and vr.test_residual_stats and vr.test_residual_stats.p95 is not None:
                all_p95s.append(vr.test_residual_stats.p95)

    return RepeatedKFoldResult(
        k=k,
        n_repeats=n_repeats,
        kfold_results=kfold_results,
        mean_test_rms=float(np.mean(all_rmses)) if all_rmses else None,
        std_test_rms=float(np.std(all_rmses, ddof=1)) if len(all_rmses) > 1 else None,
        min_test_rms=float(np.min(all_rmses)) if all_rmses else None,
        max_test_rms=float(np.max(all_rmses)) if all_rmses else None,
        mean_test_p95=float(np.mean(all_p95s)) if all_p95s else None,
        std_test_p95=float(np.std(all_p95s, ddof=1)) if len(all_p95s) > 1 else None,
        n_successful_runs=len(all_rmses),
    )


def format_kfold_result(result: KFoldResult) -> str:
    """설계 문서 18번 출력 형식.

        5-Fold Cross Validation (성공 4/5 fold)
        Test RMSE: 0.340px (mean) / 0.021px (std) / 0.310px~0.365px (min~max)
        Test P95:  0.612px (mean) / 0.045px (std)
    """
    def fmt(v: float | None) -> str:
        return f"{v:.3f}px" if v is not None else "N/A"

    lines = [f"{result.k}-Fold Cross Validation (성공 {result.n_successful_folds}/{result.k} fold)"]
    lines.append(
        f"Test RMSE: {fmt(result.mean_test_rms)} (mean) / {fmt(result.std_test_rms)} (std) / "
        f"{fmt(result.min_test_rms)}~{fmt(result.max_test_rms)} (min~max)"
    )
    lines.append(f"Test P95:  {fmt(result.mean_test_p95)} (mean) / {fmt(result.std_test_p95)} (std)")
    return "\n".join(lines)


def format_repeated_kfold_result(result: RepeatedKFoldResult) -> str:
    """설계 문서 19번 출력 형식.

        Repeated 5-Fold x 5 (총 25 fold 중 23개 성공)
        Test RMSE: Mean = 0.345px, Std = 0.024px, Min = 0.301px, Max = 0.392px
    """
    def fmt(v: float | None) -> str:
        return f"{v:.3f}px" if v is not None else "N/A"

    total_folds = result.k * result.n_repeats
    lines = [
        f"Repeated {result.k}-Fold x {result.n_repeats} "
        f"(총 {total_folds} fold 중 {result.n_successful_runs}개 성공)",
        f"Test RMSE: Mean = {fmt(result.mean_test_rms)}, Std = {fmt(result.std_test_rms)}, "
        f"Min = {fmt(result.min_test_rms)}, Max = {fmt(result.max_test_rms)}",
        f"Test P95:  Mean = {fmt(result.mean_test_p95)}, Std = {fmt(result.std_test_p95)}",
    ]
    return "\n".join(lines)

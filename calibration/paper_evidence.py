"""
camera_calibrator.calibration.paper_evidence
================================================

논문 "Beyond Reprojection RMS: Multi-Metric Evaluation of Camera Calibration
for Autonomous Driving Perception"의 quantitative evidence를 재현 가능한
CSV/JSON/텍스트 형태로 정리하는 모듈 - Paper Evidence 단계.

핵심 원칙 (요청 그대로, 반드시 지킨다):
  - 이 모듈의 어떤 함수도 calibration 결과/metric 계산 방식/Repeated K-Fold
    fold split/Hold-out 정책/Fisheye pose fallback/Straightness TEST source
    정책/Recommendation 로직을 바꾸지 않는다. 전부 이미 계산되어 있는
    ValidationResult/RepeatedKFoldResult/ParameterUncertainty를 읽어서 다른
    모양(row/summary/ranking/report)으로 재배열할 뿐이다(순수 파생 함수).
  - 25개 fold(K=5 x Repeats=5)는 서로 독립인 표본이 아니다 - repeat마다
    train set이 반복해서 겹친다. 그래서 여기서는 paired t-test 같은 통계적
    유의성 검정을 자동으로 만들지 않는다. 대신 (a) descriptive statistics
    (mean/std/median/min/max)와 (b) 같은 fold에서 어느 모델이 더 낮은
    값을 냈는지 세는 raw pairwise win count만 제공한다 - "몇 개 fold에서
    이겼는지"를 셀 뿐 통계적으로 유의하다고 주장하지 않는다.
  - Single Hold-out vs Repeated CV의 ranking 변화("split sensitivity")는
    버그도 실패 판정도 아닌 diagnostic이다 - recommendation에 자동으로
    반영되지 않는다. paper report/analysis 용도로만 쓰인다.
  - "어느 모델이 좋다/나쁘다"는 결론 문장을 자동 생성하지 않는다 - raw
    quantitative evidence와 계산된 값의 나열만 만든다.
  - ROI(evaluation region of interest)는 기본값이 Full Image이고, 사용자가
    명시적으로 좁히지 않는 한 어떤 기존 metric도 바뀌지 않는다. error가
    큰 영역을 사후에 몰래 제외하는 방식(cherry-picking)은 이 모듈 어디에도
    없다 - ROI를 바꾸면 그 사실과 정확한 pixel/normalized 좌표가 함께
    report/export에 남는다.
"""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from calibration.kfold import _fold_status
from calibration.types import (
    CameraConfig,
    CameraModelType,
    Dataset,
    KFoldResult,
    ParameterUncertainty,
    PatternConfig,
    RepeatedKFoldResult,
    SpatialErrorMap,
    ValidationResult,
)

# Test RMS/P95/Edge RMS/Straightness 전부 "작을수록 좋은" error metric이다 -
# pairwise win count/descriptive summary가 공통으로 순회하는 목록.
PAPER_FOLD_METRICS = ["test_rms", "test_p95", "test_edge_rms", "test_straightness"]


def _fmt(v: float | None, ndigits: int = 3) -> str:
    return f"{v:.{ndigits}f}" if v is not None else "N/A"


def _fmt_score(v: float | None) -> str:
    return f"{v:.1f}/100" if v is not None else "N/A"


# ---------------------------------------------------------------------------
# 1) Repeated K-Fold raw fold-level rows / repeat-level summary
# ---------------------------------------------------------------------------

def kfold_raw_rows(model: CameraModelType, result: RepeatedKFoldResult) -> list[dict]:
    """RepeatedKFoldResult 안에 이미 보존되어 있는 kfold_results/fold_validation_results
    를 model/repeat_index/fold_index/seed가 붙은 flat row로 펼친다 - 25개
    (K=5 x Repeats=5) fold의 raw evidence를 하나도 버리지 않는다.

    fold_index는 각 KFoldResult.fold_validation_results 리스트 안에서의
    위치(0부터)다 - split_k_folds()/compute_kfold_validation()의 fold 스킵
    조건(train 프레임 부족 등)이 model과 무관하게 동일한 dataset/k/seed에서
    결정되므로, 같은 (repeat_index, fold_index)는 Brown-Conrady/Rational/
    Fisheye 사이에서 항상 같은 fold(같은 test_frame_ids)를 가리킨다 -
    test_repeated_kfold_all_models_share_fold_partition()이 이미 이 사실을
    검증한다.
    """
    rows: list[dict] = []
    for repeat_index, kf in enumerate(result.kfold_results):
        for fold_index, vr in enumerate(kf.fold_validation_results):
            straightness = vr.straightness_residual if vr.straightness_source == "test" else None
            p95 = vr.test_residual_stats.p95 if vr.test_residual_stats else None
            rows.append({
                "model": model.value,
                "repeat_index": repeat_index,
                "fold_index": fold_index,
                "seed": kf.seed,
                "train_frame_count": len(vr.train_frame_ids),
                "test_frame_count": len(vr.test_frame_ids),
                "successful_test_frames": len(vr.per_frame_error),
                "failed_test_frames": len(vr.failed_test_frame_ids),
                "fold_status": _fold_status(vr),
                "test_rms": vr.test_rms,
                "test_p95": p95,
                "test_edge_rms": vr.edge_rms,
                "test_straightness": straightness,
                "straightness_source": vr.straightness_source,
            })
    return rows


def repeat_level_rows(model: CameraModelType, result: RepeatedKFoldResult) -> list[dict]:
    """5개 repeat 각각의 "5-fold 평균"(KFoldResult가 이미 계산해 둔
    mean_test_rms 등)을 그대로 꺼내 row로 만든다 - "repeat를 바꿔도 같은
    경향이 유지되는지"를 25-fold 전체 std와 별개로 볼 수 있게 한다.
    """
    rows: list[dict] = []
    for repeat_index, kf in enumerate(result.kfold_results):
        rows.append({
            "model": model.value,
            "repeat_index": repeat_index,
            "seed": kf.seed,
            "k": kf.k,
            "successful_folds": kf.n_successful_folds,
            "mean_test_rms": kf.mean_test_rms,
            "mean_test_p95": kf.mean_test_p95,
            "mean_test_edge_rms": kf.mean_edge_rms,
            "mean_test_straightness": kf.mean_test_straightness if kf.n_straightness_folds > 0 else None,
            "n_straightness_folds": kf.n_straightness_folds,
        })
    return rows


def descriptive_stats(values: list[float | None]) -> dict:
    """mean/std/median/min/max - None은 표본에서 제외(missing으로 취급,
    0이나 다른 값으로 대체하지 않는다)."""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "n": len(clean),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(clean) > 1 else None,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def model_metric_descriptive_summary(rows_by_model: dict[CameraModelType, list[dict]]) -> list[dict]:
    """모델별 mean/std/median/min/max(+ 25 fold 중 성공한 개수)를
    PAPER_FOLD_METRICS 각각에 대해 계산한다."""
    out: list[dict] = []
    for model, rows in rows_by_model.items():
        successful = sum(1 for r in rows if r["fold_status"] in ("fully_successful", "partial_success"))
        entry: dict = {
            "model": model.value,
            "total_folds": len(rows),
            "successful_folds": successful,
        }
        for metric in PAPER_FOLD_METRICS:
            stats = descriptive_stats([r[metric] for r in rows])
            for key, val in stats.items():
                entry[f"{metric}_{key}"] = val
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# 2) Pairwise fold win counts (raw comparison, no significance test)
# ---------------------------------------------------------------------------

def pairwise_win_counts(rows_by_model: dict[CameraModelType, list[dict]], metric: str) -> list[dict]:
    """metric(예: "test_rms")에서 모델 쌍마다 같은 (repeat_index, fold_index)
    fold의 값을 비교해 승/패/동점을 센다. 값이 더 작은 쪽이 win(전부 error
    metric - 작을수록 좋음을 가정). 두 모델 중 하나라도 그 fold에서 값이
    없으면(fold가 실패했거나 해당 metric을 계산 못 함) 비교 대상에서
    제외한다 - wins + losses + ties == comparable_folds가 항상 성립한다.

    이것은 통계적 유의성 검정이 아니다 - 25개 fold가 서로 독립 표본이
    아니므로(repeat마다 train set이 겹침) paired t-test류를 자동 생성하지
    않는다는 것이 이번 작업의 명시적 요구사항이다.
    """
    by_model_key: dict[CameraModelType, dict[tuple[int, int], float | None]] = {
        model: {(r["repeat_index"], r["fold_index"]): r.get(metric) for r in rows}
        for model, rows in rows_by_model.items()
    }
    models = list(rows_by_model.keys())
    results: list[dict] = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            common_keys = sorted(set(by_model_key[a]) & set(by_model_key[b]))
            wins_a = wins_b = ties = comparable = 0
            for key in common_keys:
                va, vb = by_model_key[a][key], by_model_key[b][key]
                if va is None or vb is None:
                    continue
                comparable += 1
                if va < vb:
                    wins_a += 1
                elif vb < va:
                    wins_b += 1
                else:
                    ties += 1
            results.append({
                "metric": metric,
                "model_a": a.value, "model_b": b.value,
                "wins_a": wins_a, "wins_b": wins_b, "ties": ties,
                "comparable_folds": comparable,
            })
    return results


# ---------------------------------------------------------------------------
# 3) Single Hold-out vs Repeated CV ranking sensitivity
# ---------------------------------------------------------------------------

def _rank_ascending(values: dict[CameraModelType, float | None]) -> list[tuple[CameraModelType, float]]:
    items = [(m, v) for m, v in values.items() if v is not None]
    items.sort(key=lambda item: item[1])
    return items


def ranking_sensitivity(
    single_holdout: dict[CameraModelType, ValidationResult],
    repeated: dict[CameraModelType, RepeatedKFoldResult],
) -> dict[str, dict]:
    """metric별로 Single Hold-out ranking과 Repeated CV(mean) ranking을
    나란히 계산하고 순위가 바뀌었는지("ranking_changed")만 표시한다 - 이
    자체를 실패/버그로 판정하지 않고, 어느 쪽이 "맞다"고 자동으로 정하지도
    않는다. Recommendation 로직에는 전혀 반영되지 않는 순수 진단이다.
    """
    def holdout_straightness(vr: ValidationResult | None) -> float | None:
        if vr is None or vr.straightness_source != "test":
            return None
        return vr.straightness_residual

    def repeated_straightness(r: RepeatedKFoldResult | None) -> float | None:
        if r is None or r.n_straightness_folds <= 0:
            return None
        return r.mean_test_straightness

    metric_value_pairs = {
        "test_rms": (
            {m: (vr.test_rms if vr else None) for m, vr in single_holdout.items()},
            {m: (r.mean_test_rms if r else None) for m, r in repeated.items()},
        ),
        "test_p95": (
            {
                m: (vr.test_residual_stats.p95 if vr and vr.test_residual_stats else None)
                for m, vr in single_holdout.items()
            },
            {m: (r.mean_test_p95 if r else None) for m, r in repeated.items()},
        ),
        "test_edge_rms": (
            {m: (vr.edge_rms if vr else None) for m, vr in single_holdout.items()},
            {m: (r.mean_edge_rms if r else None) for m, r in repeated.items()},
        ),
        "test_straightness": (
            {m: holdout_straightness(vr) for m, vr in single_holdout.items()},
            {m: repeated_straightness(r) for m, r in repeated.items()},
        ),
    }

    result: dict[str, dict] = {}
    for metric, (single_vals, repeated_vals) in metric_value_pairs.items():
        single_rank = _rank_ascending(single_vals)
        repeated_rank = _rank_ascending(repeated_vals)
        result[metric] = {
            "single_holdout_ranking": [{"model": m.value, "value": v} for m, v in single_rank],
            "repeated_cv_ranking": [{"model": m.value, "value": v} for m, v in repeated_rank],
            "ranking_changed": [m for m, _ in single_rank] != [m for m, _ in repeated_rank],
        }
    return result


# ---------------------------------------------------------------------------
# 4) Split manifest - Brown/Rational/Fisheye가 동일 fold partition을
#    썼는지 검증 가능한 형태로 train/test frame id를 남긴다.
# ---------------------------------------------------------------------------

def split_manifest(results: dict[CameraModelType, RepeatedKFoldResult]) -> dict:
    if not results:
        return {"models": [], "folds": []}
    models = list(results.keys())
    reference_model = models[0]
    reference_kfolds = results[reference_model].kfold_results

    folds_manifest: list[dict] = []
    for repeat_index, ref_kf in enumerate(reference_kfolds):
        for fold_index, ref_vr in enumerate(ref_kf.fold_validation_results):
            entry = {
                "repeat_index": repeat_index, "fold_index": fold_index, "seed": ref_kf.seed,
                "train_frame_ids": list(ref_vr.train_frame_ids),
                "test_frame_ids": list(ref_vr.test_frame_ids),
            }
            for model in models[1:]:
                kfolds = results[model].kfold_results
                if repeat_index >= len(kfolds):
                    continue
                other_folds = kfolds[repeat_index].fold_validation_results
                if fold_index >= len(other_folds):
                    continue
                other_vr = other_folds[fold_index]
                if set(other_vr.test_frame_ids) != set(ref_vr.test_frame_ids):
                    raise ValueError(
                        f"Fold partition mismatch at repeat={repeat_index} fold={fold_index}: "
                        f"{reference_model.value} vs {model.value} test frame ids differ - "
                        "same-seed cross-model fairness guarantee was violated."
                    )
            folds_manifest.append(entry)
    return {"models": [m.value for m in models], "folds": folds_manifest}


# ---------------------------------------------------------------------------
# 5) Evaluation ROI - 기본값 Full Image, 사용자가 명시적으로만 좁힌다.
# ---------------------------------------------------------------------------

@dataclass
class EvaluationROI:
    """평가에 쓸 image 영역. 기본값(full_image_roi())은 현재 결과를 그대로
    유지한다 - error가 큰 영역을 사후에 몰래 제외하는 cherry-picking을
    막기 위해, 모든 좌표(pixel/normalized)를 report/export에 그대로 남기고
    모든 모델/fold에 동일하게 적용해야 한다(이 dataclass 자체는 "정의"만
    담고, 실제로 어디에 적용할지는 호출부의 책임이다).
    """
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    image_width: int
    image_height: int
    label: str = "Full Image"

    @property
    def is_full_image(self) -> bool:
        return (
            self.x_min <= 0 and self.y_min <= 0
            and self.x_max >= self.image_width and self.y_max >= self.image_height
        )

    def normalized(self) -> dict:
        w = max(self.image_width, 1)
        h = max(self.image_height, 1)
        return {
            "x_min": self.x_min / w, "y_min": self.y_min / h,
            "x_max": self.x_max / w, "y_max": self.y_max / h,
        }

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "pixel": {"x_min": self.x_min, "y_min": self.y_min, "x_max": self.x_max, "y_max": self.y_max},
            "normalized": self.normalized(),
            "is_full_image": self.is_full_image,
        }


def full_image_roi(width: int, height: int) -> EvaluationROI:
    return EvaluationROI(
        x_min=0, y_min=0, x_max=width, y_max=height,
        image_width=width, image_height=height, label="Full Image",
    )


def filter_points_by_roi(rows: list[dict], roi: EvaluationROI) -> list[dict]:
    """spatial evidence point row(x,y 포함)를 ROI로 걸러낸다 - Full Image
    ROI라면 사실상 전부 통과(원래 결과와 동일)한다."""
    return [r for r in rows if roi.x_min <= r["x"] <= roi.x_max and roi.y_min <= r["y"] <= roi.y_max]


def roi_recomputed_stats(rows: list[dict]) -> dict:
    """ROI로 걸러진 point들의 magnitude로부터 RMS/P95를 다시 계산한다.
    원본(Full Image) test_rms/test_p95 등은 이 함수가 절대 건드리지 않는다 -
    "ROI-restricted"라는 이름이 붙은 별도 값으로만 report/export에 추가된다.
    """
    mags = [r["magnitude"] for r in rows if r.get("magnitude") is not None]
    if not mags:
        return {"n_points": 0, "rms": None, "p95": None}
    arr = np.asarray(mags, dtype=np.float64)
    return {
        "n_points": len(mags),
        "rms": float(np.sqrt(np.mean(arr ** 2))),
        "p95": float(np.percentile(arr, 95)),
    }


# ---------------------------------------------------------------------------
# 6) 논문 실험 조건 metadata
# ---------------------------------------------------------------------------

@dataclass
class PaperExperimentMetadata:
    resolution: str
    approximate_fov_deg: Optional[float]
    target_type: str
    board_geometry: str
    num_usable_frames: int
    k: int
    n_repeats: int
    base_seed: int
    camera_installation: str = "behind windshield"
    target_location: str = "outside windshield"
    evaluation_roi: Optional[dict] = None
    opencv_version: str = ""
    git_commit_sha: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _git_commit_sha() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def build_paper_metadata(
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    dataset: Dataset | None,
    k: int,
    n_repeats: int,
    base_seed: int,
    roi: EvaluationROI | None = None,
) -> PaperExperimentMetadata:
    import cv2

    usable = 0
    if dataset is not None:
        usable = sum(1 for f in dataset.enabled_frames if f.detection and f.detection.success)
    board = f"{pattern_config.type.value} {pattern_config.squares_x}x{pattern_config.squares_y}, square={pattern_config.square_size}m"
    if pattern_config.marker_size:
        board += f", marker={pattern_config.marker_size}m"
    roi_dict = (roi or full_image_roi(camera_config.width, camera_config.height)).to_dict()
    return PaperExperimentMetadata(
        resolution=f"{camera_config.width}x{camera_config.height}",
        approximate_fov_deg=camera_config.hfov_deg,
        target_type=pattern_config.type.value,
        board_geometry=board,
        num_usable_frames=usable,
        k=k, n_repeats=n_repeats, base_seed=base_seed,
        evaluation_roi=roi_dict,
        opencv_version=cv2.__version__,
        git_commit_sha=_git_commit_sha(),
    )


# ---------------------------------------------------------------------------
# 7) Paper summary report (raw evidence only - no auto-generated conclusions)
# ---------------------------------------------------------------------------

def build_paper_summary_report(
    single_holdout: dict[CameraModelType, ValidationResult],
    repeated: dict[CameraModelType, RepeatedKFoldResult],
    stability_by_model: dict[CameraModelType, ParameterUncertainty | None] | None = None,
    spatial_summary_by_model: dict[CameraModelType, SpatialErrorMap | None] | None = None,
    metadata: PaperExperimentMetadata | None = None,
) -> str:
    """6개 section(Single Hold-out / Repeated 5-Fold x 5 / Pairwise Fold
    Wins / Split Sensitivity / Parameter Stability / Spatial Accuracy)으로
    raw evidence를 정리한다. "Fisheye가 가장 좋다"류의 결론 문장은 어디에도
    생성하지 않는다 - 계산된 숫자를 그대로 나열할 뿐이다.
    """
    lines: list[str] = []

    if metadata is not None:
        lines.append("=== Experiment Metadata ===")
        for key, value in metadata.to_dict().items():
            lines.append(f"{key}: {value}")
        lines.append("")

    lines.append("=== 1. Single Hold-out Results ===")
    for model, vr in single_holdout.items():
        if vr is None:
            lines.append(f"{model.value}: N/A")
            continue
        p95 = vr.test_residual_stats.p95 if vr.test_residual_stats else None
        straight = vr.straightness_residual if vr.straightness_source == "test" else None
        lines.append(
            f"{model.value}: Train RMS={_fmt(vr.train_rms)}, Test RMS={_fmt(vr.test_rms)}, "
            f"P95={_fmt(p95)}, Edge RMS={_fmt(vr.edge_rms)}, "
            f"Straightness={_fmt(straight)} (source={vr.straightness_source})"
        )
    lines.append("")

    any_repeated = next(iter(repeated.values()), None)
    header_k = any_repeated.k if any_repeated else "?"
    header_n = any_repeated.n_repeats if any_repeated else "?"
    lines.append(f"=== 2. Repeated {header_k}-Fold x {header_n} ===")
    rows_by_model = {m: kfold_raw_rows(m, r) for m, r in repeated.items()}
    summary_rows = model_metric_descriptive_summary(rows_by_model)
    summary_by_model = {row["model"]: row for row in summary_rows}
    for model, r in repeated.items():
        lines.append(
            f"{model.value}: successful folds {r.n_successful_runs}/{r.total_folds} "
            f"(fully {r.fully_successful_folds}, partial {r.partial_success_folds}, failed {r.failed_folds})"
        )
        stats = summary_by_model.get(model.value, {})
        for metric in PAPER_FOLD_METRICS:
            lines.append(
                f"  {metric}: mean={_fmt(stats.get(f'{metric}_mean'))} std={_fmt(stats.get(f'{metric}_std'))} "
                f"median={_fmt(stats.get(f'{metric}_median'))} min={_fmt(stats.get(f'{metric}_min'))} "
                f"max={_fmt(stats.get(f'{metric}_max'))} (n={stats.get(f'{metric}_n', 0)})"
            )
    lines.append("")

    lines.append("=== 3. Pairwise Fold Wins (raw count, not a significance test) ===")
    for metric in PAPER_FOLD_METRICS:
        for w in pairwise_win_counts(rows_by_model, metric):
            lines.append(
                f"{metric}: {w['model_a']} vs {w['model_b']} -> "
                f"{w['model_a']} wins {w['wins_a']}/{w['comparable_folds']}, "
                f"{w['model_b']} wins {w['wins_b']}/{w['comparable_folds']}, "
                f"ties {w['ties']}"
            )
    lines.append("")

    lines.append("=== 4. Split Sensitivity (Single Hold-out ranking vs Repeated CV ranking) ===")
    for metric, info in ranking_sensitivity(single_holdout, repeated).items():
        single_order = " > ".join(f"{e['model']}({_fmt(e['value'])})" for e in info["single_holdout_ranking"])
        repeated_order = " > ".join(f"{e['model']}({_fmt(e['value'])})" for e in info["repeated_cv_ranking"])
        lines.append(f"{metric}:")
        lines.append(f"  Single Hold-out ranking: {single_order or 'N/A'}")
        lines.append(f"  Repeated CV ranking:     {repeated_order or 'N/A'}")
        lines.append(f"  ranking_changed: {info['ranking_changed']}")
    lines.append("")

    lines.append("=== 5. Parameter Stability ===")
    for model, pu in (stability_by_model or {}).items():
        if pu is None:
            lines.append(f"{model.value}: N/A")
            continue
        total = pu.n_bootstrap_total if pu.n_bootstrap_total is not None else "N/A"
        success = pu.n_bootstrap_success if pu.n_bootstrap_success is not None else "N/A"
        lines.append(
            f"{model.value}: method={pu.method}, bootstrap success={success}/{total}, "
            f"Paper Intrinsic Stability={_fmt_score(pu.paper_intrinsic_stability)}, "
            f"All-Parameter Stability={_fmt_score(pu.overall_stability)}"
        )
        lines.append(
            f"  fx={_fmt_score(pu.fx_stability)} fy={_fmt_score(pu.fy_stability)} "
            f"cx={_fmt_score(pu.cx_stability)} cy={_fmt_score(pu.cy_stability)}"
        )
        if pu.lowest_stability_parameter is not None:
            lines.append(
                f"  lowest-stability parameter: {pu.lowest_stability_parameter} "
                f"({_fmt_score(pu.lowest_stability_value)})"
            )
        near_zero = [s.label or f"d{s.index}" for s in pu.distortion_stats if s.near_zero_reference]
        if near_zero:
            lines.append(f"  near-zero reference distortion coefficients: {', '.join(near_zero)}")
    lines.append("")

    lines.append("=== 6. Spatial Accuracy (held-out test residual) ===")
    if spatial_summary_by_model:
        for model, smap in spatial_summary_by_model.items():
            if smap is None:
                lines.append(f"{model.value}: N/A")
                continue
            cell_rms = [c.rms for c in smap.cells if c.rms is not None]
            lines.append(
                f"{model.value}: {smap.rows}x{smap.cols} grid, "
                f"cell RMS range {_fmt(min(cell_rms)) if cell_rms else 'N/A'} ~ "
                f"{_fmt(max(cell_rms)) if cell_rms else 'N/A'} px"
            )
    else:
        lines.append("N/A (spatial evidence not provided to this report)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 8) Export - paper_summary.csv / repeated_kfold_folds.csv /
#    repeated_kfold_repeats.csv / pairwise_win_counts.csv /
#    stability_parameters.csv / split_manifest.json / paper_metrics.json
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> str:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(path)


def export_paper_metrics(
    output_dir: str | Path,
    single_holdout: dict[CameraModelType, ValidationResult],
    repeated: dict[CameraModelType, RepeatedKFoldResult],
    stability_by_model: dict[CameraModelType, ParameterUncertainty | None] | None = None,
    metadata: PaperExperimentMetadata | None = None,
    spatial_rows_by_model: dict[CameraModelType, list[dict]] | None = None,
) -> dict[str, str]:
    """논문용 raw evidence를 output_dir에 한 번에 저장한다. 기존 OpenCV
    YAML export(export/*.py)와는 완전히 별도의 경로/파일명을 쓴다 -
    calibration 결과를 하나도 다시 계산하지 않고 이미 있는 값을 파일로
    옮겨 적을 뿐이다.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    stability_by_model = stability_by_model or {}

    rows_by_model = {m: kfold_raw_rows(m, r) for m, r in repeated.items()}
    fold_rows = [row for rows in rows_by_model.values() for row in rows]
    written["repeated_kfold_folds.csv"] = _write_csv(
        out / "repeated_kfold_folds.csv", fold_rows,
        fieldnames=[
            "model", "repeat_index", "fold_index", "seed", "train_frame_count",
            "test_frame_count", "successful_test_frames", "failed_test_frames",
            "fold_status", "test_rms", "test_p95", "test_edge_rms",
            "test_straightness", "straightness_source",
        ],
    )

    repeat_rows = [row for m, r in repeated.items() for row in repeat_level_rows(m, r)]
    written["repeated_kfold_repeats.csv"] = _write_csv(
        out / "repeated_kfold_repeats.csv", repeat_rows,
        fieldnames=[
            "model", "repeat_index", "seed", "k", "successful_folds",
            "mean_test_rms", "mean_test_p95", "mean_test_edge_rms",
            "mean_test_straightness", "n_straightness_folds",
        ],
    )

    win_rows = [w for metric in PAPER_FOLD_METRICS for w in pairwise_win_counts(rows_by_model, metric)]
    written["pairwise_win_counts.csv"] = _write_csv(
        out / "pairwise_win_counts.csv", win_rows,
        fieldnames=["metric", "model_a", "model_b", "wins_a", "wins_b", "ties", "comparable_folds"],
    )

    stability_rows: list[dict] = []
    for model, pu in stability_by_model.items():
        if pu is None:
            continue
        # model당 한 번만 계산되는 provenance(방법/집계 stability)를 매
        # parameter row에 그대로 반복해 붙인다 - Paper Intrinsic Stability
        # (fx/fy/cx/cy만)와 All-Parameter Stability(distortion 포함 가능,
        # recommender.py가 실제로 쓰는 legacy overall_stability)가 CSV
        # 한 줄만 봐도 절대 헷갈리지 않도록 명시적으로 분리된 열로 둔다.
        model_context = {
            "model": model.value,
            "method": pu.method,
            "paper_intrinsic_stability": pu.paper_intrinsic_stability,
            "all_parameter_stability": pu.overall_stability,
        }
        for label, ref, mean, std, ci_low, ci_high, cv, stab in (
            ("fx", pu.fx_reference, pu.fx_mean, pu.fx_std, pu.fx_ci_low, pu.fx_ci_high, pu.fx_relative_cv, pu.fx_stability),
            ("fy", pu.fy_reference, pu.fy_mean, pu.fy_std, pu.fy_ci_low, pu.fy_ci_high, pu.fy_relative_cv, pu.fy_stability),
            ("cx", pu.cx_reference, pu.cx_mean, pu.cx_std, pu.cx_ci_low, pu.cx_ci_high, pu.cx_relative_cv, pu.cx_stability),
            ("cy", pu.cy_reference, pu.cy_mean, pu.cy_std, pu.cy_ci_low, pu.cy_ci_high, pu.cy_relative_cv, pu.cy_stability),
        ):
            stability_rows.append({
                **model_context,
                "parameter": label, "category": "intrinsic",
                "reference": ref, "mean": mean, "std": std, "cv": cv,
                "ci_low": ci_low, "ci_high": ci_high, "stability_score": stab,
                "near_zero_reference": False, "diagnostic": None,
            })
        for stat in pu.distortion_stats:
            stability_rows.append({
                **model_context,
                "parameter": stat.label or f"d{stat.index}", "category": "distortion",
                "reference": stat.reference, "mean": stat.mean, "std": stat.std, "cv": stat.relative_cv,
                "ci_low": stat.ci_low, "ci_high": stat.ci_high, "stability_score": stat.stability_score,
                "near_zero_reference": stat.near_zero_reference, "diagnostic": stat.diagnostic,
            })
    written["stability_parameters.csv"] = _write_csv(
        out / "stability_parameters.csv", stability_rows,
        fieldnames=[
            "model", "method", "paper_intrinsic_stability", "all_parameter_stability",
            "parameter", "category", "reference", "mean", "std", "cv",
            "ci_low", "ci_high", "stability_score", "near_zero_reference", "diagnostic",
        ],
    )

    manifest = split_manifest(repeated)
    manifest_path = out / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    written["split_manifest.json"] = str(manifest_path)

    summary_rows = model_metric_descriptive_summary(rows_by_model)
    summary_fieldnames = list(summary_rows[0].keys()) if summary_rows else [
        "model", "total_folds", "successful_folds",
    ]
    written["paper_summary.csv"] = _write_csv(out / "paper_summary.csv", summary_rows, fieldnames=summary_fieldnames)

    paper_metrics = {
        "metadata": metadata.to_dict() if metadata else None,
        "single_holdout": {
            m.value: {
                "train_rms": vr.train_rms, "test_rms": vr.test_rms,
                "test_p95": vr.test_residual_stats.p95 if vr.test_residual_stats else None,
                "test_edge_rms": vr.edge_rms,
                "test_straightness": vr.straightness_residual if vr.straightness_source == "test" else None,
                "straightness_source": vr.straightness_source,
            }
            for m, vr in single_holdout.items() if vr is not None
        },
        "repeated_kfold_summary": summary_rows,
        "pairwise_win_counts": win_rows,
        "ranking_sensitivity": ranking_sensitivity(single_holdout, repeated),
        "stability": stability_rows,
    }
    metrics_path = out / "paper_metrics.json"
    metrics_path.write_text(json.dumps(paper_metrics, indent=2, default=str), encoding="utf-8")
    written["paper_metrics.json"] = str(metrics_path)

    if spatial_rows_by_model:
        for model, rows in spatial_rows_by_model.items():
            path = out / f"spatial_residuals_{model.value}.csv"
            _write_csv(
                path, rows,
                fieldnames=["x", "y", "dx", "dy", "magnitude", "model", "frame_id", "grid_row", "grid_col"],
            )
            written[f"spatial_residuals_{model.value}.csv"] = str(path)

    return written

"""
scripts/tune_model_score_weights.py
========================================

설계 문서 8번 Model Score 가중치(w_train, w_test, w_edge, w_line, w_complexity)를
"정답을 아는" 합성 시나리오로 튜닝한다.

실제 카메라로 찍은 데이터셋이 없으므로, 대신 세 가지 "진짜 카메라"를 합성으로
만든다 - 각각 어떤 모델이 정답인지 우리가 이미 알고 있다:

    1. true_pinhole   : 왜곡이 전혀 없는 이상적인 카메라 -> 정답은 Pinhole
                        (복잡한 모델은 노이즈에 과적합할 뿐이므로 오컴의 면도날)
    2. true_extended  : 중간 정도의 방사+접선 왜곡(k1,k2,p1,p2) -> 정답은 Extended Pinhole
    3. true_fisheye   : cv2.fisheye 왜곡 모델(k1~k4)의 넓은 화각 -> 정답은 Fisheye

각 시나리오를 여러 랜덤 시드로 반복해(포즈 다양성에 따라 우연히 결과가
바뀌지 않는지 확인) 3모델 계산 + Hold-out validation까지 실제로 돌리고,
그 결과(CalibrationResult, ValidationResult)를 캐싱한다.

캐싱한 뒤에는 recommender.compute_model_scores()를 그대로 재사용해서
(재구현하지 않음 - 실제 프로덕션 코드와 반드시 같은 로직으로 평가해야 튜닝
결과가 의미 있다) 다양한 가중치 후보에 대해 "정답 모델을 맞췄는가"를
채점한다. 후보 가중치는 Dirichlet 분포로 무작위 샘플링한다(w1~w5 >= 0,
합=1 제약을 자연스럽게 만족시키는 표준적인 방법).

실행:
    python scripts/tune_model_score_weights.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from calibration.compare import run_all_models
from calibration.recommender import compute_model_scores
from calibration.types import (
    CameraConfig,
    CameraModelType,
    Dataset,
    DetectionResult,
    Frame,
    FrameStatus,
    ImageInfo,
    ModelScoreWeights,
    PatternConfig,
    PatternType,
)
from calibration.validation import validate_all_models

# ---------------------------------------------------------------------------
# 공통 설정
# ---------------------------------------------------------------------------

PATTERN = PatternConfig(
    type=PatternType.CHARUCO, squares_x=7, squares_y=5,
    square_size=0.04, marker_size=0.03, dictionary="DICT_5X5_100",
)
CAMERA = CameraConfig(width=1920, height=1080)
W, H = CAMERA.width, CAMERA.height
CENTER = (W / 2, H / 2)


def _board_geometry(corners_2d: np.ndarray) -> tuple[float, tuple[float, float], float]:
    """detector.py의 _compute_board_geometry와 동일한 방식 - stratified split이
    합성 데이터에서도 실제와 비슷하게 동작하게 하기 위함.
    """
    pts = corners_2d.reshape(-1, 2).astype(np.float32)
    hull = cv2.convexHull(pts)
    hull_area = cv2.contourArea(hull)
    area_ratio = float(hull_area / (W * H))
    cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    rect = cv2.minAreaRect(pts)
    return area_ratio, (cx, cy), float(rect[-1])


def _build_synthetic_frames(
    true_K: np.ndarray,
    true_D: np.ndarray,
    projection: str,  # "pinhole" or "fisheye"
    n_frames: int,
    seed: int,
    pixel_noise_std: float = 0.0,
) -> list[Frame]:
    """3D ChArUco 격자를 다양한 포즈로 투영해 "실제로 촬영한 것처럼" 프레임을 만든다.
    이미지 렌더링/검출을 거치지 않고 3D->2D 사영을 직접 계산하므로 빠르고,
    검출 알고리즘의 노이즈 없이 "카메라 모델 자체"만 순수하게 검증할 수 있다.

    pixel_noise_std > 0이면 코너 좌표에 가우시안 노이즈를 더해 실제 코너
    검출기의 서브픽셀 오차를 흉내낸다 - 완벽한 수학적 투영만 쓰면 문제가
    너무 쉬워져(모든 가중치가 100% 정답률을 내서) 가중치 차이가 안 드러난다.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard(
        (PATTERN.squares_x, PATTERN.squares_y), PATTERN.square_size, PATTERN.marker_size, aruco_dict
    )
    pts3d = board.getChessboardCorners().astype(np.float32)
    n_corners = pts3d.shape[0]
    ids = np.arange(n_corners, dtype=np.int32).reshape(-1, 1)

    rng = np.random.default_rng(seed)
    frames: list[Frame] = []

    attempts = 0
    while len(frames) < n_frames and attempts < n_frames * 10:
        attempts += 1
        # fisheye 시나리오는 일부러 더 넓은 각도/근접 포즈를 섞어서 광각 렌즈 촬영을 흉내낸다.
        angle_scale = 0.9 if projection == "fisheye" else 0.5
        rvec = (rng.random(3) - 0.5) * angle_scale
        tvec = np.array([
            (rng.random() - 0.5) * 0.35,
            (rng.random() - 0.5) * 0.35,
            0.35 + rng.random() * 0.4,
        ])

        try:
            if projection == "fisheye":
                projected, _ = cv2.fisheye.projectPoints(
                    pts3d.reshape(-1, 1, 3).astype(np.float64), rvec, tvec, true_K, true_D
                )
            else:
                projected, _ = cv2.projectPoints(pts3d.reshape(-1, 1, 3), rvec, tvec, true_K, true_D)
        except cv2.error:
            continue
        projected = projected.reshape(-1, 2)

        if np.any(~np.isfinite(projected)):
            continue
        if np.any(projected[:, 0] < 0) or np.any(projected[:, 0] > W):
            continue
        if np.any(projected[:, 1] < 0) or np.any(projected[:, 1] > H):
            continue

        if pixel_noise_std > 0:
            projected = projected + rng.normal(0, pixel_noise_std, projected.shape)

        area_ratio, center_px, tilt = _board_geometry(projected)
        image_id = f"synth_{len(frames):03d}"
        info = ImageInfo(image_id=image_id, path="-", width=W, height=H)
        det = DetectionResult(
            image_id=image_id, success=True,
            corners=projected.reshape(-1, 1, 2).astype(np.float32),
            object_points=pts3d.reshape(-1, 1, 3), ids=ids, num_corners=n_corners,
            board_area_ratio=area_ratio, board_center_px=center_px, board_tilt_deg=tilt,
        )
        frames.append(Frame(image_info=info, detection=det, status=FrameStatus.DETECTED))

    return frames


# ---------------------------------------------------------------------------
# 정답이 있는 세 시나리오
# ---------------------------------------------------------------------------

FX = FY = 1000.0
TRUE_K_PINHOLE = np.array([[FX, 0, CENTER[0]], [0, FY, CENTER[1]], [0, 0, 1]])

# 각 시나리오는 (진짜 카메라 파라미터, 투영 방식, 정답 모델, 프레임 수, 픽셀 노이즈)로 구성.
# "clean"(노이즈 없는 완벽한 수학적 투영) 버전은 1차 실험에서 어떤 가중치를 써도
# 항상 100% 정답이라 튜닝 신호가 전혀 없었다 - 그래서 최종 세트에서는 뺐다.
# 노이즈 있는/데이터 적은 "현실적으로 어려운" 시나리오만 남겨서 계산 비용도 줄인다.
SCENARIOS = {
    "true_pinhole_noisy": dict(
        true_K=TRUE_K_PINHOLE, true_D=np.zeros(5), projection="pinhole",
        ground_truth=CameraModelType.PINHOLE, n_frames=16, noise=0.4,
    ),
    "true_extended_noisy": dict(
        true_K=TRUE_K_PINHOLE, true_D=np.array([-0.22, 0.08, 0.002, -0.001, 0.0]), projection="pinhole",
        ground_truth=CameraModelType.EXTENDED_PINHOLE, n_frames=16, noise=0.4,
    ),
    "true_extended_mild_lowdata": dict(
        # 왜곡이 약하고(mild) 프레임도 적어서(lowdata) Pinhole과 구분하기 어려운
        # 경계 케이스 - 복잡도 페널티(w_complexity)가 너무 세면 여기서 오답(Pinhole)을
        # 고르기 쉬워진다. 튜닝이 실제로 뭔가를 하는지 보여주는 핵심 시나리오.
        true_K=TRUE_K_PINHOLE, true_D=np.array([-0.06, 0.01, 0.0, 0.0, 0.0]), projection="pinhole",
        ground_truth=CameraModelType.EXTENDED_PINHOLE, n_frames=9, noise=0.3,
    ),
    "true_fisheye_noisy": dict(
        true_K=np.array([[700.0, 0, CENTER[0]], [0, 700.0, CENTER[1]], [0, 0, 1]]),
        true_D=np.array([0.05, 0.01, -0.01, 0.002]), projection="fisheye",
        ground_truth=CameraModelType.FISHEYE, n_frames=14, noise=0.4,
    ),
}

SEEDS = [1, 2]
HOLDOUT_SEEDS = [11]  # 가중치 탐색에는 안 쓰고, 최종 평가에만 쓰는 별도 시드
# (이 프로젝트가 스스로 강조하는 "test intrinsic 재최적화 금지" 원칙과 같은 이유:
#  탐색에 쓴 시드로만 평가하면 그 노이즈 실현값에 과적합됐는지 알 수 없다.)


def build_cached_results(seeds):
    """각 시나리오 x 시드마다 실제 3모델 계산 + Hold-out validation을 한 번만
    돌려서 결과를 캐싱한다. 이후 가중치 탐색은 이 캐시만 갖고 하므로 빠르다.
    """
    cache = []
    for name, cfg in SCENARIOS.items():
        for seed in seeds:
            frames = _build_synthetic_frames(
                cfg["true_K"], cfg["true_D"], cfg["projection"], cfg["n_frames"], seed,
                pixel_noise_std=cfg["noise"],
            )
            if len(frames) < 8:
                print(f"  [경고] {name}/seed={seed}: 유효 프레임이 {len(frames)}장뿐 - 건너뜀")
                continue
            dataset = Dataset(frames=frames)

            results = run_all_models(dataset, CAMERA)
            calibration_results = {r.model_name: r for r in results}
            validation_results = validate_all_models(dataset, CAMERA, PATTERN, test_ratio=0.3, seed=seed)

            cache.append({
                "scenario": name,
                "seed": seed,
                "ground_truth": cfg["ground_truth"],
                "calibration_results": calibration_results,
                "validation_results": validation_results,
            })
            statuses = {m.value: r.success for m, r in calibration_results.items()}
            print(f"  {name}/seed={seed}: 계산 완료 (success={statuses})")
    return cache


def save_cache(cache, path: str) -> None:
    import pickle
    with open(path, "wb") as f:
        pickle.dump(cache, f)


def load_cache(path: str):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate_weights(cache, weights: ModelScoreWeights) -> tuple[float, float]:
    """주어진 가중치로 캐시된 모든 시나리오를 채점.
    Returns: (정답률, 평균 마진 - 정답을 맞춘 경우 1등과 2등 점수 차이의 평균)
    """
    correct = 0
    margins = []
    for entry in cache:
        scores = compute_model_scores(
            entry["calibration_results"], entry["validation_results"], weights=weights
        )
        eligible = [s for s in scores if entry["calibration_results"][s.model_name].success]
        if not eligible:
            continue
        ranked = sorted(eligible, key=lambda s: s.score)
        best = ranked[0]
        is_correct = best.model_name == entry["ground_truth"]
        if is_correct:
            correct += 1
            if len(ranked) > 1:
                margins.append(ranked[1].score - ranked[0].score)

    accuracy = correct / len(cache) if cache else 0.0
    avg_margin = float(np.mean(margins)) if margins else 0.0
    return accuracy, avg_margin


def random_search(cache, n_candidates: int = 3000, seed: int = 0):
    """Dirichlet 분포로 (w1..w5 >= 0, 합=1) 제약을 만족하는 후보를 무작위로
    많이 뽑아서 가장 좋은 걸 고른다 - 5차원이라 촘촘한 grid search는 비효율적.
    """
    rng = np.random.default_rng(seed)
    best_weights = ModelScoreWeights()
    best_acc, best_margin = evaluate_weights(cache, best_weights)
    print(f"\n기본 가중치 성능: 정답률={best_acc:.1%}  평균마진={best_margin:.4f}")
    print(f"  {best_weights}")

    for _ in range(n_candidates):
        w = rng.dirichlet(np.ones(5))
        candidate = ModelScoreWeights(
            w_train=float(w[0]), w_test=float(w[1]), w_edge=float(w[2]),
            w_line=float(w[3]), w_complexity=float(w[4]),
        )
        acc, margin = evaluate_weights(cache, candidate)
        if (acc, margin) > (best_acc, best_margin):
            best_acc, best_margin, best_weights = acc, margin, candidate

    return best_weights, best_acc, best_margin


def main():
    print("=== 1. 탐색용(train) 시나리오 캐싱 ===")
    train_cache = build_cached_results(SEEDS)
    print(f"총 {len(train_cache)}개 케이스 캐싱 완료")

    print("\n=== 2. 가중치 무작위 탐색 (Dirichlet random search, 탐색용 캐시만 사용) ===")
    best_weights, train_acc, train_margin = random_search(train_cache, n_candidates=6000)

    print(f"\n[탐색용 세트] 기본 가중치 vs 튜닝된 가중치")
    default_acc, default_margin = evaluate_weights(train_cache, ModelScoreWeights())
    print(f"  기본:   정답률={default_acc:.1%}  마진={default_margin:.4f}")
    print(f"  튜닝됨: 정답률={train_acc:.1%}  마진={train_margin:.4f}")

    print("\n=== 3. 별도 시드(held-out)로 최종 검증 - 탐색에 안 쓴 노이즈로 재확인 ===")
    holdout_cache = build_cached_results(HOLDOUT_SEEDS)
    holdout_default_acc, holdout_default_margin = evaluate_weights(holdout_cache, ModelScoreWeights())
    holdout_tuned_acc, holdout_tuned_margin = evaluate_weights(holdout_cache, best_weights)
    print(f"  [held-out] 기본:   정답률={holdout_default_acc:.1%}  마진={holdout_default_margin:.4f}")
    print(f"  [held-out] 튜닝됨: 정답률={holdout_tuned_acc:.1%}  마진={holdout_tuned_margin:.4f}")

    print(f"\n최적 가중치: {best_weights}")

    print("\n=== held-out 세트 시나리오별 상세 (튜닝된 가중치 기준) ===")
    for entry in holdout_cache:
        scores = compute_model_scores(
            entry["calibration_results"], entry["validation_results"], weights=best_weights
        )
        eligible = [s for s in scores if entry["calibration_results"][s.model_name].success]
        ranked = sorted(eligible, key=lambda s: s.score)
        picked = ranked[0].model_name if ranked else None
        mark = "OK" if picked == entry["ground_truth"] else "X "
        print(
            f"  [{mark}] {entry['scenario']:26s} seed={entry['seed']}  "
            f"정답={entry['ground_truth'].value:16s}  선택={picked.value if picked else 'N/A'}"
        )

    if holdout_tuned_acc >= holdout_default_acc:
        print(
            f"\n=> held-out에서도 튜닝된 가중치가 기본보다 같거나 나음 "
            f"({holdout_tuned_acc:.1%} >= {holdout_default_acc:.1%}) - 과적합 아님, 채택 권장"
        )
    else:
        print(
            f"\n=> held-out에서는 오히려 기본이 더 나음 "
            f"({holdout_default_acc:.1%} > {holdout_tuned_acc:.1%}) - 탐색용 시드에 과적합된 것으로 보임, 기본 유지 권장"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-cache", type=int, default=None,
        help="이 시드 하나만으로 캐시를 빌드해 파일로 저장하고 종료 (여러 번의 짧은 실행으로 나눠 돌릴 때 사용)",
    )
    parser.add_argument("--out", type=str, default=None, help="--build-cache와 함께 사용할 출력 pickle 경로")
    parser.add_argument(
        "--search", nargs="+", default=None,
        help="주어진 pickle 파일들을 합쳐 탐색용 캐시로 쓰고 랜덤서치 실행",
    )
    parser.add_argument(
        "--holdout", nargs="+", default=None,
        help="--search와 함께: held-out 평가용 pickle 파일들",
    )
    parser.add_argument("--n-candidates", type=int, default=6000)
    args = parser.parse_args()

    if args.build_cache is not None:
        cache = build_cached_results([args.build_cache])
        out = args.out or f"/tmp/cache_seed{args.build_cache}.pkl"
        save_cache(cache, out)
        print(f"\n캐시 저장 완료: {out} ({len(cache)}개 케이스)")
    elif args.search:
        train_cache = []
        for p in args.search:
            train_cache.extend(load_cache(p))
        print(f"탐색용 캐시 로드: {len(train_cache)}개 케이스 (from {args.search})")

        best_weights, train_acc, train_margin = random_search(train_cache, n_candidates=args.n_candidates)
        default_acc, default_margin = evaluate_weights(train_cache, ModelScoreWeights())
        print(f"\n[탐색용 세트] 기본: 정답률={default_acc:.1%} 마진={default_margin:.4f}")
        print(f"[탐색용 세트] 튜닝: 정답률={train_acc:.1%} 마진={train_margin:.4f}")
        print(f"\n최적 가중치: {best_weights}")

        if args.holdout:
            holdout_cache = []
            for p in args.holdout:
                holdout_cache.extend(load_cache(p))
            print(f"\nheld-out 캐시 로드: {len(holdout_cache)}개 케이스 (from {args.holdout})")
            h_default_acc, h_default_margin = evaluate_weights(holdout_cache, ModelScoreWeights())
            h_tuned_acc, h_tuned_margin = evaluate_weights(holdout_cache, best_weights)
            print(f"[held-out] 기본: 정답률={h_default_acc:.1%} 마진={h_default_margin:.4f}")
            print(f"[held-out] 튜닝: 정답률={h_tuned_acc:.1%} 마진={h_tuned_margin:.4f}")

            print("\n=== held-out 상세 ===")
            for entry in holdout_cache:
                scores = compute_model_scores(
                    entry["calibration_results"], entry["validation_results"], weights=best_weights
                )
                eligible = [s for s in scores if entry["calibration_results"][s.model_name].success]
                ranked = sorted(eligible, key=lambda s: s.score)
                picked = ranked[0].model_name if ranked else None
                mark = "OK" if picked == entry["ground_truth"] else "X "
                print(
                    f"  [{mark}] {entry['scenario']:26s} seed={entry['seed']}  "
                    f"정답={entry['ground_truth'].value:16s}  선택={picked.value if picked else 'N/A'}"
                )
    else:
        main()

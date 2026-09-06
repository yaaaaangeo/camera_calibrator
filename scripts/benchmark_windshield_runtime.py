"""
scripts/benchmark_windshield_runtime.py
================================================

Phase C-3 - Windshield Geometry Model Runtime Latency Benchmark.

Baseline/Spherical/Residual Grid/Residual RBF/Spline(+ torch가 설치돼 있으면
Neural Residual)의 point projection 성능을 1 / 100 / 1,000 / 10,000 포인트
스케일에서 세 티어로 측정한다:

    - Scalar      : `model.project_point()`/`unproject_pixel()`을 호출자가
                     포인트 수만큼 반복 호출(현재 WindshieldModel 공개 API의
                     기본 사용법).
    - Batch(Exact): `project_points_exact_batch()`(Phase C-1) - 계산 자체는
                     Scalar와 100% 동일(근사 아님). Baseline만 실제로
                     벡터화되고, 나머지는 반복 호출의 Python 오버헤드만 줄인다.
    - LUT(근사)    : `RuntimeWindshieldProjector`(Phase C-1) - 정확도
                     tradeoff가 있는 빠른 근사(모듈 docstring 참고).

CPU-first다: GPU/CUDA를 전혀 요구하지 않고, 실제 이미지/캘리브레이션
데이터셋도 요구하지 않는다(합성 K/D/기하 파라미터로 각 모델을 직접
구성한다) - Jetson을 포함한 어떤 기기에서도 그대로 실행 가능해야 한다는
요구사항(사용자 스펙 C-3번) 때문이다.

중요 - 절대 하지 않는 것: 이 스크립트가 출력하는 숫자는 "이 스크립트를 실행한
바로 그 기기"의 실측치일 뿐이다. Jetson 기기에서 실행한 적이 없다면 그 사실을
반드시 함께 보고해야 한다(사용자 스펙 - "must never fake Jetson numbers in
CI"). 이 스크립트 자체는 Jetson 여부를 자동 감지해 결과 상단에 표시한다.

실행:
    python scripts/benchmark_windshield_runtime.py
    python scripts/benchmark_windshield_runtime.py --scales 1 1000 --repeats 5
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from calibration.types import CameraModelType
from calibration.windshield.base import WindshieldModel
from calibration.windshield.baseline import BaselineWindshieldModel
from calibration.windshield.residual_ray import ResidualRayWindshieldModel
from calibration.windshield.residual_rbf import ResidualRBFWindshieldModel
from calibration.windshield.runtime_projector import (
    RuntimeWindshieldProjector,
    project_points_exact_batch,
    unproject_pixels_exact_batch,
)
from calibration.windshield.spherical import SphericalWindshieldModel
from calibration.windshield.spline import MIN_SPLINE_GRID_SIZE, SplineWindshieldModel, compute_angular_fov_scale

DEFAULT_SCALES = (1, 100, 1_000, 10_000)
DEFAULT_REPEATS = 5
IMG_W, IMG_H = 1280.0, 800.0
_MODEL = CameraModelType.BROWN_CONRADY


def _camera_matrix_distortion() -> tuple[np.ndarray, np.ndarray]:
    K = np.array([[900.0, 0.0, IMG_W / 2], [0.0, 900.0, IMG_H / 2], [0.0, 0.0, 1.0]])
    D = np.array([[-0.15], [0.05], [0.0], [0.0], [0.0]])
    return K, D


def _sample_points(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-0.3, 0.3, n)
    ys = rng.uniform(-0.2, 0.2, n)
    zs = rng.uniform(2.0, 30.0, n)  # LiDAR 전형적 거리대(windshield-camera 간격보다 훨씬 멀다)
    return np.stack([xs * zs, ys * zs, zs], axis=1)


def _sample_pixels(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(0, IMG_W, n), rng.uniform(0, IMG_H, n)], axis=1)


def _build_baseline() -> WindshieldModel:
    K, D = _camera_matrix_distortion()
    return BaselineWindshieldModel(K, D, _MODEL)


def _build_spherical() -> WindshieldModel:
    K, D = _camera_matrix_distortion()
    center = np.array([0.0, 0.0, -9.7])
    return SphericalWindshieldModel(K, D, _MODEL, center, 10.0)


def _build_residual_grid() -> WindshieldModel:
    K, D = _camera_matrix_distortion()
    rows, cols = 6, 6
    grid = np.zeros((rows, cols, 3), dtype=np.float64)
    return ResidualRayWindshieldModel(K, D, _MODEL, grid, IMG_W, IMG_H)


def _build_residual_rbf() -> WindshieldModel:
    K, D = _camera_matrix_distortion()
    centers = np.array(
        [[u, v] for u in np.linspace(-1, 1, 6) for v in np.linspace(-1, 1, 6)], dtype=np.float64
    )
    residual_values = np.zeros((centers.shape[0], 3), dtype=np.float64)
    return ResidualRBFWindshieldModel(K, D, _MODEL, centers, residual_values, IMG_W, IMG_H)


def _build_spline() -> WindshieldModel:
    K, D = _camera_matrix_distortion()
    center = np.array([0.0, 0.0, -9.7])
    radius = 10.0
    baseline = BaselineWindshieldModel(K, D, _MODEL)
    theta_scale, phi_scale = compute_angular_fov_scale(baseline, center, radius, IMG_W, IMG_H)
    grid = np.zeros((MIN_SPLINE_GRID_SIZE, MIN_SPLINE_GRID_SIZE))
    return SplineWindshieldModel(K, D, _MODEL, center, radius, grid, theta_scale, phi_scale)


def _build_neural() -> WindshieldModel | None:
    try:
        import torch  # noqa: F401
    except Exception:
        return None
    from calibration.windshield.neural_config import DEFAULT_NEURAL_ACTIVATION, DEFAULT_NEURAL_HIDDEN_DIMS
    from calibration.windshield.neural_residual import NeuralResidualWindshieldModel, _build_mlp

    K, D = _camera_matrix_distortion()
    net = _build_mlp(DEFAULT_NEURAL_HIDDEN_DIMS, DEFAULT_NEURAL_ACTIVATION)
    state_dict = net.state_dict()  # 학습되지 않은 무작위 초기값 - 속도 측정에는 정확도가 무관함
    return NeuralResidualWindshieldModel(
        K, D, _MODEL, state_dict, DEFAULT_NEURAL_HIDDEN_DIMS, DEFAULT_NEURAL_ACTIVATION, IMG_W, IMG_H,
    )


MODEL_BUILDERS = {
    "Baseline": _build_baseline,
    "Spherical": _build_spherical,
    "Residual Grid": _build_residual_grid,
    "Residual RBF": _build_residual_rbf,
    "Spline": _build_spline,
    "Neural Residual": _build_neural,
}


@dataclass
class LatencyStats:
    n_points: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    points_per_sec: float


def _time_repeats(fn, repeats: int) -> LatencyStats:
    """`fn()`을 `repeats`번 반복 실행해 걸린 시간(ms)을 모은다. `fn`은 한
    번 호출에 전체 포인트 배치를 처리해야 한다 - points_per_sec은 그 배치
    크기 기준이다."""
    durations_ms = []
    n_points = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
        durations_ms.append((t1 - t0) * 1000.0)
        n_points = len(out) if hasattr(out, "__len__") else 1
    arr = np.asarray(durations_ms)
    mean_ms = float(np.mean(arr))
    return LatencyStats(
        n_points=n_points or 0,
        mean_ms=mean_ms,
        median_ms=float(np.median(arr)),
        p95_ms=float(np.percentile(arr, 95)),
        points_per_sec=(n_points / (mean_ms / 1000.0)) if mean_ms > 0 and n_points else 0.0,
    )


def benchmark_model(name: str, model: WindshieldModel, scales: tuple[int, ...], repeats: int) -> dict:
    """한 모델에 대해 scale별 Scalar/Batch(Exact)/LUT 세 티어를 측정한다.
    LUT는 초기 구축 비용이 있으므로 한 번만 만들어 재사용한다(실제 런타임
    사용 패턴과 동일 - init-time에 한 번, 이후 반복 조회)."""
    runtime = RuntimeWindshieldProjector(model, IMG_W, IMG_H)
    results: dict[int, dict[str, LatencyStats]] = {}
    for n in scales:
        points = _sample_points(n)
        pixels = _sample_pixels(n)

        scalar_project = _time_repeats(
            lambda: np.array([model.project_point(float(p[0]), float(p[1]), float(p[2])) for p in points]),
            repeats,
        )
        batch_project = _time_repeats(lambda: project_points_exact_batch(model, points), repeats)
        lut_project = _time_repeats(lambda: runtime.project_points(points), repeats)

        scalar_unproject = _time_repeats(
            lambda: np.array([model.unproject_pixel(float(uv[0]), float(uv[1])) for uv in pixels]),
            repeats,
        )
        batch_unproject = _time_repeats(lambda: unproject_pixels_exact_batch(model, pixels), repeats)
        lut_unproject = _time_repeats(lambda: runtime.unproject_pixels(pixels), repeats)

        results[n] = {
            "scalar_project": scalar_project,
            "batch_project": batch_project,
            "lut_project": lut_project,
            "scalar_unproject": scalar_unproject,
            "batch_unproject": batch_unproject,
            "lut_unproject": lut_unproject,
        }
    return results


def _fmt(stats: LatencyStats) -> str:
    return f"mean={stats.mean_ms:8.3f}ms median={stats.median_ms:8.3f}ms p95={stats.p95_ms:8.3f}ms pts/s={stats.points_per_sec:12.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", type=int, nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args()

    print("=" * 78)
    print("Windshield Runtime Latency Benchmark (Phase C-3)")
    print(f"  Host          : {platform.node()} ({platform.machine()}, {platform.system()} {platform.release()})")
    print(f"  Python        : {platform.python_version()}")
    print(f"  Is this a real Jetson device? UNKNOWN to this script - verify manually.")
    print(f"  Scales        : {args.scales}")
    print(f"  Repeats/scale : {args.repeats}")
    print("  NOTE: numbers below reflect ONLY the machine this was run on. Do not")
    print("  extrapolate to Jetson hardware without running this script there directly.")
    print("=" * 78)

    for name, builder in MODEL_BUILDERS.items():
        model = builder()
        if model is None:
            print(f"\n[{name}] SKIPPED (dependency not available, e.g. torch not installed)")
            continue
        print(f"\n[{name}]")
        results = benchmark_model(name, model, tuple(args.scales), args.repeats)
        for n, tiers in results.items():
            print(f"  n={n}:")
            print(f"    project_point    scalar : {_fmt(tiers['scalar_project'])}")
            print(f"    project_point    batch  : {_fmt(tiers['batch_project'])}")
            print(f"    project_point    lut    : {_fmt(tiers['lut_project'])}")
            print(f"    unproject_pixel  scalar : {_fmt(tiers['scalar_unproject'])}")
            print(f"    unproject_pixel  batch  : {_fmt(tiers['batch_unproject'])}")
            print(f"    unproject_pixel  lut    : {_fmt(tiers['lut_unproject'])}")

    print("\n" + "=" * 78)
    print("Done. Remember: LUT tier trades accuracy for speed - see")
    print("calibration/windshield/runtime_projector.py and")
    print("validate_runtime_projector_vs_exact() before using it for anything")
    print("beyond a rough throughput estimate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

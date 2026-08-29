"""
scripts/diagnose_lidar_plane_selection.py
=============================================

AUTO ROI가 실제 캘리브레이션 보드 대신 엉뚱한 평면(바닥/벽/차체 등)을 고르고
있는 게 아닌지 직접 확인하는 진단 스크립트.

확인 항목 (요청받은 3가지 그대로):
  1) 로드된 포인트클라우드의 실제 포인트 개수와, NaN/Inf/원점 패딩 포인트 비율.
     "ROI Points"가 이미지 해상도와 우연히 같은 값(예: 1280*720=921600)으로
     찍힌다면, 그게 진짜 유효 포인트 수인지 아니면 무효 리턴이 안 걸러진
     것인지 여기서 바로 드러남. (camera_lidar.lidar_detector는 이제 NaN/Inf는
     자동으로 걸러내지만, 유효한데 전부 (0,0,0)인 패딩 포인트는 걸러내지
     않으므로 이 스크립트가 별도로 집계함.)
  2) RANSAC이 실제로 시도한 매 평면 후보(peel 순서대로)의 centroid/normal/
     inlier 수/평면 내 XY extent를 전부 출력. 카메라 앞쪽, 보드 크기(약
     0.5x0.4m 안팎) 평면인지, 훨씬 크고 먼 환경 평면(바닥/벽)인지 여기서
     판단 가능.
  3) 각 후보의 inlier 포인트를 후보별 .pcd 파일로 저장. CloudCompare, PCL
     viewer, rviz 등 기존에 쓰던 3D 뷰어로 열어서 실제로 보드 모양인지 눈으로
     확인할 수 있음 (candidate_00_..., candidate_01_... 순서로 저장되며,
     파일명에 채택 여부/상태가 붙음).

사용법 (레포 루트에서 실행):
    # 단일 .pcd/.ply 프레임 파일
    python scripts/diagnose_lidar_plane_selection.py --pcd frame_0001.pcd \
        --out-dir ./lidar_diag

    # rosbag (rosbag1 .bag 또는 rosbag2 디렉터리) 안의 특정 프레임
    python scripts/diagnose_lidar_plane_selection.py --rosbag my.bag \
        --topic /lidar/points --frame-index 0 --out-dir ./lidar_diag

    # target_config.yaml을 바꿔 쓰는 경우
    python scripts/diagnose_lidar_plane_selection.py --pcd frame_0001.pcd \
        --target-config target_config.yaml --max-planes 20
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# 이 스크립트는 camera_calibrator/scripts/ 아래에 있으므로, 패키지를 찾으려면
# 부모(레포 루트) 디렉터리를 sys.path에 넣어야 한다.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from camera_lidar.lidar_detector import (  # noqa: E402
        PlaneCandidateInfo,
        detect_lidar_target_auto,
    )
    from camera_lidar.target_config import TargetConfig, load_target_config  # noqa: E402
    from camera_lidar.types import PointCloudFrame  # noqa: E402
    from input.lidar import load_lidar_from_rosbag, read_pcd, read_ply_ascii  # noqa: E402
except ModuleNotFoundError as e:
    print(
        f"[오류] 필요한 패키지를 찾을 수 없습니다 ({e}).\n"
        "레포 루트(camera_calibrator/)에서 python scripts/diagnose_lidar_plane_selection.py로 실행하세요."
    )
    sys.exit(1)


def _write_pcd_ascii(path: str, points: np.ndarray) -> None:
    """Minimal ASCII PCD writer (x y z 필드만) -- CloudCompare/PCL viewer/
    rviz 등 대부분의 뷰어가 바로 열 수 있는 가장 단순한 포맷."""
    n = points.shape[0]
    with open(path, "w") as f:
        f.write(
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z\n"
            "SIZE 4 4 4\n"
            "TYPE F F F\n"
            "COUNT 1 1 1\n"
            f"WIDTH {n}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {n}\n"
            "DATA ascii\n"
        )
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def _load_points(args: argparse.Namespace) -> np.ndarray:
    if args.pcd:
        ext = os.path.splitext(args.pcd)[1].lower()
        if ext == ".pcd":
            pts = read_pcd(args.pcd)
        elif ext == ".ply":
            pts = read_ply_ascii(args.pcd)
        else:
            print(f"[오류] 지원하지 않는 확장자: {ext!r} (.pcd 또는 .ply만 지원)")
            sys.exit(1)
        return np.asarray(pts[:, :3], dtype=np.float64)

    if args.rosbag:
        if not args.topic:
            print("[오류] --rosbag을 쓸 때는 --topic도 반드시 지정해야 합니다.")
            sys.exit(1)
        result = load_lidar_from_rosbag(args.rosbag, sensor_spec=_dummy_sensor_spec(), topic=args.topic)
        if args.frame_index >= len(result.frames):
            print(f"[오류] frame-index={args.frame_index}, 하지만 토픽에 프레임이 {len(result.frames)}개뿐입니다.")
            sys.exit(1)
        frame = result.frames[args.frame_index]
        for w in result.warnings:
            print(f"[경고] {w}")
        return np.asarray(frame.load()[:, :3], dtype=np.float64)

    print("[오류] --pcd 또는 --rosbag 중 하나를 지정해야 합니다.")
    sys.exit(1)


def _dummy_sensor_spec():
    from input.lidar import LidarSensorSpec
    return LidarSensorSpec()


def _report_point_validity(raw_points: np.ndarray) -> np.ndarray:
    n_total = raw_points.shape[0]
    finite_mask = np.all(np.isfinite(raw_points), axis=1)
    n_finite = int(finite_mask.sum())
    origin_mask = np.all(np.abs(raw_points) < 1e-9, axis=1) & finite_mask
    n_origin = int(origin_mask.sum())

    print("=" * 78)
    print("[1] 포인트 유효성 점검")
    print("=" * 78)
    print(f"  전체 로드된 포인트 수         : {n_total}")
    print(f"  유한(NaN/Inf 아님) 포인트 수   : {n_finite}  ({n_finite / max(n_total, 1) * 100:.1f}%)")
    print(f"  정확히 (0,0,0)인 포인트 수    : {n_origin}  ({n_origin / max(n_total, 1) * 100:.1f}%)"
          + ("   <- 무효 리턴 패딩(organized cloud)일 가능성" if n_origin > n_total * 0.05 else ""))
    if n_total in (1280 * 720, 640 * 480, 1920 * 1080):
        print(f"  ※ 포인트 수 {n_total}가 이미지 해상도와 정확히 일치합니다 -- "
              f"organized point cloud(무효 리턴도 포함된 grid)일 가능성이 높습니다.")
    print(f"  (camera_lidar.lidar_detector는 이제 NaN/Inf 포인트를 detect_lidar_target*"
          f" 내부에서 자동으로 걸러냅니다. 정확히 (0,0,0)인 유효-형식-이지만-무효-리턴인 점은"
          f" 걸러지지 않으므로, 비중이 크다면 로더 단계에서 별도 필터링을 추가하는 게 좋습니다.)")
    return raw_points[finite_mask]


def _report_candidate(info: PlaneCandidateInfo) -> None:
    cx, cy, cz = info.centroid
    nx, ny, nz = info.normal
    ex, ey = info.extent_xy
    range_m = float(np.linalg.norm(info.centroid))
    status = "EXTENT-REJECTED (보드보다 훨씬 큼 -> 경계 추적 안 함)" if info.extent_rejected else "경계 추적 진행됨"
    print(f"\n  --- candidate {info.index} ---")
    print(f"    centroid       : ({cx:.3f}, {cy:.3f}, {cz:.3f}) m   [센서로부터 거리 {range_m:.3f} m]")
    print(f"    normal         : ({nx:.4f}, {ny:.4f}, {nz:.4f})")
    print(f"    inliers        : {info.inlier_count}  (전체의 {info.inlier_ratio * 100:.2f}%)")
    print(f"    평면 내 extent  : {ex:.3f} m x {ey:.3f} m")
    print(f"    상태           : {status}")
    if info.stage is not None:
        print(f"    boundary points: {info.stage.boundary_point_count}")
        print(f"    circle 후보 수  : {info.stage.circle_candidate_count}")
        print(f"    valid circles  : {info.stage.valid_circle_count} / 4")
        if info.stage.failure_reason is not None:
            print(f"    실패 사유       : {info.stage.failure_reason}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pcd", help="단일 .pcd 또는 .ply 프레임 파일")
    src.add_argument("--rosbag", help="rosbag1 .bag 파일 또는 rosbag2 디렉터리")
    ap.add_argument("--topic", help="--rosbag 사용 시 PointCloud2 토픽 이름")
    ap.add_argument("--frame-index", type=int, default=0, help="--rosbag 사용 시 몇 번째 프레임을 볼지 (기본 0)")
    ap.add_argument("--target-config", help="target_config.yaml 경로 (생략 시 기본 TargetConfig() 사용)")
    ap.add_argument("--max-planes", type=int, default=20, help="AUTO ROI가 시도할 최대 평면 개수 (기본 20)")
    ap.add_argument("--out-dir", help="후보별 inlier 포인트를 .pcd로 저장할 디렉터리 (생략 시 저장 안 함)")
    args = ap.parse_args()

    raw_points = _load_points(args)
    points = _report_point_validity(raw_points)

    target = load_target_config(args.target_config) if args.target_config else TargetConfig()
    cloud = PointCloudFrame(timestamp=0.0, points=points, frame_id="diagnostic")

    candidates: list[PlaneCandidateInfo] = []
    result = detect_lidar_target_auto(
        cloud, target, max_planes=args.max_planes, on_plane_candidate=candidates.append,
    )

    print("\n" + "=" * 78)
    print(f"[2] 평면 후보 {len(candidates)}개 상세 (peel 순서대로)")
    print("=" * 78)
    board_w = target.delta_width_circles + 2 * target.circle_radius
    board_h = target.delta_height_circles + 2 * target.circle_radius
    print(f"  (참고: target_config 기준 보드 크기 대략 {board_w:.2f} m x {board_h:.2f} m)")
    for info in candidates:
        _report_candidate(info)

    print("\n" + "=" * 78)
    print("[3] 최종 선택 결과")
    print("=" * 78)
    print(f"  success            : {result.success}")
    print(f"  failure_reason     : {result.failure_reason}")
    print(f"  selected_plane_index: {result.selected_plane_index}")
    print(f"  valid_circle_count : {result.valid_circle_count} / 4")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        print("\n" + "=" * 78)
        print(f"[3] 후보별 포인트를 {args.out_dir} 에 .pcd로 저장 중...")
        print("=" * 78)
        for info in candidates:
            status = "rejected" if info.extent_rejected else (
                "success" if (info.stage and info.stage.success) else "circles_not_found"
            )
            selected_tag = "_SELECTED" if info.index == result.selected_plane_index else ""
            fname = f"candidate_{info.index:02d}_{status}{selected_tag}.pcd"
            out_path = os.path.join(args.out_dir, fname)
            _write_pcd_ascii(out_path, info.points)
            print(f"  {fname}  ({info.points.shape[0]} points)")
        print("\n  CloudCompare나 PCL viewer, rviz 등으로 각 파일을 열어서 실제 보드 모양인지 확인하세요.")
        print(f"  보드는 대략 {board_w:.2f} x {board_h:.2f} m의 평평한 사각형이고, 지름 "
              f"{target.circle_radius * 2:.2f} m짜리 구멍 4개가 코너 쪽에 뚫려 있어야 합니다.")


if __name__ == "__main__":
    main()

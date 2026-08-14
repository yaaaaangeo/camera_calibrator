"""
diagnose_edge_rms.py
=====================

camera_calibrator 프로젝트의 두 "가장자리" 정의(코너 낱개 기반 Coverage Map vs
프레임 중심 1점 기반 classify_regions)가 실제로 얼마나 다른 결과를 내는지
직접 검증하는 진단 스크립트.

사용법:
    python diagnose_edge_rms.py <이미지 폴더> --squares-x 11 --squares-y 8 \
        --square-size 0.020 --marker-size 0.015 --width 640 --height 480

출력:
    1) 프레임별 board_center_px 좌표와, classify_regions() 기준으로 어느
       영역(center/left/right/top/bottom/corner)에 속하는지
    2) 영역별 프레임 개수 집계 (Edge RMS가 왜 N/A인지 여기서 바로 보임)
    3) 비교용으로 4x4 Coverage Map(코너 낱개 기반) 집계도 함께 출력
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

# 이 스크립트를 camera_calibrator 저장소 루트(예: ~/camera_calibrator)에 놓고
# 실행한다고 가정한다. sys.path에 스크립트 자신의 위치를 넣어서, repo 루트가
# 아닌 다른 폴더에서 실행해도 calibration 패키지를 찾을 수 있게 한다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from calibration.detector import build_charuco_detector, detect_charuco  # noqa: E402
    from calibration.models.common import classify_regions  # noqa: E402
except ModuleNotFoundError as e:
    print(
        f"[오류] calibration 패키지를 찾을 수 없습니다 ({e}).\n"
        "이 스크립트를 camera_calibrator 저장소 루트에 저장한 뒤 실행하세요.\n"
        "예: ~/camera_calibrator/diagnose_edge_rms.py"
    )
    sys.exit(1)


def build_board(squares_x: int, squares_y: int, square_size: float, marker_size: float, dict_name: str):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_size, marker_size, dictionary)
    return board


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir", help="이미지들이 들어있는 폴더 (jpg/png)")
    ap.add_argument("--squares-x", type=int, required=True)
    ap.add_argument("--squares-y", type=int, required=True)
    ap.add_argument("--square-size", type=float, required=True, help="미터 단위")
    ap.add_argument("--marker-size", type=float, required=True, help="미터 단위")
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--dict", default="DICT_4X4_100")
    ap.add_argument("--grid-rows", type=int, default=4)
    ap.add_argument("--grid-cols", type=int, default=4)
    args = ap.parse_args()

    board = build_board(args.squares_x, args.squares_y, args.square_size, args.marker_size, args.dict)
    detector = build_charuco_detector(board)

    paths = sorted(
        glob.glob(os.path.join(args.image_dir, "*.jpg"))
        + glob.glob(os.path.join(args.image_dir, "*.jpeg"))
        + glob.glob(os.path.join(args.image_dir, "*.png"))
    )
    if not paths:
        print(f"이미지가 없습니다: {args.image_dir}")
        return

    w, h = args.width, args.height
    region_counts: dict[str, int] = {
        "center": 0, "left": 0, "right": 0, "top": 0, "bottom": 0, "corner": 0,
    }
    grid_counts = np.zeros((args.grid_rows, args.grid_cols), dtype=int)
    total_frames = 0
    frame_rows = []

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        det = detect_charuco(img, board, image_id=os.path.basename(p), detector=detector)
        if not det.success or det.board_center_px is None:
            frame_rows.append((os.path.basename(p), None, None, "검출 실패"))
            continue

        total_frames += 1
        cx, cy = det.board_center_px
        regions = classify_regions(cx, cy, w, h)
        for r in regions:
            region_counts[r] += 1
        frame_rows.append((os.path.basename(p), round(cx, 1), round(cy, 1), ",".join(regions)))

        # 코너 낱개 기반 4x4 grid (Coverage Map과 동일 로직)
        cell_w, cell_h = w / args.grid_cols, h / args.grid_rows
        for x, y in det.corners.reshape(-1, 2):
            col = min(int(x // cell_w), args.grid_cols - 1)
            row = min(int(y // cell_h), args.grid_rows - 1)
            if 0 <= row < args.grid_rows and 0 <= col < args.grid_cols:
                grid_counts[row, col] += 1

    print("=" * 70)
    print(f"총 {len(paths)}장 중 검출 성공 {total_frames}장")
    print("=" * 70)
    print("\n[프레임별 board_center_px -> classify_regions() 결과]")
    print(f"{'파일명':<30}{'cx':>8}{'cy':>8}   영역")
    for name, cx, cy, regions in frame_rows:
        cx_s = f"{cx}" if cx is not None else "-"
        cy_s = f"{cy}" if cy is not None else "-"
        print(f"{name:<30}{cx_s:>8}{cy_s:>8}   {regions}")

    print("\n[영역별 프레임 개수] (Edge RMS = left/right/top/bottom/corner 평균, center는 제외됨)")
    for k, v in region_counts.items():
        marker = "  <- Edge RMS에 포함 안 됨" if k == "center" else ""
        print(f"  {k:<8}: {v:>3}장{marker}")

    edge_frame_count = sum(v for k, v in region_counts.items() if k != "center")
    print(f"\n  => center를 제외한 edge 계열 영역에 걸린 프레임(중복 포함): {edge_frame_count}장")
    if edge_frame_count == 0:
        print("  => 이래서 Edge RMS가 N/A로 나옵니다: 모든 프레임의 보드 중심이 이미지 중앙 1/3x1/3 안에만 있음.")
    else:
        print("  => edge 영역에 걸리는 프레임이 있는데도 N/A라면 다른 원인(검출 실패, id 불일치 등)일 수 있습니다.")

    print(f"\n[비교: 4x4 Coverage Map (코너 낱개 기준, {args.grid_rows}x{args.grid_cols})]")
    for r in range(args.grid_rows):
        print("  " + " ".join(f"{grid_counts[r, c]:>5}" for c in range(args.grid_cols)))
    print(
        "\n  (Coverage Map은 '코너 점' 단위 통계라 보드가 크면 가장자리 셀에도 값이 찍히지만,\n"
        "   Edge RMS는 '프레임 중심 1점'만 보므로 서로 다른 결론이 날 수 있습니다.)"
    )


if __name__ == "__main__":
    main()

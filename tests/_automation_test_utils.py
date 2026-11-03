"""Shared helpers for tests/test_automation_*.py.

물리적으로 일관된 합성 ChArUco 이미지 렌더러: 알려진 K, D, 보드 자세로
각 출력 픽셀의 광선을 보드 평면과 교차시켜 텍스처를 샘플링한다. 기존
conftest의 합성 데이터(이미지마다 임의 homography)는 배선 확인용이라 calibration
결과가 물리적으로 의미가 없으므로, 자동 판정(PASS 경로)·재현성 검증에는 이
렌더러를 쓴다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMG_W, IMG_H = 1280, 720
TRUE_K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]])
TRUE_D = np.array([-0.12, 0.05, 0.0, 0.0, 0.0])

SQUARES_X, SQUARES_Y = 7, 5
SQUARE_SIZE, MARKER_SIZE = 0.04, 0.03
DICTIONARY = "DICT_5X5_100"
_TEXTURE_PX_PER_M = 4000


def pattern_section() -> dict:
    return {
        "type": "charuco",
        "squares_x": SQUARES_X,
        "squares_y": SQUARES_Y,
        "square_size": SQUARE_SIZE,
        "marker_size": MARKER_SIZE,
        "dictionary": DICTIONARY,
    }


def _board_texture() -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_SIZE, MARKER_SIZE, dictionary)
    size = (int(SQUARES_X * SQUARE_SIZE * _TEXTURE_PX_PER_M), int(SQUARES_Y * SQUARE_SIZE * _TEXTURE_PX_PER_M))
    image = board.generateImage(size, marginSize=0)
    # 인쇄물처럼 30~225 범위로 대비를 낮춰 기존 노출/포화 선별 기준에 걸리지 않게 한다.
    return (30 + image.astype(np.float32) * (195.0 / 255.0)).astype(np.uint8)


def _render(texture: np.ndarray, R: np.ndarray, t: np.ndarray, background: int = 110) -> np.ndarray:
    u, v = np.meshgrid(np.arange(IMG_W, dtype=np.float32), np.arange(IMG_H, dtype=np.float32))
    pixels = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(pixels, TRUE_K, TRUE_D).reshape(-1, 2)
    rays = np.concatenate([normalized, np.ones((normalized.shape[0], 1), np.float32)], axis=1)
    normal = R[:, 2]
    depth = (normal @ t) / (rays @ normal)
    board_xy = (rays * depth[:, None] - t[None, :]) @ R  # R^T (P - t)
    map_x = (board_xy[:, 0] * _TEXTURE_PX_PER_M).reshape(IMG_H, IMG_W).astype(np.float32)
    map_y = (board_xy[:, 1] * _TEXTURE_PX_PER_M).reshape(IMG_H, IMG_W).astype(np.float32)
    behind = (depth <= 0).reshape(IMG_H, IMG_W)
    map_x[behind] = -1
    map_y[behind] = -1
    return cv2.remap(texture, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=background)


def _sample_poses(count: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    bw, bh = SQUARES_X * SQUARE_SIZE, SQUARES_Y * SQUARE_SIZE
    center = np.array([bw / 2, bh / 2, 0.0])
    outline = np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]], dtype=float)
    grid = [(x, y) for y in (0.2, 0.5, 0.8) for x in (0.18, 0.5, 0.82)]
    poses: list[tuple[np.ndarray, np.ndarray]] = []
    while len(poses) < count:
        gx, gy = grid[len(poses) % len(grid)]
        u = gx * IMG_W + rng.uniform(-40, 40)
        v = gy * IMG_H + rng.uniform(-30, 30)
        z = rng.uniform(0.45, 0.9)
        rvec = np.deg2rad([rng.uniform(-30, 30), rng.uniform(-30, 30), rng.uniform(-15, 15)])
        R, _ = cv2.Rodrigues(rvec)
        t = z * (np.linalg.inv(TRUE_K) @ np.array([u, v, 1.0])) - R @ center
        projected, _ = cv2.projectPoints(outline, cv2.Rodrigues(R)[0], t, TRUE_K, TRUE_D)
        p = projected.reshape(-1, 2)
        if (p[:, 0] > 15).all() and (p[:, 0] < IMG_W - 15).all() and (p[:, 1] > 15).all() and (p[:, 1] < IMG_H - 15).all():
            poses.append((R, t))
    return poses


def render_consistent_charuco_dataset(out_dir: Path, count: int = 24, seed: int = 7) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    texture = _board_texture()
    paths = []
    for index, (R, t) in enumerate(_sample_poses(count, seed)):
        image = cv2.GaussianBlur(_render(texture, R, t), (3, 3), 0.6)
        path = out_dir / f"view_{index:02d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        paths.append(path)
    return paths


def write_blank_images(out_dir: Path, count: int, size: tuple[int, int] = (64, 48), prefix: str = "blank") -> list[Path]:
    """패턴이 없는 작은 이미지 - 입력 검증/실패 경로 테스트용 (빠름)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = out_dir / f"{prefix}_{index:02d}.png"
        cv2.imwrite(str(path), np.full((size[1], size[0], 3), 127, dtype=np.uint8))
        paths.append(path)
    return paths


def base_profile(**overrides) -> dict:
    profile = {
        "profile_version": 1,
        "camera": {"model": "TEST-CAM"},
        "pattern": pattern_section(),
        "calibration": {"test_ratio": 0.25, "seed": 42, "jobs": 1},
        "image_selection": {"enabled": True},
        "acceptance": {},
        "output": {"exports": ["opencv", "ros", "report", "json", "csv"]},
    }
    for key, value in overrides.items():
        profile[key] = value
    return profile

"""
scripts/generate_kalibr_aprilgrid_fixture.py
==================================================

tests/assets/aprilgrid/kalibr_aprilgrid_6x6.png 생성 스크립트.

IMPORTANT - 이 스크립트가 만드는 이미지는 Kalibr(github.com/ethz-asl/kalibr)를
실제로 설치/실행해서 뽑은 원본 출력이 아니다. 이 환경에는 Kalibr도, 실제
카메라도 없다. 대신 Kalibr의 공개된 타겟 생성 규칙(kalibr_create_target_pdf.py
및 aprilgrid.yaml 스키마)을 최대한 충실히 재현한다:

  - 태그 계열: tag36h11 (Kalibr AprilGrid 기본값), OpenCV DICT_APRILTAG_36h11로 렌더링.
    36h11 codeword는 표준 규격이라 생성기가 달라도 태그 내용 자체는 동일하다.
  - ID 배치: row-major (왼쪽 위 0번, 오른쪽으로 증가, 다음 줄로 이동) - Kalibr
    AprilGrid와 이 프로젝트(calibration/detector.py: build_aprilgrid_dictionary
    docstring)가 공통으로 쓰는 관례.
  - 간격 규칙: Kalibr의 tagSpacing은 "태그 한 변 길이(tagSize) 대비 여백 비율"이다.
    export/kalibr.py의 build_kalibr_aprilgrid_target()도 동일하게
    tagSpacing = (square_size - marker_size) / marker_size 로 변환한다 - 이
    스크립트도 같은 공식으로 픽셀 간격을 정한다.
  - 흰 배경 위에 검은 테두리가 포함된 태그 이미지(cv2.aruco.generateImageMarker,
    Kalibr PDF의 흰 종이 위 검은 마커와 동일한 명암 배치)를 격자로 배치한다.

실제로 다른 점(투명하게 명시): Kalibr는 LaTeX/matplotlib로 PDF를 만들고
사용자가 그걸 인쇄해서 실제 카메라로 촬영한다 - 그 과정에서 생기는 인쇄
DPI/서브픽셀 안티에일리어싱/렌즈 왜곡/조명 편차는 이 합성 이미지에는 없다.
그래서 "OpenCV AprilTag36h11 detector가 Kalibr 스타일 타겟의 태그 내용과
row-major ID 배치를 검출할 수 있는가"는 검증하지만, "실제 인쇄물+카메라
촬영본까지 완벽 호환되는가"는 이 fixture만으로는 완전히 보장하지 못한다.
tests/assets/aprilgrid/README.md에 실제 Kalibr 산출물로 교체하는 방법을
적어뒀다 - 파일명만 그대로 바꿔치기하면 테스트 코드는 수정할 필요 없다.

재실행 가능 - 매번 같은 파일을 덮어쓴다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "assets" / "aprilgrid"
OUT_PATH = OUT_DIR / "kalibr_aprilgrid_6x6.png"

TAG_FAMILY_DICT = "DICT_APRILTAG_36h11"
SQUARES_X = 6
SQUARES_Y = 6
TAG_SIZE_M = 0.05      # marker_size (물리적 태그 한 변)
TAG_SPACING_RATIO = 0.3  # Kalibr 기본값 - export/kalibr.py의 tagSpacing 공식과 동일
SQUARE_SIZE_M = TAG_SIZE_M * (1.0 + TAG_SPACING_RATIO)  # tag origin 간 pitch

TAG_PX = 140
MARGIN_PX = 110
PITCH_PX = int(round(TAG_PX * (SQUARE_SIZE_M / TAG_SIZE_M)))  # = TAG_PX * (1 + spacing ratio)


def generate() -> np.ndarray:
    dict_id = getattr(cv2.aruco, TAG_FAMILY_DICT)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

    width = MARGIN_PX * 2 + (SQUARES_X - 1) * PITCH_PX + TAG_PX
    height = MARGIN_PX * 2 + (SQUARES_Y - 1) * PITCH_PX + TAG_PX
    gray = np.full((height, width), 255, dtype=np.uint8)

    total_tags = SQUARES_X * SQUARES_Y
    for marker_id in range(total_tags):
        row = marker_id // SQUARES_X
        col = marker_id % SQUARES_X
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, TAG_PX)
        y = MARGIN_PX + row * PITCH_PX
        x = MARGIN_PX + col * PITCH_PX
        gray[y:y + TAG_PX, x:x + TAG_PX] = marker

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image = generate()
    ok = cv2.imwrite(str(OUT_PATH), image)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {OUT_PATH}")
    print(f"wrote {OUT_PATH} ({image.shape[1]}x{image.shape[0]})")


if __name__ == "__main__":
    main()

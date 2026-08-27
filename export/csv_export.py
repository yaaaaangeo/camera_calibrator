"""
camera_calibrator.export.csv_export
========================================

설계 문서 11번 - CSV export. UI의 Dataset 탭(ui/dataset_view.py) 표와 같은
내용을 스프레드시트(Excel, Google Sheets, pandas)에서 바로 열어 분석할 수
있게 내보낸다 - "나쁜 프레임들의 공통점을 찾아본다"거나 "선명도와 재투영
오차의 상관관계를 그려본다" 같은, 이 앱 UI만으로는 하기 힘든 사후 분석을
사용자가 직접 할 수 있게 하는 게 목적이다.

pandas 의존성을 추가하지 않는다 - 표준 라이브러리 csv 모듈로 충분하고,
이 프로젝트의 "필요 없는 의존성은 안 늘린다" 원칙과도 맞는다.
"""

from __future__ import annotations

import csv
from pathlib import Path

from calibration.types import Dataset

_FIELDNAMES = [
    "image_id", "path", "status", "num_corners", "sharpness", "brightness",
    "board_area_ratio", "board_tilt_deg", "reprojection_error_px",
    "quality_detection_score", "quality_geometric_score", "quality_overall_score", "quality_grade",
]


def dataset_to_rows(dataset: Dataset) -> list[dict]:
    """CSV로 쓸 행(dict) 리스트를 만든다. 파일 I/O 없이 이 함수만으로도
    테스트나 다른 용도(예: pandas.DataFrame(dataset_to_rows(...)))에 바로 쓸 수 있다.
    """
    rows = []
    for frame in dataset.frames:
        det = frame.detection
        q = frame.quality
        rows.append({
            "image_id": frame.image_info.image_id,
            "path": frame.image_info.path,
            "status": frame.status.value,
            "num_corners": det.num_corners if det else None,
            "sharpness": frame.image_info.sharpness,
            "brightness": frame.image_info.brightness,
            "board_area_ratio": det.board_area_ratio if det else None,
            "board_tilt_deg": det.board_tilt_deg if det else None,
            "reprojection_error_px": frame.reprojection_error,
            "quality_detection_score": q.detection_score if q else None,
            "quality_geometric_score": q.geometric_score if q else None,
            "quality_overall_score": q.overall_score if q else None,
            "quality_grade": q.grade.value if q else None,
        })
    return rows


def export_csv(dataset: Dataset, path: str) -> str:
    rows = dataset_to_rows(dataset)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path

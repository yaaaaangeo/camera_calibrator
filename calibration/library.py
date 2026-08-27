"""
camera_calibrator.calibration.library
=========================================

카메라(센서)별 과거 캘리브레이션 결과를 디스크에 영구 보관하는 "도서관".

계산이 끝날 때마다 결과를 통째로 복사해 남겨서, 원본 촬영 폴더
(live_captures, rosbag 추출 임시 폴더 등)를 나중에 지우거나 정리해도
Library 탭에서는 계속 그 결과를 재현/조회할 수 있어야 한다. 그래서 이
모듈은 실행마다 별도 폴더에:
    library/<sensor_name>/<timestamp>/
        images/          - 이 실행에 쓰인 이미지 원본 복사본
        project.ccproj   - 전체 CalibrationProject (project_io.py 재사용,
                            이미지 경로는 위 images/ 를 가리키도록 다시 씀 -
                            그래야 나중에 이 프로젝트를 다시 열어 재학습/
                            재검증 등에 쓸 수 있다)
        summary.json     - 목록 화면이 project.ccproj 전체(코너 좌표 등 포함돼
                            무거움)를 매번 파싱하지 않도록 모델별 RMS 같은
                            핵심 수치만 뽑아둔 가벼운 요약
을 남긴다.
"""

from __future__ import annotations

import copy
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from calibration.types import (
    CalibrationProject,
    CalibrationResult,
    CameraConfig,
    CameraModelType,
    Dataset,
    ModelScore,
    PatternConfig,
    ValidationResult,
)
from calibration.project_io import load_project, save_project

LIBRARY_DIR_NAME = "library"


def library_root() -> Path:
    return Path.cwd() / LIBRARY_DIR_NAME


@dataclass
class LibraryModelSummary:
    success: bool = False
    rms_error: float | None = None
    test_rms: float | None = None
    distortion_count: int | None = None
    error_message: str | None = None


@dataclass
class LibraryRunSummary:
    """summary.json과 1:1 대응하는, 목록 화면이 바로 읽는 가벼운 요약."""
    run_dir: str
    sensor_name: str
    created_at: str
    num_images: int
    pattern_type: str
    models: dict[str, LibraryModelSummary] = field(default_factory=dict)
    project_file: str = "project.ccproj"
    sample_image: str | None = None  # run_dir 기준 상대 경로
    note: str = ""  # 사용자가 남긴 짧은 메모 - "이건 어떤 것에 대한 기록인지"


def _sanitize(name: str) -> str:
    keep = "-_.() "
    cleaned = "".join(c if c.isalnum() or c in keep else "_" for c in name)
    return cleaned.strip() or "camera"


def _unique_run_dir(sensor: str) -> Path:
    """타임스탬프(초 단위)가 같은 실행 두 개가 같은 run_dir로 서로 덮어쓰지
    않도록, 이미 폴더가 있으면 뒤에 -2, -3 ...을 붙여 비어있는 경로를 찾는다.
    """
    base = library_root() / sensor / datetime.now().strftime("%Y%m%d_%H%M%S")
    if not base.exists():
        return base
    for suffix in range(2, 1000):
        candidate = base.parent / f"{base.name}-{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"고유한 Library run 폴더를 찾지 못했습니다: {base}")


def save_calibration_run(
    dataset: Dataset,
    camera_config: CameraConfig,
    pattern_config: PatternConfig,
    calibration_results: dict[CameraModelType, CalibrationResult],
    validation_results: dict[CameraModelType, ValidationResult],
    model_scores: list[ModelScore] | None = None,
) -> Path:
    """현재 계산 결과(이미지 포함)를 도서관에 통째로 복사해 저장한다.

    호출부가 들고 있는 dataset은 다른 워크스페이스가 계속 참조 중일 수 있으므로
    건드리지 않는다 - deepcopy한 사본의 이미지 경로만 새 위치로 바꿔 저장한다.
    """
    sensor = _sanitize(camera_config.sensor_name or "camera")
    run_dir = _unique_run_dir(sensor)
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    copied_dataset = copy.deepcopy(dataset)
    sample_image_rel: str | None = None
    for frame in copied_dataset.frames:
        src = Path(frame.image_info.path)
        if not src.exists():
            continue
        dest = images_dir / f"{frame.image_info.image_id}{src.suffix or '.jpg'}"
        try:
            shutil.copy2(src, dest)
        except OSError:
            continue
        frame.image_info.path = str(dest)
        if sample_image_rel is None and frame.detection is not None and frame.detection.success:
            sample_image_rel = str(dest.relative_to(run_dir))

    project = CalibrationProject(
        project_name=sensor,
        camera_config=camera_config,
        pattern_config=pattern_config,
        dataset=copied_dataset,
        calibration_results=calibration_results,
        validation_results=validation_results,
        model_scores=model_scores or [],
    )
    # summary.json을 project.ccproj보다 먼저 만든다 - 코너 좌표까지 포함된
    # project.ccproj 저장(수백 장이면 수 초)이 중간에 실패하거나 스레드가
    # 강제 종료돼도, 목록 화면이 읽는 summary.json이 없는 "반쯤 저장된"
    # 폴더가 남지 않는다.
    summary = _build_summary(
        run_dir, sensor, copied_dataset, pattern_config,
        calibration_results, validation_results, sample_image_rel,
    )
    _write_summary(run_dir, summary)
    save_project(project, str(run_dir / "project.ccproj"))
    return run_dir


def _build_summary(
    run_dir: Path,
    sensor: str,
    dataset: Dataset,
    pattern_config: PatternConfig,
    calibration_results: dict[CameraModelType, CalibrationResult],
    validation_results: dict[CameraModelType, ValidationResult],
    sample_image_rel: str | None,
) -> LibraryRunSummary:
    return LibraryRunSummary(
        run_dir=str(run_dir),
        sensor_name=sensor,
        created_at=datetime.now().isoformat(timespec="seconds"),
        num_images=len(dataset.frames),
        pattern_type=getattr(pattern_config.type, "value", str(pattern_config.type)),
        models={
            # calibration_results/validation_results가 QThread 시그널(Signal(dict))을
            # 건너온 경우, PySide6가 str-Enum(CameraModelType) 키를 평범한 str로
            # 낮춰서 넘기는 경우가 있다 (dict 동등성/해시는 str과 동일해서 .get()
            # 조회는 계속 맞지만 .value 접근은 죽는다) - 항상 다시 enum으로
            # 정규화한 뒤 .value에 접근한다.
            CameraModelType(model).value: LibraryModelSummary(
                success=result.success,
                rms_error=result.rms_error,
                test_rms=(
                    validation_results[model].test_rms
                    if validation_results.get(model) is not None else None
                ),
                distortion_count=(len(result.distortion) if result.distortion is not None else None),
                error_message=result.error_message,
            )
            for model, result in calibration_results.items()
        },
        sample_image=sample_image_rel,
    )


def _write_summary(run_dir: Path, summary: LibraryRunSummary) -> None:
    (run_dir / "summary.json").write_text(
        json.dumps(_summary_to_dict(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _summary_to_dict(summary: LibraryRunSummary) -> dict:
    return {
        "run_dir": summary.run_dir,
        "sensor_name": summary.sensor_name,
        "created_at": summary.created_at,
        "num_images": summary.num_images,
        "pattern_type": summary.pattern_type,
        "project_file": summary.project_file,
        "sample_image": summary.sample_image,
        "note": summary.note,
        "models": {
            name: {
                "success": m.success,
                "rms_error": m.rms_error,
                "test_rms": m.test_rms,
                "distortion_count": m.distortion_count,
                "error_message": m.error_message,
            }
            for name, m in summary.models.items()
        },
    }


def _dict_to_summary(data: dict) -> LibraryRunSummary:
    return LibraryRunSummary(
        run_dir=data.get("run_dir", ""),
        sensor_name=data.get("sensor_name", "camera"),
        created_at=data.get("created_at", ""),
        num_images=data.get("num_images", 0),
        pattern_type=data.get("pattern_type", ""),
        project_file=data.get("project_file", "project.ccproj"),
        sample_image=data.get("sample_image"),
        note=data.get("note", ""),
        models={
            name: LibraryModelSummary(**m) for name, m in data.get("models", {}).items()
        },
    )


def list_cameras() -> list[str]:
    root = library_root()
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def list_runs(sensor_name: str) -> list[LibraryRunSummary]:
    """최신 실행이 먼저 오도록 정렬해서 반환한다 (타임스탬프 폴더명 기준)."""
    camera_dir = library_root() / sensor_name
    if not camera_dir.exists():
        return []
    runs = []
    for run_dir in sorted(camera_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        summary: LibraryRunSummary | None = None
        if summary_path.exists():
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
                summary = _dict_to_summary(data)
            except (OSError, json.JSONDecodeError):
                summary = None
        if summary is None:
            # summary.json이 없거나 깨진 경우 - project.ccproj가 있다면
            # 거기서 다시 만들어 채워 넣는다(자가 복구). 저장 중 강제 종료나
            # 예외로 project.ccproj까지만 저장된 실행 기록도 이걸로 계속 조회
            # 가능하게 남긴다.
            summary = _rebuild_summary_from_project(run_dir, sensor_name)
        if summary is not None:
            runs.append(summary)
    return runs


def _rebuild_summary_from_project(run_dir: Path, sensor_name: str) -> LibraryRunSummary | None:
    project_path = run_dir / "project.ccproj"
    if not project_path.exists():
        return None
    try:
        project, _missing = load_project(str(project_path))
    except Exception:  # noqa: BLE001 - 복구 시도 실패는 이 run을 그냥 건너뛴다
        return None

    sample_image_rel = None
    for frame in project.dataset.frames:
        if frame.detection is not None and frame.detection.success:
            try:
                sample_image_rel = str(Path(frame.image_info.path).relative_to(run_dir))
            except ValueError:
                sample_image_rel = None
            if sample_image_rel is not None:
                break

    summary = _build_summary(
        run_dir, sensor_name, project.dataset, project.pattern_config,
        project.calibration_results, project.validation_results, sample_image_rel,
    )
    try:
        _write_summary(run_dir, summary)
    except OSError:
        pass
    return summary


def load_run_project(run_dir: str) -> tuple[CalibrationProject, list[str]]:
    """이 run의 project.ccproj를 로드한다. 반환값 2번째는 missing_images(project_io.py 규약)."""
    return load_project(str(Path(run_dir) / "project.ccproj"))


def update_run_note(run_dir: str, note: str) -> LibraryRunSummary | None:
    """이 run에 짧은 메모를 남기거나 바꾼다 - "이건 뭐에 대한 기록인지" 목록
    화면에서 바로 보이게 하기 위함. summary.json이 없으면(구버전 기록) 먼저
    project.ccproj에서 다시 만든 뒤 메모를 얹는다.
    """
    run_path = Path(run_dir)
    summary_path = run_path / "summary.json"
    summary: LibraryRunSummary | None = None
    if summary_path.exists():
        try:
            summary = _dict_to_summary(json.loads(summary_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            summary = None
    if summary is None:
        summary = _rebuild_summary_from_project(run_path, run_path.parent.name)
    if summary is None:
        return None
    summary.note = note
    _write_summary(run_path, summary)
    return summary


def delete_run(run_dir: str) -> bool:
    """계산 기록 하나(이미지 사본 포함)를 통째로 삭제한다. 되돌릴 수 없다."""
    run_path = Path(run_dir)
    root = library_root().resolve()
    try:
        resolved = run_path.resolve()
    except OSError:
        return False
    # library/ 바깥 경로를 실수로 넘겨받아도 지우지 않도록 방어.
    if root not in resolved.parents:
        return False
    if not resolved.exists():
        return False
    shutil.rmtree(resolved)
    return True


def delete_camera(sensor_name: str) -> bool:
    """카메라 하나(그 카메라의 모든 계산 기록)를 통째로 삭제한다. 되돌릴 수 없다."""
    camera_path = library_root() / sensor_name
    if not camera_path.exists():
        return False
    shutil.rmtree(camera_path)
    return True

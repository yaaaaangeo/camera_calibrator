"""run_manifest.json: provenance and status record of one automated run.

기존 ``manifest.json``(OutputManager)은 그대로 두고, 자동화 실행 정보는 이
파일에 따로 기록한다. 상태는 ``running`` → ``completed`` | ``failed`` 한
방향으로만 바뀐다. 프로세스가 중간에 죽으면 ``running``으로 남으므로 정상
완료와 구분된다.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any
import uuid

from automation import AUTOMATION_SCHEMA_VERSION

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

REPO_ROOT = Path(__file__).resolve().parent.parent


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def new_run_id(moment: datetime | None = None) -> str:
    moment = moment or utc_now()
    return f"{moment.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"


def atomic_write_text(path: str | Path, text: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)
    return destination


def atomic_write_json(path: str | Path, data: Any) -> Path:
    # 기존 json_safe로 numpy/Enum을 풀고 inf/NaN을 null로 바꿔 표준 JSON만 쓴다
    # (예: 계산 실패 모델의 추천 점수는 내부적으로 inf).
    from calibration.json_utils import json_safe

    text = json.dumps(json_safe(data, ndarray_wrapper=False), ensure_ascii=False, indent=2,
                      default=str, allow_nan=False)
    return atomic_write_text(path, text + "\n")


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(paths: list[str]) -> dict:
    """입력 이미지 파일 내용 해시.

    aggregate는 (파일명, 내용 해시)만으로 만든다 - 폴더를 다른 위치로 옮겨도
    내용이 같으면 같은 값이 나온다. 파일 순서는 CLI가 정렬해서 쓰므로 정렬된
    순서로 합친다.
    """
    files = []
    for p in paths:
        path = Path(p)
        files.append({
            "path": str(path.resolve()),
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    aggregate = hashlib.sha256(
        "\n".join(f"{f['name']}:{f['sha256']}" for f in sorted(files, key=lambda f: f["name"])).encode("utf-8")
    ).hexdigest()
    return {"count": len(files), "aggregate_sha256": aggregate, "files": files}


def detection_fingerprint(dataset) -> str:
    """기존 ``calibration.cache.dataset_fingerprint``(코너/물체점 바이트 포함)를
    해시 문자열로 요약한다. 같은 이미지·같은 패턴 설정이면 같은 값이 나온다."""
    from calibration.cache import dataset_fingerprint

    return hashlib.sha256(repr(dataset_fingerprint(dataset)).encode("utf-8")).hexdigest()


def canonical_sha256(data: Any) -> str:
    return hashlib.sha256(
        json.dumps(data, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _git(args: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True,
    )
    return completed.stdout


def git_info(repo_root: Path = REPO_ROOT) -> dict:
    """코드 버전. 커밋되지 않은 변경이 있으면 dirty=true와 파일 목록을 남긴다 -
    commit SHA만으로는 실제 실행 코드를 재현할 수 없기 때문이다.

    소스 checkout(``repo_root/.git``이 있는 경우)에서 실행할 때만 git을 조회한다.
    ``pip install .``로 설치된 복사본은 git 정보가 없으므로 "unavailable"로 남기고,
    대신 environment_info()의 설치 패키지 버전을 본다. (venv가 저장소 안에 있으면
    site-packages에서 git을 호출해도 저장소 HEAD가 나와 설치본과 다른 코드를
    가리킬 수 있으므로 이를 피한다.)"""
    if not (Path(repo_root) / ".git").exists():
        return {"available": False, "error": f"not a git checkout: {repo_root}"}
    try:
        commit = _git(["rev-parse", "HEAD"], repo_root).strip()
        branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root).strip()
        # porcelain 형식은 "XY path" - 앞 공백이 상태 코드의 일부이므로 strip하지 않는다.
        status = _git(["status", "--porcelain", "--untracked-files=no"], repo_root)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}
    modified = [line[3:] for line in status.splitlines() if len(line) > 3]
    return {
        "available": True,
        "commit_sha": commit,
        "branch": branch,
        "dirty": bool(modified),
        "modified_tracked_files": modified,
    }


def environment_info() -> dict:
    versions: dict[str, str | None] = {}
    for module_name in ("numpy", "scipy", "cv2", "yaml"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", None)
        except ImportError:
            versions[module_name] = None
    from importlib import metadata

    from export.output_manager import APPLICATION_NAME, APPLICATION_VERSION

    try:
        installed = metadata.version("camera-calibrator")
    except metadata.PackageNotFoundError:
        installed = None
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "libraries": versions,
        "application": {"name": APPLICATION_NAME, "version": APPLICATION_VERSION},
        "source_root": str(REPO_ROOT),
        "installed_distribution_version": installed,
    }


class RunManifest:
    """run_manifest.json의 메모리 표현. 값이 바뀔 때마다 디스크에 원자적으로 기록한다."""

    def __init__(self, path: str | Path, data: dict):
        self.path = Path(path)
        self.data = data

    @classmethod
    def create(cls, path: str | Path, *, run_id: str, started_at: datetime, **fields: Any) -> "RunManifest":
        data = {
            "schema_version": AUTOMATION_SCHEMA_VERSION,
            "run_id": run_id,
            "status": STATUS_RUNNING,
            "started_at": iso(started_at),
            "finished_at": None,
            "failure": None,
            "stages": [],
            **fields,
        }
        manifest = cls(path, data)
        manifest.save()
        return manifest

    @property
    def status(self) -> str:
        return self.data["status"]

    def save(self) -> None:
        atomic_write_json(self.path, self.data)

    def update(self, **fields: Any) -> None:
        self.data.update(fields)
        self.save()

    def start_stage(self, name: str) -> None:
        self.data["stages"].append({
            "name": name, "status": STATUS_RUNNING, "started_at": iso(utc_now()),
            "finished_at": None, "error": None,
        })
        self.save()

    def _current(self, name: str) -> dict:
        for stage in reversed(self.data["stages"]):
            if stage["name"] == name:
                return stage
        raise KeyError(name)

    def finish_stage(self, name: str, **details: Any) -> None:
        stage = self._current(name)
        stage.update(status=STATUS_COMPLETED, finished_at=iso(utc_now()), **details)
        self.save()

    def fail_stage(self, name: str, error: str) -> None:
        stage = self._current(name)
        stage.update(status=STATUS_FAILED, finished_at=iso(utc_now()), error=error)
        self.save()

    def mark_completed(self) -> None:
        self.data.update(status=STATUS_COMPLETED, finished_at=iso(utc_now()), failure=None)
        self.save()

    def mark_failed(self, stage: str, error: str) -> None:
        running = [s for s in self.data["stages"] if s["status"] == STATUS_RUNNING]
        for s in running:
            s.update(status=STATUS_FAILED, finished_at=iso(utc_now()), error=s["error"] or error)
        self.data.update(
            status=STATUS_FAILED,
            finished_at=iso(utc_now()),
            failure={"stage": stage, "error": error},
        )
        self.save()

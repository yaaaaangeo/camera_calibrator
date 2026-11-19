"""
export.git_upload
=================

캘리브레이션 결과 폴더를 **결과 전용 Git 저장소**에 올린다 (Camera Intrinsic 과
CAMERA–LiDAR 가 같이 쓴다). GUI 의 [Git 업로드] 버튼이 사람이 확인한 뒤에만 부른다.

동작
  1. 저장소를 ``~/.cache/calibration-tool/results-repo/<주소 해시>/`` 에 clone 해 두고
     (다음부터는 fetch 만), 대상 브랜치의 최신 상태로 맞춘다. 이 체크아웃은 이 모듈
     전용이라 매번 원격 상태로 되돌려도 된다. 브랜치가 원격에 없으면 빈(orphan)
     브랜치로 새로 만든다 — 코드 저장소의 파일이 결과에 섞이지 않게.
  2. 설정(``GitUploadConfig``)에서 고른 파일을 경로 규칙대로 저장소 안에 복사한다.
     기본은 결과값만:
       intrinsic/{vehicle}/{camera}/{session}/camera_<model>.yaml, camera_info.yaml
       extrinsic/{vehicle}/{session}/{camera}/extrinsic.txt, result.yaml
     차량을 골랐으면 차량 정보도 같이 올린다 (``upload_vehicle``):
       vehicles/{vehicle}/vehicle.yaml   카메라 이름·토픽·렌즈 (shared.vehicle)
       vehicles/{vehicle}/lenses.yaml    그 차량 카메라들이 쓰는 렌즈 프리셋 값
     다른 PC 는 [GitHub에서 가져오기] 로 이 둘을 받아 같은 차량 정보를 쓴다 (``fetch_vehicles``).
     원본 캡처(mcap)는 고를 수 없다 - 크기 때문에 로컬 결과 폴더에만 남는다.
  3. 바뀐 것이 있으면 commit 하고 push 한다. 다른 사람이 먼저 push 해서 거부되면
     rebase 후 한 번 더 시도한다.

인증은 사용자의 git 설정(ssh 키, credential helper)을 그대로 쓴다. GUI 에는 터미널이
없으므로 ``GIT_TERMINAL_PROMPT=0`` 으로 비밀번호를 묻지 않고 바로 실패시킨 뒤 안내한다.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from shared.naming import safe_name

CONFIG_PATH = Path.home() / ".config" / "calibration-tool" / "git_upload.json"
CACHE_ROOT = Path.home() / ".cache" / "calibration-tool" / "results-repo"
MAX_FILE_MB = 50  # 이보다 큰 파일은 git 에 올리지 않는다 (원본 캡처 등)
GIT_TIMEOUT_S = 180


class GitUploadError(RuntimeError):
    pass


# 올릴 수 있는 파일 (키 -> 설명). 설정 화면의 체크박스가 이 순서로 나온다.
INTRINSIC_FILES = {
    "models": "모델별 결과 YAML (camera_<model>.yaml)",
    "camera_info": "ROS camera_info.yaml",
    "report_html": "리포트 report.html",
    "report_json": "전체 수치 calibration.json",
    "project": "프로젝트 파일 calibration.ccproj (다시 열기용)",
}
EXTRINSIC_FILES = {
    "extrinsic.txt": "extrinsic.txt (FAST-LIVO2 형식)",
    "result.yaml": "result.yaml (R, t, 4x4)",
    "boards.yaml": "boards.yaml (보드별 구멍 중심)",
    "report.txt": "report.txt (보고서, 텍스트)",
    "report.html": "report.html (보고서, 인트린직 리포트와 같은 형식)",
    "overlay.png": "overlay.png (오버레이 이미지, 수 MB)",
    "session.yaml": "session.yaml (세션 요약, 세션당 1개)",
}
INTRINSIC_PATH_FIELDS = ("vehicle", "camera", "session", "date")
VEHICLES_DIR = "vehicles"  # 저장소 안 차량 정보 위치 (고정)
EXTRINSIC_PATH_FIELDS = ("vehicle", "session", "date", "camera")


@dataclass
class GitUploadConfig:
    remote: str = ""
    branch: str = "main"
    # 커밋 작성자. 비어 있으면 이 PC 의 git 설정(user.name / user.email)을 쓴다.
    author_name: str = ""
    author_email: str = ""
    # 저장소 안 경로 규칙과 올릴 파일. 기본은 결과값만.
    intrinsic_path: str = "intrinsic/{vehicle}/{camera}/{session}"  # 차량 없음 -> no_vehicle
    intrinsic_files: list = field(default_factory=lambda: ["models", "camera_info"])
    intrinsic_recommended_only: bool = False  # 모델 YAML 중 추천 모델 하나만
    extrinsic_path: str = "extrinsic/{vehicle}/{session}/{camera}"
    extrinsic_files: list = field(default_factory=lambda: ["extrinsic.txt", "result.yaml"])
    upload_vehicle: bool = True  # 차량 정보(vehicles/<차량>/vehicle.yaml, lenses.yaml)도 같이


def load_config(path: Path | None = None) -> GitUploadConfig:
    try:
        data = json.loads(Path(path or CONFIG_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return GitUploadConfig()
    config = GitUploadConfig()
    for key, default in asdict(config).items():
        value = data.get(key, default)
        if isinstance(default, bool):
            value = bool(value)
        elif isinstance(default, list):
            value = [str(v) for v in value] if isinstance(value, list) else default
        else:
            value = str(value) if value is not None else default
        setattr(config, key, value)
    config.branch = config.branch or "main"
    return config


def check_path_template(template: str, fields: Iterable[str]) -> str:
    """경로 규칙의 문제를 설명하는 문장, 문제가 없으면 빈 문자열."""
    template = template.strip().strip("/")
    if not template:
        return "경로 규칙이 비어 있습니다"
    allowed = set(fields)
    used = set(re.findall(r"{(\w*)}", template))
    unknown = used - allowed
    if unknown:
        return f"모르는 항목: {', '.join('{' + u + '}' for u in sorted(unknown))} (쓸 수 있는 것: {', '.join('{' + f + '}' for f in fields)})"
    if any(part in ("", ".", "..") for part in template.split("/")):
        return "경로에 빈 칸, '.', '..' 은 쓸 수 없습니다"
    if "{session}" not in template:
        return "{session} 이 없으면 세션끼리 같은 경로에 덮어씁니다"
    return ""


def render_path(template: str, **values: str) -> str:
    text = template.strip().format(**{k: safe_component(v, k) if v else "" for k, v in values.items()})
    return "/".join(part for part in text.split("/") if part)


def connection_check(remote: str) -> str:
    """원격 저장소에 접근되면 빈 문자열, 아니면 이유."""
    try:
        _git(["ls-remote", "--heads", remote.strip()])
    except GitUploadError as exc:
        return str(exc)
    return ""


def git_identity() -> tuple[str, str]:
    """이 PC 의 git 전역 설정에 있는 (user.name, user.email). 없으면 빈 문자열."""
    out = []
    for key in ("user.name", "user.email"):
        try:
            done = subprocess.run(["git", "config", "--global", "--get", key], capture_output=True, text=True, timeout=10)
            out.append(done.stdout.strip() if done.returncode == 0 else "")
        except (OSError, subprocess.SubprocessError):
            out.append("")
    return out[0], out[1]


def save_config(config: GitUploadConfig, path: Path | None = None) -> None:
    path = Path(path or CONFIG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")


def safe_component(text: str, fallback: str = "unnamed") -> str:
    """저장소 경로 한 칸으로 쓸 수 있는 이름 (글자·숫자·-·_·. 만)."""
    return safe_name(text, fallback=fallback, keep_dot=True, strip="._")


@dataclass
class UploadFile:
    source: Path | None
    dest: str  # 저장소 안 경로 (POSIX)
    data: bytes | None = None  # 파일 대신 이 내용을 쓴다 (예: 렌즈 프리셋 묶음)

    @property
    def size(self) -> int:
        return len(self.data) if self.data is not None else self.source.stat().st_size


@dataclass
class UploadPlan:
    files: list[UploadFile] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # 너무 커서 뺀 파일

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)


@dataclass
class UploadResult:
    commit: str = ""
    branch: str = ""
    remote: str = ""
    files: int = 0
    nothing_to_commit: bool = False


def _cache_dir(remote: str) -> Path:
    return CACHE_ROOT / hashlib.sha1(remote.encode("utf-8")).hexdigest()[:12]


def _git(args: list[str], cwd: Path | None = None, check: bool = True,
         extra_env: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never", LC_ALL="C", **(extra_env or {}))
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")
    try:
        done = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True,
                              timeout=GIT_TIMEOUT_S)
    except FileNotFoundError as exc:
        raise GitUploadError("git 이 설치되어 있지 않습니다 (Ubuntu: sudo apt install git)") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitUploadError(f"git {args[0]} 이(가) {GIT_TIMEOUT_S}초 안에 끝나지 않았습니다 (네트워크 확인)") from exc
    if check and done.returncode != 0:
        raise GitUploadError(_explain(args, done.stderr or done.stdout))
    return done


def _explain(args: list[str], output: str) -> str:
    text = output.strip()
    low = text.lower()
    hint = ""
    if any(k in low for k in ("authentication", "could not read username", "permission denied", "terminal prompts disabled",
                              "403", "401", "host key verification")):
        hint = ("\n\n인증에 실패했습니다. 이 PC 의 git 이 저장소에 접근할 수 있어야 합니다:\n"
                " - SSH 주소(git@github.com:...)와 등록된 ssh 키, 또는\n"
                " - HTTPS 주소와 credential helper (예: gh auth login / git credential-manager)\n"
                "터미널에서 'git ls-remote <주소>' 가 비밀번호 없이 되면 여기서도 됩니다.")
    elif "please tell me who you are" in low or "user.email" in low:
        hint = ("\n\n커밋 작성자 정보가 없습니다. 터미널에서 한 번 설정하세요:\n"
                "  git config --global user.name \"이름\"\n  git config --global user.email \"메일\"")
    elif "repository not found" in low or "does not appear to be a git repository" in low:
        hint = "\n\n저장소를 찾지 못했습니다. 주소와 접근 권한을 확인하세요."
    return f"git {' '.join(args[:2])} 실패:\n{text[-1500:]}{hint}"


def _prepare_checkout(remote: str, branch: str, log: Callable[[str], None]) -> Path:
    repo = _cache_dir(remote)
    if (repo / ".git").exists():
        current = _git(["remote", "get-url", "origin"], repo, check=False).stdout.strip()
        if current != remote:
            shutil.rmtree(repo)
    if not (repo / ".git").exists():
        log("저장소 가져오는 중 (처음 한 번)…")
        repo.parent.mkdir(parents=True, exist_ok=True)
        _git(["clone", "--quiet", remote, str(repo)])
    log("최신 상태 확인 중…")
    _git(["fetch", "--quiet", "--prune", "origin"], repo)
    has_remote_branch = _git(["rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{branch}"], repo,
                             check=False).returncode == 0
    _git(["reset", "--quiet", "--hard"], repo, check=False)
    _git(["clean", "-fdq"], repo, check=False)
    if has_remote_branch:
        _git(["checkout", "--quiet", "-B", branch, f"origin/{branch}"], repo)
    else:
        # 결과 전용 브랜치를 처음 만든다: 다른 브랜치 내용을 물려받지 않게 orphan
        # (switch --orphan 은 작업 트리도 비운다). 지난번 실패로 남은 로컬 브랜치는 버린다.
        _git(["checkout", "--quiet", "--detach"], repo, check=False)
        _git(["branch", "-D", branch], repo, check=False)
        _git(["switch", "--quiet", "--orphan", branch], repo)
    return repo


def upload(config: GitUploadConfig, plan: UploadPlan, message: str,
           log: Callable[[str], None] = lambda _m: None) -> UploadResult:
    remote, branch = config.remote.strip(), (config.branch.strip() or "main")
    if not remote:
        raise GitUploadError("결과 저장소 주소가 비어 있습니다")
    if not plan.files:
        raise GitUploadError("올릴 파일이 없습니다")
    who = {}
    if config.author_name.strip() and config.author_email.strip():
        name, email = config.author_name.strip(), config.author_email.strip()
        who = {"GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email,
               "GIT_COMMITTER_NAME": name, "GIT_COMMITTER_EMAIL": email}
    repo = _prepare_checkout(remote, branch, log)

    log(f"파일 {len(plan.files)}개 복사 중…")
    for f in plan.files:
        target = repo / f.dest
        target.parent.mkdir(parents=True, exist_ok=True)
        if f.data is not None:
            target.write_bytes(f.data)
        else:
            shutil.copy2(f.source, target)
    tops = sorted({f.dest.split("/")[0] for f in plan.files})
    _git(["add", "--all", "--", *tops], repo)
    result = UploadResult(branch=branch, remote=remote, files=len(plan.files))
    if _git(["diff", "--cached", "--quiet"], repo, check=False).returncode == 0:
        result.nothing_to_commit = True
        result.commit = _git(["rev-parse", "--short", "HEAD"], repo, check=False).stdout.strip()
        return result

    log("커밋 중…")
    _git(["commit", "--quiet", "-m", message], repo, extra_env=who)
    for attempt in range(2):
        log("업로드(push) 중…")
        pushed = _git(["push", "--quiet", "origin", f"HEAD:refs/heads/{branch}"], repo, check=False)
        if pushed.returncode == 0:
            break
        rejected = any(k in (pushed.stderr or "").lower() for k in ("rejected", "fetch first", "non-fast-forward"))
        if attempt == 0 and rejected:
            log("다른 사람이 먼저 올렸습니다. 합친 뒤 다시 시도…")
            _git(["fetch", "--quiet", "origin"], repo)
            _git(["rebase", "--quiet", f"origin/{branch}"], repo, extra_env=who)
            continue
        raise GitUploadError(_explain(["push", "origin"], pushed.stderr or pushed.stdout))
    result.commit = _git(["rev-parse", "--short", "HEAD"], repo).stdout.strip()
    return result


# --------------------------------------------------------------------- 계획 만들기


def _add(plan: UploadPlan, source: Path, dest: str) -> None:
    if not source.is_file():
        return
    if source.stat().st_size > MAX_FILE_MB * 1024 * 1024:
        plan.skipped.append(dest)
    else:
        plan.files.append(UploadFile(source, dest))


def _date_of(session: str) -> str:
    m = re.match(r"(\d{4}-\d{2}-\d{2})", session)
    return m.group(1) if m else session


def plan_intrinsic(session_dir: Path, camera_name: str, config: GitUploadConfig | None = None,
                   recommended: str | None = None, vehicle: str = "") -> tuple[UploadPlan, str]:
    """Output Session → 설정에서 고른 파일. `recommended` 는 추천 모델 (예: 'brown_conrady').

    `vehicle` 이 비어 있으면 "차량 없음" - 경로의 {vehicle} 자리에 ``no_vehicle``.
    """
    from shared.vehicle import NO_VEHICLE_PATH

    config = config or GitUploadConfig()
    session_dir = Path(session_dir)
    camera = safe_component(camera_name, "camera")
    car = safe_component(vehicle, "vehicle") if vehicle else NO_VEHICLE_PATH
    problem = check_path_template(config.intrinsic_path, INTRINSIC_PATH_FIELDS)
    if problem:
        raise GitUploadError(f"Camera Intrinsic 경로 규칙: {problem}")
    base = render_path(config.intrinsic_path, vehicle=car, camera=camera, session=session_dir.name,
                       date=_date_of(session_dir.name))
    chosen = set(config.intrinsic_files)
    plan = UploadPlan()
    if "models" in chosen:
        for path in sorted((session_dir / "intrinsic").glob("camera_*.yaml")):
            if path.name == "camera_info.yaml":
                continue
            if config.intrinsic_recommended_only and recommended:
                from export.output_manager import model_filename

                if path.name != f"camera_{model_filename(recommended)}.yaml":
                    continue
            _add(plan, path, f"{base}/{path.name}")
    extra = {"camera_info": "intrinsic/camera_info.yaml", "report_html": "reports/report.html",
             "report_json": "reports/calibration.json", "project": "project/calibration.ccproj"}
    for key, rel in extra.items():
        if key in chosen:
            _add(plan, session_dir / rel, f"{base}/{Path(rel).name}")
    if vehicle and config.upload_vehicle:
        add_vehicle_files(plan, vehicle)
    return plan, f"intrinsic: {car + '/' if vehicle else ''}{camera} {session_dir.name}"


def plan_extrinsic(session_dir: Path, vehicle: str, cameras: Iterable[str],
                   config: GitUploadConfig | None = None) -> tuple[UploadPlan, str]:
    """CAMERA–LiDAR 세션 → 저장한 카메라마다 설정에서 고른 파일."""
    config = config or GitUploadConfig()
    session_dir = Path(session_dir)
    car = safe_component(vehicle, "vehicle")
    names = list(cameras)
    problem = check_path_template(config.extrinsic_path, EXTRINSIC_PATH_FIELDS)
    if problem:
        raise GitUploadError(f"CAMERA–LiDAR 경로 규칙: {problem}")
    values = dict(vehicle=car, session=session_dir.name, date=_date_of(session_dir.name))
    plan = UploadPlan()
    seen = set()
    for name in names:
        base = render_path(config.extrinsic_path, camera=name, **values)
        for filename in config.extrinsic_files:
            source = session_dir / filename if filename == "session.yaml" else session_dir / name / filename
            dest = f"{base}/{filename}"
            if filename == "session.yaml":
                # 세션당 하나: 카메라 칸({camera})을 뺀 경로에
                dest = render_path(config.extrinsic_path.replace("{camera}", ""), camera="", **values) + "/session.yaml"
            if dest not in seen:
                seen.add(dest)
                _add(plan, source, dest)
    if config.upload_vehicle:
        add_vehicle_files(plan, vehicle)
    return plan, f"extrinsic: {car} {session_dir.name}" + (f" ({', '.join(names)})" if names else "")


# --------------------------------------------------------------------- 차량 정보


def vehicle_lens_entries(vehicle: str, presets_path: Path | None = None) -> list[dict]:
    """차량 카메라들이 쓰는 렌즈 프리셋 항목 (camera_intrinsics.yaml 의 cameras: 항목 그대로)."""
    from shared import camera as shared_camera
    from shared import vehicle as shared_vehicle

    wanted = set(shared_vehicle.used_presets(vehicle))
    return [e for e in shared_camera.load_preset_entries(presets_path) if str(e.get("name")) in wanted]


def add_vehicle_files(plan: UploadPlan, vehicle: str, presets_path: Path | None = None) -> None:
    """vehicles/<차량>/vehicle.yaml 과 lenses.yaml 을 올릴 목록에 넣는다."""
    import yaml

    from shared import vehicle as shared_vehicle

    profile = shared_vehicle.profile_path(vehicle)
    if not profile.is_file():
        return
    base = f"{VEHICLES_DIR}/{safe_component(vehicle, 'vehicle')}"
    _add(plan, profile, f"{base}/vehicle.yaml")
    lenses = vehicle_lens_entries(vehicle, presets_path)
    if lenses:
        text = yaml.safe_dump({"cameras": lenses}, sort_keys=False, allow_unicode=True)
        plan.files.append(UploadFile(None, f"{base}/lenses.yaml", text.encode("utf-8")))


@dataclass
class RemoteVehicle:
    name: str
    profile: dict
    lenses: list  # camera_intrinsics.yaml 의 cameras: 항목
    status: str = ""  # "새 차량" / "같음" / "다름" (로컬과 비교)


def _filled(d: dict) -> dict:
    return {k: v for k, v in d.items() if v not in (None, "", [], {})}


def fetch_vehicles(config: GitUploadConfig, log: Callable[[str], None] = lambda _m: None) -> list[RemoteVehicle]:
    """결과 저장소의 vehicles/ 를 읽는다 (업로드와 같은 캐시 체크아웃을 쓴다)."""
    import yaml

    from shared import vehicle as shared_vehicle

    remote, branch = config.remote.strip(), (config.branch.strip() or "main")
    if not remote:
        raise GitUploadError("결과 저장소 주소가 비어 있습니다 (⚙ Git 설정)")
    repo = _prepare_checkout(remote, branch, log)
    root = repo / VEHICLES_DIR
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        try:
            profile = yaml.safe_load((d / "vehicle.yaml").read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        try:
            lenses = (yaml.safe_load((d / "lenses.yaml").read_text(encoding="utf-8")) or {}).get("cameras") or []
        except (OSError, yaml.YAMLError):
            lenses = []
        name = str(profile.get("vehicle") or d.name)
        local = shared_vehicle.profile_path(name)
        if not local.is_file():
            status = "새 차량"
        else:
            same = _filled(shared_vehicle.load(name)) == _filled({**profile, "vehicle": name})
            status = "같음" if same else "다름"
        out.append(RemoteVehicle(name, profile, list(lenses), status))
    log(f"차량 {len(out)}대")
    return out


def import_vehicle(rv: RemoteVehicle, presets_path: Path | None = None, root: Path | None = None) -> list[str]:
    """저장소의 차량 정보를 이 PC 에 반영한다. 바꾼 것들의 설명을 돌려준다.

    * 렌즈 프리셋: 같은 이름이 있으면 저장소 값으로 바꾼다.
    * 차량 프로필: 저장소 값이 우선이고, 저장소에 없는 이 PC 의 카메라·렌즈 항목은 남긴다.
      바꾸기 전 파일은 vehicle.yaml.bak 으로 둔다.
    """
    from shared import camera as shared_camera
    from shared import vehicle as shared_vehicle

    done = []
    for entry in rv.lenses:
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        fields = {k: v for k, v in entry.items() if k not in ("name", "note")}
        lens = shared_camera.LensParams(
            str(fields.get("model", "pinhole")), float(fields["fx"]), float(fields["fy"]),
            float(fields["cx"]), float(fields["cy"]),
            {k: float(fields.get(k, 0.0)) for k in shared_camera.COEFF_KEYS})
        shared_camera.add_lens_preset(name, lens, str(entry.get("note") or ""), presets_path)
        done.append(f"렌즈 {name}")
    path = shared_vehicle.profile_path(rv.name, root)
    local = shared_vehicle.load(rv.name, root) if path.is_file() else {}
    merged = {**local, **rv.profile, "vehicle": rv.name}
    for key in ("cameras", "lenses"):
        if local.get(key) or rv.profile.get(key):
            merged[key] = {**(local.get(key) or {}), **(rv.profile.get(key) or {})}
    if path.is_file():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    shared_vehicle.save(rv.name, merged, root)
    done.append(f"차량 {rv.name}")
    return done

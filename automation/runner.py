"""End-to-end automated intrinsic calibration run.

흐름 (각 단계는 run_manifest.json의 stages에 기록된다)::

    preflight (세션 생성 전, 실패 시 아무 것도 만들지 않음)
      프로파일 검증 → 입력 이미지 확인 → 해상도 확인
    session
      OutputManager 세션 생성 → run_manifest.json(status=running)
    input_fingerprint   입력 파일 sha256
    image_selection     기존 detect_dataset + select_calibration_images
    calibration         자식 프로세스에서 기존 app.cli.main 실행
    result_collection   calibration.ccproj를 기존 load_project로 읽기
    acceptance          automation.acceptance로 PASS/WARN/FAIL
    finalize            run_summary.json, manifest.json exports 기록, 완료 표시

원본 이미지는 읽기만 한다. 실행마다 새 세션 폴더를 만들므로 이전 결과를
덮어쓰지 않는다.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

from automation.acceptance import (
    FAIL,
    PASS,
    WARN,
    AcceptanceEvidence,
    collect_evidence,
    evaluate_acceptance,
)
from automation.identity import CameraIdentity
from automation.profile import (
    AutomationProfile,
    ProfileError,
    build_cli_options,
    options_to_argv,
    parse_cli_argv,
    parse_cli_config,
    write_cli_config,
)
from automation.run_manifest import (
    STATUS_COMPLETED,
    RunManifest,
    atomic_write_json,
    canonical_sha256,
    detection_fingerprint,
    environment_info,
    git_info,
    input_fingerprint,
    iso,
    new_run_id,
    sha256_file,
    utc_now,
)

EXIT_PASS = 0
EXIT_INPUT_ERROR = 1
EXIT_RUN_FAILED = 2
EXIT_WARN = 3
EXIT_FAIL = 4

_EXIT_BY_VERDICT = {PASS: EXIT_PASS, WARN: EXIT_WARN, FAIL: EXIT_FAIL}

AUTOMATION_DIR = "automation"
RUN_MANIFEST_NAME = "run_manifest.json"

# CLI --export 이름 → --json-summary의 exported_files 키
_EXPORT_SUMMARY_KEYS = {
    "opencv": "opencv_yaml",
    "ros": "ros_yaml",
    "report": "report_html",
    "json": "json",
    "csv": "csv",
    "kalibr": "kalibr_target_yaml",
}


class RunFailure(RuntimeError):
    """세션 생성 이후의 실행 실패. stage 이름과 함께 기록된다."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


@dataclass
class RunOutcome:
    exit_code: int
    run_id: str | None
    status: str
    verdict: str | None
    session_dir: str | None
    message: str


# ---------------------------------------------------------------------------
# Preflight (세션 생성 전)
# ---------------------------------------------------------------------------


def resolve_input_images(items: list[str]) -> list[str]:
    """기존 CLI와 같은 규칙(파일/디렉터리/glob)으로 이미지 목록을 만들고 절대경로로 고정한다."""
    from app.cli import CliError, _resolve_image_path_items

    if not items:
        raise ProfileError("입력 이미지가 지정되지 않았습니다 (--images).")
    for item in items:
        if not any(ch in item for ch in "*?[]") and not Path(item).exists():
            raise ProfileError(f"입력 경로가 존재하지 않습니다: {item}")
    try:
        paths = _resolve_image_path_items(items)
    except CliError as exc:
        raise ProfileError(str(exc)) from None
    resolved = [str(Path(p).resolve()) for p in paths]

    # 기존 파이프라인은 파일 stem을 image_id로 쓴다. stem이 겹치면 Train/Test
    # 분할 기록에서 서로 다른 이미지를 구분할 수 없으므로 미리 막는다.
    seen: dict[str, str] = {}
    for p in resolved:
        stem = Path(p).stem
        if stem in seen:
            raise ProfileError(f"파일 이름(stem)이 같은 이미지가 있습니다: {seen[stem]} / {p}")
        seen[stem] = p
    return resolved


def check_image_sizes(paths: list[str], expected: tuple[int, int] | None) -> tuple[int, int]:
    import cv2

    sizes: dict[tuple[int, int], list[str]] = {}
    for p in paths:
        image = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ProfileError(f"이미지를 읽을 수 없습니다: {p}")
        h, w = image.shape[:2]
        sizes.setdefault((w, h), []).append(p)
    if len(sizes) > 1:
        summary = ", ".join(f"{w}x{h} ({len(v)}장)" for (w, h), v in sizes.items())
        raise ProfileError(f"입력 이미지 해상도가 섞여 있습니다: {summary}")
    actual = next(iter(sizes))
    if expected is not None and tuple(expected) != actual:
        raise ProfileError(
            f"프로파일 해상도 {expected[0]}x{expected[1]}와 실제 이미지 해상도 "
            f"{actual[0]}x{actual[1]}가 다릅니다."
        )
    return actual


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _run_image_selection(profile: AutomationProfile, pattern_config, camera_config, paths: list[str],
                         jobs: int):
    """반환: (선택 경로, 기록용 요약, ImageSelectionResult 또는 None).

    선별 manifest 파일은 여기서 쓰지 않는다 - 기존 CLI ``--session-dir``은 결과가
    이미 있는 세션을 거부하므로, 계산이 끝난 뒤 reports/에 기록한다.
    """
    """GUI의 ImageSelectionWorker와 같은 호출 순서로 기존 선별 기능을 실행한다."""
    from calibration.detector import detect_dataset
    from calibration.frame_quality import compute_frame_quality_scores
    from calibration.image_selection import select_calibration_images
    from calibration.models.common import infer_image_size
    from calibration.quality import analyze_dataset_quality

    if not profile.image_selection_enabled:
        # 선별을 끄면 검출은 기존 CLI가 한 번만 수행한다 (검출 전부 실패 시 CLI 종료 코드 1).
        return paths, {"enabled": False, "num_input": len(paths), "selected_count": len(paths)}, None

    dataset = detect_dataset(
        paths, pattern_config,
        parallel=jobs != 1,
        max_workers=None if jobs in (0, 1) else jobs,
    )
    if dataset.num_detected == 0:
        raise RunFailure("image_selection", "모든 입력 이미지에서 패턴 검출에 실패했습니다. 패턴 설정을 확인하세요.")

    analyze_dataset_quality(dataset, camera_config)
    image_size = infer_image_size(dataset, camera_config)
    compute_frame_quality_scores(dataset, pattern_config, image_size, use_reprojection=False)
    result = select_calibration_images(dataset, profile.image_selection)
    if result.selected_count == 0:
        raise RunFailure("image_selection", f"선별된 이미지가 없습니다: {result.summary_text()}")
    return list(result.selected_paths), {
        "enabled": True,
        "num_input": len(paths),
        "num_detected": dataset.num_detected,
        "requested_count": result.requested_count,
        "selected_count": result.selected_count,
        "counts": result.counts,
        "position_counts": result.position_counts,
        "distance_counts": result.distance_counts,
        "tilt_counts": result.tilt_counts,
        "warnings": result.warnings,
    }, result


def cli_command(config_path: Path, opencv_threads: int | None = None) -> list[str]:
    """기존 CLI 실행 명령. opencv_threads가 있으면 CLI 실행 전에 cv2.setNumThreads만 호출한다."""
    if opencv_threads is None:
        return [sys.executable, "-m", "app.cli", "--config", str(config_path)]
    bootstrap = (
        f"import sys, cv2; cv2.setNumThreads({int(opencv_threads)}); "
        "from app.cli import main; sys.exit(main(sys.argv[1:]))"
    )
    return [sys.executable, "-c", bootstrap, "--config", str(config_path)]


def _run_cli_child(config_path: Path, cwd: Path, log_path: Path, timeout_sec: float | None,
                   opencv_threads: int | None = None) -> int:
    """기존 CLI를 그대로 실행한다: ``python -m app.cli --config <config_path>``."""
    import app

    env = dict(os.environ)
    # 저장소에서 바로 실행할 때(설치하지 않은 경우)에도 같은 소스의 app 패키지를 쓰도록
    # 부모가 import한 app 패키지의 루트를 앞에 둔다. 설치된 환경에서는 site-packages다.
    package_root = str(Path(app.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, env.get("PYTHONPATH")]))
    # CLI는 한글/기호(✓ ⚠)를 출력한다. 로그 파일로 리다이렉트될 때 Windows 기본
    # 코드페이지로 인코딩 오류가 나지 않도록 UTF-8을 강제한다.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    with open(log_path, "w", encoding="utf-8") as log_file:
        try:
            completed = subprocess.run(
                cli_command(config_path, opencv_threads),
                cwd=cwd, env=env, stdout=log_file, stderr=subprocess.STDOUT,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            raise RunFailure("calibration", f"계산이 제한 시간({timeout_sec}s)을 초과해 중단했습니다.") from None
    return completed.returncode


def _tail(path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def _model_summary(project) -> dict:
    models = {}
    for model, cal in project.calibration_results.items():
        val = project.validation_results.get(model)
        entry: dict[str, Any] = {
            "success": cal.success,
            "error_message": cal.error_message,
            "train_rms_px": cal.rms_error,  # Legacy alias; explicitly full-fit below.
            "full_fit_rms_px": cal.rms_error,
            "parameter_scope": "full_fit_enabled_data_with_recorded_model_exclusions",
            "fit_frame_ids": list(cal.per_frame_error),
            "optimization_audit": cal.optimization_audit,
            "camera_matrix": cal.camera_matrix.tolist() if cal.camera_matrix is not None else None,
            "distortion": cal.distortion.reshape(-1).tolist() if cal.distortion is not None else None,
        }
        if val is not None:
            stats = val.test_residual_stats
            entry["holdout"] = {
                "success": val.success,
                "error_message": val.error_message,
                "train_rms_px": val.train_rms,
                "test_rms_px": val.test_rms,
                "parameter_scope": "train_only; not exported full-fit K/D",
                "camera_matrix": val.train_camera_matrix.tolist() if val.train_camera_matrix is not None else None,
                "distortion": val.train_distortion.tolist() if val.train_distortion is not None else None,
                "optimization_audit": val.optimization_audit,
                "edge_rms_px": val.edge_rms,
                "test_p95_px": stats.p95 if stats else None,
                "test_p99_px": stats.p99 if stats else None,
                "num_train_frames": len(val.train_frame_ids),
                "num_test_frames": len(val.test_frame_ids),
                "failed_test_frame_ids": list(val.failed_test_frame_ids),
                "evidence_gate": val.evidence_gate.status if val.evidence_gate else None,
            }
        models[getattr(model, "value", str(model))] = entry
    return models


def _recommendation_summary(project, cli_args) -> dict:
    scores = project.model_scores or []
    recommended = next((s for s in scores if s.is_recommended), None)
    final = project.final_result
    return {
        "recommended_model": recommended.model_name.value if recommended else None,
        "chosen_model": final.chosen_model.value if final else None,
        "chosen_by": "forced (--model)" if getattr(cli_args, "model", None) else "recommender",
        "selection_reasons": list(recommended.selection_reasons) if recommended else [],
        "selection_confidence": recommended.selection_confidence if recommended else None,
        "selection_confidence_level": recommended.selection_confidence_level if recommended else None,
        "selection_confidence_reason": recommended.selection_confidence_reason if recommended else None,
        "scores": [
            {
                "model": s.model_name.value,
                "score": s.score,
                "is_recommended": s.is_recommended,
                "selection_status": s.selection_status,
                "ineligibility_reason": s.selection_ineligibility_reason,
            }
            for s in scores
        ],
        "overall_grade": final.overall_grade.value if final else None,
        "confidence_level": final.confidence.level if final and final.confidence else None,
        "confidence_score": final.confidence.score if final and final.confidence else None,
    }


def _split_summary(project, cli_args) -> dict:
    """모델별 실제 Train/Test 이미지 ID를 기록하고 누수·공유 여부를 확인한다."""
    splits = {
        getattr(m, "value", str(m)): (list(v.train_frame_ids), list(v.test_frame_ids))
        for m, v in project.validation_results.items()
    }
    unique = {(tuple(tr), tuple(te)) for tr, te in splits.values()}
    leakage = {m: sorted(set(tr) & set(te)) for m, (tr, te) in splits.items() if set(tr) & set(te)}
    shared = len(unique) == 1
    summary: dict[str, Any] = {
        "method": "calibration.validation.split_train_test (position x distance stratified)",
        "seed": cli_args.seed,
        "test_ratio": cli_args.test_ratio,
        "shared_across_models": shared,
        "leakage_free": not leakage,
        "leakage": leakage,
        "reproducibility_note": (
            "같은 입력 이미지·같은 패턴 설정이면 검출 결과(detection_fingerprint)가 같고, "
            "같은 seed/test_ratio로 같은 분할이 다시 만들어진다. 아래 ID 목록으로 실제 분할을 직접 확인할 수 있다."
        ),
    }
    if shared and splits:
        train_ids, test_ids = next(iter(splits.values()))
        summary["train_frame_ids"] = train_ids
        summary["test_frame_ids"] = test_ids
    else:
        summary["per_model"] = {m: {"train_frame_ids": tr, "test_frame_ids": te} for m, (tr, te) in splits.items()}
    return summary


def _effective_cli_options(cli_args) -> dict:
    keys = ("pattern", "squares_x", "squares_y", "square_size", "marker_size", "dictionary",
            "circle_grid_type", "aprilgrid_variant", "width", "height", "models", "model",
            "test_ratio", "seed", "calibration_method", "outlier", "corner_outlier",
            "max_iterations", "diagnostic", "kfold", "kfold_repeats", "repeatability",
            "bootstrap_ci", "n_bootstrap", "jobs", "export")
    return {k: getattr(getattr(cli_args, k, None), "value", getattr(cli_args, k, None)) for k in keys}


def _check_exports(session_dir: Path, cli_summary: dict, exports: list[str]) -> dict:
    exported = cli_summary.get("exported_files") or {}
    missing = []
    outside = []
    checked = {}
    for export in exports:
        key = _EXPORT_SUMMARY_KEYS.get(export)
        path = exported.get(key) if key else None
        if not path or not Path(path).is_file():
            missing.append(export)
            continue
        resolved = Path(path).resolve()
        if session_dir not in resolved.parents:
            outside.append(str(resolved))
        checked[export] = str(resolved)
    if outside:
        raise RunFailure("result_collection", f"결과 파일이 고정 세션 밖에 저장됐습니다: {outside}")
    if missing:
        raise RunFailure("result_collection", f"요청한 Export 결과가 없습니다: {missing}")
    return checked


def _record_session_exports(session_dir: Path, entries: dict[str, Path]) -> None:
    """기존 manifest.json의 exports에 자동화 산출물을 같은 형식으로 추가한다."""
    from export.output_manager import OutputManager

    manager = OutputManager(output_root=session_dir.parent)
    manager.session_dir = session_dir
    manager.manifest = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    for key, path in entries.items():
        manager.record_export(key, path)


def _list_outputs(session_dir: Path, exclude: set[Path]) -> list[dict]:
    files = []
    for path in sorted(p for p in session_dir.rglob("*") if p.is_file()):
        if path in exclude or path.name.endswith(".tmp"):
            continue
        files.append({
            "path": path.relative_to(session_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return files


def _failure_acceptance(stage: str, message: str) -> dict:
    return {
        "verdict": FAIL,
        "criteria_complete": False,
        "reasons": [f"실행 실패 ({stage}): {message}"],
        "fail_reasons": [f"실행 실패 ({stage}): {message}"],
        "warn_reasons": [],
        "notes": [],
        "fail_kinds": ["calculation_failure" if stage in ("calibration", "image_selection") else "execution_failure"],
        "warn_kinds": [],
        "failed_stage": stage,
        "checks": [],
        "chosen_model": None,
        "recommended_model": None,
        "note": "계산 또는 결과 수집이 끝나지 않아 품질 평가를 할 수 없습니다.",
    }


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run_automation(
    profile: AutomationProfile,
    identity: CameraIdentity,
    image_inputs: list[str],
    *,
    log: Callable[[str], None] = print,
) -> RunOutcome:
    # ---- preflight: 실패하면 세션을 만들지 않는다 ----
    try:
        image_paths = resolve_input_images(image_inputs)
        expected = None
        if profile.camera.get("width") is not None:
            expected = (profile.camera["width"], profile.camera["height"])
        width, height = check_image_sizes(image_paths, expected)
        # 선별 전체 목록 기준으로 CLI가 받아들이는 설정인지 확인 (경로는 아직 미정)
        validation_options = build_cli_options(
            profile, image_paths=image_paths, sensor_name=identity.label,
            save_project="x.ccproj", json_summary="x.json", log_file="x.log",
        )
        _, pattern_config = parse_cli_argv(options_to_argv(validation_options))
    except ProfileError as exc:
        return RunOutcome(EXIT_INPUT_ERROR, None, "rejected", None, None, str(exc))

    from calibration.types import CameraConfig
    from export.output_manager import OutputManager

    started = utc_now()
    run_id = new_run_id(started)
    profile_dict = profile.to_dict()
    try:
        manager = OutputManager(profile.output_root, camera_name=identity.label)
        # 이름만 선점한 빈 폴더 - manifest.json과 결과는 기존 CLI가 --session-dir로 만든다
        session_dir = manager.reserve_session_directory(identity.label)
        auto_dir = session_dir / AUTOMATION_DIR
        auto_dir.mkdir(parents=True, exist_ok=True)
        manifest = RunManifest.create(
            session_dir / RUN_MANIFEST_NAME,
            run_id=run_id,
            started_at=started,
            camera=identity.to_dict() | {"image_width": width, "image_height": height},
            session={"session_id": session_dir.name, "session_dir": str(session_dir)},
            profile={"resolved": profile_dict, "sha256": canonical_sha256(profile_dict)},
            inputs={"arguments": list(image_inputs), "num_images": len(image_paths)},
            code_version=git_info(),
            environment=environment_info(),
        )
        atomic_write_json(auto_dir / "profile_resolved.json", profile_dict)
    except OSError as exc:
        return RunOutcome(EXIT_RUN_FAILED, run_id, "failed", None, None,
                          f"결과 세션을 만들 수 없습니다 (output root: {profile.output_root or '기본값'}): {exc}")
    log(f"Run {run_id} → {session_dir}")

    stage = "input_fingerprint"
    cli_args = None
    try:
        manifest.start_stage(stage)
        fingerprint = input_fingerprint(image_paths)
        manifest.data["inputs"]["fingerprint"] = fingerprint
        manifest.finish_stage(stage, aggregate_sha256=fingerprint["aggregate_sha256"])

        stage = "image_selection"
        manifest.start_stage(stage)
        log("패턴 검출 및 이미지 선별 중...")
        camera_config = CameraConfig(width=width, height=height, sensor_name=identity.label)
        jobs = int(profile.calibration.get("jobs", 1))
        selected, selection, selection_result = _run_image_selection(
            profile, pattern_config, camera_config, image_paths, jobs,
        )
        # 선별 기준은 결과(모델 비교)에 영향을 주므로 실제 적용한 설정을 그대로 남긴다
        selection["config"] = dataclasses.asdict(profile.image_selection)
        manifest.data["image_selection"] = selection
        manifest.finish_stage(stage, selected_count=len(selected))
        log(f"  선별 결과: {len(selected)}/{len(image_paths)}장")

        stage = "calibration"
        manifest.start_stage(stage)
        ccproj = session_dir / "project" / "calibration.ccproj"
        cli_summary_path = auto_dir / "cli_summary.json"
        options = build_cli_options(
            profile, image_paths=selected, sensor_name=identity.label,
            save_project=str(ccproj), json_summary=str(cli_summary_path),
            log_file=str(auto_dir / "cli_debug.log"), session_dir=str(session_dir),
        )
        cli_args, _ = parse_cli_argv(options_to_argv(options))
        config_path = write_cli_config(options, auto_dir / "cli_config.yaml")
        # 실제 실행 경로(--config)가 검증 경로(argv)와 같은 설정으로 해석되는지 대조
        config_args, _ = parse_cli_config(config_path)
        mismatched = sorted(d for d in options if getattr(cli_args, d, None) != getattr(config_args, d, None))
        if mismatched:
            raise RunFailure(stage, f"CLI config 해석 결과가 검증 결과와 다릅니다: {mismatched}")
        manifest.data["calibration"] = {
            "invocation": f"python -m app.cli --config {AUTOMATION_DIR}/cli_config.yaml",
            "opencv_threads": profile.opencv_threads,
            "command": cli_command(Path(AUTOMATION_DIR) / "cli_config.yaml", profile.opencv_threads),
            "config_file": f"{AUTOMATION_DIR}/cli_config.yaml",
            "argv": options_to_argv(options),
            "effective_options": _effective_cli_options(cli_args),
        }
        manifest.save()
        log("4개 모델 계산 + Hold-out 검증 + 추천 (기존 CLI) 실행 중...")
        t0 = time.monotonic()
        exit_code = _run_cli_child(config_path, session_dir, auto_dir / "cli_output.log", profile.timeout_sec,
                                   profile.opencv_threads)
        elapsed = round(time.monotonic() - t0, 2)
        if exit_code != 0:
            kind = {1: "입력 문제", 2: "계산 실패"}.get(exit_code, "비정상 종료")
            raise RunFailure(
                stage,
                f"기존 CLI가 종료 코드 {exit_code}({kind})로 끝났습니다.\n{_tail(auto_dir / 'cli_output.log')}",
            )
        manifest.finish_stage(stage, cli_exit_code=exit_code, elapsed_sec=elapsed)
        if selection_result is not None:
            from calibration.image_selection import write_selection_manifest

            selection_manifest = write_selection_manifest(
                selection_result, session_dir / "reports" / "image_selection_manifest.json",
            )
            selection["manifest"] = str(selection_manifest)
            manifest.save()

        stage = "result_collection"
        manifest.start_stage(stage)
        from calibration.project_io import load_project

        if not ccproj.is_file() or not cli_summary_path.is_file():
            raise RunFailure(stage, "CLI 결과(calibration.ccproj 또는 cli_summary.json)가 고정 세션에 없습니다.")
        cli_summary = json.loads(cli_summary_path.read_text(encoding="utf-8"))
        project, missing_images = load_project(str(ccproj))
        exported = _check_exports(session_dir, cli_summary, profile.exports)
        manifest.data["dataset"] = {
            "num_total": project.dataset.num_total,
            "num_detected": project.dataset.num_detected,
            "num_enabled": project.dataset.num_enabled,
            "detection_fingerprint_sha256": detection_fingerprint(project.dataset),
            "image_ids": {f.image_info.image_id: f.image_info.path for f in project.dataset.frames},
            "missing_source_images": missing_images,
        }
        manifest.data["split"] = _split_summary(project, cli_args)
        manifest.data["results"] = {
            "recommendation": _recommendation_summary(project, cli_args),
            "models": _model_summary(project),
        }
        manifest.finish_stage(stage)

        stage = "acceptance"
        manifest.start_stage(stage)
        evidence: AcceptanceEvidence = collect_evidence(project)
        acceptance = evaluate_acceptance(evidence, profile.acceptance)
        acceptance_path = atomic_write_json(auto_dir / "acceptance.json", {
            "run_id": run_id, **acceptance.to_dict(),
        })
        manifest.data["acceptance"] = {
            "verdict": acceptance.verdict,
            "criteria_complete": acceptance.criteria_complete,
            "reasons": acceptance.reasons,
            "fail_kinds": acceptance.fail_kinds,
            "warn_kinds": acceptance.warn_kinds,
            "file": f"{AUTOMATION_DIR}/acceptance.json",
        }
        manifest.finish_stage(stage, verdict=acceptance.verdict)

        stage = "finalize"
        manifest.start_stage(stage)
        recommendation = manifest.data["results"]["recommendation"]
        chosen = recommendation["chosen_model"]
        chosen_metrics = evidence.metrics
        summary_path = atomic_write_json(auto_dir / "run_summary.json", {
            "run_id": run_id,
            "status": STATUS_COMPLETED,
            "camera": identity.to_dict(),
            "session_dir": str(session_dir),
            "verdict": acceptance.verdict,
            "criteria_complete": acceptance.criteria_complete,
            "reasons": acceptance.reasons,
            "chosen_model": chosen,
            "recommended_model": recommendation["recommended_model"],
            "metrics": chosen_metrics,
            "overall_grade": recommendation["overall_grade"],
            "confidence_level": recommendation["confidence_level"],
            "images": {
                "input": len(image_paths),
                "selected": len(selected),
                "used": project.dataset.num_enabled,
            },
            "exports": {k: str(Path(v).relative_to(session_dir).as_posix()) for k, v in exported.items()},
            "approval_note": "이 결과는 차량 적용 승인이 아닙니다. 적용은 별도 승인 절차가 필요합니다.",
        })
        session_exports = {
            "automation.run_manifest": session_dir / RUN_MANIFEST_NAME,
            "automation.acceptance": acceptance_path,
            "automation.run_summary": summary_path,
        }
        if selection.get("manifest"):
            session_exports["image_selection.manifest"] = Path(selection["manifest"])
        _record_session_exports(session_dir, session_exports)
        manifest.data["outputs"] = _list_outputs(session_dir, exclude={session_dir / RUN_MANIFEST_NAME})
        manifest.finish_stage(stage)
        manifest.mark_completed()
    except KeyboardInterrupt:
        manifest.mark_failed(stage, "사용자가 실행을 중단했습니다 (interrupted).")
        log(f"중단됨 [{stage}]")
        return RunOutcome(130, run_id, "failed", None, str(session_dir), "interrupted")
    except Exception as exc:  # noqa: BLE001 - 모든 실패를 run_manifest에 남기는 경계
        failed_stage = exc.stage if isinstance(exc, RunFailure) else stage
        message = str(exc) if isinstance(exc, RunFailure) else f"{type(exc).__name__}: {exc}"
        try:
            atomic_write_json(auto_dir / "acceptance.json", {
                "run_id": run_id, **_failure_acceptance(failed_stage, message),
            })
        except OSError:
            pass
        try:
            manifest.mark_failed(failed_stage, message)
        except OSError as write_exc:
            message += f"\n(run_manifest.json 기록도 실패: {write_exc})"
        log(f"실패 [{failed_stage}]: {message}")
        return RunOutcome(EXIT_RUN_FAILED, run_id, "failed", FAIL, str(session_dir), message)

    log(f"완료: verdict={acceptance.verdict} chosen_model={chosen} ({iso(utc_now())})")
    return RunOutcome(
        _EXIT_BY_VERDICT[acceptance.verdict], run_id, STATUS_COMPLETED, acceptance.verdict,
        str(session_dir), "; ".join(acceptance.reasons),
    )

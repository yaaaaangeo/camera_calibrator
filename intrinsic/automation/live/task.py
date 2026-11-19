"""One isolated preliminary fit or existing P1 run. No Qt/ROS in this process."""
from __future__ import annotations
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import threading
import time

from automation.live.windows_job import join_job_from_environment
join_job_from_environment()

from automation.identity import build_identity
from automation.live.config import resolve_live_profile
from automation.live.process import process_alive, stop_tree
from automation.run_manifest import sha256_file
from automation.live.storage import write_json as atomic_write_json
from calibration.project_io import project_from_dict, load_project


def execute(request):
    import cv2
    from automation.live.readiness import preflight_observability
    from automation.runner import run_automation
    from calibration.diagnosis import diagnose_calibration
    from calibration.occlusion import observable_coverage
    from automation.live.assessment import review_result
    profile = resolve_live_profile(request['profile'])
    cv2.setNumThreads(profile.p1.opencv_threads or 1)
    for path, digest in request['hashes'].items():
        if sha256_file(path) != digest:
            raise ValueError(f'Capture changed before computation: {path}')
    project = project_from_dict(request['project'])
    if request['kind'] == 'observability':
        return {'observability': preflight_observability(project.dataset, project.camera_config,
                                                         request['train_ids'], profile.readiness)}
    identity = build_identity(request['identity']['camera_model'], request['identity']['serial_number'])
    root = Path(request['session']) / 'runs' / f"attempt_{request['attempt']:03d}"
    p1 = replace(profile.p1, output_root=str(root))
    if profile.coverage_mask is not None:
        mask_path = root / 'capture_guidance_mask.json'
        atomic_write_json(mask_path, profile.coverage_mask)
        p1 = replace(p1, calibration={**p1.calibration, 'capture_guidance_mask': str(mask_path.resolve())})
    outcome = run_automation(p1, identity, request['paths'])
    result = {'outcome': asdict(outcome), 'acceptance': None, 'capture_recommendations': []}
    if outcome.session_dir:
        acceptance = Path(outcome.session_dir) / 'automation' / 'acceptance.json'
        if acceptance.is_file():
            result['acceptance'] = json.loads(acceptance.read_text(encoding='utf-8'))
    if outcome.status == 'completed':
        calibrated, _ = load_project(str(Path(outcome.session_dir) / 'project/calibration.ccproj'))
        chosen = calibrated.final_result.chosen_model
        diagnosis = diagnose_calibration(calibrated.calibration_results[chosen],
                                         calibrated.validation_results.get(chosen),
                                         calibrated.final_result.dataset_coverage_pct,
                                         calibrated.dataset.coverage_grid, calibrated.dataset.diversity,
                                         observable_coverage=observable_coverage(
                                             calibrated.dataset, calibrated.camera_config, profile.coverage_mask) if profile.coverage_mask is not None else None)
        result['capture_recommendations'] = [f'{r.title}: {r.action}' for r in diagnosis.capture_recommendations]
        result['recommendation'] = {'chosen_model': chosen.value,
                                    'recommended_model': next((s.model_name.value for s in calibrated.model_scores if s.is_recommended), None)}
        reports = {model.value: asdict(value.observability) if value.observability else None
                   for model, value in calibrated.calibration_results.items()}
        result['p1_outcome'] = result['outcome']
        result['p1_acceptance'] = result['acceptance']
        result['outcome'], result['acceptance'], result['p2_acceptance'] = review_result(
            result['outcome'], result['acceptance'], reports, chosen.value,
            profile.readiness, request.get('preflight'))
        path = Path(outcome.session_dir) / 'automation' / 'p2_acceptance.json'
        atomic_write_json(path, {**result['p2_acceptance'], 'acceptance': result['acceptance']})
        result['p2_acceptance_path'] = str(path)
    return result


def main():
    request_path, response_path, owner_pid = sys.argv[1:]
    done = threading.Event()
    def watch_owner():
        while not done.wait(1):
            if not process_alive(int(owner_pid)):
                # The task owns this process group/tree and its P1 descendants.
                stop_tree(os.getpid())
                os._exit(2)
    threading.Thread(target=watch_owner, daemon=True).start()
    request = json.loads(Path(request_path).read_text(encoding='utf-8'))
    start = time.perf_counter()
    try:
        result = execute(request)
    except Exception as exc:
        result = {'error': f'{type(exc).__name__}: {exc}'}
    result.update(task_id=request['task_id'], seconds=time.perf_counter() - start)
    atomic_write_json(response_path, result)
    done.set()
    return 1 if result.get('error') else 0


if __name__ == '__main__':
    raise SystemExit(main())

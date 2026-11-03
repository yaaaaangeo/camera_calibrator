"""Durable, Qt-free P2 state machine. Heavy work is requested, never run here."""
from __future__ import annotations

from dataclasses import asdict, replace
from enum import Enum
import json
import math
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

from automation.live.readiness import check_readiness
from automation.live.assessment import POLICY_VERSION, assess_observability
from automation.profile import options_to_argv, build_cli_options, parse_cli_argv
from automation.run_manifest import canonical_sha256, iso, sha256_file, utc_now
from automation.live.storage import write_json as atomic_write_json
from calibration.detector import build_detect_fn
from calibration.occlusion import validate_binding, mask_metadata
from calibration.image_quality import (compute_contrast, compute_motion_blur_score, compute_phash,
                                       compute_saturation, compute_pattern_exposure_metrics, hamming_distance)
from calibration.project_io import project_to_dict, project_from_dict
from calibration.types import CameraConfig, CalibrationProject, Dataset, Frame, ImageInfo
from export.output_manager import OutputManager


class State(str, Enum):
    IDLE = 'IDLE'
    CONNECTING = 'CONNECTING'
    CAPTURING = 'CAPTURING'
    CHECKING_READINESS = 'CHECKING_READINESS'
    CALIBRATING = 'CALIBRATING'
    EVALUATING = 'EVALUATING'
    COMPLETED = 'COMPLETED'
    ERROR = 'ERROR'
    CANCELLED = 'CANCELLED'


TERMINAL = {State.COMPLETED, State.ERROR, State.CANCELLED}


class SessionLock:
    """OS-owned lock, released on crashes; never remove/reclaim a PID file."""
    def __init__(self, session):
        self.handle = open(Path(session) / '.p2.lock', 'a+b')
        self.handle.seek(0)
        self.handle.write(b'0')
        self.handle.flush()
        self.handle.seek(0)
        try:
            if __import__('os').name == 'nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RuntimeError('P2 session is already in use') from None

    def close(self):
        if not self.handle.closed:
            self.handle.close()


class LiveController:
    def __init__(self, profile, identity, source: dict, *, output_root=None, resume=None, clock=time.monotonic):
        self.profile, self.identity, self.source, self.clock = profile, identity, source, clock
        validate_binding(profile.coverage_mask, identity=identity)
        self.started = clock()
        self.last_received = None
        self.last_saved = None
        self.last_checkpoint = clock()
        self.detect = None
        self.state = State.IDLE
        self.dataset = Dataset()
        self.camera = None
        self.processed = self.rejected = self.duplicates = 0
        self.calculations = self.recaptures = self.checks = self.reconnects = 0
        self.last_checked_count = self.last_calculated_count = 0
        self.guidance = []
        self.readiness = None
        self.pending = None
        self.attempts = []
        self.hashes = {}
        self.last_detection = None
        self.last_quality = None
        self.connection = 'waiting'
        self.seq = 0
        self.recover_retry = False
        self.last_calculation_fingerprint = None
        options = build_cli_options(profile.p1, image_paths=['validation'], sensor_name=identity.label,
                                    save_project='x', json_summary='x', log_file='x')
        _, self.pattern = parse_cli_argv(options_to_argv(options))
        self.session = Path(resume).resolve() if resume else OutputManager(output_root).reserve_session_directory(identity.label + '_capture')
        self.lock = SessionLock(self.session)
        try:
            if resume:
                self._restore()
            else:
                OutputManager(self.session.parent, camera_name=identity.label, session_dir=self.session)
                (self.session / 'captures').mkdir()
                (self.session / 'tasks').mkdir()
                self._transition(State.CONNECTING, 'start')
                manager = OutputManager(self.session.parent)
                manager.session_dir = self.session
                manager.manifest = json.loads((self.session / 'manifest.json').read_text(encoding='utf-8'))
                manager.record_export('automation.p2_manifest', self.session / 'p2_manifest.json')
                manager.record_export('automation.p2_events', self.session / 'capture_events.jsonl')
            self.detect = build_detect_fn(self.pattern)
        except BaseException:
            self.lock.close()
            raise

    @property
    def paths(self):
        return [f.image_info.path for f in self.dataset.frames]

    def event(self, kind, **fields):
        self.seq += 1
        record = {'seq': self.seq, 'at': iso(utc_now()), 'event': kind, 'state': self.state.value,
                  'processed': self.processed, 'rejected': self.rejected, 'duplicates': self.duplicates, **fields}
        with open(self.session / 'capture_events.jsonl', 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')

    def _transition(self, state, reason):
        self.state = state
        self.event('state', reason=reason)
        self.checkpoint()

    def checkpoint(self):
        project = (project_to_dict(CalibrationProject('P2 capture', self.camera, self.pattern, self.dataset))
                   if self.camera is not None else None)
        atomic_write_json(self.session / 'p2_manifest.json', {
            'schema_version': 1, 'gate_policy_version': POLICY_VERSION, 'profile_hash': self.profile.fingerprint,
            'profile': self.profile.raw, 'identity': self.identity.to_dict(), 'source': self.source,
            'coverage_mask': mask_metadata(self.profile.coverage_mask),
            'state': self.state.value, 'seq': self.seq, 'project': project, 'hashes': self.hashes,
            'processed': self.processed, 'rejected': self.rejected, 'duplicates': self.duplicates,
            'calculations': self.calculations, 'recaptures': self.recaptures, 'checks': self.checks,
            'reconnects': self.reconnects, 'last_checked_count': self.last_checked_count,
            'last_calculated_count': self.last_calculated_count, 'guidance': self.guidance,
            'readiness': self.readiness, 'pending': self.pending, 'attempts': self.attempts,
            'last_calculation_fingerprint': self.last_calculation_fingerprint})
        self.last_checkpoint = self.clock()

    def _restore(self):
        data = json.loads((self.session / 'p2_manifest.json').read_text(encoding='utf-8'))
        if data['profile_hash'] != self.profile.fingerprint or data['identity'] != self.identity.to_dict():
            raise ValueError('Recovery requires identical profile and camera identity')
        if data['source'] != self.source:
            raise ValueError('Recovery source/topic binding differs; do not mix cameras in a session')
        self.hashes = data['hashes']
        self.last_calculation_fingerprint = data.get('last_calculation_fingerprint')
        for path, digest in self.hashes.items():
            resolved = Path(path).resolve()
            if not resolved.is_relative_to(self.session / 'captures') or sha256_file(resolved) != digest:
                raise ValueError(f'Capture missing, changed, or outside session: {path}')
        if data['project']:
            project = project_from_dict(data['project'])
            self.camera, self.dataset = project.camera_config, project.dataset
        for key in ('processed', 'rejected', 'duplicates', 'calculations', 'recaptures', 'checks',
                    'reconnects', 'last_checked_count', 'last_calculated_count', 'seq',
                    'guidance', 'readiness', 'attempts'):
            setattr(self, key, data[key])
        # Recover audit counters beyond the last durable accepted-image checkpoint.
        for line in (self.session / 'capture_events.jsonl').read_text(encoding='utf-8').splitlines():
            try:
                event = json.loads(line)
                self.seq = max(self.seq, event['seq'])
                for key in ('processed', 'rejected', 'duplicates'):
                    setattr(self, key, max(getattr(self, key), event.get(key, 0)))
            except (ValueError, KeyError):
                continue  # A torn last journal record is not a completed capture.
        if data['state'] == State.COMPLETED.value:
            self.state = State.COMPLETED
            return
        pending = data.get('pending')
        self.recover_retry = data['state'] in {State.CHECKING_READINESS.value, State.CALIBRATING.value, State.EVALUATING.value}
        self.pending = None
        self.state = State.CONNECTING
        self.event('recovery', previous_state=data['state'], interrupted_task=pending)
        if pending and Path(pending['response']).is_file():
            self.pending = pending
            self.state = State(pending['state'])
            response = json.loads(Path(pending['response']).read_text(encoding='utf-8'))
            if pending['kind'] == 'calibration' and not response.get('p2_acceptance'):
                # Preserve old result files. A cached P1-only result cannot
                # silently acquire approval under the new P2 policy.
                response['p1_outcome'] = dict(response['outcome'])
                if response['outcome'].get('verdict') == 'PASS':
                    response['outcome'].update(verdict='WARN', exit_code=3)
                response['acceptance'] = dict(response.get('acceptance') or {})
                response['acceptance']['reasons'] = response['acceptance'].get('reasons', []) + ['이전 정책의 완료 결과입니다. 현재 P2 품질 검토가 필요합니다.']
                response['p2_acceptance'] = {'policy_version': POLICY_VERSION, 'observability_only': True,
                                             'review_required': True, 'legacy_result': True}
                self.event('legacy_result_review', task_id=pending['task_id'])
            self.finish_task(response)
        else:
            self.last_checked_count = max(0, len(self.paths) - self.profile.readiness.check_every_new_frames)
            self.checkpoint()

    def fail(self, code, detail):
        self.guidance = [detail]
        self.event('error', code=code, detail=detail)
        self._transition(State.ERROR, code)

    def connected(self):
        if self.state == State.CONNECTING:
            self._transition(State.CAPTURING, 'stream_available')
        self.connection = 'connected'
        self.last_received = self.clock()

    def stream_check(self):
        if self.state not in {State.CONNECTING, State.CAPTURING}:
            return False
        elapsed = self.clock() - (self.last_received if self.last_received is not None else self.started)
        if elapsed < self.profile.capture.stream_timeout_sec:
            return False
        self.connection = 'disconnected'
        if self.reconnects >= self.profile.capture.max_reconnects:
            self.fail('stream_timeout', '영상 연결이 복구되지 않았습니다. 토픽/카메라를 확인해 주세요.')
            return False
        self.reconnects += 1
        self.last_received = self.clock()
        self.guidance = ['영상 수신 중단: 같은 토픽으로 재연결 중입니다.']
        self._transition(State.CONNECTING, 'stream_timeout_reconnect')
        return True

    def _reject(self, reason, details):
        self.rejected += 1
        if reason in {'duplicate', 'duplicate_pose'}:
            self.duplicates += 1
        self.event('frame_rejected', reason=reason, **details)
        if self.clock() - self.last_checkpoint >= 2:
            self.checkpoint()
        return reason

    def accept_frame(self, image, source_timestamp=0.0, detection=None):
        if self.state not in {State.CAPTURING, State.CONNECTING}:
            return 'paused'
        self.connected()
        self.processed += 1
        now = self.clock()
        h, w = image.shape[:2]
        try:
            validate_binding(self.profile.coverage_mask, w, h)
        except ValueError as exc:
            self.fail('mask_resolution_mismatch', str(exc))
            return 'mask_resolution_mismatch'
        if self.camera is not None and (w, h) != (self.camera.width, self.camera.height):
            self.fail('resolution_changed', '입력 해상도가 바뀌었습니다. 새 세션을 시작해 주세요.')
            return 'resolution_changed'
        expected = self.profile.p1.camera
        if expected.get('width') is not None and (w, h) != (expected['width'], expected['height']):
            self.fail('resolution_mismatch', '프로파일 해상도와 입력 해상도가 다릅니다.')
            return 'resolution_mismatch'
        image_id = f'capture_{len(self.paths):06d}_' + uuid.uuid4().hex
        t0 = time.perf_counter()
        try:
            det = detection if detection is not None else self.detect(image, image_id)
            det.image_id = image_id
        except Exception as exc:
            return self._reject('detection_error', {'detail': str(exc)})
        self.last_detection = {key: getattr(det, key) for key in (
            'success', 'num_corners', 'board_center_px', 'board_area_ratio', 'board_tilt_deg',
            'min_edge_margin_px', 'failure_reason')}
        self.last_detection['processing_seconds'] = time.perf_counter() - t0
        details = {'detection': self.last_detection, 'source_timestamp': float(source_timestamp),
                   'detection_seconds': time.perf_counter() - t0}
        if not det.success or det.num_corners < self.profile.capture.min_corners:
            return self._reject('detection_failed', details)
        if self.last_saved is not None and now - self.last_saved < self.profile.capture.interval_sec:
            return self._reject('interval', details)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        metrics = compute_pattern_exposure_metrics(gray, det.corners)
        if metrics is None:
            return self._reject('pattern_roi_unavailable', details)
        quality = self.profile.p1.image_selection
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        motion = compute_motion_blur_score(gray)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(np.asarray(det.corners, np.float32)).astype(np.int32), 255)
        roi = gray[mask > 0]
        contrast = float(roi.std()) if roi.size else 0.0
        self.last_quality = {'sharpness': sharpness, 'motion_blur_score': motion, 'roi_contrast': contrast,
                             **asdict(metrics)}
        details['quality'] = self.last_quality
        if sharpness < quality.blur_min_laplacian_var:
            return self._reject('blur', details)
        if motion > self.profile.capture.motion_blur_max:
            return self._reject('motion_blur', details)
        if contrast < self.profile.capture.roi_contrast_min:
            return self._reject('roi_contrast', details)
        if not quality.brightness_min <= metrics.roi_brightness <= quality.brightness_max:
            return self._reject('exposure_brightness', details)
        if metrics.corner_clip_median > quality.corner_clip_max:
            return self._reject('exposure_clipping', details)
        phash = compute_phash(gray)
        for prior in self.dataset.frames:
            old = prior.detection
            center_distance = (math.dist(det.board_center_px, old.board_center_px) / math.hypot(w, h)
                               if det.board_center_px is not None and old.board_center_px is not None else 0)
            area_delta = (abs(det.board_area_ratio - old.board_area_ratio) / max(old.board_area_ratio, 1e-8)
                          if det.board_area_ratio is not None and old.board_area_ratio is not None else 0)
            tilt_delta = abs((det.board_tilt_deg or 0) - (old.board_tilt_deg or 0))
            same_pose = (center_distance < self.profile.capture.center_delta
                         and area_delta < self.profile.capture.relative_area_delta
                         and tilt_delta < self.profile.capture.tilt_delta_deg)
            # Hash alone can confuse small boards on plain backgrounds; retain geometry gate.
            if same_pose and (prior.image_info.phash is None or
                              hamming_distance(phash, prior.image_info.phash) <= quality.duplicate_phash_distance):
                return self._reject('duplicate', details)
            if same_pose:
                return self._reject('duplicate_pose', details)
        if len(self.paths) >= self.profile.capture.max_images:
            self.fail('capture_limit', '촬영 상한에 도달했습니다. 분포/프로파일을 수동 검토해 주세요.')
            return 'capture_limit'
        path = self.session / 'captures' / f'{image_id}.png'
        ok, encoded = cv2.imencode('.png', image)
        if not ok:
            raise OSError('PNG encoding failed')
        with open(path, 'xb') as stream:
            stream.write(encoded.tobytes())
            stream.flush()
            __import__('os').fsync(stream.fileno())
        self.camera = CameraConfig(w, h, sensor_name=self.identity.label)
        info = ImageInfo(image_id, str(path), w, h, sharpness=sharpness, brightness=float(gray.mean()),
                         contrast=compute_contrast(gray), saturation=compute_saturation(gray),
                         motion_blur_score=motion, phash=phash)
        self.dataset.frames.append(Frame(info, det))
        self.hashes[str(path)] = sha256_file(path)
        self.last_saved = now
        self.event('frame_saved', path=str(path), **details)
        self.checkpoint()
        return 'saved'

    def maybe_check(self, *, force=False):
        if self.state != State.CAPTURING or len(self.paths) < self.profile.readiness.min_images:
            return None
        required_new = self.profile.recovery.min_new_frames if self.calculations and not self.recover_retry else 0
        if len(self.paths) - self.last_calculated_count < required_new:
            return None
        new_count = len(self.paths) - self.last_checked_count
        if new_count <= 0 or (not force and new_count < self.profile.readiness.check_every_new_frames):
            return None
        if self.checks >= self.profile.readiness.max_checks:
            self.fail('readiness_limit', '준비도 검사 횟수 초과: 수동 검토가 필요합니다.')
            return None
        self.last_checked_count = len(self.paths)
        self.readiness, selected = check_readiness(self.dataset, self.camera, self.pattern, self.profile)
        self.guidance = self.readiness['reasons'] or self.readiness['coaching']
        self.event('readiness', result=self.readiness)
        if not self.readiness['ready']:
            self.checkpoint()
            return None
        fingerprint = self.calculation_fingerprint()
        if fingerprint == self.last_calculation_fingerprint and not self.recover_retry:
            self.guidance = self.readiness['coaching'] + ['선택된 학습/검증 데이터가 이전 계산과 같습니다. 같은 계산을 반복하지 않습니다.']
            self.event('unchanged_calculation_inputs', fingerprint=fingerprint)
            self.checkpoint()
            return None
        self.checks += 1
        self._transition(State.CHECKING_READINESS, 'preliminary_observability')
        return self.request_task('observability', selected)

    def calculation_fingerprint(self):
        return canonical_sha256({'selected': sorted(self.readiness.get('selected_paths', self.paths)),
                                 'train': sorted(self.readiness['train_ids']),
                                 'test': sorted(self.readiness.get('test_ids', []))})

    def request_task(self, kind, dataset=None):
        task_id = uuid.uuid4().hex
        folder = self.session / 'tasks' / task_id
        folder.mkdir()
        request = {'task_id': task_id, 'kind': kind, 'gate_policy_version': POLICY_VERSION, 'profile': self.profile.raw,
                   'identity': self.identity.to_dict(), 'session': str(self.session),
                   'paths': self.paths, 'hashes': self.hashes,
                   'project': project_to_dict(CalibrationProject('P2 task', self.camera, self.pattern,
                                                               dataset or self.dataset)),
                   'train_ids': self.readiness['train_ids'], 'attempt': self.calculations,
                   'preflight': self.readiness.get('observability')}
        atomic_write_json(folder / 'request.json', request)
        self.pending = {'task_id': task_id, 'kind': kind, 'request': str(folder / 'request.json'),
                        'response': str(folder / 'response.json'), 'state': self.state.value}
        self.checkpoint()
        return self.pending

    def finish_task(self, response):
        if not self.pending or response.get('task_id') != self.pending['task_id']:
            raise ValueError('Stale/mismatched task response')
        kind = self.pending['kind']
        if response.get('error'):
            self.pending = None
            self.fail('task_error', response['error'])
            return None
        if kind == 'observability':
            self.pending = None
            obs = response['observability']
            if obs.get('report'):
                # Reassess cached v1 reports without refitting or trusting their
                # old binary score-based gate decision.
                obs = {**obs, **assess_observability(obs['report'], self.profile.readiness)}
            self.readiness['observability'] = obs
            self.guidance = list(dict.fromkeys(obs['reasons'] + self.readiness.get('coaching', [])))
            if not obs['ready']:
                self._transition(State.CAPTURING, 'observability_insufficient')
                return None
            if self.calculations >= self.profile.recovery.max_calculations:
                self.fail('calculation_limit', '최대 계산 횟수 초과: 수동 검토가 필요합니다.')
                return None
            self.calculations += 1
            self.recover_retry = False
            self.last_calculated_count = len(self.paths)
            self.last_calculation_fingerprint = self.calculation_fingerprint()
            self._transition(State.CALIBRATING, 'readiness_satisfied')
            return self.request_task('calibration')
        self._transition(State.EVALUATING, 'p1_finished')
        self.pending = None
        self.attempts.append(response)
        outcome = response['outcome']
        acceptance = response.get('acceptance') or {}
        self.guidance = response.get('capture_recommendations') or acceptance.get('reasons', [])
        if outcome['status'] != 'completed':
            # Setup/system/export failures must not launch a blind recapture loop.
            self.fail('p1_execution_failed', outcome['message'])
        elif outcome['verdict'] == 'PASS':
            self._transition(State.COMPLETED, 'PASS')
        elif (set(acceptance.get('fail_kinds', [])) - {'quality', 'missing_metric'}):
            self.fail('invalid_or_system_result', '; '.join(acceptance.get('reasons', [])))
        elif 'criteria_missing' in acceptance.get('warn_kinds', []) or not acceptance.get('criteria_complete', False):
            self.guidance = list(dict.fromkeys(acceptance.get('reasons', [])
                + response.get('capture_recommendations', []) + self.readiness.get('coaching', [])
                + ['운영 기준/필수 지표를 검토하세요. 반복 촬영하지 않습니다.']))
            self._transition(State.COMPLETED, 'manual_review_criteria')
        elif response.get('p2_acceptance', {}).get('observability_only'):
            self.guidance = list(dict.fromkeys(acceptance.get('reasons', [])
                + response.get('capture_recommendations', []) + self.readiness.get('coaching', [])
                + ['관측성 경고만으로 반복 촬영하지 않습니다. 결과를 수동 검토하세요.']))
            self._transition(State.COMPLETED, 'manual_review_observability')
        elif response.get('capture_recommendations'):
            if self.recaptures >= self.profile.recovery.max_recaptures:
                self.fail('recapture_limit', '추가 촬영 횟수 초과: 수동 검토가 필요합니다.')
            else:
                self.recaptures += 1
                self._transition(State.CAPTURING, 'additional_capture')
        else:
            self._transition(State.COMPLETED, 'manual_review_no_capture_remedy')
        return None

    def cancel(self):
        self.pending = None
        self._transition(State.CANCELLED, 'user_cancelled')

    def snapshot(self):
        progress = None
        if self.state == State.CALIBRATING:
            for manifest in (self.session / 'runs' / f'attempt_{self.calculations:03d}').glob('*/run_manifest.json'):
                try:
                    data = json.loads(manifest.read_text(encoding='utf-8'))
                    progress = [{'name': s['name'], 'status': s['status']} for s in data.get('stages', [])]
                except (OSError, ValueError):
                    pass
        return {'state': self.state.value, 'session': str(self.session), 'connection': self.connection,
                'saved': len(self.paths), 'processed': self.processed, 'rejected': self.rejected,
                'duplicates': self.duplicates, 'calculations': self.calculations, 'recaptures': self.recaptures,
                'guidance': self.guidance, 'readiness': self.readiness, 'detection': self.last_detection,
                'quality': self.last_quality, 'calibration_progress': progress,
                  'attempts': [{'outcome': a['outcome'], 'recommendation': a.get('recommendation'),
                                'p1_verdict': a.get('p1_outcome', {}).get('verdict'),
                                'p2_acceptance_path': a.get('p2_acceptance_path')}
                                                         for a in self.attempts]}

    def close(self):
        self.lock.close()

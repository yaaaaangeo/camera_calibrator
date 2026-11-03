from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

from automation.identity import build_identity
from automation.live.config import load_live_profile
from automation.live.controller import LiveController, State, TERMINAL
from automation.live.process import TaskProcess
from automation.live.sources import ImageReplay, BagReplay, RosSource


def run(controller, source, *, emit=None, poll_sec=.05, replay_interval=.1):
    if emit is None:
        emit = lambda value: print(value, flush=True)
    task = None
    task_started = None
    last_emit = 0.0
    next_frame = 0.0
    cancel_file = controller.session / 'cancel.request'
    cancel_file.unlink(missing_ok=True)  # explicit resume clears only its own control flag
    def status():
        snapshot = controller.snapshot()
        if hasattr(source, 'buffer'):
            from dataclasses import asdict
            snapshot['buffer'] = asdict(source.buffer.stats())
            controller.event('source_stats', stats=snapshot['buffer'])
        emit(json.dumps(snapshot, ensure_ascii=False, allow_nan=False))
    status()
    try:
        if controller.state not in TERMINAL:
            try:
                source.start()
            except Exception as exc:
                if not isinstance(source, RosSource):
                    raise
                source.stop()
                controller.event('connect_error', detail=str(exc))
        if controller.pending and controller.state in {State.CHECKING_READINESS, State.CALIBRATING}:
            task = TaskProcess(controller.pending)
            task_started = time.monotonic()
        while controller.state not in TERMINAL:
            now = time.monotonic()
            if cancel_file.exists():
                if task is not None:
                    task.stop()
                    task = None
                controller.cancel()
                break
            if task is not None:
                timeout = (controller.profile.readiness.timeout_sec if controller.pending['kind'] == 'observability'
                           else controller.profile.recovery.calculation_timeout_sec)
                if now - task_started > timeout:
                    task.stop()
                    task = None
                    controller.fail('task_timeout', '계산 제한 시간을 초과했습니다. 수동 검토 후 재시작해 주세요.')
                    break
                if task.poll() is not None:
                    code = task.poll()
                    task.close()
                    task = None
                    response = Path(controller.pending['response'])
                    if not response.is_file():
                        controller.fail('task_crashed', f'계산 프로세스 종료({code}), 완료 결과 없음')
                        break
                    new_task = controller.finish_task(json.loads(response.read_text(encoding='utf-8')))
                    source.clear()
                    controller.last_received = now
                    if new_task:
                        task = TaskProcess(new_task)
                        task_started = now
            else:
                if isinstance(source, RosSource):
                    while source.errors:
                        controller.event('decode_error', detail=source.errors.popleft())
                    if controller.stream_check():
                        source.stop()
                        deadline = time.monotonic() + controller.profile.capture.reconnect_interval_sec
                        while time.monotonic() < deadline and not cancel_file.exists():
                            time.sleep(.05)
                        try:
                            source.start()
                        except Exception as exc:
                            controller.event('reconnect_error', detail=str(exc))
                if controller.state in {State.CONNECTING, State.CAPTURING} and now >= next_frame:
                    frame = source.poll()
                    if frame is not None:
                        controller.accept_frame(*frame)
                    next_frame = now + (replay_interval if not isinstance(source, RosSource) else poll_sec)
                    request = (controller.maybe_check(force=True) if source.exhausted
                               else controller.maybe_check())
                    if request:
                        source.clear()
                        task = TaskProcess(request)
                        task_started = now
                    elif source.exhausted:
                        controller.fail('input_exhausted', '재생 입력이 끝났습니다. 준비도 미충족: ' + '; '.join(controller.guidance))
            if now - last_emit >= .5:
                status()
                last_emit = now
            time.sleep(poll_sec)
    except KeyboardInterrupt:
        if task is not None:
            task.stop()
            task = None
        controller.cancel()
    except Exception as exc:
        if task is not None:
            task.stop()
            task = None
        try:
            controller.fail('runtime_error', f'{type(exc).__name__}: {exc}')
        except OSError:
            emit(json.dumps({'state': 'ERROR', 'session': str(controller.session), 'guidance': ['세션 저장 오류: ' + str(exc)]}))
    finally:
        if task is not None:
            task.stop()
        source.stop()
        status()
        controller.close()
    return 0 if controller.state == State.COMPLETED else 130 if controller.state == State.CANCELLED else 2


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m automation live')
    parser.add_argument('--profile', default='default')
    parser.add_argument('--serial', required=True)
    parser.add_argument('--camera-model', required=True)
    parser.add_argument('--output-root')
    parser.add_argument('--resume', help='Existing P2 capture session; profile/identity/source must match')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--replay', help='Saved-image directory')
    group.add_argument('--bag', help='ROS1/ROS2 bag path')
    group.add_argument('--ros', action='store_true')
    parser.add_argument('--topic')
    parser.add_argument('--msg-type', default='sensor_msgs/msg/Image')
    parser.add_argument('--replay-interval', type=float, default=1.1)
    from automation.publish.cli import add_options
    add_options(parser)
    from automation.factory.service import add_options as add_factory_options
    add_factory_options(parser)
    args = parser.parse_args(argv)
    try:
        if not args.replay and not args.topic:
            parser.error('--topic is required for ROS/rosbag')
        if not 0 < args.replay_interval <= 60:
            parser.error('--replay-interval must be >0 and <=60')
        profile = load_live_profile(args.profile)
        identity = build_identity(args.camera_model, args.serial)
        if args.replay:
            binding = {'kind': 'images', 'path': str(Path(args.replay).expanduser().resolve())}
            source = ImageReplay(binding['path'])
        elif args.bag:
            binding = {'kind': 'bag', 'path': str(Path(args.bag).expanduser().resolve()), 'topic': args.topic}
            source = BagReplay(binding['path'], args.topic)
        else:
            binding = {'kind': 'ros', 'topic': args.topic, 'msg_type': args.msg_type}
            source = RosSource(args.topic, args.msg_type)
        controller = LiveController(profile, identity, binding, output_root=args.output_root, resume=args.resume)
    except (ValueError, OSError, RuntimeError) as exc:
        print(json.dumps({'state': 'ERROR', 'guidance': [str(exc)]}, ensure_ascii=False))
        return 1
    code=run(controller, source, replay_interval=args.replay_interval)
    factory_event=None
    if controller.state==State.COMPLETED and controller.attempts:
        from automation.factory.service import post_compare
        session=controller.attempts[-1]['outcome'].get('session_dir')
        if session:
            factory_event=post_compare(session,args)
    if args.publish and controller.state==State.COMPLETED and controller.attempts:
        from automation.publish.service import post_publish
        session=controller.attempts[-1]['outcome'].get('session_dir')
        if session:
            options={'full_frame':args.publish_full_frame,'roi':args.publish_roi}
            if factory_event and factory_event.get('report'):
                options['factory_report']=factory_event['report']
            print(json.dumps({'event':'publication',**post_publish(session,args.publish_config,**options)},ensure_ascii=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())

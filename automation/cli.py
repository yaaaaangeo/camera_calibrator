"""Command line for the automation layer.

    python -m automation run --profile default \\
        --images ./captures --serial SN123 --camera-model SG3S-ISX031C

    python -m automation check --profile default [--images ./captures]

``--profile``은 YAML 파일 경로 또는 내장 프로파일 이름(``default``)이다.

종료 코드 (run):
    0 = 실행 완료, 판정 PASS
    3 = 실행 완료, 판정 WARN (검토 필요 또는 운영 기준 미정)
    4 = 실행 완료, 판정 FAIL (품질 기준 미달)
    1 = 프로파일/입력 오류 - 계산을 시작하지 않았고 세션도 만들지 않음
    2 = 실행 실패 - 세션의 run_manifest.json에 실패 단계와 원인이 기록됨

``--camera-model``은 카메라 하드웨어 모델명(예: SG3S-ISX031C)이다. 기존
app.cli의 ``--model``(최종 calibration 모델 강제 선택)과 혼동하지 않도록,
calibration 모델 강제 선택은 프로파일 ``calibration.model``로만 지정한다.
"""

from __future__ import annotations

import argparse
import sys

from automation.identity import IdentityError, build_identity
from automation.profile import ProfileError, load_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m automation",
        description="Camera intrinsic calibration 자동 실행 (기존 app.cli 계산 파이프라인 사용).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("종료 코드 (run):", 1)[1] if __doc__ else None,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser('experiment', help='Integrated experiment preflight/run/retry (experiment --help)', add_help=False)
    sub.add_parser('factory', help='P4 Factory reference lookup/comparison (factory --help)', add_help=False)
    sub.add_parser('live', help='P2 실시간 자동 촬영·준비도 검사·P1 계산 (live --help)', add_help=False)
    sub.add_parser('publish', help='P3 결과 패키지/Outbox/게시 (publish --help)', add_help=False)
    sub.add_parser('history', help='P3 카메라별 로컬 이력/비교 (history --help)', add_help=False)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--profile", required=True, help="프로파일 YAML 경로 또는 내장 이름 (예: default)")
        p.add_argument(
            "--camera-model", "--model", dest="camera_model",
            help="카메라 하드웨어 모델명. 프로파일 camera.model보다 우선",
        )
        p.add_argument("--seed", type=int, help="Train/Hold-out 분할 seed. 프로파일 calibration.seed보다 우선")
        p.add_argument("--output-root", help="결과 루트 폴더. 프로파일 output.root보다 우선")

    run = sub.add_parser("run", help="검출→선별→계산→Hold-out→추천→판정→저장 전체 실행")
    add_common(run)
    run.add_argument("--images", nargs="+", required=True, help="이미지 파일/디렉터리/glob (기존 CLI와 같은 규칙)")
    run.add_argument("--serial", required=True, help="카메라 Serial Number (수동 입력)")
    run.add_argument("--quiet", action="store_true", help="진행 메시지를 출력하지 않음")
    from automation.publish.cli import add_options
    add_options(run)
    from automation.factory.service import add_options as add_factory_options
    add_factory_options(run)

    check = sub.add_parser("check", help="계산 없이 프로파일·입력만 검증")
    add_common(check)
    check.add_argument("--images", nargs="+", help="지정하면 이미지 경로·해상도도 확인")
    check.add_argument("--serial", help="지정하면 Serial 형식도 확인")
    return parser


def _load(args):
    return load_profile(
        args.profile,
        camera_model=args.camera_model,
        seed=args.seed,
        output_root=args.output_root,
    )


def _cmd_check(args) -> int:
    from automation.runner import check_image_sizes, resolve_input_images

    try:
        profile = _load(args)
        if args.serial is not None or profile.camera.get("model"):
            build_identity(profile.camera.get("model"), args.serial or "CHECK-ONLY")
        if args.images:
            paths = resolve_input_images(args.images)
            expected = None
            if profile.camera.get("width") is not None:
                expected = (profile.camera["width"], profile.camera["height"])
            width, height = check_image_sizes(paths, expected)
            print(f"이미지 {len(paths)}장, 해상도 {width}x{height}")
    except (ProfileError, IdentityError) as exc:
        print(f"설정 오류:\n{exc}", file=sys.stderr)
        return 1
    print("프로파일 검증 통과")
    return 0


def _cmd_run(args) -> int:
    from automation.runner import run_automation

    try:
        profile = _load(args)
        identity = build_identity(profile.camera.get("model"), args.serial)
    except (ProfileError, IdentityError) as exc:
        print(f"설정 오류 - 계산을 시작하지 않았습니다:\n{exc}", file=sys.stderr)
        return 1

    log = (lambda _msg: None) if args.quiet else print
    outcome = run_automation(profile, identity, args.images, log=log)
    if outcome.status == "rejected":
        print(f"입력 오류 - 계산을 시작하지 않았습니다:\n{outcome.message}", file=sys.stderr)
    elif outcome.status == "failed":
        print(f"실행 실패 (run_id={outcome.run_id}): {outcome.session_dir}", file=sys.stderr)
    else:
        print(f"run_id={outcome.run_id} verdict={outcome.verdict}")
        print(f"결과: {outcome.session_dir}")
    factory_event = None
    if outcome.status == 'completed' and outcome.session_dir:
        from automation.factory.service import post_compare
        factory_event = post_compare(outcome.session_dir, args)
    if getattr(args,'publish',False) and outcome.session_dir:
        import json
        from automation.publish.service import post_publish
        options={'full_frame':args.publish_full_frame,'roi':args.publish_roi}
        if factory_event and factory_event.get('report'):
            options['factory_report']=factory_event['report']
        publication=post_publish(outcome.session_dir,args.publish_config,**options)
        print(json.dumps({'event':'publication',**publication},ensure_ascii=False))
    return outcome.exit_code


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == 'experiment':
        from automation.experiment.cli import main as experiment_main
        return experiment_main(arguments[1:])
    if arguments and arguments[0] == 'factory':
        from automation.factory.cli import main as factory_main
        return factory_main(arguments[1:])
    if arguments and arguments[0] in {'publish','history'}:
        from automation.publish.cli import main as publish_main
        return publish_main(arguments[1:],history=arguments[0]=='history')
    if arguments and arguments[0] == 'live':
        from automation.live.cli import main as live_main
        return live_main(arguments[1:])
    args = build_parser().parse_args(argv)
    if args.command == "check":
        return _cmd_check(args)
    return _cmd_run(args)

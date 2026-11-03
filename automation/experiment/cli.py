import argparse
import json
import sys


def main(argv=None):
    parser=argparse.ArgumentParser(prog='python -m automation experiment')
    parser.add_argument('action',choices=['preflight','run','retry-publish'])
    parser.add_argument('--profile',required=True)
    parser.add_argument('--dry-run',action='store_true',help='Read-only plan; no output/lock/outbox/calibration/push')
    parser.add_argument('--network',action='store_true',help='Opt-in read-only ls-remote probe (no push); ignored during dry-run')
    parser.add_argument('--verify-write',action='store_true',help='Preflight owned temporary write/fsync probe; ignored during dry-run')
    parser.add_argument('--reset-failed',action='store_true',help='Explicit publication retry after correcting permanent error')
    parser.add_argument('--reuse-session',help='Adopt a completed image-input Session after identity/input/pattern verification; no recalibration')
    args=parser.parse_args(argv)
    if args.reuse_session and args.action!='run':
        parser.error('--reuse-session is only supported by run (including run --dry-run)')
    try:
        from automation.experiment.config import load_experiment
        exp=load_experiment(args.profile)
        if args.dry_run or args.action=='preflight':
            from automation.experiment.preflight import preflight
            result=preflight(exp,network=args.network and not args.dry_run,dry_run=args.dry_run,verify_write=args.verify_write,
                publish_only=args.action=='retry-publish')
            if args.action=='retry-publish':result['plan']=['Reuse saved result/outbox','Retry publication only; no input/calibration/Factory fit']
            if args.reuse_session:
                result['plan']=['Adopt explicit historical image Session after identity/input/pattern checks (no calculation)']+result['plan'][3:]
            print(json.dumps(result,ensure_ascii=False,indent=2));return 1 if result['status']=='FAIL' else 0
        from automation.experiment.runner import run_experiment
        return run_experiment(exp,retry_only=args.action=='retry-publish',reset_failed=args.reset_failed,reuse_session=args.reuse_session)
    except Exception as exc:
        from automation.publish.security import redact
        print(json.dumps({'status':'FAIL','error':redact(getattr(exc,'code',str(exc)))},ensure_ascii=False),file=sys.stderr)
        return 1

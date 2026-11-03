from __future__ import annotations
import argparse
import json
from pathlib import Path


def add_options(parser):
    parser.add_argument('--publish',action='store_true',help='Post-export outbox/publication; disabled unless explicitly requested')
    parser.add_argument('--publish-config',help='Separate JSON publication settings; enabled defaults to false')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--publish-full-frame',action='store_true',help='User declaration for compatible history comparisons')
    group.add_argument('--publish-roi',nargs=4,type=int,metavar=('X','Y','WIDTH','HEIGHT'))


def main(argv=None,*,history=False):
    from automation.publish.config import load_config
    from automation.publish.service import outbox_for,publish_session
    from automation.publish.git_publisher import GitPublisher
    from automation.publish.base import PublishError
    parser=argparse.ArgumentParser(prog='python -m automation '+('history' if history else 'publish'))
    parser.add_argument('--publish-config')
    if history:
        parser.add_argument('--serial',required=True);parser.add_argument('--camera-model')
        parser.add_argument('--repository',help='Read-only local results checkout; default: local outbox')
        parser.add_argument('--compare',nargs=2,metavar=('RUN_A','RUN_B'))
    else:
        parser.add_argument('action',nargs='?',choices=['retry'])
        parser.add_argument('--session');parser.add_argument('--job')
        parser.add_argument('--factory-report', help='Optional session-bound P4 comparison JSON')
        parser.add_argument('--reset-failed',action='store_true',help='Explicitly reset permanently failed jobs after fixing configuration/auth')
        group=parser.add_mutually_exclusive_group()
        group.add_argument('--full-frame',action='store_true',help='User declaration: images use the full sensor coordinate system')
        group.add_argument('--roi',nargs=4,type=int,metavar=('X','Y','WIDTH','HEIGHT'))
    args=parser.parse_args(argv)
    try:
        cfg=load_config(args.publish_config);box=outbox_for(cfg)
        if history:
            from automation.publish.history import list_history,compare_runs
            records=list_history(args.repository or box.root,args.serial,args.camera_model)
            if args.compare:
                a,b=([r for r in records if r['run_id']==name] for name in args.compare)
                if len(a)!=1 or len(b)!=1:raise PublishError('HISTORY_RUN_NOT_UNIQUE')
                output=compare_runs(a[0],b[0])
            else:output={'runs':records}
        elif args.action=='retry':
            states=[box.process(s['job_id'],GitPublisher(cfg),reset_failed=args.reset_failed)
                    for s in box.list() if not args.job or s['job_id']==args.job]
            output={'jobs':states}
        elif args.session:
            options={'full_frame':args.full_frame,'roi':args.roi}
            if args.factory_report:
                options['factory_report']=args.factory_report
            output=publish_session(args.session,cfg,**options)
        else:parser.error('--session or retry is required')
        print(json.dumps(output,ensure_ascii=False,allow_nan=False))
        return 2 if output.get('publication_status') in {'FAILED','RETRY_WAIT'} or any(s['status'] in {'FAILED','RETRY_WAIT'} for s in output.get('jobs',[])) else 0
    except Exception as exc:
        print(json.dumps({'publication_status':'FAILED','error':exc.code if isinstance(exc,PublishError) else 'PUBLICATION_COMMAND_FAILED'}))
        return 2

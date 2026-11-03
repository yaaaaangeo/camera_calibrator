"""Optional post-calculation hook; failures never alter the Local verdict."""
import json
from pathlib import Path


def add_options(parser):
    parser.add_argument('--factory', help='Optional P4 reference JSON envelope')
    parser.add_argument('--factory-seeded', action='store_true')
    parser.add_argument('--factory-coordinates', help='Explicit Session coordinate-system JSON')


def post_compare(session, args):
    if not getattr(args, 'factory', None):
        return None
    try:
        from automation.factory.compare import compare_session
        if not args.factory_coordinates:
            raise ValueError('--factory-coordinates is required to compare; Local result preserved')
        out, report = compare_session(session, args.factory, seeded=args.factory_seeded,
            coordinates=json.loads(Path(args.factory_coordinates).read_text(encoding='utf-8-sig')))
        event = {'status': report['status'], 'conclusion': report['conclusion'], 'report': str(out/'factory_comparison.json')}
    except Exception as exc:
        event = {'status': 'INCOMPARABLE', 'error': str(exc), 'local_result_preserved': True}
        # Persist a separate diagnostic even when loading fails. Do not attach an
        # incomplete comparison to P3 as if it were a completed result.
        try:
            from uuid import uuid4
            out = Path(session).resolve().parent / 'factory_comparisons' / uuid4().hex
            out.mkdir(parents=True, exist_ok=False)
            report = {'schema_version':1, 'status':'INCOMPARABLE', 'conclusion':'INCOMPARABLE',
                'candidates':{}, 'error':str(exc), 'local_result_preserved':True}
            path = out/'factory_comparison.json'
            path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
            event['diagnostic_report'] = str(path)
        except OSError:
            event['diagnostic_report_status'] = 'WRITE_FAILED'
    print(json.dumps({'event': 'factory_comparison', **event}, ensure_ascii=False))
    return event

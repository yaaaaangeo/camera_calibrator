from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from automation.factory.provider import FactoryError, FileProvider, SensingSDKProvider


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m automation factory')
    sub = p.add_subparsers(dest='command', required=True)
    lookup = sub.add_parser('lookup')
    lookup.add_argument('--serial', required=True)
    lookup.add_argument('--camera-model', required=True)
    lookup.add_argument('--factory', help='Reference JSON envelope; omission reports SDK unavailable')
    compare = sub.add_parser('compare')
    compare.add_argument('--session', required=True)
    compare.add_argument('--factory', required=True)
    compare.add_argument('--seeded', action='store_true')
    compare.add_argument('--coordinates', required=True, help='Session coordinate-system JSON (explicit declaration)')
    compare.add_argument('--output-root')
    args = p.parse_args(argv)
    try:
        if args.command == 'lookup':
            provider = FileProvider(args.factory) if args.factory else SensingSDKProvider()
            r = provider.lookup(args.camera_model, args.serial)
            print(json.dumps({'status': 'LOADED', 'authenticated': False, 'metadata': r.metadata, 'K': r.K.tolist(), 'D': r.D.tolist()}))
            return 0
        from automation.factory.compare import compare_session
        out, report = compare_session(args.session, args.factory, seeded=args.seeded,
            coordinates=json.loads(Path(args.coordinates).read_text(encoding='utf-8-sig')), output_root=args.output_root)
        print(json.dumps({'status': report['status'], 'conclusion': report['conclusion'], 'output': str(out)}))
        return 3 if report['conclusion'] == 'INCOMPARABLE' else 0
    except (FactoryError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({'status': 'INCOMPARABLE', 'error': str(exc)}), file=sys.stderr)
        return 1

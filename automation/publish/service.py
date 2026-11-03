"""Post-export orchestration: publication never changes calibration outcomes."""
from __future__ import annotations
from pathlib import Path

from automation.publish.base import PublishError
from automation.publish.config import load_config
from automation.publish.outbox import Outbox
from automation.publish.git_publisher import GitPublisher


def outbox_for(config):
    from export.output_manager import DEFAULT_OUTPUT_ROOT
    return Outbox(config.outbox or DEFAULT_OUTPUT_ROOT/'publish_outbox',config)


def publish_session(session,config,**options):
    box=outbox_for(config)
    try:state=box.enqueue(session,**options)
    except Exception as exc:
        error=exc.code if isinstance(exc,PublishError) else 'BUNDLE_PREPARATION_FAILED'
        details=exc.details if isinstance(exc,PublishError) else {}
        box.record_error(error,details)
        return {'publication_status':'FAILED','error':error,'details':details,'calibration_unchanged':True}
    state=box.process(state['job_id'],GitPublisher(config))
    return {'publication_status':state['status'],'job_id':state['job_id'],
            'receipt':state.get('receipt'),'error':state.get('error'),'calibration_unchanged':True}


def post_publish(session,config_path=None,**options):
    try:return publish_session(session,load_config(config_path),**options)
    except Exception as exc:
        return {'publication_status':'FAILED','error':exc.code if isinstance(exc,PublishError) else 'PUBLICATION_SETUP_FAILED',
                'calibration_unchanged':True}

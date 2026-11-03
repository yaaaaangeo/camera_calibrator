from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
from time import monotonic, sleep, time
from types import SimpleNamespace

from automation.publish.storage import Lock, write_json, confined
from automation.publish.security import redact
from automation.run_manifest import canonical_sha256, sha256_file, iso, utc_now
from automation.experiment.preflight import preflight


def fingerprint_inputs(source):
    path=Path(source['path'])
    files=sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    if source['kind']=='images':files=[p for p in files if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}]
    if not files:raise ValueError('No input files to fingerprint')
    return canonical_sha256([(str(p.relative_to(path)) if path.is_dir() else p.name,sha256_file(p)) for p in files])


def run_experiment(exp, *, retry_only=False, reset_failed=False, reuse_session=None, emit=print):
    folder=exp.folder
    for name in ('.lock','experiment.json','events.jsonl','capture.log'):
        confined(exp.output,'experiments/'+exp.job+'/'+name,must_exist=False)
    # Collision validation precedes lock/manifest creation.
    if folder.exists() and not (folder/'experiment.json').is_file():
        raise ValueError('Existing job folder without experiment manifest; no overwrite')
    if retry_only and not (folder/'experiment.json').is_file():raise ValueError('No experiment result to republish')
    if not retry_only:
        report=preflight(exp,verify_write=True)
        emit(json.dumps({'event':'preflight',**report},ensure_ascii=False))
        if report['status']=='FAIL':return 1
    else:
        if exp.publisher.enabled and not exp.raw.get('publish',{}).get('results_repository'):
            raise ValueError('Explicit results-only repository declaration required')
    folder.mkdir(parents=True,exist_ok=True)
    with Lock(folder/'.lock'):
        manifest_path=folder/'experiment.json'
        state=json.loads(manifest_path.read_text(encoding='utf8')) if manifest_path.exists() else {
            'schema_version':1,'fingerprint':exp.fingerprint,'job_id':exp.job,'status':'RUNNING',
            'started_at':iso(utc_now()),'stages':[]}
        if state['fingerprint']!=exp.fingerprint:raise ValueError('Job fingerprint mismatch; use a new job_id')
        if exp.live.coverage_mask is not None:
            from calibration.occlusion import mask_metadata
            state['coverage_mask'] = mask_metadata(exp.live.coverage_mask)
        if state.get('finished_at'):state['previous_finished_at']=state.pop('finished_at')
        state.update(status='RUNNING',failure=None,last_started_at=iso(utc_now()))
        if not retry_only:state['preflight']=report
        state['camera']=exp.identity.to_dict();state['input_source']=exp.source
        def save():write_json(manifest_path,state)
        def event(kind,**fields):
            data={'at':iso(utc_now()),'event':kind,**fields}
            with (folder/'events.jsonl').open('a',encoding='utf8') as f:
                f.write(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
            emit(json.dumps(data,ensure_ascii=False))
        @contextmanager
        def stage(name):
            record={'name':name,'started_at':iso(utc_now()),'status':'RUNNING'}
            state['stages'].append(record);state['active_stage']=name;save();start=monotonic()
            event('stage_started',stage=name)
            try:
                yield
            except BaseException as exc:
                record.update(status='INTERRUPTED' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'FAILED',error=redact(str(exc)))
                raise
            else:record['status']='COMPLETED'
            finally:
                record['seconds']=monotonic()-start;record['finished_at']=iso(utc_now());save()
                event('stage_finished',stage=name,status=record['status'],seconds=record['seconds'])
        save()
        try:
            if reuse_session and not state.get('calibration_session'):
                with stage('adopt_existing_result'):
                    from automation.publish.bundle import resolve_session
                    from calibration.project_io import load_project
                    from automation.profile import build_cli_options,options_to_argv,parse_cli_argv
                    session=resolve_session(reuse_session)
                    result=json.loads((session/'run_manifest.json').read_text(encoding='utf8'))
                    if result.get('status')!='completed' or any(result.get('camera',{}).get(k)!=v
                        for k,v in exp.identity.to_dict().items() if k!='source'):
                        raise ValueError('Existing result is incomplete or camera identity mismatches')
                    if exp.source['kind']!='images':raise ValueError('Historical adoption requires exact original image inputs; bag provenance cannot be inferred')
                    recorded=sorted(x['sha256'] for x in result['inputs']['fingerprint']['files'])
                    from automation.live.sources import ImageReplay
                    actual=sorted(sha256_file(p) for p in ImageReplay(exp.source['path']).paths)
                    if not recorded or recorded!=actual:raise ValueError('Existing result input image contents mismatch')
                    project,_=load_project(str(session/'project/calibration.ccproj'))
                    options=build_cli_options(exp.live.p1,image_paths=['validate'],sensor_name=exp.identity.label,
                        save_project='unused',json_summary='unused',log_file='unused')
                    _,pattern=parse_cli_argv(options_to_argv(options))
                    if asdict(pattern)!=asdict(project.pattern_config):raise ValueError('Existing result physical pattern differs from profile')
                    acceptance_path=session/'automation/p2_acceptance.json'
                    acceptance=json.loads((acceptance_path if acceptance_path.exists() else session/'automation/acceptance.json').read_text(encoding='utf8'))
                    verdict=acceptance.get('verdict')
                    if verdict not in {'PASS','WARN','FAIL'}:raise ValueError('Existing result verdict unavailable')
                    state.update(calibration_session=str(session),calibration_verdict=verdict,
                        result_manifest_sha256=sha256_file(session/'run_manifest.json'),input_sha256=fingerprint_inputs(exp.source),
                        reuse_policy='Explicit historical result; current calibration/acceptance settings are not reapplied')
                    save()
            elif reuse_session and Path(reuse_session).resolve()!=Path(state['calibration_session']).resolve():
                raise ValueError('Job already binds another completed Session')
            if not retry_only and not state.get('calibration_session'):
                with stage('input_fingerprint'):
                    current=fingerprint_inputs(exp.source)
                    if state.get('input_sha256') and state['input_sha256']!=current:
                        raise ValueError('Input content changed; new job_id required')
                    state['input_sha256']=current
                with stage('capture_readiness_calibration_export'):
                    from automation.live.controller import LiveController,State
                    from automation.live.sources import ImageReplay,BagReplay
                    from automation.live.cli import run
                    binding={k:v for k,v in exp.source.items() if k!='replay_interval'}
                    captures=folder/'capture_runs'
                    resume=state.get('capture_session')
                    if not resume and captures.exists():
                        candidates=list(captures.glob('**/p2_manifest.json'))
                        if len(candidates)!=1:raise ValueError('Interrupted capture initialization requires manual review')
                        resume=str(candidates[0].parent)
                    controller=LiveController(exp.live,exp.identity,binding,output_root=captures,resume=resume)
                    state['capture_session']=str(controller.session);save()
                    def log_snapshot(line):
                        with (folder/'capture.log').open('a',encoding='utf8') as f:f.write(line+'\n')
                    try:
                        source=(ImageReplay(binding['path']) if binding['kind']=='images' else BagReplay(binding['path'],binding['topic'],extract_fps=binding.get('extract_fps')))
                        code=run(controller,source,emit=log_snapshot,replay_interval=exp.source['replay_interval'])
                    finally:controller.close()
                    if controller.state!=State.COMPLETED or not controller.attempts:
                        if code==130:raise KeyboardInterrupt('Capture cancelled; rerun the same job to resume')
                        raise RuntimeError('; '.join(controller.guidance) or 'Capture did not complete')
                    outcome=controller.attempts[-1]['outcome']
                    state['calibration_session']=outcome['session_dir'];state['calibration_outcome']=outcome
                    # Preserve effective P2 verdict separately from pipeline execution.
                    state['calibration_verdict']=outcome['verdict']
                    state['result_manifest_sha256']=sha256_file(Path(outcome['session_dir'])/'run_manifest.json')
                    state['p2_task_timing']=[]
                    for response in controller.session.glob('tasks/*/response.json'):
                        data=json.loads(response.read_text(encoding='utf8'))
                        request=response.with_name('request.json')
                        kind=json.loads(request.read_text(encoding='utf8')).get('kind') if request.exists() else 'unknown'
                        state['p2_task_timing'].append({'task':response.parent.name,'kind':kind,'seconds':data.get('seconds'),'error':data.get('error')})
                    save()
            elif not retry_only:
                with stage('input_fingerprint'):
                    if fingerprint_inputs(exp.source)!=state['input_sha256']:
                        raise ValueError('Input content changed; completed result preserved, use a new job_id')
                event('calibration_reused',verdict=state.get('calibration_verdict'))
            if not state.get('calibration_session'):raise ValueError('No completed calibration result to publish')
            session=Path(state['calibration_session'])
            state['p1_stage_timing']=json.loads((session/'run_manifest.json').read_text(encoding='utf8')).get('stages',[])
            from datetime import datetime
            for timing in state['p1_stage_timing']:
                if timing.get('started_at') and timing.get('finished_at'):
                    timing['seconds']=(datetime.fromisoformat(timing['finished_at'])-datetime.fromisoformat(timing['started_at'])).total_seconds()
            if sha256_file(session/'run_manifest.json')!=state['result_manifest_sha256']:
                raise ValueError('Saved calibration manifest changed; manual review required')
            if not retry_only and exp.factory.get('reference') and Path(exp.factory['reference']).is_file():
                state.pop('factory_skip',None)
                factory_hash=canonical_sha256({k:(sha256_file(v) if k in {'reference','coordinates'} and Path(v).is_file() else v)
                    for k,v in exp.factory.items()})
                if state.get('factory_fingerprint')!=factory_hash:
                    with stage('factory_comparison'):
                        from automation.factory.service import post_compare
                        args=SimpleNamespace(factory=exp.factory.get('reference'),factory_coordinates=exp.factory.get('coordinates'),factory_seeded=exp.factory.get('seeded',False))
                        state['factory']=post_compare(session,args);state['factory_fingerprint']=factory_hash
                        state.pop('publication',None)  # new comparison must be packaged, never silently reuse an old bundle
                        save()
            elif not retry_only:
                if state.get('factory'):
                    state.pop('factory',None);state.pop('factory_fingerprint',None);state.pop('publication',None)
                state['factory_skip']={'status':'SKIPPED','reason':'NO_FACTORY_REFERENCE' if not exp.factory.get('reference') else 'FACTORY_FILE_NOT_FOUND','seeded_executed':False}
                save();event('factory_skipped',**state['factory_skip'])
            with stage('publication'):
                from automation.publish.service import outbox_for
                from automation.publish.git_publisher import GitPublisher
                box=outbox_for(exp.publisher)
                if state.get('publication',{}).get('job_id'):
                    job=state['publication']['job_id']
                else:
                    options={'full_frame':exp.raw.get('publish',{}).get('full_frame',False)}
                    if (state.get('factory') or {}).get('report'):options['factory_report']=state['factory']['report']
                    pending=box.enqueue(session,**options);job=pending['job_id']
                    state['publication']={'job_id':job,'outbox':str(box.root),'status':pending['status']};save()
                if Path(state['publication']['outbox']).resolve()!=box.root:
                    raise ValueError('Existing job is bound to another outbox')
                publication=box.process(job,GitPublisher(exp.publisher),reset_failed=reset_failed)
                state['publication']={**publication,'outbox':str(box.root)};save()
                # Opt-in orchestration only: retain Outbox's target, bounded
                # attempts, permanent-error policy and exponential backoff.
                while exp.raw.get('publish',{}).get('retry_in_process',False) and publication['status']=='RETRY_WAIT':
                    delay=max(0.,(publication.get('next_attempt_at') or 0)-time())
                    event('publication_retry_wait',job_id=job,attempts=publication['attempts'],seconds=delay)
                    sleep(delay)
                    publication=box.process(job,GitPublisher(exp.publisher))
                    state['publication']={**publication,'outbox':str(box.root)};save()
                if publication['status'] in {'FAILED','RETRY_WAIT'}:
                    raise RuntimeError('Publication '+str(publication.get('error'))+'; calibration preserved; use retry-publish')
            state.update(status='COMPLETED',active_stage=None,finished_at=iso(utc_now()),failure=None);save()
            event('completed',verdict=state['calibration_verdict'],publication=state['publication']['status'])
            return {'PASS':0,'WARN':3,'FAIL':4}.get(state['calibration_verdict'],3)
        except BaseException as exc:
            state.update(status='INTERRUPTED' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'FAILED',
                failure={'stage':state.get('active_stage'),'reason':redact(str(exc))},finished_at=iso(utc_now()))
            save();event('failed',**state['failure'])
            if isinstance(exc,(KeyboardInterrupt,SystemExit)):return 130
            return 2

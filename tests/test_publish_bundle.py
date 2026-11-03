import json
from pathlib import Path
import pytest
from automation.publish.bundle import build_bundle,verify_bundle
from automation.publish.base import PublishError
from automation.publish.config import load_config,validate_remote
from tests._publish_test_utils import make_session,write,read,snapshot


def test_allowlisted_bundle_preserves_source_and_numeric_exports(tmp_path):
    s=make_session(tmp_path/'session',p2=True);before=snapshot(s)
    (s/'private-cache.bin').write_bytes(b'not for git')
    before=snapshot(s)
    m=build_bundle(s,tmp_path/'bundle',full_frame=True)
    assert verify_bundle(tmp_path/'bundle')==m and snapshot(s)==before
    assert m['summary']['acceptance_verdict']=='WARN'
    assert m['summary']['approval_status']=='UNREVIEWED'
    assert not (tmp_path/'bundle/private-cache.bin').exists()
    assert 'script' not in (tmp_path/'bundle/report.html').read_text(encoding='utf-8')
    from calibration.calibration_io import load_standard_calibration
    import numpy as np
    original=load_standard_calibration(str(s/'intrinsic/camera_brown_conrady.yaml'))
    staged=load_standard_calibration(str(tmp_path/'bundle/intrinsic/camera_brown_conrady.yaml'))
    np.testing.assert_array_equal(original.camera_matrix,staged.camera_matrix)
    np.testing.assert_array_equal(original.distortion,staged.distortion)


@pytest.mark.parametrize('missing',['run_manifest.json','reports/report.html','reports/calibration.json','automation/acceptance.json','intrinsic/camera_info.yaml','intrinsic/camera_brown_conrady.yaml'])
def test_required_file_missing_is_not_success(tmp_path,missing):
    s=make_session(tmp_path/'session');(s/missing).unlink()
    with pytest.raises(PublishError) as error:build_bundle(s,tmp_path/'bundle')
    assert error.value.details.get('missing')


@pytest.mark.parametrize('path',['../secret.json','C:/Users/private.json','/tmp/private.json','intrinsic/../../secret.json','intrinsic\\secret.json'])
def test_manifest_output_traversal_rejected(tmp_path,path):
    s=make_session(tmp_path/'session');m=read(s/'run_manifest.json');m['outputs']=[{'path':path}];write(s/'run_manifest.json',m)
    with pytest.raises(PublishError):build_bundle(s,tmp_path/'bundle')


def test_sensitive_paths_tokens_and_html_assets_do_not_leave_session(tmp_path,monkeypatch):
    token='ghp_'+'z'*32;monkeypatch.setenv('TEST_PUBLISH_TOKEN',token)
    s=make_session(tmp_path/'session');m=read(s/'run_manifest.json')
    m['results']['recommendation']['selection_reasons']=[f'private C:\\Users\\Alice\\secret.json {token}', '/home/alice/private/file.txt']
    m['results']['token']=token;write(s/'run_manifest.json',m)
    build_bundle(s,tmp_path/'bundle')
    texts=''.join(p.read_text(encoding='utf-8') for p in (tmp_path/'bundle').rglob('*') if p.is_file())
    assert token not in texts and 'Alice' not in texts and '/home/alice' not in texts


def test_content_hash_ignores_run_time_but_not_results_camera_or_verdict(tmp_path):
    def bundle(name,**kwargs):
        session=make_session(tmp_path/name,**kwargs)
        if name=='b':
            manifest=read(session/'run_manifest.json')
            manifest.update(started_at='2030-01-01T00:00:00+00:00',finished_at='2030-01-01T00:01:00+00:00')
            write(session/'run_manifest.json',manifest)
            calibration=read(session/'reports/calibration.json');calibration['generated_at']='2030-01-01'
            write(session/'reports/calibration.json',calibration)
        return build_bundle(session,tmp_path/(name+'-bundle'))['content_hash']
    a=bundle('a',run_id='RUN-1');b=bundle('b',run_id='RUN-2')
    assert a==b
    assert bundle('c',delta=.01)!=a
    assert bundle('d',serial='SN456')!=a
    assert bundle('e',verdict='FAIL')!=a


@pytest.mark.parametrize('serial',['UNKNOWN','NONE','UNSET',''])
def test_unconfirmed_serial_is_not_invented(tmp_path,serial):
    s=make_session(tmp_path/'s',serial=serial)
    with pytest.raises(ValueError):build_bundle(s,tmp_path/'bundle')


def test_large_required_files_fail_and_large_unrelated_cache_is_ignored(tmp_path):
    s=make_session(tmp_path/'s');(s/'cache.bin').write_bytes(b'x'*(11*1024*1024))
    build_bundle(s,tmp_path/'good')
    (s/'reports/report.html').write_bytes(b'x'*(11*1024*1024))
    with pytest.raises(PublishError,match='FILE_TOO_LARGE'):build_bundle(s,tmp_path/'bad')


def test_tampered_or_extra_bundle_files_rejected(tmp_path):
    build_bundle(make_session(tmp_path/'s'),tmp_path/'b')
    (tmp_path/'b/calibration.json').write_text('{}',encoding='utf-8')
    with pytest.raises(PublishError,match='HASH'):verify_bundle(tmp_path/'b')


@pytest.mark.parametrize('remote',['https://token@github.com/org/results.git','https://github.com/org/results?token=x','git@github.com:yaaaaangeo/camera_calibrator.git','https://github.com/yaaaaangeo/camera_calibrator','-upload-pack=bad'])
def test_unsafe_or_source_remotes_rejected(remote):
    with pytest.raises(PublishError):validate_remote(remote)


def test_disabled_defaults_and_secret_config_keys_rejected(tmp_path):
    assert not load_config().enabled
    p=tmp_path/'cfg.json';write(p,{'enabled':True,'remote':'https://github.com/org/results.git','token':'secret'})
    with pytest.raises(PublishError):load_config(p)


def test_execution_error_is_not_normal_bundle(tmp_path):
    s=make_session(tmp_path/'s');m=read(s/'run_manifest.json');m['status']='failed';write(s/'run_manifest.json',m)
    with pytest.raises(PublishError,match='NOT_COMPLETED'):build_bundle(s,tmp_path/'b')


def test_p2_verdict_overrides_p1_without_changing_either_source(tmp_path):
    s=make_session(tmp_path/'s',verdict='PASS',p2=True);p2=read(s/'automation/p2_acceptance.json')
    p2['verdict']='WARN';p2['acceptance']['verdict']='WARN';write(s/'automation/p2_acceptance.json',p2)
    before=snapshot(s);m=build_bundle(s,tmp_path/'b')
    assert m['summary']['acceptance_verdict']=='WARN'
    assert read(tmp_path/'b/p1_acceptance.json')['verdict']=='PASS' and snapshot(s)==before


def test_case_sensitive_camera_identity_has_distinct_safe_paths(tmp_path):
    a=build_bundle(make_session(tmp_path/'a',serial='SN123'),tmp_path/'ab')
    b=build_bundle(make_session(tmp_path/'b',serial='sn123'),tmp_path/'bb')
    assert a['run_path'].lower()!=b['run_path'].lower()


def test_inconsistent_yaml_and_json_are_not_published(tmp_path):
    s=make_session(tmp_path/'s');m=read(s/'run_manifest.json')
    m['results']['models']['brown_conrady']['camera_matrix'][0][0]+=1;write(s/'run_manifest.json',m)
    with pytest.raises(PublishError,match='PARAMETER_MISMATCH'):build_bundle(s,tmp_path/'b')


def test_unexpected_directory_cannot_be_copied_by_publisher(tmp_path):
    build_bundle(make_session(tmp_path/'s'),tmp_path/'b')
    (tmp_path/'b/private-cache').mkdir()
    with pytest.raises(PublishError,match='UNEXPECTED_BUNDLE_DIRECTORY'):verify_bundle(tmp_path/'b')


def test_link_guard_runs_before_intrinsic_directory_traversal(tmp_path,monkeypatch):
    build_bundle(make_session(tmp_path/'s'),tmp_path/'b')
    original=Path.is_symlink;target=tmp_path/'b/intrinsic'
    monkeypatch.setattr(Path,'is_symlink',lambda p:p==target or original(p))
    with pytest.raises(PublishError,match='UNSAFE_LINK'):verify_bundle(tmp_path/'b')

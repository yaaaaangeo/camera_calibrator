from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
from calibration.types import CalibrationResult,CameraModelType,CameraConfig,PatternConfig,PatternType
from export.opencv import export_opencv_yaml
from export.ros import export_ros_camera_info


def make_session(root,*,run_id='RUN-1',serial='SN123',camera_model='CAM',verdict='WARN',delta=0.,p2=False):
    root=Path(root);root.mkdir(parents=True)
    (root/'reports').mkdir();(root/'automation').mkdir();(root/'intrinsic').mkdir()
    K=np.array([[800.+delta,0,320.],[0,805.,240.],[0,0,1.]])
    D=np.array([-.1,.02,0,0,0.]).reshape(-1,1)
    result=CalibrationResult(CameraModelType.BROWN_CONRADY,camera_matrix=K,distortion=D,success=True,rms_error=.3)
    camera=CameraConfig(640,480,sensor_name=camera_model+'_'+serial)
    export_opencv_yaml(result,camera,PatternConfig(PatternType.CHARUCO,7,5,.04,marker_size=.03),str(root/'intrinsic/camera_brown_conrady.yaml'))
    export_ros_camera_info(result,camera,str(root/'intrinsic/camera_info.yaml'))
    inputs=[{'path':str(root/'private'/'photo.png'),'name':'photo.png','sha256':hashlib.sha256(b'image').hexdigest()}]
    model={'success':True,'camera_matrix':K.tolist(),'distortion':D.ravel().tolist(),
           'train_rms_px':.3,'holdout':{'test_rms_px':.4,'edge_rms_px':.5,'test_p95_px':.6}}
    manifest={'schema_version':1,'run_id':run_id,'status':'completed',
        'started_at':'2026-11-02T01:00:00+00:00','finished_at':'2026-11-02T01:01:00+00:00',
        'camera':{'camera_model':camera_model,'serial_number':serial,'image_width':640,'image_height':480},
        'code_version':{'available':True,'commit_sha':'a'*40,'dirty':False},
        'inputs':{'fingerprint':{'aggregate_sha256':'b'*64,'count':1,'files':inputs}},
        'split':{'train_frame_ids':['photo'],'test_frame_ids':['other']},
        'results':{'models':{'brown_conrady':model},'recommendation':{
            'chosen_model':'brown_conrady','recommended_model':'brown_conrady','selection_reasons':['Hold-out evidence']}},
        'outputs':[]}
    acceptance={'run_id':run_id,'verdict':verdict,'criteria_complete':False,'reasons':['Review required']}
    write(root/'run_manifest.json',manifest)
    write(root/'reports/calibration.json',{'generated_at':'2026-11-02','chosen_model':'brown_conrady','models':{'brown_conrady':model}})
    write(root/'automation/acceptance.json',acceptance)
    write(root/'automation/run_summary.json',{'metrics':{'test_rms_px':.4,'stability_pct':90.}})
    (root/'reports/report.html').write_text('<html><script>bad()</script>private original</html>',encoding='utf-8')
    if p2:write(root/'automation/p2_acceptance.json',{'verdict':verdict,'acceptance':acceptance})
    return root


def write(path,data):Path(path).write_text(json.dumps(data),encoding='utf-8')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def git(*args,cwd=None):
    p=subprocess.run(['git','-c','user.name=Test','-c','user.email=test@localhost','-c','commit.gpgsign=false',*map(str,args)],cwd=cwd,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    return p.stdout.strip()
def bare(path):git('init','--bare',path);return Path(path).resolve()
def snapshot(root):return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(root).rglob('*') if p.is_file()}

"""Run two full agentic variants, then score, audit, and release the 122B queue."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from agentic_sandbox import atomic_json, digest

def run(config_path):
    config_path = Path(config_path).resolve()
    c = json.loads(config_path.read_text())
    root = Path(c['root']);repo=Path(c['repo']);logs=root/'logs';logs.mkdir(exist_ok=True)
    protocol = json.loads(Path(c['protocol']).read_text())
    assert protocol['frozen']
    actual_image = subprocess.check_output(['docker','image','inspect',
        'vllm/vllm-openai:v0.19.0','--format','{{.Id}}'],text=True).strip()
    assert actual_image == protocol['sandbox_image_id'], 'Sandbox image changed'
    assert json.loads((root/'sandbox-parity.json').read_text())['passed']
    assert json.loads((root/'sandbox-isolation.json').read_text())['passed']
    children = []
    def status(phase,**kwargs):
        atomic_json(root/'status.json',{'phase':phase,'updated_unix':time.time(),
            'protocol_sha256':digest(c['protocol']),**kwargs})
    try:
        status('starting_inference')
        for gpu,v in zip(c['inference_gpus'],protocol['variants']):
            name=v['id']
            if (root/'predictions'/name/'complete.json').exists():
                continue
            ipc=Path(c['ipc_root']+'-'+str(gpu));ipc.mkdir(parents=True,exist_ok=True)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),VLLM_WORKER_MULTIPROC_METHOD='spawn',
                     VLLM_RPC_BASE_PATH=str(ipc),TMPDIR=str(ipc),OMP_NUM_THREADS='1',
                     OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            log=(logs/(name+'.log')).open('a')
            proc=subprocess.Popen([c['inference_python'],'-u',str(repo/'scripts/infer_agentic.py'),
                '--config',str(config_path),'--variant',name],env=env,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,cwd=repo)
            children.append((name,proc))
            atomic_json(root/(name+'-process.json'),{'pid':proc.pid,'gpu':gpu,'started_unix':time.time()})
        while any(proc.poll() is None for _,proc in children):
            for name,proc in children:
                if proc.poll() not in (None,0):
                    raise RuntimeError(f'Inference failed: {name}, returncode={proc.returncode}')
            counts = {v['id']:len(list((root/'predictions'/v['id']).glob('*/generation.json'))) for v in protocol['variants']}
            status('agentic_inference',completed_tasks=counts,
                   children={name:proc.pid for name,proc in children if proc.poll() is None})
            time.sleep(15)
        for name,proc in children:
            assert proc.returncode == 0
        status('scoring')
        (root/'scores').mkdir(exist_ok=True)
        for v in protocol['variants']:
            name=v['id']
            command=[c['host_python'],str(repo/'scripts/score.py'),
                '--manifest',c['manifest'],'--protocol',c['protocol'],'--references',c['references'],
                '--predictions',str(root/'predictions'/name),'--renders',str(root/'renders'/name),
                '--clip',c['clip'],'--output',str(root/'scores'/(name+'.json')),'--split','all']
            with (logs/(name+'-score.log')).open('w') as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,cwd=repo)
        status('auditing_and_reporting')
        with (logs/'final-report.log').open('w') as log:
            subprocess.run([c['host_python'],str(repo/'scripts/report_agentic.py'),'--config',str(config_path)],
                           check=True,stdout=log,stderr=subprocess.STDOUT,cwd=repo)
        atomic_json(root/'postprocess-complete.json',{'completed_unix':time.time(),
                    'report_output':c['report_output'],'protocol_sha256':digest(c['protocol'])})
        status('complete')
        pid=c.get('qwen122_pipeline_pid')
        if pid:
            cmdline=Path(f'/proc/{pid}/cmdline')
            if cmdline.exists():
                text=cmdline.read_bytes().replace(b'\0',b' ').decode(errors='replace')
                if 'run_full_pipeline.py' in text and 'eval-qwen122-20261015/run-config.json' in text:
                    os.kill(pid,signal.SIGCONT)
                    atomic_json(root/'qwen122-resumed.json',{'pid':pid,'unix_time':time.time()})
    except BaseException as error:
        atomic_json(root/'error.json',{'error':repr(error),'traceback':traceback.format_exc(),'unix_time':time.time()})
        status('error',error=repr(error))
        # Stop our own inference sessions. Completed raw outputs remain available for exact resume.
        for _,proc in children:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    run(p.parse_args().config)

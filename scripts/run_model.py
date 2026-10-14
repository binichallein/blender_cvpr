"""Resume one model split, rendering completed generations while inference continues."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from render_one import render

def main():
    p=argparse.ArgumentParser()
    for key in ['model','manifest','protocol','output','blender','infinigen','clip','references']:
        p.add_argument('--'+key,required=True)
    p.add_argument('--inference-gpu',required=True)
    p.add_argument('--render-gpu',required=True)
    p.add_argument('--split',choices=['subset','all'],default='subset')
    a=p.parse_args()
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    scripts=Path(__file__).resolve().parent
    m=json.loads(Path(a.manifest).read_text())
    ids=m['subset_ids'] if a.split=='subset' else [t['id'] for t in m['all_tasks']]
    tasks=[t for t in m['all_tasks'] if t['id'] in ids]
    predictions=root/'predictions';renders=root/'renders'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.inference_gpu,TOKENIZERS_PARALLELISM='false',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8')
    command=[sys.executable,'-u',str(scripts/'infer.py'),'--model',a.model,
             '--manifest',a.manifest,'--protocol',a.protocol,'--output',str(predictions),'--split',a.split]
    handled=set()
    with (root/(a.split+'-inference.log')).open('a') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        while True:
            for task in tasks:
                if task['id'] in handled: continue
                generation=predictions/task['id']/'generation.json'
                if not generation.exists(): continue
                row=json.loads(generation.read_text())
                if row['ok']:
                    r=render(task,predictions/task['id']/'prediction.py',renders/task['id']/'prediction',
                             a.blender,a.infinigen,a.render_gpu)
                    print(json.dumps({'task':task['id'],'render_ok':r['ok'],'seconds':r.get('seconds')}),flush=True)
                handled.add(task['id'])
            status=proc.poll()
            if status is not None:
                if status: raise RuntimeError(f'Inference exited {status}; see inference log, resume after fixing infrastructure')
                if len(handled)==len(tasks): break
                raise RuntimeError(f'Incomplete inference: {len(handled)}/{len(tasks)}')
            time.sleep(2)
    reference_deadline=time.monotonic()+3600
    while not all((Path(a.references)/t['id']/k/'render.json').exists() for t in tasks for k in ['start','goal']):
        if (Path(a.references)/'batch.json').exists() or time.monotonic()>reference_deadline:
            raise RuntimeError('Reference rendering incomplete or failed; inspect reference logs')
        time.sleep(5)
    command=[sys.executable,'-u',str(scripts/'score.py')]
    for key,value in {'manifest':a.manifest,'protocol':a.protocol,'references':a.references,
                      'predictions':str(predictions),'renders':str(renders),'clip':a.clip,
                      'output':str(root/(a.split+'-scores.json')),'split':a.split}.items():
        command.extend(['--'+key,value])
    subprocess.run(command,check=True)
if __name__=='__main__': main()

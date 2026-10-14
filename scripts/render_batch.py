"""Render references/controls or predictions with one worker per specified GPU."""
import argparse
import concurrent.futures
import json
import queue
from pathlib import Path
from render_one import render
from calibration_controls import create

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--blender',required=True)
    p.add_argument('--infinigen',required=True)
    p.add_argument('--gpus',default='2,3')
    p.add_argument('--mode',choices=['calibration','references','predictions'],required=True)
    p.add_argument('--split',choices=['subset','all'],default='subset')
    p.add_argument('--predictions')
    p.add_argument('--samples',type=int,default=512)
    a=p.parse_args()
    m=json.loads(Path(a.manifest).read_text());out=Path(a.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    ids=m['calibration_ids'] if a.mode=='calibration' else m['subset_ids'] if a.split=='subset' else [t['id'] for t in m['all_tasks']]
    if a.mode=='calibration': create(m,out/'controls')
    jobs=queue.Queue()
    for t in m['all_tasks']:
        if t['id'] not in ids: continue
        if a.mode=='predictions':
            script=Path(a.predictions)/t['id']/'prediction.py'
            if script.exists(): jobs.put((t,'prediction',script,20261015))
        else:
            for kind in ['start','goal']:
                jobs.put((t,kind,Path(t['directory'])/(kind+'.py'),20261015))
            if a.mode=='calibration':
                for kind in ['near','half']:
                    jobs.put((t,kind,out/'controls'/t['id']/(kind+'.py'),20261015))
                jobs.put((t,'repeat',Path(t['directory'])/'goal.py',20261015))
                jobs.put((t,'alternate_seed',Path(t['directory'])/'goal.py',20261016))
    results=[]
    def worker(gpu):
        while True:
            try: t,kind,script,seed=jobs.get_nowait()
            except queue.Empty: break
            dest=out/t['id']/kind
            try:
                r=render(t,script,dest,a.blender,a.infinigen,gpu,a.samples,seed)
            except Exception as e:
                r={'ok':False,'error':repr(e)}
            row={'task':t['id'],'kind':kind,**r}
            results.append(row)
            print(json.dumps({k:row.get(k) for k in ['task','kind','ok','seconds','error']}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.gpus.split(','))) as ex:
        list(ex.map(worker,a.gpus.split(',')))
    (out/'batch.json').write_text(json.dumps(results,indent=2))
    assert all(r['ok'] for r in results) or a.mode=='predictions', 'Reference failures; inspect Blender logs'
if __name__=='__main__': main()

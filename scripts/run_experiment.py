"""Run both 50-task screens, then only qualifying models on all released tasks."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from validate_screen import validate

def main():
    p=argparse.ArgumentParser()
    for key in ['model8','model32','manifest','protocol','output','blender','infinigen','clip','references']:
        p.add_argument('--'+key,required=True)
    p.add_argument('--backend',choices=['transformers','vllm'],default='vllm')
    p.add_argument('--inference-python',default=sys.executable)
    a=p.parse_args()
    for key,value in vars(a).items():
        if key!='backend': setattr(a,key,str(Path(value).resolve()))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    scripts=Path(__file__).resolve().parent
    (out/'source.json').write_text(json.dumps({'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=scripts.parent,text=True).strip(),
        'git_status':subprocess.check_output(['git','status','--porcelain'],cwd=scripts.parent,text=True)},indent=2))
    preflight=validate(a.manifest,a.protocol,a.references)
    (out/'preflight.json').write_text(json.dumps(preflight,indent=2))
    models=[('qwen3-vl-8b',a.model8,'2','0'),('qwen3-vl-32b',a.model32,'3','1')]
    def run_models(selected,split):
        children=[]
        for name,model,inference_gpu,render_gpu in selected:
            command=[sys.executable,'-u',str(scripts/'run_model.py')]
            options={'model':model,'manifest':a.manifest,'protocol':a.protocol,'output':str(out/name),
                     'blender':a.blender,'infinigen':a.infinigen,'clip':a.clip,'references':a.references,
                     'inference-gpu':inference_gpu,'render-gpu':render_gpu,'split':split,
                     'backend':a.backend,'inference-python':a.inference_python}
            for k,v in options.items(): command.extend(['--'+k,v])
            with (out/(name+'-'+split+'.log')).open('a') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            children.append((name,child))
            print('STARTED',name,split,child.pid,flush=True)
        outcomes=[(name,child.wait()) for name,child in children]
        assert all(rc==0 for name,rc in outcomes),outcomes
    run_models(models,'subset')
    qualified=[]
    for item in models:
        score=json.loads((out/item[0]/'subset-scores.json').read_text())
        print('SCREEN_RESULT',item[0],json.dumps(score['summary']),flush=True)
        if score['expand_to_full']: qualified.append(item)
    if qualified:
        command=[sys.executable,'-u',str(scripts/'render_batch.py'),'--manifest',a.manifest,
                 '--output',a.references,'--blender',a.blender,'--infinigen',a.infinigen,
                 '--mode','references','--split','all','--gpus','0,1,2,3']
        with (out/'full-references.log').open('a') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        run_models(qualified,'all')
    result={}
    for name,_,_,_ in models:
        result[name]={'subset':json.loads((out/name/'subset-scores.json').read_text())}
        full=out/name/'all-scores.json'
        if full.exists(): result[name]['full']=json.loads(full.read_text())
    (out/'complete.json').write_text(json.dumps(result,indent=2))
    print('EXPERIMENT_COMPLETE',flush=True)
if __name__=='__main__': main()

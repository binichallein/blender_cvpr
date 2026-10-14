"""Fit a model-independent visual-success rule on separate rendered controls."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime,timezone
from metrics import Metrics, relative_error

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--renders',required=True)
    p.add_argument('--clip',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    out=Path(a.output)
    if out.exists(): raise ValueError('Protocol already frozen; do not overwrite')
    m=json.loads(Path(a.manifest).read_text());root=Path(a.renders)
    metric=Metrics(a.clip)
    rows=[]
    for task in m['all_tasks']:
        if task['id'] not in m['calibration_ids']: continue
        d=root/task['id'];scores={}
        for kind in ['start','goal','repeat','near','half','alternate_seed']:
            assert json.loads((d/kind/'render.json').read_text())['ok'], (task['id'],kind)
        for kind in ['start','repeat','near','half','alternate_seed']:
            scores[kind]=metric.compare(d/kind,d/'goal',task['views'])
        try:
            scores['release_goal']=metric.compare(Path(task['directory'])/'renders/goal',d/'goal',task['views'])
        except ValueError as e:
            scores['release_goal']={'diagnostic':str(e)}
        rows.append({'task':task['id'],'category':task['category'],'scores':scores})
    categories={}
    for cat in sorted({r['category'] for r in rows}):
        group=[r for r in rows if r['category']==cat]
        floor={k:max(r['scores']['repeat'][k] for r in group) for k in ['pl','nclip']}
        positives=[];negatives=[]
        for r in group:
            assert r['scores']['start']['pl']>floor['pl']*10, 'Task change below render-noise resolution'
            r['relative']={kind:relative_error(r['scores'][kind],r['scores']['start'],floor) for kind in ['repeat','near','half','start']}
            positives.extend(r['relative'][k]['pl'] for k in ['repeat','near'])
            negatives.extend(r['relative'][k]['pl'] for k in ['half','start'])
        pos=max(positives);neg=min(negatives)
        assert pos<neg, f'Controls overlap in {cat}: near={pos}, wrong={neg}'
        threshold=math.sqrt(pos*neg) if pos>0 else neg/2
        categories[cat]={'noise_floor':floor,'rule':{'pl':threshold},
                         'max_positive_relative_pl':pos,'min_negative_relative_pl':neg}
    result={'frozen':True,'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
            'name':'calibrated_relative_photometric_success_v1',
            'official_binary_score':False,
            'definition':'PL residual above the category noise floor, divided by start residual, <= category threshold. N-CLIP is reported separately.',
            'threshold_selection':'Geometric midpoint between largest positive and smallest negative relative PL; fail closed if controls overlap.',
            'calibration_limitations':'Only two tasks per category; near/half parameter edits are operational controls, not human correctness labels. These tasks are excluded from subset selection but remain part of a conditional full run.',
            'category_rules':categories,'calibration_ids':m['calibration_ids'],
            'archive_sha256':m['archive_sha256'],'controls':rows,
            'render_settings':{'samples':512,'seed':20261015,'resolution':[512,512],'engine':'CYCLES','device':'CUDA','timeout_seconds':600},
            'noise_policy':'Same-seed rerenders define the floor; alternate-seed renders are diagnostics only because formal evaluation fixes seed 20261015.',
            'input_views':'all matched released current/target image pairs','max_new_tokens':8192,
            'generation':'greedy BF16 one-shot; no retries or repairs'}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps({'protocol_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'categories':categories},indent=2))
if __name__=='__main__': main()

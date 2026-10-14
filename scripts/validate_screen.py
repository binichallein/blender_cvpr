"""Check complete, non-degenerate reference coverage before model inference."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

def validate(manifest,protocol,references):
    m=json.loads(Path(manifest).read_text());p=json.loads(Path(protocol).read_text())
    ids=m['subset_ids'];assert len(ids)==len(set(ids))==50
    tasks=[t for t in m['all_tasks'] if t['id'] in ids]
    counts=collections.Counter(t['category'] for t in tasks)
    assert len(counts)==5 and set(counts.values())=={10}
    assert not set(ids)&set(m.get('invalid_task_ids',[]))
    rows=[]
    for t in tasks:
        root=Path(references)/t['id']
        for kind in ['start','goal']:
            meta=json.loads((root/kind/'render.json').read_text())
            assert meta['ok'] and meta['samples']==512 and meta['seed']==20261015
            assert meta['script_sha256']==hashlib.sha256((Path(t['directory'])/(kind+'.py')).read_bytes()).hexdigest()
        per_view=[]
        for view in t['views']:
            x=np.asarray(Image.open(root/'start'/view).convert('RGB'),dtype=np.float32)/255
            y=np.asarray(Image.open(root/'goal'/view).convert('RGB'),dtype=np.float32)/255
            assert x.shape==y.shape==(512,512,3)
            per_view.append(float(np.mean((x-y)**2)))
        value=float(np.mean(per_view));floor=p['category_rules'][t['category']]['noise_floor']['pl']
        assert value>floor+1e-8, f'Unresolvable reference change: {t["id"]} PL={value}, floor={floor}'
        rows.append({'task':t['id'],'start_pl':value,'noise_floor':floor})
    return {'ok':True,'category_counts':dict(counts),'rows':rows,
            'manifest_sha256':hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
            'protocol_sha256':hashlib.sha256(Path(protocol).read_bytes()).hexdigest()}

def main():
    p=argparse.ArgumentParser()
    for k in ['manifest','protocol','references','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();result=validate(a.manifest,a.protocol,a.references)
    Path(a.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({'ok':result['ok'],'counts':result['category_counts']}),flush=True)
if __name__=='__main__':main()

"""Complete-coverage scoring. Execution/parse failures remain in the denominator."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from metrics import Metrics,relative_error,passes,should_expand

def summarize(rows):
    nominal=len(rows)
    invalid=[{'task':r['task'],'reason':r['invalid_data']} for r in rows if r.get('invalid_data')]
    rows=[r for r in rows if not r.get('invalid_data')]
    valid=[r for r in rows if r.get('metrics')]
    result={'tasks':nominal,'scorable_tasks':len(rows),'invalid_data':invalid,'successes':sum(r['success'] for r in rows),
            'success_rate':sum(r['success'] for r in rows)/len(rows) if rows else None,
            'rendered_tasks':len(valid),'failures':dict(collections.Counter(r.get('failure') for r in rows if r.get('failure')))}
    result['valid_only_mean']={k:float(np.mean([r['metrics'][k] for r in valid])) if valid else None for k in ['pl','nclip']}
    result['start_mean']={k:float(np.mean([r['start_metrics'][k] for r in rows])) if rows else None for k in ['pl','nclip']}
    return result

def main():
    p=argparse.ArgumentParser()
    for name in ['manifest','protocol','references','predictions','renders','clip','output']:
        p.add_argument('--'+name,required=True)
    p.add_argument('--split',choices=['subset','all'],default='subset')
    a=p.parse_args()
    m=json.loads(Path(a.manifest).read_text());protocol=json.loads(Path(a.protocol).read_text())
    assert protocol['frozen']
    protocol_sha=hashlib.sha256(Path(a.protocol).read_bytes()).hexdigest()
    ids=m['subset_ids'] if a.split=='subset' else [t['id'] for t in m['all_tasks']]
    tasks=[t for t in m['all_tasks'] if t['id'] in ids]
    root=Path(a.predictions);renders=Path(a.renders);refs=Path(a.references)
    assert all((root/t['id']/'generation.json').exists() for t in tasks),'Incomplete inference'
    metric=Metrics(a.clip);rows=[]
    for t in tasks:
        ref=refs/t['id'];pred=renders/t['id']/'prediction'
        for kind in ['start','goal']:
            assert json.loads((ref/kind/'render.json').read_text())['ok'], (t['id'],kind)
        start=metric.compare(ref/'start',ref/'goal',t['views'])
        generation=json.loads((root/t['id']/'generation.json').read_text())
        assert generation['protocol_sha256']==protocol_sha,'Protocol mismatch'
        row={'task':t['id'],'category':t['category'],'success':False,'start_metrics':start,
             'generation_seconds':generation['seconds'],'input_tokens':generation.get('input_tokens'),
             'output_tokens':generation.get('output_tokens')}
        rule=protocol['category_rules'][t['category']]
        if t['id'] in m.get('invalid_task_ids',[]):
            row['invalid_data']='identical_start_goal_program'
            rows.append(row)
            continue
        if start['pl']<=rule['noise_floor']['pl']+1e-8:
            row['invalid_data']='reference_change_below_noise_floor'
            rows.append(row)
            continue
        if not generation['ok']:
            row['failure']='generation_or_parse';row['detail']=generation.get('error')
        elif not (pred/'render.json').exists():
            row['failure']='execution'
        elif not all((pred/v).exists() for v in t['views']):
            row['failure']='missing_view'
        else:
            row['metrics']=metric.compare(pred,ref/'goal',t['views'])
            rule=protocol['category_rules'][t['category']]
            row['relative_error']=relative_error(row['metrics'],start,rule['noise_floor'])
            row['success']=start['pl']>rule['noise_floor']['pl']+1e-8 and passes(row['relative_error'],rule['rule'])
            if start['pl']<=rule['noise_floor']['pl']+1e-8:
                row['failure']='below_calibrated_noise_floor'
            row['render_seconds']=json.loads((pred/'render.json').read_text())['seconds']
        rows.append(row)
    summary=summarize(rows)
    result={'split':a.split,'protocol_sha256':protocol_sha,'summary':summary,
            'by_category':{cat:summarize([r for r in rows if r['category']==cat]) for cat in sorted({r['category'] for r in rows})},
            'rows':rows}
    if a.split=='subset': result['expand_to_full']=should_expand(summary['successes'],len(rows))
    else:
        result['screening_subset']=summarize([r for r in rows if r['task'] in m['subset_ids']])
        result['remaining_tasks']=summarize([r for r in rows if r['task'] not in m['subset_ids']])
        result['excluding_calibration']=summarize([r for r in rows if r['task'] not in m['calibration_ids']])
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
if __name__=='__main__': main()

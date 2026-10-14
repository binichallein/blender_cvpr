"""Deterministic controls created before model inference; never shown to models."""
import ast
import json
from pathlib import Path

def interpolate(start, goal, fraction):
    if type(start) in (float, int) and type(goal) in (float, int):
        return float(start + fraction * (goal-start))
    if type(start) in (tuple, list) and type(goal) is type(start) and len(start)==len(goal):
        return type(goal)(interpolate(a,b,fraction) for a,b in zip(start,goal))
    raise ValueError("Non-numeric interpolation")

def assignments(source):
    values={}
    for n in ast.walk(ast.parse(source)):
        if isinstance(n,ast.Assign) and len(n.targets)==1:
            try: values[ast.unparse(n.targets[0])]=ast.literal_eval(n.value)
            except (ValueError,TypeError): pass
    return values

def control(task, fraction):
    d=Path(task['directory'])
    start=(d/'start.py').read_text()
    goal=(d/'goal.py').read_text()
    if task['category'] in ('blendshape','lighting','placement'):
        starts=assignments(start)
        tree=ast.parse(goal)
        changed=0
        for n in ast.walk(tree):
            if not isinstance(n,ast.Assign) or len(n.targets)!=1: continue
            key=ast.unparse(n.targets[0])
            if key not in starts: continue
            try:
                value=ast.literal_eval(n.value)
                if value==starts[key]: continue
                mixed=interpolate(starts[key],value,fraction)
            except (ValueError,TypeError): continue
            n.value=ast.parse(repr(mixed),mode='eval').body
            changed+=1
        assert changed>0,task['id']
        return ast.unparse(ast.fix_missing_locations(tree))+'\n'
    replacements={
      'geometry30': [('9.3000', '4.6500')],
      'geometry38': [('value.outputs[0].default_value = 1.4000',
                      'value.outputs[0].default_value = '+str(6.9+fraction*(1.4-6.9)))],
      'material23': [("'Distance': 3.5000", "'Distance': "+str(3.5*fraction))],
      'material6': [('[0.1470, 0.0482, 0.0273, 1.0000]',
                     repr(interpolate([0.0138,0.0119,0.1470,1.0],[0.1470,0.0482,0.0273,1.0],fraction))),
                    ('position = 0.5000','position = '+str(0.8227+fraction*(0.5-0.8227))),
                    ('position = 0.0000','position = '+str(0.2864*(1-fraction)))]
    }
    if task['id']=='geometry30':
        replacements['geometry30']=[('9.3000',str(9.30001 if fraction==0.95 else 4.65+fraction*4.65))]
    for old,new in replacements[task['id']]:
        assert old in goal,(task['id'],old)
        goal=goal.replace(old,new)
    ast.parse(goal)
    return goal

def create(manifest, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for t in manifest['all_tasks']:
        if t['id'] not in manifest['calibration_ids']: continue
        d=output/t['id'];d.mkdir(exist_ok=True)
        for name,f in [('near',0.95),('half',0.5)]:
            (d/(name+'.py')).write_text(control(t,f))
    (output/'design.json').write_text(json.dumps({
      'positive':'reference and 95% goal controls',
      'negative':'unchanged start and 50% goal controls',
      'note':'geometry30 near uses scale 9.30001 after visual audit found 2.5% scale changes alter discrete topology; wrong scale 6.975. geometry38 uses Value 6.9 to 1.4; material6 perturbs goal colors/ramp positions',
      'ids':manifest['calibration_ids']},indent=2))

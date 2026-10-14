"""Pre-inference audit; replace identical-program screen tasks within their category."""
import ast
from pathlib import Path

def audit_manifest(manifest):
    tasks=manifest['all_tasks']
    invalid=[]
    for t in tasks:
        d=Path(t['directory'])
        start=ast.dump(ast.parse((d/'start.py').read_text()),include_attributes=False)
        goal=ast.dump(ast.parse((d/'goal.py').read_text()),include_attributes=False)
        if start==goal: invalid.append(t['id'])
    manifest['invalid_task_ids']=invalid
    manifest.setdefault('original_subset_ids',manifest['subset_ids'].copy())
    replacements=manifest.setdefault('subset_replacements',{})
    for old in list(manifest['subset_ids']):
        if old not in invalid: continue
        task=next(t for t in tasks if t['id']==old)
        forbidden=set(invalid+manifest['subset_ids']+manifest['calibration_ids'])
        candidates=[t for t in tasks if t['category']==task['category'] and t['id'] not in forbidden]
        candidates.sort(key=lambda t:(t['base_group']!=task['base_group'],t['number']))
        replacement=candidates[0]['id']
        manifest['subset_ids'][manifest['subset_ids'].index(old)]=replacement
        replacements[old]={'replacement':replacement,'reason':'start and goal have identical Python AST'}
    return manifest

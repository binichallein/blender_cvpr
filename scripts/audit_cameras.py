"""Run with Blender: read object names from each base file without loading its scene."""
import argparse
import json
import sys
from pathlib import Path
import bpy
p=argparse.ArgumentParser()
p.add_argument('--manifest',required=True)
p.add_argument('--output',required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
m=json.loads(Path(a.manifest).read_text());inventory={}
for t in m['all_tasks']:
    key=t['base_group']
    if key in inventory:continue
    with bpy.data.libraries.load(t['base_blend'],link=True) as (source,target):
        inventory[key]=[name for name in source.objects if name.startswith('Camera')]
Path(a.output).write_text(json.dumps(inventory,indent=2))
print('CAMERA_INVENTORY',len(inventory),flush=True)

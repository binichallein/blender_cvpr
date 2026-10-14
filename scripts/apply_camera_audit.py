"""Drop orphan image views lacking a corresponding base-scene evaluation camera."""
import argparse
import json
from pathlib import Path

def apply(manifest, inventory):
    changes={}
    for t in manifest['all_tasks']:
        t.setdefault('released_views',t['views'].copy())
        available=set(inventory[t['base_group']])
        t['views']=[v for v in t['released_views'] if 'Camera'+Path(v).stem.removeprefix('render') in available]
        assert t['views'],f"No usable camera views: {t['id']}"
        missing=[v for v in t['released_views'] if v not in t['views']]
        if missing: changes[t['id']]=missing
    manifest['unavailable_camera_views']=changes
    return manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True);p.add_argument('--cameras',required=True)
    a=p.parse_args();path=Path(a.manifest)
    m=apply(json.loads(path.read_text()),json.loads(Path(a.cameras).read_text()))
    path.write_text(json.dumps(m,indent=2))
    print(json.dumps(m['unavailable_camera_views'],indent=2))
if __name__=='__main__':main()

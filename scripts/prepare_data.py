"""Validate/extract the release and select a fixed, scene-stratified subset."""
import argparse
import collections
import hashlib
import json
import random
import re
import zipfile
from pathlib import Path

CATEGORIES = ['blendshape', 'geometry', 'lighting', 'material', 'placement']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as z:
        for name in z.namelist():
            assert (args.out / name).resolve().is_relative_to(args.out.resolve())
        bad = z.testzip()
        assert bad is None, bad
        z.extractall(args.out)
    root = args.out / 'bench_data'
    bases = list((root / 'blender_files').glob('*.blend'))
    tasks = []
    for taskdir in root.iterdir():
        match = re.fullmatch(r'(blendshape|geometry|lighting|material|placement)(\d+)', taskdir.name)
        if not match:
            continue
        category, n = match[1], int(match[2])
        matches = []
        for base in bases:
            cat, lo, hi = base.stem.split('_')
            if cat == category and int(lo) <= n <= int(hi):
                matches.append(base)
        assert len(matches) == 1, (taskdir, matches)
        assert (taskdir / 'start.py').is_file() and (taskdir / 'goal.py').is_file()
        views = sorted(set(p.name for p in (taskdir / 'renders/start').glob('*.png')) &
                       set(p.name for p in (taskdir / 'renders/goal').glob('*.png')))
        assert views, taskdir
        tasks.append({'id': taskdir.name, 'category': category, 'number': n,
                      'directory': str(taskdir), 'base_blend': str(matches[0]),
                      'base_group': matches[0].stem, 'views': views})
    tasks.sort(key=lambda t: (CATEGORIES.index(t['category']), t['number']))
    rng = random.Random(20261015)
    selected = []
    calibration = []
    for cat in CATEGORIES:
        groups = collections.defaultdict(list)
        for t in tasks:
            if t['category'] == cat:
                groups[t['base_group']].append(t)
        keys = sorted(groups)
        rng.shuffle(keys)
        for values in groups.values():
            rng.shuffle(values)
        cat_selected = []
        while len(cat_selected) < 10:
            for key in keys:
                if groups[key] and len(cat_selected) < 10:
                    cat_selected.append(groups[key].pop())
        selected.extend(cat_selected)
        remaining = [t for key in keys for t in groups[key]]
        rng.shuffle(remaining)
        calibration.extend(remaining[:2])
    digest = hashlib.sha256()
    with args.archive.open('rb') as f:
        for b in iter(lambda: f.read(16 << 20), b''):
            digest.update(b)
    manifest = {'seed': 20261015, 'selection': 'round-robin across shuffled shared base scenes',
                'archive_sha256': digest.hexdigest(), 'all_tasks': tasks,
                'subset_ids': [t['id'] for t in selected],
                'calibration_ids': [t['id'] for t in calibration]}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ['archive_sha256','subset_ids','calibration_ids']}, indent=2), flush=True)

if __name__ == '__main__':
    main()

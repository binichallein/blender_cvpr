"""Audit supplied images against reference-program renders, independently of model answers."""
import argparse
import concurrent.futures
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path):
    with Image.open(path) as image:
        original_size = list(image.size)
        image = image.convert('RGB').resize((512, 512), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float64) / 255.0, original_size


def main():
    parser = argparse.ArgumentParser()
    for key in ['manifest', 'references', 'output']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--flag-mse', type=float, default=0.005)
    args = parser.parse_args()
    tasks = json.loads(Path(args.manifest).read_text())['all_tasks']
    refs = Path(args.references)

    def compare(task):
        directory = refs / task['id']
        if not all((directory / kind / 'render.json').exists() for kind in ['start', 'goal']):
            if args.partial:
                return None
            raise ValueError('Reference missing: ' + task['id'])
        per_view = []
        for view in task['views']:
            arrays, sizes = {}, {}
            for kind in ['start', 'goal']:
                arrays[kind], sizes[kind] = load_image(Path(task['directory']) / 'renders' / kind / view)
                arrays['reference_' + kind], _ = load_image(directory / kind / view)
            loss = lambda x, y: float(np.mean(np.square(arrays[x] - arrays[y])))
            per_view.append({
                'view': view, 'supplied_sizes': sizes,
                'start_reference_mse': loss('start', 'reference_start'),
                'goal_reference_mse': loss('goal', 'reference_goal'),
                'supplied_change_mse': loss('start', 'goal'),
                'reference_change_mse': loss('reference_start', 'reference_goal'),
            })
        maximum = max(max(v['start_reference_mse'], v['goal_reference_mse']) for v in per_view)
        return {'task': task['id'], 'category': task['category'], 'per_view': per_view,
                'max_view_reference_mse': maximum, 'flagged': maximum > args.flag_mse}

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        rows = [r for r in pool.map(compare, tasks) if r is not None]
    result = {
        'purpose': 'post-hoc data-quality sensitivity analysis; does not alter main score or model input',
        'uses_model_outputs': False,
        'method': 'RGB MSE after Lanczos resizing supplied images to 512x512; flag if either start or goal exceeds threshold in any view',
        'flag_mse': args.flag_mse, 'complete': len(rows) == len(tasks),
        'flagged_tasks': sorted(r['task'] for r in rows if r['flagged']), 'rows': rows,
        'limitations': 'A flag can reflect noise, different rendering settings, scene state or camera mismatch; it is not automatically an invalid task.',
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()

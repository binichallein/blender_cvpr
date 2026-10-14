"""Independent float64 pixel audit of complete full-set results and execution coverage."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from infer import build_messages, extract_code


def pixels(path):
    with Image.open(path) as image:
        array = np.asarray(image.convert('RGB'), dtype=np.float64) / 255.0
    assert array.shape == (512, 512, 3), (str(path), array.shape)
    return array


def main():
    parser = argparse.ArgumentParser()
    for name in ['runs', 'manifest', 'protocol', 'references', 'output']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    runs, references = Path(args.runs), Path(args.references)
    protocol = json.loads(Path(args.protocol).read_text())
    protocol_sha = hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest()
    manifest = json.loads(Path(args.manifest).read_text())
    results = json.loads((runs / 'complete.json').read_text())
    variants = {v['id']: v for v in protocol['variants']}
    assert set(results) == set(variants)
    scores = {name: {row['task']: row for row in result['rows']} for name, result in results.items()}
    task_ids = {task['id'] for task in manifest['all_tasks']}
    for name in variants:
        assert set(scores[name]) == task_ids
        assert len(results[name]['rows']) == len(task_ids)
        config = json.loads((runs / name / 'predictions/config.json').read_text())
        assert config['protocol_sha256'] == protocol_sha
        assert config['max_new_tokens'] == 32768
        assert config['sampling'] == protocol['sampling']
        assert config['variant'] == variants[name]
    audit = {name: {'tasks_checked': 0, 'predictions_recomputed': 0,
                    'successes_recomputed': 0, 'max_pixel_loss_difference': 0.0,
                    'max_reference_loss_difference': 0.0} for name in variants}
    for task in manifest['all_tasks']:
        task_id = task['id']
        views = task['views']
        starts = [pixels(references / task_id / 'start' / view) for view in views]
        goals = [pixels(references / task_id / 'goal' / view) for view in views]
        start_mse = float(np.mean([np.mean(np.square(x-y)) for x, y in zip(starts, goals)]))
        expected_input = build_messages(task)
        floor = protocol['category_rules'][task['category']]['noise_floor']['pl']
        threshold = protocol['category_rules'][task['category']]['rule']['pl']
        invalid = task_id in manifest.get('invalid_task_ids', []) or start_mse <= floor + 1e-8
        for name in variants:
            score = scores[name][task_id]
            pred = runs / name / 'predictions' / task_id
            generation = json.loads((pred / 'generation.json').read_text())
            completion = json.loads((pred / 'completion.json').read_text())
            assert json.loads((pred / 'input.json').read_text()) == expected_input
            assert generation['protocol_sha256'] == protocol_sha
            assert generation['thinking'] == variants[name]['thinking']
            assert generation['output_tokens'] == len(completion['token_ids']) <= 32768
            assert generation['input_tokens'] + generation['output_tokens'] <= protocol['context_capacity']
            assert generation['truncated'] == (completion['finish_reason'] == 'length')
            assert (pred / 'response.txt').read_text() == completion['text']
            assert bool(score.get('invalid_data')) == invalid, (name, task_id, 'invalid data disagreement')
            summary = audit[name]
            summary['tasks_checked'] += 1
            reference_difference = abs(start_mse - score['start_metrics']['pl'])
            summary['max_reference_loss_difference'] = max(
                reference_difference, summary['max_reference_loss_difference'])
            assert reference_difference < 2e-7, (name, task_id, 'reference loss disagreement')
            if generation['ok']:
                assert not generation['truncated']
                assert (pred / 'prediction.py').read_text() == extract_code((pred / 'final.txt').read_text())
            if invalid:
                assert not score['success']
                continue
            if score.get('metrics'):
                render_root = runs / name / 'renders' / task_id / 'prediction'
                meta = json.loads((render_root / 'render.json').read_text())
                assert meta['ok']
                assert meta['script_sha256'] == hashlib.sha256((pred / 'prediction.py').read_bytes()).hexdigest()
                loss = float(np.mean([np.mean(np.square(pixels(render_root / view) - goal))
                                      for view, goal in zip(views, goals)]))
                relative = max(0.0, loss-floor) / max(start_mse-floor, 1e-8)
                success = relative <= threshold
                difference = abs(loss-score['metrics']['pl'])
                summary['max_pixel_loss_difference'] = max(difference, summary['max_pixel_loss_difference'])
                assert difference < 2e-7, (name, task_id, 'prediction loss disagreement')
                assert success == score['success'], (name, task_id, 'threshold decision disagreement')
                summary['predictions_recomputed'] += 1
                summary['successes_recomputed'] += int(success)
            else:
                assert not score['success'] and score.get('failure')
        print('AUDITED', task_id, flush=True)
    for name in variants:
        assert audit[name]['successes_recomputed'] == results[name]['summary']['successes']
    report = {'passed': True, 'protocol_sha256': protocol_sha,
              'method': 'independent float64 RGB MSE and binary decisions; input/output and execution provenance',
              'variants': audit}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

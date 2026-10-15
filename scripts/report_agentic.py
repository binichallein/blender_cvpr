"""Independent final-selection audit and compact agentic benchmark report."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
from agentic_core import TOTAL_TOKENS, MAX_RENDERS, MAX_TURNS, select_attempt, visible_history
from agentic_sandbox import digest, atomic_json
from infer import build_messages


def pixels(path):
    with Image.open(path) as image:
        array = np.asarray(image.convert('RGB'), dtype=np.float64) / 255
    assert array.shape == (512, 512, 3)
    return array


def run(config_path):
    c = json.loads(Path(config_path).read_text())
    root = Path(c['root'])
    p = json.loads(Path(c['protocol']).read_text())
    protocol_sha = digest(c['protocol'])
    m = json.loads(Path(c['manifest']).read_text())
    refs = Path(c['references'])
    results = {v['id']: json.loads((root / 'scores' / (v['id'] + '.json')).read_text()) for v in p['variants']}
    baseline = json.loads((Path(c['baseline_results']) / 'results.json').read_text())
    audit, statistics, comparisons, flat = {}, {}, {}, []
    input_script = __import__('agentic_core').SYSTEM
    for v in p['variants']:
        name = v['id']
        rows = {row['task']: row for row in results[name]['rows']}
        assert set(rows) == {t['id'] for t in m['all_tasks']} and len(results[name]['rows']) == len(rows)
        predictions = root / 'predictions' / name
        inference_config = json.loads((predictions / 'config.json').read_text())
        assert inference_config['protocol_sha256'] == protocol_sha
        assert inference_config['variant'] == v
        checked = {'tasks': 0, 'turns': 0, 'renders': 0, 'successes_recomputed': 0,
                   'max_pixel_loss_difference': 0.0, 'max_reference_loss_difference': 0.0}
        generations = []
        for task in m['all_tasks']:
            tid = task['id']
            d = predictions / tid
            g = json.loads((d / 'generation.json').read_text())
            state = json.loads((d / 'state.json').read_text())
            generations.append(g)
            assert g['protocol_sha256'] == protocol_sha == state['protocol_sha256']
            assert g['thinking'] == v['thinking']
            assert g['output_tokens'] <= TOTAL_TOKENS and g['turns'] <= MAX_TURNS
            assert g['render_attempts'] == len(state['attempts']) <= MAX_RENDERS
            assert g['stop_reason'] in ('submitted','token_budget','turn_budget')
            expected = build_messages(task)
            expected[0]['content'] = [{'type':'text','text':input_script}]
            expected[1]['content'][-1] = {'type':'text','text':'Use run_blender to try a complete revised script, inspect feedback, and call submit to finish.'}
            assert json.loads((d / 'input.json').read_text()) == expected
            assert state['messages'][:2] == expected
            total, inputs = 0, 0
            for index in range(1, g['turns']+1):
                turn = d / f'turn-{index:03d}'
                completion = json.loads((turn / 'completion.json').read_text())
                turn_input = json.loads((turn / 'input.json').read_text())
                actual_expected = visible_history(state['messages'][:2 + 2*(index-1)])
                assert turn_input == actual_expected, (name,tid,index,'visible history changed')
                assert completion['max_tokens'] == min(32768,TOTAL_TOKENS-total)
                assert len(completion['token_ids']) <= completion['max_tokens']
                assert completion['input_tokens'] + len(completion['token_ids']) <= p['context_capacity']
                total += len(completion['token_ids']); inputs += completion['input_tokens']
                checked['turns'] += 1
            assert g['output_tokens'] == total == state['output_tokens']
            assert g['input_tokens'] == inputs
            chosen = select_attempt(state['attempts'],state.get('submitted'))
            assert g['selected_attempt'] == (chosen['attempt_id'] if chosen else None)
            assert g['ok'] == (chosen is not None)
            for attempt in state['attempts']:
                result = attempt['result']
                assert result['script_sha256'] == digest(attempt['script'])
                assert result['views'] == task['views'] and result['samples'] == 512 and result['seed'] == 20261015
                # Inspect actual Docker mounts, including adapter-translated source paths.
                destinations = {mount['Destination'] for mount in result['mounts']}
                assert destinations == {'/opt/blender','/opt/infinigen','/task/base','/task/input',
                                        '/runtime','/driver','/opt/blender/3.6/scripts/addons/cycles/lib','/out'}
                assert all(not mount['RW'] for mount in result['mounts'] if mount['Destination'] != '/out')
                checked['renders'] += 1
            if chosen:
                assert digest(d / 'prediction.py') == g['selected_script_sha256'] == digest(chosen['script'])
            score = rows[tid]
            starts = [pixels(refs / tid / 'start' / view) for view in task['views']]
            goals = [pixels(refs / tid / 'goal' / view) for view in task['views']]
            start = float(np.mean([np.mean((x-y)**2) for x,y in zip(starts,goals)]))
            floor = p['category_rules'][task['category']]['noise_floor']['pl']
            threshold = p['category_rules'][task['category']]['rule']['pl']
            invalid = tid in m['invalid_task_ids'] or start <= floor+1e-8
            assert bool(score.get('invalid_data')) == invalid
            difference = abs(start-score['start_metrics']['pl'])
            checked['max_reference_loss_difference'] = max(difference,checked['max_reference_loss_difference'])
            assert difference < 2e-7
            if score.get('metrics'):
                assert chosen and chosen['result']['ok']
                render_root = root / 'renders' / name / tid / 'prediction'
                for view in task['views']:
                    assert digest(render_root/view) == digest(Path(chosen['output'])/view)
                loss = float(np.mean([np.mean((pixels(render_root/view)-goal)**2) for view,goal in zip(task['views'],goals)]))
                relative = max(0.0,loss-floor)/max(start-floor,1e-8)
                assert (relative <= threshold) == score['success']
                difference = abs(loss-score['metrics']['pl'])
                checked['max_pixel_loss_difference'] = max(difference,checked['max_pixel_loss_difference'])
                assert difference < 2e-7
                checked['successes_recomputed'] += int(score['success'])
            else:
                assert not score['success']
            checked['tasks'] += 1
            flat.append({'variant':name, 'task':tid,'category':task['category'],
                         'invalid_data':score.get('invalid_data',''),'success':score['success'],
                         'pl':score.get('metrics',{}).get('pl'),'nclip':score.get('metrics',{}).get('nclip'),
                         'failure':score.get('failure',''),
                         **{k:g[k] for k in ['output_tokens','reasoning_tokens','input_tokens','turns',
                            'render_attempts','selected_attempt','explicit_submission','stop_reason','truncated_turns','parse_errors']}})
        assert checked['successes_recomputed'] == results[name]['summary']['successes']
        audit[name] = checked
        statistics[name] = {
            'total_output_tokens':sum(g['output_tokens'] for g in generations),
            'total_input_tokens':sum(g['input_tokens'] for g in generations),
            'total_reasoning_tokens':sum(g['reasoning_tokens'] for g in generations),
            'total_render_attempts':sum(g['render_attempts'] for g in generations),
            'mean_render_attempts':float(np.mean([g['render_attempts'] for g in generations])),
            'stop_reasons':dict(collections.Counter(g['stop_reason'] for g in generations)),
            'explicit_submissions':sum(g['explicit_submission'] for g in generations),
            'truncated_turns':sum(g['truncated_turns'] for g in generations),
            'parse_errors':sum(g['parse_errors'] for g in generations)}
        mode = 'think' if v['thinking'] else 'nothink'
        old_name = 'qwen3.6-27b-'+mode
        old = {r['task']:r for r in baseline[old_name]['rows']}
        valid = [tid for tid in rows if not rows[tid].get('invalid_data')]
        comparisons[name] = {'baseline':old_name,'scorable_tasks':len(valid),
            'both_success':[tid for tid in valid if rows[tid]['success'] and old[tid]['success']],
            'rescued':[tid for tid in valid if rows[tid]['success'] and not old[tid]['success']],
            'regressed':[tid for tid in valid if not rows[tid]['success'] and old[tid]['success']],
            'both_fail':[tid for tid in valid if not rows[tid]['success'] and not old[tid]['success']]}
    output = Path(c['report_output']); output.mkdir(parents=True,exist_ok=True)
    for filename,data in [('results.json',results),('generation-statistics.json',statistics),
                          ('paired-vs-oneshot.json',comparisons),('final-audit.json',{
                              'passed':True,'protocol_sha256':protocol_sha,'variants':audit})]:
        atomic_json(output/filename,data)
    with (output/'tasks.csv').open('w') as file:
        writer = csv.DictWriter(file,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    for filename in ['protocol.json','sandbox-parity.json','sandbox-isolation.json']:
        shutil.copyfile(root/filename,output/filename)
    atomic_json(output/'model-configs.json',{v['id']:json.loads((root/'predictions'/v['id']/'config.json').read_text()) for v in p['variants']})
    source_hashes = {name:digest(Path(c['repo'])/'scripts'/name) for name in [
        'agentic_core.py','agentic_sandbox.py','infer_agentic.py','run_agentic.py','report_agentic.py',
        'blender_render.py','score.py','metrics.py']}
    atomic_json(output/'sources.json',{'sha256':source_hashes,'git_head':subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=c['repo'],text=True).strip(),'manifest_sha256':digest(c['manifest'])})
    lines = ['# Qwen3.6-27B Blender tool-agent evaluation','',
        'All 280 released tasks per mode; 277 scorable after the same three no-op exclusions. Same BF16 checkpoint, native thinking on/off, vLLM continuous batching.',
        '',
        'The agent receives the original script and paired images, may execute a complete replacement script up to five times, inspect errors/renders, and select its final attempt. No hidden score, reference program, or helper model is exposed.',
        '',
        '**Budget:** 65536 total output tokens per trajectory (reasoning and tool syntax included), 32768 per response, at most 12 responses. This exceeds the earlier one-shot 32768-token budget, so differences do not isolate feedback alone.',
        '',
        '| Mode | Final success | Prior one-shot | Rescued / regressed | Mean executions |',
        '|---|---:|---:|---:|---:|']
    for v in p['variants']:
        name=v['id'];mode='think' if v['thinking'] else 'nothink'
        s=results[name]['summary'];old=baseline['qwen3.6-27b-'+mode]['summary'];pair=comparisons[name]
        lines.append(f"| {mode} | {s['successes']}/{s['scorable_tasks']} ({s['success_rate']:.2%}) | {old['successes']}/{old['scorable_tasks']} | {len(pair['rescued'])} / {len(pair['regressed'])} | {statistics[name]['mean_render_attempts']:.2f} |")
    lines += ['','## Category results','','| Category | nothink | think |','|---|---:|---:|']
    for category in sorted(results[p['variants'][0]['id']]['by_category']):
        row = [category]
        for v in p['variants']:
            s=results[v['id']]['by_category'][category];row.append(f"{s['successes']}/{s['scorable_tasks']}")
        lines.append('| '+' | '.join(row)+' |')
    lines += ['','## Interpretation and verification','',
        'Success uses the previously frozen, category-specific calibrated photometric error thresholds; this is a derived binary metric, not an official BlenderGym success score. Threshold calibration used only two tasks per category. Continuous PL and corrected CLIP distance, execution coverage, and calibration-excluded summaries are included in results.json.',
        '',
        'The main score uses the agent-selected attempt. If it exhausts its budget without submission, the most recent attempted script is used, including execution failures. No oracle best-of-five selection is used. Raw intermediate traces stay on ml2.',
        '',
        'Each execution uses a fresh network-disabled container with only task inputs and runtime dependencies mounted. Cameras and render settings match the previous run: Blender 3.6, CUDA Cycles, 512×512, 512 samples, seed 20261015. The same cached A100 kernel is reused. Five category smoke renders are checked against the original reference renders before launch.',
        '',
        'Independent float64 pixel audit verifies all final success decisions; trace audit checks input history, token totals, budgets, selected-script/image hashes, and actual container mount boundaries. Existing released-image/reference mismatches remain inherited from the baseline and are not silently corrected.',
        '',
        'Files: results.json, tasks.csv, paired-vs-oneshot.json, generation-statistics.json, protocol.json, final-audit.json, model-configs.json, sources.json.',
        '']
    (output/'README.md').write_text('\n'.join(lines))
    atomic_json(root/'final-audit.json',{'passed':True,'protocol_sha256':protocol_sha,'variants':audit})
    print(json.dumps({name:results[name]['summary'] for name in results},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    run(parser.parse_args().config)

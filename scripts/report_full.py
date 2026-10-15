"""Publish complete six-variant scores with transparent denominators and token usage."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from score import summarize


def load(path):
    return json.loads(Path(path).read_text())


def number(value):
    return f'{value:.6f}' if value is not None else 'N/A'


def rate(summary):
    if not summary['scorable_tasks']:
        return '0/0 (N/A)'
    return f"{summary['successes']}/{summary['scorable_tasks']} ({summary['success_rate']:.2%})"


def main():
    parser = argparse.ArgumentParser()
    for name in ['runs', 'manifest', 'protocol', 'output', 'model-metadata', 'reference-audit']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    runs, out = Path(args.runs), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest, protocol = load(args.manifest), load(args.protocol)
    results = load(runs / 'complete.json')
    variants = {v['id']: v for v in protocol['variants']}
    protocol_sha = hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest()
    ids = {task['id'] for task in manifest['all_tasks']}
    assert set(results) == set(variants)
    for name, result in results.items():
        assert result['protocol_sha256'] == protocol_sha
        assert len(result['rows']) == len(ids)
        assert {row['task'] for row in result['rows']} == ids
        assert result['summary'] == summarize(result['rows'])
    invalid_sets = [{(r['task'], r['invalid_data']) for r in result['rows'] if r.get('invalid_data')}
                    for result in results.values()]
    assert all(value == invalid_sets[0] for value in invalid_sets)
    shutil.copy2(runs / 'complete.json', out / 'results.json')
    shutil.copy2(args.protocol, out / 'protocol.json')
    reference_audit = load(args.reference_audit)
    assert reference_audit['complete'] and {r['task'] for r in reference_audit['rows']} == ids
    flagged = set(reference_audit['flagged_tasks'])
    sensitivity = {name: summarize([row for row in result['rows'] if row['task'] not in flagged])
                   for name, result in results.items()}
    shutil.copy2(args.reference_audit, out / 'reference-image-audit.json')
    (out / 'data-sensitivity.json').write_text(json.dumps(sensitivity, indent=2) + '\n')
    configurations, statistics, sources = {}, {}, {}
    generations = {}
    for name in variants:
        predictions = runs / name / 'predictions'
        configurations[name] = load(predictions / 'config.json')
        assert configurations[name]['protocol_sha256'] == protocol_sha
        sources[name] = load(runs / name / 'source.json')
        sources[name]['earlier_inference_sources'] = [
            load(p) for p in sorted((runs / name).glob('source-history-*.json'))]
        sources[name]['invocations'] = [load(p) for p in sorted(predictions.glob('invocation-*.json'))]
        generations[name] = {task: load(predictions / task / 'generation.json') for task in ids}
        records = list(generations[name].values())
        execution_errors = []
        for row in results[name]['rows']:
            if row.get('failure') not in ('execution', 'missing_view'):
                continue
            directory = runs / name / 'renders' / row['task'] / 'prediction'
            messages = []
            log = directory / 'blender.log'
            if log.exists():
                messages = [line for line in log.read_text(errors='replace').splitlines()
                            if line.startswith(('KeyError:', 'TypeError:', 'AttributeError:',
                                'RuntimeError:', 'ValueError:', 'NameError:', 'IndexError:',
                                'ZeroDivisionError:', 'ModuleNotFoundError:', 'MemoryError:'))]
            failure_file = directory / 'failure.json'
            failure = load(failure_file) if failure_file.exists() else {}
            execution_errors.append({'task': row['task'], 'failure': row['failure'],
                                     'returncode': failure.get('returncode'), 'errors': messages[-8:]})
        timings = [load(p) for p in sorted(predictions.glob('timing-*.json'))]
        complete_timing = sum(t['completed_in_this_invocation'] for t in timings) == len(ids)
        replica_runs = [load(p) for p in sorted(predictions.glob('replica-runs/*/run.json'))]
        generation_seconds = sum(t['generation_wall_seconds'] for t in timings) if complete_timing else None
        if replica_runs:
            # Parallel process times must never be added and called elapsed wall time.
            generation_seconds = None
            if complete_timing and all('generation_started_unix' in t for t in timings):
                intervals = sorted((t['generation_started_unix'], t['generation_finished_unix']) for t in timings)
                merged = []
                for start, end in intervals:
                    if merged and start <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], end)
                    else:
                        merged.append([start, end])
                generation_seconds = sum(end-start for start, end in merged)
        statistics[name] = {
            'generated_tasks': len(records),
            'execution_failures': execution_errors,
            'total_output_tokens': sum(r['output_tokens'] for r in records),
            'reasoning_tokens': sum(r.get('reasoning_tokens', 0) for r in records),
            'answer_tokens': sum(r.get('answer_tokens', 0) for r in records),
            'unclassified_token_tasks': [r['task'] for r in records if 'reasoning_tokens' not in r],
            'truncated_tasks': sorted(r['task'] for r in records if r['truncated']),
            'generation_or_parse_failures': sorted(r['task'] for r in records if not r['ok']),
            'reasoning_not_terminated_tasks': sorted(r['task'] for r in records
                if variants[name]['thinking'] and not r.get('reasoning_terminated')),
            'timing_records': timings,
            'replica_runs': replica_runs,
            'complete_timing_coverage': complete_timing,
            'generation_wall_seconds': generation_seconds,
        }
    for filename, value in [('model-configs.json', configurations),
                            ('generation-statistics.json', statistics), ('sources.json', sources)]:
        (out / filename).write_text(json.dumps(value, indent=2) + '\n')
    coordinator_sources = [load(p) for p in sorted(runs.glob('source-*.json'))]
    (out / 'coordinator-sources.json').write_text(json.dumps(coordinator_sources, indent=2) + '\n')
    meta_out = out / 'model-metadata'
    meta_out.mkdir(exist_ok=True)
    for path in sorted(Path(args.model_metadata).glob('*-files.json')):
        shutil.copy2(path, meta_out / path.name)
    paired = {}
    for family in ['qwen3-vl-8b', 'qwen3-vl-32b', 'qwen3.6-27b']:
        off = {r['task']: r for r in results[family + '-nothink']['rows']}
        on = {r['task']: r for r in results[family + '-think']['rows']}
        eligible = sorted(task for task in ids if not off[task].get('invalid_data'))
        paired[family] = {
            'scorable_tasks': len(eligible),
            'both_success': [t for t in eligible if off[t]['success'] and on[t]['success']],
            'only_thinking_success': [t for t in eligible if not off[t]['success'] and on[t]['success']],
            'only_nonthinking_success': [t for t in eligible if off[t]['success'] and not on[t]['success']],
            'both_fail': [t for t in eligible if not off[t]['success'] and not on[t]['success']],
        }
    (out / 'paired-comparison.json').write_text(json.dumps(paired, indent=2) + '\n')
    fields = ['variant', 'task', 'category', 'thinking', 'screening_subset', 'calibration_task',
              'success', 'invalid_data', 'failure', 'pl', 'nclip', 'relative_pl',
              'input_tokens', 'output_tokens', 'reasoning_tokens', 'answer_tokens',
              'truncated', 'reasoning_terminated', 'generation_seconds']
    with (out / 'tasks.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for name, result in results.items():
            for row in result['rows']:
                generation = generations[name][row['task']]
                writer.writerow({
                    'variant': name, 'thinking': variants[name]['thinking'],
                    'screening_subset': row['task'] in manifest['subset_ids'],
                    'calibration_task': row['task'] in manifest['calibration_ids'],
                    **{k: row.get(k, '') for k in ['task', 'category', 'success', 'invalid_data',
                                                 'failure', 'input_tokens', 'output_tokens',
                                                 'generation_seconds']},
                    **{k: generation.get(k, '') for k in ['reasoning_tokens', 'answer_tokens',
                                                         'truncated', 'reasoning_terminated']},
                    'pl': row.get('metrics', {}).get('pl', ''),
                    'nclip': row.get('metrics', {}).get('nclip', ''),
                    'relative_pl': row.get('relative_error', {}).get('pl', ''),
                })
    text = [
        '# BlenderGym 全量 thinking / non-thinking 对比', '',
        f'六组均对发布包的 {len(ids)} 题重新生成答案；不使用上一轮 11/50 扩测门槛，也不复用上一轮答案。',
        '本轮统一 32768 个输出 token（思考与最终代码合计），BF16，temperature=0.6、top_p=0.95、top_k=20、seed=20261015。'
        '使用 vLLM 连续批处理，完成一题就补入下一题；每题一次回答，无工具调用、重试修复或挑选最佳答案。', '',
        '| 变体 | 成功数 / 可评分题数 | 有效渲染 | 平均 PL | 平均 N-CLIP |',
        '|---|---:|---:|---:|---:|',
    ]
    for name, result in results.items():
        s = result['summary']
        means = s['valid_only_mean']
        text.append(f"| {name} | {rate(s)} | {s['rendered_tasks']} | "
                    f"{number(means['pl'])} | {number(means['nclip'])} |")
    text += [
        '',
        'PL 与 N-CLIP 越低越好；这里的连续误差均值仅覆盖成功渲染的有效题目。'
        '代码格式、输出截断和执行失败保留在成功率分母中。数据无有效差异的题目统一排除，详见逐题结果。',
        '',
        '| 类别 | ' + ' | '.join(variants) + ' |',
        '|---|' + '---:|' * len(variants),
    ]
    for category in ['blendshape', 'geometry', 'lighting', 'material', 'placement']:
        text.append('| ' + category + ' | ' + ' | '.join(
            rate(results[name]['by_category'][category]) for name in variants) + ' |')
    text += [
        '',
        '补充的数据一致性分析：本轮进行中发现部分发布图片与场景文件的重渲染不一致。'
        '例如 lighting24 的后两个视角存在明显差异；相机恢复操作只产生约 1e-6 的矩阵数值差异，不能解释这种视角变化。'
        '以下分析是在运行期间提出的事后检查，不改变主分数、输入或成功阈值。',
        f"将发布图片用 Lanczos 缩放到 512×512 后，任一当前/目标视角与参考渲染的 RGB MSE 超过 "
        f"{reference_audit['flag_mse']} 则标记。共标记 {len(flagged)} 题；高误差也可能来自噪声或渲染设置差异，不能直接认定每道标记题无效。",
        '',
        '| 变体 | 排除图片差异标记题后的成功数 / 可评分题数 |',
        '|---|---:|',
    ]
    for name in variants:
        text.append(f"| {name} | {rate(sensitivity[name])} |")
    text += ['', '| 变体 | 原筛选 50 题 | 其余题目 | 排除阈值校准题 |',
             '|---|---:|---:|---:|']
    for name, result in results.items():
        text.append(f"| {name} | {rate(result['screening_subset'])} | "
                    f"{rate(result['remaining_tasks'])} | {rate(result['excluding_calibration'])} |")
    text += ['', '| 变体 | 总输出 token | 思考 token | 截断题数 | 格式/生成失败 | 执行失败 | 生成墙钟秒 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, stats in statistics.items():
        seconds = stats['generation_wall_seconds']
        duration = f'{seconds:.1f}' if seconds is not None else 'N/A（断点续跑计时不全）'
        text.append(f"| {name} | {stats['total_output_tokens']} | {stats['reasoning_tokens']} | "
                    f"{len(stats['truncated_tasks'])} | {len(stats['generation_or_parse_failures'])} | {len(stats['execution_failures'])} | {duration} |")
    text += [
        '',
        '生成墙钟时间包含图像预处理和排队，不包含模型加载、通用图片检查、Blender 渲染及评分。'
        '32B 每个实例使用双卡张量并行，其他模型每实例单卡。本轮 Qwen3.6 thinking 续跑时将未完成题分给两个相同的单卡副本；'
        '已经完成的回答保留，中断时尚未完成的请求按原参数恢复。各组时间不能直接作为相同硬件吞吐量比较，'
        '中断前未完成请求的额外计算不包含在最终回答 token 统计中；完整计时覆盖不足时标为 N/A。',
        '思考和答案 token 按原生 `</think>` 分界统计，分隔符本身不属于两者，因此两者之和可能略小于总输出。'
        '未结束思考即耗尽预算的回答记为截断失败，只执行最终答案中的完整程序。',
        '',
        'Qwen3-VL 的两种模式使用官方 Instruct / Thinking 不同权重，差异不能单独归因于开关；'
        'Qwen3.6-27B 使用同一多模态权重的 enable_thinking=False / True。',
        '输入均为起始代码和有效相机的当前/目标图片。成功判定沿用模型评测前冻结的类别相对像素误差阈值，'
        '属于本实验的近似视觉成功指标，不是 BlenderGym 官方二元分数。每类只有两题用于阈值校准，'
        '因此另外列出排除校准题的分数。',
        '',
        '数据处理：blendshape73、geometry13、geometry44 起始和目标程序相同；'
        'lighting1–5 的 render5.png 无对应相机，输入与评分均去掉该视角。'
        '参考变化低于渲染噪声阈值的任务也会被判为数据无效。'
        f'本轮实际排除 {len(invalid_sets[0])} 题，所有排除原因见 results.json。',
        '',
        '本轮解码采样和预算不同于先前贪心 8192-token 筛选，应作为独立实验阅读。',
        '',
        '文件：[逐题 CSV](tasks.csv)、[完整分数](results.json)、[配对结果](paired-comparison.json)、'
        '[生成统计](generation-statistics.json)、[模型配置](model-configs.json)、[冻结协议](protocol.json)、'
        '[来源与代码版本](sources.json)、[协调程序版本](coordinator-sources.json)、[图片一致性审查](reference-image-audit.json)、[补充分数](data-sensitivity.json)、'
        '[独立像素及输入输出复核](final-audit.json)、[续跑文件完整性核验](resume-audit.json)、[续跑记录](replica-transition.json)。权重、原始回答、思考文本、Python 程序与渲染图片保存在 ml2。',
    ]
    (out / 'README.md').write_text('\n'.join(text) + '\n')
    print(out, flush=True)


if __name__ == '__main__':
    main()

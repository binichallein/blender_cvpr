"""Export a compact Chinese report and per-task CSV after the coordinator finishes."""
import argparse
import csv
import json
import shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();runs=Path(a.runs);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    def fmt(value): return f'{value:.6f}' if value is not None else 'N/A'
    results=json.loads((runs/'complete.json').read_text())
    shutil.copy2(runs/'complete.json',out/'results.json')
    text=['# Qwen3-VL BlenderGym 单次生成评测','','每个模型先测相同的 50 题，五类各 10 题。达到 11/50 才扩展到全量。模型只接收起始代码及有效相机的成对当前/目标图像，输出完整 Python 程序；没有工具调用、重试或代码修复。','',
          '| 模型 | 成功数 | 成功率 | 可渲染 | 平均 PL | 平均标准 N-CLIP |','|---|---:|---:|---:|---:|---:|']
    for name,result in results.items():
        s=result['subset']['summary'];mean=s['valid_only_mean']
        text.append(f"| {name} | {s['successes']}/50 | {s['success_rate']:.1%} | {s['rendered_tasks']}/50 | {fmt(mean['pl'])} | {fmt(mean['nclip'])} |")
    text+=['','两个模型均使用 vLLM AsyncLLM 连续批处理（BF16、贪心解码、最多 8192 个输出 token）。完成一题即保存并补充下一请求；早期 Transformers 试跑单独保留，不计入结果。','','连续误差只在成功渲染的样本上平均，须结合可渲染数量阅读；程序或输出格式失败仍计入二元成功率的分母。PL 和 N-CLIP 越低越好。','','| 类别 | 8B 成功数 | 32B 成功数 |','|---|---:|---:|']
    for cat in ['blendshape','geometry','lighting','material','placement']:
        counts=[results[n]['subset']['by_category'][cat]['successes'] for n in ['qwen3-vl-8b','qwen3-vl-32b']]
        text.append(f'| {cat} | {counts[0]}/10 | {counts[1]}/10 |')
    text+=['','成功判定使用模型测试前冻结的、按类别校准的相对像素误差阈值，属于本实验的近似视觉成功指标，不是 BlenderGym 官方二元分数。CLIP 使用标准单次缩放预处理。详细定义见 [协议](../../docs/protocol.md)，逐题结果见 [results.json](results.json) 和 [tasks.csv](tasks.csv)。','',
           '发布包有 280 题。三题的起始/目标程序相同：blendshape73、geometry13、geometry44；抽样中的前两题已在模型测试前替换为同类题。lighting1–5 的 render5.png 没有对应场景相机，输入和评分均剔除此视角。全部修正不依赖模型答题结果。']
    for name,result in results.items():
        if 'full' not in result:
            text.append(f"\n{name} 未达到扩测门槛，没有进行全量推理。")
        else:
            s=result['full']['summary'];remaining=result['full']['remaining_tasks'];clean=result['full']['excluding_calibration']
            text.append(f"\n{name} 全量：{s['successes']}/{s['scorable_tasks']}（{s['success_rate']:.1%}），发布任务总数 {s['tasks']}，数据问题 {len(s['invalid_data'])} 题；扩展部分 {remaining['successes']}/{remaining['scorable_tasks']}。排除用于校准的任务后为 {clean['successes']}/{clean['scorable_tasks']}。全量结果受 50 题筛选门槛条件化。")
    text+=['','每类只用两道独立任务校准阈值，因此二元成功率适合本轮筛选与模型比较，不能代替人工对复杂场景正确性的完整审查。该实验测量一次性看图改代码能力，与原项目多轮搜索设置不同。']
    (out/'README.md').write_text('\n'.join(text)+'\n')
    with (out/'tasks.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['model','split','task','category','success','invalid_data','failure','pl','nclip','relative_pl','input_tokens','output_tokens','generation_seconds'])
        writer.writeheader()
        for model,result in results.items():
            for split,scores in result.items():
                for row in scores['rows']:
                    writer.writerow({'model':model,'split':split,
                        **{k:row.get(k,'') for k in ['task','category','success','invalid_data','failure','input_tokens','output_tokens','generation_seconds']},
                        'pl':row.get('metrics',{}).get('pl',''),'nclip':row.get('metrics',{}).get('nclip',''),
                        'relative_pl':row.get('relative_error',{}).get('pl','')})
    config={}
    for model in results:
        config[model]=json.loads((runs/model/'predictions/config.json').read_text())
    (out/'model-configs.json').write_text(json.dumps(config,indent=2))
    if (runs/'source.json').exists(): shutil.copy2(runs/'source.json',out/'source.json')
    print(out,flush=True)
if __name__=='__main__': main()

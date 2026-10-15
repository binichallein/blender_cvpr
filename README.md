# BlenderGym model evaluation

Reproducible, non-agentic evaluation of Qwen vision-language models on BlenderGym.

The full experiment is complete: six variants each generated fresh answers for all 280 released tasks on ml2, using vLLM continuous batching and a 32,768-token output budget including reasoning and final code.

| Model family | Non-thinking | Thinking |
|---|---:|---:|
| Qwen3-VL-8B | 12/277 (4.33%) | 12/277 (4.33%) |
| Qwen3-VL-32B | 14/277 (5.05%) | 18/277 (6.50%) |
| Qwen3.6-27B | 19/277 (6.86%) | 17/277 (6.14%) |

These are one-shot success rates under the visual thresholds frozen before model evaluation, not an official BlenderGym binary score. Three tasks with identical start and goal programs are excluded consistently, leaving 277 scorable tasks. Truncation and execution failures remain in the denominator.

See the [full report, category results, continuous PL/N-CLIP metrics, per-task CSV and audits](results/2026-10-15-full-thinking/README.md). The report also gives scores excluding calibration tasks and a separately labeled sensitivity analysis for reference-image inconsistencies.

Qwen3-VL uses separate official Instruct and Thinking checkpoints. Qwen3.6-27B uses the same multimodal weights with native thinking disabled or enabled.

Independent checks passed for all 1,680 input/output records and 1,505 scorable rendered predictions. Float64 pixel-loss recomputation reproduced every reported success decision. The 44 completed answers retained during replica continuation were unchanged across 308 checked files.

See the [full comparison protocol and running notes](docs/full-thinking-plan.md) and [frozen configuration](docs/protocol-full-thinking.json). Raw responses, reasoning, scripts, rendered images, datasets and weights remain on ml2.

The earlier greedy 8,192-token, 50-task screening experiment remains separate: Qwen3-VL-8B-Instruct and Qwen3-VL-32B-Instruct each achieved 2/50 and did not pass its 11/50 expansion gate. See the [original screening report](results/2026-10-15-vllm/README.md), [screening protocol](docs/protocol.md), [task selection](docs/subset.json), and [screening instructions](docs/run.md).

## Initial screening protocol

Historical protocol for the first 50-task experiment; the subsequent full-evaluation request superseded its expansion gate.

- One-shot image-conditioned editing of the complete starter Python program.
- Identical scene-stratified subset: 10 tasks from each of five categories.
- Calibrate and freeze a derived visual-success definition before model inference. Report original photometric loss and N-CLIP separately.
- Independently expand a model to the full available task set only after at least 11 successes among its 50 subset tasks.
- Save all responses, scripts, rendering errors and scores. Never expose goal scripts to the evaluated model.

## Sources

- [BlenderGym](https://github.com/richard-guyunqi/BlenderGym-Open), pinned to `67c8b48bf8c548c701b6873db6fcc6527d756385`.
- [Benchmark data](https://huggingface.co/datasets/richard-guyunqi/BG_bench_data).
- [Custom Infinigen](https://github.com/richard-guyunqi/infinigen), pinned to `5d998a433f327cb27656bcf19ba557d614fe5f9b`.
- Blender 3.6.0.

The current release has 280 task directories, rather than the paper's 245. Lighting IDs 26–30 are absent. Task discovery uses actual files, not consecutive-ID assumptions.

Datasets, weights and large experiment artifacts are stored outside this repository. This repository does not redistribute third-party scene assets or models.

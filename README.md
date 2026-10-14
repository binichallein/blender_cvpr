# BlenderGym model evaluation

Reproducible, non-agentic evaluation of Qwen vision-language models on BlenderGym.

The initial experiment completed the fixed 50-task screen on ml2 with vLLM continuous batching. Qwen3-VL-8B-Instruct and Qwen3-VL-32B-Instruct each achieved 2/50 under the frozen derived visual-success rule. Neither reached the 11/50 gate, so that screening experiment did not expand to a full run.

See the [full report, category breakdown and per-task results](results/2026-10-15-vllm/README.md). These are one-shot results under our calibrated rule, not an official BlenderGym binary score.

See [the frozen protocol](docs/protocol.md), [fixed task selection and audits](docs/subset.json), and [running instructions](docs/run.md).

A separate full-set experiment is now running on ml2: Qwen3-VL-8B and 32B (official Instruct and Thinking checkpoints), plus Qwen3.6-27B with native thinking enabled and disabled. All six variants receive 32,768 output tokens and all 280 tasks, without a screening gate. See the [full comparison plan](docs/full-thinking-plan.md) and [frozen configuration](docs/protocol-full-thinking.json). No results from this new experiment are claimed yet.

## Initial screening protocol

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

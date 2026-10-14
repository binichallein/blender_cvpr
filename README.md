# BlenderGym model evaluation

Reproducible, non-agentic evaluation of Qwen3-VL Instruct models on BlenderGym.

Status: calibration is frozen; final reference preflight is in progress. No model scores are claimed yet.

See [the frozen protocol](docs/protocol.md), [fixed task selection and audits](docs/subset.json), and [running instructions](docs/run.md).

## Protocol

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

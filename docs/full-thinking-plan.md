# Full thinking comparison

Run all six variants on all 280 released tasks. The previous 11/50 gate is explicitly superseded by the user's full-evaluation request. Reuse frozen visual thresholds and reference renders, but generate fresh responses for every variant/task. The original 50-task experiment remains unchanged.

- Qwen3-VL-8B and 32B: compare official Instruct and Thinking checkpoints.
- Qwen3.6-27B: one multimodal checkpoint with enable_thinking=False / True.
- User-selected output cap: 32,768 total tokens, including reasoning and final code.
- Common decoding: temperature 0.6, top_p 0.95, top_k 20, fixed per-request seed 20261015, one response, no repairs.
- The new sampling and token budget make this a separate experiment from the earlier greedy 8,192-token screen.
- Native templates enable thinking; no /think or /nothink prompt substitution.
- Decode and preserve reasoning separately. Only the final answer after the reasoning terminator may be parsed as a candidate program.
- Preserve every model format, truncation and execution failure in the denominator. Reference/data problems must be audited independently and applied equally to all variants.
- Report all released tasks, scorable tasks, the original screening subset, new tasks and scores excluding calibration tasks.
- Use vLLM continuous batching, initially 16 active sequences / 32 outstanding requests, BF16, context capacity 49,152. Check generic smoke inputs before running benchmark requests.
- All runtime files, weights and images stay on ml2; publish code, configurations and compact results to GitHub.

Official references: [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), [Qwen3-VL-8B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking), [Qwen3-VL-32B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking).


## Running and resuming

Run reference scenes for all tasks with `render_batch.py --mode references --split all`, then start `run_full.py` with the manifest, frozen protocol, output root, Blender and Infinigen paths, CLIP weights, reference root, model weights root, inference Python, and reference-process PID. The coordinator allocates GPUs 0–1 to rendering once reference rendering finishes, and GPUs 2–3 to inference. It requires four visible GPUs; 32B variants use tensor parallelism across the two inference GPUs.

Use a short local IPC path such as `/dev/shm/bg-full` for vLLM; a deep project path can exceed the Unix-domain socket path limit. Runtime status and process IDs are recorded in `status.json` and each variant's `source.json`. A singleton file lock prevents concurrent coordinators. A resumed coordinator adopts still-running inference only when PID, command, variant and output path match.

Completed outputs are saved before parsing and reused on resume. Infrastructure failures stop that variant, while malformed or truncated model answers are retained as model failures. ModelScope downloads become ready only after all indexed weight shards and required tokenizer/processor files exist.

After all six variants are scored, `report_full.py` exports the report, per-task CSV, generation statistics, paired outcomes, configuration and model file provenance. Large artifacts remain outside Git.


## Reference image audit added during the run

Before seeing any full-model scores, a reference-only check found that some supplied images differ from renders of the released scene and reference program. Lighting24 views 3–4 visibly differ; camera restoration changes the world matrices by only about 1e-6 and does not explain the discrepancy. All model inputs and frozen main scoring rules remain unchanged.

A supplementary, explicitly post-hoc sensitivity analysis flags tasks if any supplied start/goal view has RGB MSE greater than 0.005 against its corresponding reference render, after Lanczos resizing to 512 by 512. This audit reads no model answers. Flags can reflect noise, settings or scene/camera differences and are not automatically declared invalid data. Report the original main scores plus scores excluding flagged tasks.

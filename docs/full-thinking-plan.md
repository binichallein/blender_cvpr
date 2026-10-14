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

# Qwen3.5-122B-A10B full evaluation

Both native thinking modes use the same unquantized BF16 checkpoint with four-way tensor parallelism on ml2's 4 x A100 80GB. All code, weights, predictions, and renders remain on ml2. Only code and compact reports are published.

- Each mode generates one response for each of the same 280 tasks. No screening gate, tools, repair loop, or answer selection.
- Reuse the prior inputs, 49152-token context capacity, 32768 total output-token budget including reasoning, sampling parameters, references, renderer, frozen category thresholds, and invalid-task exclusions. Expected main denominator: 277.
- Use vLLM AsyncLLM continuous batching. Run nothink then thinking with all four GPUs; render after inference releases the GPUs.
- Validate the native chat template and eight synthetic color-image requests before benchmark generation. Persist raw completions before parsing; preserve every completed answer on resume.
- Export per-task results, category scores, truncation and execution errors, token usage, timing, model file hashes, and paired success differences.
- Independently recompute float64 pixel losses and success decisions, and validate all 560 input/output records before publishing scores.

Protocol: [protocol-qwen35-122b.json](protocol-qwen35-122b.json).
Model: https://huggingface.co/Qwen/Qwen3.5-122B-A10B
The binary success rule is our frozen operational metric, not an official BlenderGym pass/fail score.

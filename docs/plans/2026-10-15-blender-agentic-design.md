# Blender tool agent design

Approved choice: a Blender-specific tool agent; Qwen3.6-27B native thinking on/off; all 280 tasks per mode. User chose 65536 generated tokens per entire trajectory and at most five Blender executions.

## Interface and selection
The model receives the same original start program and all paired released start/target views. Two native Qwen tools: run_blender(code), which runs a complete replacement script in a fresh base scene and returns errors or all fixed-camera renders; submit(attempt_id), which selects a previously executed version. No verifier, similarity score, goal program, reference rerender, or thresholds are model-visible. Main score uses the selected version. On budget exhaustion or absent submission, use the most recent attempted script (even if it failed); never select by correctness.

## Budgets
65536 total generated tokens including reasoning/tool syntax, maximum 32768 per response, five execution attempts (failed execution counts), 12 model responses. Both modes share these limits and sampling settings from the one-shot protocol. Generation truncation consumes its tokens and yields an explicit budget warning, allowing recovery within the remaining total. Context capacity 98304; preserve text and reasoning, but retain only the latest tool observation images plus original paired images. No opaque summarizer.

## Execution isolation
Run generated scripts inside disposable network-disabled Docker containers with dropped capabilities, no-new-privileges, read-only root, resource limits, and only per-task base.blend/code, renderer runtime, Infinigen dependencies, and writable output mounted. Never mount the dataset directory, goals, score files, credentials, repository, or host root. Use the same Blender 3.6/Cycles CUDA 512x512, 512 samples and seed as the baseline. Restore cameras. The sandbox is benchmark answer isolation, not a claim of protection against kernel exploits.

## Scheduling and records
vLLM AsyncLLM, BF16, TP1. One replica per mode on two GPUs, two shared render GPUs coordinated with file locks. Continuous task replenishment while render feedback is awaited. Persist every raw completion before parsing, every model-visible input, tool arguments/results and images, budgets and selection. Resume completed turns and render caches without regenerating an answer.

## Reporting
Reuse the frozen category rules and invalid-task exclusions (277 scorable tasks), calibrated on the earlier run. Report final-selection scores, per-category results, failure types, tokens, tool usage and paired one-shot changes. Explicitly state the larger agentic token budget prevents attributing all changes solely to feedback. Audit score decisions independently in float64; export compact provenance and results to GitHub. Leave 122B download running; resume its suspended GPU pipeline after agentic releases GPUs.

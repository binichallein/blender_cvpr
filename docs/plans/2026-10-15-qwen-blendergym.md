# Qwen BlenderGym Evaluation Implementation Plan


**Goal:** Evaluate Qwen3-VL-8B-Instruct and Qwen3-VL-32B-Instruct on the same 50 tasks (10 from each of five categories); evaluate a model on the complete available benchmark only if it succeeds on at least 11 subset tasks.

**Architecture:** An isolated evaluation harness next to the untouched official repository. One-shot multimodal code generation, followed by trusted rerendering and objective image metrics. Freeze subset selection and calibrated success criteria before model inference; persist every request, response, execution result and score for resumability.

**Tech Stack:** Python, Qwen3-VL via Transformers or vLLM, Blender 3.6.0 and the official custom Infinigen fork, PyTorch CLIP, SSH ml2.

User confirmed both VL Instruct model names and a reference/control-calibrated success definition. No Harbor conversion. No model training. Model-independent setup and protocol calibration precede model results.

## Task 1: Resources and inputs

- Record GPU, CPU, disk, versions, source revisions, and model shard inventory.
- Store all large files on a shared remote filesystem, because root disk is nearly full.
- Download the official archive on the remote machine, validate ZIP and SHA-256, extract safely and create immutable task inventory.
- Select 10 tasks per category with a documented deterministic, base-scene-aware rule, avoiding missing lighting IDs.

## Task 2: Reproducible rendering

- Install Blender 3.6.0 and the benchmark's custom Infinigen dependencies in isolated paths.
- Run reference start/goal renders from each category; fix infrastructure defects with minimal reproductions.
- Freeze views, resolution, samples, seed, camera handling, timeouts and failure policy.
- Preserve original scripts and scene assets. Never send goal scripts to model.

## Task 3: Calibration and checks

- Establish numerical rerender noise with reference controls and controlled wrong/near-goal scenes.
- Select and freeze an explicitly named approximate visual-success rule, documenting its limits and control outcomes.
- Validate scorer on identical images, deliberately wrong outputs, missing views, invalid code, and protocol gating at 10 vs 11 successes.
- Save original PL and N-CLIP separately from the derived binary success label; do not claim unavailable Chamfer results.

## Task 4: Model adapters and 50-task runs

- Use both models with identical one-shot prompts, all matched current/goal views, full start code, deterministic generation and one final script per task.
- Use shared model settings and sufficient output tokens, record truncation and parsing failures as failures; no model-driven repair or hidden goal access.
- Run jobs detached with per-task checkpoint files, command, PID and logs.
- Perform a multimodal interface smoke check before launching; do not tune protocol using task performance.
- Report each of the 50 tasks even on failure, with category counts, PL/N-CLIP, runtime and output artifacts.

## Task 5: Conditional full run and report

- For each model independently, run remaining available tasks only when subset success count >= 11.
- Reuse subset outcomes in the full result, with identical protocol. Report full and remaining-only scores because the trigger uses the subset.
- Validate manifest coverage and metric aggregation; distinguish model failures from unavailable infrastructure.
- Write a Chinese report and machine-readable results, retain artifacts on ml2, and state all remaining limitations.

Execution is already authorized by the user. Proceed within this scope without additional implementation approval; ask only if essential information is missing.

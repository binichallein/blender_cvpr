# Full Qwen3.6-27B Blender tool-agent evaluation

This run uses a Blender-specific native-tool agent, selected by the user. Both native thinking modes run all 280 released tasks using the same BF16 checkpoint and the earlier frozen calibration (277 scorable tasks).

The user selected **65536 output tokens per complete trajectory**, including reasoning and tool-call syntax, with at most **five executions**. Each response is capped at 32768 tokens, and a trajectory has at most 12 assistant responses. The earlier one-shot baseline used 32768 tokens, so score changes combine feedback and increased budget.

Tools: run_blender(code) executes a complete replacement script in a fresh base scene and returns execution errors or fixed-camera images; submit(attempt_id) selects a previous attempt. The final submitted version is scored. Without a submission, use the latest attempted script, including failures. Never choose an answer using a hidden score. Invalid tool calls use response/token budget; failed Blender executions use execution budget.

The model receives all paired released input images and start.py. All text and reasoning history is retained; only the latest tool image set is retained alongside the original paired images. No goal.py, hidden reference render, score threshold, numerical score, or helper VLM enters the conversation.

Execution uses network-disabled Docker containers, a read-only root, no capabilities, a non-root user, and per-task mounts. Blender 3.6/CUDA Cycles uses the same 512x512, 512 samples, seed and cached A100 kernel as the baseline. Container device mapping resolves GPU UUIDs to actual Linux device minors on ml2.

vLLM AsyncLLM uses continuous batching, one TP1 replica per mode on GPUs 0 and 1, with GPUs 2 and 3 sharing the render queue. Responses are saved before tool parsing. Resume reuses completed responses and render artifacts; infrastructure failures stop the run rather than scoring subsequent tasks as wrong.

Preflight passed both modes' real native-tool/image-feedback smoke tests, five-category render parity (max per-view MSE 5.79e-9), and sandbox visibility/read-only checks. Unit and fault-injection tests cover budget limits, selecting earlier attempts, fallback to failed latest attempts, and recovery without regenerated answers.

## Operation on ml2

All artifacts: /limx_embap/tos/user/Yaleon/projects/blender-research/eval-agentic-20261015.

Launch or resume:

    /root/miniconda3/bin/python scripts/run_agentic.py --config /limx_embap/tos/user/Yaleon/projects/blender-research/eval-agentic-20261015/run-config.json

The coordinator saves status.json, per-mode logs and full traces, then runs complete scoring and independent float64/final-selection audits. The 122B download continues; its GPU pipeline resumes after agentic completion. Results are exported to results/2026-10-15-agentic-qwen36 for review and publication.

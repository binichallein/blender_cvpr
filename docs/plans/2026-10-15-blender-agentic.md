# Blender Tool Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run and audit full Qwen3.6-27B thinking/non-thinking BlenderGym tool-agent evaluations.
**Architecture:** Native Qwen tool calls through vLLM AsyncLLM, a resumable per-task state machine, isolated Docker Blender execution, and host-only scoring.
**Tech Stack:** Python, vLLM 0.19, Blender 3.6, Docker, existing NumPy/CLIP scoring.

---

### Task 1: Isolated rendering
Create scripts/agentic_sandbox.py and tests/test_agentic.py. Build deterministic mount/argument construction, validate paths and cache provenance, retain full local logs but expose bounded execution errors. Test no dataset/goal mounts and exact candidate selection; verify CUDA and all five task import families in existing container runtime.

### Task 2: Tool protocol and budgets
Create scripts/agentic_core.py with two schemas, agent prompt, native Qwen tool parser adapter, trajectory accounting, image-history policy, final selection. Add tests for malformed calls, reasoning boundaries, exhaustion, no oracle fallback, and missing artifacts.

### Task 3: Inference driver
Create scripts/infer_agentic.py. Reuse native tokenizer and AsyncLLM settings; persist each completion before parsing, render with shared GPU locks, recover turns deterministically. Smoke-test both thinking modes with synthetic image/tool observations before benchmark. Freeze protocol with script hashes.

### Task 4: Coordinator, audit and report
Create scripts/run_agentic.py and scripts/report_agentic.py. Start two replicas, wait for both, score complete task coverage using existing score.py, independently recompute pixel decisions and validate final-selection provenance. Export compact result data and comparison to the prior one-shot run. Resume Qwen122 only after both inference and scoring release resources.

### Task 5: Launch and completion
Run unit tests and compile checks; commit and push implementation. Launch detached coordinator on ml2 with persistent status/logs. Monitor real progress and resolve infrastructure problems without discarding model failures. On completion review/push audited results and document remaining limitations.

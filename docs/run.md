# Running on a remote GPU machine

Keep the official source trees, scene archive, weights and runtime outputs outside this repository. Use Blender 3.6.0 and the source revisions in README.md. The scoring environment uses Python 3.13, torch 2.9.1+cu129, transformers 5.5.0, NumPy 2.4.2 and Pillow 12.1.1. Inference uses a separate Python 3.12 environment with vLLM 0.19.0, torch 2.10.0+cu129 and transformers 4.57.6. Set VLLM_PYTHON to that environment’s Python executable. Blender uses its own Python 3.10.

All commands below run on the remote machine. Set BENCH_ARCHIVE, RUN_DIR, BLENDER, INFINIGEN, CLIP_DIR, MODEL8 and MODEL32 to absolute paths. CLIP_DIR contains the public openai/clip-vit-base-patch32 config, preprocessor config and weights.

```bash
python scripts/prepare_data.py --archive "$BENCH_ARCHIVE" --out "$RUN_DIR/data"
"$BLENDER" --background --factory-startup --python scripts/audit_cameras.py -- --manifest "$RUN_DIR/data/manifest.json" --output "$RUN_DIR/data/cameras.json"
python scripts/apply_camera_audit.py --manifest "$RUN_DIR/data/manifest.json" --cameras "$RUN_DIR/data/cameras.json"
python -m unittest discover -s tests -v
python scripts/render_batch.py --manifest "$RUN_DIR/data/manifest.json" --output "$RUN_DIR/calibration" --blender "$BLENDER" --infinigen "$INFINIGEN" --mode calibration --gpus 2,3
python scripts/calibrate.py --manifest "$RUN_DIR/data/manifest.json" --renders "$RUN_DIR/calibration" --clip "$CLIP_DIR" --output "$RUN_DIR/protocol.json"
python scripts/render_batch.py --manifest "$RUN_DIR/data/manifest.json" --output "$RUN_DIR/references" --blender "$BLENDER" --infinigen "$INFINIGEN" --mode references --gpus 0,1
python scripts/run_experiment.py --model8 "$MODEL8" --model32 "$MODEL32" --manifest "$RUN_DIR/data/manifest.json" --protocol "$RUN_DIR/protocol.json" --output "$RUN_DIR/models-vllm" --backend vllm --inference-python "$VLLM_PYTHON" --blender "$BLENDER" --infinigen "$INFINIGEN" --clip "$CLIP_DIR" --references "$RUN_DIR/references"
```

Launch the experiment with nohup or a scheduler and redirect output to a persistent log. The coordinator runs 8B inference on GPU 2 and 32B inference on GPU 3, with candidate rendering on GPUs 0 and 1. Successful reference renders and completed generations are reused on resume. Cache identities include the candidate script and renderer hashes. Do not change protocol, prompts or generation settings during a run.

The scorer retains all model failures in the success denominator. Data problems are reported separately. Three release tasks have identical start and goal programs (blendshape73, geometry13, geometry44); the screening manifest replaces the two sampled instances before inference. Full-run reports show both released and scorable task counts, plus results excluding the calibration tasks.

Inference uses AsyncLLM with continuous batching: four active sequences and eight outstanding requests per model. Every completed request is saved immediately and its slot replenished, without waiting for a fixed batch. Rendering consumes these results concurrently. Infrastructure errors cancel outstanding requests and stop the split; model parsing/execution failures remain scored failures.

An earlier incomplete Transformers pilot is retained separately on the remote machine. The formal results use vLLM for both models, restarted from the same frozen 50 tasks. No accuracy-based threshold changes were made at the backend switch. The frozen protocol hash remains 38842a28676e90c5635729e5dd9ae7a517a4324aadf251ecc27f537bc3cf3468.

The complete release was also checked for input size without generating answers: the maximum is 8,331 tokens (geometry51). A conditional full run uses a 17,408-token context capacity, preserving the 8,192-token output cap, full images and starter code. Screening uses 16,384, sufficient for its maximum 6,718-token input. The full-stage configuration is saved separately as config-full.json; all shared generation settings must match the screening configuration.

A render-drain bug was fixed after generation: completed inference could be mistaken for incomplete work when some outputs arrived after the renderer had passed their position in its scan. Resuming reused all 100 generated responses and preserved prior execution failures. The report includes both initial-generation and resume source records. No model answers were regenerated.

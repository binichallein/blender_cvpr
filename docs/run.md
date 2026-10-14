# Running on a remote GPU machine

Keep the official source trees, scene archive, weights and runtime outputs outside this repository. Use Blender 3.6.0 and the source revisions in README.md. The inference environment uses Python 3.13, torch 2.9.1+cu129, transformers 5.5.0, accelerate 1.13.0, NumPy 2.4.2 and Pillow 12.1.1. Blender uses its own Python 3.10.

All commands below run on the remote machine. Set BENCH_ARCHIVE, RUN_DIR, BLENDER, INFINIGEN, CLIP_DIR, MODEL8 and MODEL32 to absolute paths. CLIP_DIR contains the public openai/clip-vit-base-patch32 config, preprocessor config and weights.

```bash
python scripts/prepare_data.py --archive "$BENCH_ARCHIVE" --out "$RUN_DIR/data"
"$BLENDER" --background --factory-startup --python scripts/audit_cameras.py -- --manifest "$RUN_DIR/data/manifest.json" --output "$RUN_DIR/data/cameras.json"
python scripts/apply_camera_audit.py --manifest "$RUN_DIR/data/manifest.json" --cameras "$RUN_DIR/data/cameras.json"
python -m unittest discover -s tests -v
python scripts/render_batch.py --manifest "$RUN_DIR/data/manifest.json" --output "$RUN_DIR/calibration" --blender "$BLENDER" --infinigen "$INFINIGEN" --mode calibration --gpus 2,3
python scripts/calibrate.py --manifest "$RUN_DIR/data/manifest.json" --renders "$RUN_DIR/calibration" --clip "$CLIP_DIR" --output "$RUN_DIR/protocol.json"
python scripts/render_batch.py --manifest "$RUN_DIR/data/manifest.json" --output "$RUN_DIR/references" --blender "$BLENDER" --infinigen "$INFINIGEN" --mode references --gpus 0,1
python scripts/run_experiment.py --model8 "$MODEL8" --model32 "$MODEL32" --manifest "$RUN_DIR/data/manifest.json" --protocol "$RUN_DIR/protocol.json" --output "$RUN_DIR/models" --blender "$BLENDER" --infinigen "$INFINIGEN" --clip "$CLIP_DIR" --references "$RUN_DIR/references"
```

Launch the experiment with nohup or a scheduler and redirect output to a persistent log. The coordinator runs 8B inference on GPU 2 and 32B inference on GPU 3, with candidate rendering on GPUs 0 and 1. Successful reference renders and completed generations are reused on resume. Cache identities include the candidate script and renderer hashes. Do not change protocol, prompts or generation settings during a run.

The scorer retains all model failures in the success denominator. Data problems are reported separately. Three release tasks have identical start and goal programs (blendshape73, geometry13, geometry44); the screening manifest replaces the two sampled instances before inference. Full-run reports show both released and scorable task counts, plus results excluding the calibration tasks.

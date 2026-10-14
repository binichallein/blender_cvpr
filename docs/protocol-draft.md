# One-shot evaluation protocol (calibration pending)

Models: Qwen3-VL-8B-Instruct and Qwen3-VL-32B-Instruct, BF16, greedy decoding, 8192 output tokens, one attempt. No model tools, retries or code repairs.

The input contains the complete start script and all available matched current/target image pairs, labeled by camera. The output is one complete Python program. Hidden goal programs are used only by the evaluator.

The release contains 280 tasks: 85 blendshape, 55 geometry, 45 lighting, 45 material, 50 placement. Lighting IDs 26–30 are absent. A deterministic seed selects 10 tasks per category, round-robin over shuffled base scenes. Two other tasks per category supply calibration controls.

Evaluation loads each shared base scene afresh, executes the program, restores evaluation cameras, and renders all provided views at 512×512, Cycles CUDA, 512 samples, seed 20261015. Per-view RGB MSE (PL) and CLIP ViT-B/32 cosine distance (N-CLIP) are averaged per task. No Chamfer score is implemented.

Calibration uses goal rerenders, same-seed repeats to measure the actual evaluation noise and a second seed as a diagnostic, 95%-goal near controls, 50%-goal wrong controls, and unchanged-start controls. Exact control construction is in calibration_controls.py. We will freeze a category-specific visual-success rule after checking control separation and before any benchmark model predictions. This derived binary metric is not an official BlenderGym score.

Every selected task remains in the denominator. Model parsing, truncation, execution and missing-view failures are failures. Reference/environment failures block scoring until resolved. A model triggers evaluation on the remaining 230 tasks only after completing all 50 and achieving at least 11 successes.

The original project's multi-round search is not reproduced: this experiment measures one-shot image-conditioned program editing. Full results include the selected subset and are conditional on passing its screening gate.

CLIP preprocessing uses standard PIL RGB input with one rescaling step. The upstream helper converts to a [0,1] tensor and then invokes the default rescaling processor, so our N-CLIP implementation corrects that preprocessing issue. It is reported as standard N-CLIP, not a bitwise reproduction of upstream helper output. PL follows the same RGB MSE formula.

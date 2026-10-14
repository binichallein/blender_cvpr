# Frozen visual-success criterion

Frozen before benchmark model inference. The immutable JSON is [protocol.json](protocol.json).

For each task, let E be view-averaged RGB MSE to its goal rerender, E_start the initial-scene MSE, and f the category same-seed repeat floor. The normalized residual is max(0, E-f) / max(E_start-f, 1e-8). A valid execution succeeds if this residual is at most the category threshold.

| Category | Residual threshold | Approximate required MSE reduction |
|---|---:|---:|
| blendshape | 0.155908 | 84.41% |
| geometry | 0.185679 | 81.43% |
| lighting | 0.012521 | 98.75% |
| material | 0.009745 | 99.03% |
| placement | 0.308227 | 69.18% |

These are operational visual thresholds, not an official binary metric or human correctness labels. Only two calibration tasks per category support them. Standard N-CLIP is reported separately and does not determine success. See [calibration-audit.md](calibration-audit.md) for model-independent corrections to controls and the random-seed policy.

The 50-task subset excludes identical start/goal programs. Full-run reporting separates all released tasks, scorable tasks and calibration-excluded scores. Program errors and generation failures count against model success; data problems are listed separately.

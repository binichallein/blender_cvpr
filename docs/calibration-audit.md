# Calibration audit before model inference

The 50 screening tasks are unchanged. No benchmark model output was generated during calibration.

- blendshape6: start-to-goal PL was 0.0000251, below the second-seed reference difference of 0.0000515. Its parameter controls cannot reliably establish a visual correctness boundary. Replaced only this calibration task with blendshape1, the first unused non-subset task in numeric order.
- geometry30: a 2.5% wave-scale perturbation caused a large topology change. Visual inspection rejected it as a near-correct positive. The positive now uses wave scale 9.30001 versus target 9.3; the negative remains 6.975.
- Alternate-seed diagnostic noise was large in placement24 (PL 0.001372 versus start-to-goal 0.002737), so it does not represent the fixed-seed evaluation pipeline. The noise floor is instead measured by independent same-seed rerenders; alternate-seed renders are preserved separately. Changes below that floor are explicitly marked unresolved and never counted as successful unchanged outputs.
- Final calibration inputs and render logs remain in the remote experiment directory. The original geometry30 near control had relative PL 1.0741 versus 1.0445 for its negative, so its positive label was rejected; its construction is retained in commit 9ecfe93. The frozen protocol will record the final control measurements and category thresholds.

A full camera-name audit of all 38 base scenes found orphan render5.png files in lighting1–5: Camera5 is absent. Those five tasks use their three valid matched views (1–3) for both input and scoring. All other task view lists are unchanged. This correction occurred before any benchmark model output; calibrated thresholds and their JSON hash remain unchanged.

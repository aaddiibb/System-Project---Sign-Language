PHASE 8 HANDOFF — MOTION-S / SIGNVRSE

Best Phase-7 model: sc_point
Best self-conditioned variant: sc_point; selected probabilities: {'point': 0.15, 'span': 0.15}
Intervention: two-pass self-conditioned history corruption (not exact scheduled sampling).
Architecture changed: NO. Phase-3 changed: NO. Phase-6 length predictor changed: NO.

Oracle-length original -> best:
micro token accuracy 0.138210 -> 0.147933
normalized edit distance 0.850525 -> 0.838971
normalized feature MAE 0.856556 -> 0.847493
collapse 0.510861 -> 0.474658
repetition 0.474626 -> 0.468999

Late-sequence accuracy improved: True
First-error propagation improved: True
Corrupted-history robustness improved: True
Remaining weakness: EXPOSURE_BIAS_REDUCED
Recommended Phase 8: refine the best exposure-aware model; then test repetition/dynamics regularization

No skeleton/BVH conversion, MPJPE, or semantic sign-quality claim is supported.

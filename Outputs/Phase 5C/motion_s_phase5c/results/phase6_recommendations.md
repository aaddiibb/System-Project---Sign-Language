# Phase 6 recommendations

These recommendations are ranked from measured Phase-4C/5B/5C evidence. They are proposals only;
Phase 6 is not implemented here.

## 1. Explicit duration and stopping control — highest priority

- Add a separate motion-token length predictor or explicit stop/EOS head.
- Condition decoding on the predicted target length and impose a defensible length window.
- Diagnose EOS supervision and max-length termination using `eos_motion_relation.csv`.

Evidence: oracle signed frame bias is **555.817** and end-to-end bias is
**551.270** frames. This dominates the oracle/end-to-end difference.

## 2. Mitigate autoregressive exposure bias and collapse

- Evaluate scheduled sampling or an autoregressive curriculum.
- Compare with non-autoregressive/masked temporal token generation.
- Add repetition-aware validation diagnostics and sequence-level early stopping criteria.

Evidence: **728** oracle and **723** end-to-end sequences were
flagged for collapse, so ground-truth gloss alone does not solve the problem.

## 3. Hierarchical coarse-to-fine RVQ prediction

- Predict the base level first, then condition residual levels on the base trajectory.
- Reweight residual-level losses according to measured ablation gains.
- Use `rvq_level_ablation.csv`; performed in this run: **False**.

## 4. Motion-dynamics objectives

- Consider velocity/smoothness losses or temporal-consistency regularization in token/model space.
- Validate that improvements reduce collapse without suppressing legitimate fast hand articulation.
- Use empirical GT velocity percentiles instead of fixed arbitrary thresholds.

## 5. Phase-3 gloss improvements — lower priority

Oracle-to-end-to-end normalized-feature degradation is only **0.006420**, so Phase 3
is not the leading bottleneck. Revisit it after length and collapse failures improve.

## Unresolved representation requirement

The exact 668-D index layout and verified feature-to-joint/BVH inverse remain unresolved. Obtain the
provider preprocessing/inverse utility before adding MPJPE, skeleton rendering, or generated BVH.

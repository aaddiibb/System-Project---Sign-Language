PHASE 7 HANDOFF — MOTION-S / SIGNVRSE

1. Length predictor
Frozen Phase-3 T5 encoder (512 dimensions), attention-mask-aware mean pooling,
Linear(512,256), GELU, Dropout(0.1), Linear(256,1), trained on log1p token length.
Best checkpoint: length_model/best_model.pt; best epoch: 48. Only the MLP head was trained.

2. Data and selection
Locked Phase-1 fingerprint: e2acdf4d6dce91775239ad50be3866dbc6f5da94d940204b62456cdb77fc4b0d. Supported T<=512 counts: {'train': 9936, 'validation': 1243, 'test': 1243}.
Model selection and optional scalar calibration used validation only. Test was evaluated after freezing.
Training-derived inference bounds: [24,469]. Calibration selected: True.

3. Exact-length generation
The Phase-4C checkpoint remained frozen. PAD/BOS/EOS logits were excluded by taking argmax over IDs 0..511,
and generation ran for exactly the requested temporal length. Oracle length is diagnostic only.

4. Results
Validation GT-gloss length MAE: 12.046661 tokens.
Test GT-gloss length MAE: 12.587289 tokens.
Test predicted-gloss length MAE: 19.785197 tokens.
Original oracle duration MAE: 590.874497 frames.
Oracle-length duration MAE: 1.490748 frames.
Predicted-length end-to-end duration MAE: 79.077233 frames.
Original oracle collapse rate: 0.585680; oracle-length: 0.510861.
Original oracle repetition: 0.498969; oracle-length: 0.474626.

5. Scientific interpretation
Length-only duration fraction removed: 0.997477.
Dominant remaining bottleneck: EXPOSURE_BIAS.
Recommended Phase 7: target autoregressive exposure bias.
Do not claim semantic sign quality, compute MPJPE, construct a skeleton, or export BVH until the exact 668-D
layout and provider inverse are verified.

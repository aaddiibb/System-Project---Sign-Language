PHASE 5C INPUT CONTRACT — MOTION-S / SIGNVRSE

1. Verified RVQ decoder architecture
Candidate B strict-loaded from the official rvq_vae_best.pth:
ConvTranspose1d 256->512 (k=3,s=2,p=1,op=1), ReLU, BatchNorm1d;
ConvTranspose1d 512->512 (k=3,s=2,p=1,op=1), ReLU, BatchNorm1d;
ConvTranspose1d 512->512 (k=3,s=1,p=1,op=0), ReLU, BatchNorm1d;
ConvTranspose1d 512->668 (k=3,s=1,p=1,op=0), no final activation.

2. Token shape
[6,T], integer IDs 0..511. One selected [256]-vector from each codebook is summed.

3. Continuous motion output
[4*T,668], float32. Generated durations are untrimmed.

4. Normalization
Decoder output is inverse transformed with empirically recovered Motion-S feature normalization:
motion = normalized_motion * safe_std + mean. For std < 1e-6, safe_std=1.0.
These are not claimed to be official provider statistics.

5. Decoded locations
Oracle: /kaggle/working/motion_s_phase5/decoded/oracle
End-to-end: /kaggle/working/motion_s_phase5/decoded/end2end

6. Ground-truth Motion-Features
/kaggle/input/competitions/motion-s-hierarchical-text-to-motion-generation-for-sign-language/Motion-Features

7. Qualitative sample IDs
3299621, 3907151, 7236920, 310984, 6665221, 942487, 5936652, 3603901, 1187771, 244607, 283953, 4664643, 244757, 4497772, 2291880

8. Quantitative findings
Oracle duration MAE: 590.874497 frames
End-to-end duration MAE: 592.197104 frames
Oracle normalized feature MAE: 0.931877
End-to-end normalized feature MAE: 0.938296
Oracle velocity MAE: 0.096447
End-to-end velocity MAE: 0.095699

9. Collapse / instability findings
Motion-collapse flags — oracle: 728, end-to-end: 723
Extreme-motion flags — oracle: 43, end-to-end: 45
Thresholds are empirical ground-truth test-distribution percentiles; inspect diagnostics CSVs.

10. Explicit unresolved item
The semantic layout / joint conversion of the 668-dimensional Motion-S representation must be
verified before skeleton/BVH visualization.

PHASE 5B INPUT CONTRACT — MOTION-S / SIGNVRSE

Required artifacts:
/kaggle/working/motion_s_phase5/diagnostics/decoder_contract.json
/kaggle/working/motion_s_phase5/diagnostics/motion_mean_668.npy
/kaggle/working/motion_s_phase5/diagnostics/motion_std_668.npy

Reconstruct the verified Candidate-B decoder and strict-load decoder weights from:
/kaggle/input/models/antonygithinji/motion-s-vae-rvq/pytorch/default/3/rvq_vae_best.pth

RVQ token input: [6,T]
Valid token IDs: 0..511
Latent after six codebook lookups and summation: [256,T]
Decoded normalized motion: [4T,668]
Final inverse-normalized continuous motion: [4T,668]

Normalization is an empirically recovered per-feature z-score computed from every valid
Motion-Features file. It is not claimed to be official provider normalization.

Ground-truth reconstruction may trim only the decoder's padded tail when the original frame count
is known. Do not trim Phase-4 generated sequences.

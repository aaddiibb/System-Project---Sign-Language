PHASE 5 INPUT CONTRACT — MOTION-S / SIGNVRSE

Best Phase-4 checkpoint: /kaggle/working/motion_s_phase4/best_model/model.pt
Architecture: frozen Phase-3 T5-small encoder; 512->256 projection; six Embedding(515,64) tables;
Linear(384->256); four-layer causal Transformer decoder; d_model=256, 8 heads, FFN=1024;
six independent Linear(256->515) heads. Dropout was 0.1 during full training.
Special IDs (model only): PAD=512, BOS=513, EOS=514.
RVQ level order: ['base_tokens', 'residual_1', 'residual_2', 'residual_3', 'residual_4', 'residual_5']
Each real codebook size: 512. Valid IDs sent to RVQ-VAE: 0..511 only.
Canonical generated sequence shape: [6,T]. Supported maximum T: 512.
T>512 samples were not truncated; they are listed in diagnostics/phase4c_long_sequence_policy.csv.
Phase-1 fingerprint: e2acdf4d6dce91775239ad50be3866dbc6f5da94d940204b62456cdb77fc4b0d
Frozen encoder dependency: /kaggle/input/datasets/adibraian/phase-3/motion_s_phase3/best_model
Tokenizer dependency: /kaggle/input/datasets/adibraian/phase-3/motion_s_phase3/tokenizer

Reload: instantiate GlossMotionTransformer(load_frozen_encoder()), load best_model/model.pt,
then load motion_state_dict with strict=False; only gloss_encoder.* keys may be missing.
gloss_to_motion_tokens(): after reload, call
output = gloss_to_motion_tokens(model, "ME WANT WATER DRINK//"); tokens = output["tokens"].
Pass an already-produced gloss WITHOUT a 'text to gloss:' prefix. The function starts all six levels
with BOS, suppresses PAD/BOS, uses synchronized EOS, and returns only IDs 0..511.

NPZ format: sample_ids[N], lengths[N], offsets[N+1], tokens_flat[6,sum(lengths)].
For sample i: tokens_flat[:, offsets[i]:offsets[i+1]] has shape [6,lengths[i]].
Oracle Phase-5 test decoding input: results/test_oracle_tokens.npz
End-to-end Phase-5 test decoding input: results/test_end2end_tokens.npz

Official Motion-S RVQ-VAE decoder/checkpoint: not embedded here. Use the exact provider artifact
identified by the Motion-S competition/data source and verify its six 512-entry codebooks before decoding.
Never pass PAD/BOS/EOS to that decoder.

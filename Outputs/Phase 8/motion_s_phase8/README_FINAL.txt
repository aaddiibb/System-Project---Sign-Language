MOTION-S FINAL SYSTEM

Final generator: dynamics
Checkpoint: final_model/best_motion_generator.pt
Phase-3 model/tokenizer: /kaggle/input/datasets/adibraian/phase-3/motion_s_phase3/best_model / /kaggle/input/datasets/adibraian/phase-3/motion_s_phase3/tokenizer
Length predictor: /kaggle/input/datasets/adibraian/phase-6-output/motion_s_phase6/length_model/best_model.pt, calibrated bounds [24,469]
RVQ decoder: /kaggle/input/models/antonygithinji/motion-s-vae-rvq/pytorch/default/3/rvq_vae_best.pth
Normalization: /kaggle/input/datasets/adibraian/phase-5b-output/motion_s_phase5/diagnostics/motion_mean_668.npy and /kaggle/input/datasets/adibraian/phase-5b-output/motion_s_phase5/diagnostics/motion_std_668.npy

EXACT RELOAD ORDER
1. Load the Phase-3 tokenizer and AutoModelForSeq2SeqLM from the manifest paths with local_files_only=True; set eval() and freeze it.
2. Construct the unchanged Phase-4/7 MotionModel, attach the Phase-3 encoder, then load final_model/best_motion_generator.pt['motion_state_dict']; set eval() and freeze it.
3. Construct the unchanged Phase-6 LengthHead, load length_model/best_model.pt['length_head_state_dict'], and apply the calibration and [24,469] bounds recorded in its config.json.
4. Construct the verified MotionDecoder, load decoder.* tensors from rvq_vae_best.pth, and use the six rvq.quantizers.q.embedding tensors without modification.
5. Load motion_mean_668.npy and motion_std_668.npy for decoder de-normalization.
6. Import final_pipeline/generate_motion.py, call install_pipeline(text_to_gloss, predict_motion_length, generate_rvq_tokens, decode_motion_s_tokens), then call generate_motion(...).

EXECUTED NOTEBOOK EXAMPLE
result = generate_motion("I want water", return_intermediates=True)
Expected: gloss string; integer T; rvq_tokens int64 [6,T] IDs 0..511; motion float32 [4T,668].

The notebook has already installed and validated these frozen functions. model_manifest.json records paths, hashes, IDs, versions, and shape contracts. No verified skeleton/BVH inverse exists.

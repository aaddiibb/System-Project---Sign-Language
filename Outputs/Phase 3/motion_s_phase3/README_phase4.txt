# Phase 4 hand-off

- Locked Phase-1 fingerprint: `e2acdf4d6dce91775239ad50be3866dbc6f5da94d940204b62456cdb77fc4b0d`
- For Phase-4 **training**, join the saved Phase-1 training CSV to motion resources and use its
  ground-truth `gloss` / `model_gloss` target. Do not replace training glosses with Phase-3 predictions.
- For end-to-end inference, call `predict_gloss(text)["predicted_gloss"]` and pass that value to Phase 4.
- Validation/test Phase-3 predictions are saved only for evaluation, analysis, and optional end-to-end testing.
- Best text-to-gloss model: `/kaggle/working/motion_s_phase3/best_model`
- Tokenizer: `/kaggle/working/motion_s_phase3/tokenizer`
- Selected decoder: `beam_4`

Fresh-notebook reload:

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/kaggle/working/motion_s_phase3/tokenizer", local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained("/kaggle/working/motion_s_phase3/best_model", local_files_only=True)

Apply the exact `text to gloss: ` prefix and the lengths recorded in `phase3_config.json`.
Phase 4 does not need to load this model while learning gloss-to-motion; load it only for
end-to-end English-to-motion inference or held-out pipeline evaluation.

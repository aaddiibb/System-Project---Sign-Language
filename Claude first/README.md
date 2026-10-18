# Motion-S Kaggle Pipeline — Quick Reference

## Files
| File | Purpose |
|------|---------|
| `motion_s_pipeline.py` | Full pipeline — paste into a Kaggle notebook |
| `motion_s_NOTEBOOK.ipynb` | Ready-to-upload notebook wrapper |

---

## Architecture at a Glance

```
TEXT INPUT
    │
    ▼
┌─────────────────────────────┐
│  Semantic Retriever         │  TF-IDF + SBERT dual-encoder
│  (Tier 1)                   │  Score ≥ 0.75 → use blended retrieval
└────────────────┬────────────┘
                 │ score < 0.75
                 ▼
┌─────────────────────────────┐
│  Progressive Transformer    │  Text → Pose seq2seq
│  (Tier 2)                   │  + velocity loss + stop token
└────────────────┬────────────┘
                 │ if 0.4 ≤ score < 0.75
                 │ blend retrieval + generation
                 ▼
┌─────────────────────────────┐
│  Diffusion Refiner          │  Optional, GPU only
│  (Tier 3)                   │  DDPM on pose latent space
└────────────────┬────────────┘
                 ▼
┌─────────────────────────────┐
│  Post-processing            │  Root normalise + cosine smooth
└────────────────┬────────────┘
                 ▼
         submission/*.npy
         submission.csv
```

---

## What each tier contributes

### Tier 1 — Semantic Retrieval
- **From public notebook**: `sigmaborov/motion-s-text-to-sign-tf-idf-knn`
- **Upgraded**: Added Sentence-BERT dense encoder alongside TF-IDF sparse encoder
- **Blending**: Weighted sum of top-k retrieved poses by similarity score
- **Best for**: Sentences that repeat or paraphrase training data

### Tier 2 — Progressive Transformer
- **From**: Saunders et al. ECCV 2020 + SLRTP2025 winners
- **Improvements over baseline**:
  - Pre-LN (norm_first=True) for stable deep training
  - Velocity loss → smooth motion
  - Stop token → variable-length generation
  - Mixed precision (AMP) training
  - Cosine LR schedule
- **Best for**: Unseen sentence structures

### Tier 2.5 — Blended zone
- When retrieval score is 0.40–0.75, linearly interpolate
- Acts as a soft ensemble

### Tier 3 — Diffusion Refiner (GPU only)
- **From**: Sign-IDD (AAAI 2025), G2P-DDM (AAAI 2024)
- **Architecture**: Lightweight 1-D convolutional Transformer denoiser
- **Mode**: Partial diffusion (add noise at t=30 → denoise → refined output)
- **Best for**: Smoothing and improving realism of Transformer outputs

### Post-processing
- **Root normalisation**: Centres hip, scales to shoulder width → signer-agnostic
- **Cosine smoothing**: 5-frame Hann window → removes per-frame jitter

---

## Evaluation Metrics

| Metric | Formula | Direction |
|--------|---------|-----------|
| DTW-MJE | DTW alignment cost + mean joint error | ↓ lower better |
| BLEU-4 (back-translation) | BLEU-4 of sign→text vs original text | ↑ higher better |
| FID | Fréchet distance of pose distributions | ↓ lower better |

---

## How to run on Kaggle

```python
# Cell 1 — Install
!pip install -q sentence-transformers dtw-python sacrebleu transformers

# Cell 2 — Run
exec(open('/kaggle/input/YOUR_DATASET/motion_s_pipeline.py').read())
main()

# Cell 3 — Submit
# Zip the /kaggle/working/submission/ folder or use submission.csv
```

---

## Expected data layout

```
/kaggle/input/motion-s-hierarchical-text-to-motion-generation-for-sign-language/
    train.csv           — id, text, num_frames
    test.csv            — id, text
    train_poses/        — {id}.npy  shape (T, 128, 3)
```

Adjust `load_data()` and `load_pose()` if column names differ from the above.

---

## Key hyperparameters to tune

| Parameter | Default | Notes |
|-----------|---------|-------|
| `retrieval_threshold` | 0.75 | Raise → more retrieval; lower → more generation |
| `alpha` (retriever) | 0.4 | Balance TF-IDF vs SBERT |
| `d_model` | 256 | Increase to 512 on A100 |
| `num_enc/dec_layers` | 4 | Increase for more capacity |
| `lambda_vel` | 0.5 | Higher → smoother but may reduce BLEU |
| `smooth_window` | 5 | Higher → smoother but may blur fast signs |
| `diff_steps` | 100 | 50–200 range; lower = faster inference |

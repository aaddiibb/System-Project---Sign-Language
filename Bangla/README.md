# BdSL MediaPipe Extraction Pipeline

**Ban-Sign-Sent-9K → MediaPipe Holistic (SLRTP2025 Format)**

Extracts 543 body/hand/face landmarks from ~9,610 BdSL sign language videos across multiple Kaggle sessions.

---

## Pipeline Overview

```
Session 0                Sessions 1-7              Final Session
┌──────────────┐    ┌────────────────────┐    ┌──────────────────┐
│ Download &   │    │ MediaPipe Holistic │    │ Merge + Normalize│
│ Extract      │───▶│ Extraction         │───▶│ + Train/Val Split│
│ Split batches│    │ (~1,373 vids each) │    │                  │
└──────────────┘    └────────────────────┘    └──────────────────┘
  ~1-2 hours          ~8-10 hours × 7           ~1-2 hours
```

**Total: ~9 Kaggle sessions, ~60-75 hours compute**

---

## Session-by-Session Instructions

### Session 0: Download & Setup
**File:** `Session_0_Download_and_Setup.py`
**Accelerator:** CPU (no GPU needed)
**Time:** ~1-2 hours

1. Create new Kaggle Notebook
2. Paste the script into a code cell
3. Run — it downloads Ban-Sign-Sent-9K from HuggingFace
4. After completion, create a Kaggle Dataset from `/kaggle/working/bdsl_dataset/`
5. Name it: **`bdsl-ban-sign-sent-9k-extracted`**

**Output:**
- `videos/` — all extracted video files
- `metadata/master_videos.csv` — full video listing
- `metadata/batches/batch_0.csv` through `batch_6.csv` — processing splits

---

### Sessions 1-7: Batch Processing
**File:** `Session_1to7_Process_Batch.py`
**Accelerator:** CPU (MediaPipe runs on CPU)
**Time:** ~8-10 hours per session

For each session:

1. Create new Kaggle Notebook
2. **Add dataset:** `bdsl-ban-sign-sent-9k-extracted` (from Session 0)
3. **Change `BATCH_ID`** at the top of the script:
   | Session | BATCH_ID |
   |---------|----------|
   | 1       | 0        |
   | 2       | 1        |
   | 3       | 2        |
   | 4       | 3        |
   | 5       | 4        |
   | 6       | 5        |
   | 7       | 6        |
4. Paste the script and run
5. Create Kaggle Dataset from output: **`bdsl-batch-{BATCH_ID}-mediapipe`**

**Output per batch:**
- `poses/*.npy` — landmark arrays, shape `(T, 543, 3)`
- `batch_{ID}_mapping.csv` — sentence/signer → file mapping
- `batch_stats.json` — processing statistics
- `checkpoint.json` — for resuming interrupted sessions

**Landmark layout (543 total):**
| Index Range | Count | Body Part   |
|-------------|-------|-------------|
| 0-32        | 33    | Body pose   |
| 33-53       | 21    | Left hand   |
| 54-74       | 21    | Right hand  |
| 75-542      | 468   | Face mesh   |

---

### Final Session: Merge
**File:** `Session_Final_Merge.py`
**Accelerator:** CPU
**Time:** ~1-2 hours

1. Create new Kaggle Notebook
2. **Add ALL 8 datasets:**
   - `bdsl-ban-sign-sent-9k-extracted`
   - `bdsl-batch-0-mediapipe`
   - `bdsl-batch-1-mediapipe`
   - `bdsl-batch-2-mediapipe`
   - `bdsl-batch-3-mediapipe`
   - `bdsl-batch-4-mediapipe`
   - `bdsl-batch-5-mediapipe`
   - `bdsl-batch-6-mediapipe`
3. Paste the script and run
4. Create Kaggle Dataset: **`bdsl-mediapipe-final`**

**What it does:**
- Merges all 7 batch outputs
- Normalizes: neck at origin, scaled by shoulder width
- Creates signer-independent train/val split
- Validates shapes and ranges

**Final output:**
```
bdsl_final_dataset/
├── train/                    # ~8,000 .npy files
├── val/                      # ~1,600 .npy files
├── poses_normalized/         # All files (train + val)
├── train_manifest.csv
├── val_manifest.csv
└── dataset_metadata.json
```

---

## Using the Final Dataset

```python
import numpy as np
import os

# Load one sample
poses = np.load("train/sentence1_signer1.npy")  # (T, 543, 3)

# Access body parts
body_pose   = poses[:, 0:33, :]    # (T, 33, 3)
left_hand   = poses[:, 33:54, :]   # (T, 21, 3)
right_hand  = poses[:, 54:75, :]   # (T, 21, 3)
face_mesh   = poses[:, 75:543, :]  # (T, 468, 3)

# Upper body only (for sign language, legs aren't needed)
upper_body  = poses[:, :75, :]     # (T, 75, 3) — pose + both hands
```

---

## Troubleshooting

**"Batch CSV not found"**
→ Check your dataset name matches exactly. The script tries multiple paths.

**Session times out before finishing**
→ The script saves checkpoints every 50 videos. Re-run the same session — it skips already-processed videos automatically.

**Low hand detection rate (<50%)**
→ Normal for some videos where hands are occluded or out of frame. The pipeline interpolates missing frames.

**"No video files found" in Session 0**
→ The HuggingFace dataset structure may have changed. Check the download directory contents printed in the output and adjust `VIDEO_DIR`.

**Disk space issues**
→ Each batch output is ~2-4 GB. Make sure Kaggle working directory has space. You can delete checkpoint.json after uploading the dataset.

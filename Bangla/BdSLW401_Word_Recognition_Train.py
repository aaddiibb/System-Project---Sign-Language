"""
================================================================================
BdSL WORD RECOGNITION — Training notebook (Kaggle)
================================================================================
GOAL: train a model that takes the MediaPipe landmarks of ONE signed word and
      predicts which Bangla word it is. You then run the saved model locally on
      a webcam (see sign_to_word_webcam.py) to get: sign a word -> Bangla word.

WHAT RUNS WHERE:
  - THIS notebook (Kaggle, GPU): load BdSLW401 -> train classifier -> save model.
  - Webcam inference: runs on YOUR computer (Kaggle has no camera).

IMPORTANT — BdSLW401's exact on-disk format is not publicly documented, so this
notebook INSPECTS the dataset first (Step 1) and adapts (Step 2). If Step 2 finds
0 samples, read the Step 1 printout and tell me the structure — the loader is the
only part that depends on the dataset's exact layout.

FEATURE SPEC (identical here and in the webcam script — this MUST match):
  - MediaPipe Holistic: pose(33) + left_hand(21) + right_hand(21) = 75 points
    (face is NOT used for word recognition), each (x, y, z) -> 225 values/frame.
  - Per frame: center on shoulder midpoint, scale by shoulder width.
  - Each sign resampled to T_FIXED frames -> input shape (T_FIXED, 225).
================================================================================
"""

import os, sys, glob, json, time, math
import numpy as np

# ---------------------------------------------------------------- CONFIG
INPUT_BASE = "/kaggle/input"
WORK_DIR   = "/kaggle/working"
OUT_DIR    = os.path.join(WORK_DIR, "bdsl_word_model")
os.makedirs(OUT_DIR, exist_ok=True)

POSE_N, LH_N, RH_N = 33, 21, 21
N_PTS = POSE_N + LH_N + RH_N          # 75
FEAT  = N_PTS * 3                      # 225
T_FIXED = 48                          # frames per sign after resampling
L_SH, R_SH = 11, 12                   # shoulder indices within the pose block

# Your 34-word household vocabulary (used to FILTER BdSLW401 to a subset IF the
# dataset provides a Bangla word <-> sample mapping; otherwise we train on all
# classes found and you filter later). Top entries are the real vocab.
MY_WORDS = ["না","খেতেন","খাবে","দুপুরে","রাতে","বিকালে","সন্ধ্যায়","সকালে","কফি","চা",
            "মাংস","ডাল","ফল","শাক","সব্জী","খান","চাচা","মা","দাদী","দাদা","খেয়েছিল",
            "খায়নি","খালা","বাবা","চাচী","খালু","খেয়েছে"]
FILTER_TO_MY_WORDS = True   # set False to train on ALL words BdSLW401 provides
MAX_PER_CLASS = 400         # cap samples/class (keeps runtime + memory sane)

# ============================================================================
# STEP 1 — INSPECT the attached dataset (so we know what we're loading)
# ============================================================================
print("=" * 70); print("STEP 1: Inspecting /kaggle/input ..."); print("=" * 70)

def walk_summary(base, max_show=40):
    exts = {}
    sample_files = []
    for root, dirs, files in os.walk(base):
        for f in files:
            e = os.path.splitext(f)[1].lower()
            exts[e] = exts.get(e, 0) + 1
            if len(sample_files) < max_show:
                sample_files.append(os.path.join(root, f))
    return exts, sample_files

exts, samples = walk_summary(INPUT_BASE)
print("File-type counts under /kaggle/input:")
for e, c in sorted(exts.items(), key=lambda x: -x[1]):
    print(f"   {e or '<none>':10s} : {c}")
print("\nTop-level folders:")
for d in sorted(glob.glob(os.path.join(INPUT_BASE, "*"))):
    print("  ", d)
print("\nA few sample file paths:")
for s in samples[:15]:
    print("  ", s)

# ============================================================================
# STEP 2 — ADAPTIVE LOADER: landmark files if present, else video extraction
# ============================================================================
print("\n" + "=" * 70); print("STEP 2: Loading samples ..."); print("=" * 70)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
LAND_EXTS  = {".npy", ".csv", ".parquet", ".pkl", ".json"}

def label_from_path(path):
    """Guess the word/label for a sample from its folder/file name.
    BdSLW401 uses IDs like 'W043'. Common layouts: <word>/<clip>.ext or
    <WID>/<clip>.ext or <WID>_<signer>_...ext. Adjust here if Step 1 shows
    a different layout."""
    parts = os.path.normpath(path).split(os.sep)
    base  = os.path.splitext(parts[-1])[0]
    # try: a 'W\d+' token anywhere in the path
    import re
    for token in [base] + parts[::-1]:
        m = re.search(r"W\d{2,4}", token)
        if m:
            return m.group(0)
    # else: use the immediate parent folder name as the label
    return parts[-2] if len(parts) >= 2 else base

def coerce_landmarks(arr):
    """Coerce a loaded array to (T, 75, 3). Handles (T,225), (T,75,3),
    (T,543,3)->subset, (T,224) RQE-ish. Returns None if unusable."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        T, D = a.shape
        if D == FEAT:                 # (T, 225)
            return a.reshape(T, N_PTS, 3)
        if D == 224:                  # paper's 75*3-1 encoding; pad the dropped dim
            a = np.concatenate([a, np.zeros((T, 1), np.float32)], axis=1)
            return a.reshape(T, N_PTS, 3)
        if D == 543 * 3:              # full holistic flattened
            full = a.reshape(T, 543, 3)
            return holistic543_to_75(full)
    if a.ndim == 3:
        T, P, C = a.shape
        if P == N_PTS and C == 3:     # (T,75,3)
            return a
        if P == 543 and C == 3:       # (T,543,3) full holistic -> subset
            return holistic543_to_75(a)
    return None

def holistic543_to_75(full):
    """full (T,543,3): pose[0:33], lh[33:54], rh[54:75] (your own layout)."""
    return full[:, :75, :].astype(np.float32)

def load_landmark_file(path):
    e = os.path.splitext(path)[1].lower()
    try:
        if e == ".npy":
            arr = np.load(path, allow_pickle=True)
        elif e == ".csv":
            arr = np.loadtxt(path, delimiter=",")
        elif e == ".parquet":
            import pandas as pd; arr = pd.read_parquet(path).values
        elif e == ".json":
            arr = np.array(json.load(open(path)))
        elif e == ".pkl":
            import pickle; arr = pickle.load(open(path, "rb"))
        else:
            return None
        return coerce_landmarks(arr)
    except Exception:
        return None

# ---- shared feature pipeline (MUST match the webcam script) ----
def normalize_seq(seq):
    """seq (T,75,3) -> normalized (T,75,3): center on shoulder midpoint,
    scale by shoulder width, per frame. Missing (all-zero) points stay zero."""
    out = seq.copy().astype(np.float32)
    for t in range(out.shape[0]):
        ls, rs = out[t, L_SH], out[t, R_SH]
        if np.all(ls == 0) or np.all(rs == 0):
            continue
        neck = (ls + rs) / 2.0
        sd = np.linalg.norm(ls - rs) or 1.0
        m = ~np.all(out[t] == 0, axis=1)
        out[t][m] = (out[t][m] - neck) / sd
    return out

def resample_T(seq, T=T_FIXED):
    """Resample (T0,75,3) to (T,75,3) by linear interpolation over time."""
    T0 = seq.shape[0]
    if T0 == 0:
        return np.zeros((T, N_PTS, 3), np.float32)
    if T0 == T:
        return seq
    idx = np.linspace(0, T0 - 1, T)
    lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, T0 - 1)
    frac = (idx - lo)[:, None, None]
    return (seq[lo] * (1 - frac) + seq[hi] * frac).astype(np.float32)

def to_features(seq):
    return resample_T(normalize_seq(seq)).reshape(T_FIXED, FEAT)

# ---- MediaPipe extraction (only used if the dataset is VIDEO) ----
_holistic = None
def extract_from_video(path, stride=1):
    global _holistic
    try:
        import cv2, mediapipe as mp
    except Exception:
        os.system(f"{sys.executable} -m pip install -q mediapipe opencv-python-headless")
        import cv2, mediapipe as mp
    if _holistic is None:
        _holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False, model_complexity=1)
    cap = cv2.VideoCapture(path); frames = []; i = 0
    while True:
        ok, img = cap.read()
        if not ok: break
        if i % stride == 0:
            res = _holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            frames.append(landmarks_to_75(res))
        i += 1
    cap.release()
    return np.array(frames, np.float32) if frames else None

def landmarks_to_75(res):
    pts = np.zeros((N_PTS, 3), np.float32)
    if res.pose_landmarks:
        for j, lm in enumerate(res.pose_landmarks.landmark[:POSE_N]):
            pts[j] = [lm.x, lm.y, lm.z]
    if res.left_hand_landmarks:
        for j, lm in enumerate(res.left_hand_landmarks.landmark[:LH_N]):
            pts[POSE_N + j] = [lm.x, lm.y, lm.z]
    if res.right_hand_landmarks:
        for j, lm in enumerate(res.right_hand_landmarks.landmark[:RH_N]):
            pts[POSE_N + LH_N + j] = [lm.x, lm.y, lm.z]
    return pts

# ---- discover samples ----
all_files = []
for root, _, files in os.walk(INPUT_BASE):
    for f in files:
        all_files.append(os.path.join(root, f))

land_files  = [p for p in all_files if os.path.splitext(p)[1].lower() in LAND_EXTS]
video_files = [p for p in all_files if os.path.splitext(p)[1].lower() in VIDEO_EXTS]
# ignore obvious non-sample landmark files (mapping/readme csvs are usually few)
use_video = len(land_files) < 50 and len(video_files) > 50
print(f"  landmark files: {len(land_files)} | video files: {len(video_files)}")
print(f"  -> loading via {'VIDEO extraction' if use_video else 'landmark files'}")

# optional word-id filter: look for a word list / mapping file to map Bangla->WID
wid_filter = None
if FILTER_TO_MY_WORDS:
    map_candidates = [p for p in all_files
                      if os.path.splitext(p)[1].lower() in {".csv", ".txt", ".json"}
                      and any(k in os.path.basename(p).lower()
                              for k in ["word", "label", "class", "vocab", "map", "list"])]
    for mc in map_candidates:
        try:
            txt = open(mc, encoding="utf-8", errors="ignore").read()
            if any(w in txt for w in MY_WORDS):
                import re
                wid_filter = set()
                for line in txt.splitlines():
                    if any(w in line for w in MY_WORDS):
                        m = re.search(r"W\d{2,4}", line)
                        if m: wid_filter.add(m.group(0))
                if wid_filter:
                    print(f"  Found word mapping: {mc} -> {len(wid_filter)} target IDs")
                    break
        except Exception:
            pass
    if not wid_filter:
        print("  No Bangla word<->ID mapping found; training on ALL classes found.\n"
              "  (You can map your 34 words to IDs later, or set FILTER_TO_MY_WORDS=False.)")

# ---- load ----
X, y = [], []
per_class = {}
src = video_files if use_video else land_files
print(f"  Processing {len(src)} candidate samples ...")
for i, path in enumerate(src):
    lab = label_from_path(path)
    if wid_filter is not None and lab not in wid_filter:
        continue
    if per_class.get(lab, 0) >= MAX_PER_CLASS:
        continue
    seq = extract_from_video(path, stride=2) if use_video else load_landmark_file(path)
    if seq is None or seq.shape[0] < 4:
        continue
    X.append(to_features(seq)); y.append(lab)
    per_class[lab] = per_class.get(lab, 0) + 1
    if i % 500 == 0:
        print(f"    [{i}/{len(src)}] loaded={len(X)} classes={len(per_class)}")

print(f"\n  Loaded {len(X)} samples across {len(set(y))} classes.")
if len(X) == 0:
    raise SystemExit(
        "No samples loaded. Read the STEP 1 printout above and adjust "
        "label_from_path()/coerce_landmarks() to the real structure.")

X = np.stack(X).astype(np.float32)                      # (N, T_FIXED, FEAT)
classes = sorted(set(y))
cls2idx = {c: i for i, c in enumerate(classes)}
y_idx = np.array([cls2idx[c] for c in y], np.int64)
print(f"  X shape: {X.shape} | classes: {len(classes)}")

# ============================================================================
# STEP 3 — Train a small GRU word classifier (PyTorch)
# ============================================================================
print("\n" + "=" * 70); print("STEP 3: Training ..."); print("=" * 70)
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("  device:", dev)

# signer-independent-ish split would need signer ids; we do a stratified random split
rng = np.random.default_rng(42)
perm = rng.permutation(len(X))
n_val = max(1, int(0.15 * len(X)))
val_idx, tr_idx = perm[:n_val], perm[n_val:]

def loader(idx, bs, shuffle):
    ds = TensorDataset(torch.tensor(X[idx]), torch.tensor(y_idx[idx]))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)

tr = loader(tr_idx, 64, True); va = loader(val_idx, 128, False)

class GRUClassifier(nn.Module):
    def __init__(self, feat=FEAT, hid=256, n_cls=len(classes), p=0.3):
        super().__init__()
        self.gru = nn.GRU(feat, hid, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=p)
        self.head = nn.Sequential(nn.LayerNorm(hid*2), nn.Dropout(p),
                                  nn.Linear(hid*2, n_cls))
    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out.mean(dim=1))      # temporal mean-pool

model = GRUClassifier().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)

best_acc, best_state = 0.0, None
for epoch in range(60):
    model.train()
    for xb, yb in tr:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
    sched.step()
    # eval
    model.eval(); correct = tot = 0
    with torch.no_grad():
        for xb, yb in va:
            xb, yb = xb.to(dev), yb.to(dev)
            correct += (model(xb).argmax(1) == yb).sum().item(); tot += len(yb)
    acc = correct / max(1, tot)
    if acc > best_acc:
        best_acc = acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if epoch % 5 == 0 or epoch == 59:
        print(f"  epoch {epoch:2d}  val_acc={acc:.3f}  (best={best_acc:.3f})")

print(f"\n  BEST val accuracy: {best_acc:.3f}")

# ============================================================================
# STEP 4 — Save model + label map + feature config (for the webcam script)
# ============================================================================
print("\n" + "=" * 70); print("STEP 4: Saving ..."); print("=" * 70)
if best_state is not None:
    model.load_state_dict(best_state)
torch.save(model.state_dict(), os.path.join(OUT_DIR, "word_gru.pt"))
json.dump({"classes": classes,
           "feat": FEAT, "n_pts": N_PTS, "t_fixed": T_FIXED,
           "pose_n": POSE_N, "lh_n": LH_N, "rh_n": RH_N,
           "l_sh": L_SH, "r_sh": R_SH, "val_acc": best_acc},
          open(os.path.join(OUT_DIR, "config.json"), "w"), ensure_ascii=False, indent=2)
print(f"  Saved to {OUT_DIR}:  word_gru.pt , config.json")
print("  Download these two files and use them in sign_to_word_webcam.py.")
print("\nDONE.")

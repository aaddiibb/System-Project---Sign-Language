"""
================================================================================
BdSLW60 — NOTEBOOK 2 of 2 : TRAIN  (word recognition from saved landmarks)
================================================================================
PREREQ: run Notebook 1, save its output as a dataset (e.g. 'bdslw60-landmarks'),
and ATTACH that dataset to this notebook (Add Input -> Datasets).

WHAT THIS DOES:
  - Load the pre-extracted landmark .npy + labels.csv (fast, no video)
  - (optional) keep only the words your household app uses
  - Train a GRU word classifier
  - Save  word_gru.pt + config.json  -> use with sign_to_word_webcam.py on your PC

FEATURE SPEC (identical to Notebook 1 + the webcam script):
  pose(33)+left_hand(21)+right_hand(21)=75 pts; per-frame center on shoulders &
  scale by shoulder width; resample each clip to T_FIXED frames -> (T_FIXED,225).
================================================================================
"""

import os, glob, csv, json
import numpy as np

# ------------------------------------------------------------------ CONFIG
INPUT_BASE = "/kaggle/input"          # the attached bdslw60-landmarks dataset
WORK_DIR   = "/kaggle/working"
OUT_DIR    = os.path.join(WORK_DIR, "bdsl_word_model")
os.makedirs(OUT_DIR, exist_ok=True)

POSE_N, LH_N, RH_N = 33, 21, 21
N_PTS = POSE_N + LH_N + RH_N          # 75
FEAT  = N_PTS * 3                      # 225
T_FIXED = 48
L_SH, R_SH = 11, 12

# Your 34 household words. If BdSLW60's labels are in Bangla and overlap these,
# we keep only the overlap (a tighter, more accurate model). If labels don't
# match (e.g. English glosses / numeric IDs), we train on ALL words found.
MY_WORDS = ["না","খেতেন","খাবে","দুপুরে","রাতে","বিকালে","সন্ধ্যায়","সকালে","কফি","চা",
            "মাংস","ডাল","ফল","শাক","সব্জী","খান","চাচা","মা","দাদী","দাদা","খেয়েছিল",
            "খায়নি","খালা","বাবা","চাচী","খালু","খেয়েছে"]
FILTER_TO_MY_WORDS = True
EPOCHS = 60

# ============================================================================
# STEP 1 — Locate the landmark dataset (folder with labels.csv + *.npy)
# ============================================================================
print("=" * 70); print("STEP 1: Finding the landmark dataset ..."); print("=" * 70)

def find_dataset_root():
    for root, _, files in os.walk(INPUT_BASE):
        if "labels.csv" in files and glob.glob(os.path.join(root, "*.npy")):
            return root
    return None

ROOT = find_dataset_root()
if ROOT is None:
    raise SystemExit("Could not find labels.csv + *.npy under /kaggle/input.\n"
                     "Attach the dataset you saved from Notebook 1.")
print("  Dataset root:", ROOT)

# ============================================================================
# STEP 2 — Load clips + labels
# ============================================================================
print("\n" + "=" * 70); print("STEP 2: Loading ..."); print("=" * 70)

rows = []
with open(os.path.join(ROOT, "labels.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append((r["file"], r["label"]))
print(f"  {len(rows)} clips in labels.csv")

all_labels = sorted(set(lab for _, lab in rows))
overlap = [w for w in MY_WORDS if w in all_labels]
print(f"  distinct labels: {len(all_labels)}")
print(f"  overlap with your 34 words: {len(overlap)} -> {overlap}")

keep_labels = None
if FILTER_TO_MY_WORDS and len(overlap) >= 2:
    keep_labels = set(overlap)
    print(f"  Training on the {len(keep_labels)}-word overlap.")
else:
    print("  Labels don't match your Bangla words (or too few) -> training on ALL "
          f"{len(all_labels)} labels. (Map IDs->Bangla later if needed.)")

# ---- shared feature pipeline ----
def normalize_seq(seq):
    out = seq.copy().astype(np.float32)
    for t in range(out.shape[0]):
        ls, rs = out[t, L_SH], out[t, R_SH]
        if np.all(ls == 0) or np.all(rs == 0): continue
        neck = (ls + rs) / 2.0; sd = np.linalg.norm(ls - rs) or 1.0
        m = ~np.all(out[t] == 0, axis=1)
        out[t][m] = (out[t][m] - neck) / sd
    return out

def resample_T(seq, T=T_FIXED):
    T0 = seq.shape[0]
    if T0 == 0: return np.zeros((T, N_PTS, 3), np.float32)
    if T0 == T: return seq
    idx = np.linspace(0, T0 - 1, T)
    lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, T0 - 1)
    fr = (idx - lo)[:, None, None]
    return (seq[lo]*(1-fr) + seq[hi]*fr).astype(np.float32)

def to_features(seq):
    return resample_T(normalize_seq(seq)).reshape(T_FIXED, FEAT)

X, y = [], []
for name, lab in rows:
    if keep_labels is not None and lab not in keep_labels:
        continue
    p = os.path.join(ROOT, name)
    try:
        seq = np.load(p).astype(np.float32)
    except Exception:
        continue
    if seq.ndim != 3 or seq.shape[1] != N_PTS:
        continue
    X.append(to_features(seq)); y.append(lab)

if not X:
    raise SystemExit("No samples after loading. Check the dataset / filter.")
X = np.stack(X).astype(np.float32)
classes = sorted(set(y)); cls2idx = {c: i for i, c in enumerate(classes)}
y_idx = np.array([cls2idx[c] for c in y], np.int64)
print(f"  X shape: {X.shape} | classes: {len(classes)}")

# ============================================================================
# STEP 3 — Train a GRU word classifier
# ============================================================================
print("\n" + "=" * 70); print("STEP 3: Training ..."); print("=" * 70)
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

dev = "cuda" if torch.cuda.is_available() else "cpu"; print("  device:", dev)
rng = np.random.default_rng(42); perm = rng.permutation(len(X))
n_val = max(1, int(0.15 * len(X))); val_idx, tr_idx = perm[:n_val], perm[n_val:]

def loader(idx, bs, sh):
    ds = TensorDataset(torch.tensor(X[idx]), torch.tensor(y_idx[idx]))
    return DataLoader(ds, batch_size=bs, shuffle=sh)
tr, va = loader(tr_idx, 64, True), loader(val_idx, 128, False)

class GRUClassifier(nn.Module):
    def __init__(self, feat=FEAT, hid=256, n_cls=len(classes), p=0.3):
        super().__init__()
        self.gru = nn.GRU(feat, hid, 2, batch_first=True, bidirectional=True, dropout=p)
        self.head = nn.Sequential(nn.LayerNorm(hid*2), nn.Dropout(p), nn.Linear(hid*2, n_cls))
    def forward(self, x):
        o, _ = self.gru(x); return self.head(o.mean(1))

model = GRUClassifier().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

best, best_state = 0.0, None
for ep in range(EPOCHS):
    model.train()
    for xb, yb in tr:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
    sched.step()
    model.eval(); c = t = 0
    with torch.no_grad():
        for xb, yb in va:
            xb, yb = xb.to(dev), yb.to(dev)
            c += (model(xb).argmax(1) == yb).sum().item(); t += len(yb)
    acc = c / max(1, t)
    if acc > best: best, best_state = acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if ep % 5 == 0 or ep == EPOCHS-1:
        print(f"  epoch {ep:2d}  val_acc={acc:.3f}  (best={best:.3f})")
print(f"\n  BEST val accuracy: {best:.3f}")

# ============================================================================
# STEP 4 — Save model + config (for sign_to_word_webcam.py)
# ============================================================================
print("\n" + "=" * 70); print("STEP 4: Saving ..."); print("=" * 70)
if best_state is not None: model.load_state_dict(best_state)
torch.save(model.state_dict(), os.path.join(OUT_DIR, "word_gru.pt"))
json.dump({"classes": classes, "feat": FEAT, "n_pts": N_PTS, "t_fixed": T_FIXED,
           "pose_n": POSE_N, "lh_n": LH_N, "rh_n": RH_N, "l_sh": L_SH, "r_sh": R_SH,
           "val_acc": best},
          open(os.path.join(OUT_DIR, "config.json"), "w"), ensure_ascii=False, indent=2)
print(f"  Saved: {OUT_DIR}/word_gru.pt , config.json")
print("  Download both and run sign_to_word_webcam.py on your PC.")
print("DONE.")

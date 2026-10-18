"""
================================================================================
BdSLW60 — NOTEBOOK 2 (v2) : TRAIN word recognition  [augmented + leak-free]
================================================================================
PREREQ: attach the landmark dataset you saved from Notebook 1 (bdslw60-landmarks).

WHAT CHANGED vs v1 (and why your live demo was failing):
 1. LEAK-FREE SPLIT. Each video holds ~6 near-identical repetitions of one sign.
    v1 split clips RANDOMLY, so near-duplicates landed in BOTH train and val ->
    val_acc looked great while the model had really just memorised recordings.
    v2 groups clips by recording (runs of identical consecutive labels = one
    video) and holds out WHOLE recordings. The accuracy it prints is honest.
 2. AUGMENTATION. Rotation / scale / shift / time-warp / jitter / hand-dropout /
    mirror-flip. This is the big one for generalising to YOU on YOUR webcam.
 3. VELOCITY FEATURES. Signs are motion; we feed position + frame-to-frame
    delta (225 -> 450 features).
 4. ROTATION NORMALISATION. Shoulders rotated level, so camera tilt stops
    mattering.
 5. Reports top-1 AND top-3 plus the worst-confused words.
 6. Bangla word map is EMBEDDED -> config.json comes out ready; no patch needed.

OUTPUT: word_gru.pt + config.json  -> use with sign_to_word_webcam.py
The feature spec here is mirrored EXACTLY in the webcam script via config.json.
================================================================================
"""

import os, glob, csv, json
import numpy as np

# ------------------------------------------------------------------ CONFIG
INPUT_BASE = "/kaggle/input"
OUT_DIR    = "/kaggle/working/bdsl_word_model"
os.makedirs(OUT_DIR, exist_ok=True)

POSE_N, LH_N, RH_N = 33, 21, 21
N_PTS   = POSE_N + LH_N + RH_N        # 75
T_FIXED = 48
L_SH, R_SH = 11, 12

USE_VELOCITY = True                   # position + delta
ROT_NORM     = True                   # level the shoulders
HAND_LOCAL   = True                   # wrist-centred handshape block (see below)
FEAT  = N_PTS*3*(2 if USE_VELOCITY else 1) + ((LH_N+RH_N)*3 if HAND_LOCAL else 0)  # 576
EPOCHS    = 80
VAL_FRAC  = 0.20
AUG       = True

# BdSLW60 official word list (from BdSLW60.pdf) -> baked into config.json
WORDMAP = {
    "W1":"বাবা","W2":"আত্মীয়","W3":"ভাই","W4":"বোন","W5":"বউ","W6":"চাচা","W7":"চাচী",
    "W8":"দাদা/নানা","W9":"দাদী/নানী","W10":"দায়িত্ব","W11":"দেবর","W12":"দুলাভাই",
    "W19":"কন্যা","W20":"মা","W37":"আম","W38":"আলু","W39":"আনারস","W40":"আঙ্গুর",
    "W41":"আপেল","W42":"বিস্কুট","W43":"বরই","W44":"কেক","W45":"চা","W46":"চাল",
    "W47":"চিনি","W48":"চিপস","W49":"চকলেট","W50":"ডাল","W91":"বোতাম","W92":"টুপি",
    "W93":"চাদর","W94":"চিরুনি","W95":"চশমা","W96":"চুড়ি","W97":"ক্লিপ","W98":"ক্রিম",
    "W99":"তথ্য","W100":"দেনাদার","W111":"জমজ","W112":"জুতা","W211":"টুথপেস্ট",
    "W212":"টিশার্ট","W213":"টিউবলাইট","W214":"টিভি","W215":"এসি","W216":"অ্যাপার্টমেন্ট",
    "W217":"অডিও ক্যাসেট","W218":"আয়না","W219":"বালতি","W220":"বালু","W351":"এইডস",
    "W352":"বাত","W353":"ব্যান্ডেজ","W354":"ক্যাপসুল","W355":"চিকিৎসা","W356":"চোখ ওঠা",
    "W357":"ডেঙ্গু","W358":"ডক্টর","W359":"দংশন","W360":"দুর্বল",
}

# ============================================================================
# STEP 1 — Locate dataset
# ============================================================================
print("=" * 70); print("STEP 1: Finding the landmark dataset ..."); print("=" * 70)

def find_dataset_root():
    for root, _, files in os.walk(INPUT_BASE):
        if "labels.csv" in files and glob.glob(os.path.join(root, "*.npy")):
            return root
    return None

ROOT = find_dataset_root()
if ROOT is None:
    raise SystemExit("No labels.csv + *.npy under /kaggle/input. "
                     "Attach the dataset saved from Notebook 1.")
print("  Dataset root:", ROOT)

rows = []
with open(os.path.join(ROOT, "labels.csv"), encoding="utf-8") as f:
    rd = csv.DictReader(f)
    has_signer = "signer" in (rd.fieldnames or [])
    for r in rd:
        rows.append((r["file"], r["label"], r.get("signer")))
print(f"  {len(rows)} clips | signer column: {'YES' if has_signer else 'no (will group by recording)'}")

# ============================================================================
# STEP 2 — Feature pipeline  (MUST match sign_to_word_webcam.py exactly)
# ============================================================================
def normalize_seq(seq):
    """Center on neck, scale by shoulder width, optionally level the shoulders."""
    out = seq.copy().astype(np.float32)
    for t in range(out.shape[0]):
        ls, rs = out[t, L_SH], out[t, R_SH]
        if np.all(ls == 0) or np.all(rs == 0):
            continue
        neck = (ls + rs) / 2.0
        sd = np.linalg.norm(ls - rs) or 1.0
        m = ~np.all(out[t] == 0, axis=1)
        out[t][m] = (out[t][m] - neck) / sd
        if ROT_NORM:
            v = out[t, R_SH] - out[t, L_SH]
            ang = np.arctan2(v[1], v[0])
            c, s = np.cos(-ang), np.sin(-ang)
            xy = out[t][m][:, :2]
            out[t][m] = np.concatenate(
                [np.stack([xy[:, 0]*c - xy[:, 1]*s, xy[:, 0]*s + xy[:, 1]*c], 1),
                 out[t][m][:, 2:]], axis=1)
    return out

def resample_T(seq, T=T_FIXED):
    T0 = seq.shape[0]
    if T0 == 0: return np.zeros((T, N_PTS, 3), np.float32)
    if T0 == T: return seq.astype(np.float32)
    idx = np.linspace(0, T0 - 1, T)
    lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, T0 - 1)
    fr = (idx - lo)[:, None, None]
    return (seq[lo]*(1-fr) + seq[hi]*fr).astype(np.float32)

def to_features(seq):
    f = seq.reshape(seq.shape[0], N_PTS*3)
    if USE_VELOCITY:
        d = np.diff(f, axis=0, prepend=f[:1])
        f = np.concatenate([f, d], axis=1)
    return f.astype(np.float32)

def hand_local(seq):
    """Wrist-centred, hand-size-scaled finger coordinates = HANDSHAPE signal.
    Why this matters: after shoulder-width normalisation a whole hand spans
    ~0.3 units, so the difference between a curled and an extended finger is
    ~0.02 — while an arm swing is ~1.0. The model was drowning in gross arm
    motion and nearly blind to handshape, which is what separates most words.
    Rescaling each hand into its own frame makes that detail visible."""
    T = seq.shape[0]
    out = np.zeros((T, LH_N + RH_N, 3), np.float32)
    for bi, off in enumerate((POSE_N, POSE_N + LH_N)):
        blk = seq[:, off:off+LH_N, :]
        wrist = blk[:, 0:1, :]
        span = np.linalg.norm(blk[:, 9, :] - blk[:, 0, :], axis=1)   # wrist->middle MCP
        ok = (~np.all(blk == 0, axis=(1, 2))) & (span > 1e-6)
        sc = np.where(span > 1e-6, span, 1.0)[:, None, None]
        loc = (blk - wrist) / sc
        out[ok, bi*LH_N:(bi+1)*LH_N, :] = loc[ok]
    return out

# ---- augmentation (training only) ----
POSE_PAIRS = [(1,4),(2,5),(3,6),(7,8),(9,10),(11,12),(13,14),(15,16),
              (17,18),(19,20),(21,22),(23,24),(25,26),(27,28),(29,30),(31,32)]

def flip_seq(s):
    """Mirror left<->right in NORMALISED space (x -> -x + swap L/R indices)."""
    p = s[:, :POSE_N, :].copy()
    for a, b in POSE_PAIRS:
        p[:, [a, b], :] = p[:, [b, a], :]
    lh = s[:, POSE_N:POSE_N+LH_N, :].copy()
    rh = s[:, POSE_N+LH_N:, :].copy()
    out = np.concatenate([p, rh, lh], axis=1)      # hands swap slots
    out[..., 0] *= -1.0
    return out

def time_crop(seq, rng):
    T0 = seq.shape[0]
    if T0 < 8: return seq
    frac = rng.uniform(0.80, 1.0)
    n = max(6, int(T0 * frac))
    a = rng.integers(0, T0 - n + 1)
    return seq[a:a+n]

def geom_aug(s, rng):
    out = s.copy()
    miss = np.all(s == 0, axis=2)                  # keep missing points at 0
    if rng.random() < 0.5:
        out = flip_seq(out); miss = np.all(out == 0, axis=2)
    ang = np.deg2rad(rng.uniform(-15, 15))
    c, sn = np.cos(ang), np.sin(ang)
    x, y = out[..., 0].copy(), out[..., 1].copy()
    out[..., 0] = x*c - y*sn
    out[..., 1] = x*sn + y*c
    out[..., :2] *= rng.uniform(0.85, 1.15)        # scale
    out[..., 0] += rng.uniform(-0.15, 0.15)        # shift
    out[..., 1] += rng.uniform(-0.15, 0.15)
    out += rng.normal(0, 0.012, out.shape)         # jitter
    if rng.random() < 0.10:                        # simulate a lost hand
        if rng.random() < 0.5: out[:, POSE_N:POSE_N+LH_N, :] = 0
        else:                  out[:, POSE_N+LH_N:, :] = 0
        miss = np.all(out == 0, axis=2)
    out[miss] = 0.0
    return out.astype(np.float32)

def build(seq, aug=False, rng=None):
    if aug: seq = time_crop(seq, rng)
    s = resample_T(normalize_seq(seq))
    if aug: s = geom_aug(s, rng)
    f = to_features(s)
    if HAND_LOCAL:
        f = np.concatenate([f, hand_local(s).reshape(s.shape[0], -1)], axis=1)
    return f.astype(np.float32)

# ============================================================================
# STEP 3 — Load clips + build a LEAK-FREE split
# ============================================================================
print("\n" + "=" * 70); print("STEP 3: Loading + splitting ..."); print("=" * 70)

X_raw, y, signers = [], [], []
for name, lab, sg in rows:
    try:
        seq = np.load(os.path.join(ROOT, name)).astype(np.float32)
    except Exception:
        continue
    if seq.ndim != 3 or seq.shape[1] != N_PTS or seq.shape[0] < 4:
        continue
    X_raw.append(seq); y.append(lab); signers.append(sg)

classes = sorted(set(y), key=lambda s: (len(s), s))
cls2idx = {c: i for i, c in enumerate(classes)}
y_idx = np.array([cls2idx[c] for c in y], np.int64)
print(f"  {len(X_raw)} clips | {len(classes)} words")

# group = one recording. Consecutive clips with the same label came from the
# same video (Notebook 1 wrote trials of a video back-to-back), so a run of
# identical labels is one recording. Hold out WHOLE recordings -> no leakage.
if has_signer and all(s for s in signers):
    groups = np.array([f"{s}" for s in signers])
    print("  Grouping by SIGNER (true signer-independent split).")
else:
    g, gid = [], 0
    for i in range(len(y)):
        if i > 0 and y[i] != y[i-1]: gid += 1
        g.append(gid)
    groups = np.array(g)
    print(f"  Grouping by RECORDING ({len(set(g))} groups) — no same-video "
          "clips shared between train and val.")

rng = np.random.default_rng(42)
val_groups = set()
by_lab = {}
for i, lab in enumerate(y):
    by_lab.setdefault(lab, set()).add(groups[i])
for lab, gs in by_lab.items():                      # hold out per word
    gs = sorted(gs)
    k = max(1, int(round(VAL_FRAC * len(gs))))
    val_groups.update(rng.choice(np.array(gs, dtype=object), size=k, replace=False).tolist())

val_mask = np.array([g in val_groups for g in groups])
tr_idx = np.where(~val_mask)[0]; val_idx = np.where(val_mask)[0]
print(f"  train={len(tr_idx)}  val={len(val_idx)}  (val recordings held out entirely)")

# ============================================================================
# STEP 4 — Train
# ============================================================================
print("\n" + "=" * 70); print("STEP 4: Training ..."); print("=" * 70)
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

dev = "cuda" if torch.cuda.is_available() else "cpu"; print("  device:", dev)

class SignDS(Dataset):
    def __init__(self, idx, train):
        self.idx = idx; self.train = train
        self.cache = None
        if not train:                                   # val is deterministic
            self.cache = [build(X_raw[i]) for i in idx]
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        i = self.idx[k]
        if self.cache is not None:
            f = self.cache[k]
        else:
            r = np.random.default_rng()
            f = build(X_raw[i], aug=AUG, rng=r)
        return torch.from_numpy(f), int(y_idx[i])

tr = DataLoader(SignDS(tr_idx, True), batch_size=64, shuffle=True,
                num_workers=2, drop_last=False)
va = DataLoader(SignDS(val_idx, False), batch_size=128, shuffle=False)

class GRUClassifier(nn.Module):
    def __init__(self, feat=FEAT, hid=256, n_cls=len(classes), p=0.3):
        super().__init__()
        self.inorm = nn.LayerNorm(feat)
        self.gru = nn.GRU(feat, hid, 2, batch_first=True, bidirectional=True, dropout=p)
        self.head = nn.Sequential(nn.LayerNorm(hid*2), nn.Dropout(p), nn.Linear(hid*2, n_cls))
    def forward(self, x):
        o, _ = self.gru(self.inorm(x))
        return self.head(o.mean(1))

model = GRUClassifier().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

best, best3, best_state = 0.0, 0.0, None
for ep in range(EPOCHS):
    model.train()
    for xb, yb in tr:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
    sched.step()
    model.eval(); c1 = c3 = n = 0
    with torch.no_grad():
        for xb, yb in va:
            xb, yb = xb.to(dev), yb.to(dev)
            lo = model(xb)
            t3 = lo.topk(3, dim=1).indices
            c1 += (t3[:, 0] == yb).sum().item()
            c3 += (t3 == yb[:, None]).any(1).sum().item()
            n += len(yb)
    a1, a3 = c1/max(1, n), c3/max(1, n)
    if a1 > best:
        best, best3 = a1, a3
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if ep % 5 == 0 or ep == EPOCHS-1:
        print(f"  epoch {ep:2d}  top1={a1:.3f}  top3={a3:.3f}   (best top1={best:.3f})")

print(f"\n  BEST  top-1 = {best:.3f}   top-3 = {best3:.3f}   [held-out recordings]")

# per-class accuracy -> which words to re-record / inspect
if best_state is not None: model.load_state_dict(best_state)
model.eval()
cor = np.zeros(len(classes)); tot = np.zeros(len(classes))
with torch.no_grad():
    for xb, yb in va:
        pr = model(xb.to(dev)).argmax(1).cpu().numpy(); yb = yb.numpy()
        for p, t in zip(pr, yb):
            tot[t] += 1; cor[t] += (p == t)
acc = np.divide(cor, np.maximum(tot, 1))
worst = np.argsort(acc)[:10]
print("  weakest words:", [(classes[i], WORDMAP.get(classes[i], classes[i]),
                            f"{acc[i]:.2f}") for i in worst])

# ============================================================================
# STEP 5 — Save
# ============================================================================
print("\n" + "=" * 70); print("STEP 5: Saving ..."); print("=" * 70)
torch.save(model.state_dict(), os.path.join(OUT_DIR, "word_gru.pt"))
json.dump({"classes": classes, "feat": FEAT, "n_pts": N_PTS, "t_fixed": T_FIXED,
           "pose_n": POSE_N, "lh_n": LH_N, "rh_n": RH_N, "l_sh": L_SH, "r_sh": R_SH,
           "use_velocity": USE_VELOCITY, "rot_norm": ROT_NORM,
           "hand_local": HAND_LOCAL,
           "val_acc": best, "val_top3": best3,
           "id2bn": {c: WORDMAP.get(c, c) for c in classes}},
          open(os.path.join(OUT_DIR, "config.json"), "w"),
          ensure_ascii=False, indent=2)
print(f"  Saved word_gru.pt + config.json (Bangla map already inside)")
print("  Download both, put next to sign_to_word_webcam.py, run it.")
print("DONE.")

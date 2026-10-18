"""
================================================================================
SIGN -> BANGLA WORD  (live webcam, v2)      — run on YOUR computer, not Kaggle
================================================================================
TWO MODES (press M to switch):
  HOLD mode (default, recommended):  press SPACE to start recording a sign,
      press SPACE again to stop -> it classifies that clip. No guessing where
      the sign begins/ends, so what you see is the model's TRUE quality.
  AUTO mode: old behaviour — pauses (hands still) split words automatically.

KEYS:  SPACE start/stop (HOLD)   M switch mode   C clear sentence   Q quit

SETUP (one time):
    pip install torch mediapipe opencv-python pillow numpy
Next to this script put (from the Kaggle Notebook-2 output):
    word_gru.pt      config.json
MediaPipe models + a Bangla font download themselves on first run.

The feature pipeline below is driven by config.json so it stays byte-identical
to training (position + velocity, shoulder-normalised, shoulders levelled).
================================================================================
"""

import os, json, urllib.request
import numpy as np
import cv2
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python as mptasks
from mediapipe.tasks.python import vision as mpvision
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
CFG  = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))

CLASSES = CFG["classes"]
ID2BN   = CFG.get("id2bn", {})
def display_label(c): return ID2BN.get(c, c)
FEAT    = CFG["feat"]
N_PTS   = CFG["n_pts"]
T_FIXED = CFG["t_fixed"]
POSE_N, LH_N, RH_N = CFG["pose_n"], CFG["lh_n"], CFG["rh_n"]
L_SH, R_SH = CFG["l_sh"], CFG["r_sh"]
USE_VELOCITY = CFG.get("use_velocity", False)
ROT_NORM     = CFG.get("rot_norm", False)
HAND_LOCAL   = CFG.get("hand_local", False)

# ---- auto-mode segmentation tuning ----
MOTION_ON, STILL_FRAMES, MIN_SIGN_FRAMES = 0.012, 8, 8
CONF_MIN = 0.35

# ---- MediaPipe Tasks models (same as the prep notebook) ----
POSE_VARIANT = "lite"
POSE_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            f"pose_landmarker_{POSE_VARIANT}/float16/latest/"
            f"pose_landmarker_{POSE_VARIANT}.task")
HAND_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task")

def _get(url, name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        print("downloading", name, "..."); urllib.request.urlretrieve(url, p)
    return p

pose = mpvision.PoseLandmarker.create_from_options(
    mpvision.PoseLandmarkerOptions(
        base_options=mptasks.BaseOptions(model_asset_path=_get(POSE_URL, "pose_landmarker.task")),
        running_mode=mpvision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5))
hands = mpvision.HandLandmarker.create_from_options(
    mpvision.HandLandmarkerOptions(
        base_options=mptasks.BaseOptions(model_asset_path=_get(HAND_URL, "hand_landmarker.task")),
        running_mode=mpvision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.5, min_tracking_confidence=0.5))

# ---- Bangla font ----
NOTO_URL = ("https://raw.githubusercontent.com/googlefonts/noto-fonts/main/"
            "hinted/ttf/NotoSansBengali/NotoSansBengali-Regular.ttf")
NOTO_LOCAL = os.path.join(HERE, "NotoSansBengali-Regular.ttf")

def find_bangla_font():
    for c in [NOTO_LOCAL, r"C:\Windows\Fonts\Nirmala.ttf", r"C:\Windows\Fonts\NirmalaUI.ttf",
              r"C:\Windows\Fonts\Vrinda.ttf", r"C:\Windows\Fonts\Shonar Bangla.ttf",
              "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf",
              "/System/Library/Fonts/Supplemental/Bangla MN.ttc"]:
        if os.path.exists(c): return c
    try:
        print("No Bangla font found; downloading Noto Sans Bengali ...")
        urllib.request.urlretrieve(NOTO_URL, NOTO_LOCAL); return NOTO_LOCAL
    except Exception as e:
        print("  font download failed:", e); return None

FONT_PATH = find_bangla_font()
def _load(sz):
    if not FONT_PATH: return None
    try:    return ImageFont.truetype(FONT_PATH, sz, layout_engine=ImageFont.Layout.RAQM)
    except Exception: return ImageFont.truetype(FONT_PATH, sz)
BIG, MID, SMALL = _load(58), _load(30), _load(24)

def draw_bn(img_bgr, text, pos, font, color=(0, 230, 0)):
    if not text: return img_bgr
    if font is None:
        cv2.putText(img_bgr, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2); return img_bgr
    im = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(im).text(pos, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

# ---- model ----
class GRUClassifier(nn.Module):
    def __init__(self, feat=FEAT, hid=256, n_cls=len(CLASSES), p=0.3):
        super().__init__()
        self.inorm = nn.LayerNorm(feat)
        self.gru = nn.GRU(feat, hid, 2, batch_first=True, bidirectional=True, dropout=p)
        self.head = nn.Sequential(nn.LayerNorm(hid*2), nn.Dropout(p), nn.Linear(hid*2, n_cls))
    def forward(self, x):
        o, _ = self.gru(self.inorm(x))
        return self.head(o.mean(1))

DEV = "cuda" if torch.cuda.is_available() else "cpu"
model = GRUClassifier().to(DEV)
sd = torch.load(os.path.join(HERE, "word_gru.pt"), map_location=DEV)
try:
    model.load_state_dict(sd)
except RuntimeError as e:
    raise SystemExit("word_gru.pt does not match config.json — download BOTH "
                     f"from the same Notebook-2 run.\n{e}")
model.eval()

# ---- feature pipeline (IDENTICAL to Notebook 2) ----
def res_to_75(pres, hres):
    pts = np.zeros((N_PTS, 3), np.float32)
    if pres.pose_landmarks:
        for j, lm in enumerate(pres.pose_landmarks[0][:POSE_N]):
            pts[j] = [lm.x, lm.y, lm.z]
    if hres.hand_landmarks:
        for lms, handed in zip(hres.hand_landmarks, hres.handedness):
            label = handed[0].category_name if handed else "Right"
            off = POSE_N if label == "Left" else POSE_N + LH_N
            for j, lm in enumerate(lms[:LH_N]):
                pts[off + j] = [lm.x, lm.y, lm.z]
    return pts

def normalize_seq(seq):
    out = seq.copy().astype(np.float32)
    for t in range(out.shape[0]):
        ls, rs = out[t, L_SH], out[t, R_SH]
        if np.all(ls == 0) or np.all(rs == 0): continue
        neck = (ls + rs) / 2.0
        sd_ = np.linalg.norm(ls - rs) or 1.0
        m = ~np.all(out[t] == 0, axis=1)
        out[t][m] = (out[t][m] - neck) / sd_
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

def hand_local(seq):
    """Build the wrist-centred, hand-size-scaled features used for training."""
    T = seq.shape[0]
    out = np.zeros((T, LH_N + RH_N, 3), np.float32)
    for bi, off in enumerate((POSE_N, POSE_N + LH_N)):
        blk = seq[:, off:off+LH_N, :]
        wrist = blk[:, 0:1, :]
        span = np.linalg.norm(blk[:, 9, :] - blk[:, 0, :], axis=1)
        ok = (~np.all(blk == 0, axis=(1, 2))) & (span > 1e-6)
        sc = np.where(span > 1e-6, span, 1.0)[:, None, None]
        loc = (blk - wrist) / sc
        out[ok, bi*LH_N:(bi+1)*LH_N, :] = loc[ok]
    return out

def to_features(seq):
    f = seq.reshape(seq.shape[0], N_PTS*3)
    if USE_VELOCITY:
        d = np.diff(f, axis=0, prepend=f[:1])
        f = np.concatenate([f, d], axis=1)
    if HAND_LOCAL:
        f = np.concatenate([f, hand_local(seq).reshape(seq.shape[0], -1)], axis=1)
    if f.shape[1] != FEAT:
        raise ValueError(
            f"Feature pipeline produced {f.shape[1]} values per frame; "
            f"config/model expects {FEAT}."
        )
    return f.astype(np.float32)

def classify(buf, k=3):
    seq = resample_T(normalize_seq(np.array(buf, np.float32)))
    x = torch.from_numpy(to_features(seq)[None]).to(DEV)
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0].cpu().numpy()
    top = p.argsort()[::-1][:k]
    return [(CLASSES[i], float(p[i])) for i in top]

def hand_motion(a, b):
    h = slice(POSE_N, N_PTS)
    a, b = a[h], b[h]
    m = (~np.all(a == 0, axis=1)) & (~np.all(b == 0, axis=1))
    if m.sum() == 0: return 0.0
    return float(np.linalg.norm(a[m] - b[m], axis=1).mean())

# ---- main loop ----
cap = cv2.VideoCapture(0)
if not cap.isOpened(): raise SystemExit("Could not open webcam (index 0).")

HOLD = True                 # start in hold-to-capture mode
recording = False
buf, still, prev = [], 0, None
top3, sentence = [], []
ts = 0
print("HOLD mode. SPACE = start/stop a sign | M = mode | C = clear | Q = quit")

while True:
    ok, frame = cap.read()
    if not ok: break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ts += 1
    mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    pts = res_to_75(pose.detect_for_video(mpimg, ts),
                    hands.detect_for_video(mpimg, ts))

    if HOLD:
        if recording: buf.append(pts)
    else:
        moving = prev is not None and hand_motion(prev, pts) > MOTION_ON
        if moving:
            buf.append(pts); still = 0
        else:
            still += 1
            if still == STILL_FRAMES and len(buf) >= MIN_SIGN_FRAMES:
                top3 = classify(buf)
                if top3[0][1] >= CONF_MIN: sentence.append(display_label(top3[0][0]))
                buf = []
            if still > STILL_FRAMES: buf = []
    prev = pts

    # ---------- overlay ----------
    disp = cv2.flip(frame, 1)
    mode = "HOLD  (SPACE=record)" if HOLD else "AUTO  (pause=split)"
    cv2.putText(disp, mode, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    if HOLD and recording:
        cv2.circle(disp, (disp.shape[1]-30, 30), 11, (0, 0, 255), -1)
        cv2.putText(disp, f"REC {len(buf)}", (disp.shape[1]-140, 37),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if top3:
        w, c = top3[0]
        disp = draw_bn(disp, display_label(w), (18, 52), BIG,
                       (0, 230, 0) if c >= CONF_MIN else (0, 180, 230))
        cv2.putText(disp, f"{c*100:.0f}%", (18, 132),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        for i, (w2, c2) in enumerate(top3[1:], start=0):   # runners-up
            disp = draw_bn(disp, f"{display_label(w2)}  {c2*100:.0f}%",
                           (18, 150 + i*34), MID, (170, 170, 170))

    disp = draw_bn(disp, " ".join(sentence[-8:]), (18, disp.shape[0]-52), SMALL,
                   (255, 255, 255))
    cv2.imshow("Sign -> Bangla word", disp)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"): break
    if k == ord("c"): sentence = []
    if k == ord("m"): HOLD = not HOLD; buf, recording, still = [], False, 0
    if k == ord(" ") and HOLD:
        if not recording:
            buf, recording = [], True
        else:
            recording = False
            if len(buf) >= 6:
                top3 = classify(buf)
                if top3[0][1] >= CONF_MIN: sentence.append(display_label(top3[0][0]))
            else:
                print("too short — hold SPACE through the whole sign")

cap.release(); cv2.destroyAllWindows()

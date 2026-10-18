"""
================================================================================
SIGN -> BANGLA WORD  (live webcam)   — run on YOUR computer, not Kaggle
================================================================================
You sign ONE word to the camera, pause, and the predicted Bangla word appears
on screen. Pauses separate words (hands still = boundary).

SETUP (one time):
    pip install torch mediapipe opencv-python pillow numpy
Put these next to this script (download them from the Kaggle notebook output):
    word_gru.pt        (trained model weights)
    config.json        (classes + feature spec)

RUN:
    python sign_to_word_webcam.py
Press  q  to quit.

NOTE: the feature pipeline here is IDENTICAL to the training notebook
(BdSLW401_Word_Recognition_Train.py). If you change it there, change it here too.
================================================================================
"""

import os, json, time
import numpy as np
import cv2
import torch
import torch.nn as nn
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
CFG  = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))

CLASSES = CFG["classes"]
FEAT    = CFG["feat"]        # 225
N_PTS   = CFG["n_pts"]       # 75
T_FIXED = CFG["t_fixed"]     # 48
POSE_N, LH_N, RH_N = CFG["pose_n"], CFG["lh_n"], CFG["rh_n"]
L_SH, R_SH = CFG["l_sh"], CFG["r_sh"]

# ---- pause / segmentation tuning ----
MOTION_ON   = 0.012   # mean hand-point movement/frame to count as "signing"
STILL_FRAMES = 8      # this many still frames in a row = word boundary
MIN_SIGN_FRAMES = 8   # ignore blips shorter than this
CONF_MIN = 0.35       # below this, show "?" instead of a wrong guess

# ---- Bangla font (OpenCV can't draw Bengali; use PIL) ----
def find_bangla_font():
    candidates = [
        r"C:\Windows\Fonts\Nirmala.ttf",
        r"C:\Windows\Fonts\NirmalaB.ttf",
        r"C:\Windows\Fonts\Shonar Bangla.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Bangla MN.ttc",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

FONT_PATH = find_bangla_font()
BIG  = ImageFont.truetype(FONT_PATH, 64) if FONT_PATH else None
SMALL= ImageFont.truetype(FONT_PATH, 26) if FONT_PATH else None

def draw_bangla(frame_bgr, text, pos, font, color=(0, 230, 0)):
    """Draw Bengali text onto a cv2 BGR frame via PIL."""
    if font is None:
        cv2.putText(frame_bgr, "(install Bangla font)", pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2); return frame_bgr
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(img).text(pos, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ---- model (MUST match the training notebook) ----
class GRUClassifier(nn.Module):
    def __init__(self, feat=FEAT, hid=256, n_cls=len(CLASSES), p=0.3):
        super().__init__()
        self.gru = nn.GRU(feat, hid, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=p)
        self.head = nn.Sequential(nn.LayerNorm(hid*2), nn.Dropout(p),
                                  nn.Linear(hid*2, n_cls))
    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out.mean(dim=1))

DEV = "cuda" if torch.cuda.is_available() else "cpu"
model = GRUClassifier().to(DEV)
model.load_state_dict(torch.load(os.path.join(HERE, "word_gru.pt"), map_location=DEV))
model.eval()

# ---- feature pipeline (IDENTICAL to training) ----
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

def normalize_seq(seq):
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
    T0 = seq.shape[0]
    if T0 == 0: return np.zeros((T, N_PTS, 3), np.float32)
    if T0 == T: return seq
    idx = np.linspace(0, T0 - 1, T)
    lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, T0 - 1)
    frac = (idx - lo)[:, None, None]
    return (seq[lo] * (1 - frac) + seq[hi] * frac).astype(np.float32)

def to_features(seq):
    return resample_T(normalize_seq(seq)).reshape(1, T_FIXED, FEAT)

def classify(buffer):
    feats = torch.tensor(to_features(np.array(buffer, np.float32))).to(DEV)
    with torch.no_grad():
        prob = torch.softmax(model(feats), dim=1)[0].cpu().numpy()
    k = int(prob.argmax())
    return CLASSES[k], float(prob[k])

def hand_motion(prev, cur):
    """mean movement of hand points between two frames (ignores missing)."""
    h = slice(POSE_N, N_PTS)  # the 42 hand points
    a, b = prev[h], cur[h]
    m = (~np.all(a == 0, axis=1)) & (~np.all(b == 0, axis=1))
    if m.sum() == 0: return 0.0
    return float(np.linalg.norm(a[m] - b[m], axis=1).mean())

# ---- main loop ----
holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam (index 0).")

buffer = []           # frames while signing
still = 0             # consecutive still frames
prev_pts = None
last_word, last_conf = "", 0.0
sentence = []         # accumulated recognized words (your 'gloss' sequence)

print("Ready. Sign a word, then pause. Press q to quit.")
while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)
    res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pts = landmarks_to_75(res)

    moving = False
    if prev_pts is not None:
        moving = hand_motion(prev_pts, pts) > MOTION_ON
    prev_pts = pts

    if moving:
        buffer.append(pts); still = 0
    else:
        still += 1
        # boundary: enough stillness AND a real sign was collected
        if still == STILL_FRAMES and len(buffer) >= MIN_SIGN_FRAMES:
            word, conf = classify(buffer)
            last_word, last_conf = word, conf
            if conf >= CONF_MIN:
                sentence.append(word)
            buffer = []
        if still > STILL_FRAMES:
            buffer = []   # keep it clear while idle

    # ---- overlay ----
    status = "SIGNING..." if moving else "pause"
    cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 200, 255) if moving else (150, 150, 150), 2)
    show = last_word if last_conf >= CONF_MIN else (last_word + " ?")
    frame = draw_bangla(frame, show, (20, 70), BIG,
                        (0, 230, 0) if last_conf >= CONF_MIN else (0, 180, 230))
    if last_word:
        cv2.putText(frame, f"conf {last_conf:.2f}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    frame = draw_bangla(frame, " ".join(sentence[-8:]), (20, frame.shape[0]-60),
                        SMALL, (255, 255, 255))

    cv2.imshow("Sign -> Bangla word  (q=quit, c=clear)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"): break
    if key == ord("c"): sentence = []

cap.release(); cv2.destroyAllWindows()

"""
================================================================================
BdSLW60 — NOTEBOOK 1 of 2 : PREP  (download -> extract landmarks -> save)
================================================================================
PURPOSE (run this ONCE):
  - Download BdSLW60 from Zenodo (internet must be ON in notebook settings)
  - Extract MediaPipe landmarks (pose+hands = 75 points) from every clip
  - Save compact per-clip .npy + labels.csv  ->  /kaggle/working/bdslw60_landmarks

THEN:
  - Save Version (Save & Run All) so the output persists
  - Output tab -> "New Dataset"  ->  name it e.g.  bdslw60-landmarks
  - Notebook 2 attaches that dataset and trains (fast, no re-download/extract)

WHY: raw video is big + slow to process. Doing extraction ONCE and storing tiny
landmark .npy files means every future training run is fast and offline.

FEATURE SPEC (kept identical in Notebook 2 and the webcam script):
  pose(33) + left_hand(21) + right_hand(21) = 75 points x (x,y,z).
  We save RAW landmarks here as (T,75,3); normalization happens at train time.
================================================================================
"""

import os, sys, glob, json, csv, time
import numpy as np

# ------------------------------------------------------------------ CONFIG
# BdSLW60 is ALREADY on Kaggle -> attach it:  hasaniut/BdSLW60
#   Add Input -> Datasets -> search "BdSLW60" -> add.
# We read it straight from /kaggle/input (no download). The Zenodo download is
# only a fallback if, for some reason, no dataset is attached.
OUT_DIR = "/kaggle/working/bdslw60_landmarks"   # <- becomes your reusable dataset
os.makedirs(OUT_DIR, exist_ok=True)

def _find_attached_input():
    """Return /kaggle/input if an attached dataset has video/landmark files."""
    exts = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".npy")
    for root, _, files in os.walk("/kaggle/input"):
        if any(f.lower().endswith(exts) for f in files):
            return "/kaggle/input"
    return None

ATTACHED = _find_attached_input()
if ATTACHED:
    DL_DIR = ATTACHED                            # use the attached Kaggle dataset
    print("Using attached Kaggle dataset at /kaggle/input (no download needed).")
else:
    DL_DIR = "/kaggle/working/bdslw60_raw"        # Zenodo fallback target
    os.makedirs(DL_DIR, exist_ok=True)
    ZENODO_RECORD = "6779843"  # NOTE: this record is actually SignBD-Word (~24GB);
                               # prefer attaching hasaniut/BdSLW60 instead.

POSE_N, LH_N, RH_N = 33, 21, 21
N_PTS = POSE_N + LH_N + RH_N                     # 75
FRAME_STRIDE = 2          # sample every Nth video frame (speed vs detail)
MAX_PER_CLASS = 10_000    # cap clips per word (BdSLW60 is small, keep high)
MODEL_COMPLEXITY = 1      # MediaPipe Holistic complexity (0=fast,1=balanced)

# ============================================================================
# STEP 1 — Get the data (attached Kaggle dataset preferred; Zenodo fallback)
# ============================================================================
print("=" * 70); print("STEP 1: Locating data ..."); print("=" * 70)
import requests, zipfile

def already_downloaded():
    return any(True for _ in glob.iglob(os.path.join(DL_DIR, "**", "*"), recursive=True))

if ATTACHED:
    print("  Reading attached dataset from", DL_DIR, "(skip download).")
elif already_downloaded():
    print("  Data already present in", DL_DIR, "- skipping download.")
else:
    print("  No attached dataset found; falling back to Zenodo download.")
    print("  (Recommended instead: attach hasaniut/BdSLW60 and re-run.)")
    rec = requests.get(f"https://zenodo.org/api/records/{ZENODO_RECORD}").json()
    files = rec.get("files", [])
    print(f"  Record has {len(files)} file(s).")
    for f in files:
        name = f["key"]; url = f["links"]["self"]; size_mb = f.get("size", 0) / 1e6
        dst = os.path.join(DL_DIR, name)
        print(f"  Downloading {name} ({size_mb:.0f} MB) ...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dst, "wb") as out:
                for chunk in r.iter_content(1 << 20):
                    out.write(chunk)
        if name.lower().endswith(".zip"):
            print("    unzipping ...")
            try:
                with zipfile.ZipFile(dst) as z:
                    z.extractall(DL_DIR)
                os.remove(dst)
            except Exception as e:
                print("    (unzip failed:", e, ")")
    print("  Download complete.")

# ============================================================================
# STEP 2 — Inspect structure (so we know how it's organized)
# ============================================================================
print("\n" + "=" * 70); print("STEP 2: Inspecting downloaded data ..."); print("=" * 70)
exts = {}; samples = []
for root, _, fs in os.walk(DL_DIR):
    for f in fs:
        e = os.path.splitext(f)[1].lower(); exts[e] = exts.get(e, 0) + 1
        if len(samples) < 20: samples.append(os.path.join(root, f))
print("  File types:")
for e, c in sorted(exts.items(), key=lambda x: -x[1]):
    print(f"    {e or '<none>':8s} : {c}")
print("  Sample paths:")
for s in samples[:15]:
    print("   ", s.replace(DL_DIR, "."))

# ============================================================================
# STEP 3 — Load / extract landmarks  (adaptive: video OR provided landmarks)
# ============================================================================
print("\n" + "=" * 70); print("STEP 3: Extracting landmarks ..."); print("=" * 70)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
LAND_EXTS  = {".npy", ".csv", ".parquet", ".pkl"}

def label_from_path(path):
    """Word/label for a clip. BdSLW60 clips are typically organized as
    <word>/<clip> or <signer>/<word>/<clip> or <id>_<...>. Uses the immediate
    parent folder by default; falls back to a numeric/ID token in the name.
    >>> If STEP 2 shows a different layout, adjust this one function. <<<"""
    import re
    parts = os.path.normpath(path).split(os.sep)
    base  = os.path.splitext(parts[-1])[0]
    parent = parts[-2] if len(parts) >= 2 else base
    # if parent looks like a signer id (e.g. U1/P01), go one level up
    if re.fullmatch(r"[A-Za-z]{1,3}\d{1,3}", parent) and len(parts) >= 3:
        return parts[-3]
    return parent

_holi = None
def extract_video(path):
    global _holi
    import cv2, mediapipe as mp
    if _holi is None:
        _holi = mp.solutions.holistic.Holistic(static_image_mode=False,
                                                model_complexity=MODEL_COMPLEXITY)
    cap = cv2.VideoCapture(path); frames = []; i = 0
    while True:
        ok, img = cap.read()
        if not ok: break
        if i % FRAME_STRIDE == 0:
            res = _holi.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            frames.append(res_to_75(res))
        i += 1
    cap.release()
    return np.array(frames, np.float32) if frames else None

def res_to_75(res):
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

def load_landmarks(path):
    e = os.path.splitext(path)[1].lower()
    try:
        if e == ".npy":   arr = np.load(path, allow_pickle=True)
        elif e == ".csv": arr = np.loadtxt(path, delimiter=",")
        elif e == ".parquet":
            import pandas as pd; arr = pd.read_parquet(path).values
        elif e == ".pkl":
            import pickle; arr = pickle.load(open(path, "rb"))
        else: return None
        a = np.asarray(arr, np.float32)
        if a.ndim == 2 and a.shape[1] == N_PTS*3: return a.reshape(-1, N_PTS, 3)
        if a.ndim == 3 and a.shape[1] == N_PTS and a.shape[2] == 3: return a
        if a.ndim == 3 and a.shape[1] >= N_PTS:  return a[:, :N_PTS, :]
        return None
    except Exception:
        return None

all_files = [os.path.join(r, f) for r, _, fs in os.walk(DL_DIR) for f in fs]
videos = [p for p in all_files if os.path.splitext(p)[1].lower() in VIDEO_EXTS]
lands  = [p for p in all_files if os.path.splitext(p)[1].lower() in LAND_EXTS]
use_video = len(videos) > len(lands)
print(f"  videos={len(videos)}  landmark_files={len(lands)}  -> "
      f"{'extracting from VIDEO' if use_video else 'loading landmark files'}")
if use_video:
    try:
        import cv2, mediapipe  # noqa
    except Exception:
        print("  installing mediapipe/opencv ...")
        os.system(f"{sys.executable} -m pip install -q mediapipe opencv-python-headless")

src = videos if use_video else lands
rows = []            # (saved_npy_name, label)
per_class = {}
t0 = time.time()
for i, path in enumerate(src):
    lab = label_from_path(path)
    if per_class.get(lab, 0) >= MAX_PER_CLASS:
        continue
    seq = extract_video(path) if use_video else load_landmarks(path)
    if seq is None or seq.shape[0] < 4:
        continue
    name = f"{len(rows):06d}.npy"
    np.save(os.path.join(OUT_DIR, name), seq.astype(np.float32))
    rows.append((name, lab)); per_class[lab] = per_class.get(lab, 0) + 1
    if i % 200 == 0:
        el = time.time() - t0
        print(f"    [{i}/{len(src)}] saved={len(rows)} classes={len(per_class)} "
              f"({el:.0f}s)")

print(f"\n  Extracted {len(rows)} clips across {len(per_class)} words.")
if len(rows) == 0:
    raise SystemExit("No clips extracted. Check STEP 2 structure and fix "
                     "label_from_path()/load_landmarks().")

# ============================================================================
# STEP 4 — Save labels + metadata (and copy any word-list file we can find)
# ============================================================================
print("\n" + "=" * 70); print("STEP 4: Saving labels ..."); print("=" * 70)
with open(os.path.join(OUT_DIR, "labels.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["file", "label"])
    for name, lab in rows: w.writerow([name, lab])

classes = sorted(per_class.keys())
json.dump({"classes": classes,
           "n_pts": N_PTS, "pose_n": POSE_N, "lh_n": LH_N, "rh_n": RH_N,
           "counts": per_class, "total": len(rows)},
          open(os.path.join(OUT_DIR, "meta.json"), "w"),
          ensure_ascii=False, indent=2)

# copy any mapping / word-list file from the raw dataset (helps map IDs->Bangla)
import shutil
for p in all_files:
    bn = os.path.basename(p).lower()
    if os.path.splitext(p)[1].lower() in {".csv", ".txt", ".json", ".xlsx"} and \
       any(k in bn for k in ["word", "label", "class", "vocab", "gloss", "map", "list"]):
        try:
            shutil.copy2(p, os.path.join(OUT_DIR, "SRC_" + os.path.basename(p)))
        except Exception:
            pass

print(f"  Saved: labels.csv ({len(rows)} rows), meta.json")
print(f"  Classes found ({len(classes)}): {classes[:30]}{' ...' if len(classes)>30 else ''}")
print(f"\n  Output folder: {OUT_DIR}")
print("\n  NEXT: Save Version (Save & Run All) -> Output tab -> 'New Dataset'")
print("        Name it 'bdslw60-landmarks' and attach it in Notebook 2.")
print("DONE.")

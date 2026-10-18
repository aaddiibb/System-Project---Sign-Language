"""
================================================================================
BdSL Retrieval-Only Sign Language Generation — DEMO
================================================================================

WHAT THIS DOES:
  Give it a Bangla text query (or let it pick a random sample from the dataset),
  and it:
    1. Searches the TRAIN corpus for the closest matching sentence
         exact match  ->  fuzzy TF-IDF char n-gram  ->  (optional) LaBSE semantic
    2. Loads that sentence's pose sequence (T, 543, 3)
    3. Renders it as a sign-language skeleton animation (inline video + saved GIF)

  "Retrieval-only" = we never GENERATE new poses. We find the best existing
  pose sequence in the dataset and play it back. This is exactly what the 2025
  SOTA systems (USTC-MoE @ SLRTP, SOKE @ ICCV) do at the sentence level.

KAGGLE SETUP:
  Add these input datasets to the notebook:
    1. "bdsl-final-normalized"   (your merged train/val poses)
    2. "bdsl-mediapipe-final"    (OR the session-0 dataset — needed for the
                                   mapping CSVs that hold the Bangla sentence text)
  CPU is fine. Then paste this script and Run All.

HOW TO USE:
  - Leave QUERY_TEXT = None  ->  picks a random val sample and signs it.
  - Set   QUERY_TEXT = "আমি ভালো আছি"  ->  signs your own text.

================================================================================
"""

import os
import re
import csv
import glob
import base64
import random
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_BASE = "/kaggle/input"
WORK_DIR = "/kaggle/working"

# Where the merged dataset lives. We auto-discover it, but this is the default.
FINAL_DATASET_HINTS = ["bdsl-final-normalized", "bdsl_final_dataset"]

QUERY_TEXT = None        # None -> random val sample; or set a Bangla string
RANDOM_SEED = None       # set an int for reproducible random pick, else None
TOP_N = 5                # how many ranked candidates to print
USE_LABSE = False        # semantic fallback; needs sentence-transformers + model
FACE_POINTS = True       # draw the 468 face landmarks (faint). False = cleaner.
MAX_RENDER_FRAMES = 240  # cap animation length for speed

# MediaPipe Holistic landmark layout (from the dataset metadata)
POSE_SLICE = slice(0, 33)      # 33 body
LH_SLICE = slice(33, 54)       # 21 left hand
RH_SLICE = slice(54, 75)       # 21 right hand
FACE_SLICE = slice(75, 543)    # 468 face

POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12

# Skeleton edges -------------------------------------------------------------
# Upper-body pose connections (indices into the 33 pose landmarks)
POSE_EDGES = [
    (11, 12),                       # shoulders
    (11, 13), (13, 15),             # left upper arm -> forearm
    (12, 14), (14, 16),             # right upper arm -> forearm
    (11, 23), (12, 24), (23, 24),   # torso
    (0, 11), (0, 12),               # nose -> shoulders (neck-ish)
]
# 21-point hand connections (MediaPipe standard), indices LOCAL to the hand
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (5, 9), (9, 10), (10, 11), (11, 12),    # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm
]

BENGALI_RE = re.compile(r"[ঀ-৿]")


def has_bengali(s):
    return bool(BENGALI_RE.search(str(s)))


# ============================================================================
# STEP 1: Locate the final dataset (train/ and val/ with .npy poses)
# ============================================================================
print("=" * 70)
print("BdSL Retrieval-Only Sign Generation — Demo")
print("=" * 70)


def find_dataset_root():
    """Find the directory that directly contains train/ and val/ with .npy files."""
    for root, dirs, _ in os.walk(INPUT_BASE):
        if "train" in dirs and "val" in dirs:
            t = os.path.join(root, "train")
            if glob.glob(os.path.join(t, "*.npy")):
                return root
    return None


DATASET_ROOT = find_dataset_root()
if DATASET_ROOT is None:
    raise SystemExit(
        "ERROR: Could not find a folder containing train/ and val/ *.npy files.\n"
        "  Add the 'bdsl-final-normalized' dataset to this notebook."
    )

TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")
print(f"\n  Dataset root: {DATASET_ROOT}")


def index_poses(pose_dir):
    """filename stem -> dict(path, sentence_id, signer_id). Stem = {sentence}_{signer}."""
    out = {}
    for p in sorted(glob.glob(os.path.join(pose_dir, "*.npy"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            sentence_id, signer_id = parts
        else:
            sentence_id, signer_id = stem, "unknown"
        out[stem] = {"path": p, "sentence_id": sentence_id, "signer_id": signer_id}
    return out


train_index = index_poses(TRAIN_DIR)
val_index = index_poses(VAL_DIR)
print(f"  Train pose files: {len(train_index)}")
print(f"  Val pose files:   {len(val_index)}")

# ============================================================================
# STEP 2: Load Bangla sentence text from Bangla_Sign_Sentence.xlsx
# ============================================================================
# The pose mapping CSVs only carry (sentence_id, signer_id, pose_file, num_frames)
# -- NO text. The sentence text lives in "Bangla_Sign_Sentence.xlsx", sheet
# "Training Data", with columns: Sentence ID | Natural Sentence | Sign Sentence.
#   - Natural Sentence = normal Bangla  (THIS is the retrieval query/key)
#   - Sign Sentence    = the signed gloss order (kept for reference)
# Sentence IDs match the pose filenames exactly (verified 100% overlap).
# ============================================================================
print(f"\n{'=' * 70}")
print("Step 2: Loading sentence text from Bangla_Sign_Sentence.xlsx...")

# sentence_id -> natural Bangla text   (and a parallel sign-gloss map)
sentence_text = {}
sign_gloss = {}


def find_sentence_xlsx():
    """Locate Bangla_Sign_Sentence.xlsx in /kaggle/input or beside this script."""
    search_roots = [INPUT_BASE, WORK_DIR, os.getcwd()]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        hits = glob.glob(os.path.join(root, "**", "Bangla_Sign_Sentence.xlsx"),
                         recursive=True)
        if hits:
            return hits[0]
    # last resort: any .xlsx that has a 'Sentence ID'-style sheet
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for x in glob.glob(os.path.join(root, "**", "*.xlsx"), recursive=True):
            return x
    return None


XLSX_PATH = find_sentence_xlsx()
if XLSX_PATH:
    print(f"  Found: {XLSX_PATH}")
    try:
        import openpyxl
        wb = openpyxl.load_workbook(XLSX_PATH, read_only=True, data_only=True)

        # prefer the "Training Data" sheet; else first sheet
        ws = wb["Training Data"] if "Training Data" in wb.sheetnames else wb.worksheets[0]
        rows = list(ws.iter_rows(values_only=True))
        header = [str(h).strip().lower() if h is not None else "" for h in rows[0]]

        # locate columns by header name (robust to reordering)
        def col(idx_names, default):
            for i, h in enumerate(header):
                if any(k in h for k in idx_names):
                    return i
            return default

        c_id = col(["sentence id", "sentence_id", "sentenceid", "id"], 0)
        c_nat = col(["natural"], 1)
        c_sign = col(["sign"], 2)

        for r in rows[1:]:
            if r[c_id] is None:
                continue
            sid = str(r[c_id]).strip()
            # ids may come through as "23511" or "23511.0"
            if sid.endswith(".0"):
                sid = sid[:-2]
            nat = (str(r[c_nat]).strip() if c_nat < len(r) and r[c_nat] else "")
            sgn = (str(r[c_sign]).strip() if c_sign < len(r) and r[c_sign] else "")
            if nat:
                sentence_text[sid] = nat
            if sgn:
                sign_gloss[sid] = sgn
        print(f"  Loaded {len(sentence_text)} sentence texts "
              f"(sheet: '{ws.title}', columns: id={c_id}, natural={c_nat}, sign={c_sign})")
    except Exception as e:
        print(f"  ERROR reading xlsx: {e}")
else:
    print("  Bangla_Sign_Sentence.xlsx NOT FOUND in /kaggle/input.")


def text_for(entry):
    return sentence_text.get(entry["sentence_id"])


covered = sum(1 for e in train_index.values() if text_for(e))
print(f"  Train files with text: {covered}/{len(train_index)}")

HAS_TEXT = covered > 0
if not HAS_TEXT:
    print("\n  WARNING: No Bangla text matched the pose files.")
    print("  Retrieval will fall back to matching by sentence_id only.")
    print("  FIX: add 'Bangla_Sign_Sentence.xlsx' to one of your Kaggle input")
    print("  datasets (it is only ~92 KB) so this notebook can read it.")

# ============================================================================
# STEP 3: Build the retrieval corpus (unique TRAIN sentences)
# ============================================================================
print(f"\n{'=' * 70}")
print("Step 3: Building retrieval index over TRAIN sentences...")

# corpus_sentences: sentence_id -> {text, files:[stems...]}
corpus = {}
for stem, e in train_index.items():
    sid = e["sentence_id"]
    corpus.setdefault(sid, {"text": text_for(e) or "", "files": []})
    corpus[sid]["files"].append(stem)

corpus_ids = list(corpus.keys())
corpus_texts = [corpus[s]["text"] for s in corpus_ids]
print(f"  Corpus size: {len(corpus_ids)} unique train sentences")


def normalize_text(s):
    """Light normalization for exact matching."""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def strip_text(s):
    """Aggressive: drop whitespace + punctuation for a looser exact match."""
    s = normalize_text(s)
    s = re.sub(r"[।,.!?;:\"'()\-\s]", "", s)
    return s


# exact-match tables
exact_map = {}
loose_map = {}
for sid, txt in zip(corpus_ids, corpus_texts):
    if txt:
        exact_map.setdefault(normalize_text(txt), sid)
        loose_map.setdefault(strip_text(txt), sid)

# TF-IDF char n-gram index (only over sentences that have text)
tfidf = None
tfidf_matrix = None
tfidf_ids = []
if HAS_TEXT:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    tfidf_ids = [s for s, t in zip(corpus_ids, corpus_texts) if t]
    tfidf_corpus = [corpus[s]["text"] for s in tfidf_ids]
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    tfidf_matrix = tfidf.fit_transform(tfidf_corpus)
    print(f"  TF-IDF index: {tfidf_matrix.shape[0]} sentences, "
          f"{tfidf_matrix.shape[1]} features")

# optional LaBSE semantic index
labse = None
labse_emb = None
if USE_LABSE and HAS_TEXT:
    try:
        from sentence_transformers import SentenceTransformer
        labse = SentenceTransformer("sentence-transformers/LaBSE")
        labse_emb = labse.encode(tfidf_corpus, normalize_embeddings=True,
                                 show_progress_bar=True)
        print(f"  LaBSE index: {labse_emb.shape}")
    except Exception as e:
        print(f"  LaBSE unavailable ({e}); continuing with TF-IDF only.")
        labse = None

# ============================================================================
# STEP 4: The retrieval function
# ============================================================================
print(f"\n{'=' * 70}")
print("Step 4: Retrieval ready.")


def retrieve(query_text, top_n=TOP_N):
    """Return (best_sentence_id, method, ranked[list of (sid, score, text)])."""
    q = normalize_text(query_text)

    # Tier 1: exact
    if q in exact_map:
        sid = exact_map[q]
        return sid, "exact", [(sid, 1.0, corpus[sid]["text"])]
    if strip_text(q) in loose_map:
        sid = loose_map[strip_text(q)]
        return sid, "exact-loose", [(sid, 0.99, corpus[sid]["text"])]

    # Tier 2: TF-IDF fuzzy
    if tfidf is not None:
        from sklearn.metrics.pairwise import cosine_similarity
        qv = tfidf.transform([q])
        sims = cosine_similarity(qv, tfidf_matrix).ravel()
        order = np.argsort(-sims)[:top_n]
        ranked = [(tfidf_ids[i], float(sims[i]), corpus[tfidf_ids[i]]["text"])
                  for i in order]
        best_sid, best_score = ranked[0][0], ranked[0][1]

        # Tier 3: LaBSE semantic re-rank if fuzzy is weak
        if labse is not None and best_score < 0.5:
            qe = labse.encode([q], normalize_embeddings=True)
            sem = (labse_emb @ qe[0])
            order = np.argsort(-sem)[:top_n]
            ranked = [(tfidf_ids[i], float(sem[i]), corpus[tfidf_ids[i]]["text"])
                      for i in order]
            return ranked[0][0], "semantic-labse", ranked
        return best_sid, "fuzzy-tfidf", ranked

    # Fallback: no text available -> match by sentence_id string
    if query_text in corpus:
        return query_text, "id-exact", [(query_text, 1.0, "")]
    return None, "no-match", []


def pose_for_sentence(sid):
    """Pick a pose file for a sentence_id (prefer train). Returns (stem, poses)."""
    if sid in corpus and corpus[sid]["files"]:
        stem = corpus[sid]["files"][0]
        return stem, np.load(train_index[stem]["path"])
    # fall back to any val file with that sentence
    for stem, e in val_index.items():
        if e["sentence_id"] == sid:
            return stem, np.load(e["path"])
    return None, None


# ============================================================================
# STEP 5: Skeleton rendering (OpenCV, MediaPipe-style, exports MP4)
# ============================================================================
print(f"\n{'=' * 70}")
print("Step 5: Preparing renderer...")

import cv2

# --- Rendering config --------------------------------------------------------
CANVAS_W, CANVAS_H = 540, 680     # output video size
FPS = 25                          # smooth playback
SMOOTH_WINDOW = 3                 # temporal smoothing (odd int, 1 = off)
BG_COLOR = (250, 249, 246)        # near-white (BGR)
COL_BODY = (150, 90, 40)          # slate blue (BGR)
COL_LHAND = (60, 60, 200)         # warm red  (left hand)
COL_RHAND = (70, 160, 40)         # green     (right hand)
COL_FACE = (205, 200, 195)        # faint grey face mesh
COL_JOINT = (90, 70, 60)

# Build connection edges (global indices into the 543 layout).
# Prefer MediaPipe's official graphs; fall back to hardcoded if unavailable.
def build_edges():
    pose_e, lh_e, rh_e, face_e = [], [], [], []
    try:
        import mediapipe as mp
        pose_e = [(a, b) for a, b in mp.solutions.pose.POSE_CONNECTIONS]
        hc = [(a, b) for a, b in mp.solutions.hands.HAND_CONNECTIONS]
        lh_e = [(a + 33, b + 33) for a, b in hc]
        rh_e = [(a + 54, b + 54) for a, b in hc]
        fc = mp.solutions.face_mesh.FACEMESH_CONTOURS  # clean outline, not full mesh
        face_e = [(a + 75, b + 75) for a, b in fc]
        print("  Using MediaPipe connection graphs (incl. face contours).")
    except Exception:
        pose_e = POSE_EDGES
        lh_e = [(a + 33, b + 33) for a, b in HAND_EDGES]
        rh_e = [(a + 54, b + 54) for a, b in HAND_EDGES]
        face_e = []  # no face without mediapipe
        print("  MediaPipe not available — using built-in body+hand graph.")
    return pose_e, lh_e, rh_e, face_e


POSE_E, LH_E, RH_E, FACE_E = build_edges()


def _smooth(poses, k):
    """Moving-average temporal smoothing over valid (non-zero) landmarks."""
    if k <= 1:
        return poses
    T = poses.shape[0]
    out = poses.copy()
    half = k // 2
    valid = ~np.all(poses == 0, axis=2)  # (T, 543)
    for t in range(T):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        window = poses[lo:hi]                       # (w,543,3)
        wvalid = valid[lo:hi][..., None]            # (w,543,1)
        denom = wvalid.sum(axis=0)                  # (543,1)
        summed = (window * wvalid).sum(axis=0)      # (543,3)
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = np.where(denom > 0, summed / np.maximum(denom, 1), 0.0)
        out[t] = np.where(valid[t][..., None], avg, 0.0)
    return out


def render_pose(poses, title="", out_mp4=None, out_gif=None):
    """Render a (T,543,3) pose sequence as a clean skeleton MP4.
    Returns inline-displayable HTML (falls back to a GIF if MP4 can't embed)."""
    T = poses.shape[0]
    if T > MAX_RENDER_FRAMES:
        idx = np.linspace(0, T - 1, MAX_RENDER_FRAMES).astype(int)
        poses = poses[idx]
        T = poses.shape[0]

    poses = _smooth(poses.astype(np.float32), SMOOTH_WINDOW)

    # Bounding box from all non-zero points -> map tightly into the canvas.
    pts = poses.reshape(-1, 3)
    pts = pts[~np.all(pts == 0, axis=1)]
    if len(pts) == 0:
        raise ValueError("All landmarks are zero — nothing to render.")
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()

    # keep aspect ratio, leave a margin
    margin = 0.12
    span_x = (xmax - xmin) or 1e-3
    span_y = (ymax - ymin) or 1e-3
    draw_w = CANVAS_W * (1 - 2 * margin)
    draw_h = CANVAS_H * (1 - 2 * margin)
    scale = min(draw_w / span_x, draw_h / span_y)
    # center the figure
    off_x = (CANVAS_W - span_x * scale) / 2 - xmin * scale
    off_y = (CANVAS_H - span_y * scale) / 2 - ymin * scale

    def to_px(pt):
        return (int(pt[0] * scale + off_x), int(pt[1] * scale + off_y))

    def valid(pt):
        return not (pt[0] == 0 and pt[1] == 0 and pt[2] == 0)

    def draw_edges(canvas, P, edges, color, thickness):
        for a, b in edges:
            pa, pb = P[a], P[b]
            if valid(pa) and valid(pb):
                cv2.line(canvas, to_px(pa), to_px(pb), color, thickness, cv2.LINE_AA)

    frames = []
    for t in range(T):
        P = poses[t]
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)

        # face first (thin, subtle), then body, then hands on top
        if FACE_POINTS:
            draw_edges(canvas, P, FACE_E, COL_FACE, 1)
        draw_edges(canvas, P, POSE_E, COL_BODY, 4)
        draw_edges(canvas, P, LH_E, COL_LHAND, 2)
        draw_edges(canvas, P, RH_E, COL_RHAND, 2)

        # joint dots for hands (so fingers read clearly even when small)
        for sl, col in [(LH_SLICE, COL_LHAND), (RH_SLICE, COL_RHAND)]:
            for pt in P[sl]:
                if valid(pt):
                    cv2.circle(canvas, to_px(pt), 3, col, -1, cv2.LINE_AA)
        # shoulder/elbow/wrist dots
        for i in (11, 12, 13, 14, 15, 16):
            if valid(P[i]):
                cv2.circle(canvas, to_px(P[i]), 4, COL_JOINT, -1, cv2.LINE_AA)

        # caption bar
        if title:
            for j, line in enumerate(title.split("\n")[:2]):
                cv2.putText(canvas, line, (14, 26 + j * 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 70, 70), 1, cv2.LINE_AA)
        frames.append(canvas)

    # --- write MP4 (for download / high quality) ---
    # Only H.264 (avc1) is reliably playable inline in a browser <video>.
    # mp4v still writes a valid file that plays in VLC etc. and downloads fine.
    mp4_ok = False
    mp4_h264 = False
    if out_mp4:
        for fourcc_name in ("avc1", "mp4v"):
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
                vw = cv2.VideoWriter(out_mp4, fourcc, FPS, (CANVAS_W, CANVAS_H))
                if not vw.isOpened():
                    vw.release()
                    continue
                for fr in frames:
                    vw.write(fr)
                vw.release()
                if os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 0:
                    mp4_ok = True
                    mp4_h264 = (fourcc_name == "avc1")
                    print(f"  Saved MP4 ({fourcc_name}): {out_mp4}")
                    break
            except Exception as e:
                print(f"  (MP4 {fourcc_name} failed: {e})")

    # --- always also write a GIF (guaranteed inline playback) ---
    gif_ok = False
    if out_gif:
        try:
            import imageio
            rgb = [cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) for fr in frames]
            imageio.mimsave(out_gif, rgb, fps=FPS, loop=0)
            gif_ok = True
            print(f"  Saved GIF: {out_gif}")
        except Exception as e:
            print(f"  (GIF save failed: {e})")

    # --- build inline display HTML ---
    # Prefer an H.264 MP4 inline (crisp + scrubbable); otherwise show the GIF so
    # the notebook always animates, while the MP4 stays available for download.
    from IPython.display import HTML
    if mp4_h264:
        b64 = base64.b64encode(open(out_mp4, "rb").read()).decode()
        return HTML(f'<video controls autoplay loop style="max-width:420px">'
                    f'<source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>')
    if gif_ok:
        b64 = base64.b64encode(open(out_gif, "rb").read()).decode()
        note = "" if not mp4_ok else "<br><small>Higher-quality MP4 saved for download.</small>"
        return HTML(f'<img src="data:image/gif;base64,{b64}" style="max-width:420px">{note}')
    if mp4_ok:
        b64 = base64.b64encode(open(out_mp4, "rb").read()).decode()
        return HTML(f'<video controls autoplay loop style="max-width:420px">'
                    f'<source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>')
    return HTML("<p>Render produced no displayable output.</p>")


# ============================================================================
# STEP 6: DEMO — random (or custom) input -> sign language
# ============================================================================
print(f"\n{'=' * 70}")
print("Step 6: Running demo...")
print("=" * 70)

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

# Decide the query
if QUERY_TEXT is not None:
    query = QUERY_TEXT
    source = "custom"
    gt_stem = None
else:
    # random val sample
    gt_stem = random.choice(list(val_index.keys()))
    gt_entry = val_index[gt_stem]
    query = text_for(gt_entry) or gt_entry["sentence_id"]
    source = "random val sample"

print(f"\n  Query source: {source}")
print(f"  INPUT TEXT:   {query}")
if gt_stem:
    print(f"  (ground-truth val file: {gt_stem}, signer {val_index[gt_stem]['signer_id']})")

best_sid, method, ranked = retrieve(query)

print(f"\n  Retrieval method: {method}")
if ranked:
    print(f"  Top {len(ranked)} candidates:")
    for rank, (sid, score, txt) in enumerate(ranked, 1):
        marker = " <== chosen" if sid == best_sid else ""
        print(f"    {rank}. [{score:.3f}] {sid}: {txt}{marker}")

if best_sid is None:
    raise SystemExit("\n  No match found. Check that text/ids loaded correctly.")

stem, poses = pose_for_sentence(best_sid)
if poses is None:
    raise SystemExit(f"\n  Matched sentence {best_sid} but found no pose file.")

matched_text = corpus.get(best_sid, {}).get("text", "")
gloss = sign_gloss.get(best_sid, "")

print(f"\n  Matched sentence: {matched_text or best_sid}")
if gloss:
    print(f"  Sign gloss order: {gloss}")
print(f"  Retrieved pose file: {stem}")
print(f"  Signer: {train_index.get(stem, {}).get('signer_id', '?')}")
print(f"  Pose shape: {poses.shape}  (frames, landmarks, xyz)")
print(f"\n  Rendering sign-language animation...")

title = f"IN: {query}\nSIGN: {gloss or matched_text or best_sid}  ({method})"
out_mp4 = os.path.join(WORK_DIR, f"sign_{best_sid}.mp4")
out_gif = os.path.join(WORK_DIR, f"sign_{best_sid}.gif")
html = render_pose(poses, title=title, out_mp4=out_mp4, out_gif=out_gif)

# Display inline in the notebook
from IPython.display import display
display(html)

# ----------------------------------------------------------------------------
# Export landmarks to JSON for the 3D avatar viewer (avatar_viewer.html).
#
# IMPORTANT: the avatar retargeter (Kalidokit) needs RAW MediaPipe image
# landmarks (y-down, ~[0,1] range). Our train/val poses are neck-centered and
# per-frame shoulder-scaled, which breaks Kalidokit's 2D heuristics and makes
# the avatar's arms fly up. So for the avatar we export the ORIGINAL raw pose
# from the batch dataset (bdsl-mediapipe-final/batch_*/poses/<stem>.npy).
# The MP4/GIF skeleton above still uses the normalized poses — that's fine.
# ----------------------------------------------------------------------------
raw_poses = None
raw_candidates = glob.glob(os.path.join(INPUT_BASE, "**", f"{stem}.npy"), recursive=True)
# prefer a path under the raw batch dataset, not the normalized final dataset
for c in raw_candidates:
    low = c.lower()
    if "batch" in low and "poses" in low and "normalized" not in low:
        try:
            raw_poses = np.load(c)
            print(f"  Avatar will use RAW landmarks from: {c}")
            break
        except Exception:
            pass
if raw_poses is None:
    raw_poses = poses  # fallback: normalized (avatar may be less stable)
    print("  NOTE: raw landmarks not found; avatar JSON falls back to normalized "
          "poses (arms may be unstable). Add the 'bdsl-mediapipe-final' dataset "
          "to export raw landmarks for a clean avatar.")

out_json = os.path.join(WORK_DIR, f"sign_{best_sid}.json")
avatar_payload = {
    "sentence_id": str(best_sid),
    "text": matched_text,
    "gloss": gloss,
    "fps": FPS,
    "coords": "mediapipe_image_raw" if raw_poses is not poses else "normalized",
    "num_frames": int(raw_poses.shape[0]),
    "layout": {"pose": [0, 33], "left_hand": [33, 54],
               "right_hand": [54, 75], "face": [75, 543]},
    # round to 4 decimals -> much smaller JSON, no visible quality loss
    "frames": np.round(raw_poses.astype(np.float32), 4).tolist(),
}
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(avatar_payload, f, ensure_ascii=False)
print(f"  Saved avatar JSON: {out_json}  "
      f"({os.path.getsize(out_json) / 1024:.0f} KB)")

print(f"\n{'=' * 70}")
print("DONE. The animation above is the generated sign for your input.")
print(f"Downloadable files in {WORK_DIR}:")
print(f"    {os.path.basename(out_mp4)}   (skeleton video)")
print(f"    {os.path.basename(out_gif)}   (inline preview)")
print(f"    {os.path.basename(out_json)}  (feed this to avatar_viewer.html for a 3D avatar)")
print("=" * 70)
print("""
  Re-run tips:
    - Set QUERY_TEXT = "your Bangla sentence"  to sign custom text.
    - Set RANDOM_SEED = 42  for a reproducible random pick.
    - Set USE_LABSE = True (and add sentence-transformers) for semantic match.
""")

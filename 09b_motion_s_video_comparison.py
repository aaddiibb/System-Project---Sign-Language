# %% [markdown]
# # Motion-S — Phase 9b: Video and Smoothness Comparison (GT vs Phase 8 vs Phase 9)
#
# **Inference / visualization only.** Renders held-out test sentences side by side as 55-joint skeleton videos
# (ground truth, Phase-8 generator, Phase-9 concatenative synthesis) and measures smoothness on reconstructed 3D joints.
#
# Fairness rules:
# * The Phase-9 synthesizer is rebuilt from the **training split only**, so the shown test sentences are never copied.
# * All three motions use the ground-truth duration (oracle length) so the panels stay time-aligned.
# * "Smoother" is judged **relative to ground truth**: a frozen skeleton has zero jerk but is not better motion.
#   Each metric is reported next to the GT value; closer to GT is better.
#
# Skeleton reconstruction uses the same offline Kiseki package (commit 07fd2bc) as the final demo.

# %%
from __future__ import annotations

import ast
import base64
import io
import json
import os
import re
import sys
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from scipy.sparse.linalg import lsqr
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from IPython.display import HTML, Image, Video, display
except ImportError:
    display = print
    Video = Image = HTML = None

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
EXPECTED_RVQ_SHA = "0329cf9197f3ba8088e79ce519c422b97ba118ae8046f3dc608c6d4e1904b25f"
LEVELS = ["base_tokens", "residual_1", "residual_2", "residual_3", "residual_4", "residual_5"]
FPS = 30

# Change these to inspect other held-out sentences (Phase-1 test split IDs) or another custom sentence.
SAMPLE_IDS = ["1002815", "1012610", "1042390"]
CUSTOM_SENTENCE = "There is a book on the table"
# True: zero the global root rotation/translation velocities (features 0-2) before building the skeleton, so the
# signer always faces the camera and stays in place. Arm, hand, and finger motion is unchanged.
LOCK_FACING = True
SMOOTHNESS_SAMPLES = int(os.environ.get("MOTION_S_SMOOTHNESS_SAMPLES", 200))  # length-stratified test subset for the aggregate smoothness table

ON_KAGGLE = Path("/kaggle/input").exists()
SEARCH = [Path("/kaggle/input"), Path("/kaggle/working")] if ON_KAGGLE else [Path(os.environ.get("MOTION_S_ROOT", Path.cwd()))] + [Path(x) for x in os.environ.get("MOTION_S_EXTRA_ROOTS", "").split(os.pathsep) if x]
OUT = Path("/kaggle/working/motion_s_phase9b") if ON_KAGGLE else Path(os.environ.get("MOTION_S_PHASE9B_OUT", Path.cwd() / "Outputs/Phase 9b/motion_s_phase9b"))
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Kaggle:", ON_KAGGLE, "| device:", DEVICE, "| output:", OUT)


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def find_dirs(predicate, label, required=True):
    found = []
    for base in SEARCH:
        if not base.exists():
            continue
        for current, dirs, _ in os.walk(base):
            candidate = Path(current)
            if candidate.resolve() == OUT.resolve():
                dirs[:] = []
                continue
            if predicate(candidate):
                found.append(candidate.resolve())
                dirs[:] = []
    found = sorted(dict.fromkeys(found), key=lambda p: (len(p.parts), str(p)))
    if not found and required:
        raise RuntimeError(f"Missing {label}")
    print(f"{label}: {found[0] if found else 'not attached'}")
    return found[0] if found else None


def find_file(name, required=True):
    found = sorted({p.resolve() for base in SEARCH if base.exists() for p in base.rglob(name) if p.is_file()}, key=str)
    if not found and required:
        raise RuntimeError(f"Missing {name}")
    return found[0] if found else None


P1 = find_dirs(lambda p: (p / "dataset_config.json").is_file() and (p / "splits/train_split.csv").is_file(), "Phase 1")
P3 = find_dirs(lambda p: all((p / x).is_file() for x in ("phase3_report.json", "best_model/config.json", "tokenizer/tokenizer_config.json")), "Phase 3")
P5 = find_dirs(lambda p: all((p / "diagnostics" / x).is_file() for x in ("motion_mean_668.npy", "motion_std_668.npy")), "Phase 5B normalization")
P8 = find_dirs(lambda p: (p / "phase8_report.json").is_file() and (p / "decoded/final_oracle").is_dir(), "Phase 8 decoded outputs", required=False)
P9 = find_dirs(lambda p: (p / "phase9_report.json").is_file() and (p / "diagnostics/selection.json").is_file(), "Phase 9 outputs", required=False)
FEATURE_DIR = find_dirs(lambda p: re.sub(r"[^a-z0-9]", "", p.name.casefold()) == "motionfeatures" and any(p.glob("*.npy")), "Motion-Features (real GT)", required=False)
SBERT_DIR = find_dirs(lambda p: (p / "modules.json").is_file() and (p / "config_sentence_transformers.json").is_file() and "minilm" in str(p).lower(), "MiniLM sentence encoder", required=False)
RVQ_CKPT = find_file("rvq_vae_best.pth")
selection = read_json(P9 / "diagnostics/selection.json") if P9 else {"max_ngram": 8, "medoid_pool": 64, "join_radius": 1}
# Unseen-gloss fallback: closest meaning (MiniLM, validated threshold) when available, otherwise closest spelling.
SEMANTIC_THRESHOLD = selection.get("semantic_threshold", 0.6) if SBERT_DIR is not None else None
print("Phase-9 frozen hyperparameters:", {k: selection[k] for k in ("max_ngram", "medoid_pool", "join_radius")}, "| semantic threshold:", SEMANTIC_THRESHOLD)

# %% [markdown]
# ## Offline Kiseki skeleton renderer
#
# The notebook build embeds the exact Kiseki package as base64 (no internet needed). When running the `.py` locally,
# the vendored `.tmp-kiseki` folder is used instead.

# %%
KISEKI_ARCHIVE_B64 = "__KISEKI_ARCHIVE_B64__"
vendor_root = OUT / "_offline_vendor"
if not KISEKI_ARCHIVE_B64.startswith("__"):
    if not (vendor_root / "kiseki/sample.bvh").is_file():
        vendor_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(KISEKI_ARCHIVE_B64))) as archive:
            archive.extractall(vendor_root)
    sys.path.insert(0, str(vendor_root))
else:
    vendored = (".tmp-kiseki", "Outputs/Last final demo run/motion_s_final_demo/_offline_vendor")
    local = next((b / v for b in SEARCH for v in vendored if (b / v / "kiseki/sample.bvh").is_file()), None)
    if local is None:
        raise RuntimeError("Kiseki unavailable: build the notebook with build_phase9b_notebook.py")
    sys.path.insert(0, str(local))
import kiseki
from kiseki import visualize
from kiseki.core import parse_bvh_hierarchy, reconstruct_positions

BVH = Path(kiseki.__file__).parent / "sample.bvh"
JOINT_NAMES, PARENTS = parse_bvh_hierarchy(str(BVH))
NJ = len(JOINT_NAMES)
J = {name: i for i, name in enumerate(JOINT_NAMES)}
WRISTS = [J["left_wrist"], J["right_wrist"]]
FINGERTIPS = [J[f"{side}_{finger}3"] for side in ("left", "right") for finger in ("index", "middle", "pinky", "ring", "thumb")]
HAND_JOINTS = list(range(J["left_wrist"], J["left_thumb3"] + 1)) + list(range(J["right_wrist"], J["right_thumb3"] + 1))
print(NJ, "joints")


def lock_facing(motion):
    """Remove whole-body turning and travel: root angular velocity (0) and root linear XZ velocity (1, 2)."""
    motion = np.array(motion, dtype=np.float32, copy=True)
    if LOCK_FACING:
        motion[:, 0:3] = 0.0
    return motion


def joints(motion):
    positions, _ = reconstruct_positions(lock_facing(motion).astype(np.float64), NJ, PARENTS)
    return positions  # [N,55,3]

# %% [markdown]
# ## Data, RVQ decoder, and the Phase-9 synthesizer (training split only)

# %%
def parse_cell(value):
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = [int(x) for x in re.split(r"[\s,]+", text) if x]
    return np.asarray(parsed, dtype=np.int64)


def load_split(name):
    file = {"train": "train_split.csv", "test": "test_split.csv"}[name]
    frame = pd.read_csv(P1 / "splits" / file, dtype={"id": str}, low_memory=False)
    frame["tokens"] = [np.stack([parse_cell(row[c]) for c in LEVELS]) for _, row in frame.iterrows()]
    frame["T"] = frame.tokens.map(lambda t: t.shape[1])
    return frame[frame["T"] <= 512].reset_index(drop=True)


train_frame, test_frame = load_split("train"), load_split("test")
test_index = test_frame.set_index("id")


class MotionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 512, 3, 2, 1, output_padding=1), nn.ReLU(), nn.BatchNorm1d(512),
            nn.ConvTranspose1d(512, 512, 3, 2, 1, output_padding=1), nn.ReLU(), nn.BatchNorm1d(512),
            nn.ConvTranspose1d(512, 512, 3, 1, 1), nn.ReLU(), nn.BatchNorm1d(512),
            nn.ConvTranspose1d(512, 668, 3, 1, 1))

    def forward(self, x):
        return self.decoder(x)


payload = torch.load(RVQ_CKPT, map_location="cpu", weights_only=True)
for key in ("model_state_dict", "state_dict"):
    if isinstance(payload, dict) and key in payload:
        payload = payload[key]
        break
rvq_state = {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}
decoder = MotionDecoder()
decoder.load_state_dict({k[8:]: v for k, v in rvq_state.items() if k.startswith("decoder.")}, strict=True)
decoder = decoder.to(DEVICE).eval()
codebooks = torch.stack([rvq_state[f"rvq.quantizers.{q}.embedding"].float() for q in range(6)]).to(DEVICE)
motion_mean = torch.from_numpy(np.load(P5 / "diagnostics/motion_mean_668.npy").astype(np.float32)).to(DEVICE)
motion_std_np = np.load(P5 / "diagnostics/motion_std_668.npy").astype(np.float32)
motion_std = torch.from_numpy(np.where(motion_std_np >= 1e-6, motion_std_np, 1.0).astype(np.float32)).to(DEVICE)


@torch.inference_mode()
def decode_tokens(tokens, joins=(), radius=0):
    ids = torch.as_tensor(np.asarray(tokens), dtype=torch.long, device=DEVICE)
    latent = torch.zeros((256, ids.shape[1]), device=DEVICE)
    for q in range(6):
        latent += codebooks[q][:, ids[q]]
    T = latent.shape[1]
    for j in (joins if radius > 0 else ()):
        a, b = max(0, j - radius - 1), min(T - 1, j + radius)
        if b - a >= 2:
            w = torch.linspace(0, 1, b - a + 1, device=DEVICE)
            latent[:, a:b + 1] = latent[:, a:a + 1] * (1 - w) + latent[:, b:b + 1] * w
    return (decoder(latent[None]).squeeze(0).T * motion_std + motion_mean).cpu().numpy().astype(np.float32)


WORD = re.compile(r"[A-Za-z0-9\-']+")


def ext_units(gloss):
    tail = re.sub(r"[A-Za-z0-9\-' ]", "", str(gloss))[-3:]
    return ["<s>"] + WORD.findall(str(gloss).upper()) + ["</s>" + ("?" if "?" in tail else "//")]


class ConcatSynth:
    """Identical to the Phase-9 synthesizer (09_motion_s_concatenative_synthesis.py)."""

    def __init__(self, glosses, tokens, max_ngram=8, medoid_pool=64, seed=42, damp=0.05):
        self.max_ngram, self.medoid_pool, self.seed = max_ngram, medoid_pool, seed
        self.tokens = [np.asarray(t, dtype=np.int64) for t in tokens]
        self.units = [ext_units(g) for g in glosses]
        vocab = sorted({u for us in self.units for u in us})
        self.vid = {u: i for i, u in enumerate(vocab)}
        rows = [r for r, us in enumerate(self.units) for _ in us]
        cols = [self.vid[u] for us in self.units for u in us]
        X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(self.units), len(vocab)))
        y = np.array([t.shape[1] for t in self.tokens], float)
        self.duration = np.maximum(lsqr(X, y, damp=damp, atol=1e-10, btol=1e-10, iter_lim=20000)[0], 1.0)
        self.word_vocab = [u for u in vocab if not u.startswith("<")]
        self.bounds = []
        for us, t in zip(self.units, self.tokens):
            b = np.concatenate([[0.0], np.cumsum([self.duration[self.vid[u]] for u in us])])
            b = np.rint(b * (t.shape[1] / b[-1])).astype(int)
            b[-1] = t.shape[1]
            self.bounds.append(b)
        self.index = defaultdict(list)
        for s, us in enumerate(self.units):
            for i in range(len(us)):
                for j in range(i + 1, min(len(us), i + max_ngram) + 1):
                    self.index[tuple(us[i:j])].append((s, i, j))
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(self.word_vocab)
        self.char_matrix = self.char_vectorizer.transform(self.word_vocab)
        self._medoids, self._nearest = {}, {}
        self.encoder = None

    def set_semantic(self, encoder, threshold):
        self.encoder, self.semantic_threshold, self._nearest = encoder, threshold, {}
        if encoder is not None:
            self.vocab_embeddings = encoder.encode([w.lower() for w in self.word_vocab], normalize_embeddings=True, batch_size=256)
        return self

    def nearest_word(self, word):
        if word not in self._nearest:
            choice = None
            if self.encoder is not None:
                similarity = self.vocab_embeddings @ self.encoder.encode([word.lower()], normalize_embeddings=True)[0]
                if similarity.max() >= self.semantic_threshold:
                    choice = self.word_vocab[int(np.argmax(similarity))]
            if choice is None:
                similarity = (self.char_vectorizer.transform([word]) @ self.char_matrix.T).toarray()[0]
                choice = self.word_vocab[int(np.argmax(similarity))]
            self._nearest[word] = choice
        return self._nearest[word]

    def span(self, s, i, j):
        return self.tokens[s][:, self.bounds[s][i]:self.bounds[s][j]]

    def pick(self, key):
        if key in self._medoids:
            return self._medoids[key]
        occurrences = self.index[key]
        if len(occurrences) > self.medoid_pool:
            local = np.random.default_rng(self.seed + zlib.crc32(" | ".join(key).encode("utf-8")))
            occurrences = [occurrences[k] for k in local.choice(len(occurrences), self.medoid_pool, replace=False)]
        spans = [s for s in (self.span(*o) for o in occurrences) if s.shape[1] > 0] or [self.span(*occurrences[0])]
        if len(spans) <= 2:
            best = spans[0]
        else:
            n = int(np.median([s.shape[1] for s in spans]))
            stacked = np.stack([s[:, np.linspace(0, s.shape[1] - 1, n).round().astype(int)] for s in spans])
            best = spans[int(np.argmax([(stacked == stacked[k]).mean(axis=(1, 2)).sum() for k in range(len(stacked))]))]
        self._medoids[key] = best
        return best

    def synthesize(self, gloss):
        us = [u if u in self.vid else self.nearest_word(u) for u in ext_units(gloss)]
        self.last_substitutions = {u: self.nearest_word(u) for u in ext_units(gloss) if u not in self.vid}
        pieces, i = [], 0
        while i < len(us):
            j = min(len(us), i + self.max_ngram)
            while j > i + 1 and tuple(us[i:j]) not in self.index:
                j -= 1
            piece = self.pick(tuple(us[i:j]))
            if piece.shape[1] > 0:
                pieces.append(piece)
            i = j
        return np.concatenate(pieces, axis=1), np.cumsum([p.shape[1] for p in pieces])[:-1].tolist()


synth = ConcatSynth(train_frame.gloss.astype(str).tolist(), train_frame.tokens.tolist(), selection["max_ngram"], selection["medoid_pool"])
if SEMANTIC_THRESHOLD is not None:
    from sentence_transformers import SentenceTransformer
    synth.set_semantic(SentenceTransformer(str(SBERT_DIR), device=str(DEVICE)), SEMANTIC_THRESHOLD)
RADIUS = int(selection["join_radius"])


def phase9_motion(gloss, T=None):
    tokens, joins = synth.synthesize(gloss)
    if T is not None and tokens.shape[1] != T:  # oracle length: nearest-index resample, junctions rescaled
        scale = T / tokens.shape[1]
        tokens = tokens[:, np.linspace(0, tokens.shape[1] - 1, T).round().astype(int)]
        joins = [int(round(j * scale)) for j in joins]
    return decode_tokens(tokens, joins, RADIUS)


def ground_truth_motion(sid):
    if FEATURE_DIR is not None and (FEATURE_DIR / f"{sid}.npy").is_file():
        return np.load(FEATURE_DIR / f"{sid}.npy").astype(np.float32), "real Motion-Features"
    return decode_tokens(test_index.tokens[sid]), "RVQ reconstruction of GT tokens"


def phase8_motion(sid):
    path = P8 / "decoded/final_oracle" / f"{sid}.npy" if P8 else None
    return np.load(path).astype(np.float32) if path is not None and path.is_file() else None

# %% [markdown]
# ## Smoothness metrics on reconstructed 3D joints
#
# * **jerk** — mean |d³x/dt³| over all joints (lower = smoother; GT is the target, not zero)
# * **hand jerk** — same, on the 32 hand joints (finger jitter shows up here)
# * **wrist speed** — mean wrist speed; collapsed/frozen motion has very low values
# * **frozen fraction** — share of frames where both wrists move < 1 unit/s

# %%
def smoothness(motion):
    p = joints(motion)
    vel = np.diff(p, axis=0) * FPS
    jerk = np.linalg.norm(np.diff(p, 3, axis=0), axis=-1) * FPS ** 3
    wrist_speed = np.linalg.norm(vel[:, WRISTS], axis=-1)
    return {"jerk": float(jerk.mean()), "hand_jerk": float(jerk[:, HAND_JOINTS].mean()), "wrist_speed": float(wrist_speed.mean()),
            "frozen_fraction": float((wrist_speed.max(axis=1) < 1.0).mean())}

# %% [markdown]
# ## Side-by-side held-out videos (GT | Phase 8 | Phase 9)

# %%
BONES = [(i, p) for i, p in enumerate(PARENTS) if p >= 0]
HAND_SET = set(HAND_JOINTS)
UPPER_BODY = list(range(J["spine1"], NJ))
BONES = [(i, p) for i, p in BONES if i in UPPER_BODY]


def side_by_side(panels, path, title):
    """Front-view (x/y) skeleton animation, pelvis-centred, one column per motion."""
    projected = []
    for _, motion in panels:
        p = joints(motion)
        p = p - p[:, :1, :] * np.array([1, 0, 1])  # remove root drift horizontally
        projected.append(p)
    n = min(len(p) for p in projected)
    upper = np.concatenate([p[:n, UPPER_BODY] for p in projected])  # crop to the signing space (spine and above)
    xlim = (upper[..., 0].min() - 5, upper[..., 0].max() + 5)
    ylim = (upper[..., 1].min() - 5, upper[..., 1].max() + 5)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 5.6))
    fig.subplots_adjust(top=0.80, wspace=0.05)
    artists = []
    for ax, (label, _) in zip(axes, panels):
        ax.set(xlim=xlim, ylim=ylim, aspect="equal", xticks=[], yticks=[])
        ax.set_title(label, fontsize=11)
        lines = [ax.plot([], [], color="tab:red" if i in HAND_SET else "tab:blue", lw=1.2 if i in HAND_SET else 2)[0] for i, _ in BONES]
        trail = [ax.plot([], [], color="tab:orange", lw=0.8, alpha=0.6)[0] for _ in WRISTS]
        artists.append((lines, trail))
    fig.suptitle(title, fontsize=11, y=0.98)

    def update(frame):
        changed = []
        for p, (lines, trail) in zip(projected, artists):
            for line, (i, parent) in zip(lines, BONES):
                line.set_data([p[frame, i, 0], p[frame, parent, 0]], [p[frame, i, 1], p[frame, parent, 1]])
            for t, w in zip(trail, WRISTS):
                lo = max(0, frame - 30)
                t.set_data(p[lo:frame + 1, w, 0], p[lo:frame + 1, w, 1])
            changed += lines + trail
        return changed

    anim = FuncAnimation(fig, update, frames=n, interval=1000 / FPS, blit=True)
    if matplotlib.animation.writers.is_available("ffmpeg"):
        path = path.with_suffix(".mp4")
        anim.save(str(path), writer=FFMpegWriter(fps=FPS, bitrate=2000, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"]), dpi=90)
    else:
        path = path.with_suffix(".gif")
        anim.save(str(path), writer=PillowWriter(fps=FPS // 2), dpi=60)
    plt.close(fig)
    return path


sample_rows = []
for sid in SAMPLE_IDS:
    if sid not in test_index.index:
        print("Skipping", sid, "(not in the supported test split)")
        continue
    row = test_index.loc[sid]
    gt, gt_source = ground_truth_motion(sid)
    panels = [(f"Ground truth", gt)]
    p8 = phase8_motion(sid)
    if p8 is not None:
        panels.append(("Phase 8 generator", p8))
    p9 = phase9_motion(row.gloss, int(row["T"]))
    panels.append(("Phase 9 concatenative", p9))
    for label, motion in panels:
        np.save(OUT / f"{sid}_{re.sub(r'[^a-z0-9]+', '_', label.lower())}.npy", motion, allow_pickle=False)
        sample_rows.append({"sample_id": sid, "motion": label, **smoothness(motion)})
    video = side_by_side(panels, OUT / f"compare_{sid}", f"{sid}: \"{row.sentence}\"\nGloss: {row.gloss}  (GT = {gt_source})")
    print("Saved", video)
    if Video is not None and video.suffix == ".mp4":
        display(Video(str(video), embed=True, width=1000))
    elif Image is not None:
        display(Image(filename=str(video)))
sample_table = pd.DataFrame(sample_rows)
sample_table.to_csv(OUT / "smoothness_selected_samples.csv", index=False)
display(sample_table.round(3))

# %% [markdown]
# ## Aggregate smoothness over a length-stratified test subset
#
# Ratios are relative to ground truth (1.0 = GT-like). A ratio far below 1 for wrist speed together with a high frozen
# fraction indicates collapsed motion; a ratio far above 1 for jerk indicates jitter.

# %%
subset = test_frame.sort_values("T", kind="stable").iloc[np.linspace(0, len(test_frame) - 1, SMOOTHNESS_SAMPLES).round().astype(int)]
rows = []
for number, (sid, gloss, T) in enumerate(zip(subset.id, subset.gloss, subset["T"]), 1):
    rows.append({"sample_id": sid, "motion": "ground_truth", **smoothness(ground_truth_motion(sid)[0])})
    p8 = phase8_motion(sid)
    if p8 is not None:
        rows.append({"sample_id": sid, "motion": "phase8", **smoothness(p8)})
    rows.append({"sample_id": sid, "motion": "phase9", **smoothness(phase9_motion(gloss, int(T)))})
    if number % 50 == 0:
        print("smoothness", number, "/", len(subset), flush=True)
detail = pd.DataFrame(rows)
detail.to_csv(OUT / "smoothness_detail.csv", index=False)
summary = detail.groupby("motion")[["jerk", "hand_jerk", "wrist_speed", "frozen_fraction"]].median()
for column in ("jerk", "hand_jerk", "wrist_speed"):
    summary[f"{column}_ratio_to_gt"] = summary[column] / summary.loc["ground_truth", column]
summary.to_csv(OUT / "smoothness_summary.csv")
display(summary.round(3))

# %% [markdown]
# ## Custom English sentence → Phase 9 motion → Kiseki video
#
# Uses the Phase-3 text→gloss model, then the Phase-9 synthesizer (own lexicon length).

# %%
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(P3 / "tokenizer", local_files_only=True, use_fast=True)
phase3_model = AutoModelForSeq2SeqLM.from_pretrained(P3 / "best_model", local_files_only=True).to(DEVICE).eval()
beams = 4 if read_json(P3 / "phase3_report.json").get("selected_decoding") == "beam_4" else 1
with torch.inference_mode():
    encoded = tokenizer("text to gloss: " + CUSTOM_SENTENCE.strip(), truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
    custom_gloss = tokenizer.decode(phase3_model.generate(**encoded, max_length=128, num_beams=beams, do_sample=False)[0], skip_special_tokens=True).strip()
custom_motion = phase9_motion(custom_gloss)
np.save(OUT / "custom_phase9_motion.npy", custom_motion, allow_pickle=False)  # raw synthesized motion
custom_path = OUT / "custom_phase9_motion_view.npy"  # copy rendered by Kiseki (facing locked if LOCK_FACING)
np.save(custom_path, lock_facing(custom_motion), allow_pickle=False)
print(f"'{CUSTOM_SENTENCE}' -> {custom_gloss} -> {custom_motion.shape}")
print("Unseen glosses replaced:", synth.last_substitutions or "none (all signs known)")
print("Smoothness:", {k: round(v, 3) for k, v in smoothness(custom_motion).items()})
use_mp4 = matplotlib.animation.writers.is_available("ffmpeg")
custom_video = visualize(custom_path, output_path=OUT / ("custom_phase9_3d.mp4" if use_mp4 else "custom_phase9_3d.gif"), bvh_path=BVH, fixed_view="front", fps=FPS,
                         tracking=True, title=f"Phase 9: {CUSTOM_SENTENCE}", trails="fingertips", trail_length=30, save_grid=True, grid_frames=9, display=False)
print("Saved", custom_video)
if Video is not None and use_mp4:
    display(Video(str(custom_video), embed=True, width=900))

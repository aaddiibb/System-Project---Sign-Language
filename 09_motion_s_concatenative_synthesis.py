# %% [markdown]
# # Motion-S / SignVRSE — Phase 9: Gloss-Aligned Concatenative Motion Synthesis
#
# **Why this phase exists.** Post-hoc analysis of the Phase 1–8 outputs showed:
#
# 1. The Phase-8 generator barely beats non-learned TF-IDF retrieval in feature space and collapses on ~50% of samples.
#    Its teacher-forced level-0 accuracy (81%) is largely copy-previous behaviour (copy-previous alone scores 62.5%).
# 2. Motion length is almost exactly **additive over gloss units**: a per-gloss duration lexicon fits training lengths
#    with ~0.65-token residual MAE, versus 12-token MAE for the Phase-6 T5 length head.
# 3. Sentences sharing a gloss share **identical 6-level RVQ token spans** (e.g. a median 20-token identical prefix
#    for sentences starting with HE), and ~51% of validation token positions are covered by exact 8-token training spans.
#
# The corpus therefore behaves like concatenated per-sign clips. This phase replaces autoregressive token generation
# with **unit selection** (as in concatenative TTS / sign-language production): segment every training sequence with the
# duration lexicon, then synthesize a new sequence by copying the longest matching multi-gloss spans (medoid variant),
# and crossfade the RVQ latent at junctions.
#
# Selection policy: all hyperparameters are chosen on **validation only**; test is evaluated once after freezing.
# Metrics use the exact Phase-8 definitions. If the Motion-Features directory is attached, motion metrics use real
# ground truth (directly comparable to Phase 8); otherwise they use the RVQ reconstruction of GT tokens as a proxy
# (correlation 0.99 with real-feature MAE on Phase-8 outputs) and the report says so.
# No skeleton/BVH/MPJPE or semantic sign-correctness claim is made.

# %%
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import pickle
import random
import re
import sys
import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import lsqr
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from IPython.display import display
except ImportError:  # plain-script execution
    display = print

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
FINGERPRINT = "e2acdf4d6dce91775239ad50be3866dbc6f5da94d940204b62456cdb77fc4b0d"
EXPECTED_RVQ_SHA = "0329cf9197f3ba8088e79ce519c422b97ba118ae8046f3dc608c6d4e1904b25f"
EXPECTED_SUPPORTED = {"train": 9936, "validation": 1243, "test": 1243}
LEVELS = ["base_tokens", "residual_1", "residual_2", "residual_3", "residual_4", "residual_5"]


@dataclass(frozen=True)
class Config:
    seed: int = 42
    max_t: int = 512
    # Validation-only search grids
    max_ngram_candidates: tuple[int, ...] = (2, 3, 4, 8)
    medoid_pool_candidates: tuple[int, ...] = (8, 24, 64)
    join_radius_candidates: tuple[int, ...] = (0, 1, 2, 3)
    # Unseen-gloss fallback: closest-meaning known gloss (MiniLM) when cosine >= threshold, else closest spelling.
    semantic_threshold_candidates: tuple[float, ...] = (0.5, 0.6, 0.7)
    motion_selection_samples: int = 256
    near_zero_std: float = 1e-6
    lsqr_damp: float = 0.05
    save_decoded_test: bool = False


CFG = Config()
ON_KAGGLE = Path("/kaggle/input").exists()
SEARCH = [Path("/kaggle/input"), Path("/kaggle/working")] if ON_KAGGLE else [Path(os.environ.get("MOTION_S_ROOT", Path.cwd()))] + [Path(x) for x in os.environ.get("MOTION_S_EXTRA_ROOTS", "").split(os.pathsep) if x]
OUT = Path("/kaggle/working/motion_s_phase9") if ON_KAGGLE else Path(os.environ.get("MOTION_S_PHASE9_OUT", Path.cwd() / "Outputs/Phase 9/motion_s_phase9"))
DIAG, RESULTS, FIGURES, TOKENS, FINAL = (OUT / "diagnostics", OUT / "results", OUT / "figures", OUT / "tokens", OUT / "final_pipeline")
for path in (OUT, DIAG, RESULTS, FIGURES, TOKENS, FINAL):
    path.mkdir(parents=True, exist_ok=True)
random.seed(CFG.seed)
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Python", sys.version.split()[0], "| PyTorch", torch.__version__, "| device", DEVICE, "| Kaggle", ON_KAGGLE)
print("Search roots:", SEARCH, "\nOutput:", OUT)
display(pd.Series(asdict(CFG), name="value").to_frame())


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    return value


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(safe(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

# %% [markdown]
# ## 9.0 — Artifact discovery
#
# Required: Phase-1 splits, provider RVQ checkpoint (SHA-verified), Phase-5B normalization, Phase-3 text→gloss model and
# predictions. Optional: Motion-Features (real-GT motion metrics), Phase-8 report (comparison), competition `test.csv`
# (submission).

# %%
def find_dirs(predicate, label, required=True):
    found = []
    for base in SEARCH:
        if not base.exists():
            continue
        for current, dirs, _ in os.walk(base):
            candidate = Path(current)
            if OUT.resolve() == candidate.resolve():
                dirs[:] = []
                continue
            if predicate(candidate):
                found.append(candidate.resolve())
                dirs[:] = []
    found = sorted(dict.fromkeys(found), key=lambda p: (len(p.parts), str(p)))
    if not found and required:
        raise RuntimeError(f"Missing {label}")
    if found:
        print(f"{label}: {found[0]}" + (f"  (+{len(found) - 1} other candidates)" if len(found) > 1 else ""))
    return found[0] if found else None


def find_file(name, expected_sha=None, required=True):
    found = sorted({p.resolve() for base in SEARCH if base.exists() for p in base.rglob(name) if p.is_file()}, key=str)
    if expected_sha:
        found = [p for p in found if sha(p) == expected_sha]
    if not found and required:
        raise RuntimeError(f"Missing {name}")
    return found[0] if found else None


P1 = find_dirs(lambda p: (p / "dataset_config.json").is_file() and (p / "splits/train_split.csv").is_file(), "Phase 1")
P3 = find_dirs(lambda p: all((p / x).is_file() for x in ("phase3_report.json", "best_model/config.json", "tokenizer/tokenizer_config.json", "results/test_predictions.csv", "results/validation_predictions.csv")), "Phase 3")
P5 = find_dirs(lambda p: all((p / "diagnostics" / x).is_file() for x in ("motion_mean_668.npy", "motion_std_668.npy", "empirical_motion_thresholds.json")), "Phase 5B normalization")
P8 = find_dirs(lambda p: (p / "phase8_report.json").is_file(), "Phase 8", required=False)
SBERT_DIR = find_dirs(lambda p: (p / "modules.json").is_file() and (p / "config_sentence_transformers.json").is_file() and "minilm" in str(p).lower(), "MiniLM sentence encoder", required=False)
FEATURE_DIR = find_dirs(lambda p: re.sub(r"[^a-z0-9]", "", p.name.casefold()) == "motionfeatures" and any(p.glob("*.npy")), "Motion-Features", required=False)
COMPETITION = find_dirs(lambda p: (p / "test.csv").is_file() and (p / "train.csv").is_file(), "Competition data", required=False)
RVQ_CKPT = find_file("rvq_vae_best.pth", EXPECTED_RVQ_SHA)
P1CFG = read_json(P1 / "dataset_config.json")
if P1CFG.get("split_fingerprint_sha256") != FINGERPRINT:
    raise RuntimeError("Phase-1 fingerprint mismatch")
REFERENCE = "real_motion_features" if FEATURE_DIR else "rvq_reconstruction_proxy"
print("Motion-metric reference:", REFERENCE)
write_json(DIAG / "artifact_verification.json", {"phase1": P1, "phase3": P3, "phase5b": P5, "phase8": P8, "motion_features": FEATURE_DIR,
                                                  "competition": COMPETITION, "rvq_checkpoint": RVQ_CKPT, "rvq_sha256": EXPECTED_RVQ_SHA,
                                                  "motion_metric_reference": REFERENCE, "phase1_fingerprint": FINGERPRINT})

# %% [markdown]
# ## 9.1 — Locked data and verified decoder

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
    array = np.asarray(parsed, dtype=np.int64)
    if array.ndim != 1 or array.size == 0 or array.min() < 0 or array.max() > 511:
        raise ValueError("invalid token cell")
    return array


def load_split(name):
    file = {"train": "train_split.csv", "validation": "val_split.csv", "test": "test_split.csv"}[name]
    frame = pd.read_csv(P1 / "splits" / file, dtype={"id": str}, low_memory=False)
    frame["tokens"] = [np.stack([parse_cell(row[c]) for c in LEVELS]) for _, row in frame.iterrows()]
    frame["T"] = frame.tokens.map(lambda t: t.shape[1])
    frame = frame[frame["T"] <= CFG.max_t].reset_index(drop=True)
    if len(frame) != EXPECTED_SUPPORTED[name]:
        raise RuntimeError(f"{name}: {len(frame)} supported rows, expected {EXPECTED_SUPPORTED[name]}")
    return frame


train_frame, val_frame, test_frame = load_split("train"), load_split("validation"), load_split("test")
phase3_val = pd.read_csv(P3 / "results/validation_predictions.csv", dtype={"sample_id": str}).set_index("sample_id").predicted_gloss
phase3_test = pd.read_csv(P3 / "results/test_predictions.csv", dtype={"sample_id": str}).set_index("sample_id").predicted_gloss
val_frame["predicted_gloss"] = val_frame.id.map(phase3_val).fillna("").astype(str)
test_frame["predicted_gloss"] = test_frame.id.map(phase3_test).fillna("").astype(str)
print({k: len(v) for k, v in {"train": train_frame, "validation": val_frame, "test": test_frame}.items()})


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
motion_mean = np.load(P5 / "diagnostics/motion_mean_668.npy").astype(np.float32)
motion_std = np.load(P5 / "diagnostics/motion_std_668.npy").astype(np.float32)
active = motion_std >= CFG.near_zero_std
mean_t, std_t = torch.from_numpy(motion_mean).to(DEVICE), torch.from_numpy(np.where(active, motion_std, 1.0).astype(np.float32)).to(DEVICE)
thresholds = read_json(P5 / "diagnostics/empirical_motion_thresholds.json")
feature_index = {p.stem: p for p in FEATURE_DIR.glob("*.npy")} if FEATURE_DIR else {}


def tokens_to_latent(tokens):
    ids = torch.as_tensor(np.asarray(tokens), dtype=torch.long, device=DEVICE)
    latent = torch.zeros((256, ids.shape[1]), device=DEVICE)
    for q in range(6):
        latent += codebooks[q][:, ids[q]]
    return latent


def smooth_joins(latent, joins, radius):
    """Linear crossfade of the summed RVQ latent [256,T] across each copied-span junction."""
    if radius <= 0 or not joins:
        return latent
    latent = latent.clone()
    T = latent.shape[1]
    for j in joins:
        a, b = max(0, j - radius - 1), min(T - 1, j + radius)
        if b - a < 2:
            continue
        w = torch.linspace(0, 1, b - a + 1, device=latent.device)
        latent[:, a:b + 1] = latent[:, a:a + 1] * (1 - w) + latent[:, b:b + 1] * w
    return latent


@torch.inference_mode()
def decode_latent(latent):
    motion = (decoder(latent[None]).squeeze(0).T * std_t + mean_t).cpu().numpy().astype(np.float32)
    if motion.shape != (4 * latent.shape[1], 668) or not np.isfinite(motion).all():
        raise RuntimeError("decode invariant failed")
    return motion


def decode_tokens(tokens, joins=(), radius=0):
    return decode_latent(smooth_joins(tokens_to_latent(tokens), list(joins), radius))


def reference_motion(sample_id, gt_tokens):
    if feature_index:
        return np.load(feature_index[str(sample_id)], allow_pickle=False).astype(np.float32)
    return decode_tokens(gt_tokens)

# %% [markdown]
# ## 9.2 — Dataset-structure diagnostics (training split only)

# %%
WORD = re.compile(r"[A-Za-z0-9\-']+")


def words(gloss):
    return WORD.findall(str(gloss).upper())


def end_marker(gloss):
    tail = re.sub(r"[A-Za-z0-9\-' ]", "", str(gloss))[-3:]
    return "?" if "?" in tail else "//"


def ext_units(gloss):
    return ["<s>"] + words(gloss) + ["</s>" + end_marker(gloss)]


def common_prefix(a, b):
    n = min(a.shape[1], b.shape[1])
    equal = (a[:, :n] == b[:, :n]).all(0)
    return n if equal.all() else int(np.argmin(equal))


rng = np.random.default_rng(CFG.seed)
structure = {}
for first in ("HE", "ME", "SHE"):
    group = train_frame[train_frame.gloss.map(lambda g: len(words(g)) > 1 and words(g)[0] == first)]
    picks = rng.choice(len(group), size=(200, 2))
    structure[f"shared_prefix_tokens_first_word_{first}_median"] = float(np.median([common_prefix(group.tokens.iloc[i], group.tokens.iloc[j]) for i, j in picks if i != j]))
picks = rng.choice(len(train_frame), size=(300, 2))
structure["shared_prefix_tokens_random_pairs_median"] = float(np.median([common_prefix(train_frame.tokens.iloc[i], train_frame.tokens.iloc[j]) for i, j in picks if i != j]))
copy_prev = np.zeros(6)
count = 0
for t in val_frame.tokens:
    copy_prev += (t[:, 1:] == t[:, :-1]).sum(1)
    count += t.shape[1] - 1
structure["copy_previous_token_accuracy_validation"] = (copy_prev / count).tolist()
print(json.dumps(structure, indent=1))

# %% [markdown]
# ## 9.3 — Unit-selection synthesizer
#
# * **Duration lexicon:** `T = Σ duration(unit)` over units `<s>, gloss words…, </s>{//|?}`, fitted by damped least
#   squares on training token lengths. Each training sequence is segmented at the cumulative durations, rescaled to its
#   true length.
# * **Unit selection:** left-to-right longest match of unit n-grams (≤ `max_ngram`) against the training index; multi-word
#   matches keep natural transitions. Among occurrences the **medoid** span (maximum token agreement with the other
#   occurrences) is chosen — the most typical recorded variant.
# * **OOV fallback:** an unseen gloss word is replaced by the character-n-gram-nearest training gloss.
# * **Junction smoothing:** RVQ latent crossfade over `radius` tokens around each junction.

# %%
class ConcatSynth:
    def __init__(self, glosses, tokens, max_ngram=8, medoid_pool=64, seed=0, damp=0.05):
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
        self.length_fit_residual_mae = float(np.abs(X @ self.duration - y).mean())
        self.word_vocab = [u for u in vocab if not u.startswith("<")]
        self.bounds = [self._bounds(us, t.shape[1]) for us, t in zip(self.units, self.tokens)]
        self.index = defaultdict(list)
        for s, us in enumerate(self.units):
            for i in range(len(us)):
                for j in range(i + 1, min(len(us), i + max_ngram) + 1):
                    self.index[tuple(us[i:j])].append((s, i, j))
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(self.word_vocab)
        self.char_matrix = self.char_vectorizer.transform(self.word_vocab)
        self._medoids, self._nearest = {}, {}

    def _bounds(self, us, T):
        b = np.concatenate([[0.0], np.cumsum([self.duration[self.vid[u]] for u in us])])
        b = np.rint(b * (T / b[-1])).astype(int)
        b[-1] = T
        return b

    def span(self, s, i, j):
        b = self.bounds[s]
        return self.tokens[s][:, b[i]:b[j]]

    @staticmethod
    def resample(tokens, n):
        return tokens[:, np.linspace(0, tokens.shape[1] - 1, n).round().astype(int)]

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
            stacked = np.stack([self.resample(s, n) for s in spans])
            agreement = [(stacked == stacked[k]).mean(axis=(1, 2)).sum() for k in range(len(stacked))]
            best = spans[int(np.argmax(agreement))]
        self._medoids[key] = best
        return best

    def set_semantic(self, encoder, threshold):
        """Enable closest-meaning fallback for unseen glosses (e.g. AUTOMOBILE -> CAR instead of AUTOMATIC)."""
        self.encoder, self.semantic_threshold, self._nearest = encoder, threshold, {}
        if encoder is not None and not hasattr(self, "vocab_embeddings"):
            self.vocab_embeddings = encoder.encode([w.lower() for w in self.word_vocab], normalize_embeddings=True, batch_size=256)
        return self

    def nearest_word(self, word):
        if word not in self._nearest:
            choice = None
            if getattr(self, "encoder", None) is not None:
                similarity = self.vocab_embeddings @ self.encoder.encode([word.lower()], normalize_embeddings=True)[0]
                if similarity.max() >= self.semantic_threshold:
                    choice = self.word_vocab[int(np.argmax(similarity))]
            if choice is None:
                similarity = (self.char_vectorizer.transform([word]) @ self.char_matrix.T).toarray()[0]
                choice = self.word_vocab[int(np.argmax(similarity))]
            self._nearest[word] = choice
        return self._nearest[word]

    def known_units(self, gloss):
        return [u if u in self.vid else self.nearest_word(u) for u in ext_units(gloss)]

    def synthesize(self, gloss):
        """Returns (tokens int64 [6,T], junction indices)."""
        us = self.known_units(gloss)
        pieces, i = [], 0
        while i < len(us):
            j = min(len(us), i + self.max_ngram)
            while j > i + 1 and tuple(us[i:j]) not in self.index:
                j -= 1
            piece = self.pick(tuple(us[i:j]))
            if piece.shape[1] > 0:
                pieces.append(piece)
            i = j
        tokens = np.concatenate(pieces, axis=1).astype(np.int64)
        return tokens, np.cumsum([p.shape[1] for p in pieces])[:-1].tolist()

    def predict_length(self, gloss):
        return int(round(sum(self.duration[self.vid[u]] for u in self.known_units(gloss))))

# %% [markdown]
# ## 9.4 — Metrics (exact Phase-8 definitions)

# %%
try:
    from rapidfuzz.distance import Levenshtein

    def edit_distance(a, b):
        return int(Levenshtein.distance(a, b))
except ImportError:
    def edit_distance(a, b):
        previous = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            current = [i]
            for j, y in enumerate(b, 1):
                current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + int(x != y)))
            previous = current
        return previous[-1]


def token_metrics(frame, mapping, label):
    rows = []
    level_correct, level_total = np.zeros(6), np.zeros(6)
    micro_correct = micro_total = 0
    for sid, gt in zip(frame.id, frame.tokens):
        pred = mapping[sid]
        n = min(gt.shape[1], pred.shape[1])
        per, ed = [], []
        for q in range(6):
            c = int((gt[q, :n] == pred[q, :n]).sum())
            level_correct[q] += c
            level_total[q] += n
            per.append(c / max(n, 1))
            ed.append(edit_distance(gt[q].tolist(), pred[q].tolist()) / max(gt.shape[1], pred.shape[1], 1))
        c = int((gt[:, :n] == pred[:, :n]).sum())
        micro_correct += c
        micro_total += 6 * n
        rows.append({"sample_id": sid, "model": label, "aligned_token_accuracy": c / max(6 * n, 1), "normalized_token_edit_distance": float(np.mean(ed)),
                     **{f"level_{q}_accuracy": per[q] for q in range(6)}, "token_length_error": pred.shape[1] - gt.shape[1]})
    detail = pd.DataFrame(rows)
    return {"model": label, "aligned_token_accuracy": float(detail.aligned_token_accuracy.mean()), "micro_token_accuracy": micro_correct / max(micro_total, 1),
            "normalized_token_edit_distance": float(detail.normalized_token_edit_distance.mean()), "per_level_accuracy": (level_correct / np.maximum(level_total, 1)).tolist(),
            "length_mae_tokens": float(detail.token_length_error.abs().mean()), "signed_length_bias_tokens": float(detail.token_length_error.mean())}, detail


def z(m):
    return ((m[:, active] - motion_mean[active]) / motion_std[active]).astype(np.float32)


def resample_time(m, n):
    if len(m) == n:
        return m
    return F.interpolate(torch.from_numpy(np.ascontiguousarray(m.T))[None], size=n, mode="linear", align_corners=True).squeeze(0).T.numpy()


def velocity(m):
    value = z(m)
    return np.sqrt(np.mean(np.square(np.diff(value, axis=0), dtype=np.float64), axis=1)) if len(value) > 1 else np.empty(0)


def repetition(values, max_lag=180):
    x = np.asarray(values, dtype=np.float64)
    x = x - x.mean() if len(x) else x
    correlations = []
    for lag in range(2, min(max_lag, len(x) // 2) + 1):
        a, b = x[:-lag], x[lag:]
        d = np.sqrt(np.dot(a, a) * np.dot(b, b))
        if d > 0:
            correlations.append(float(np.dot(a, b) / d))
    return max(correlations) if correlations else float("nan")


def motion_row(sid, pred, gt):
    aligned = resample_time(pred, len(gt))
    zp, zg = z(aligned), z(gt)
    vel = velocity(pred)
    mean_vel = float(vel.mean()) if len(vel) else 0.0
    variance = float(np.mean(np.var(z(pred), axis=0, dtype=np.float64)))
    collapse = mean_vel < float(thresholds["static_velocity_threshold_gt_percentile_1"]) or variance < float(thresholds["static_variance_threshold_gt_percentile_1"])
    maxz = float(np.max(np.abs(z(pred))))
    extreme = (float(pred.min()) < float(thresholds["extreme_minimum_threshold_gt_sample_min_percentile_0_1"]) or float(pred.max()) > float(thresholds["extreme_maximum_threshold_gt_sample_max_percentile_99_9"])
               or maxz > float(thresholds["extreme_normalized_magnitude_threshold_gt_percentile_99_9"]))
    return {"sample_id": sid, "normalized_feature_mae": float(np.mean(np.abs(zp - zg))), "time_normalized_raw_mae": float(np.mean(np.abs(aligned - gt))),
            "velocity_mae": float(np.mean(np.abs(np.diff(zp, axis=0) - np.diff(zg, axis=0)))), "motion_variance": variance, "collapse": collapse,
            "repetition_score": repetition(vel), "extreme": extreme, "duration_error": len(pred) - len(gt)}


def motion_metrics(frame, mapping, joins, radius, label, save_dir=None):
    rows = []
    for number, (sid, gt_tokens) in enumerate(zip(frame.id, frame.tokens), 1):
        pred = decode_tokens(mapping[sid], joins.get(sid, ()), radius)
        rows.append(motion_row(sid, pred, reference_motion(sid, gt_tokens)))
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            np.save(Path(save_dir) / f"{sid}.npy", pred, allow_pickle=False)
        if number % 250 == 0:
            print(label, "motion", number, "/", len(frame), flush=True)
    detail = pd.DataFrame(rows)
    return {"model": label, "motion_metric_reference": REFERENCE, "normalized_feature_mae": float(detail.normalized_feature_mae.mean()),
            "time_normalized_raw_mae": float(detail.time_normalized_raw_mae.mean()), "velocity_mae": float(detail.velocity_mae.mean()),
            "motion_variance": float(detail.motion_variance.mean()), "collapse_rate": float(detail.collapse.mean()), "repetition_score": float(detail.repetition_score.mean()),
            "extreme_motion_rate": float(detail.extreme.mean()), "duration_mae_frames": float(detail.duration_error.abs().mean()),
            "signed_duration_bias": float(detail.duration_error.mean())}, detail


def resample_tokens(tokens, T):
    return tokens[:, np.linspace(0, tokens.shape[1] - 1, T).round().astype(int)]


def run(synth, frame, gloss_col="gloss", oracle_length=True):
    """Synthesize for every row. Oracle-length mode nearest-index resamples tokens to the GT length."""
    tokens, joins = {}, {}
    for sid, gloss, T in zip(frame.id, frame[gloss_col], frame["T"]):
        tok, jn = synth.synthesize(gloss)
        if oracle_length and tok.shape[1] != T:
            scale = T / tok.shape[1]
            tok, jn = resample_tokens(tok, T), [int(round(j * scale)) for j in jn]
        tokens[sid], joins[sid] = tok, jn
    return tokens, joins

# %% [markdown]
# ## 9.5 — Validation-only hyperparameter selection
#
# Stage A (tokens, full validation): `max_ngram × medoid_pool`, ranked by edit distance then micro accuracy.
# Stage B (motion, length-stratified validation subset): junction radius, ranked by collapse then feature MAE.

# %%
search_rows = []
for max_ngram in CFG.max_ngram_candidates:
    for pool in CFG.medoid_pool_candidates:
        synth = ConcatSynth(train_frame.gloss.tolist(), train_frame.tokens.tolist(), max_ngram, pool, CFG.seed, CFG.lsqr_damp)
        tokens, _ = run(synth, val_frame)
        summary, _ = token_metrics(val_frame, tokens, f"ngram{max_ngram}_pool{pool}")
        search_rows.append({"max_ngram": max_ngram, "medoid_pool": pool, "micro_token_accuracy": summary["micro_token_accuracy"],
                            "normalized_token_edit_distance": summary["normalized_token_edit_distance"], "level_0_accuracy": summary["per_level_accuracy"][0]})
        print(search_rows[-1], flush=True)
token_search = pd.DataFrame(search_rows).sort_values(["normalized_token_edit_distance", "micro_token_accuracy"], ascending=[True, False])
token_search.to_csv(RESULTS / "validation_token_search.csv", index=False)
best_ngram, best_pool = int(token_search.iloc[0].max_ngram), int(token_search.iloc[0].medoid_pool)
synth = ConcatSynth(train_frame.gloss.tolist(), train_frame.tokens.tolist(), best_ngram, best_pool, CFG.seed, CFG.lsqr_damp)
print("Selected max_ngram", best_ngram, "medoid_pool", best_pool, "| length-fit residual MAE (train):", round(synth.length_fit_residual_mae, 3))

subset = val_frame.sort_values("T", kind="stable").iloc[np.linspace(0, len(val_frame) - 1, CFG.motion_selection_samples).round().astype(int)].reset_index(drop=True)
subset_tokens, subset_joins = run(synth, subset)
radius_rows = []
for radius in CFG.join_radius_candidates:
    summary, _ = motion_metrics(subset, subset_tokens, subset_joins, radius, f"radius{radius}")
    radius_rows.append({"join_radius": radius, **{k: v for k, v in summary.items() if k != "model"}})
    print(radius_rows[-1], flush=True)
radius_search = pd.DataFrame(radius_rows).sort_values(["collapse_rate", "normalized_feature_mae"])
radius_search.to_csv(RESULTS / "validation_join_radius_search.csv", index=False)
best_radius = int(radius_search.iloc[0].join_radius)

# Stage C (tokens, validation sentences containing unseen glosses): spelling vs closest-meaning fallback threshold.
sentence_encoder = None
if SBERT_DIR is not None:
    from sentence_transformers import SentenceTransformer
    sentence_encoder = SentenceTransformer(str(SBERT_DIR), device=str(DEVICE))
oov_val = val_frame[[any(u not in synth.vid for u in ext_units(g)) for g in val_frame.gloss]].reset_index(drop=True)
fallback_rows = []
for threshold in ([None] + list(CFG.semantic_threshold_candidates if sentence_encoder is not None else [])):
    synth.set_semantic(sentence_encoder if threshold is not None else None, threshold)
    tokens, _ = run(synth, oov_val)
    summary, _ = token_metrics(oov_val, tokens, f"fallback_{threshold}")
    fallback_rows.append({"semantic_threshold": threshold, "fallback": "spelling" if threshold is None else "meaning>=threshold, else spelling",
                          "micro_token_accuracy": summary["micro_token_accuracy"], "normalized_token_edit_distance": summary["normalized_token_edit_distance"], "samples": len(oov_val)})
    print(fallback_rows[-1], flush=True)
fallback_search = pd.DataFrame(fallback_rows).sort_values(["normalized_token_edit_distance", "micro_token_accuracy"], ascending=[True, False])
fallback_search.to_csv(RESULTS / "validation_oov_fallback_search.csv", index=False)
best_threshold = fallback_search.iloc[0].semantic_threshold
best_threshold = None if pd.isna(best_threshold) else float(best_threshold)
synth.set_semantic(sentence_encoder if best_threshold is not None else None, best_threshold)
if sentence_encoder is None:
    print("MiniLM not attached: unseen glosses use the closest-spelling fallback")
selection = {"split": "validation only", "max_ngram": best_ngram, "medoid_pool": best_pool, "join_radius": best_radius,
             "oov_fallback": "spelling" if best_threshold is None else "meaning", "semantic_threshold": best_threshold, "sentence_encoder": SBERT_DIR,
             "token_ranking": ["lower edit distance", "higher micro accuracy"], "motion_ranking": ["lower collapse", "lower feature MAE"], "test_used": False}
write_json(DIAG / "selection.json", selection)
print("FROZEN:", selection)

val_tokens, val_joins = run(synth, val_frame)
val_token_summary, _ = token_metrics(val_frame, val_tokens, "phase9_validation_oracle")
val_motion_summary, _ = motion_metrics(val_frame, val_tokens, val_joins, best_radius, "phase9_validation_oracle")
display(pd.DataFrame([{**val_token_summary, **val_motion_summary}]).T)

# %% [markdown]
# ## 9.6 — Frozen test evaluation (run once)
#
# * **Oracle length** (GT gloss, tokens resampled to the GT length) — comparable with Phase-8 `oracle_length_test`.
# * **GT gloss, own length** — the competition setting (the competition `test.csv` provides the gloss).
# * **Deployable** (Phase-3 predicted gloss, own lexicon length) — comparable with Phase-8 `end_to_end_test`.

# %%
results = {}
details = {}
for label, gloss_col, oracle in (("phase9_oracle_length", "gloss", True), ("phase9_gt_gloss_own_length", "gloss", False), ("phase9_end2end", "predicted_gloss", False)):
    tokens, joins = run(synth, test_frame, gloss_col, oracle)
    token_summary, token_detail = token_metrics(test_frame, tokens, label)
    save_dir = OUT / "decoded" / label if CFG.save_decoded_test else None
    motion_summary, motion_detail = motion_metrics(test_frame, tokens, joins, best_radius, label, save_dir)
    results[label] = {**token_summary, **motion_summary}
    details[label] = token_detail.set_index("sample_id").join(motion_detail.set_index("sample_id"), rsuffix="_motion").reset_index()
    details[label].to_csv(RESULTS / f"{label}_test_metrics.csv", index=False)
    ids = list(tokens)
    np.savez_compressed(TOKENS / f"{label}_test.npz", sample_ids=np.asarray(ids), offsets=np.concatenate([[0], np.cumsum([tokens[i].shape[1] for i in ids])]),
                        tokens_flat=np.concatenate([tokens[i] for i in ids], axis=1))
    print(label, {k: results[label][k] for k in ("micro_token_accuracy", "normalized_token_edit_distance", "normalized_feature_mae", "collapse_rate", "length_mae_tokens")}, flush=True)

# %% [markdown]
# ## 9.7 — Comparison with Phase 8

# %%
comparison = []
if P8:
    p8 = read_json(P8 / "phase8_report.json")
    for name, item in (("phase8_oracle_length", p8["oracle_length_test"]), ("phase8_end2end", p8["end_to_end_test"])):
        comparison.append({"method": name, "motion_metric_reference": "real_motion_features", **{k: v for k, v in item.items() if not isinstance(v, list) and k != "model"},
                           **{f"level_{q}_accuracy": item["per_level_accuracy"][q] for q in range(6)}})
        # Same-reference re-score of the saved Phase-8 decoded motions, so motion metrics are apples-to-apples.
        decoded_dir = P8 / "decoded" / {"phase8_oracle_length": "final_oracle", "phase8_end2end": "final_end2end"}[name]
        if REFERENCE != "real_motion_features" and decoded_dir.is_dir() and all((decoded_dir / f"{sid}.npy").is_file() for sid in test_frame.id):
            rows = [motion_row(sid, np.load(decoded_dir / f"{sid}.npy").astype(np.float32), reference_motion(sid, gt)) for sid, gt in zip(test_frame.id, test_frame.tokens)]
            rescored = pd.DataFrame(rows)
            comparison.append({"method": name + "_rescored", "motion_metric_reference": REFERENCE, **{k: v for k, v in item.items() if not isinstance(v, list) and k != "model"},
                               "normalized_feature_mae": rescored.normalized_feature_mae.mean(), "time_normalized_raw_mae": rescored.time_normalized_raw_mae.mean(),
                               "velocity_mae": rescored.velocity_mae.mean(), "motion_variance": rescored.motion_variance.mean(), "collapse_rate": rescored.collapse.mean(),
                               "repetition_score": rescored.repetition_score.mean(), "extreme_motion_rate": rescored.extreme.mean()})
for name, item in results.items():
    comparison.append({"method": name, **{k: v for k, v in item.items() if not isinstance(v, list) and k != "model"},
                       **{f"level_{q}_accuracy": item["per_level_accuracy"][q] for q in range(6)}})
comparison = pd.DataFrame(comparison)
columns = ["method", "motion_metric_reference", "micro_token_accuracy", "aligned_token_accuracy", "normalized_token_edit_distance", "normalized_feature_mae",
           "velocity_mae", "motion_variance", "collapse_rate", "repetition_score", "extreme_motion_rate", "duration_mae_frames", "length_mae_tokens"]
comparison = comparison[[c for c in columns if c in comparison]]
comparison.to_csv(RESULTS / "phase8_vs_phase9_test.csv", index=False)
display(comparison)
if REFERENCE != "real_motion_features":
    print("NOTE: Phase-9 motion metrics use the RVQ-reconstruction proxy reference; plain Phase-8 rows use real Motion-Features.\n"
          "Compare motion metrics against the *_rescored Phase-8 rows (same proxy reference). Token metrics are directly comparable.")

# Per-level and temporal-progress diagnostics
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
levels = np.arange(6)
if P8:
    axes[0].bar(levels - 0.2, p8["oracle_length_test"]["per_level_accuracy"], 0.4, label="Phase 8 (oracle length)")
axes[0].bar(levels + 0.2, results["phase9_oracle_length"]["per_level_accuracy"], 0.4, label="Phase 9 (oracle length)")
axes[0].set(xticks=levels, xticklabels=["base", "r1", "r2", "r3", "r4", "r5"], ylabel="Token accuracy", title="Per-level accuracy (test)")
axes[0].legend()
oracle_tokens, _ = run(synth, test_frame, "gloss", True)
correct, total = np.zeros(10), np.zeros(10)
for sid, gt in zip(test_frame.id, test_frame.tokens):
    pred = oracle_tokens[sid]
    for t in range(gt.shape[1]):
        b = min(9, int(10 * t / gt.shape[1]))
        correct[b] += int((gt[:, t] == pred[:, t]).sum())
        total[b] += 6
axes[1].plot(range(10), correct / np.maximum(total, 1), marker="o", label="Phase 9")
axes[1].set(xticks=range(10), xticklabels=[f"{i * 10}%" for i in range(10)], xlabel="Normalized progress", ylabel="Token accuracy", title="Accuracy vs. sequence progress (test)")
axes[1].legend()
fig.tight_layout()
fig.savefig(FIGURES / "phase9_accuracy_overview.png", dpi=160)
plt.close(fig)

# Qualitative feature trajectories: GT vs Phase-9 (vs Phase-8 when its decoded outputs are attached)
p8_decoded = P8 / "decoded/final_oracle" if P8 and (P8 / "decoded/final_oracle").is_dir() else None
ranked = details["phase9_oracle_length"].sort_values("normalized_feature_mae").reset_index(drop=True)
picked = list(dict.fromkeys(ranked.sample_id.iloc[[0, 1, len(ranked) // 2, len(ranked) // 2 + 1, -2, -1]].astype(str)))
feature_ids = thresholds.get("high_variance_feature_indices", [0, 1, 2, 3])[:4]
oracle_joins = run(synth, test_frame, "gloss", True)[1]
for sid in picked:
    gt_tokens = test_frame.set_index("id").tokens[sid]
    gt = reference_motion(sid, gt_tokens)
    p9 = resample_time(decode_tokens(oracle_tokens[sid], oracle_joins[sid], best_radius), len(gt))
    panels = [("Ground truth" + (" (RVQ proxy)" if REFERENCE != "real_motion_features" else ""), gt), ("Phase 9 concatenative", p9)]
    if p8_decoded and (p8_decoded / f"{sid}.npy").is_file():
        panels.append(("Phase 8 generator", resample_time(np.load(p8_decoded / f"{sid}.npy"), len(gt))))
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 3 * len(panels)), sharex=True)
    for ax, (title, motion) in zip(axes, panels):
        for f in feature_ids:
            ax.plot(motion[:, f], label=f"f{f}")
        ax.set_title(f"{title} — {sid}")
    axes[0].legend(ncol=4)
    fig.tight_layout()
    fig.savefig(FIGURES / f"case_{sid}.png", dpi=130)
    plt.close(fig)

# %% [markdown]
# ## 9.8 — Final inference API and competition submission
#
# `generate_motion(text)` = Phase-3 text→gloss → unit selection → junction-smoothed RVQ decode. When a gloss is already
# available (competition setting) call `generate_motion_from_gloss(gloss)`.
# The deployable synthesizer is rebuilt on **train + validation + test** token data (all labelled data), because no
# further evaluation happens after this point; the frozen hyperparameters are unchanged.

# %%
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(P3 / "tokenizer", local_files_only=True, use_fast=True)
phase3_model = AutoModelForSeq2SeqLM.from_pretrained(P3 / "best_model", local_files_only=True).to(DEVICE).eval()
phase3_beams = 4 if read_json(P3 / "phase3_report.json").get("selected_decoding") == "beam_4" else 1

if COMPETITION:
    competition_train = pd.read_csv(COMPETITION / "train.csv", dtype={"id": str}, low_memory=False)

    def usable_tokens(row):
        """[6,T] tokens, or None for rows with missing/invalid/misaligned levels (Phase 1 quarantined these)."""
        try:
            levels = [parse_cell(row[c]) for c in LEVELS]
        except (ValueError, TypeError):
            return None
        return np.stack(levels) if len({len(x) for x in levels}) == 1 else None

    competition_train["tokens"] = [usable_tokens(row) for _, row in competition_train.iterrows()]
    skipped = competition_train[competition_train.tokens.isna() | competition_train.gloss.isna()]
    competition_train = competition_train.drop(skipped.index).reset_index(drop=True)
    print(f"Competition train.csv: {len(competition_train)} usable rows, {len(skipped)} skipped (missing/invalid tokens or gloss): {skipped.id.tolist()[:10]}")
    deploy_glosses, deploy_tokens = competition_train.gloss.astype(str).tolist(), competition_train.tokens.tolist()
else:
    labelled = pd.concat([train_frame, val_frame, test_frame], ignore_index=True)
    deploy_glosses, deploy_tokens = labelled.gloss.astype(str).tolist(), labelled.tokens.tolist()
deploy_synth = ConcatSynth(deploy_glosses, deploy_tokens, best_ngram, best_pool, CFG.seed, CFG.lsqr_damp)
deploy_synth.set_semantic(sentence_encoder if best_threshold is not None else None, best_threshold)
print("Deployable synthesizer built on", len(deploy_glosses), "labelled sequences")


@torch.inference_mode()
def text_to_gloss(text):
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if not cleaned:
        raise ValueError("English text must not be empty")
    encoded = tokenizer("text to gloss: " + cleaned, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
    output = phase3_model.generate(**encoded, max_length=128, num_beams=phase3_beams, do_sample=False, early_stopping=phase3_beams > 1)
    return re.sub(r"\s+", " ", tokenizer.decode(output[0], skip_special_tokens=True)).strip()


def generate_motion_from_gloss(gloss, return_intermediates=False):
    tokens, joins = deploy_synth.synthesize(gloss)
    motion = decode_tokens(tokens, joins, best_radius)
    if motion.dtype != np.float32 or motion.shape != (4 * tokens.shape[1], 668) or not np.isfinite(motion).all():
        raise RuntimeError("Final API invariant failed")
    result = {"gloss": gloss, "token_length": int(tokens.shape[1]), "rvq_tokens": tokens, "junctions": joins, "motion": motion}
    return result if return_intermediates else motion


def generate_motion(english_text, return_intermediates=False):
    gloss = text_to_gloss(english_text)
    result = generate_motion_from_gloss(gloss, True)
    result = {"text": str(english_text), "predicted_gloss": gloss, **result}
    return result if return_intermediates else result["motion"]


api_rows = []
for _, row in test_frame.iloc[np.linspace(0, len(test_frame) - 1, 20).round().astype(int)].iterrows():
    try:
        r = generate_motion(str(row.sentence), True)
        api_rows.append({"sample_id": row.id, "predicted_gloss": r["predicted_gloss"], "token_length": r["token_length"],
                         "token_ids_valid": bool(r["rvq_tokens"].min() >= 0 and r["rvq_tokens"].max() <= 511), "motion_shape_valid": r["motion"].shape == (4 * r["token_length"], 668),
                         "finite": bool(np.isfinite(r["motion"]).all()), "error": ""})
    except Exception as error:
        api_rows.append({"sample_id": row.id, "predicted_gloss": "", "token_length": None, "token_ids_valid": False, "motion_shape_valid": False, "finite": False, "error": repr(error)})
api_validation = pd.DataFrame(api_rows)
api_validation.to_csv(FINAL / "api_validation.csv", index=False)
api_passed = bool(api_validation[["token_ids_valid", "motion_shape_valid", "finite"]].all().all())
example = generate_motion("I want water", True)
print("Example:", example["predicted_gloss"], "| T =", example["token_length"], "| motion", example["motion"].shape, "| API passed:", api_passed)

submission_written = False
if COMPETITION:
    competition_test = pd.read_csv(COMPETITION / "test.csv", dtype={"id": str})
    rows = []
    for _, row in competition_test.iterrows():
        gloss = str(row.gloss) if "gloss" in competition_test and isinstance(row.gloss, str) and row.gloss.strip() else text_to_gloss(row.sentence)
        tokens, _ = deploy_synth.synthesize(gloss)
        rows.append({"id": row.id, **{level: " ".join(map(str, tokens[q].tolist())) for q, level in enumerate(LEVELS)}})
    pd.DataFrame(rows).to_csv(OUT / "submission.csv", index=False)
    submission_written = True
    print("Wrote", OUT / "submission.csv", "with", len(rows), "rows")

with (FINAL / "concat_synth_state.pkl").open("wb") as handle:
    pickle.dump({"config": {"max_ngram": best_ngram, "medoid_pool": best_pool, "join_radius": best_radius, "seed": CFG.seed, "damp": CFG.lsqr_damp},
                 "glosses": deploy_glosses, "tokens": deploy_tokens}, handle)
write_json(FINAL / "system_contract.json", {"input": "English UTF-8 text (or gloss string)", "phase3_output": "gloss string",
                                            "synthesizer_output": "int64 [6,T], IDs 0..511, T from the duration lexicon",
                                            "decoder_output": "float32 [4T,668], finite, junction-smoothed (radius %d)" % best_radius,
                                            "rebuild": "ConcatSynth(glosses, tokens, **config) from concat_synth_state.pkl"})

# %% [markdown]
# ## 9.9 — Report

# %%
report = {"phase": "Phase 9 - Gloss-aligned concatenative motion synthesis", "phase1_fingerprint": FINGERPRINT, "motion_metric_reference": REFERENCE,
          "structure_diagnostics": {**structure, "duration_lexicon_train_residual_mae_tokens": synth.length_fit_residual_mae},
          "selection": selection, "validation_oracle_length": {**val_token_summary, **val_motion_summary}, "test": results,
          "phase8_comparison_available": bool(P8), "api_validation_passed": api_passed, "submission_written": submission_written,
          "neural_generator_used": False, "semantic_motion_quality_claimed": False, "skeleton_conversion_verified": False, "bvh_export_verified": False,
          "limitations": ["Retrieval/concatenation cannot produce signs absent from the training data; OOV glosses use the nearest-spelled known gloss.",
                          "Exploits the corpus's concatenated-clip construction; gains may not transfer to continuously signed corpora.",
                          "Deployable accuracy is bounded by Phase-3 text-to-gloss quality.",
                          "Test split was inspected in earlier phases; Phase 9 choices used validation only."]}
write_json(OUT / "phase9_report.json", report)
print("=" * 60)
print("PHASE 9 COMPLETE | reference:", REFERENCE)
for name, item in results.items():
    print(f"{name:28s} micro {item['micro_token_accuracy']:.4f} | edit {item['normalized_token_edit_distance']:.4f} | feature MAE {item['normalized_feature_mae']:.4f} | "
          f"collapse {item['collapse_rate']:.4f} | length MAE {item['length_mae_tokens']:.2f} tok")
print("=" * 60)

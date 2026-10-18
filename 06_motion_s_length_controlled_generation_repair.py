# %% [markdown]
# # Motion-S / SignVRSE — Phase 6: Length-Controlled Motion Generation Repair
#
# This notebook is a causal repair experiment. It freezes the completed Phase-3, Phase-4C,
# and Phase-5A models; trains only a gloss-to-length head; and compares original free-running,
# oracle-length, and predicted-length decoding. It remains strictly in verified token/feature space.

# %%
from __future__ import annotations

import ast
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from torch.utils.data import DataLoader, TensorDataset

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EXPECTED_FINGERPRINT = "e2acdf4d6dce91775239ad50be3866dbc6f5da94d940204b62456cdb77fc4b0d"
EXPECTED_SPLIT_SIZES = {"train": 9970, "validation": 1246, "test": 1247}
EXPECTED_SUPPORTED = {"train": 9936, "validation": 1243, "test": 1243}
LEVEL_ORDER = ["base_tokens", "residual_1", "residual_2", "residual_3", "residual_4", "residual_5"]
RVQ_LEVELS, CODEBOOK_SIZE = 6, 512
PAD_ID, BOS_ID, EOS_ID, MOTION_VOCAB_SIZE = 512, 513, 514, 515
MOTION_DIM, LATENT_DIM, HIDDEN_DIM, TEMPORAL_EXPANSION = 668, 256, 512, 4


@dataclass(frozen=True)
class Phase6Config:
    seed: int = 42
    kaggle_input: str = "/kaggle/input"
    kaggle_working: str = "/kaggle/working"
    output_dir: str = "/kaggle/working/motion_s_phase6"
    supported_max_t: int = 512
    encoder_batch_size: int = 64
    length_batch_size: int = 256
    length_learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_epochs: int = 50
    early_stopping_patience: int = 6
    dropout: float = 0.1
    baseline_reproduction_samples: int = 8
    generate_validation_tokens: bool = True
    near_zero_std: float = 1e-6


CFG = Phase6Config()
OUTPUT = Path(CFG.output_dir)
DATA, LENGTH_MODEL, TOKENS, DECODED, DIAGNOSTICS, RESULTS, FIGURES = (
    OUTPUT / "data", OUTPUT / "length_model", OUTPUT / "tokens", OUTPUT / "decoded",
    OUTPUT / "diagnostics", OUTPUT / "results", OUTPUT / "figures",
)
ORACLE_DECODED = DECODED / "oracle_length"
PRED_ORACLE_DECODED = DECODED / "predicted_length_oracle_gloss"
PRED_END_DECODED = DECODED / "predicted_length_end2end"
for directory in (OUTPUT, DATA, LENGTH_MODEL, TOKENS, DECODED, ORACLE_DECODED,
                  PRED_ORACLE_DECODED, PRED_END_DECODED, DIAGNOSTICS, RESULTS, FIGURES):
    directory.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


set_seed(CFG.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda": raise RuntimeError("Phase 6 requires a Kaggle GPU accelerator")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, ensure_ascii=False, allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def canonical(value: str) -> str: return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle: return json.load(handle)


print("Python:", sys.version.split()[0], "PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
display(pd.Series(asdict(CFG), name="value").to_frame())

# %% [markdown]
# ## 6.0 — Artifact discovery and immutable-contract checks

# %%
SEARCH_BASES = [Path(CFG.kaggle_input), Path(CFG.kaggle_working)]


def roots_below(predicate) -> list[Path]:
    found: list[Path] = []
    for base in SEARCH_BASES:
        if not base.exists(): continue
        for current, directories, _ in os.walk(base):
            candidate = Path(current)
            if candidate.resolve() == OUTPUT.resolve(): directories[:] = []; continue
            if predicate(candidate): found.append(candidate.resolve()); directories[:] = []
    return sorted(dict.fromkeys(found), key=str)


def choose_root(label: str, predicate, identity_relpaths: list[str]) -> Path:
    roots = roots_below(predicate)
    if not roots: raise RuntimeError(f"Missing {label}. Searched {[str(x) for x in SEARCH_BASES]}")
    signatures = {}
    for root in roots:
        signature = []
        for relative in identity_relpaths:
            path = root / relative
            if path.is_file(): signature.append((relative, sha256_file(path)))
        signatures[root] = tuple(signature)
    if len(set(signatures.values())) != 1:
        raise RuntimeError(f"Ambiguous non-identical {label} roots: {[(str(k), v) for k, v in signatures.items()]}")
    working = [root for root in roots if str(root).startswith(CFG.kaggle_working)]
    selected = sorted(working or roots, key=lambda p: (len(p.parts), str(p)))[0]
    if len(roots) > 1: print(f"{label}: identical duplicate roots found; selected {selected}")
    else: print(f"{label}: {selected}")
    return selected


def phase1_predicate(path: Path) -> bool:
    return ((path / "dataset_config.json").is_file() and (path / "splits/train_split.csv").is_file()
            and any((path / "splits" / name).is_file() for name in ("val_split.csv", "validation_split.csv"))
            and (path / "splits/test_split.csv").is_file())


def phase3_predicate(path: Path) -> bool:
    required = all((path / rel).is_file() for rel in ("phase3_report.json", "best_model/config.json",
                                                       "tokenizer/tokenizer_config.json",
                                                       "results/validation_predictions.csv",
                                                       "results/test_predictions.csv"))
    weights = any((path / "best_model" / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))
    return required and weights


def phase4_predicate(path: Path) -> bool:
    return all((path / rel).is_file() for rel in ("phase4c_report.json", "best_model/model.pt",
                                                   "best_model/model_config.json",
                                                   "results/test_oracle_tokens.npz",
                                                   "results/test_end2end_tokens.npz"))


def phase5b_predicate(path: Path) -> bool:
    return all((path / rel).is_file() for rel in ("phase5b_report.json", "diagnostics/decoder_contract.json",
                                                   "diagnostics/motion_mean_668.npy",
                                                   "diagnostics/motion_std_668.npy",
                                                   "diagnostics/empirical_motion_thresholds.json",
                                                   "results/oracle_motion_metrics.csv",
                                                   "results/end2end_motion_metrics.csv",
                                                   "decoded/oracle_manifest.csv", "decoded/end2end_manifest.csv"))


def phase5c_predicate(path: Path) -> bool:
    return all((path / rel).is_file() for rel in ("phase5c_report.json",
                                                   "diagnostics/phase5c_empirical_thresholds.json",
                                                   "results/overgeneration_tail_analysis.csv"))


PHASE1_ROOT = choose_root("Phase-1 root", phase1_predicate, ["dataset_config.json", "file_manifest.csv"])
PHASE3_ROOT = choose_root("Phase-3 root", phase3_predicate,
                          ["phase3_report.json", "best_model/config.json", "best_model/model.safetensors",
                           "best_model/pytorch_model.bin", "tokenizer/tokenizer_config.json"])
PHASE4_ROOT = choose_root("Phase-4C root", phase4_predicate,
                          ["phase4c_report.json", "best_model/model.pt", "best_model/model_config.json"])
PHASE5B_ROOT = choose_root("Phase-5B root", phase5b_predicate,
                           ["phase5b_report.json", "diagnostics/decoder_contract.json",
                            "diagnostics/motion_mean_668.npy", "diagnostics/motion_std_668.npy"])
PHASE5C_ROOT = choose_root("Phase-5C root", phase5c_predicate,
                           ["phase5c_report.json", "diagnostics/phase5c_empirical_thresholds.json"])


def resolve_checkpoint() -> Path:
    candidates = []
    for base in SEARCH_BASES:
        if base.exists(): candidates.extend(path.resolve() for path in base.rglob("rvq_vae_best.pth") if path.is_file())
    candidates = sorted(dict.fromkeys(candidates), key=str)
    if not candidates: raise RuntimeError("Official rvq_vae_best.pth is missing")
    hashes = {path: sha256_file(path) for path in candidates}
    contract = read_json(PHASE5B_ROOT / "diagnostics/decoder_contract.json")
    expected = contract.get("checkpoint_sha256")
    if expected:
        candidates = [path for path in candidates if hashes[path] == expected]
        if not candidates: raise RuntimeError("No RVQ checkpoint matches decoder_contract checkpoint_sha256")
    if len({hashes[path] for path in candidates}) != 1:
        raise RuntimeError(f"Ambiguous non-identical official checkpoints: {[(str(p), hashes[p]) for p in candidates]}")
    selected = candidates[0]; print("Official RVQ-VAE checkpoint:", selected); return selected


CHECKPOINT_PATH = resolve_checkpoint()
PHASE1_CONFIG = read_json(PHASE1_ROOT / "dataset_config.json")
PHASE3_REPORT = read_json(PHASE3_ROOT / "phase3_report.json")
PHASE4_REPORT = read_json(PHASE4_ROOT / "phase4c_report.json")
PHASE5B_REPORT = read_json(PHASE5B_ROOT / "phase5b_report.json")
PHASE5C_REPORT = read_json(PHASE5C_ROOT / "phase5c_report.json")
ID_COL = PHASE1_CONFIG["id_column"]
GLOSS_COL = PHASE1_CONFIG["gloss_column"]
SENTENCE_COL = PHASE1_CONFIG.get("sentence_column", "sentence")
TOKEN_COLS = list(PHASE1_CONFIG["motion_token_columns"])
if PHASE1_CONFIG.get("split_fingerprint_sha256") != EXPECTED_FINGERPRINT: raise RuntimeError("Phase-1 fingerprint mismatch")
if TOKEN_COLS != LEVEL_ORDER: raise RuntimeError(f"Unexpected RVQ level order: {TOKEN_COLS}")
if PHASE3_REPORT.get("phase3_complete") is not True: raise RuntimeError("Phase 3 is incomplete")
if PHASE4_REPORT.get("phase4c_complete") is not True: raise RuntimeError("Phase 4C is incomplete")
if PHASE5B_REPORT.get("phase5b_complete") is not True: raise RuntimeError("Phase 5B is incomplete")
if PHASE5C_REPORT.get("phase5c_complete") is not True: raise RuntimeError("Phase 5C is incomplete")


def locate_train_csv() -> Path:
    candidates = []
    for base in SEARCH_BASES:
        if not base.exists(): continue
        for path in base.rglob("train.csv"):
            try:
                columns = set(pd.read_csv(path, nrows=0).columns)
                if {ID_COL, GLOSS_COL, *TOKEN_COLS}.issubset(columns): candidates.append(path.resolve())
            except Exception: pass
    candidates = sorted(dict.fromkeys(candidates), key=str)
    if not candidates: raise RuntimeError("Official token train.csv is missing")
    hashes = {path: sha256_file(path) for path in candidates}
    if len(set(hashes.values())) != 1:
        raise RuntimeError(f"Ambiguous non-identical token train.csv files: {[(str(p), hashes[p]) for p in candidates]}")
    selected = candidates[0]
    if len(candidates) > 1: print(f"Official token train.csv: identical copies found; selected {selected}")
    else: print("Official token train.csv:", selected)
    return selected


def locate_motion_features() -> Path:
    candidates = []
    for base in SEARCH_BASES:
        if not base.exists(): continue
        for current, directories, _ in os.walk(base):
            for directory in directories:
                if canonical(directory) == "motionfeatures": candidates.append((Path(current) / directory).resolve())
    candidates = sorted(dict.fromkeys(candidates), key=str)
    if len(candidates) != 1: raise RuntimeError(f"Expected exactly one Motion-Features directory; found {[str(x) for x in candidates]}")
    print("Motion-Features:", candidates[0]); return candidates[0]


MOTION_S_TRAIN_CSV = locate_train_csv()
MOTION_FEATURES = locate_motion_features()
TOKENIZER_PATH, PHASE3_MODEL_PATH = PHASE3_ROOT / "tokenizer", PHASE3_ROOT / "best_model"
PHASE4_CHECKPOINT = PHASE4_ROOT / "best_model/model.pt"
DECODER_CONTRACT_PATH = PHASE5B_ROOT / "diagnostics/decoder_contract.json"
MOTION_MEAN_PATH = PHASE5B_ROOT / "diagnostics/motion_mean_668.npy"
MOTION_STD_PATH = PHASE5B_ROOT / "diagnostics/motion_std_668.npy"
for label, path in {
    "Phase-1 splits": PHASE1_ROOT / "splits", "Phase-3 model": PHASE3_MODEL_PATH,
    "Phase-3 tokenizer": TOKENIZER_PATH, "Phase-3 predictions": PHASE3_ROOT / "results",
    "Phase-4 checkpoint": PHASE4_CHECKPOINT, "Phase-4 token metadata": PHASE4_ROOT / "best_model/token_config.json",
    "Phase-5A decoder contract": DECODER_CONTRACT_PATH, "Phase-5B report": PHASE5B_ROOT / "phase5b_report.json",
    "Phase-5C report": PHASE5C_ROOT / "phase5c_report.json", "official train.csv": MOTION_S_TRAIN_CSV,
}.items(): print(f"{label}: {path}")

artifact_verification = {
    "phase1_fingerprint": EXPECTED_FINGERPRINT, "phase1": PHASE1_ROOT, "phase3": PHASE3_ROOT,
    "phase4": PHASE4_ROOT, "phase5b": PHASE5B_ROOT, "phase5c": PHASE5C_ROOT,
    "official_train_csv": MOTION_S_TRAIN_CSV, "motion_features": MOTION_FEATURES,
    "rvq_checkpoint": CHECKPOINT_PATH, "rvq_checkpoint_sha256": sha256_file(CHECKPOINT_PATH),
    "phase4_checkpoint_sha256": sha256_file(PHASE4_CHECKPOINT), "passed": True,
}
write_json(DIAGNOSTICS / "artifact_verification.json", artifact_verification)

# %% [markdown]
# ## 6.2 — Locked length targets and descriptive analysis

# %%
def split_path(name: str) -> Path:
    names = {"train": ["train_split.csv"], "validation": ["val_split.csv", "validation_split.csv"],
             "test": ["test_split.csv"]}[name]
    matches = [PHASE1_ROOT / "splits" / item for item in names if (PHASE1_ROOT / "splits" / item).is_file()]
    if len(matches) != 1: raise RuntimeError(f"Expected one {name} split; found {matches}")
    return matches[0]


official = pd.read_csv(MOTION_S_TRAIN_CSV, dtype={ID_COL: str}, low_memory=False)
official[ID_COL] = official[ID_COL].astype(str)
if official[ID_COL].duplicated().any(): raise RuntimeError("Duplicate IDs in official train.csv")
official_indexed = official.set_index(ID_COL, drop=False)


def parse_token_cell(value: Any) -> np.ndarray:
    """Parse the official whitespace format plus JSON/Python-list exported copies."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise RuntimeError("Missing token cell")
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        raise RuntimeError("Blank/null token cell")
    parsed: Any
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            # The official Motion-S train.csv stores levels as plain space-separated IDs.
            # Comma-separated unbracketed exports are accepted too.
            if any(mark in text for mark in ("[", "]", "{", "}", "|", ";")):
                raise RuntimeError(f"Corrupted token serialization: {text[:80]!r}")
            try:
                parsed = [int(piece) for piece in re.split(r"[\s,]+", text) if piece]
            except ValueError as error:
                raise RuntimeError(f"Non-integer token serialization: {text[:80]!r}") from error
    array = np.asarray(parsed)
    if array.dtype == object or array.ndim != 1 or array.size == 0:
        raise RuntimeError(f"Expected one non-empty token level [T], got {array.shape}")
    try:
        numeric = array.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise RuntimeError("Non-numeric token value") from error
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise RuntimeError("Token sequence contains NaN/Inf or non-integer values")
    return numeric.astype(np.int64)


def parse_tokens_from_row(row: pd.Series) -> np.ndarray:
    levels = [parse_token_cell(row[column]) for column in TOKEN_COLS]
    if any(level.ndim != 1 or len(level) == 0 for level in levels): raise RuntimeError(f"Malformed tokens for {row[ID_COL]}")
    if len({len(level) for level in levels}) != 1: raise RuntimeError(f"RVQ length mismatch for {row[ID_COL]}")
    tokens = np.stack(levels)
    if int(tokens.min()) < 0 or int(tokens.max()) > 511: raise RuntimeError(f"Invalid token ID for {row[ID_COL]}")
    return tokens


TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*|[/?.!,;:]", flags=re.UNICODE)
def word_count(value: Any) -> int: return len(TOKEN_PATTERN.findall(str(value)))


split_frames: dict[str, pd.DataFrame] = {}
target_parts = []
for split_name, expected in EXPECTED_SPLIT_SIZES.items():
    locked = pd.read_csv(split_path(split_name), dtype={ID_COL: str}, low_memory=False)
    locked[ID_COL] = locked[ID_COL].astype(str)
    if len(locked) != expected or not locked[ID_COL].is_unique: raise RuntimeError(f"Locked {split_name} mismatch")
    missing = sorted(set(locked[ID_COL]) - set(official_indexed.index))
    if missing: raise RuntimeError(f"Official train.csv missing {split_name} IDs: {missing[:10]}")
    merged = locked.drop(columns=[c for c in TOKEN_COLS if c in locked], errors="ignore").merge(
        official[[ID_COL, *TOKEN_COLS]], on=ID_COL, how="left", validate="one_to_one")
    lengths = []
    for _, row in merged.iterrows(): lengths.append(parse_tokens_from_row(row).shape[1])
    merged["motion_token_length"] = np.asarray(lengths, dtype=np.int64)
    if "token_length" in locked and not np.array_equal(pd.to_numeric(locked["token_length"]).astype(int), merged["motion_token_length"]):
        raise RuntimeError(f"Phase-1 length metadata mismatch in {split_name}")
    supported = merged.loc[merged["motion_token_length"] <= CFG.supported_max_t].copy()
    if len(supported) != EXPECTED_SUPPORTED[split_name]:
        raise RuntimeError(f"{split_name}: expected {EXPECTED_SUPPORTED[split_name]} supported IDs, found {len(supported)}")
    supported["split"] = split_name
    supported["gloss_token_count"] = supported[GLOSS_COL].map(word_count)
    supported["sentence_word_count"] = supported[SENTENCE_COL].map(word_count)
    supported["motion_frame_length"] = TEMPORAL_EXPANSION * supported["motion_token_length"]
    split_frames[split_name] = supported.reset_index(drop=True)
    target_parts.append(supported[[ID_COL, "split", GLOSS_COL, SENTENCE_COL, "gloss_token_count",
                                   "sentence_word_count", "motion_token_length", "motion_frame_length"]])

length_targets = pd.concat(target_parts, ignore_index=True).rename(columns={ID_COL: "sample_id", GLOSS_COL: "gloss", SENTENCE_COL: "sentence"})
length_targets.to_csv(DATA / "length_targets.csv", index=False)
if len(set(split_frames["train"][ID_COL]) & set(split_frames["validation"][ID_COL])): raise RuntimeError("Train/validation leakage")
if len(set(split_frames["train"][ID_COL]) & set(split_frames["test"][ID_COL])): raise RuntimeError("Train/test leakage")


def spearman(x, y) -> float:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < 2 or frame.x.nunique() < 2 or frame.y.nunique() < 2: return float("nan")
    return float(frame.x.rank(method="average").corr(frame.y.rank(method="average")))


def distribution(values: pd.Series) -> dict[str, float]:
    x = values.astype(float)
    return {"min": float(x.min()), "mean": float(x.mean()), "median": float(x.median()), "std": float(x.std(ddof=0)),
            **{f"p{q}": float(np.percentile(x, q)) for q in (10, 25, 75, 90, 95, 99)}, "max": float(x.max())}


length_distribution = {"splits": {}, "relationships": {}}
for name, frame in split_frames.items():
    length_distribution["splits"][name] = distribution(frame["motion_token_length"])
    length_distribution["relationships"][name] = {
        "gloss_words_pearson": float(frame["gloss_token_count"].corr(frame["motion_token_length"])),
        "gloss_words_spearman": spearman(frame["gloss_token_count"], frame["motion_token_length"]),
        "sentence_words_pearson": float(frame["sentence_word_count"].corr(frame["motion_token_length"])),
        "sentence_words_spearman": spearman(frame["sentence_word_count"], frame["motion_token_length"]),
    }
write_json(DIAGNOSTICS / "length_distribution.json", length_distribution)
display(pd.DataFrame(length_distribution["splits"]).T)

# %% [markdown]
# ## 6.3 — Median and gloss-count linear baselines

# %%
def length_metrics(y_true, y_pred) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.float64); prediction = np.asarray(y_pred, dtype=np.float64)
    error = prediction - truth; absolute = np.abs(error); relative = absolute / np.maximum(truth, 1)
    return {
        "mae_tokens": float(absolute.mean()), "rmse_tokens": float(np.sqrt(np.mean(error ** 2))),
        "median_absolute_error_tokens": float(np.median(absolute)), "mean_signed_error_tokens": float(error.mean()),
        "spearman": spearman(pd.Series(truth), pd.Series(prediction)), "relative_mae": float(relative.mean()),
        "within_5_tokens": float((absolute <= 5).mean()), "within_10_tokens": float((absolute <= 10).mean()),
        "within_20_tokens": float((absolute <= 20).mean()), "within_10_percent": float((relative <= .10).mean()),
        "within_20_percent": float((relative <= .20).mean()), "approx_frame_mae": float(4 * absolute.mean()),
    }


train_frame, validation_frame, test_frame = split_frames["train"], split_frames["validation"], split_frames["test"]
train_y = train_frame["motion_token_length"].to_numpy(float)
val_y = validation_frame["motion_token_length"].to_numpy(float)
training_median = float(np.median(train_y))
linear_slope, linear_intercept = np.polyfit(train_frame["gloss_token_count"].to_numpy(float), train_y, 1)
baseline_rows = []
for label, prediction in (
    ("training_median", np.full(len(validation_frame), training_median)),
    ("linear_gloss_token_count", linear_slope * validation_frame["gloss_token_count"].to_numpy(float) + linear_intercept),
):
    baseline_rows.append({"baseline": label, "split": "validation", **length_metrics(val_y, np.rint(prediction))})
pd.DataFrame(baseline_rows).to_csv(RESULTS / "length_baseline_metrics.csv", index=False)
display(pd.DataFrame(baseline_rows))

# %% [markdown]
# ## 6.4–6.7 — Frozen T5 length predictor, validation selection, and evaluation

# %%
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError as error:
    raise ImportError("Transformers must be available in the Kaggle image") from error

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True, use_fast=True)
seq2seq = AutoModelForSeq2SeqLM.from_pretrained(PHASE3_MODEL_PATH, local_files_only=True)
frozen_encoder = seq2seq.get_encoder().to(DEVICE).eval()
for parameter in frozen_encoder.parameters(): parameter.requires_grad = False
encoder_dimension = int(frozen_encoder.config.d_model)
del seq2seq; gc.collect(); torch.cuda.empty_cache()


@torch.inference_mode()
def encode_glosses(texts: list[str], label: str) -> np.ndarray:
    chunks = []
    for start in range(0, len(texts), CFG.encoder_batch_size):
        batch_text = texts[start:start + CFG.encoder_batch_size]
        encoded = tokenizer(batch_text, padding=True, truncation=False, return_tensors="pt")
        input_ids = encoded["input_ids"].to(DEVICE); mask = encoded["attention_mask"].to(DEVICE)
        hidden = frozen_encoder(input_ids=input_ids, attention_mask=mask).last_hidden_state.float()
        weights = mask.unsqueeze(-1).float()
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        chunks.append(pooled.cpu().numpy().astype(np.float32))
        if (start // CFG.encoder_batch_size + 1) % 25 == 0: print(f"{label}: {min(start + len(batch_text), len(texts))}/{len(texts)}")
    return np.concatenate(chunks, axis=0)


train_embeddings = encode_glosses(train_frame[GLOSS_COL].astype(str).tolist(), "train GT-gloss encoding")
validation_gt_embeddings = encode_glosses(validation_frame[GLOSS_COL].astype(str).tolist(), "validation GT-gloss encoding")


class LengthHead(nn.Module):
    def __init__(self, input_dimension: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dimension, 256), nn.GELU(), nn.Dropout(CFG.dropout), nn.Linear(256, 1))
    def forward(self, pooled: torch.Tensor) -> torch.Tensor: return self.network(pooled).squeeze(-1)


def embedding_loader(embeddings: np.ndarray, targets: np.ndarray, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(CFG.seed)
    dataset = TensorDataset(torch.from_numpy(embeddings), torch.from_numpy(np.log1p(targets).astype(np.float32)))
    return DataLoader(dataset, batch_size=CFG.length_batch_size, shuffle=shuffle, generator=generator,
                      num_workers=0, pin_memory=True)


train_loader = embedding_loader(train_embeddings, train_y, True)
val_loader = embedding_loader(validation_gt_embeddings, val_y, False)
length_head = LengthHead(encoder_dimension).to(DEVICE)
optimizer = torch.optim.AdamW(length_head.parameters(), lr=CFG.length_learning_rate, weight_decay=CFG.weight_decay)
criterion = nn.SmoothL1Loss()
train_lower = int(train_frame["motion_token_length"].min())
train_upper = min(int(train_frame["motion_token_length"].max()), CFG.supported_max_t)


@torch.inference_mode()
def predict_from_embeddings(model: LengthHead, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    model.eval(); raw_parts = []
    for start in range(0, len(embeddings), CFG.length_batch_size):
        logits = model(torch.from_numpy(embeddings[start:start + CFG.length_batch_size]).to(DEVICE)).float()
        raw_parts.append(torch.expm1(logits).cpu().numpy())
    raw = np.concatenate(raw_parts)
    rounded = np.rint(raw)
    clipped = np.clip(rounded, train_lower, train_upper).astype(np.int64)
    return raw, clipped, int((rounded != clipped).sum())


best_mae, best_epoch, wait = float("inf"), -1, 0
history = []
for epoch in range(1, CFG.max_epochs + 1):
    length_head.train(); losses = []
    for embeddings_batch, targets_batch in train_loader:
        embeddings_batch = embeddings_batch.to(DEVICE, non_blocking=True); targets_batch = targets_batch.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(length_head(embeddings_batch), targets_batch)
        loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    _, val_prediction, _ = predict_from_embeddings(length_head, validation_gt_embeddings)
    val_mae = float(np.mean(np.abs(val_prediction - val_y)))
    with torch.inference_mode():
        val_losses = [float(criterion(length_head(x.to(DEVICE)), y.to(DEVICE))) for x, y in val_loader]
    history.append({"epoch": epoch, "train_smooth_l1": float(np.mean(losses)),
                    "validation_smooth_l1": float(np.mean(val_losses)), "validation_mae_tokens": val_mae})
    print(f"Epoch {epoch:02d}: train={np.mean(losses):.6f} val={np.mean(val_losses):.6f} val_MAE={val_mae:.4f}")
    if val_mae < best_mae - 1e-9:
        best_mae, best_epoch, wait = val_mae, epoch, 0
        torch.save({"length_head_state_dict": {k: v.detach().cpu() for k, v in length_head.state_dict().items()},
                    "best_epoch": epoch, "validation_mae_tokens": val_mae,
                    "phase1_fingerprint": EXPECTED_FINGERPRINT}, LENGTH_MODEL / "best_model.pt")
    else:
        wait += 1
        if wait >= CFG.early_stopping_patience: print("Validation-only early stopping triggered"); break

history_frame = pd.DataFrame(history)
history_frame.to_csv(LENGTH_MODEL / "training_history.csv", index=False)
checkpoint = torch.load(LENGTH_MODEL / "best_model.pt", map_location="cpu", weights_only=True)
length_head = LengthHead(encoder_dimension).to(DEVICE)
length_head.load_state_dict(checkpoint["length_head_state_dict"], strict=True); length_head.eval()
for parameter in length_head.parameters(): parameter.requires_grad = False

val_raw, val_uncalibrated, val_clipped = predict_from_embeddings(length_head, validation_gt_embeddings)
calibration_slope, calibration_intercept = np.polyfit(val_raw.astype(float), val_y.astype(float), 1)
val_calibrated = np.clip(np.rint(calibration_slope * val_raw + calibration_intercept), train_lower, train_upper).astype(int)
use_calibration = bool(np.mean(np.abs(val_calibrated - val_y)) + 1e-9 < np.mean(np.abs(val_uncalibrated - val_y)))
print("Validation-only scalar calibration:", {"slope": calibration_slope, "intercept": calibration_intercept,
                                                "selected": use_calibration})


def apply_length_model(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    raw, uncalibrated, uncalibrated_clip_count = predict_from_embeddings(length_head, embeddings)
    if use_calibration:
        calibrated_rounded = np.rint(calibration_slope * raw + calibration_intercept)
        selected = np.clip(calibrated_rounded, train_lower, train_upper).astype(int)
        selected_clip_count = int((calibrated_rounded != selected).sum())
    else:
        selected, selected_clip_count = uncalibrated, uncalibrated_clip_count
    return uncalibrated, selected, selected_clip_count


# Test is touched only after checkpoint selection and validation-only calibration are frozen.
test_gt_embeddings = encode_glosses(test_frame[GLOSS_COL].astype(str).tolist(), "test GT-gloss encoding")
phase3_validation = pd.read_csv(PHASE3_ROOT / "results/validation_predictions.csv", dtype={"sample_id": str})
phase3_test = pd.read_csv(PHASE3_ROOT / "results/test_predictions.csv", dtype={"sample_id": str})


def join_phase3(base: pd.DataFrame, predictions: pd.DataFrame, split_name: str) -> pd.DataFrame:
    predictions["sample_id"] = predictions["sample_id"].astype(str)
    required = {"sample_id", "predicted_gloss"}
    if not required.issubset(predictions): raise RuntimeError(f"Phase-3 {split_name} missing {required - set(predictions)}")
    columns = ["sample_id", "predicted_gloss"] + [c for c in ("lexical_token_f1", "lexical_exact_match") if c in predictions]
    merged = base.merge(predictions[columns], left_on=ID_COL, right_on="sample_id", how="left", validate="one_to_one")
    if merged["predicted_gloss"].isna().any(): raise RuntimeError(f"Incomplete Phase-3 {split_name} join")
    return merged


validation_with_phase3 = join_phase3(validation_frame, phase3_validation, "validation")
test_with_phase3 = join_phase3(test_frame, phase3_test, "test")
validation_pred_embeddings = encode_glosses(validation_with_phase3["predicted_gloss"].astype(str).tolist(), "validation predicted-gloss encoding")
test_pred_embeddings = encode_glosses(test_with_phase3["predicted_gloss"].astype(str).tolist(), "test predicted-gloss encoding")
test_uncal_gt, test_selected_gt, test_clip_gt = apply_length_model(test_gt_embeddings)
val_uncal_gt, val_selected_gt, val_clip_gt = apply_length_model(validation_gt_embeddings)
val_uncal_pred, val_selected_pred, val_clip_pred = apply_length_model(validation_pred_embeddings)
test_uncal_pred, test_selected_pred, test_clip_pred = apply_length_model(test_pred_embeddings)

length_metric_rows = []
for split_name, gloss_source, truth, uncalibrated, selected, clip_count in (
    ("validation", "ground_truth", val_y, val_uncal_gt, val_selected_gt, val_clip_gt),
    ("test", "ground_truth", test_frame["motion_token_length"], test_uncal_gt, test_selected_gt, test_clip_gt),
    ("validation", "phase3_predicted", val_y, val_uncal_pred, val_selected_pred, val_clip_pred),
    ("test", "phase3_predicted", test_frame["motion_token_length"], test_uncal_pred, test_selected_pred, test_clip_pred),
):
    for calibration, prediction in (("uncalibrated", uncalibrated), ("selected_policy", selected)):
        length_metric_rows.append({"split": split_name, "gloss_source": gloss_source, "calibration": calibration,
                                   "clipped_predictions": clip_count, **length_metrics(truth, prediction)})
length_predictor_metrics = pd.DataFrame(length_metric_rows)
length_predictor_metrics.to_csv(RESULTS / "length_predictor_metrics.csv", index=False)
write_json(LENGTH_MODEL / "config.json", {
    "architecture": f"frozen T5 encoder ({encoder_dimension}) -> masked mean -> Linear({encoder_dimension},256) -> GELU -> Dropout(0.1) -> Linear(256,1)",
    "target": "log1p(motion_token_length)", "loss": "SmoothL1Loss", "optimizer": "AdamW",
    "encoder_frozen": True, "motion_model_frozen": True, "best_epoch": best_epoch,
    "selection": "lowest validation GT-gloss token MAE", "training_bounds": [train_lower, train_upper],
    "calibration": {"fit_split": "validation only", "slope": calibration_slope, "intercept": calibration_intercept,
                    "selected": use_calibration}, "config": asdict(CFG),
})

# %% [markdown]
# ## 6.1 and 6.8–6.10 — Reload frozen Phase 4, reproduce free-run decoding, and generate exact lengths

# %%
MODEL_CONFIG = read_json(PHASE4_ROOT / "best_model/model_config.json")


class GlossMotionTransformer(nn.Module):
    def __init__(self, gloss_encoder: nn.Module):
        super().__init__(); self.gloss_encoder = gloss_encoder
        d_model = int(MODEL_CONFIG["d_model"]); d_level = int(MODEL_CONFIG["d_level"])
        self.encoder_projection = nn.Linear(int(gloss_encoder.config.d_model), d_model)
        self.level_embeddings = nn.ModuleList([nn.Embedding(MOTION_VOCAB_SIZE, d_level, padding_idx=PAD_ID) for _ in range(6)])
        self.motion_projection = nn.Linear(6 * d_level, d_model)
        self.temporal_positions = nn.Embedding(int(MODEL_CONFIG["max_model_positions"]), d_model)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=int(MODEL_CONFIG["num_heads"]),
            dim_feedforward=int(MODEL_CONFIG["ffn_dimension"]), dropout=float(MODEL_CONFIG["dropout"]),
            activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=int(MODEL_CONFIG["decoder_layers"]), norm=nn.LayerNorm(d_model))
        self.heads = nn.ModuleList([nn.Linear(d_model, MOTION_VOCAB_SIZE) for _ in range(6)])

    def encode_gloss(self, input_ids, attention_mask):
        with torch.no_grad(): hidden = self.gloss_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.encoder_projection(hidden)

    def decode(self, tokens, memory, gloss_attention_mask, padding_mask):
        embeddings = torch.cat([self.level_embeddings[level](tokens[:, level]) for level in range(6)], dim=-1)
        positions = torch.arange(tokens.shape[-1], device=tokens.device).unsqueeze(0)
        embedded = self.motion_projection(embeddings) + self.temporal_positions(positions)
        causal = torch.triu(torch.ones(tokens.shape[-1], tokens.shape[-1], dtype=torch.bool, device=tokens.device), diagonal=1)
        hidden = self.decoder(tgt=embedded, memory=memory, tgt_mask=causal,
                              tgt_key_padding_mask=padding_mask, memory_key_padding_mask=~gloss_attention_mask.bool())
        return torch.stack([head(hidden) for head in self.heads], dim=1)


phase4_model = GlossMotionTransformer(frozen_encoder).to(DEVICE)
phase4_checkpoint = torch.load(PHASE4_CHECKPOINT, map_location="cpu", weights_only=True)
incompatible = phase4_model.load_state_dict(phase4_checkpoint["motion_state_dict"], strict=False)
expected_missing = [name for name in phase4_model.state_dict() if name.startswith("gloss_encoder.")]
if sorted(incompatible.missing_keys) != sorted(expected_missing) or incompatible.unexpected_keys:
    raise RuntimeError(f"Phase-4 checkpoint incompatibility: {incompatible}")
phase4_model.eval()
for parameter in phase4_model.parameters(): parameter.requires_grad = False
if any(parameter.requires_grad for parameter in phase4_model.parameters()): raise RuntimeError("Phase-4 model is not frozen")
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class PackedTokens:
    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as data:
            self.ids = data["sample_ids"].astype(str); self.lengths = data["lengths"].astype(int)
            self.offsets = data["offsets"].astype(int); self.flat = data["tokens_flat"].astype(np.int64)
        if self.flat.ndim != 2 or self.flat.shape[0] != 6 or not np.array_equal(np.diff(self.offsets), self.lengths):
            raise RuntimeError(f"Malformed packed tokens: {path}")
        if len(self.offsets) != len(self.ids) + 1 or self.offsets[0] != 0 or self.offsets[-1] != self.flat.shape[1]:
            raise RuntimeError(f"Invalid packed offsets: {path}")
        if self.flat.size and (self.flat.min() < 0 or self.flat.max() > 511): raise RuntimeError(f"Invalid tokens: {path}")
        self.index = {sample_id: i for i, sample_id in enumerate(self.ids)}
    def sequence(self, sample_id: str) -> np.ndarray:
        i = self.index[str(sample_id)]; return self.flat[:, self.offsets[i]:self.offsets[i + 1]]


FREE_ORACLE = PackedTokens(PHASE4_ROOT / "results/test_oracle_tokens.npz")
FREE_END = PackedTokens(PHASE4_ROOT / "results/test_end2end_tokens.npz")
if set(FREE_ORACLE.ids) != set(test_frame[ID_COL].astype(str)) or set(FREE_END.ids) != set(test_frame[ID_COL].astype(str)):
    raise RuntimeError("Saved Phase-4 test token IDs do not equal the locked supported test IDs")


@torch.inference_mode()
def free_run_batch(glosses: list[str]) -> list[np.ndarray]:
    """Exact Phase-4C greedy policy, preserving its original batch semantics."""
    encoded = tokenizer([str(gloss) for gloss in glosses], padding=True, truncation=False, return_tensors="pt")
    ids = encoded["input_ids"].to(DEVICE); mask = encoded["attention_mask"].to(DEVICE)
    with torch.autocast("cuda", dtype=AMP_DTYPE): memory = phase4_model.encode_gloss(ids, mask)
    batch_size = len(glosses)
    history = torch.full((batch_size, 6, 1), BOS_ID, dtype=torch.long, device=DEVICE)
    padding = torch.zeros((batch_size, 1), dtype=torch.bool, device=DEVICE)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)
    generated: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
    for step in range(CFG.supported_max_t + 1):
        with torch.autocast("cuda", dtype=AMP_DTYPE): logits = phase4_model.decode(history, memory, mask, padding)[:, :, -1].float()
        logits[:, :, PAD_ID] = -torch.inf; logits[:, :, BOS_ID] = -torch.inf
        choice = logits.argmax(-1)
        all_eos = (choice == EOS_ID).all(1) & ~finished
        finished |= all_eos
        if step == CFG.supported_max_t:
            finished[:] = True
            break
        real = logits[:, :, :512].argmax(-1)
        choice = torch.where((choice == EOS_ID) & ~all_eos[:, None], real, choice)
        append = torch.full_like(choice, PAD_ID); active = ~finished; append[active] = choice[active]
        for index in range(batch_size):
            if bool(active[index]): generated[index].append(choice[index].cpu())
        history = torch.cat([history, append.unsqueeze(-1)], dim=-1)
        padding = torch.cat([padding, finished.unsqueeze(-1)], dim=1)
        if bool(finished.all()): break
    return [(torch.stack(items, dim=1).numpy().astype(np.int64) if items else np.empty((6, 0), np.int64))
            for items in generated]


reproduction_ids = test_frame.sort_values(["motion_token_length", ID_COL]).iloc[
    np.linspace(0, len(test_frame) - 1, CFG.baseline_reproduction_samples).round().astype(int)
][ID_COL].astype(str).tolist()
reproduction_id_set = set(reproduction_ids)

# Phase 4C evaluated test samples in fixed length buckets. Reconstruct the entire original batch
# containing each probe so batch-shape-dependent GPU roundoff does not create autoregressive drift.
bucket_boundaries = (64, 96, 128, 181, 256, 384, 512)
bucket_batch_sizes = (8, 8, 8, 8, 4, 2, 1)
test_buckets: list[list[int]] = [[] for _ in bucket_boundaries]
for index, length in enumerate(test_frame["motion_token_length"].astype(int)):
    bucket = next(i for i, upper in enumerate(bucket_boundaries) if length <= upper)
    test_buckets[bucket].append(index)
regenerated_by_id: dict[str, np.ndarray] = {}
for indices, batch_size in zip(test_buckets, bucket_batch_sizes):
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        batch_ids = test_frame.iloc[batch_indices][ID_COL].astype(str).tolist()
        if not reproduction_id_set.intersection(batch_ids): continue
        batch_glosses = test_frame.iloc[batch_indices][GLOSS_COL].astype(str).tolist()
        batch_outputs = free_run_batch(batch_glosses)
        regenerated_by_id.update(dict(zip(batch_ids, batch_outputs)))
if not reproduction_id_set.issubset(regenerated_by_id):
    raise RuntimeError(f"Failed to reconstruct Phase-4 reproduction probes: {sorted(reproduction_id_set - set(regenerated_by_id))}")

reproduction_rows = []
for sample_id in reproduction_ids:
    regenerated, saved = regenerated_by_id[sample_id], FREE_ORACLE.sequence(sample_id)
    reference = parse_tokens_from_row(official_indexed.loc[sample_id])
    common = min(regenerated.shape[1], saved.shape[1])
    agreement = float((regenerated[:, :common] == saved[:, :common]).mean()) if common else float(regenerated.shape[1] == saved.shape[1])
    saved_aligned = min(reference.shape[1], saved.shape[1]); regenerated_aligned = min(reference.shape[1], regenerated.shape[1])
    saved_accuracy = float((reference[:, :saved_aligned] == saved[:, :saved_aligned]).mean()) if saved_aligned else 0.0
    regenerated_accuracy = float((reference[:, :regenerated_aligned] == regenerated[:, :regenerated_aligned]).mean()) if regenerated_aligned else 0.0
    reproduction_rows.append({"sample_id": sample_id, "saved_length": saved.shape[1], "regenerated_length": regenerated.shape[1],
                              "absolute_length_difference": abs(saved.shape[1] - regenerated.shape[1]),
                              "prefix_token_agreement": agreement, "saved_aligned_token_accuracy": saved_accuracy,
                              "regenerated_aligned_token_accuracy": regenerated_accuracy,
                              "absolute_metric_difference": abs(saved_accuracy - regenerated_accuracy),
                              "valid_ids": bool(regenerated.size == 0 or (regenerated.min() >= 0 and regenerated.max() <= 511))})
reproduction_passed = bool(all(row["valid_ids"] and row["absolute_length_difference"] <= 8 and row["absolute_metric_difference"] <= .02
                               for row in reproduction_rows))
write_json(DIAGNOSTICS / "phase4_baseline_reproduction.json", {"samples": reproduction_rows, "passed": reproduction_passed,
    "tolerance": {"max_absolute_length_difference": 8, "maximum_aligned_accuracy_difference": .02},
    "batch_policy": {"boundaries": bucket_boundaries, "batch_sizes": bucket_batch_sizes},
    "prefix_agreement_is_diagnostic_not_a_gate": True})
if not reproduction_passed:
    display(pd.DataFrame(reproduction_rows))
    raise RuntimeError("Phase-4 baseline reproduction failed; inspect the displayed per-sample differences")


def batch_size_for_length(length: int) -> int:
    if length <= 181: return 8
    if length <= 256: return 4
    if length <= 384: return 2
    return 1


@torch.inference_mode()
def exact_length_generate(frame: pd.DataFrame, gloss_column: str, target_lengths: np.ndarray, label: str) -> dict[str, np.ndarray]:
    records = [(i, str(frame.iloc[i][ID_COL]), str(frame.iloc[i][gloss_column]), int(target_lengths[i])) for i in range(len(frame))]
    records.sort(key=lambda x: (x[3], x[1])); outputs = {}
    cursor = 0
    while cursor < len(records):
        size = batch_size_for_length(records[cursor][3]); batch = records[cursor:cursor + size]; cursor += len(batch)
        texts = [item[2] for item in batch]; lengths = [item[3] for item in batch]; maximum = max(lengths)
        encoded = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        ids = encoded["input_ids"].to(DEVICE); mask = encoded["attention_mask"].to(DEVICE)
        with torch.autocast("cuda", dtype=AMP_DTYPE): memory = phase4_model.encode_gloss(ids, mask)
        history = torch.full((len(batch), 6, 1), BOS_ID, dtype=torch.long, device=DEVICE)
        padding = torch.zeros((len(batch), 1), dtype=torch.bool, device=DEVICE); steps = []
        for _ in range(maximum):
            with torch.autocast("cuda", dtype=AMP_DTYPE): logits = phase4_model.decode(history, memory, mask, padding)[:, :, -1].float()
            choice = logits[:, :, :512].argmax(-1)  # PAD/BOS/EOS are structurally impossible.
            steps.append(choice.cpu()); history = torch.cat([history, choice.unsqueeze(-1)], dim=-1)
            padding = torch.cat([padding, torch.zeros((len(batch), 1), dtype=torch.bool, device=DEVICE)], dim=1)
        stacked = torch.stack(steps, dim=-1).numpy().astype(np.int64)
        for index, (_, sample_id, _, target) in enumerate(batch): outputs[sample_id] = stacked[index, :, :target].copy()
        if len(outputs) % 100 < len(batch): print(f"{label}: {len(outputs)}/{len(records)}")
    if any(array.shape != (6, int(target_lengths[frame.index[frame[ID_COL].astype(str) == sid][0]])) for sid, array in outputs.items()):
        raise RuntimeError(f"{label}: exact-length invariant failed")
    if any(array.min() < 0 or array.max() > 511 for array in outputs.values()): raise RuntimeError(f"{label}: invalid generated ID")
    return outputs


def save_token_npz(path: Path, ordered_ids: list[str], token_map: dict[str, np.ndarray]) -> None:
    arrays = [token_map[str(sample_id)] for sample_id in ordered_ids]
    lengths = np.asarray([x.shape[1] for x in arrays], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    flat = np.concatenate(arrays, axis=1)
    np.savez_compressed(path, sample_ids=np.asarray(ordered_ids, dtype=str), lengths=lengths, offsets=offsets, tokens_flat=flat)


test_ids = test_frame[ID_COL].astype(str).tolist()
oracle_length_test = exact_length_generate(test_frame, GLOSS_COL, test_frame["motion_token_length"].to_numpy(int), "oracle-length test")
predicted_oracle_test = exact_length_generate(test_frame, GLOSS_COL, test_selected_gt, "predicted-length GT-gloss test")
predicted_end_test = exact_length_generate(test_with_phase3, "predicted_gloss", test_selected_pred, "predicted-length end-to-end test")
save_token_npz(TOKENS / "oracle_length_test.npz", test_ids, oracle_length_test)
save_token_npz(TOKENS / "predicted_length_oracle_gloss_test.npz", test_ids, predicted_oracle_test)
save_token_npz(TOKENS / "predicted_length_end2end_test.npz", test_ids, predicted_end_test)

if CFG.generate_validation_tokens:
    val_ids = validation_frame[ID_COL].astype(str).tolist()
    oracle_length_val = exact_length_generate(validation_frame, GLOSS_COL, val_y.astype(int), "oracle-length validation")
    predicted_oracle_val = exact_length_generate(validation_frame, GLOSS_COL, val_selected_gt, "predicted-length GT-gloss validation")
    predicted_end_val = exact_length_generate(validation_with_phase3, "predicted_gloss", val_selected_pred, "predicted-length end-to-end validation")
    save_token_npz(TOKENS / "oracle_length_validation.npz", val_ids, oracle_length_val)
    save_token_npz(TOKENS / "predicted_length_oracle_gloss_validation.npz", val_ids, predicted_oracle_val)
    save_token_npz(TOKENS / "predicted_length_end2end_validation.npz", val_ids, predicted_end_val)

# %% [markdown]
# ## 6.11 and 6.17 — Token-space metrics and first-segment agreement

# %%
try:
    from rapidfuzz.distance import Levenshtein
    def edit_distance(a, b): return int(Levenshtein.distance(a, b))
except ImportError:
    def edit_distance(a, b):
        previous = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            current = [i]
            for j, y in enumerate(b, 1): current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + int(x != y)))
            previous = current
        return previous[-1]


def reference_tokens(sample_id: str) -> np.ndarray: return parse_tokens_from_row(official_indexed.loc[str(sample_id)])


def evaluate_tokens(label: str, token_map: dict[str, np.ndarray]) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []; level_correct = np.zeros(6); level_total = np.zeros(6); micro_correct = micro_total = 0
    for sample_id in test_ids:
        reference, prediction = reference_tokens(sample_id), token_map[sample_id]
        aligned = min(reference.shape[1], prediction.shape[1]); level_accuracy = []; edits = []
        for level in range(6):
            correct = int((reference[level, :aligned] == prediction[level, :aligned]).sum()) if aligned else 0
            level_correct[level] += correct; level_total[level] += aligned; level_accuracy.append(correct / max(aligned, 1))
            edits.append(edit_distance(reference[level].tolist(), prediction[level].tolist()) / max(reference.shape[1], prediction.shape[1], 1))
        correct = int((reference[:, :aligned] == prediction[:, :aligned]).sum()) if aligned else 0
        micro_correct += correct; micro_total += 6 * aligned; error = prediction.shape[1] - reference.shape[1]
        rows.append({"sample_id": sample_id, "variant": label, "ground_truth_length": reference.shape[1],
                     "generated_length": prediction.shape[1], "length_error": error, "absolute_length_error": abs(error),
                     "relative_length_error": abs(error) / reference.shape[1], "aligned_token_accuracy": correct / max(6 * aligned, 1),
                     "normalized_token_edit_distance": float(np.mean(edits)),
                     **{f"level_{i}_prefix_accuracy": level_accuracy[i] for i in range(6)}})
    frame = pd.DataFrame(rows)
    summary = {"variant": label, "samples": len(frame), "length_mae": float(frame.absolute_length_error.mean()),
               "length_rmse": float(np.sqrt(np.mean(frame.length_error.astype(float) ** 2))),
               "mean_signed_length_error": float(frame.length_error.mean()),
               "relative_length_error": float(frame.relative_length_error.mean()),
               "length_spearman": spearman(frame.ground_truth_length, frame.generated_length),
               "aligned_token_accuracy": float(frame.aligned_token_accuracy.mean()),
               "micro_aligned_token_accuracy": micro_correct / max(micro_total, 1),
               "per_level_prefix_accuracy": (level_correct / np.maximum(level_total, 1)).tolist(),
               "normalized_token_edit_distance": float(frame.normalized_token_edit_distance.mean())}
    return summary, frame


free_oracle_map = {sid: FREE_ORACLE.sequence(sid) for sid in test_ids}
free_end_map = {sid: FREE_END.sequence(sid) for sid in test_ids}
token_variants = {
    "original_free_run_oracle": free_oracle_map, "original_free_run_end2end": free_end_map,
    "oracle_gt_length": oracle_length_test, "predicted_length_gt_gloss": predicted_oracle_test,
    "predicted_length_predicted_gloss": predicted_end_test,
}
token_summaries, token_frames = {}, {}
for label, mapping in token_variants.items(): token_summaries[label], token_frames[label] = evaluate_tokens(label, mapping)


def prefix_agreement(first: np.ndarray, second: np.ndarray) -> float:
    common = min(first.shape[1], second.shape[1]); return float((first[:, :common] == second[:, :common]).mean()) if common else float("nan")


prefix_rows = []
for sample_id in test_ids:
    prefix_rows.extend([
        {"sample_id": sample_id, "comparison": "free_oracle_vs_oracle_gt_length", "prefix_token_agreement": prefix_agreement(free_oracle_map[sample_id], oracle_length_test[sample_id])},
        {"sample_id": sample_id, "comparison": "free_oracle_vs_predicted_length_gt_gloss", "prefix_token_agreement": prefix_agreement(free_oracle_map[sample_id], predicted_oracle_test[sample_id])},
        {"sample_id": sample_id, "comparison": "free_end2end_vs_predicted_length_end2end", "prefix_token_agreement": prefix_agreement(free_end_map[sample_id], predicted_end_test[sample_id])},
    ])
pd.DataFrame(prefix_rows).to_csv(RESULTS / "first_segment_agreement.csv", index=False)

# %% [markdown]
# ## 6.12–6.16 — Fixed decoder, feature-space metrics, and fair collapse thresholds

# %%
contract = read_json(DECODER_CONTRACT_PATH)
if "candidateb" not in canonical(json.dumps(contract, sort_keys=True)): raise RuntimeError("Decoder contract is not verified Candidate B")
motion_mean = np.load(MOTION_MEAN_PATH, allow_pickle=False).astype(np.float32)
motion_std = np.load(MOTION_STD_PATH, allow_pickle=False).astype(np.float32)
if motion_mean.shape != (668,) or motion_std.shape != (668,) or not np.isfinite(motion_mean).all() or not np.isfinite(motion_std).all():
    raise RuntimeError("Invalid Phase-5A normalization")
active_dimensions = motion_std >= CFG.near_zero_std; safe_std = motion_std.copy(); safe_std[~active_dimensions] = 1.0


class MotionDecoder(nn.Module):
    def __init__(self):
        super().__init__(); self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 512, 3, 2, 1, output_padding=1), nn.ReLU(), nn.BatchNorm1d(512),
            nn.ConvTranspose1d(512, 512, 3, 2, 1, output_padding=1), nn.ReLU(), nn.BatchNorm1d(512),
            nn.ConvTranspose1d(512, 512, 3, 1, 1), nn.ReLU(), nn.BatchNorm1d(512),
            nn.ConvTranspose1d(512, 668, 3, 1, 1))
    def forward(self, x): return self.decoder(x)


checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
state = checkpoint
for key in ("model_state_dict", "state_dict"):
    if isinstance(state, dict) and key in state and isinstance(state[key], dict): state = state[key]; break
state = {key: value for key, value in state.items() if isinstance(value, torch.Tensor)}
decoder_state = {key[len("decoder."):]: value for key, value in state.items() if key.startswith("decoder.")}
motion_decoder = MotionDecoder(); load_result = motion_decoder.load_state_dict(decoder_state, strict=True)
if load_result.missing_keys or load_result.unexpected_keys: raise RuntimeError("Strict decoder load failed")
codebooks = []
for level in range(6):
    tensor = state[f"rvq.quantizers.{level}.embedding"]
    if tuple(tensor.shape) != (256, 512): raise RuntimeError(f"Wrong codebook shape at L{level}: {tuple(tensor.shape)}")
    codebooks.append(tensor.float())
codebooks = torch.stack(codebooks).to(DEVICE); motion_decoder = motion_decoder.to(DEVICE).eval()
mean_tensor = torch.from_numpy(motion_mean).to(DEVICE); std_tensor = torch.from_numpy(safe_std).to(DEVICE)


@torch.inference_mode()
def decode_tokens(tokens: np.ndarray) -> np.ndarray:
    if tokens.ndim != 2 or tokens.shape[0] != 6 or tokens.shape[1] <= 0 or tokens.min() < 0 or tokens.max() > 511:
        raise RuntimeError(f"Invalid decoder input {tokens.shape}")
    ids = torch.as_tensor(tokens, dtype=torch.long, device=DEVICE)
    latent = torch.zeros((256, tokens.shape[1]), device=DEVICE)
    for level in range(6): latent += codebooks[level][:, ids[level]]
    normalized = motion_decoder(latent.unsqueeze(0)).squeeze(0).transpose(0, 1)
    motion = (normalized * std_tensor + mean_tensor).cpu().numpy().astype(np.float32)
    if motion.shape != (4 * tokens.shape[1], 668) or not np.isfinite(motion).all(): raise RuntimeError("Decoded motion invariant failed")
    return motion


gt_index = {path.stem: path for path in MOTION_FEATURES.glob("*.npy")}
if any(sample_id not in gt_index for sample_id in test_ids): raise RuntimeError("Missing test Motion-Features")
def load_gt(sample_id: str) -> np.ndarray:
    value = np.load(gt_index[sample_id], allow_pickle=False).astype(np.float32)
    if value.ndim != 2 or value.shape[1] != 668 or not np.isfinite(value).all(): raise RuntimeError(f"Invalid GT motion {sample_id}")
    return value


def decode_collection(label: str, mapping: dict[str, np.ndarray], directory: Path) -> None:
    for number, sample_id in enumerate(test_ids, 1):
        np.save(directory / f"{sample_id}.npy", decode_tokens(mapping[sample_id]), allow_pickle=False)
        if number % 100 == 0: print(f"{label} decode: {number}/{len(test_ids)}")


decode_collection("oracle length", oracle_length_test, ORACLE_DECODED)
decode_collection("predicted length + GT gloss", predicted_oracle_test, PRED_ORACLE_DECODED)
decode_collection("predicted length end-to-end", predicted_end_test, PRED_END_DECODED)


def linear_resample(motion: np.ndarray, target: int) -> np.ndarray:
    if len(motion) == target: return motion.copy()
    tensor = torch.from_numpy(motion.T).unsqueeze(0)
    return F.interpolate(tensor, size=target, mode="linear", align_corners=True).squeeze(0).T.numpy()


def normalized_active(motion: np.ndarray) -> np.ndarray:
    return ((motion[:, active_dimensions] - motion_mean[active_dimensions]) / motion_std[active_dimensions]).astype(np.float32)


def velocity_magnitude(motion: np.ndarray) -> np.ndarray:
    z = normalized_active(motion)
    return np.sqrt(np.mean(np.square(np.diff(z, axis=0), dtype=np.float64), axis=1)) if len(z) >= 2 else np.empty(0)


def repetition_score(values: np.ndarray, max_lag: int = 180) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(x) < 6: return float("nan")
    x -= x.mean(); limit = min(max_lag, len(x) // 2); correlations = []
    for lag in range(2, limit + 1):
        a, b = x[:-lag], x[lag:]; denominator = np.sqrt(np.dot(a, a) * np.dot(b, b))
        if denominator > 0: correlations.append(float(np.dot(a, b) / denominator))
    return max(correlations) if correlations else float("nan")


thresholds = read_json(PHASE5B_ROOT / "diagnostics/empirical_motion_thresholds.json")
static_velocity_threshold = float(thresholds["static_velocity_threshold_gt_percentile_1"])
static_variance_threshold = float(thresholds["static_variance_threshold_gt_percentile_1"])
extreme_min_threshold = float(thresholds["extreme_minimum_threshold_gt_sample_min_percentile_0_1"])
extreme_max_threshold = float(thresholds["extreme_maximum_threshold_gt_sample_max_percentile_99_9"])
extreme_z_threshold = float(thresholds["extreme_normalized_magnitude_threshold_gt_percentile_99_9"])


def feature_metrics(sample_id: str, predicted: np.ndarray) -> dict[str, Any]:
    ground_truth = load_gt(sample_id); resampled = linear_resample(predicted, len(ground_truth))
    raw_difference = resampled.astype(np.float64) - ground_truth.astype(np.float64)
    z_pred, z_gt = normalized_active(resampled), normalized_active(ground_truth)
    z_difference = z_pred.astype(np.float64) - z_gt.astype(np.float64)
    velocity_difference = np.diff(z_pred, axis=0).astype(np.float64) - np.diff(z_gt, axis=0).astype(np.float64)
    velocity = velocity_magnitude(predicted)
    mean_velocity = float(velocity.mean()) if velocity.size else 0.0
    variance = float(np.mean(np.var(normalized_active(predicted), axis=0, dtype=np.float64)))
    minimum, maximum = float(predicted.min()), float(predicted.max())
    maximum_z = float(np.max(np.abs(normalized_active(predicted))))
    return {"sample_id": sample_id, "gt_frames": len(ground_truth), "pred_frames": len(predicted),
            "frame_error": len(predicted) - len(ground_truth), "absolute_frame_error": abs(len(predicted) - len(ground_truth)),
            "time_normalized_raw_MAE": float(np.mean(np.abs(raw_difference))),
            "normalized_feature_MAE": float(np.mean(np.abs(z_difference))),
            "velocity_MAE": float(np.mean(np.abs(velocity_difference))), "mean_velocity_magnitude": mean_velocity,
            "feature_temporal_variance": variance, "repetition_score": repetition_score(velocity),
            "collapse_flag": bool(mean_velocity < static_velocity_threshold or variance < static_variance_threshold),
            "extreme_motion_flag": bool(minimum < extreme_min_threshold or maximum > extreme_max_threshold or maximum_z > extreme_z_threshold)}


def evaluate_decoded(label: str, directory: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    for number, sample_id in enumerate(test_ids, 1):
        rows.append(feature_metrics(sample_id, np.load(directory / f"{sample_id}.npy", allow_pickle=False)))
        if number % 100 == 0: print(f"{label} metrics: {number}/{len(test_ids)}")
    frame = pd.DataFrame(rows)
    summary = {"method": label, "samples": len(frame), "duration_mae_frames": float(frame.absolute_frame_error.mean()),
               "duration_rmse_frames": float(np.sqrt(np.mean(frame.frame_error.astype(float) ** 2))),
               "duration_mean_signed_error": float(frame.frame_error.mean()),
               "time_normalized_raw_mae": float(frame.time_normalized_raw_MAE.mean()),
               "normalized_feature_mae": float(frame.normalized_feature_MAE.mean()),
               "velocity_mae": float(frame.velocity_MAE.mean()), "motion_variance": float(frame.feature_temporal_variance.mean()),
               "repetition_score": float(frame.repetition_score.mean()), "collapse_rate": float(frame.collapse_flag.mean()),
               "extreme_motion_rate": float(frame.extreme_motion_flag.mean())}
    return summary, frame


new_feature = {}
for label, directory in (("oracle_gt_length", ORACLE_DECODED), ("predicted_length_gt_gloss", PRED_ORACLE_DECODED),
                         ("predicted_length_predicted_gloss", PRED_END_DECODED)):
    summary, frame = evaluate_decoded(label, directory); new_feature[label] = (summary, frame)

phase5_oracle_metrics = pd.read_csv(PHASE5B_ROOT / "results/oracle_motion_metrics.csv", dtype={"sample_id": str})
phase5_end_metrics = pd.read_csv(PHASE5B_ROOT / "results/end2end_motion_metrics.csv", dtype={"sample_id": str})
phase5_collapse = pd.read_csv(PHASE5B_ROOT / "diagnostics/motion_collapse_flags.csv", dtype={"sample_id": str})
phase5_extreme = pd.read_csv(PHASE5B_ROOT / "diagnostics/extreme_motion_flags.csv", dtype={"sample_id": str})


def bool_series(series: pd.Series) -> pd.Series:
    return series if pd.api.types.is_bool_dtype(series) else series.astype(str).str.casefold().map({"true": True, "false": False})


def old_summary(label: str, frame: pd.DataFrame, variant: str) -> dict[str, Any]:
    collapse = phase5_collapse.loc[phase5_collapse.variant == variant].set_index("sample_id").loc[test_ids]
    extreme = phase5_extreme.loc[phase5_extreme.variant == variant].set_index("sample_id").loc[test_ids]
    repetitions = []
    directory = PHASE5B_ROOT / "decoded" / ("oracle" if variant == "oracle" else "end2end")
    for sample_id in test_ids: repetitions.append(repetition_score(velocity_magnitude(np.load(directory / f"{sample_id}.npy", allow_pickle=False))))
    return {"method": label, "samples": len(frame), "duration_mae_frames": float(frame.absolute_frame_error.mean()),
            "duration_rmse_frames": float(np.sqrt(np.mean(frame.frame_error.astype(float) ** 2))),
            "duration_mean_signed_error": float(frame.frame_error.mean()),
            "time_normalized_raw_mae": float(frame.time_normalized_raw_MAE.mean()),
            "normalized_feature_mae": float(frame.normalized_feature_MAE.mean()), "velocity_mae": float(frame.velocity_MAE.mean()),
            "motion_variance": float(frame.feature_temporal_variance.mean()), "repetition_score": float(np.nanmean(repetitions)),
            "collapse_rate": float(bool_series(collapse.static_motion).mean()),
            "extreme_motion_rate": float(bool_series(extreme.extreme_motion).mean())}


main_summaries = [
    old_summary("original_free_run_oracle", phase5_oracle_metrics, "oracle"),
    old_summary("original_free_run_end2end", phase5_end_metrics, "end-to-end"),
    new_feature["oracle_gt_length"][0], new_feature["predicted_length_gt_gloss"][0],
    new_feature["predicted_length_predicted_gloss"][0],
]
main_comparison = pd.DataFrame(main_summaries)
main_comparison.to_csv(RESULTS / "phase6_main_comparison.csv", index=False)

for label, filename in (("oracle_gt_length", "oracle_length_metrics.csv"),
                        ("predicted_length_gt_gloss", "predicted_length_oracle_metrics.csv"),
                        ("predicted_length_predicted_gloss", "predicted_length_end2end_metrics.csv")):
    merged = token_frames[label].merge(new_feature[label][1], on="sample_id", validate="one_to_one")
    merged.to_csv(RESULTS / filename, index=False)

collapse_reanalysis = pd.concat([frame.assign(variant=label) for label, (_, frame) in new_feature.items()], ignore_index=True)
collapse_reanalysis.to_csv(DIAGNOSTICS / "collapse_reanalysis.csv", index=False)

# %% [markdown]
# ## 6.18–6.23 — Tail repair, predictor errors, causal finding, figures, and handoff

# %%
tail_analysis = pd.read_csv(PHASE5C_ROOT / "results/overgeneration_tail_analysis.csv", dtype={"sample_id": str})
selected_tails = tail_analysis.loc[(tail_analysis["variant"] == "end-to-end") &
                                   (tail_analysis["empirical_tail_behavior"] == "REPETITIVE_TAIL")].drop_duplicates("sample_id")
old_end_collapse = phase5_collapse.loc[phase5_collapse.variant == "end-to-end"].set_index("sample_id")
new_end_flags = new_feature["predicted_length_predicted_gloss"][1].set_index("sample_id")
tail_repair_rows = []
for _, row in selected_tails.iterrows():
    sample_id = str(row.sample_id); new_motion = np.load(PRED_END_DECODED / f"{sample_id}.npy", allow_pickle=False)
    gt_frames = int(row.gt_frames); new_tail = new_motion[gt_frames:]
    tail_repair_rows.append({"sample_id": sample_id, "previous_predicted_frames": int(row.predicted_frames),
        "new_predicted_frames": len(new_motion), "GT_frames": gt_frames, "previous_tail_frames": int(row.excess_frames),
        "new_tail_frames": max(len(new_motion) - gt_frames, 0), "previous_repetition_score": row.tail_repetition_score,
        "new_repetition_score": repetition_score(velocity_magnitude(new_tail)) if len(new_tail) else float("nan"),
        "previous_collapse_flag": bool_series(old_end_collapse.loc[[sample_id], "static_motion"]).iloc[0],
        "new_collapse_flag": bool(new_end_flags.loc[sample_id, "collapse_flag"])})
pd.DataFrame(tail_repair_rows, columns=["sample_id", "previous_predicted_frames", "new_predicted_frames", "GT_frames",
    "previous_tail_frames", "new_tail_frames", "previous_repetition_score", "new_repetition_score",
    "previous_collapse_flag", "new_collapse_flag"]).to_csv(RESULTS / "repetitive_tail_repair.csv", index=False)

error_analysis = pd.DataFrame({"sample_id": test_ids, "true_length": test_frame.motion_token_length.to_numpy(int),
                               "predicted_length": test_selected_pred, "gloss_word_count": test_frame.gloss_token_count.to_numpy(int)})
error_analysis["residual"] = error_analysis.predicted_length - error_analysis.true_length
error_analysis["absolute_error"] = error_analysis.residual.abs()
train_vocabulary = {token.casefold() for text in train_frame[GLOSS_COL].astype(str) for token in TOKEN_PATTERN.findall(text)}
error_analysis["oov_gloss_terms"] = [sum(token.casefold() not in train_vocabulary for token in TOKEN_PATTERN.findall(text))
                                      for text in test_with_phase3.predicted_gloss.astype(str)]
if "lexical_token_f1" in test_with_phase3: error_analysis["phase3_gloss_f1"] = test_with_phase3.lexical_token_f1.to_numpy(float)
error_analysis.to_csv(DIAGNOSTICS / "length_predictor_test_errors.csv", index=False)
diagnostic_correlations = {column: spearman(error_analysis.absolute_error, error_analysis[column])
                           for column in ("gloss_word_count", "true_length", "oov_gloss_terms")}
if "phase3_gloss_f1" in error_analysis: diagnostic_correlations["phase3_gloss_f1"] = spearman(error_analysis.absolute_error, error_analysis.phase3_gloss_f1)
write_json(DIAGNOSTICS / "length_predictor_diagnostics.json", {
    "correlations_with_absolute_error_spearman": diagnostic_correlations,
    "short_half_mean_signed_error": float(error_analysis.loc[error_analysis.true_length <= error_analysis.true_length.median(), "residual"].mean()),
    "long_half_mean_signed_error": float(error_analysis.loc[error_analysis.true_length > error_analysis.true_length.median(), "residual"].mean()),
    "training_bounds": [train_lower, train_upper], "calibration_selected": use_calibration,
})


def save_length_figures() -> None:
    truth = error_analysis.true_length; prediction = error_analysis.predicted_length; residual = error_analysis.residual
    fig, ax = plt.subplots(figsize=(6, 6)); ax.scatter(truth, prediction, s=12, alpha=.35); ax.plot([0, 512], [0, 512], "k--")
    ax.set(xlabel="True token length", ylabel="Predicted token length", title="Length predictor: test predicted gloss")
    fig.tight_layout(); fig.savefig(FIGURES / "length_predictor_scatter.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(truth, residual, s=12, alpha=.35); ax.axhline(0, color="black", linestyle="--")
    ax.set(xlabel="True token length", ylabel="Prediction residual", title="Length residual vs true length")
    fig.tight_layout(); fig.savefig(FIGURES / "length_residuals.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.hist(residual, bins=50); ax.axvline(0, color="black", linestyle="--")
    ax.set(xlabel="Token-length residual", ylabel="Samples", title="Length error histogram")
    fig.tight_layout(); fig.savefig(FIGURES / "length_error_histogram.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5)); ax.bar(main_comparison.method, main_comparison.duration_mae_frames)
    ax.tick_params(axis="x", rotation=25); ax.set(ylabel="Duration MAE (frames)", title="Phase 6 duration comparison")
    fig.tight_layout(); fig.savefig(FIGURES / "duration_comparison.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5)); ax.bar(main_comparison.method, main_comparison.collapse_rate)
    ax.tick_params(axis="x", rotation=25); ax.set(ylabel="Collapse rate", title="Fair-threshold collapse comparison")
    fig.tight_layout(); fig.savefig(FIGURES / "collapse_comparison.png", dpi=170); plt.close(fig)


save_length_figures()
summary_by_method = main_comparison.set_index("method").to_dict("index")
free_oracle = summary_by_method["original_free_run_oracle"]
free_end = summary_by_method["original_free_run_end2end"]
oracle_exact = summary_by_method["oracle_gt_length"]
pred_oracle = summary_by_method["predicted_length_gt_gloss"]
pred_end = summary_by_method["predicted_length_predicted_gloss"]


def delta(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
    keys = ["duration_mae_frames", "duration_mean_signed_error", "normalized_feature_mae", "velocity_mae",
            "collapse_rate", "repetition_score", "extreme_motion_rate"]
    return {key: float(second[key] - first[key]) for key in keys}


duration_fraction_removed = 1.0 - oracle_exact["duration_mae_frames"] / max(free_oracle["duration_mae_frames"], 1e-12)
deep_accuracy = float(np.mean(token_summaries["oracle_gt_length"]["per_level_prefix_accuracy"][3:]))
base_accuracy = float(token_summaries["oracle_gt_length"]["per_level_prefix_accuracy"][0])
exposure_evidence = oracle_exact["collapse_rate"] >= .25 or oracle_exact["repetition_score"] >= free_oracle["repetition_score"] * .9
hierarchy_evidence = base_accuracy - deep_accuracy >= .10
if duration_fraction_removed >= .80 and exposure_evidence and hierarchy_evidence: dominant = "MIXED"
elif duration_fraction_removed >= .80 and exposure_evidence: dominant = "EXPOSURE_BIAS"
elif hierarchy_evidence: dominant = "RVQ_HIERARCHY"
else: dominant = "LENGTH" if duration_fraction_removed < .80 else "MIXED"
recommendation = {"LENGTH": "integrate learned length conditioning", "EXPOSURE_BIAS": "target autoregressive exposure bias",
                  "RVQ_HIERARCHY": "investigate hierarchical coarse-to-fine RVQ prediction",
                  "MIXED": "target exposure bias first, then evaluate hierarchical RVQ prediction"}[dominant]
causal_analysis = {
    "length_only_benefit_free_oracle_to_oracle_length": delta(free_oracle, oracle_exact),
    "length_predictor_gap_oracle_length_to_predicted_length_gt_gloss": delta(oracle_exact, pred_oracle),
    "phase3_propagation_predicted_length_gt_to_predicted_gloss": delta(pred_oracle, pred_end),
    "duration_mae_fraction_removed_by_oracle_length": duration_fraction_removed,
    "remaining_collapse_after_oracle_length": oracle_exact["collapse_rate"],
    "oracle_length_base_vs_deep_level_accuracy_gap": base_accuracy - deep_accuracy,
    "dominant_remaining_failure": dominant, "recommended_next_phase": recommendation,
}
write_json(RESULTS / "phase6_causal_analysis.json", causal_analysis)

phase6_complete = bool(reproduction_passed and len(length_targets) == sum(EXPECTED_SUPPORTED.values())
                       and len(test_ids) == EXPECTED_SUPPORTED["test"]
                       and all(len(mapping) == len(test_ids) for mapping in (oracle_length_test, predicted_oracle_test, predicted_end_test))
                       and all((directory / f"{sample_id}.npy").is_file() for directory in (ORACLE_DECODED, PRED_ORACLE_DECODED, PRED_END_DECODED) for sample_id in test_ids))
selected_metric = lambda split, source: length_predictor_metrics.query(
    "split == @split and gloss_source == @source and calibration == 'selected_policy'").iloc[0].to_dict()
phase6_report = {
    "phase": "Phase 6 - Length-Controlled Motion Generation Repair", "phase6_complete": phase6_complete,
    "phase1_fingerprint": EXPECTED_FINGERPRINT, "motion_model_retrained": False,
    "length_predictor": {"architecture": f"frozen T5 {encoder_dimension} -> 256 -> 1 log length", "encoder_frozen": True,
        "best_epoch": best_epoch, "validation": selected_metric("validation", "ground_truth"),
        "test_oracle_gloss": selected_metric("test", "ground_truth"),
        "test_predicted_gloss": selected_metric("test", "phase3_predicted")},
    "baseline_free_running": {"oracle_frame_mae": free_oracle["duration_mae_frames"],
        "oracle_signed_bias": free_oracle["duration_mean_signed_error"], "end2end_frame_mae": free_end["duration_mae_frames"],
        "end2end_signed_bias": free_end["duration_mean_signed_error"]},
    "oracle_length": oracle_exact, "predicted_length_oracle_gloss": pred_oracle,
    "predicted_length_end2end": pred_end, "token_space": token_summaries, "causal_analysis": causal_analysis,
    "dominant_remaining_failure": dominant, "recommended_next_phase": recommendation, "ready_for_phase7": phase6_complete,
    "semantic_motion_quality_claimed": False, "mpjpe_calculated": False, "windowed_eos_experiment": "not performed; exact-length policy is primary",
}
write_json(OUTPUT / "phase6_config.json", {"config": asdict(CFG), "resolved_paths": artifact_verification,
                                            "selection_and_calibration": "validation only"})
write_json(OUTPUT / "phase6_report.json", phase6_report)

readme = f"""PHASE 7 HANDOFF — MOTION-S / SIGNVRSE

1. Length predictor
Frozen Phase-3 T5 encoder ({encoder_dimension} dimensions), attention-mask-aware mean pooling,
Linear({encoder_dimension},256), GELU, Dropout(0.1), Linear(256,1), trained on log1p token length.
Best checkpoint: length_model/best_model.pt; best epoch: {best_epoch}. Only the MLP head was trained.

2. Data and selection
Locked Phase-1 fingerprint: {EXPECTED_FINGERPRINT}. Supported T<=512 counts: {EXPECTED_SUPPORTED}.
Model selection and optional scalar calibration used validation only. Test was evaluated after freezing.
Training-derived inference bounds: [{train_lower},{train_upper}]. Calibration selected: {use_calibration}.

3. Exact-length generation
The Phase-4C checkpoint remained frozen. PAD/BOS/EOS logits were excluded by taking argmax over IDs 0..511,
and generation ran for exactly the requested temporal length. Oracle length is diagnostic only.

4. Results
Validation GT-gloss length MAE: {selected_metric('validation', 'ground_truth')['mae_tokens']:.6f} tokens.
Test GT-gloss length MAE: {selected_metric('test', 'ground_truth')['mae_tokens']:.6f} tokens.
Test predicted-gloss length MAE: {selected_metric('test', 'phase3_predicted')['mae_tokens']:.6f} tokens.
Original oracle duration MAE: {free_oracle['duration_mae_frames']:.6f} frames.
Oracle-length duration MAE: {oracle_exact['duration_mae_frames']:.6f} frames.
Predicted-length end-to-end duration MAE: {pred_end['duration_mae_frames']:.6f} frames.
Original oracle collapse rate: {free_oracle['collapse_rate']:.6f}; oracle-length: {oracle_exact['collapse_rate']:.6f}.
Original oracle repetition: {free_oracle['repetition_score']:.6f}; oracle-length: {oracle_exact['repetition_score']:.6f}.

5. Scientific interpretation
Length-only duration fraction removed: {duration_fraction_removed:.6f}.
Dominant remaining bottleneck: {dominant}.
Recommended Phase 7: {recommendation}.
Do not claim semantic sign quality, compute MPJPE, construct a skeleton, or export BVH until the exact 668-D
layout and provider inverse are verified.
"""
(OUTPUT / "README_phase7.txt").write_text(readme, encoding="utf-8")

inventory_rows = []
for path in sorted(OUTPUT.rglob("*")):
    if path.is_file() and path.name != "artifact_inventory.csv":
        inventory_rows.append({"relative_path": str(path.relative_to(OUTPUT)), "bytes": path.stat().st_size,
                               "sha256": sha256_file(path)})
pd.DataFrame(inventory_rows).to_csv(OUTPUT / "artifact_inventory.csv", index=False)

val_metric = selected_metric("validation", "ground_truth")
test_gt_metric = selected_metric("test", "ground_truth")
test_pred_metric = selected_metric("test", "phase3_predicted")
print("=" * 60); print("PHASE 6 COMPLETE"); print("=" * 60)
print("Phase-4 model: FROZEN")
print(f"Length predictor validation MAE: {val_metric['mae_tokens']:.3f} tokens")
print(f"Test GT-gloss MAE: {test_gt_metric['mae_tokens']:.3f} tokens")
print(f"Test predicted-gloss MAE: {test_pred_metric['mae_tokens']:.3f} tokens")
print(f"Test predicted-gloss Spearman: {test_pred_metric['spearman']:.6f}")
print("-" * 60); print("ORIGINAL FREE-RUN")
print(f"Oracle duration MAE: {free_oracle['duration_mae_frames']:.3f} frames")
print(f"End-to-end duration MAE: {free_end['duration_mae_frames']:.3f} frames")
print(f"Oracle signed bias: {free_oracle['duration_mean_signed_error']:+.3f} frames")
print(f"End-to-end signed bias: {free_end['duration_mean_signed_error']:+.3f} frames")
print("-" * 60); print("ORACLE-LENGTH DIAGNOSTIC")
print(f"Duration MAE: {oracle_exact['duration_mae_frames']:.3f}")
print(f"Normalized feature MAE: {oracle_exact['normalized_feature_mae']:.6f}")
print(f"Velocity MAE: {oracle_exact['velocity_mae']:.6f}")
print(f"Collapse rate: {oracle_exact['collapse_rate']:.6f}")
print(f"Repetition: {oracle_exact['repetition_score']:.6f}")
print("-" * 60); print("PREDICTED-LENGTH + GT GLOSS")
print(f"Duration MAE: {pred_oracle['duration_mae_frames']:.3f}")
print(f"Normalized feature MAE: {pred_oracle['normalized_feature_mae']:.6f}")
print(f"Collapse rate: {pred_oracle['collapse_rate']:.6f}")
print("-" * 60); print("PREDICTED-LENGTH END-TO-END")
print(f"Duration MAE: {pred_end['duration_mae_frames']:.3f}")
print(f"Signed duration bias: {pred_end['duration_mean_signed_error']:+.3f}")
print(f"Normalized feature MAE: {pred_end['normalized_feature_mae']:.6f}")
print(f"Velocity MAE: {pred_end['velocity_mae']:.6f}")
print(f"Collapse rate: {pred_end['collapse_rate']:.6f}")
print(f"Repetition: {pred_end['repetition_score']:.6f}")
print("-" * 60); print("CAUSAL FINDING")
print(f"Length-only improvement: {duration_fraction_removed:.2%} of oracle duration MAE removed")
print(f"Remaining collapse after oracle length: {oracle_exact['collapse_rate']:.2%}")
print(f"Dominant remaining bottleneck: {dominant}")
print(f"Recommended Phase 7: {recommendation}")
print("READY FOR PHASE 7:", "YES" if phase6_complete else "NO")
print("=" * 60)

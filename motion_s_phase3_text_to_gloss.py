# %% [markdown]
# # Motion-S — Phase 3: English Text to Sign Gloss
#
# This standalone Kaggle notebook consumes the immutable Phase-1 CSV splits and trains a
# real sequence-to-sequence model. It never recreates the split, never retrieves a training
# gloss, and contains no motion generation. Internet must be disabled: attach a complete
# Hugging Face T5-small model/tokenizer dataset and set `model_path` if auto-discovery is
# ambiguous.

# %% [markdown]
# ## 3.1 Reproducible environment and experiment contract
#
# **Expected output:** package versions, GPU, all hyperparameters, and deterministic-mode
# status. The output folder is `/kaggle/working/motion_s_phase3`.

# %%
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from torch.utils.data import DataLoader, Dataset


EXPECTED_PHASE1_FINGERPRINT = "e2acdf4d6dce91775239ad50be3866dbc6f5da94d940204b62456cdb77fc4b0d"


@dataclass(frozen=True)
class Phase3Config:
    seed: int = 42
    kaggle_input: str = "/kaggle/input"
    phase1_root: Optional[str] = None
    output_dir: str = "/kaggle/working/motion_s_phase3"

    model_name: str = "t5-small"
    model_path: Optional[str] = None
    prefix: str = "text to gloss: "
    length_coverage: float = 0.995
    input_length_cap: int = 256
    target_length_cap: int = 128

    debug_samples: int = 32
    debug_epochs: int = 30
    debug_batch_size: int = 8
    debug_learning_rate: float = 5e-4
    debug_min_loss_reduction: float = 0.50
    debug_min_token_f1: float = 0.70

    epochs: int = 12
    train_batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.08
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    num_workers: int = 2
    beam_width: int = 4


CFG3 = Phase3Config()
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

random.seed(CFG3.seed)
np.random.seed(CFG3.seed)
torch.manual_seed(CFG3.seed)
torch.cuda.manual_seed_all(CFG3.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT = Path(CFG3.output_dir)
DIAGNOSTICS = OUTPUT / "diagnostics"
FIGURES = OUTPUT / "figures"
RESULTS = OUTPUT / "results"
BEST_MODEL_DIR = OUTPUT / "best_model"
TOKENIZER_DIR = OUTPUT / "tokenizer"
for directory in (OUTPUT, DIAGNOSTICS, FIGURES, RESULTS, BEST_MODEL_DIR, TOKENIZER_DIR):
    directory.mkdir(parents=True, exist_ok=True)

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: no CUDA GPU. The notebook is correct but full training will be slow.")
display(pd.Series(asdict(CFG3), name="value").to_frame())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


# %% [markdown]
# ## 3.2 Load and lock the Phase-1 split
#
# The package is discovered by its complete artifact contract. Its manifest fingerprint must
# match the fixed Phase-1 fingerprint below; mismatch is a hard error, not a warning.

# %%
def is_phase1_root(path: Path) -> bool:
    required = (
        path / "dataset_config.json",
        path / "split_report.json",
        path / "sample_manifest.csv",
        path / "splits" / "train_split.csv",
        path / "splits" / "val_split.csv",
        path / "splits" / "test_split.csv",
    )
    return all(item.is_file() for item in required)


def locate_phase1_root() -> Path:
    if CFG3.phase1_root:
        root = Path(CFG3.phase1_root)
        if not is_phase1_root(root):
            raise FileNotFoundError(f"Incomplete Phase-1 package: {root}")
        return root.resolve()
    candidates: list[Path] = []
    inherited = Path("/kaggle/working/motion_s_phase1")
    if is_phase1_root(inherited):
        candidates.append(inherited.resolve())
    input_root = Path(CFG3.kaggle_input)
    if input_root.exists():
        candidates.extend(
            path.parent.resolve()
            for path in input_root.rglob("dataset_config.json")
            if is_phase1_root(path.parent)
        )
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one complete Phase-1 package, found "
            f"{[str(x) for x in candidates]}. Set CFG3.phase1_root explicitly."
        )
    return candidates[0]


PHASE1_ROOT = locate_phase1_root()
with (PHASE1_ROOT / "dataset_config.json").open(encoding="utf-8") as handle:
    dataset_config = json.load(handle)
with (PHASE1_ROOT / "split_report.json").open(encoding="utf-8") as handle:
    split_report = json.load(handle)

ID_COL = dataset_config["id_column"]
SENTENCE_COL = dataset_config["sentence_column"]
GLOSS_COL = dataset_config["gloss_column"]

train_df = pd.read_csv(PHASE1_ROOT / "splits/train_split.csv", dtype={ID_COL: str}, low_memory=False)
val_df = pd.read_csv(PHASE1_ROOT / "splits/val_split.csv", dtype={ID_COL: str}, low_memory=False)
test_df = pd.read_csv(PHASE1_ROOT / "splits/test_split.csv", dtype={ID_COL: str}, low_memory=False)
manifest_df = pd.read_csv(PHASE1_ROOT / "sample_manifest.csv", dtype={"sample_id": str})

for split_name, frame in (("train", train_df), ("validation", val_df), ("test", test_df)):
    missing = [c for c in (ID_COL, SENTENCE_COL, GLOSS_COL) if c not in frame.columns]
    if missing:
        raise RuntimeError(f"{split_name} split is missing {missing}")
    if frame[ID_COL].duplicated().any():
        raise RuntimeError(f"Duplicate IDs inside {split_name}")

train_ids, val_ids, test_ids = map(set, (train_df[ID_COL], val_df[ID_COL], test_df[ID_COL]))
assert train_ids.isdisjoint(val_ids) and train_ids.isdisjoint(test_ids) and val_ids.isdisjoint(test_ids)
assert len(train_df) + len(val_df) + len(test_df) == len(manifest_df)

fingerprint_payload = "\n".join(
    f"{row.sample_id},{row.split}"
    for row in manifest_df[["sample_id", "split"]].sort_values("sample_id").itertuples(index=False)
)
OBSERVED_FINGERPRINT = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()
if OBSERVED_FINGERPRINT != EXPECTED_PHASE1_FINGERPRINT:
    raise RuntimeError(
        "Wrong Phase-1 split. Expected " + EXPECTED_PHASE1_FINGERPRINT +
        ", observed " + OBSERVED_FINGERPRINT
    )
if split_report.get("split_fingerprint_sha256") != OBSERVED_FINGERPRINT:
    raise RuntimeError("split_report.json and sample_manifest.csv disagree")
if split_report.get("split_verified") is not True:
    raise RuntimeError("Phase-1 split_report does not mark this split verified")

print("PHASE 3 DATA LOADING")
print("Phase-1 package:", PHASE1_ROOT)
print("Train samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Test samples:", len(test_df))
print("ID column:", ID_COL)
print("Sentence column:", SENTENCE_COL)
print("Gloss column:", GLOSS_COL)
print("Split ID intersections:", {
    "train_validation": len(train_ids & val_ids), "train_test": len(train_ids & test_ids),
    "validation_test": len(val_ids & test_ids),
})
print("Training count expected by Phase 2 / observed:", 9970, "/", len(train_df))
print("Locked fingerprint:", OBSERVED_FINGERPRINT)


# %% [markdown]
# ## 3.3 Gloss-label audit and explicit cleaning policy
#
# `original_gloss` is immutable. `model_gloss` only normalizes Unicode whitespace; punctuation
# and the terminal `//` convention are preserved for learning. Thresholds are fitted on the
# training split using robust distribution statistics, then applied unchanged to all splits.
# Rows with empty/non-string labels or control characters are ineligible for training. Merely
# suspicious labels remain eligible and are surfaced instead of silently discarded.

# %%
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(value: Any) -> str:
    if pd.isna(value):
        return ""
    return WHITESPACE_RE.sub(" ", str(value)).strip()


def max_consecutive_token_repeat(tokens: list[str]) -> int:
    if not tokens:
        return 0
    best = run = 1
    for previous, current in zip(tokens, tokens[1:]):
        run = run + 1 if current == previous else 1
        best = max(best, run)
    return best


def max_consecutive_pattern_repeat(tokens: list[str], max_pattern: int = 4) -> int:
    best = 1 if tokens else 0
    for width in range(1, min(max_pattern, len(tokens) // 2) + 1):
        for start in range(0, len(tokens) - 2 * width + 1):
            pattern = tokens[start:start + width]
            repeats = 1
            cursor = start + width
            while cursor + width <= len(tokens) and tokens[cursor:cursor + width] == pattern:
                repeats += 1
                cursor += width
            best = max(best, repeats)
    return best


def base_audit(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    records = []
    for row in frame[[ID_COL, SENTENCE_COL, GLOSS_COL]].itertuples(index=False, name=None):
        sample_id, sentence, raw_gloss = row
        missing = pd.isna(raw_gloss) or not isinstance(raw_gloss, str)
        original = "" if pd.isna(raw_gloss) else str(raw_gloss)
        model = normalize_whitespace(original)
        tokens = model.split()
        punctuation = "".join(ch for ch in model if not ch.isalnum() and not ch.isspace() and ch != "_")
        unusual = "".join(sorted(set(re.findall(r"[^\w\s/?.,'!:\-]", model, flags=re.UNICODE))))
        records.append({
            "sample_id": str(sample_id), "split": split_name,
            "sentence": normalize_whitespace(sentence), "original_gloss": original,
            "model_gloss": model, "missing_or_nonstring": bool(missing),
            "empty_after_whitespace_normalization": not bool(model),
            "has_control_character": bool(CONTROL_RE.search(original)),
            "has_terminal_double_slash": bool(re.search(r"//\s*$", model)),
            "has_question_mark": "?" in model, "punctuation_signature": punctuation,
            "unusual_characters": unusual,
            "token_count": len(tokens), "unique_token_count": len(set(tokens)),
            "repeat_fraction": 0.0 if not tokens else 1.0 - len(set(tokens)) / len(tokens),
            "max_consecutive_token_repeat": max_consecutive_token_repeat(tokens),
            "max_consecutive_pattern_repeat": max_consecutive_pattern_repeat(tokens),
        })
    return pd.DataFrame(records)


audit_parts = [
    base_audit(train_df, "train"), base_audit(val_df, "validation"), base_audit(test_df, "test")
]
gloss_audit = pd.concat(audit_parts, ignore_index=True)
train_audit = gloss_audit[gloss_audit["split"] == "train"]


def robust_upper_threshold(values: pd.Series, quantile: float = 0.995) -> int:
    array = values.astype(float).to_numpy()
    q = float(np.quantile(array, quantile, method="higher"))
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    robust = median + 6.0 * 1.4826 * mad
    return max(1, int(math.ceil(max(q, robust))))


AUDIT_THRESHOLDS = {
    "method": "max(training P99.5, training median + 6 scaled MAD)",
    "long_gloss_tokens": robust_upper_threshold(train_audit["token_count"]),
    "consecutive_token_repeats": robust_upper_threshold(train_audit["max_consecutive_token_repeat"]),
    "consecutive_pattern_repeats": robust_upper_threshold(train_audit["max_consecutive_pattern_repeat"]),
}


def audit_flags(row: pd.Series) -> list[str]:
    flags = []
    if row["missing_or_nonstring"]:
        flags.append("missing_or_nonstring")
    if row["empty_after_whitespace_normalization"]:
        flags.append("empty_gloss")
    if row["has_control_character"]:
        flags.append("control_character")
    if row["unusual_characters"]:
        flags.append("unusual_character")
    if row["token_count"] > AUDIT_THRESHOLDS["long_gloss_tokens"]:
        flags.append("extreme_length")
    if row["max_consecutive_token_repeat"] > AUDIT_THRESHOLDS["consecutive_token_repeats"]:
        flags.append("repeated_token_run")
    if row["max_consecutive_pattern_repeat"] > AUDIT_THRESHOLDS["consecutive_pattern_repeats"]:
        flags.append("repeated_short_pattern")
    return flags


flag_lists = gloss_audit.apply(audit_flags, axis=1)
gloss_audit["audit_flags"] = flag_lists.map(lambda x: "|".join(x))
gloss_audit["is_suspicious"] = flag_lists.map(bool)
gloss_audit["training_eligible"] = ~(
    gloss_audit["missing_or_nonstring"]
    | gloss_audit["empty_after_whitespace_normalization"]
    | gloss_audit["has_control_character"]
)
gloss_audit["gloss"] = gloss_audit["original_gloss"]
gloss_audit["max_consecutive_repeat"] = gloss_audit["max_consecutive_token_repeat"]
gloss_audit["repeated_token_ratio"] = gloss_audit["repeat_fraction"]
gloss_audit["flags"] = gloss_audit["audit_flags"]
audit_export_columns = [
    "sample_id", "split", "sentence", "gloss", "original_gloss", "model_gloss",
    "token_count", "unique_token_count", "max_consecutive_repeat",
    "max_consecutive_pattern_repeat", "repeated_token_ratio",
    "punctuation_signature", "has_terminal_double_slash", "has_question_mark",
    "unusual_characters", "flags", "training_eligible",
]
gloss_audit[audit_export_columns].to_csv(DIAGNOSTICS / "gloss_audit.csv", index=False)
gloss_audit[audit_export_columns].to_csv(DIAGNOSTICS / "phase3_gloss_audit.csv", index=False)
write_json(DIAGNOSTICS / "gloss_audit_thresholds.json", AUDIT_THRESHOLDS)


def attach_audit(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    columns = ["sample_id", "original_gloss", "model_gloss", "audit_flags", "is_suspicious", "training_eligible"]
    attached = frame.copy()
    attached[ID_COL] = attached[ID_COL].astype(str)
    return attached.merge(
        gloss_audit.loc[gloss_audit["split"] == split_name, columns],
        left_on=ID_COL, right_on="sample_id", validate="one_to_one",
    ).drop(columns="sample_id")


train_df = attach_audit(train_df, "train")
val_df = attach_audit(val_df, "validation")
test_df = attach_audit(test_df, "test")

audit_summary = gloss_audit.groupby("split").agg(
    rows=("sample_id", "size"), suspicious=("is_suspicious", "sum"),
    unusable=("training_eligible", lambda x: int((~x).sum())),
    terminal_double_slash=("has_terminal_double_slash", "sum"),
    question_mark=("has_question_mark", "sum"),
    median_tokens=("token_count", "median"), max_tokens=("token_count", "max"),
)
gloss_length_statistics = gloss_audit.groupby("split")["token_count"].agg(
    minimum="min", median="median", mean="mean",
    p95=lambda x: x.quantile(.95), p99=lambda x: x.quantile(.99), maximum="max",
)
print("Audit thresholds:", AUDIT_THRESHOLDS)
display(audit_summary)
print("Gloss length statistics:")
display(gloss_length_statistics)
print("Most suspicious labels (nothing silently removed):")
display(
    gloss_audit[gloss_audit["is_suspicious"]]
    .sort_values(["token_count", "max_consecutive_pattern_repeat"], ascending=False)
    .head(30)[["sample_id", "split", "original_gloss", "audit_flags", "token_count",
               "max_consecutive_token_repeat", "max_consecutive_pattern_repeat"]]
)

# %% [markdown]
# ## 3.4 Vocabulary, rarity, and held-out OOV analysis
#
# Lexical analysis removes only a terminal `//` annotation marker. Training targets retain it.

# %%
TERMINAL_MARKER_RE = re.compile(r"\s*//\s*$")
# Deliberately identical to Phase 2: uppercase regex tokens with punctuation excluded.
LEXICAL_TOKEN_RE = re.compile(r"[\w']+", flags=re.UNICODE)


def raw_normalized(value: Any) -> str:
    return normalize_whitespace(value)


def lexical_tokens(value: Any) -> list[str]:
    text = TERMINAL_MARKER_RE.sub("", raw_normalized(value))
    return [token.upper() for token in LEXICAL_TOKEN_RE.findall(text)]


eligible_train = train_df[train_df["training_eligible"]].reset_index(drop=True)
TRAIN_GLOSS_COUNTS = Counter(
    token for gloss in train_df.loc[train_df["training_eligible"], "model_gloss"] for token in lexical_tokens(gloss)
)
TRAIN_GLOSS_VOCAB = set(TRAIN_GLOSS_COUNTS)


def oov_report(frame: pd.DataFrame, split_name: str) -> dict[str, Any]:
    tokens = [token for gloss in frame["model_gloss"] for token in lexical_tokens(gloss)]
    counts = Counter(tokens)
    oov = {token: count for token, count in counts.items() if token not in TRAIN_GLOSS_VOCAB}
    return {
        "split": split_name, "tokens": len(tokens), "types": len(counts),
        "oov_token_count": int(sum(oov.values())), "oov_type_count": len(oov),
        "oov_token_rate": sum(oov.values()) / max(len(tokens), 1),
        "oov_types": dict(sorted(oov.items(), key=lambda x: (-x[1], x[0]))[:100]),
    }


VOCABULARY_REPORT = {
    "training_token_count": int(sum(TRAIN_GLOSS_COUNTS.values())),
    "training_type_count": len(TRAIN_GLOSS_COUNTS),
    "training_unique_gloss_sequences": int(eligible_train["model_gloss"].nunique()),
    "training_gloss_length": {
        "mean": float(eligible_train["model_gloss"].map(lambda x: len(lexical_tokens(x))).mean()),
        "median": float(eligible_train["model_gloss"].map(lambda x: len(lexical_tokens(x))).median()),
        "p95": float(eligible_train["model_gloss"].map(lambda x: len(lexical_tokens(x))).quantile(.95)),
        "p99": float(eligible_train["model_gloss"].map(lambda x: len(lexical_tokens(x))).quantile(.99)),
        "maximum": int(eligible_train["model_gloss"].map(lambda x: len(lexical_tokens(x))).max()),
    },
    "most_frequent_30_tokens": dict(TRAIN_GLOSS_COUNTS.most_common(30)),
    "hapax_types": sum(count == 1 for count in TRAIN_GLOSS_COUNTS.values()),
    "types_frequency_lt_5": sum(count < 5 for count in TRAIN_GLOSS_COUNTS.values()),
    "types_frequency_lt_10": sum(count < 10 for count in TRAIN_GLOSS_COUNTS.values()),
    "validation": oov_report(val_df, "validation"),
    "test": oov_report(test_df, "test"),
}
write_json(DIAGNOSTICS / "vocabulary_report.json", VOCABULARY_REPORT)
display(pd.json_normalize(VOCABULARY_REPORT, sep="."))
print("Most frequent 30 training gloss tokens:")
display(pd.DataFrame(TRAIN_GLOSS_COUNTS.most_common(30), columns=["token", "count"]))

# %% [markdown]
# ## 3.5 Offline T5-small discovery and tokenizer length audit
#
# Attach a Kaggle dataset containing `config.json`, tokenizer files, and model weights.
# Auto-discovery prefers a directory whose config declares `model_type: t5` and whose name
# contains `t5-small`. `local_files_only=True` and offline environment variables prevent a
# silent download.

# %%
try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
except ImportError as exc:
    raise ImportError("This notebook requires transformers and nltk from the Kaggle image") from exc


def is_complete_hf_model(path: Path) -> bool:
    has_weights = any((path / name).is_file() for name in (
        "model.safetensors", "pytorch_model.bin"
    )) or bool(list(path.glob("model-*.safetensors"))) or bool(list(path.glob("pytorch_model-*.bin")))
    has_tokenizer = any((path / name).is_file() for name in (
        "tokenizer.json", "spiece.model"
    ))
    return (path / "config.json").is_file() and has_weights and has_tokenizer


def model_type_is_t5(path: Path) -> bool:
    try:
        with (path / "config.json").open(encoding="utf-8") as handle:
            return json.load(handle).get("model_type") == "t5"
    except (OSError, json.JSONDecodeError):
        return False


def locate_offline_model() -> Path:
    if CFG3.model_path:
        path = Path(CFG3.model_path)
        if not is_complete_hf_model(path) or not model_type_is_t5(path):
            raise FileNotFoundError(f"Configured offline T5 directory is incomplete: {path}")
        return path.resolve()
    roots = [Path(CFG3.kaggle_input), Path.home() / ".cache" / "huggingface" / "hub"]
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        for config_file in root.rglob("config.json"):
            parent = config_file.parent
            if is_complete_hf_model(parent) and model_type_is_t5(parent):
                candidates.append(parent.resolve())
    candidates = list(dict.fromkeys(candidates))
    preferred = [p for p in candidates if "t5-small" in str(p).lower() or "t5_small" in str(p).lower()]
    chosen_pool = preferred or candidates
    if len(chosen_pool) != 1:
        raise RuntimeError(
            "Could not select exactly one offline T5-small directory. Candidates: "
            f"{[str(x) for x in candidates]}. Attach the model and set CFG3.model_path."
        )
    return chosen_pool[0]


MODEL_SOURCE = locate_offline_model()
tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE, local_files_only=True, use_fast=True)
probe_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_SOURCE, local_files_only=True)
parameter_count = sum(parameter.numel() for parameter in probe_model.parameters())
model_type = probe_model.config.model_type
model_config_name = str(getattr(probe_model.config, "_name_or_path", MODEL_SOURCE))
is_t5_small_architecture = bool(
    getattr(probe_model.config, "d_model", None) == 512
    and getattr(probe_model.config, "num_layers", None) == 6
    and getattr(probe_model.config, "num_decoder_layers", 6) == 6
)
MODEL_DESCRIPTION = "T5-small" if is_t5_small_architecture else f"offline T5 equivalent ({model_config_name})"
del probe_model
gc.collect()

if model_type != "t5":
    raise RuntimeError(f"Expected a T5 checkpoint, got model_type={model_type}")
print("Offline model source:", MODEL_SOURCE)
print("Tokenizer:", tokenizer.__class__.__name__)
print("Model class: AutoModelForSeq2SeqLM ->", model_type)
print("Resolved model:", MODEL_DESCRIPTION)
if not is_t5_small_architecture:
    print("T5-small was not attached; using the only complete offline T5 encoder-decoder checkpoint.")
print("Parameters:", f"{parameter_count:,}")


def encoded_lengths(texts: list[str], batch_size: int = 512) -> np.ndarray:
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start:start + batch_size], add_special_tokens=True,
            truncation=False, padding=False, return_length=True,
        )
        if "length" in encoded:
            lengths.extend(int(x) for x in encoded["length"])
        else:
            lengths.extend(len(x) for x in encoded["input_ids"])
    return np.asarray(lengths, dtype=np.int32)


train_input_texts = (CFG3.prefix + eligible_train[SENTENCE_COL].map(normalize_whitespace)).tolist()
train_target_texts = eligible_train["model_gloss"].tolist()
input_lengths = encoded_lengths(train_input_texts)
target_lengths = encoded_lengths(train_target_texts)


def select_length(lengths: np.ndarray, cap: int, label: str) -> tuple[int, float]:
    selected = min(int(np.quantile(lengths, CFG3.length_coverage, method="higher")), cap)
    truncation_rate = float(np.mean(lengths > selected))
    if truncation_rate >= 0.01:
        raise RuntimeError(
            f"{label} cap={cap} truncates {truncation_rate:.2%}; increase the cap before training"
        )
    return selected, truncation_rate


MAX_INPUT_LENGTH, INPUT_TRUNCATION_RATE = select_length(input_lengths, CFG3.input_length_cap, "input")
MAX_TARGET_LENGTH, TARGET_TRUNCATION_RATE = select_length(target_lengths, CFG3.target_length_cap, "target")
LENGTH_REPORT = {
    "coverage_quantile": CFG3.length_coverage,
    "input": {"p50": float(np.quantile(input_lengths, .50)), "p90": float(np.quantile(input_lengths, .90)),
              "p99": float(np.quantile(input_lengths, .99)), "max": int(input_lengths.max()),
              "selected_max_length": MAX_INPUT_LENGTH, "truncation_rate": INPUT_TRUNCATION_RATE},
    "target": {"p50": float(np.quantile(target_lengths, .50)), "p90": float(np.quantile(target_lengths, .90)),
               "p99": float(np.quantile(target_lengths, .99)), "max": int(target_lengths.max()),
               "selected_max_length": MAX_TARGET_LENGTH, "truncation_rate": TARGET_TRUNCATION_RATE},
}
write_json(DIAGNOSTICS / "tokenizer_length_report.json", LENGTH_REPORT)
print("Length-selection coverage quantile:", LENGTH_REPORT["coverage_quantile"])
display(pd.DataFrame({"input": LENGTH_REPORT["input"], "target": LENGTH_REPORT["target"]}))
print("Selected lengths:", MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
print("Training truncation rates:", INPUT_TRUNCATION_RATE, TARGET_TRUNCATION_RATE)
tokenizer_examples = []
for _, row in eligible_train.sample(n=min(3, len(eligible_train)), random_state=CFG3.seed).iterrows():
    sentence = normalize_whitespace(row[SENTENCE_COL])
    gloss = row["model_gloss"]
    tokenizer_examples.append({
        "raw_sentence": sentence,
        "input_ids": tokenizer(CFG3.prefix + sentence, add_special_tokens=True)["input_ids"],
        "raw_gloss": gloss,
        "target_ids": tokenizer(text_target=gloss, add_special_tokens=True)["input_ids"],
    })
print("Tokenizer examples (untruncated):")
display(pd.DataFrame(tokenizer_examples))

# %% [markdown]
# ## 3.6 Dataset, label masking, and metric definitions
#
# The fixed prefix is added to every source. Target padding is replaced by `-100`, which
# makes the T5 loss ignore it. Metrics are reported twice: raw-normalized (punctuation kept)
# and lexical (terminal `//` removed). Token precision/recall/F1 use multiset overlap, exactly
# matching Phase 2's F1 definition.

# %%
class TextGlossDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True).copy()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        source = CFG3.prefix + normalize_whitespace(row[SENTENCE_COL])
        target = row["model_gloss"]
        source_encoded = tokenizer(
            source, max_length=MAX_INPUT_LENGTH, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        target_encoded = tokenizer(
            text_target=target, max_length=MAX_TARGET_LENGTH, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        labels = target_encoded["input_ids"].squeeze(0)
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": source_encoded["input_ids"].squeeze(0),
            "attention_mask": source_encoded["attention_mask"].squeeze(0),
            "labels": labels,
            "sample_id": str(row[ID_COL]),
            "sentence": normalize_whitespace(row[SENTENCE_COL]),
            "reference_gloss": target,
            "original_reference_gloss": str(row["original_gloss"]),
            "audit_flags": str(row["audit_flags"]),
        }


dataset_probe = TextGlossDataset(eligible_train.iloc[:2])
probe = dataset_probe[0]
assert probe["input_ids"].shape == (MAX_INPUT_LENGTH,)
assert probe["attention_mask"].shape == (MAX_INPUT_LENGTH,)
assert probe["labels"].shape == (MAX_TARGET_LENGTH,)
assert tokenizer.pad_token_id not in probe["labels"].tolist(), "Label padding must be -100"
probe_target_length = min(
    len(tokenizer(text_target=eligible_train.iloc[0]["model_gloss"], add_special_tokens=True)["input_ids"]),
    MAX_TARGET_LENGTH,
)
if probe_target_length < MAX_TARGET_LENGTH:
    assert (probe["labels"][probe_target_length:] == -100).all(), "All target padding must be -100"
print("Dataset tensor shapes:", {k: tuple(probe[k].shape) for k in ("input_ids", "attention_mask", "labels")})
print("Masked label positions:", int((probe["labels"] == -100).sum()))


BLEU_SMOOTHER = SmoothingFunction().method1


def raw_tokens(value: Any) -> list[str]:
    return raw_normalized(value).split()


def multiset_prf(reference: list[str], hypothesis: list[str]) -> tuple[float, float, float]:
    ref_counts, hyp_counts = Counter(reference), Counter(hypothesis)
    overlap = sum((ref_counts & hyp_counts).values())
    precision = overlap / max(sum(hyp_counts.values()), 1)
    recall = overlap / max(sum(ref_counts.values()), 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, reference_token in enumerate(reference, start=1):
        current = [i]
        for j, hypothesis_token in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + int(reference_token != hypothesis_token),
            ))
        previous = current
    return previous[-1]


def smoothed_bleu(reference: list[str], hypothesis: list[str]) -> float:
    if not reference or not hypothesis:
        return 0.0
    return float(sentence_bleu(
        [reference], hypothesis, weights=(.25, .25, .25, .25),
        smoothing_function=BLEU_SMOOTHER,
    ))


def per_example_metrics(reference: Any, hypothesis: Any, view: str = "lexical") -> dict[str, float]:
    tokenizer_function = raw_tokens if view == "raw" else lexical_tokens
    ref, hyp = tokenizer_function(reference), tokenizer_function(hypothesis)
    precision, recall, f1 = multiset_prf(ref, hyp)
    edit = edit_distance(ref, hyp)
    return {
        "bleu": smoothed_bleu(ref, hyp), "token_precision": precision,
        "token_recall": recall, "token_f1": f1,
        "exact_match": float(ref == hyp), "edit_distance": float(edit),
        "normalized_edit_distance": edit / max(len(ref), len(hyp), 1),
    }


def evaluate_text_pairs(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    if len(references) != len(hypotheses) or not references:
        raise ValueError("Metric inputs must be non-empty and equal length")
    output: dict[str, float] = {"n": float(len(references))}
    for view in ("raw", "lexical"):
        rows = [per_example_metrics(ref, hyp, view) for ref, hyp in zip(references, hypotheses)]
        for metric in rows[0]:
            output[f"{view}_{metric}"] = float(np.mean([row[metric] for row in rows]))
    return output


metric_unit = evaluate_text_pairs(["ME GO //"], ["ME GO //"])
assert metric_unit["raw_exact_match"] == 1 and metric_unit["lexical_token_f1"] == 1
assert lexical_tokens("ME GO //") == ["ME", "GO"]
print("Metric unit check passed:", metric_unit)


# %% [markdown]
# ## 3.7 Training, validation, and generation functions
#
# Automatic mixed precision is enabled only on CUDA. Loss is divided by the accumulation
# factor; gradients are unscaled before clipping. Each epoch generates the complete validation
# split greedily. Best-model ordering is lexical token F1 first, BLEU second.

# %%
def make_loader(frame: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(CFG3.seed)
    return DataLoader(
        TextGlossDataset(frame), batch_size=batch_size, shuffle=shuffle,
        num_workers=CFG3.num_workers, pin_memory=DEVICE.type == "cuda",
        generator=generator if shuffle else None,
    )


def tensor_batch(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {key: batch[key].to(DEVICE, non_blocking=True) for key in ("input_ids", "attention_mask", "labels")}


def autocast_context():
    return torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda")


def make_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")


@torch.no_grad()
def mean_loss(model: torch.nn.Module, loader: DataLoader) -> float:
    model.eval()
    total, examples = 0.0, 0
    for batch in loader:
        model_inputs = tensor_batch(batch)
        with autocast_context():
            loss = model(**model_inputs).loss
        batch_size = model_inputs["input_ids"].shape[0]
        total += float(loss.item()) * batch_size
        examples += batch_size
    return total / max(examples, 1)


@torch.no_grad()
def generate_predictions(model: torch.nn.Module, loader: DataLoader, num_beams: int) -> pd.DataFrame:
    model.eval()
    records: list[dict[str, Any]] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        generation_args: dict[str, Any] = {
            "input_ids": input_ids, "attention_mask": attention_mask,
            "max_length": MAX_TARGET_LENGTH, "num_beams": num_beams,
            "do_sample": False,
        }
        if num_beams > 1:
            generation_args["early_stopping"] = True
        generated = model.generate(**generation_args)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for sample_id, sentence, reference, original_reference, flags, prediction in zip(
            batch["sample_id"], batch["sentence"], batch["reference_gloss"],
            batch["original_reference_gloss"], batch["audit_flags"], decoded,
        ):
            prediction_normalized = raw_normalized(prediction)
            record = {
                "sample_id": sample_id, "sentence": sentence,
                "ground_truth_gloss": original_reference,
                "reference_gloss": reference, "raw_reference": original_reference,
                "normalized_reference": raw_normalized(reference),
                "raw_prediction": str(prediction), "normalized_prediction": prediction_normalized,
                "predicted_gloss": prediction_normalized,
                "audit_flags": flags,
            }
            for view in ("raw", "lexical"):
                record.update({f"{view}_{key}": value for key, value in per_example_metrics(
                    reference, prediction, view
                ).items()})
            records.append(record)
    return pd.DataFrame(records)


def aggregate_prediction_frame(frame: pd.DataFrame) -> dict[str, float]:
    metrics = {"n": float(len(frame))}
    for view in ("raw", "lexical"):
        for metric in ("bleu", "token_precision", "token_recall", "token_f1", "exact_match",
                       "edit_distance", "normalized_edit_distance"):
            metrics[f"{view}_{metric}"] = float(frame[f"{view}_{metric}"].mean())
    return metrics


def train_epoch(
    model: torch.nn.Module, loader: DataLoader, optimizer, scheduler, scaler,
    accumulation_steps: Optional[int] = None,
) -> dict[str, float]:
    accumulation_steps = accumulation_steps or CFG3.gradient_accumulation_steps
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, examples, last_grad_norm = 0.0, 0, 0.0
    steps = len(loader)
    for step, batch in enumerate(loader, start=1):
        model_inputs = tensor_batch(batch)
        with autocast_context():
            loss = model(**model_inputs).loss
            scaled_loss = loss / accumulation_steps
        scaler.scale(scaled_loss).backward()
        should_step = step % accumulation_steps == 0 or step == steps
        if should_step:
            scaler.unscale_(optimizer)
            last_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CFG3.max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        batch_size = model_inputs["input_ids"].shape[0]
        total_loss += float(loss.item()) * batch_size
        examples += batch_size
    return {"loss": total_loss / max(examples, 1), "grad_norm": last_grad_norm}


def fresh_model() -> torch.nn.Module:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_SOURCE, local_files_only=True)
    model.config.use_cache = False
    return model.to(DEVICE)


def better(candidate: dict[str, float], incumbent: Optional[dict[str, float]]) -> bool:
    if incumbent is None:
        return True
    candidate_pair = (candidate["lexical_token_f1"], candidate["lexical_bleu"])
    incumbent_pair = (incumbent["lexical_token_f1"], incumbent["lexical_bleu"])
    return candidate_pair > incumbent_pair


# %% [markdown]
# ## 3.8 Mandatory 32-example overfit gate
#
# This is deliberately a separate model and optimizer. Full training is unreachable unless
# loss falls by at least 50% and lexical token F1 on the same 32 examples reaches 0.70.
# A failure saves diagnostics and raises an exception so architecture/data bugs cannot hide.

# %%
debug_frame = eligible_train.sample(n=CFG3.debug_samples, random_state=CFG3.seed).reset_index(drop=True)
debug_loader = make_loader(debug_frame, CFG3.debug_batch_size, shuffle=True)
debug_eval_loader = make_loader(debug_frame, CFG3.eval_batch_size, shuffle=False)
debug_model = fresh_model()
debug_optimizer = torch.optim.AdamW(debug_model.parameters(), lr=CFG3.debug_learning_rate)
debug_scaler = make_scaler()
debug_history: list[dict[str, float]] = []
debug_initial_loss = mean_loss(debug_model, debug_eval_loader)

for debug_epoch in range(1, CFG3.debug_epochs + 1):
    started = time.time()
    epoch_stats = train_epoch(
        debug_model, debug_loader, debug_optimizer, None, debug_scaler, accumulation_steps=1
    )
    evaluated_loss = mean_loss(debug_model, debug_eval_loader)
    row = {
        "epoch": debug_epoch, "train_loss": epoch_stats["loss"],
        "evaluation_loss": evaluated_loss, "seconds": time.time() - started,
    }
    if debug_epoch == 1 or debug_epoch % 5 == 0 or debug_epoch == CFG3.debug_epochs:
        checkpoint_predictions = generate_predictions(debug_model, debug_eval_loader, num_beams=1)
        checkpoint_metrics = aggregate_prediction_frame(checkpoint_predictions)
        row["lexical_token_f1"] = checkpoint_metrics["lexical_token_f1"]
        row["lexical_exact_match"] = checkpoint_metrics["lexical_exact_match"]
        print("Debug:", row)
        display(checkpoint_predictions[[
            "sentence", "reference_gloss", "predicted_gloss", "lexical_token_f1", "lexical_exact_match"
        ]].head(3))
    debug_history.append(row)

debug_predictions = generate_predictions(debug_model, debug_eval_loader, num_beams=1)
debug_metrics = aggregate_prediction_frame(debug_predictions)
debug_final_loss = mean_loss(debug_model, debug_eval_loader)
debug_loss_reduction = 1.0 - debug_final_loss / max(debug_initial_loss, 1e-12)
DEBUG_GATE = {
    "sample_count": len(debug_frame), "initial_loss": debug_initial_loss,
    "final_loss": debug_final_loss, "loss_reduction": debug_loss_reduction,
    "lexical_token_f1": debug_metrics["lexical_token_f1"],
    "lexical_exact_match": debug_metrics["lexical_exact_match"],
    "minimum_loss_reduction": CFG3.debug_min_loss_reduction,
    "minimum_token_f1": CFG3.debug_min_token_f1,
}
DEBUG_GATE["passed"] = bool(
    DEBUG_GATE["loss_reduction"] >= CFG3.debug_min_loss_reduction
    and DEBUG_GATE["lexical_token_f1"] >= CFG3.debug_min_token_f1
)
pd.DataFrame(debug_history).to_csv(DIAGNOSTICS / "debug_overfit_history.csv", index=False)
debug_predictions.to_csv(DIAGNOSTICS / "debug_overfit_predictions.csv", index=False)
write_json(DIAGNOSTICS / "debug_overfit_gate.json", DEBUG_GATE)
display(pd.Series(DEBUG_GATE, name="debug_gate").to_frame())

del debug_model, debug_optimizer, debug_scaler
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
if not DEBUG_GATE["passed"]:
    raise RuntimeError(
        "MANDATORY OVERFIT GATE FAILED. Full training was not started. Inspect "
        f"{DIAGNOSTICS / 'debug_overfit_predictions.csv'}"
    )
print("OVERFIT GATE PASSED — full training is now permitted.")

# %% [markdown]
# ## 3.9 Full training with best-checkpoint selection and early stopping
#
# Training uses only eligible training rows. Validation uses every original validation row.
# Each epoch records loss, F1, BLEU, exact match, edit distance, learning rate, gradient norm,
# runtime, and peak GPU memory. The best checkpoint and tokenizer are self-contained.

# %%
train_loader = make_loader(eligible_train, CFG3.train_batch_size, shuffle=True)
val_loader = make_loader(val_df, CFG3.eval_batch_size, shuffle=False)

model = fresh_model()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=CFG3.learning_rate, weight_decay=CFG3.weight_decay
)
updates_per_epoch = math.ceil(len(train_loader) / CFG3.gradient_accumulation_steps)
total_updates = updates_per_epoch * CFG3.epochs
warmup_updates = int(round(total_updates * CFG3.warmup_ratio))
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_updates, num_training_steps=total_updates
)
scaler = make_scaler()

history: list[dict[str, Any]] = []
best_metrics: Optional[dict[str, float]] = None
best_epoch: Optional[int] = None
epochs_without_improvement = 0

for epoch in range(1, CFG3.epochs + 1):
    started = time.time()
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train_stats = train_epoch(model, train_loader, optimizer, scheduler, scaler)
    validation_loss = mean_loss(model, val_loader)
    validation_predictions_epoch = generate_predictions(model, val_loader, num_beams=1)
    validation_metrics_epoch = aggregate_prediction_frame(validation_predictions_epoch)
    row = {
        "epoch": epoch, "train_loss": train_stats["loss"], "validation_loss": validation_loss,
        **{f"validation_{key}": value for key, value in validation_metrics_epoch.items()},
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "grad_norm": train_stats["grad_norm"], "seconds": time.time() - started,
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated() / 1024 ** 3 if DEVICE.type == "cuda" else 0.0
        ),
    }
    history.append(row)
    print(
        f"Epoch {epoch:02d}: train={row['train_loss']:.4f} val={validation_loss:.4f} "
        f"F1={validation_metrics_epoch['lexical_token_f1']:.4f} "
        f"BLEU={validation_metrics_epoch['lexical_bleu']:.4f} "
        f"exact={validation_metrics_epoch['lexical_exact_match']:.4f} "
        f"time={row['seconds']:.1f}s"
    )
    if better(validation_metrics_epoch, best_metrics):
        best_metrics = validation_metrics_epoch.copy()
        best_epoch = epoch
        epochs_without_improvement = 0
        model.config.use_cache = True
        model.save_pretrained(BEST_MODEL_DIR, safe_serialization=True)
        tokenizer.save_pretrained(TOKENIZER_DIR)
        model.config.use_cache = False
        validation_predictions_epoch.to_csv(RESULTS / "best_epoch_validation_greedy.csv", index=False)
        write_json(OUTPUT / "best_checkpoint_metrics.json", {
            "epoch": best_epoch, "selection_rule": "lexical_token_f1_then_lexical_bleu",
            "metrics": best_metrics,
        })
        print("  Saved new best checkpoint")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= CFG3.early_stopping_patience:
            print("Early stopping: no validation improvement for", epochs_without_improvement, "epochs")
            break

history_df = pd.DataFrame(history)
history_df.to_csv(OUTPUT / "training_history.csv", index=False)
write_json(OUTPUT / "training_history.json", history)
assert best_epoch is not None and (BEST_MODEL_DIR / "config.json").is_file()

figure, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o", label="train")
axes[0].plot(history_df["epoch"], history_df["validation_loss"], marker="o", label="validation")
axes[0].set(title="Loss", xlabel="epoch", ylabel="cross-entropy"); axes[0].legend()
axes[1].plot(history_df["epoch"], history_df["validation_lexical_token_f1"], marker="o", label="token F1")
axes[1].plot(history_df["epoch"], history_df["validation_lexical_bleu"], marker="o", label="BLEU")
axes[1].set(title="Validation generation", xlabel="epoch", ylabel="score"); axes[1].legend()
figure.tight_layout()
figure.savefig(FIGURES / "training_curves.png", dpi=160, bbox_inches="tight")
plt.show()

loss_figure, loss_axis = plt.subplots(figsize=(7, 4))
loss_axis.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="train")
loss_axis.plot(history_df["epoch"], history_df["validation_loss"], marker="o", label="validation")
loss_axis.set(title="Training and validation loss", xlabel="epoch", ylabel="cross-entropy")
loss_axis.legend(); loss_figure.tight_layout()
loss_figure.savefig(FIGURES / "training_loss.png", dpi=160, bbox_inches="tight")
plt.show()

metrics_figure, metrics_axis = plt.subplots(figsize=(7, 4))
metrics_axis.plot(history_df["epoch"], history_df["validation_lexical_token_f1"], marker="o", label="token F1")
metrics_axis.plot(history_df["epoch"], history_df["validation_lexical_bleu"], marker="o", label="BLEU")
metrics_axis.plot(history_df["epoch"], history_df["validation_lexical_exact_match"], marker="o", label="exact")
metrics_axis.set(title="Validation generation metrics", xlabel="epoch", ylabel="score")
metrics_axis.legend(); metrics_figure.tight_layout()
metrics_figure.savefig(FIGURES / "validation_metrics.png", dpi=160, bbox_inches="tight")
plt.show()

print("Best epoch:", best_epoch)
display(pd.Series(best_metrics, name="best_validation_greedy").to_frame())

del model, optimizer, scheduler, scaler
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()


# %% [markdown]
# ## 3.10 Select greedy or beam-4 on validation; evaluate test once
#
# Both decoding strategies see the same locked best checkpoint and complete validation set.
# Selection again uses lexical F1 then BLEU. Only the selected strategy is used for the final
# test evaluation. Clean-subset figures are secondary; full-split metrics are primary.

# %%
best_model = AutoModelForSeq2SeqLM.from_pretrained(BEST_MODEL_DIR, local_files_only=True).to(DEVICE)
best_model.eval()

val_greedy = generate_predictions(best_model, val_loader, num_beams=1)
val_beam = generate_predictions(best_model, val_loader, num_beams=CFG3.beam_width)
val_greedy_metrics = aggregate_prediction_frame(val_greedy)
val_beam_metrics = aggregate_prediction_frame(val_beam)
DECODING_METHOD = "beam_4" if better(val_beam_metrics, val_greedy_metrics) else "greedy"
SELECTED_BEAMS = CFG3.beam_width if DECODING_METHOD == "beam_4" else 1

decoding_comparison = pd.DataFrame([
    {"method": "greedy", **val_greedy_metrics},
    {"method": "beam_4", **val_beam_metrics},
])
decoding_comparison.to_csv(OUTPUT / "decoding_comparison.csv", index=False)
val_predictions = val_beam if DECODING_METHOD == "beam_4" else val_greedy
val_metrics = val_beam_metrics if DECODING_METHOD == "beam_4" else val_greedy_metrics
val_predictions.to_csv(RESULTS / "validation_predictions.csv", index=False)

test_loader = make_loader(test_df, CFG3.eval_batch_size, shuffle=False)
test_predictions = generate_predictions(best_model, test_loader, num_beams=SELECTED_BEAMS)
test_metrics = aggregate_prediction_frame(test_predictions)
test_predictions.to_csv(RESULTS / "test_predictions.csv", index=False)


def clean_subset_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    clean = predictions[predictions["audit_flags"].fillna("") == ""]
    return aggregate_prediction_frame(clean) if len(clean) else {"n": 0.0}


EVALUATION_METRICS = {
    "selected_decoding": DECODING_METHOD,
    "validation_full": val_metrics,
    "validation_clean": clean_subset_metrics(val_predictions),
    "test_full": test_metrics,
    "test_clean": clean_subset_metrics(test_predictions),
}
write_json(OUTPUT / "evaluation_metrics.json", EVALUATION_METRICS)
print("Selected decoding:", DECODING_METHOD)
display(decoding_comparison)
print("Validation full / clean:")
display(pd.DataFrame([EVALUATION_METRICS["validation_full"], EVALUATION_METRICS["validation_clean"]],
                     index=["full", "clean"]))
print("Test full / clean:")
display(pd.DataFrame([EVALUATION_METRICS["test_full"], EVALUATION_METRICS["test_clean"]],
                     index=["full", "clean"]))


# %% [markdown]
# ## 3.11 Compare against the measured Phase-2 retrieval baselines
#
# These constants are the actual Phase-2 results supplied with this experiment—not placeholder
# targets. Phase-2 F1 and BLEU use the same lexical tokenization and multiset F1 definition.

# %%
PHASE2_BASELINES = {
    "validation": {
        "TF-IDF retrieval": {"bleu": 0.1094276713, "token_f1": 0.4129747154,
                             "exact_match": 0.0072231140, "edit_distance": 3.9374},
        "SBERT retrieval": {"bleu": 0.0973204753, "token_f1": 0.3880102876,
                            "exact_match": 0.0080256822, "edit_distance": 4.3411},
    },
    "test": {
        "TF-IDF retrieval": {"bleu": 0.1069681297, "token_f1": 0.4215114575,
                             "exact_match": 0.0064153970},
        "SBERT retrieval": {"bleu": 0.1021802074, "token_f1": 0.4020750850,
                            "exact_match": 0.0104250200},
    },
}
PHASE2_RETRIEVAL_DIAGNOSTICS = {
    "sbert_mean_validation_similarity": 0.6476238777,
    "similarity_vs_gloss_f1_spearman": 0.5226967147,
}


def comparison_rows(split_name: str, phase3_metrics: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for name, metrics in PHASE2_BASELINES[split_name].items():
        rows.append({"split": split_name, "system": name, **metrics})
    rows.append({
        "split": split_name, "system": f"Phase 3 T5-small ({DECODING_METHOD})",
        "bleu": phase3_metrics["lexical_bleu"],
        "token_f1": phase3_metrics["lexical_token_f1"],
        "exact_match": phase3_metrics["lexical_exact_match"],
        "edit_distance": phase3_metrics["lexical_edit_distance"],
    })
    return rows


baseline_comparison = pd.DataFrame(
    comparison_rows("validation", val_metrics) + comparison_rows("test", test_metrics)
)
baseline_comparison.to_csv(OUTPUT / "phase2_phase3_comparison.csv", index=False)
display(baseline_comparison)

# %% [markdown]
# ## 3.12 Error slices: length, vocabulary, and linguistic constructions
#
# Length boundaries come from training tertiles and are frozen before application to test.
# Vocabulary groups use training-gloss frequency only. Linguistic subsets use declared regex
# rules and may overlap; their observed sizes are always reported.

# %%
analysis_df = test_predictions.copy()
analysis_df["sentence_token_count"] = analysis_df["sentence"].map(lambda x: len(str(x).split()))
analysis_df["reference_gloss_token_count"] = analysis_df["reference_gloss"].map(lambda x: len(lexical_tokens(x)))


def tertile_boundaries(values: pd.Series) -> tuple[float, float]:
    lower, upper = values.quantile([1 / 3, 2 / 3]).tolist()
    return float(lower), float(upper)


train_sentence_lengths = eligible_train[SENTENCE_COL].map(lambda x: len(normalize_whitespace(x).split()))
train_gloss_lengths = eligible_train["model_gloss"].map(lambda x: len(lexical_tokens(x)))
sentence_boundaries = tertile_boundaries(train_sentence_lengths)
gloss_boundaries = tertile_boundaries(train_gloss_lengths)


def apply_length_bin(value: int, boundaries: tuple[float, float]) -> str:
    if value <= boundaries[0]:
        return "short"
    if value <= boundaries[1]:
        return "medium"
    return "long"


analysis_df["sentence_length_bin"] = analysis_df["sentence_token_count"].map(
    lambda x: apply_length_bin(x, sentence_boundaries)
)
analysis_df["reference_gloss_length_bin"] = analysis_df["reference_gloss_token_count"].map(
    lambda x: apply_length_bin(x, gloss_boundaries)
)


def metrics_by_group(frame: pd.DataFrame, column: str, analysis_name: str) -> pd.DataFrame:
    records = []
    for value, group in frame.groupby(column, dropna=False):
        records.append({"analysis": analysis_name, "group": str(value), **aggregate_prediction_frame(group)})
    return pd.DataFrame(records)


length_analysis = pd.concat([
    metrics_by_group(analysis_df, "sentence_length_bin", "sentence_length"),
    metrics_by_group(analysis_df, "reference_gloss_length_bin", "reference_gloss_length"),
], ignore_index=True)
length_analysis.to_csv(DIAGNOSTICS / "length_analysis.csv", index=False)


def vocabulary_group(reference: str) -> str:
    tokens = lexical_tokens(reference)
    if any(token not in TRAIN_GLOSS_VOCAB for token in tokens):
        return "contains_unseen_token"
    if any(TRAIN_GLOSS_COUNTS[token] < 5 for token in tokens):
        return "contains_rare_known_token_freq_lt_5"
    return "all_tokens_frequency_ge_5"


analysis_df["vocabulary_group"] = analysis_df["reference_gloss"].map(vocabulary_group)
vocabulary_analysis = metrics_by_group(analysis_df, "vocabulary_group", "reference_vocabulary")
vocabulary_analysis.to_csv(DIAGNOSTICS / "vocabulary_slice_analysis.csv", index=False)

LINGUISTIC_PATTERNS = {
    "negation": re.compile(r"\b(no|not|never|neither|nor|nothing|nobody|cannot|can't|won't|don't|doesn't|didn't)\b", re.I),
    "question": re.compile(r"\?|\b(who|what|when|where|why|how|which|whose)\b", re.I),
    "future": re.compile(r"\b(will|shall|gonna|going to|tomorrow|next week|next month|next year)\b", re.I),
    "past": re.compile(r"\b(was|were|did|had|yesterday|last week|last month|last year|\w+ed)\b", re.I),
    "modal": re.compile(r"\b(can|could|may|might|must|shall|should|will|would)\b", re.I),
}
targeted_rows = []
for subset_name, pattern in LINGUISTIC_PATTERNS.items():
    mask = analysis_df["sentence"].map(lambda x: bool(pattern.search(str(x))))
    subset = analysis_df[mask]
    row = {"subset": subset_name, "rule": pattern.pattern, "n": int(len(subset))}
    if len(subset):
        row.update(aggregate_prediction_frame(subset))
    targeted_rows.append(row)
targeted_analysis = pd.DataFrame(targeted_rows)
targeted_analysis.to_csv(DIAGNOSTICS / "targeted_linguistic_analysis.csv", index=False)
analysis_df.to_csv(DIAGNOSTICS / "test_predictions_with_slices.csv", index=False)

display(length_analysis[["analysis", "group", "n", "lexical_token_f1", "lexical_bleu", "lexical_exact_match"]])
display(vocabulary_analysis[["group", "n", "lexical_token_f1", "lexical_bleu", "lexical_exact_match"]])
display(targeted_analysis[[c for c in ("subset", "n", "lexical_token_f1", "lexical_bleu", "lexical_exact_match") if c in targeted_analysis]])

figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for axis, (analysis_name, subset) in zip(axes, length_analysis.groupby("analysis", sort=False)):
    order = [name for name in ("short", "medium", "long") if name in set(subset["group"])]
    plotted = subset.set_index("group").loc[order]
    axis.bar(plotted.index, plotted["lexical_token_f1"])
    axis.set(title=analysis_name.replace("_", " "), xlabel="bin", ylabel="lexical token F1")
figure.tight_layout()
figure.savefig(FIGURES / "length_slice_f1.png", dpi=160, bbox_inches="tight")
figure.savefig(FIGURES / "performance_by_length.png", dpi=160, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## 3.13 Generated-vocabulary/repetition audit and qualitative examples
#
# The output audit does not rewrite predictions. It measures tokens absent from training glosses,
# repeated runs/patterns, malformed characters, empty predictions, and extreme generated length.
# Good/medium/poor examples are selected deterministically from test F1 ranks.

# %%
def generated_audit_row(row: pd.Series) -> dict[str, Any]:
    tokens = lexical_tokens(row["predicted_gloss"])
    unknown = [token for token in tokens if token not in TRAIN_GLOSS_VOCAB]
    source_tokens = set(lexical_tokens(row["sentence"]))
    source_copy_non_gloss = [token for token in unknown if token in source_tokens]
    unusual_characters = "".join(sorted(set(re.findall(
        r"[^\w\s/?.,'!:\-]", str(row["predicted_gloss"]), flags=re.UNICODE
    ))))
    token_repeat = max_consecutive_token_repeat(tokens)
    pattern_repeat = max_consecutive_pattern_repeat(tokens)
    flags = []
    if not tokens:
        flags.append("empty_prediction")
    if unknown:
        flags.append("generated_out_of_training_vocabulary")
    if CONTROL_RE.search(str(row["predicted_gloss"])):
        flags.append("control_character")
    if unusual_characters:
        flags.append("malformed_or_unusual_symbol")
    if source_copy_non_gloss:
        flags.append("possible_english_filler_or_source_copy")
    if token_repeat > AUDIT_THRESHOLDS["consecutive_token_repeats"]:
        flags.append("repeated_token_run")
    if pattern_repeat > AUDIT_THRESHOLDS["consecutive_pattern_repeats"]:
        flags.append("repeated_short_pattern")
    if len(tokens) > AUDIT_THRESHOLDS["long_gloss_tokens"]:
        flags.append("extreme_length")
    return {
        "generated_token_count": len(tokens), "generated_unknown_token_count": len(unknown),
        "generated_known_gloss_token_count": len(tokens) - len(unknown),
        "generated_unknown_tokens": "|".join(sorted(set(unknown))),
        "generated_source_copy_non_gloss_tokens": "|".join(sorted(set(source_copy_non_gloss))),
        "generated_unusual_characters": unusual_characters,
        "generated_max_token_repeat": token_repeat,
        "generated_max_pattern_repeat": pattern_repeat,
        "generated_audit_flags": "|".join(flags),
    }


generated_audit = test_predictions.apply(generated_audit_row, axis=1, result_type="expand")
test_predictions_audited = pd.concat([test_predictions.reset_index(drop=True), generated_audit], axis=1)
test_predictions_audited.to_csv(RESULTS / "test_predictions.csv", index=False)
generated_token_total = int(generated_audit["generated_token_count"].sum())
GENERATED_AUDIT_SUMMARY = {
    "predictions": len(generated_audit),
    "empty_predictions": int(generated_audit["generated_audit_flags"].str.contains("empty_prediction").sum()),
    "flagged_predictions": int(generated_audit["generated_audit_flags"].ne("").sum()),
    "unknown_generated_tokens": int(generated_audit["generated_unknown_token_count"].sum()),
    "unknown_generated_token_rate": float(generated_audit["generated_unknown_token_count"].sum() / max(generated_token_total, 1)),
    "known_gloss_like_token_rate": float(generated_audit["generated_known_gloss_token_count"].sum() / max(generated_token_total, 1)),
    "possible_english_filler_or_source_copy_predictions": int(
        generated_audit["generated_source_copy_non_gloss_tokens"].ne("").sum()
    ),
    "malformed_or_unusual_symbol_predictions": int(
        generated_audit["generated_unusual_characters"].ne("").sum()
    ),
    "max_generated_token_count": int(generated_audit["generated_token_count"].max()),
    "max_consecutive_token_repeat": int(generated_audit["generated_max_token_repeat"].max()),
    "max_consecutive_pattern_repeat": int(generated_audit["generated_max_pattern_repeat"].max()),
}
write_json(DIAGNOSTICS / "generated_gloss_audit.json", GENERATED_AUDIT_SUMMARY)
display(pd.Series(GENERATED_AUDIT_SUMMARY, name="generated_gloss_audit").to_frame())


def failure_reason(row: pd.Series) -> str:
    reference = lexical_tokens(row["reference_gloss"])
    prediction = lexical_tokens(row["predicted_gloss"])
    missing = list((Counter(reference) - Counter(prediction)).elements())
    extra = list((Counter(prediction) - Counter(reference)).elements())
    reasons = []
    if not prediction:
        reasons.append("empty output")
    if str(row.get("audit_flags", "")):
        reasons.append("annotation irregularity")
    if any(token not in TRAIN_GLOSS_VOCAB for token in reference):
        reasons.append("reference contains unseen training-vocabulary token")
    elif any(TRAIN_GLOSS_COUNTS[token] < 5 for token in reference):
        reasons.append("reference contains rare token")
    if missing:
        reasons.append("missing: " + " ".join(missing[:5]))
    if extra:
        if any(token not in TRAIN_GLOSS_VOCAB for token in extra):
            reasons.append("hallucinated gloss token: " + " ".join(extra[:5]))
        else:
            reasons.append("extra sign token: " + " ".join(extra[:5]))
    if len(prediction) < len(reference):
        reasons.append("under-generation")
    elif len(prediction) > len(reference):
        reasons.append("over-generation")
    if row.get("generated_max_pattern_repeat", 0) > AUDIT_THRESHOLDS["consecutive_pattern_repeats"]:
        reasons.append("repetition")
    if len(str(row["sentence"]).split()) > sentence_boundaries[1]:
        reasons.append("long sentence")
    if LINGUISTIC_PATTERNS["negation"].search(str(row["sentence"])):
        reasons.append("negation construction")
    if reference != prediction and Counter(reference) == Counter(prediction):
        reasons.append("word-order error")
    if row["lexical_token_f1"] < .25 and prediction:
        reasons.append("possible semantic misunderstanding")
    return "; ".join(reasons) if reasons else "exact lexical match"


ranked = test_predictions_audited.sort_values(
    ["lexical_token_f1", "lexical_bleu", "sample_id"], ascending=[False, False, True]
).reset_index(drop=True)
n_examples = min(20, max(len(ranked) // 3, 1))
good_examples = ranked.head(n_examples).copy()
poor_examples = ranked.tail(n_examples).sort_values(
    ["lexical_token_f1", "lexical_bleu", "sample_id"], ascending=[True, True, True]
).copy()
middle_start = max((len(ranked) - n_examples) // 2, n_examples)
medium_examples = ranked.iloc[middle_start:middle_start + n_examples].copy()
qualitative = pd.concat([
    good_examples.assign(quality_band="good"),
    medium_examples.assign(quality_band="medium"),
    poor_examples.assign(quality_band="poor"),
], ignore_index=True)
qualitative["failure_reason"] = qualitative.apply(failure_reason, axis=1)
qualitative_columns = [
    "quality_band", "sample_id", "sentence", "reference_gloss", "predicted_gloss",
    "lexical_bleu", "lexical_token_f1", "lexical_edit_distance", "lexical_exact_match",
    "failure_reason",
]
qualitative[qualitative_columns].to_csv(DIAGNOSTICS / "qualitative_examples.csv", index=False)
for band in ("good", "medium", "poor"):
    qualitative.loc[qualitative["quality_band"] == band, qualitative_columns].to_csv(
        RESULTS / f"qualitative_{band}.csv", index=False
    )
    print(f"{band.title()} examples:")
    display(qualitative.loc[qualitative["quality_band"] == band, qualitative_columns].reset_index(drop=True))


# %% [markdown]
# ## 3.14 Reloadable inference function and custom demo
#
# The model and tokenizer are reloaded only from the exported directories. This verifies that
# the deliverable is self-contained. `predict_gloss` applies the identical prefix and lengths.

# %%
del best_model
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

deployment_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True, use_fast=True)
deployment_model = AutoModelForSeq2SeqLM.from_pretrained(BEST_MODEL_DIR, local_files_only=True).to(DEVICE)
deployment_model.eval()


@torch.no_grad()
def predict_gloss(sentence: str, decoding: str = "best") -> dict[str, Any]:
    """Generate a gloss and return text, token IDs, and length with reproducible decoding."""
    cleaned = normalize_whitespace(sentence)
    if not cleaned:
        raise ValueError("Input text must not be empty")
    decoding_key = DECODING_METHOD if decoding == "best" else decoding
    if decoding_key not in {"greedy", "beam_4"}:
        raise ValueError("decoding must be 'best', 'greedy', or 'beam_4'")
    beams = 1 if decoding_key == "greedy" else CFG3.beam_width
    inputs = deployment_tokenizer(
        CFG3.prefix + cleaned, max_length=MAX_INPUT_LENGTH, truncation=True,
        return_tensors="pt",
    ).to(DEVICE)
    kwargs: dict[str, Any] = {
        **inputs, "max_length": MAX_TARGET_LENGTH, "num_beams": beams, "do_sample": False,
    }
    if beams > 1:
        kwargs["early_stopping"] = True
    output_ids = deployment_model.generate(**kwargs)
    token_ids = output_ids[0].detach().cpu().tolist()
    return {
        "sentence": cleaned,
        "predicted_gloss": raw_normalized(deployment_tokenizer.decode(output_ids[0], skip_special_tokens=True)),
        "token_ids": token_ids,
        "generation_length": len(token_ids),
        "decoding": decoding_key,
    }


CUSTOM_SENTENCES = [
    "I want to drink water.",
    "I am going to school tomorrow.",
    "Where is the nearest hospital?",
    "She did not finish her work yesterday.",
    "Can you help me?",
    "My brother likes playing football.",
    "We will meet next week.",
    "Please open the window.",
    "The children are reading books.",
    "Why are you late?",
    "I have never visited that city.",
    "He should call his mother.",
    "The weather is very hot today.",
]
custom_demo = pd.DataFrame({
    "sentence": CUSTOM_SENTENCES,
    "predicted_gloss": [predict_gloss(sentence)["predicted_gloss"] for sentence in CUSTOM_SENTENCES],
})
custom_demo.to_csv(RESULTS / "custom_predictions.csv", index=False)
display(custom_demo)
example_result = predict_gloss("I will go home tomorrow.")
print(example_result["predicted_gloss"])
display(example_result)


# %% [markdown]
# ## 3.15 Final report, artifact inventory, and Phase-4 hand-off
#
# Phase 4 must use ground-truth Phase-1 glosses for training. Phase-3 generated gloss is for
# inference and held-out analysis only; feeding predictions into Phase-4 training would compound
# errors and change the supervised task.

# %%
phase4_text = f"""# Phase 4 hand-off

- Locked Phase-1 fingerprint: `{OBSERVED_FINGERPRINT}`
- For Phase-4 **training**, join the saved Phase-1 training CSV to motion resources and use its
  ground-truth `{GLOSS_COL}` / `model_gloss` target. Do not replace training glosses with Phase-3 predictions.
- For end-to-end inference, call `predict_gloss(text)["predicted_gloss"]` and pass that value to Phase 4.
- Validation/test Phase-3 predictions are saved only for evaluation, analysis, and optional end-to-end testing.
- Best text-to-gloss model: `{BEST_MODEL_DIR}`
- Tokenizer: `{TOKENIZER_DIR}`
- Selected decoder: `{DECODING_METHOD}`

Fresh-notebook reload:

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("{TOKENIZER_DIR}", local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained("{BEST_MODEL_DIR}", local_files_only=True)

Apply the exact `{CFG3.prefix}` prefix and the lengths recorded in `phase3_config.json`.
Phase 4 does not need to load this model while learning gloss-to-motion; load it only for
end-to-end English-to-motion inference or held-out pipeline evaluation.
"""
(OUTPUT / "README_phase4.txt").write_text(phase4_text, encoding="utf-8")

package_versions = {}
for package in ("torch", "transformers", "tokenizers", "pandas", "numpy", "nltk"):
    try:
        package_versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        package_versions[package] = "not-found"

EXPERIMENT_CONFIG = {
    "phase": 3, "task": "English text to sign gloss sequence-to-sequence generation",
    "config": asdict(CFG3), "phase1_root": str(PHASE1_ROOT),
    "phase1_split_fingerprint_sha256": OBSERVED_FINGERPRINT,
    "columns": {"id": ID_COL, "sentence": SENTENCE_COL, "gloss": GLOSS_COL},
    "split_sizes": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
    "training_rows_after_explicit_unusable_exclusion": len(eligible_train),
    "offline_model_source": str(MODEL_SOURCE), "model_type": model_type,
    "parameter_count": parameter_count, "package_versions": package_versions,
    "device": str(DEVICE), "max_input_length": MAX_INPUT_LENGTH,
    "max_target_length": MAX_TARGET_LENGTH, "selected_decoding": DECODING_METHOD,
}
write_json(OUTPUT / "experiment_config.json", EXPERIMENT_CONFIG)
write_json(OUTPUT / "phase3_config.json", EXPERIMENT_CONFIG)

tfidf_val = PHASE2_BASELINES["validation"]["TF-IDF retrieval"]
sbert_val = PHASE2_BASELINES["validation"]["SBERT retrieval"]
tfidf_test = PHASE2_BASELINES["test"]["TF-IDF retrieval"]
sbert_test = PHASE2_BASELINES["test"]["SBERT retrieval"]
SUCCESS_CRITERIA = {
    "phase1_fingerprint_exact_match": OBSERVED_FINGERPRINT == EXPECTED_PHASE1_FINGERPRINT,
    "overfit_gate_passed": DEBUG_GATE["passed"],
    "train_input_truncation_below_1_percent": INPUT_TRUNCATION_RATE < .01,
    "train_target_truncation_below_1_percent": TARGET_TRUNCATION_RATE < .01,
    "phase3_validation_f1_beats_both_phase2_retrievers": (
        val_metrics["lexical_token_f1"] > max(tfidf_val["token_f1"], sbert_val["token_f1"])
    ),
    "phase3_test_f1_beats_both_phase2_retrievers": (
        test_metrics["lexical_token_f1"] > max(tfidf_test["token_f1"], sbert_test["token_f1"])
    ),
    "best_model_reload_and_demo_succeeded": len(custom_demo) == len(CUSTOM_SENTENCES),
}


def lexical_metric_view(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "n": metrics.get("n", 0.0),
        "bleu": metrics.get("lexical_bleu", float("nan")),
        "token_precision": metrics.get("lexical_token_precision", float("nan")),
        "token_recall": metrics.get("lexical_token_recall", float("nan")),
        "token_f1": metrics.get("lexical_token_f1", float("nan")),
        "exact_match": metrics.get("lexical_exact_match", float("nan")),
        "edit_distance": metrics.get("lexical_edit_distance", float("nan")),
        "normalized_edit_distance": metrics.get("lexical_normalized_edit_distance", float("nan")),
    }


required_before_report = [
    BEST_MODEL_DIR / "config.json", TOKENIZER_DIR / "tokenizer_config.json",
    DIAGNOSTICS / "gloss_audit.csv", DIAGNOSTICS / "vocabulary_report.json",
    RESULTS / "validation_predictions.csv", RESULTS / "test_predictions.csv",
    RESULTS / "qualitative_good.csv", RESULTS / "qualitative_medium.csv",
    RESULTS / "qualitative_poor.csv", RESULTS / "custom_predictions.csv",
    FIGURES / "training_loss.png", FIGURES / "validation_metrics.png",
    FIGURES / "performance_by_length.png", OUTPUT / "training_history.json",
    OUTPUT / "phase3_config.json", OUTPUT / "README_phase4.txt",
]
missing_required = [str(path) for path in required_before_report if not path.is_file()]
if missing_required:
    raise RuntimeError(f"Cannot mark Phase 3 complete; required artifacts are missing: {missing_required}")


PHASE3_REPORT = {
    "phase": "Phase 3 - Text to Gloss",
    "phase3_complete": True,
    "model": {
        "architecture": MODEL_DESCRIPTION + " sequence-to-sequence",
        "offline_source": str(MODEL_SOURCE),
        "best_model_directory": str(BEST_MODEL_DIR),
        "tokenizer_directory": str(TOKENIZER_DIR),
        "fixed_prefix": CFG3.prefix,
        "parameter_count": parameter_count,
    },
    "training_samples": len(eligible_train),
    "validation_samples": len(val_df),
    "test_samples": len(test_df),
    "training_samples_excluded_by_quality_control": int(len(train_df) - len(eligible_train)),
    "best_epoch": best_epoch,
    "validation": lexical_metric_view(val_metrics),
    "validation_raw_representation": {
        key.removeprefix("raw_"): value for key, value in val_metrics.items() if key.startswith("raw_")
    },
    "test": lexical_metric_view(test_metrics),
    "test_raw_representation": {
        key.removeprefix("raw_"): value for key, value in test_metrics.items() if key.startswith("raw_")
    },
    "clean_test_subset": lexical_metric_view(EVALUATION_METRICS["test_clean"]),
    "phase2_comparison": {
        "same_phase1_split": True,
        "baseline_results": PHASE2_BASELINES,
        "retrieval_diagnostics": PHASE2_RETRIEVAL_DIAGNOSTICS,
        "comparison_rows": baseline_comparison.to_dict(orient="records"),
    },
    "decoding_method": DECODING_METHOD,
    "phase1_split_fingerprint_sha256": OBSERVED_FINGERPRINT,
    "data_leakage_detected": False,
    "overfit_gate": DEBUG_GATE,
    "success_criteria": SUCCESS_CRITERIA,
}
write_json(OUTPUT / "phase3_report.json", PHASE3_REPORT)
assert (OUTPUT / "phase3_report.json").is_file()

FINAL_REPORT = {
    "status": "completed",
    "scope": "Phase 3 text-to-gloss only; no motion modeling",
    "phase1_split_fingerprint_sha256": OBSERVED_FINGERPRINT,
    "split_sizes": EXPERIMENT_CONFIG["split_sizes"],
    "gloss_audit": {
        "thresholds": AUDIT_THRESHOLDS,
        "summary": audit_summary.reset_index().to_dict(orient="records"),
        "length_statistics": gloss_length_statistics.reset_index().to_dict(orient="records"),
        "cleaning_policy": (
            "original_gloss retained; model_gloss applies whitespace normalization only; "
            "empty/non-string/control-character labels excluded from training; suspicious rows retained"
        ),
    },
    "vocabulary": VOCABULARY_REPORT,
    "tokenizer_lengths": LENGTH_REPORT,
    "overfit_gate": DEBUG_GATE,
    "training": {
        "epochs_completed": len(history), "best_epoch": best_epoch,
        "selection_rule": "validation lexical token F1, then lexical BLEU",
        "best_epoch_greedy_metrics": best_metrics,
    },
    "decoding_comparison": decoding_comparison.to_dict(orient="records"),
    "selected_decoding": DECODING_METHOD,
    "evaluation": EVALUATION_METRICS,
    "phase2_comparison": baseline_comparison.to_dict(orient="records"),
    "phase2_retrieval_diagnostics": PHASE2_RETRIEVAL_DIAGNOSTICS,
    "generated_gloss_audit": GENERATED_AUDIT_SUMMARY,
    "success_criteria": SUCCESS_CRITERIA,
    "phase4_rule": "Use ground-truth gloss for Phase-4 training; predicted gloss only at inference/evaluation.",
}
write_json(OUTPUT / "final_report.json", FINAL_REPORT)

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


artifact_rows = []
for path in sorted(OUTPUT.rglob("*")):
    if path.is_file():
        artifact_rows.append({
            "relative_path": str(path.relative_to(OUTPUT)), "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
artifact_inventory = pd.DataFrame(artifact_rows)
artifact_inventory.to_csv(OUTPUT / "artifact_inventory.csv", index=False)

print("FINAL PHASE-3 REPORT")
print("Split fingerprint:", OBSERVED_FINGERPRINT)
print("Model / tokenizer:", BEST_MODEL_DIR, "/", TOKENIZER_DIR)
print("Best epoch / selected decoding:", best_epoch, "/", DECODING_METHOD)
print("Validation lexical metrics:", val_metrics)
print("Test lexical metrics:", test_metrics)
print("Success criteria:", SUCCESS_CRITERIA)
print("Phase-4 rule: use ground-truth gloss for training; predicted gloss only at inference/evaluation.")
print("Artifacts:", OUTPUT)
display(artifact_inventory)

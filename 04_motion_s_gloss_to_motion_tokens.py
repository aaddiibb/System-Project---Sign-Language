# %% [markdown]
# # Motion-S Phase 4A — Token Representation and Data Alignment
#
# **Scope:** discovery and verification only. This notebook does not define a motion
# Transformer, create special token IDs, run an overfit experiment, or train a model.
# It must establish the exact Phase-1 split, Phase-3 alignment, RVQ token axes, codebook
# sizes, sequence statistics, and all-sample token integrity before Phase 4B is permitted.
#
# **Inputs:** the saved Phase-1 output, original Motion-S/SignVRSE dataset, and saved
# Phase-3 output. **Outputs:** Phase-4A diagnostics under
# `/kaggle/working/motion_s_phase4`. Expected canonical token shape is not assumed; it is
# derived from the Phase-1 export contract and verified against every row.

# %% [markdown]
# ## 4A.0 Reproducibility and centralized configuration
#
# The configuration contains discovery overrides only. Token-vocabulary and model parameters
# remain unset until evidence is collected. If discovery finds multiple candidates, set the
# corresponding path explicitly instead of allowing the notebook to guess.

# %%
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display


EXPECTED_FINGERPRINT = "e2acdf4d6dce91775239ad50be3866dbc6f5da94d940204b62456cdb77fc4b0d"
EXPECTED_SPLIT_SIZES = {"train": 9970, "validation": 1246, "test": 1247}
OFFICIAL_RVQ_DECLARATION = {
    "provider": "Signvrse",
    "model": "Motion-S-RVQ-VAE-Tokenizer",
    "source_url": "https://huggingface.co/Signvrse/Motion-S-RVQ-VAE-Tokenizer",
    "num_quantizers": 6,
    "num_embeddings_per_quantizer": 512,
    "valid_token_id_range": [0, 511],
}


@dataclass(frozen=True)
class Phase4Config:
    seed: int = 42
    kaggle_input: str = "/kaggle/input"
    phase1_root: Optional[str] = None
    motion_s_root: Optional[str] = None
    phase3_root: Optional[str] = None
    rvq_checkpoint: Optional[str] = None
    # Official Signvrse Motion-S RVQ-VAE declaration: 6 quantizers × 512 entries.
    # The all-row audit below still rejects any ID outside 0..511.
    declared_codebook_sizes: Optional[tuple[int, ...]] = (512, 512, 512, 512, 512, 512)
    declared_codebook_sizes_source: str = OFFICIAL_RVQ_DECLARATION["source_url"]
    output_dir: str = "/kaggle/working/motion_s_phase4"
    examples_to_show: int = 5
    max_metadata_file_bytes: int = 2_000_000
    max_checkpoint_candidates: int = 10


CFG4 = Phase4Config()
random.seed(CFG4.seed)
np.random.seed(CFG4.seed)
torch.manual_seed(CFG4.seed)
torch.cuda.manual_seed_all(CFG4.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

OUTPUT = Path(CFG4.output_dir)
DIAGNOSTICS = OUTPUT / "diagnostics"
FIGURES = OUTPUT / "figures"
for directory in (OUTPUT, DIAGNOSTICS, FIGURES):
    directory.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)


print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    properties = torch.cuda.get_device_properties(0)
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory (GiB):", round(properties.total_memory / 1024 ** 3, 2))
else:
    print("GPU: unavailable (Phase 4A can run on CPU; Phase 4B training will require a GPU)")
display(pd.Series(asdict(CFG4), name="value").to_frame())

# %% [markdown]
# ## 4A.1 Locate all three input packages
#
# **Input:** Kaggle inputs or explicit config paths. **Output:** one unambiguous Phase-1
# package, one original Motion-S root, and one Phase-3 output. No row data is trusted yet.

# %%
def is_phase1_root(path: Path) -> bool:
    return all(candidate.is_file() for candidate in (
        path / "dataset_config.json", path / "split_report.json", path / "sample_manifest.csv",
        path / "splits" / "train_split.csv", path / "splits" / "val_split.csv",
        path / "splits" / "test_split.csv",
    ))


def phase3_prediction_path(path: Path, split_name: str) -> Optional[Path]:
    names = [f"{split_name}_predictions.csv"]
    folders = [path / "results", path / "predictions", path]
    for folder in folders:
        for name in names:
            candidate = folder / name
            if candidate.is_file():
                return candidate
    return None


def is_phase3_root(path: Path) -> bool:
    return (
        (path / "phase3_report.json").is_file()
        and (path / "phase3_config.json").is_file()
        and phase3_prediction_path(path, "validation") is not None
        and phase3_prediction_path(path, "test") is not None
    )


def is_motion_s_root(path: Path) -> bool:
    return (
        (path / "train.csv").is_file()
        and (path / "Train").is_dir()
        and (path / "Motion-Features").is_dir()
    )


def unique_root(kind: str, override: Optional[str], predicate, working_candidate: Optional[Path] = None) -> Path:
    if override:
        selected = Path(override)
        if not predicate(selected):
            raise FileNotFoundError(f"Configured {kind} root is incomplete: {selected}")
        return selected.resolve()
    candidates: list[Path] = []
    if working_candidate is not None and predicate(working_candidate):
        candidates.append(working_candidate.resolve())
    input_root = Path(CFG4.kaggle_input)
    if input_root.exists():
        for current_root, directories, files in os.walk(input_root):
            current = Path(current_root)
            if predicate(current):
                candidates.append(current.resolve())
                directories[:] = []
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one {kind} root, found {[str(path) for path in candidates]}. "
            f"Set the corresponding Phase4Config path explicitly."
        )
    return candidates[0]


PHASE1_ROOT = unique_root("Phase-1", CFG4.phase1_root, is_phase1_root, Path("/kaggle/working/motion_s_phase1"))
MOTION_S_ROOT = unique_root("Motion-S", CFG4.motion_s_root, is_motion_s_root)
PHASE3_ROOT = unique_root("Phase-3", CFG4.phase3_root, is_phase3_root, Path("/kaggle/working/motion_s_phase3"))
print("Phase-1 root:", PHASE1_ROOT)
print("Motion-S root:", MOTION_S_ROOT)
print("Phase-3 root:", PHASE3_ROOT)

# %% [markdown]
# ## 4A.2 Verify the immutable Phase-1 split
#
# **Expected shapes:** three tables containing 9,970 / 1,246 / 1,247 rows. IDs must be
# unique and mutually disjoint. The manifest-derived SHA-256 must exactly equal the locked
# project fingerprint. A mismatch stops execution immediately.

# %%
with (PHASE1_ROOT / "dataset_config.json").open(encoding="utf-8") as handle:
    phase1_config = json.load(handle)
with (PHASE1_ROOT / "split_report.json").open(encoding="utf-8") as handle:
    phase1_split_report = json.load(handle)

ID_COL = phase1_config["id_column"]
SENTENCE_COL = phase1_config["sentence_column"]
GLOSS_COL = phase1_config["gloss_column"]
TOKEN_COLS = list(phase1_config["motion_token_columns"])
if not TOKEN_COLS:
    raise RuntimeError("Phase-1 config has no motion-token columns")

split_paths = {
    "train": PHASE1_ROOT / "splits" / "train_split.csv",
    "validation": PHASE1_ROOT / "splits" / "val_split.csv",
    "test": PHASE1_ROOT / "splits" / "test_split.csv",
}
splits = {
    name: pd.read_csv(path, dtype={ID_COL: str}, low_memory=False)
    for name, path in split_paths.items()
}
manifest = pd.read_csv(PHASE1_ROOT / "sample_manifest.csv", dtype={"sample_id": str}, low_memory=False)

for name, frame in splits.items():
    required = [ID_COL, SENTENCE_COL, GLOSS_COL, *TOKEN_COLS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Phase-1 {name} split lacks columns: {missing}")
    if len(frame) != EXPECTED_SPLIT_SIZES[name]:
        raise RuntimeError(f"Wrong {name} count: {len(frame)} != {EXPECTED_SPLIT_SIZES[name]}")
    if frame[ID_COL].isna().any() or frame[ID_COL].duplicated().any():
        raise RuntimeError(f"Missing or duplicate IDs in {name}")

id_sets = {name: set(frame[ID_COL]) for name, frame in splits.items()}
intersections = {
    "train_validation": len(id_sets["train"] & id_sets["validation"]),
    "train_test": len(id_sets["train"] & id_sets["test"]),
    "validation_test": len(id_sets["validation"] & id_sets["test"]),
}
if any(intersections.values()):
    raise RuntimeError(f"Split leakage detected: {intersections}")
if len(manifest) != sum(EXPECTED_SPLIT_SIZES.values()) or manifest["sample_id"].duplicated().any():
    raise RuntimeError("Phase-1 manifest count/uniqueness failure")

fingerprint_payload = "\n".join(
    f"{row.sample_id},{row.split}"
    for row in manifest[["sample_id", "split"]].sort_values("sample_id").itertuples(index=False)
)
OBSERVED_FINGERPRINT = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()
fingerprint_sources = {
    "computed_from_manifest": OBSERVED_FINGERPRINT,
    "dataset_config": phase1_config.get("split_fingerprint_sha256"),
    "split_report": phase1_split_report.get("split_fingerprint_sha256"),
}
if any(value != EXPECTED_FINGERPRINT for value in fingerprint_sources.values()):
    raise RuntimeError(f"PHASE-1 FINGERPRINT FAILURE: {fingerprint_sources}")
if phase1_split_report.get("split_verified") is not True:
    raise RuntimeError("Phase-1 split_report does not mark split_verified=true")
declared_split_sizes = {
    "train": phase1_config.get("train_size"),
    "validation": phase1_config.get("val_size"),
    "test": phase1_config.get("test_size"),
}
if declared_split_sizes != EXPECTED_SPLIT_SIZES:
    raise RuntimeError(f"Phase-1 dataset_config split sizes disagree: {declared_split_sizes}")

print("PHASE-1 SPLIT: PASS")
print("Fingerprint:", OBSERVED_FINGERPRINT)
print("Counts:", {name: len(frame) for name, frame in splits.items()})
print("ID intersections:", intersections)
print("Columns:", {"id": ID_COL, "sentence": SENTENCE_COL, "gloss": GLOSS_COL, "motion_tokens": TOKEN_COLS})

# %% [markdown]
# ## 4A.3 Align Phase-3 predictions by sample ID
#
# Phase-3 files are joined to validation/test by ID with one-to-one validation. Ground-truth
# glosses must match Phase 1 after whitespace-only normalization. Predictions are never joined
# by row position and are never manually repaired.

# %%
with (PHASE3_ROOT / "phase3_report.json").open(encoding="utf-8") as handle:
    phase3_report = json.load(handle)
with (PHASE3_ROOT / "phase3_config.json").open(encoding="utf-8") as handle:
    phase3_config = json.load(handle)

for source_name, payload in (("phase3_report", phase3_report), ("phase3_config", phase3_config)):
    reported = payload.get("phase1_split_fingerprint_sha256")
    if reported is None and isinstance(payload.get("config"), dict):
        reported = payload.get("config", {}).get("phase1_split_fingerprint_sha256")
    if reported != EXPECTED_FINGERPRINT:
        raise RuntimeError(f"{source_name} has wrong/missing Phase-1 fingerprint: {reported}")

if phase3_report.get("phase3_complete") is not True:
    raise RuntimeError("Phase-3 report does not mark Phase 3 complete")
if phase3_report.get("data_leakage_detected") is not False:
    raise RuntimeError("Phase-3 report does not explicitly confirm data_leakage_detected=false")


def reported_phase3_count(split_name: str) -> Optional[int]:
    direct_aliases = {
        "train": ["training_samples", "train_samples"],
        "validation": ["validation_samples", "val_samples"],
        "test": ["test_samples"],
    }
    for alias in direct_aliases[split_name]:
        if alias in phase3_report:
            return int(phase3_report[alias])
    split_sizes = phase3_report.get("split_sizes", {})
    for alias in (split_name, "val" if split_name == "validation" else split_name):
        if alias in split_sizes:
            return int(split_sizes[alias])
    return None


reported_phase3_sizes = {name: reported_phase3_count(name) for name in EXPECTED_SPLIT_SIZES}
if reported_phase3_sizes != EXPECTED_SPLIT_SIZES:
    raise RuntimeError(f"Phase-3 report split counts disagree: {reported_phase3_sizes}")


def normalized_space(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if pd.isna(value) else str(value)).strip()


def find_column(frame: pd.DataFrame, aliases: list[str], purpose: str) -> str:
    lower_map = {str(column).lower(): str(column) for column in frame.columns}
    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    raise KeyError(f"Cannot find {purpose} in columns {list(frame.columns)}")


phase3_alignment: dict[str, pd.DataFrame] = {}
phase3_join_status: dict[str, bool] = {}
for split_name in ("validation", "test"):
    path = phase3_prediction_path(PHASE3_ROOT, split_name)
    assert path is not None
    predictions = pd.read_csv(path, low_memory=False)
    prediction_id = find_column(predictions, ["sample_id", ID_COL, "id"], "Phase-3 sample ID")
    predictions = pd.read_csv(path, dtype={prediction_id: str}, low_memory=False)
    ground_truth = find_column(
        predictions,
        ["ground_truth_gloss", "reference_gloss", "raw_reference", "normalized_reference"],
        "Phase-3 ground-truth gloss",
    )
    predicted_gloss = find_column(
        predictions, ["predicted_gloss", "normalized_prediction", "raw_prediction"],
        "Phase-3 predicted gloss",
    )
    gloss_f1 = find_column(predictions, ["lexical_token_f1", "token_f1", "gloss_token_f1"], "Phase-3 gloss F1")
    exact_match = next((column for column in predictions.columns if str(column).lower() in {
        "lexical_exact_match", "exact_match", "gloss_exact_match"
    }), None)
    audit_column = next((column for column in predictions.columns if str(column).lower() in {
        "audit_flags", "generated_audit_flags", "flags"
    }), None)
    core_columns = [prediction_id, ground_truth, predicted_gloss, gloss_f1]
    core_columns += [exact_match] if exact_match else []
    core_columns += [audit_column] if audit_column else []
    analysis_terms = ("audit", "unknown", "repeat", "unusual", "edit_distance", "bleu", "token_precision", "token_recall")
    optional_columns = [
        column for column in predictions.columns
        if column not in core_columns and any(term in str(column).lower() for term in analysis_terms)
    ]
    selected = predictions[core_columns + optional_columns].copy()
    rename = {
        prediction_id: ID_COL,
        ground_truth: "phase3_ground_truth_gloss",
        predicted_gloss: "phase3_predicted_gloss",
        gloss_f1: "phase3_gloss_f1",
    }
    if exact_match:
        rename[exact_match] = "phase3_gloss_exact_match"
    if audit_column:
        rename[audit_column] = "phase3_audit_flags"
    for column in optional_columns:
        normalized_name = re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_")
        rename[column] = f"phase3_{normalized_name}"
    selected = selected.rename(columns=rename)
    selected[ID_COL] = selected[ID_COL].astype(str)
    if len(selected) != EXPECTED_SPLIT_SIZES[split_name] or selected[ID_COL].duplicated().any():
        raise RuntimeError(f"Phase-3 {split_name} row count or ID uniqueness failure")
    base = splits[split_name][[ID_COL, SENTENCE_COL, GLOSS_COL]].copy()
    aligned = base.merge(selected, on=ID_COL, how="outer", validate="one_to_one", indicator=True)
    unmatched = aligned[aligned["_merge"] != "both"]
    mismatch = aligned[
        aligned[GLOSS_COL].map(normalized_space)
        != aligned["phase3_ground_truth_gloss"].map(normalized_space)
    ]
    aligned.to_csv(DIAGNOSTICS / f"phase3_{split_name}_alignment.csv", index=False)
    if len(unmatched) or len(mismatch):
        mismatch.head(100).to_csv(DIAGNOSTICS / f"phase3_{split_name}_gloss_mismatches.csv", index=False)
        raise RuntimeError(
            f"Phase-3 {split_name} alignment failed: unmatched={len(unmatched)}, "
            f"ground-truth gloss mismatches={len(mismatch)}"
        )
    phase3_alignment[split_name] = aligned.drop(columns="_merge")
    phase3_join_status[split_name] = True
    print(f"Phase-3 {split_name} join: PASS ({len(aligned)} rows), file={path}")

# %% [markdown]
# ## 4A.4 Establish the token source and serialization contract
#
# Phase-1 declares one JSON integer array per RVQ-level CSV column. This notebook verifies that
# declaration against all split rows and the original Motion-S `train.csv`. Column order is
# preserved exactly; no flattening or transposition is performed.

# %%
source_header = pd.read_csv(MOTION_S_ROOT / "train.csv", nrows=0)
required_source_columns = [ID_COL, GLOSS_COL, *TOKEN_COLS]
missing_source_columns = [column for column in required_source_columns if column not in source_header.columns]
if missing_source_columns:
    raise RuntimeError(f"Original Motion-S train.csv lacks Phase-1 token contract columns: {missing_source_columns}")
source_df = pd.read_csv(
    MOTION_S_ROOT / "train.csv", dtype={ID_COL: str}, usecols=required_source_columns, low_memory=False
)
if source_df[ID_COL].duplicated().any():
    raise RuntimeError("Original Motion-S train.csv has duplicate sample IDs")

TOKEN_SOURCE_DESCRIPTION = {
    "original_file": str(MOTION_S_ROOT / "train.csv"),
    "phase1_portable_files": {name: str(path) for name, path in split_paths.items()},
    "serialization": phase1_config.get("motion_token_serialization"),
    "provider_column_order": TOKEN_COLS,
    "phase1_declared_rvq_levels": phase1_config.get("rvq_levels"),
    "canonicalization_rule": "one 1-D CSV column per level; stack in provider order to [L,T]",
    "flattening_used": False,
    "official_rvq_declaration": OFFICIAL_RVQ_DECLARATION,
}
if phase1_config.get("rvq_levels") != len(TOKEN_COLS):
    raise RuntimeError("Phase-1 rvq_levels disagrees with motion_token_columns")
print(json.dumps(TOKEN_SOURCE_DESCRIPTION, indent=2))


class TokenParseError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def parse_token_cell(value: Any, exported_json_only: bool) -> np.ndarray:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise TokenParseError("missing_token", "missing token cell")
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        raise TokenParseError("missing_token", "blank/null token cell")
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError) as json_error:
        if exported_json_only:
            raise TokenParseError("corrupted_serialization", repr(json_error)) from json_error
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError) as source_error:
            if any(mark in text for mark in ("[", "]", "{", "}", "|", ";")):
                raise TokenParseError("corrupted_serialization", repr(source_error)) from source_error
            try:
                parsed = [int(piece) for piece in re.split(r"[\s,]+", text) if piece]
            except ValueError as plain_error:
                raise TokenParseError("corrupted_serialization", repr(plain_error)) from plain_error
    array = np.asarray(parsed)
    if array.dtype == object:
        raise TokenParseError("corrupted_serialization", "ragged/object sequence")
    if array.ndim != 1:
        raise TokenParseError("shape_mismatch", f"expected one level [T], got {array.shape}")
    if array.size == 0:
        raise TokenParseError("empty_token", "empty token sequence")
    try:
        numeric = array.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise TokenParseError("corrupted_serialization", "non-numeric token") from error
    if np.isnan(numeric).any() or np.isinf(numeric).any():
        raise TokenParseError("nan_or_inf", "NaN/Inf token")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise TokenParseError("non_integer_token", "non-integer token ID")
    return numeric.astype(np.int64)

# %% [markdown]
# ## 4A.5 Audit every Phase-1 token target
#
# **Expected tensor shape per valid example:** `[L,T]`, where `L` equals the discovered number
# of level columns and all levels share `T`. Every failure remains an explicit audit row; no
# missing or malformed target is replaced.

# %%
combined_parts = []
for split_name, frame in splits.items():
    part = frame.copy()
    part["_phase4a_split"] = split_name
    combined_parts.append(part)
all_split_rows = pd.concat(combined_parts, ignore_index=True)

token_by_id: dict[str, np.ndarray] = {}
audit_records: list[dict[str, Any]] = []
for _, row in all_split_rows.iterrows():
    sample_id = str(row[ID_COL])
    layers: list[np.ndarray] = []
    issues: list[str] = []
    for column in TOKEN_COLS:
        try:
            layers.append(parse_token_cell(row[column], exported_json_only=True))
        except TokenParseError as error:
            issues.append(f"{column}:{error.code}")
    shape_mismatch = False
    if len(layers) == len(TOKEN_COLS):
        lengths = {len(layer) for layer in layers}
        shape_mismatch = len(lengths) != 1
        if shape_mismatch:
            issues.append("per_level_temporal_length_mismatch")
    if not issues and not shape_mismatch:
        tokens = np.stack(layers, axis=0)
        token_by_id[sample_id] = tokens
        token_shape = f"[{tokens.shape[0]},{tokens.shape[1]}]"
        temporal_length = int(tokens.shape[1])
        minimum_token = int(tokens.min())
        maximum_token = int(tokens.max())
        negative_token_count = int((tokens < 0).sum())
        per_level_min = [int(tokens[level].min()) for level in range(tokens.shape[0])]
        per_level_max = [int(tokens[level].max()) for level in range(tokens.shape[0])]
    else:
        token_shape = None
        temporal_length = None
        minimum_token = None
        maximum_token = None
        negative_token_count = None
        per_level_min = None
        per_level_max = None
    declared_length = row.get("token_length", np.nan)
    length_metadata_mismatch = bool(
        temporal_length is not None and pd.notna(declared_length) and int(declared_length) != temporal_length
    )
    if length_metadata_mismatch:
        issues.append("phase1_token_length_metadata_mismatch")
    audit_records.append({
        "sample_id": sample_id, "split": row["_phase4a_split"],
        "token_shape": token_shape, "temporal_length": temporal_length,
        "rvq_layers": len(layers) if not issues else None,
        "minimum_token": minimum_token, "maximum_token": maximum_token,
        "negative_token_count": negative_token_count,
        "invalid_token_count": None,
        "missing_token": any("missing_token" in issue for issue in issues),
        "empty_token": any("empty_token" in issue for issue in issues),
        "has_nan_or_inf": any("nan_or_inf" in issue for issue in issues),
        "corrupted_serialization": any("corrupted_serialization" in issue for issue in issues),
        "shape_mismatch": shape_mismatch or any("shape_mismatch" in issue for issue in issues),
        "length_metadata_mismatch": length_metadata_mismatch,
        "per_level_min": json.dumps(per_level_min), "per_level_max": json.dumps(per_level_max),
        "issues": "|".join(issues), "valid_before_codebook_check": not bool(issues),
    })

token_audit = pd.DataFrame(audit_records)
print("Parsed valid token rows:", len(token_by_id), "/", len(all_split_rows))
print("Pre-codebook audit failures:", int((~token_audit["valid_before_codebook_check"]).sum()))
display(token_audit[~token_audit["valid_before_codebook_check"]].head(30))

if not token_by_id:
    token_audit.to_csv(DIAGNOSTICS / "motion_token_audit.csv", index=False)
    raise RuntimeError("No valid motion-token rows. Audit saved; Phase 4A cannot continue.")
observed_level_counts = Counter(tokens.shape[0] for tokens in token_by_id.values())
if set(observed_level_counts) != {len(TOKEN_COLS)}:
    raise RuntimeError(f"Inconsistent RVQ levels: {dict(observed_level_counts)}")

# %% [markdown]
# ## 4A.6 Inspect five real examples and derive observed token evidence
#
# These examples print source text, gloss, raw serialization, canonical `[L,T]` shape, and
# per-level ranges. Dataset-wide counters establish observed ranges but do not by themselves
# pretend to prove the pretrained RVQ codebook's allocated size.

# %%
example_ids = splits["train"][ID_COL].sample(
    n=min(CFG4.examples_to_show, len(splits["train"])), random_state=CFG4.seed
).astype(str).tolist()
for sample_id in example_ids:
    row = splits["train"].set_index(ID_COL).loc[sample_id]
    tokens = token_by_id.get(sample_id)
    print("\nSample:", sample_id)
    print("Sentence:", row[SENTENCE_COL])
    print("Gloss:", row[GLOSS_COL])
    print("Motion token type: one JSON integer array per configured RVQ-level column")
    print("Token columns/order:", TOKEN_COLS)
    print("Raw first level:", str(row[TOKEN_COLS[0]])[:240])
    if tokens is None:
        print("Token parse: FAILED — see audit")
        continue
    print("Token shape [L,T]:", tuple(tokens.shape))
    print("RVQ levels:", tokens.shape[0], "Temporal length:", tokens.shape[1])
    for level in range(tokens.shape[0]):
        print(f"Layer {level} ({TOKEN_COLS[level]}) min/max:", int(tokens[level].min()), int(tokens[level].max()))

level_frequencies = [Counter() for _ in TOKEN_COLS]
for tokens in token_by_id.values():
    for level in range(len(TOKEN_COLS)):
        level_frequencies[level].update(int(value) for value in tokens[level])

observed_ranges = []
for level, frequencies in enumerate(level_frequencies):
    ids = sorted(frequencies)
    observed_ranges.append({
        "level": level, "column": TOKEN_COLS[level],
        "observed_min_id": min(ids) if ids else None,
        "observed_max_id": max(ids) if ids else None,
        "observed_unique_ids": len(ids),
        "observed_ids_contiguous_from_zero": bool(
            ids and ids[0] == 0 and len(ids) == ids[-1] + 1
            and all(right - left == 1 for left, right in zip(ids, ids[1:]))
        ),
        "observed_minimum_codebook_size": max(ids) + 1 if ids else None,
    })
display(pd.DataFrame(observed_ranges))

# %% [markdown]
# ## 4A.7 Resolve exact codebook sizes from declarations or RVQ artifacts
#
# Observed maxima are only lower bounds. Exact sizes are accepted from: an explicit reviewed
# config override, Phase-1 declaration, unambiguous Motion-S source/config metadata, or
# codebook/quantizer tensor evidence from safely inspected checkpoints. If exact size remains
# unresolved, Phase 4A finishes with `READY TO BUILD MOTION MODEL: NO`.

# %%
metadata_evidence: list[dict[str, Any]] = []
metadata_patterns = {
    "codebook_size": re.compile(r"(?i)\b(codebook_size|num_codes|n_codes|nb_code|n_embed|num_embeddings)\b\s*[\"']?\s*[:=]\s*(\d+)"),
    "rvq_levels": re.compile(r"(?i)\b(num_quantizers|rvq_levels|num_codebooks|n_q)\b\s*[\"']?\s*[:=]\s*(\d+)"),
}
metadata_suffixes = {".json", ".yaml", ".yml", ".toml", ".py", ".txt", ".md"}
for path in MOTION_S_ROOT.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in metadata_suffixes:
        continue
    try:
        if path.stat().st_size > CFG4.max_metadata_file_bytes:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    if not re.search(r"(?i)rvq|residual.?quant|codebook|quantizer|nb_code|n_embed", text):
        continue
    for evidence_type, pattern in metadata_patterns.items():
        for match in pattern.finditer(text):
            metadata_evidence.append({
                "path": str(path.relative_to(MOTION_S_ROOT)), "type": evidence_type,
                "key": match.group(1), "value": int(match.group(2)),
                "context": text[max(0, match.start() - 80):match.end() + 80].replace("\n", " "),
            })


def checkpoint_candidates() -> list[Path]:
    if CFG4.rvq_checkpoint:
        candidate = Path(CFG4.rvq_checkpoint)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return [candidate]
    allowed = {".pt", ".pth", ".ckpt", ".safetensors"}
    candidates = []
    roots = [MOTION_S_ROOT, Path(CFG4.kaggle_input)]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            lower = str(path).lower()
            if path.is_file() and path.suffix.lower() in allowed and any(
                term in lower for term in ("rvq", "vqvae", "vq_vae", "codebook", "quantizer", "motion-s_vae")
            ):
                candidates.append(path.resolve())
    return list(dict.fromkeys(candidates))[:CFG4.max_checkpoint_candidates]


def flatten_tensor_items(value: Any, prefix: str = ""):
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_tensor_items(child, child_prefix)


checkpoint_evidence: list[dict[str, Any]] = []
checkpoint_errors: list[dict[str, str]] = []
for checkpoint_path in checkpoint_candidates():
    try:
        tensor_items: list[tuple[str, tuple[int, ...]]] = []
        loaded = None
        if checkpoint_path.suffix.lower() == ".safetensors":
            from safetensors import safe_open
            with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
                tensor_items = [(key, tuple(handle.get_slice(key).get_shape())) for key in handle.keys()]
        else:
            loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            tensor_items = [(key, tuple(tensor.shape)) for key, tensor in flatten_tensor_items(loaded)]
        for key, shape in tensor_items:
            if any(term in key.lower() for term in ("codebook", "quantizer", "embedding", "embed")) and len(shape) in (2, 3):
                checkpoint_evidence.append({
                    "path": str(checkpoint_path), "key": key, "shape": list(shape),
                    "strong_codebook_key": any(term in key.lower() for term in ("codebook", "quantizer")),
                })
        if loaded is not None:
            del loaded
    except Exception as error:
        checkpoint_errors.append({"path": str(checkpoint_path), "error": repr(error)})

write_json(DIAGNOSTICS / "rvq_metadata_evidence.json", {
    "official_provider_declaration": OFFICIAL_RVQ_DECLARATION,
    "source_metadata": metadata_evidence,
    "checkpoint_tensors": checkpoint_evidence,
    "checkpoint_errors": checkpoint_errors,
})


def resolve_codebook_sizes() -> tuple[Optional[list[int]], str, list[int]]:
    levels = len(TOKEN_COLS)
    maxima = [item["observed_max_id"] for item in observed_ranges]
    if CFG4.declared_codebook_sizes is not None:
        sizes = list(CFG4.declared_codebook_sizes)
        source = (
            "official Signvrse provider declaration recorded in Phase4Config: "
            + CFG4.declared_codebook_sizes_source
        )
    else:
        declared = phase1_config.get("codebook_sizes", phase1_config.get("codebook_size"))
        if isinstance(declared, int):
            sizes, source = [declared] * levels, "Phase-1 dataset_config declaration"
        elif isinstance(declared, list) and declared:
            sizes, source = [int(value) for value in declared], "Phase-1 dataset_config declaration"
        else:
            metadata_sizes = sorted({
                item["value"] for item in metadata_evidence
                if item["type"] == "codebook_size" and item["value"] > max(maxima)
            })
            strong_evidence = [item for item in checkpoint_evidence if item["strong_codebook_key"]]
            dimension_evidence = strong_evidence or checkpoint_evidence
            plausible_checkpoint_dims = set()
            for item in dimension_evidence:
                for dimension in item["shape"]:
                    if dimension > max(maxima):
                        plausible_checkpoint_dims.add(int(dimension))
            combined = sorted(set(metadata_sizes) | plausible_checkpoint_dims)
            if len(metadata_sizes) == 1:
                sizes, source = [metadata_sizes[0]] * levels, "unambiguous Motion-S source/config declaration"
            elif len(combined) == 1:
                sizes, source = [combined[0]] * levels, "unambiguous RVQ checkpoint tensor dimension"
            else:
                return None, "unresolved: exact declaration/checkpoint evidence is absent or ambiguous", combined
    if len(sizes) != levels:
        raise RuntimeError(f"Codebook-size count {len(sizes)} != RVQ levels {levels}")
    if any(size <= maximum for size, maximum in zip(sizes, maxima)):
        raise RuntimeError(f"Declared codebook sizes {sizes} do not contain observed maxima {maxima}")
    return sizes, source, sizes


CODEBOOK_SIZES, CODEBOOK_RESOLUTION_SOURCE, CODEBOOK_CANDIDATES = resolve_codebook_sizes()
print("Exact codebook sizes:", CODEBOOK_SIZES)
print("Resolution source:", CODEBOOK_RESOLUTION_SOURCE)
if CODEBOOK_SIZES is None:
    print("Plausible but NOT accepted candidates:", CODEBOOK_CANDIDATES)
display(pd.DataFrame(metadata_evidence).head(30))
display(pd.DataFrame(checkpoint_evidence).head(30))

# %% [markdown]
# ## 4A.8 Finalize integrity, codebook usage, and special-token evidence
#
# Invalid IDs are computed only when exact codebook sizes are known. Existing PAD/BOS/EOS are
# reported from provider metadata; no model-side IDs are allocated in Phase 4A.

# %%
invalid_by_id: dict[str, Optional[int]] = {}
for sample_id, tokens in token_by_id.items():
    if CODEBOOK_SIZES is None:
        invalid_by_id[sample_id] = None
    else:
        invalid_by_id[sample_id] = int(sum(
            ((tokens[level] < 0) | (tokens[level] >= CODEBOOK_SIZES[level])).sum()
            for level in range(len(TOKEN_COLS))
        ))
token_audit["invalid_token_count"] = token_audit["sample_id"].map(invalid_by_id)
token_audit["valid"] = (
    token_audit["valid_before_codebook_check"]
    & (token_audit["invalid_token_count"].isna() | token_audit["invalid_token_count"].eq(0))
)
token_audit["fully_valid_with_exact_codebook"] = (
    token_audit["valid_before_codebook_check"]
    & bool(CODEBOOK_SIZES is not None)
    & token_audit["invalid_token_count"].fillna(1).eq(0)
)
token_audit.to_csv(DIAGNOSTICS / "motion_token_audit.csv", index=False)

usage_records = []
for level, frequencies in enumerate(level_frequencies):
    codebook_size = CODEBOOK_SIZES[level] if CODEBOOK_SIZES else None
    for token_id, count in sorted(frequencies.items()):
        usage_records.append({
            "rvq_level": level, "column": TOKEN_COLS[level], "token_id": token_id,
            "count": count, "fraction_of_level_tokens": count / max(sum(frequencies.values()), 1),
            "valid_for_resolved_codebook": (
                0 <= token_id < codebook_size if codebook_size is not None else None
            ),
        })
codebook_usage = pd.DataFrame(usage_records)
codebook_usage.to_csv(DIAGNOSTICS / "codebook_usage.csv", index=False)

special_token_evidence = phase1_config.get("special_token_ids")
SPECIAL_TOKEN_REPORT = {
    "provider_declared_special_token_ids": special_token_evidence,
    "negative_ids_observed": int(sum((tokens < 0).sum() for tokens in token_by_id.values())),
    "pad_bos_eos_present_in_targets": (
        "not declared; no out-of-range IDs observed" if special_token_evidence is None and CODEBOOK_SIZES
        and all((value or 0) == 0 for value in invalid_by_id.values())
        else "unresolved or provider-declared; inspect evidence"
    ),
    "phase4_model_special_tokens_allocated": False,
    "note": "PAD/BOS/EOS allocation is deferred to Phase 4B and must start at/above each exact codebook size.",
}
write_json(DIAGNOSTICS / "special_token_evidence.json", SPECIAL_TOKEN_REPORT)

audit_summary = {
    "total_samples": len(token_audit),
    "valid": int(token_audit["valid"].sum()),
    "fully_valid_with_exact_codebook": int(token_audit["fully_valid_with_exact_codebook"].sum()),
    "missing": int(token_audit["missing_token"].sum()),
    "empty": int(token_audit["empty_token"].sum()),
    "nan_or_inf": int(token_audit["has_nan_or_inf"].sum()),
    "malformed": int(token_audit["corrupted_serialization"].sum()),
    "shape_mismatches": int(token_audit["shape_mismatch"].sum()),
    "length_metadata_mismatches": int(token_audit["length_metadata_mismatch"].sum()),
    "invalid_ids": (
        int(token_audit["invalid_token_count"].sum()) if CODEBOOK_SIZES is not None else None
    ),
}
display(pd.Series(audit_summary, name="token_integrity").to_frame())

# %% [markdown]
# ## 4A.9 Verify every target against the original Motion-S table
#
# This all-row comparison proves that Phase-1 IDs, glosses, RVQ level ordering, and token values
# still align with the original attached resource. Comparisons parse data; they do not compare
# dataframe positions or raw string formatting.

# %%
source_by_id = source_df.set_index(ID_COL, drop=False)
all_ids = set(all_split_rows[ID_COL].astype(str))
missing_in_source = sorted(all_ids - set(source_by_id.index.astype(str)))
source_gloss_mismatches = []
source_token_mismatches = []
source_parse_errors = []

for _, row in all_split_rows.iterrows():
    sample_id = str(row[ID_COL])
    if sample_id not in source_by_id.index:
        continue
    source_row = source_by_id.loc[sample_id]
    if normalized_space(row[GLOSS_COL]) != normalized_space(source_row[GLOSS_COL]):
        source_gloss_mismatches.append(sample_id)
    if sample_id not in token_by_id:
        continue
    try:
        source_layers = [parse_token_cell(source_row[column], exported_json_only=False) for column in TOKEN_COLS]
        if len({len(layer) for layer in source_layers}) != 1:
            source_token_mismatches.append(sample_id)
            continue
        source_tokens = np.stack(source_layers, axis=0)
        if source_tokens.shape != token_by_id[sample_id].shape or not np.array_equal(source_tokens, token_by_id[sample_id]):
            source_token_mismatches.append(sample_id)
    except TokenParseError as error:
        source_parse_errors.append({"sample_id": sample_id, "error": f"{error.code}: {error}"})

source_alignment_report = {
    "phase1_ids_missing_from_original_source": len(missing_in_source),
    "gloss_mismatches": len(source_gloss_mismatches),
    "motion_token_mismatches": len(source_token_mismatches),
    "original_source_parse_errors": len(source_parse_errors),
    "all_rows_aligned": not any((missing_in_source, source_gloss_mismatches, source_token_mismatches, source_parse_errors)),
    "mismatch_previews": {
        "missing_ids": missing_in_source[:100], "gloss": source_gloss_mismatches[:100],
        "tokens": source_token_mismatches[:100], "parse_errors": source_parse_errors[:100],
    },
}
write_json(DIAGNOSTICS / "motion_s_source_alignment.json", source_alignment_report)
print("Original Motion-S alignment:", "PASS" if source_alignment_report["all_rows_aligned"] else "FAIL")
display(pd.Series({key: value for key, value in source_alignment_report.items() if key != "mismatch_previews"}).to_frame())

# %% [markdown]
# ## 4A.10 Sequence statistics and gloss–motion length relationship
#
# Statistics use every successfully parsed target. Dynamic padding is recommended for Phase 4B;
# no arbitrary global maximum is selected here. The plot is diagnostic only.

# %%
valid_lengths = token_audit.loc[token_audit["valid_before_codebook_check"], "temporal_length"].astype(int)
length_statistics = {
    "minimum": int(valid_lengths.min()), "mean": float(valid_lengths.mean()),
    "median": float(valid_lengths.median()), "p90": float(valid_lengths.quantile(.90)),
    "p95": float(valid_lengths.quantile(.95)), "p99": float(valid_lengths.quantile(.99)),
    "maximum": int(valid_lengths.max()),
}

length_frame = all_split_rows[[ID_COL, GLOSS_COL]].copy()
length_frame[ID_COL] = length_frame[ID_COL].astype(str)
length_frame["gloss_token_length"] = length_frame[GLOSS_COL].map(lambda value: len(normalized_space(value).split()))
length_frame = length_frame.merge(
    token_audit[["sample_id", "temporal_length"]], left_on=ID_COL, right_on="sample_id",
    how="left", validate="one_to_one",
).dropna(subset=["temporal_length"])
pearson_correlation = float(length_frame["gloss_token_length"].corr(length_frame["temporal_length"], method="pearson"))
spearman_correlation = float(
    length_frame["gloss_token_length"].rank(method="average").corr(
        length_frame["temporal_length"].rank(method="average"), method="pearson"
    )
)

per_level_statistics = []
for level, frequencies in enumerate(level_frequencies):
    ids = sorted(frequencies)
    resolved_size = CODEBOOK_SIZES[level] if CODEBOOK_SIZES else None
    per_level_statistics.append({
        **observed_ranges[level], "resolved_codebook_size": resolved_size,
        "codebook_resolution_source": CODEBOOK_RESOLUTION_SOURCE,
        "codebook_utilization_percent": (
            100.0 * len(ids) / resolved_size if resolved_size else None
        ),
        "most_frequent_10": frequencies.most_common(10),
        "least_frequent_10": sorted(frequencies.items(), key=lambda item: (item[1], item[0]))[:10],
    })

ORDER_SEMANTICS = (
    "base-named level followed by residual-named levels in Phase-1 provider-column order; "
    "distinct coarse/fine meanings are not independently asserted"
    if TOKEN_COLS and "base" in TOKEN_COLS[0].lower()
    and all("residual" in column.lower() for column in TOKEN_COLS[1:])
    else "provider-column order is verified; coarse-to-fine semantics are not documented by column names"
)
MOTION_TOKEN_STATISTICS = {
    "source": TOKEN_SOURCE_DESCRIPTION,
    "shape_convention": "[L,T]",
    "rvq_levels": len(TOKEN_COLS),
    "level_order": TOKEN_COLS,
    "level_order_semantics": ORDER_SEMANTICS,
    "hierarchy_documentation_status": (
        "ordering is documented by provider column names and Phase-1 export; level meanings beyond base/residual remain unclaimed"
    ),
    "all_levels_equal_temporal_length": not bool(token_audit["shape_mismatch"].any()),
    "zero_based": all(item["observed_min_id"] == 0 for item in observed_ranges),
    "codebook_sizes": CODEBOOK_SIZES,
    "codebook_resolution_source": CODEBOOK_RESOLUTION_SOURCE,
    "sequence_length_statistics": length_statistics,
    "gloss_vs_motion_length_pearson": pearson_correlation,
    "gloss_vs_motion_length_spearman": spearman_correlation,
    "per_level": per_level_statistics,
    "special_tokens": SPECIAL_TOKEN_REPORT,
}
write_json(DIAGNOSTICS / "motion_token_statistics.json", MOTION_TOKEN_STATISTICS)
print("Motion-token length statistics:")
display(pd.Series(length_statistics).to_frame("value"))
print("Gloss length vs motion-token length correlations:", {
    "pearson": pearson_correlation, "spearman": spearman_correlation
})

figure, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(valid_lengths, bins=min(60, max(10, int(math.sqrt(len(valid_lengths))))), edgecolor="black")
axes[0].set(title="Motion-token temporal lengths", xlabel="T", ylabel="samples")
axes[1].scatter(length_frame["gloss_token_length"], length_frame["temporal_length"], alpha=.20, s=8)
axes[1].set(title=f"Gloss vs motion length (Spearman={spearman_correlation:.3f})",
            xlabel="gloss whitespace tokens", ylabel="motion-token T")
figure.tight_layout()
figure.savefig(FIGURES / "token_length_distribution.png", dpi=160, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4A.11 Final PASS/FAIL gate and saved verification report
#
# `READY TO BUILD MOTION MODEL` becomes YES only when all immutable data, Phase-3 joins,
# all token rows, exact codebook sizes, ID ranges, and original-resource alignment pass.
# Failure saves diagnostics and then stops—Phase 4B must not be appended or run yet.

# %%
checks = {
    "phase1_fingerprint": OBSERVED_FINGERPRINT == EXPECTED_FINGERPRINT,
    "split_counts": all(len(splits[name]) == count for name, count in EXPECTED_SPLIT_SIZES.items()),
    "split_ids_disjoint": not any(intersections.values()),
    "phase3_validation_join": phase3_join_status.get("validation", False),
    "phase3_test_join": phase3_join_status.get("test", False),
    "all_token_rows_parse": len(token_by_id) == len(all_split_rows),
    "rvq_level_count_consistent": set(observed_level_counts) == {len(TOKEN_COLS)},
    "all_levels_share_temporal_length": not bool(token_audit["shape_mismatch"].any()),
    "phase1_length_metadata_matches": not bool(token_audit["length_metadata_mismatch"].any()),
    "exact_codebook_sizes_resolved": CODEBOOK_SIZES is not None,
    "no_invalid_codebook_ids": bool(
        CODEBOOK_SIZES is not None and token_audit["invalid_token_count"].fillna(1).eq(0).all()
    ),
    "original_motion_s_alignment": source_alignment_report["all_rows_aligned"],
}
READY_TO_BUILD_MOTION_MODEL = all(checks.values())

PHASE4A_REPORT = {
    "phase": "Phase 4A - Token Representation and Data Alignment",
    "phase1_split_fingerprint_sha256": OBSERVED_FINGERPRINT,
    "split_sizes": {name: len(frame) for name, frame in splits.items()},
    "split_id_intersections": intersections,
    "phase3_root": str(PHASE3_ROOT),
    "phase3_joins": phase3_join_status,
    "motion_token_representation": MOTION_TOKEN_STATISTICS,
    "token_integrity": audit_summary,
    "original_source_alignment": source_alignment_report,
    "checks": checks,
    "ready_to_build_motion_model": READY_TO_BUILD_MOTION_MODEL,
    "next_phase_allowed": "Phase 4B - Model + 32-sample overfit test" if READY_TO_BUILD_MOTION_MODEL else None,
}
write_json(OUTPUT / "phase4_config.json", {
    "phase4a_config": asdict(CFG4),
    "resolved_paths": {"phase1": PHASE1_ROOT, "motion_s": MOTION_S_ROOT, "phase3": PHASE3_ROOT},
    "phase1_fingerprint": OBSERVED_FINGERPRINT,
})
write_json(OUTPUT / "phase4a_verification_report.json", PHASE4A_REPORT)

invalid_display = audit_summary["invalid_ids"] if audit_summary["invalid_ids"] is not None else "UNRESOLVED"
print("\n========================================")
print("PHASE 4A DATA/TOKEN VERIFICATION")
print("========================================")
print("Phase-1 fingerprint:", "PASS" if checks["phase1_fingerprint"] else "FAIL")
print("Train:", len(splits["train"]))
print("Validation:", len(splits["validation"]))
print("Test:", len(splits["test"]))
print("Motion-token source:", MOTION_S_ROOT / "train.csv", "+ locked Phase-1 split CSVs")
print("Token shape convention: [L,T]")
print("RVQ levels:", len(TOKEN_COLS))
print("Codebook sizes:", CODEBOOK_SIZES if CODEBOOK_SIZES is not None else "UNRESOLVED")
print("Mean motion-token length:", length_statistics["mean"])
print("P95 motion-token length:", length_statistics["p95"])
print("Maximum motion-token length:", length_statistics["maximum"])
print("Missing tokens:", audit_summary["missing"])
print("Invalid tokens:", invalid_display)
print("Phase-3 validation join:", "PASS" if checks["phase3_validation_join"] else "FAIL")
print("Phase-3 test join:", "PASS" if checks["phase3_test_join"] else "FAIL")
print("READY TO BUILD MOTION MODEL:", "YES" if READY_TO_BUILD_MOTION_MODEL else "NO")
print("========================================")
print("Individual checks:")
display(pd.Series(checks, name="passed").to_frame())

if not READY_TO_BUILD_MOTION_MODEL:
    failed = [name for name, passed in checks.items() if not passed]
    raise RuntimeError(
        "PHASE 4A GATE FAILED. Do not start Phase 4B. Failed checks: " + ", ".join(failed)
    )
print("PHASE 4A GATE PASSED. Stop here; Phase 4B is the next separately authorized stage.")

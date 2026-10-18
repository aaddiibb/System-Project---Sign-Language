# %% [markdown]
# # Motion-S Text-to-Sign — Phase 1: Dataset Verification and Preprocessing
#
# This notebook establishes the dataset contract **before any model is built**.
# It discovers the attached Kaggle input, inspects tabular metadata and motion/token
# artifacts, validates every training target, measures sequence statistics, and creates
# leakage-safe train/validation/internal-test splits.
#
# **Hard rules enforced here**
#
# - No DWPose or `(T, 128, 3)` assumption.
# - No hard-coded 668-D assumption; the feature dimension is measured.
# - No missing target is replaced with zeros.
# - RVQ layers are never flattened silently.
# - Exact normalized sentences remain in one split.
# - Four provider-incomplete records are quarantined by exact audited ID; no target is fabricated.
# - Any incomplete record outside that reviewed list remains a hard failure.
# - No downstream phase should run unless `PHASE1_READY` is `True`.
#
# The public competition page is useful context, but its advertised values are treated
# only as expectations to verify against the files attached to this notebook.

# %% [markdown]
# ## 1. Environment, reproducibility, and centralized configuration
#
# **What:** Fix random seeds, print the runtime, and keep all Phase 1 choices in one object.
#
# **Why:** Reproducible splits and explicit overrides are essential when Kaggle input mount
# names differ. `data_root=None` enables conservative auto-discovery. If discovery is
# ambiguous, set it to the exact attached dataset directory and rerun. The quarantine list
# contains only IDs already confirmed by exported diagnostics to lack both required targets.
#
# **Output:** Runtime facts and a single `CFG` object. No package installation or internet
# access is needed.

# %%
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class Config:
    seed: int = 42
    kaggle_input: str = "/kaggle/input"
    data_root: Optional[str] = None  # Set only if auto-discovery reports ambiguity.
    output_dir: str = "/kaggle/working/motion_s_phase1"

    # Explicit column overrides. Leave None to infer conservatively.
    id_column: Optional[str] = None
    sentence_column: Optional[str] = None
    gloss_column: Optional[str] = None
    token_columns: Optional[tuple[str, ...]] = None
    rvq_checkpoint: Optional[str] = None  # Optional exact path to an attached checkpoint.

    # These exact records were audited from integrity_report.csv and token_errors.csv.
    # They contain sentence/gloss annotations but neither motion arrays nor RVQ targets.
    # Any new failing ID remains a hard error.
    approved_quarantine_ids: tuple[str, ...] = (
        "1361291",
        "25980",
        "32734",
        "4808283",
    )

    # Only set if a single nested token column is genuinely ambiguous after inspection.
    # Accepted values: "L_T" for [RVQ level, time], "T_L" for [time, RVQ level].
    token_layout: Optional[str] = None

    train_fraction: float = 0.80
    validation_fraction: float = 0.10
    test_fraction: float = 0.10
    near_duplicate_threshold: float = 0.90
    near_duplicate_max_features: int = 50_000

    examples_to_show: int = 5
    numeric_chunk_rows: int = 4096
    full_numeric_scan: bool = True
    # Phase 2 consumes text/gloss, exact splits, paths, and existing token strings; it does
    # not decode RVQ tokens. Keep this False for the Phase-2 export. Set True before any
    # decoding/reconstruction phase, where the fixed provider checkpoint is mandatory.
    require_rvq_checkpoint: bool = False
    require_motion_tokens: bool = True
    strict_shape_validation: bool = True


CFG = Config()
assert math.isclose(
    CFG.train_fraction + CFG.validation_fraction + CFG.test_fraction,
    1.0,
    rel_tol=0,
    abs_tol=1e-9,
)
assert CFG.token_layout in {None, "L_T", "T_L"}

print("Python version:", sys.version.split()[0])
print("Platform:", platform.platform())
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Selected device:", device)
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("GPU name:", torch.cuda.get_device_name(0))
    print(f"GPU memory: {props.total_memory / 2**30:.2f} GiB")
else:
    print("GPU name: not available (Phase 1 is CPU-compatible)")
print("Configuration:")
display(pd.Series(asdict(CFG), name="value").to_frame())

# %% [markdown]
# ## 2. Locate the Motion-S input without assuming its mount name
#
# **What:** Score each first-level Kaggle input directory using filenames and artifact
# types. A valid candidate should contain tabular metadata and motion-related files.
#
# **Why:** Kaggle mount names can change, and selecting the first directory is unsafe when
# checkpoints or other datasets are attached.
#
# **Input:** `/kaggle/input` or `CFG.data_root`.
#
# **Output:** `DATA_ROOT`. Ambiguous discovery raises an error with candidates rather than
# guessing.

# %%
def score_dataset_root(root: Path) -> tuple[int, dict[str, int]]:
    """Score a candidate using names only; this does not assert a representation."""
    counts = Counter()
    for current_root, _, files in os.walk(root):
        for name in files:
            lower = name.lower()
            suffix = Path(name).suffix.lower()
            counts[suffix] += 1
            if lower in {"train.csv", "test.csv", "sample_submission.csv"}:
                counts[lower] += 1
            if any(key in lower for key in ("rvq", "vqvae", "vq_vae", "codebook")):
                counts["rvq_named"] += 1
    score = (
        30 * counts["train.csv"]
        + 10 * counts["test.csv"]
        + 3 * counts[".npy"]
        + 2 * counts[".bvh"]
        + 8 * counts["rvq_named"]
        + 100 * int((root / "train.csv").is_file())
    )
    return score, dict(counts)


def locate_data_root(cfg: Config) -> Path:
    if cfg.data_root is not None:
        root = Path(cfg.data_root)
        if not root.exists():
            raise FileNotFoundError(f"Configured data_root does not exist: {root}")
        return root.resolve()

    input_root = Path(cfg.kaggle_input)
    if not input_root.exists():
        raise FileNotFoundError(
            f"{input_root} does not exist. Run this notebook on Kaggle with the "
            "Motion-S competition data attached, or set CFG.data_root."
        )

    # Kaggle competition data can be nested as /kaggle/input/competitions/<slug>.
    # Prefer the exact directory containing train.csv over a broad ancestor mount.
    train_csv_parents = []
    for current_root, _, files in os.walk(input_root):
        if "train.csv" in files:
            train_csv_parents.append(Path(current_root))
    candidates = list(dict.fromkeys(train_csv_parents))
    if not candidates:
        candidates = [p for p in input_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No attached input directories found below {input_root}")

    scored = []
    for candidate in candidates:
        score, evidence = score_dataset_root(candidate)
        scored.append((score, candidate, evidence))
    scored.sort(key=lambda x: x[0], reverse=True)

    display(pd.DataFrame([
        {"score": score, "path": str(path), **evidence}
        for score, path, evidence in scored
    ]).fillna(0))

    best_score = scored[0][0]
    tied = [item for item in scored if item[0] == best_score]
    if best_score <= 0 or len(tied) != 1:
        raise RuntimeError(
            "Dataset root discovery was ambiguous. Set CFG.data_root to the exact "
            "Motion-S directory after reviewing the candidate table."
        )
    return scored[0][1].resolve()


DATA_ROOT = locate_data_root(CFG)
OUTPUT_DIR = Path(CFG.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Verified candidate DATA_ROOT:", DATA_ROOT)

# %% [markdown]
# ## 3. Complete directory traversal and compact manifest
#
# **What:** The first cell below performs the explicitly requested `os.walk` traversal and
# prints every directory with up to ten filenames. The second creates a compact manifest
# containing every file path, suffix, and byte size.
#
# **Why:** The traversal exposes the real hierarchy; the manifest supports later discovery
# without repeatedly walking approximately tens of thousands of files.
#
# **Expected output:** Many lines for the full traversal, followed by extension counts and
# the largest files. Kaggle may collapse the long cell output in its interface.

# %%
# Required complete traversal. This intentionally does not assume directory names.
for root, dirs, files in os.walk(DATA_ROOT):
    print(root)
    print(files[:10])

# %%
def build_manifest(search_root: Path) -> pd.DataFrame:
    records = []
    for root, _, files in os.walk(search_root):
        root_path = Path(root)
        for filename in files:
            path = root_path / filename
            stat = path.stat()
            records.append({
                "path": str(path),
                "relative_path": str(path.relative_to(search_root)),
                "name": path.name,
                "stem": path.stem,
                "suffix": path.suffix.lower(),
                "size_bytes": int(stat.st_size),
            })
    if not records:
        raise RuntimeError(f"No files discovered under {search_root}")
    return pd.DataFrame.from_records(records)


manifest = build_manifest(DATA_ROOT)
print(f"Discovered {len(manifest):,} files in {manifest['relative_path'].map(lambda x: str(Path(x).parent)).nunique():,} directories")
display(manifest.groupby("suffix").agg(files=("path", "size"), bytes=("size_bytes", "sum")).sort_values("files", ascending=False))
display(manifest.nlargest(20, "size_bytes")[["relative_path", "suffix", "size_bytes"]])

# %% [markdown]
# ## 4. Identify and inspect CSV/metadata tables
#
# **What:** Read every CSV under the selected root as strings, then display `head`,
# `columns`, `shape`, and `info`. Training and external-test tables are selected using both
# filenames and their observed columns.
#
# **Why:** Reading identifiers as strings preserves leading zeros. Keeping tokens as their
# original strings also prevents accidental numeric coercion before their syntax is known.
#
# **Output:** `TRAIN_CSV`, optional `EXTERNAL_TEST_CSV`, `train_df`, and resolved ID,
# sentence, and gloss columns. Ambiguous columns cause a hard stop with an override hint.

# %%
csv_paths = [Path(p) for p in manifest.loc[manifest["suffix"].eq(".csv"), "path"]]
if not csv_paths:
    raise FileNotFoundError(f"No CSV files found below {DATA_ROOT}")

csv_tables: dict[Path, pd.DataFrame] = {}
for csv_path in sorted(csv_paths):
    print("\n" + "=" * 100)
    print("CSV:", csv_path)
    table = pd.read_csv(csv_path, dtype=str, low_memory=False)
    csv_tables[csv_path] = table
    display(table.head())
    print("Columns:", table.columns.tolist())
    print("Shape:", table.shape)
    table.info()


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def table_role_score(path: Path, df: pd.DataFrame, role: str) -> int:
    name = path.name.lower()
    cols = [normalized_name(c) for c in df.columns]
    has_sentence = any(c in {"sentence", "text", "translation", "english"} for c in cols)
    has_gloss = any("gloss" in c for c in cols)
    has_token = any(any(k in c for k in ("token", "codebook", "rvq", "layer")) for c in cols)
    if role == "train":
        return 20 * (name == "train.csv") + 6 * ("train" in name) + 4 * has_sentence + 4 * has_gloss + 5 * has_token
    return 20 * (name == "test.csv") + 6 * ("test" in name and "submission" not in name) + 4 * has_sentence + 2 * has_gloss - 3 * has_token


def select_unique_table(role: str) -> Optional[Path]:
    pool = list(csv_tables.items())
    if role == "test":
        pool = [
            (path, df) for path, df in pool
            if "submission" not in path.name.lower()
            and (
                "test" in path.name.lower()
                or (
                    "train" not in path.name.lower()
                    and not any(
                        any(term in normalized_name(col) for term in ("motiontoken", "rvqtoken", "codebook", "tokenlayer"))
                        for col in df.columns
                    )
                )
            )
        ]
    ranked = sorted(
        ((table_role_score(path, df, role), path) for path, df in pool),
        key=lambda item: item[0],
        reverse=True,
    )
    if not ranked or ranked[0][0] <= 0:
        return None
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        raise RuntimeError(f"Ambiguous {role} CSV candidates: {ranked[:5]}")
    return ranked[0][1]


TRAIN_CSV = select_unique_table("train")
EXTERNAL_TEST_CSV = select_unique_table("test")
if TRAIN_CSV is None:
    raise RuntimeError("Could not identify a training CSV from observed files and columns")

train_df = csv_tables[TRAIN_CSV].copy()
external_test_df = csv_tables[EXTERNAL_TEST_CSV].copy() if EXTERNAL_TEST_CSV else None
print("Selected training CSV:", TRAIN_CSV)
print("Selected external test CSV:", EXTERNAL_TEST_CSV)


def resolve_column(
    df: pd.DataFrame,
    role: str,
    override: Optional[str],
    exact_aliases: tuple[str, ...],
    contains_aliases: tuple[str, ...] = (),
) -> str:
    if override is not None:
        if override not in df.columns:
            raise KeyError(f"Configured {role} column {override!r} is absent: {df.columns.tolist()}")
        return override

    normalized = {column: normalized_name(column) for column in df.columns}
    exact = [c for c, n in normalized.items() if n in exact_aliases]
    candidates = exact or [c for c, n in normalized.items() if any(alias in n for alias in contains_aliases)]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one {role} column, found {candidates}. "
            f"Available columns: {df.columns.tolist()}. Set the corresponding CFG override."
        )
    return candidates[0]


ID_COL = resolve_column(train_df, "ID", CFG.id_column, ("id", "sampleid", "motionid", "uid"), ("id",))
SENTENCE_COL = resolve_column(
    train_df, "sentence", CFG.sentence_column,
    ("sentence", "text", "translation", "english", "spokenlanguage"),
    ("sentence", "translation"),
)
GLOSS_COL = resolve_column(train_df, "gloss", CFG.gloss_column, ("gloss", "glosses"), ("gloss",))

for col in (ID_COL, SENTENCE_COL, GLOSS_COL):
    train_df[col] = train_df[col].astype("string")

print({"id": ID_COL, "sentence": SENTENCE_COL, "gloss": GLOSS_COL})
display(train_df[[ID_COL, SENTENCE_COL, GLOSS_COL]].head(CFG.examples_to_show))

# %% [markdown]
# ## 5. Inspect text metadata files and cross-check annotations
#
# **What:** Show real `metadata.txt`-like examples and compare parsed `SENTENCE:`/`GLOSS:`
# values with CSV rows whenever IDs can be matched.
#
# **Why:** The CSV and per-sample metadata are independent evidence for the linguistic
# pairing. Disagreement is reported; it is never silently overwritten.

# %%
text_metadata_paths = [
    Path(p) for p in manifest.loc[
        manifest["suffix"].isin([".txt", ".text", ".meta"])
        & manifest["name"].str.lower().str.contains("metadata|meta", regex=True),
        "path",
    ]
]


def parse_metadata_file(path: Path) -> dict[str, str]:
    content = path.read_text(encoding="utf-8", errors="replace")
    parsed: dict[str, str] = {}
    for line in content.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            parsed[key.strip().lower()] = value.strip()
    return parsed


print(f"Metadata-like text files: {len(text_metadata_paths):,}")
for path in text_metadata_paths[: CFG.examples_to_show]:
    print("\nPATH:", path)
    print(path.read_text(encoding="utf-8", errors="replace")[:1000])

csv_by_id = train_df.set_index(ID_COL, drop=False)
metadata_mismatches = []
metadata_checked = 0
for path in text_metadata_paths:
    sample_id = path.parent.name if path.parent.name in csv_by_id.index else path.stem
    if sample_id not in csv_by_id.index:
        continue
    meta = parse_metadata_file(path)
    row = csv_by_id.loc[sample_id]
    if isinstance(row, pd.DataFrame):
        metadata_mismatches.append({"id": sample_id, "field": "duplicate_csv_id"})
        continue
    metadata_checked += 1
    for field, col in (("sentence", SENTENCE_COL), ("gloss", GLOSS_COL)):
        if field in meta and str(row[col]).strip() != meta[field].strip():
            metadata_mismatches.append({
                "id": sample_id,
                "field": field,
                "csv": str(row[col]),
                "metadata": meta[field],
                "path": str(path),
            })

print("Metadata rows cross-checked:", metadata_checked)
print("Metadata mismatches:", len(metadata_mismatches))
if metadata_mismatches:
    display(pd.DataFrame(metadata_mismatches).head(20))

# %% [markdown]
# ## 6. Discover and inspect raw motion arrays and BVH files
#
# **What:** Match artifacts to sample IDs using exact filename/parent-name evidence, inspect
# array type, shape, dtype, range, and BVH headers.
#
# **Why:** This determines the actual representation. A two-dimensional feature matrix is
# kept as `[T_motion, D_motion]`; it is never reshaped into joints. If joint coordinates are
# needed later, Phase 5 must use documented SMPL-X/BVH conversion code.
#
# **Important shapes:** Shapes printed in this section are observed facts. No expected
# feature dimension is coded into validation.

# %%
ARRAY_SUFFIXES = {".npy", ".npz"}
array_paths = [Path(p) for p in manifest.loc[manifest["suffix"].isin(ARRAY_SUFFIXES), "path"]]
bvh_paths = [Path(p) for p in manifest.loc[manifest["suffix"].eq(".bvh"), "path"]]


def build_exact_artifact_index(paths: Iterable[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        index[path.stem].append(path)
        if path.parent.name != path.stem:
            index[path.parent.name].append(path)
    return index


array_index = build_exact_artifact_index(array_paths)


def path_kind_score(path: Path, kind: str) -> int:
    text = str(path).lower().replace("-", "_")
    token_terms = ("token", "tokens", "codebook", "codes", "rvq", "quant")
    motion_terms = ("motion", "feature", "smpl", "pose", "train")
    if kind == "token":
        return 5 * sum(term in text for term in token_terms) - 4 * sum(term in text for term in ("checkpoint", "weight", "model"))
    return 3 * sum(term in text for term in motion_terms) - 6 * sum(term in text for term in token_terms)


def resolve_artifact(sample_id: str, kind: str) -> tuple[Optional[Path], list[Path]]:
    candidates = list(dict.fromkeys(array_index.get(str(sample_id), [])))
    if not candidates:
        return None, []
    ranked = sorted(candidates, key=lambda p: (path_kind_score(p, kind), p.name == f"{sample_id}.npy"), reverse=True)
    top_score = path_kind_score(ranked[0], kind)
    tied = [p for p in ranked if path_kind_score(p, kind) == top_score]
    exact_named = [p for p in tied if p.stem == str(sample_id)]
    if len(exact_named) == 1:
        return exact_named[0], candidates
    if len(tied) == 1:
        return tied[0], candidates
    return None, candidates  # Ambiguity is an integrity failure, not a guess.


def open_array(path: Path) -> tuple[Any, Optional[str]]:
    """Open without pickle; return the array and NPZ member name if applicable."""
    if path.suffix.lower() == ".npy":
        return np.load(path, mmap_mode="r", allow_pickle=False), None
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)
        if len(keys) != 1:
            raise ValueError(f"NPZ has {len(keys)} arrays, so the motion member is ambiguous: {path} -> {keys}")
        key = keys[0]
        # NPZ members cannot remain lazy after the archive closes. Copy only this member;
        # ordinary NPY motion files remain memory-mapped.
        return np.asarray(archive[key]).copy(), key


def describe_array(path: Path) -> dict[str, Any]:
    array, member = open_array(path)
    try:
        values = np.asarray(array)
        numeric = bool(np.issubdtype(values.dtype, np.number))
        info = {
            "path": str(path),
            "member": member,
            "python_type": type(array).__name__,
            "shape": tuple(int(x) for x in values.shape),
            "ndim": int(values.ndim),
            "dtype": str(values.dtype),
            "min": float(np.nanmin(values)) if values.size and numeric else None,
            "max": float(np.nanmax(values)) if values.size and numeric else None,
            "has_nan": bool(numeric and np.isnan(values).any()),
            "has_inf": bool(numeric and np.isinf(values).any()),
        }
        return info
    finally:
        base = getattr(array, "base", None)
        if hasattr(base, "close"):
            base.close()


resolved_motion_preview = []
for sample_id in train_df[ID_COL].dropna().astype(str).head(50):
    path, candidates = resolve_artifact(sample_id, "motion")
    if path is not None and path not in resolved_motion_preview:
        resolved_motion_preview.append(path)
    if len(resolved_motion_preview) >= CFG.examples_to_show:
        break

print(f"Array files: {len(array_paths):,}; BVH files: {len(bvh_paths):,}")
if not resolved_motion_preview:
    print("No unambiguous sample-linked motion arrays were found in the first 50 IDs.")
for path in resolved_motion_preview:
    print(json.dumps(describe_array(path), indent=2))


def inspect_bvh_header(path: Path, max_lines: int = 5000) -> dict[str, Any]:
    joints = []
    channel_count = 0
    frames = None
    frame_time = None
    motion_seen = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            stripped = line.strip()
            if stripped.startswith(("ROOT ", "JOINT ")):
                joints.append(stripped.split(maxsplit=1)[1])
            elif stripped.startswith("CHANNELS "):
                parts = stripped.split()
                channel_count += int(parts[1])
            elif stripped == "MOTION":
                motion_seen = True
            elif stripped.lower().startswith("frames:"):
                frames = int(stripped.split(":", 1)[1].strip())
            elif stripped.lower().startswith("frame time:"):
                frame_time = float(stripped.split(":", 1)[1].strip())
                break
            if line_number >= max_lines:
                break
    return {
        "path": str(path),
        "motion_section": motion_seen,
        "joint_count_from_hierarchy": len(joints),
        "channel_count": channel_count,
        "frames": frames,
        "frame_time": frame_time,
        "first_joints": joints[:15],
    }


for path in bvh_paths[: CFG.examples_to_show]:
    print(json.dumps(inspect_bvh_header(path), indent=2))

# %% [markdown]
# ## 7. Discover and analyze layered motion tokens
#
# **What:** Detect token-bearing CSV columns, parse their real syntax, and infer whether a
# nested representation is `[L, T]` or `[T, L]` only when the evidence is unambiguous.
# Per-layer ranges, observed vocabulary sizes, frequency, and sequence lengths are measured.
#
# **Why:** Residual vector quantization has semantic layers. Flattening `[L, T]` into one
# sequence destroys level/time structure.
#
# **Input:** One column per layer, a single nested token column, or exact-ID token files.
#
# **Output:** Canonical tokens in shape `[L, T_token]` for analysis only. No BOS/EOS/PAD IDs
# are invented; discovered config/code evidence is printed separately.

# %%
def discover_token_columns(df: pd.DataFrame, override: Optional[tuple[str, ...]]) -> list[str]:
    if override is not None:
        missing = [c for c in override if c not in df.columns]
        if missing:
            raise KeyError(f"Configured token columns absent from CSV: {missing}")
        return list(override)
    result = []
    for col in df.columns:
        normalized = normalized_name(col)
        is_base = normalized in {"basetoken", "basetokens", "basecode", "basecodes"}
        is_residual = bool(re.fullmatch(r"residual(?:token|tokens|code|codes)?\d+", normalized))
        is_explicit_token = any(term in normalized for term in ("motiontoken", "rvqtoken", "codebook", "tokenlayer", "layertoken"))
        if is_base or is_residual or is_explicit_token:
            if not any(term in normalized for term in ("length", "count", "size", "mask")):
                result.append(col)
    if not result:
        broader = [
            c for c in df.columns
            if any(term in normalized_name(c) for term in ("token", "rvq", "code"))
            and not any(term in normalized_name(c) for term in ("length", "count", "size", "mask"))
        ]
        result = broader
    def token_column_order(column: str) -> tuple[int, int, str]:
        normalized = normalized_name(column)
        if normalized.startswith("base"):
            return (0, 0, normalized)
        residual_match = re.search(r"residual(?:token|tokens|code|codes)?(\d+)", normalized)
        if residual_match:
            return (1, int(residual_match.group(1)), normalized)
        return (2, 0, normalized)

    return sorted(result, key=token_column_order)


TOKEN_COLS = discover_token_columns(train_df, CFG.token_columns)
print("Detected token columns:", TOKEN_COLS)
if TOKEN_COLS:
    display(train_df[[ID_COL] + TOKEN_COLS].head(CFG.examples_to_show))


def parse_token_value(value: Any) -> np.ndarray:
    """Parse one token field while preserving nesting; never flatten here."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("missing token value")
    if isinstance(value, np.ndarray):
        parsed = value
    elif isinstance(value, (list, tuple)):
        parsed = np.asarray(value)
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            raise ValueError("missing token value")
        parsed_obj = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed_obj = parser(text)
                break
            except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
                pass
        if parsed_obj is None:
            # A plain one-layer sequence such as "12 4 9" or "12,4,9".
            if any(mark in text for mark in ("[", "]", "{", "}", "|", ";")):
                raise ValueError(f"unrecognized structured token syntax: {text[:120]!r}")
            pieces = [piece for piece in re.split(r"[\s,]+", text) if piece]
            parsed_obj = [int(piece) for piece in pieces]
        parsed = np.asarray(parsed_obj)

    if parsed.dtype == object:
        raise ValueError("ragged/object token structure")
    if parsed.ndim not in (1, 2):
        raise ValueError(f"tokens must be 1-D or 2-D, got shape {parsed.shape}")
    if parsed.size == 0:
        raise ValueError("empty token sequence")
    numeric = parsed.astype(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("tokens contain NaN or Inf")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("tokens contain non-integer values")
    integers = numeric.astype(np.int64)
    if (integers < 0).any():
        raise ValueError("tokens contain negative IDs")
    return integers


def infer_nested_layout(shapes: list[tuple[int, int]], override: Optional[str]) -> str:
    if override is not None:
        return override
    if not shapes:
        raise RuntimeError("No valid 2-D token shapes available for layout inference")
    first_dims = {shape[0] for shape in shapes}
    second_dims = {shape[1] for shape in shapes}
    first_small_constant = len(first_dims) == 1 and next(iter(first_dims)) <= 32
    second_small_constant = len(second_dims) == 1 and next(iter(second_dims)) <= 32
    if first_small_constant and not second_small_constant:
        return "L_T"
    if second_small_constant and not first_small_constant:
        return "T_L"
    raise RuntimeError(
        "Nested token layout is ambiguous from observed shapes. "
        f"First dimensions={sorted(first_dims)[:20]}, second dimensions={sorted(second_dims)[:20]}. "
        "Inspect the provider's decoder/code and set CFG.token_layout to 'L_T' or 'T_L'."
    )


token_parse_errors: list[dict[str, Any]] = []
raw_token_records: list[tuple[str, list[np.ndarray]]] = []
if TOKEN_COLS:
    for _, row in train_df.iterrows():
        sample_id = str(row[ID_COL])
        layers = []
        try:
            for col in TOKEN_COLS:
                layers.append(parse_token_value(row[col]))
            raw_token_records.append((sample_id, layers))
        except Exception as exc:
            token_parse_errors.append({"id": sample_id, "error": repr(exc)})

TOKEN_LAYOUT = None
if raw_token_records and len(TOKEN_COLS) == 1:
    two_d_shapes = [tuple(parts[0].shape) for _, parts in raw_token_records if parts[0].ndim == 2]
    one_d_count = sum(parts[0].ndim == 1 for _, parts in raw_token_records)
    if two_d_shapes and one_d_count:
        raise RuntimeError("The same token column mixes 1-D and 2-D structures")
    if two_d_shapes:
        TOKEN_LAYOUT = infer_nested_layout(two_d_shapes, CFG.token_layout)
    elif one_d_count:
        TOKEN_LAYOUT = "L_T"  # One observed level; canonical shape becomes [1, T].
elif raw_token_records and len(TOKEN_COLS) > 1:
    if any(layer.ndim != 1 for _, layers in raw_token_records for layer in layers):
        raise RuntimeError("With one CSV column per RVQ level, every parsed column must be 1-D")
    TOKEN_LAYOUT = "L_T"


def canonicalize_csv_tokens(parts: list[np.ndarray]) -> np.ndarray:
    """Return [L, T] only after layout has been established."""
    if len(parts) > 1:
        lengths = {len(layer) for layer in parts}
        if len(lengths) != 1:
            raise ValueError(f"per-layer token lengths disagree: {sorted(lengths)}")
        return np.stack(parts, axis=0)
    arr = parts[0]
    if arr.ndim == 1:
        return arr[None, :]
    if TOKEN_LAYOUT == "L_T":
        return arr
    if TOKEN_LAYOUT == "T_L":
        return arr.T
    raise RuntimeError("Token layout has not been established")


token_by_id: dict[str, np.ndarray] = {}
for sample_id, parts in raw_token_records:
    try:
        token_by_id[sample_id] = canonicalize_csv_tokens(parts)
    except Exception as exc:
        token_parse_errors.append({"id": sample_id, "error": repr(exc)})

if not TOKEN_COLS:
    token_named_arrays = [
        path for path in array_paths
        if path_kind_score(path, "token") > path_kind_score(path, "motion")
    ]
    print(f"No token CSV columns found; token-named array candidates: {len(token_named_arrays)}")
    # File-based tokens are deliberately inspected but not guessed when multiple arrays/members exist.
    for sample_id in train_df[ID_COL].dropna().astype(str):
        path, candidates = resolve_artifact(sample_id, "token")
        if path is None:
            token_parse_errors.append({"id": sample_id, "error": "missing or ambiguous token file", "candidates": [str(p) for p in candidates]})
            continue
        try:
            raw, _ = open_array(path)
            parsed = parse_token_value(np.asarray(raw))
            if parsed.ndim == 1:
                parsed = parsed[None, :]
            elif CFG.token_layout == "T_L":
                parsed = parsed.T
            elif CFG.token_layout != "L_T":
                raise RuntimeError("Set CFG.token_layout after inspecting file-based 2-D tokens")
            token_by_id[sample_id] = parsed
        except Exception as exc:
            token_parse_errors.append({"id": sample_id, "error": repr(exc), "path": str(path)})

print("Canonical token layout:", "[L, T]" if token_by_id else "not established")
print("Token parse/load errors:", len(token_parse_errors))
if token_parse_errors:
    token_error_preview = pd.DataFrame(token_parse_errors).head(20)
    display(token_error_preview)
    print(token_error_preview.to_string(index=False))
for sample_id, tokens in list(token_by_id.items())[: CFG.examples_to_show]:
    print({
        "id": sample_id,
        "python_type": type(tokens).__name__,
        "shape_L_T": tuple(tokens.shape),
        "dtype": str(tokens.dtype),
        "min": int(tokens.min()),
        "max": int(tokens.max()),
    })


def summarize_tokens(tokens: dict[str, np.ndarray]) -> tuple[dict[str, Any], list[Counter]]:
    if not tokens:
        return {}, []
    level_counts = Counter(arr.shape[0] for arr in tokens.values())
    if len(level_counts) != 1:
        raise RuntimeError(f"Inconsistent RVQ level counts: {dict(level_counts)}")
    levels = next(iter(level_counts))
    frequencies = [Counter() for _ in range(levels)]
    lengths = []
    for arr in tokens.values():
        assert arr.ndim == 2 and arr.shape[0] == levels
        lengths.append(int(arr.shape[1]))
        for level in range(levels):
            frequencies[level].update(arr[level].tolist())
    level_summary = []
    for level, counts in enumerate(frequencies):
        ids = np.asarray(list(counts.keys()), dtype=np.int64)
        level_summary.append({
            "level": level,
            "observed_min_id": int(ids.min()),
            "observed_max_id": int(ids.max()),
            "observed_vocabulary": int(len(ids)),
            "most_common": counts.most_common(10),
            "rare_once": int(sum(freq == 1 for freq in counts.values())),
        })
    length_array = np.asarray(lengths)
    summary = {
        "examples": len(tokens),
        "rvq_levels": levels,
        "token_length_mean": float(length_array.mean()),
        "token_length_median": float(np.median(length_array)),
        "token_length_p95": float(np.percentile(length_array, 95)),
        "token_length_p99": float(np.percentile(length_array, 99)),
        "token_length_max": int(length_array.max()),
        "per_level": level_summary,
    }
    return summary, frequencies


token_summary, token_frequencies = summarize_tokens(token_by_id)
print(json.dumps(token_summary, indent=2))

if token_frequencies:
    fig, axes = plt.subplots(len(token_frequencies), 1, figsize=(14, 3.5 * len(token_frequencies)), squeeze=False)
    for level, counts in enumerate(token_frequencies):
        ids = np.asarray(sorted(counts))
        freqs = np.asarray([counts[int(token_id)] for token_id in ids])
        axes[level, 0].bar(ids, freqs, width=1.0)
        axes[level, 0].set_title(f"Observed RVQ level {level} token frequencies")
        axes[level, 0].set_xlabel("Token ID (observed; special-token meaning not assumed)")
        axes[level, 0].set_ylabel("Count")
        axes[level, 0].set_yscale("log")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 8. Locate and safely inspect the supplied RVQ-VAE checkpoint/code
#
# **What:** Search all attached Kaggle inputs—not only the competition directory—for RVQ,
# VQ-VAE, decoder, quantizer, and codebook artifacts. Checkpoints are opened on CPU using
# `weights_only=True` where the installed PyTorch supports it, and only key/tensor-shape
# summaries are printed.
#
# **Why:** Later decoding must use the supplied fixed codebook. The observed state-dict keys
# and companion config/code establish its interface. Phase 1 does not instantiate or modify
# the decoder.
#
# **Output:** Candidate paths, checkpoint tensor shapes, and relevant code/config excerpts.
# A missing required checkpoint keeps `PHASE1_READY=False`.

# %%
ALL_INPUT_ROOT = Path(CFG.kaggle_input)
all_input_manifest = build_manifest(ALL_INPUT_ROOT)
checkpoint_suffixes = {".pt", ".pth", ".ckpt", ".safetensors"}
checkpoint_terms = ("rvq", "vqvae", "vq_vae", "codebook", "quantizer")
checkpoint_candidates = [
    Path(row.path)
    for row in all_input_manifest.itertuples()
    if row.suffix in checkpoint_suffixes
    and any(term in row.relative_path.lower() for term in checkpoint_terms)
]
if CFG.rvq_checkpoint is not None:
    configured_checkpoint = Path(CFG.rvq_checkpoint)
    if not configured_checkpoint.exists():
        raise FileNotFoundError(f"Configured RVQ checkpoint does not exist: {configured_checkpoint}")
    checkpoint_candidates.insert(0, configured_checkpoint)
checkpoint_candidates = list(dict.fromkeys(checkpoint_candidates))

code_config_candidates = []
for row in all_input_manifest.itertuples():
    if row.suffix not in {".py", ".json", ".yaml", ".yml", ".toml"}:
        continue
    path = Path(row.path)
    path_match = any(
        term in row.relative_path.lower()
        for term in ("rvq", "vqvae", "vq_vae", "codebook", "quantizer", "decoder")
    )
    content_match = False
    if row.suffix == ".py" and row.size_bytes <= 5_000_000:
        content = path.read_text(encoding="utf-8", errors="replace")
        content_match = bool(re.search(r"RVQ|ResidualVQ|VQ.?VAE|codebook|quantizer", content, flags=re.IGNORECASE))
    if path_match or content_match:
        code_config_candidates.append(path)
code_config_candidates = list(dict.fromkeys(code_config_candidates))

print("RVQ/VQ checkpoint candidates:")
for path in checkpoint_candidates:
    print(" -", path)
print("Relevant code/config candidates:")
for path in code_config_candidates[:50]:
    print(" -", path)


def load_checkpoint_safely(path: Path) -> Any:
    if path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("safetensors checkpoint found but safetensors is unavailable") from exc
        return load_file(str(path), device="cpu")
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=True)


def tensor_inventory(obj: Any, prefix: str = "", limit: int = 200) -> list[dict[str, Any]]:
    rows = []
    stack = [(prefix, obj)]
    while stack and len(rows) < limit:
        name, value = stack.pop()
        if torch.is_tensor(value):
            rows.append({"key": name, "shape": tuple(value.shape), "dtype": str(value.dtype)})
        elif isinstance(value, dict):
            for key, child in reversed(list(value.items())):
                stack.append((f"{name}.{key}" if name else str(key), child))
    return rows


checkpoint_inspection = []
codebook_tensor_evidence = []
for checkpoint_path in checkpoint_candidates:
    try:
        checkpoint = load_checkpoint_safely(checkpoint_path)
        inventory = tensor_inventory(checkpoint)
        checkpoint_inspection.append({
            "path": str(checkpoint_path),
            "top_level_type": type(checkpoint).__name__,
            "top_level_keys": list(checkpoint.keys())[:50] if isinstance(checkpoint, dict) else None,
            "tensor_count_shown": len(inventory),
            "tensor_inventory": inventory,
        })
        print("\nCheckpoint:", checkpoint_path)
        if isinstance(checkpoint, dict):
            print("Top-level keys:", list(checkpoint.keys())[:50])
        display(pd.DataFrame(inventory).head(100))
        for item in inventory:
            if any(term in item["key"].lower() for term in ("codebook", "quantizer", "embedding", "embed")):
                codebook_tensor_evidence.append({"checkpoint": str(checkpoint_path), **item})
        del checkpoint
    except Exception as exc:
        checkpoint_inspection.append({"path": str(checkpoint_path), "error": repr(exc)})
        print("Could not inspect checkpoint safely:", checkpoint_path, repr(exc))

special_token_evidence = []
for path in code_config_candidates[:10]:
    print("\n" + "=" * 100)
    print(path)
    content = path.read_text(encoding="utf-8", errors="replace")
    print(content[:5000])
    for line_number, line in enumerate(content.splitlines(), start=1):
        if re.search(r"\b(pad|padding|bos|eos|sos|start_token|end_token)\b", line, flags=re.IGNORECASE):
            special_token_evidence.append({
                "path": str(path),
                "line": line_number,
                "text": line.strip()[:500],
            })

print("Special-token declarations found in provider code/config:")
if special_token_evidence:
    display(pd.DataFrame(special_token_evidence).head(100))
else:
    print(
        "None located. Dataset RVQ IDs are therefore treated as codebook IDs only; "
        "Phase 4 must add non-colliding BOS/EOS/PAD IDs explicitly and document them."
    )
print("Codebook/quantizer-like checkpoint tensors (shape evidence, not guessed semantics):")
if codebook_tensor_evidence:
    display(pd.DataFrame(codebook_tensor_evidence))
else:
    print("None identified by state-dict key name; inspect the full inventory/provider code above.")

# %% [markdown]
# ## 9. Full dataset-integrity validation
#
# **What:** Validate every training row and every linked motion target. Numeric arrays are
# opened lazily/memory-mapped and scanned in row chunks so the full dataset is not loaded
# into RAM.
#
# **Why:** Missing, ambiguous, malformed, NaN, infinite, or zero-only targets must stop the
# pipeline. There is no zero substitution.
#
# **Expected input shape:** Determined from files. For a consistent feature representation,
# all valid arrays should normally be rank 2 with one stable final feature dimension, but
# this cell reports what it sees before enforcing consistency.
#
# **Output:** One row per training example in `integrity_df` and the required aggregate
# counts. Full scanning can take several minutes for a large corpus.

# %%
def scan_numeric_array(path: Path, chunk_rows: int, full_scan: bool) -> dict[str, Any]:
    array, member = open_array(path)
    try:
        arr = np.asarray(array)
        result = {
            "motion_path": str(path),
            "npz_member": member,
            "shape": tuple(int(x) for x in arr.shape),
            "ndim": int(arr.ndim),
            "dtype": str(arr.dtype),
            "num_values": int(arr.size),
            "has_nan": False,
            "has_inf": False,
            "zero_only": False,
            "numeric": bool(np.issubdtype(arr.dtype, np.number)),
        }
        if not result["numeric"] or arr.size == 0:
            return result

        saw_nonzero = False
        if full_scan and arr.ndim >= 1:
            for start in range(0, arr.shape[0], chunk_rows):
                chunk = np.asarray(arr[start : start + chunk_rows])
                result["has_nan"] = result["has_nan"] or bool(np.isnan(chunk).any())
                result["has_inf"] = result["has_inf"] or bool(np.isinf(chunk).any())
                saw_nonzero = saw_nonzero or bool(np.any(chunk != 0))
        else:
            sample = arr if arr.ndim == 0 else arr[: min(arr.shape[0], chunk_rows)]
            result["has_nan"] = bool(np.isnan(sample).any())
            result["has_inf"] = bool(np.isinf(sample).any())
            saw_nonzero = bool(np.any(sample != 0))
        result["zero_only"] = not saw_nonzero
        return result
    finally:
        base = getattr(array, "base", None)
        if hasattr(base, "close"):
            base.close()


integrity_records = []
started = time.perf_counter()
for row_number, row in train_df.iterrows():
    sample_id = str(row[ID_COL])
    motion_path, candidates = resolve_artifact(sample_id, "motion")
    record: dict[str, Any] = {
        "row_index": int(row_number),
        "id": sample_id,
        "sentence_missing": pd.isna(row[SENTENCE_COL]) or not str(row[SENTENCE_COL]).strip(),
        "gloss_missing": pd.isna(row[GLOSS_COL]) or not str(row[GLOSS_COL]).strip(),
        "motion_missing": motion_path is None and not candidates,
        "motion_ambiguous": motion_path is None and bool(candidates),
        "motion_candidates": [str(path) for path in candidates],
        "token_missing": sample_id not in token_by_id,
        "token_shape": tuple(token_by_id[sample_id].shape) if sample_id in token_by_id else None,
        "motion_path": str(motion_path) if motion_path is not None else None,
        "npz_member": None,
        "shape": None,
        "ndim": None,
        "dtype": None,
        "num_values": None,
        "numeric": None,
        "has_nan": None,
        "has_inf": None,
        "zero_only": None,
        "scan_error": None,
    }
    if motion_path is not None:
        try:
            record.update(scan_numeric_array(
                motion_path,
                chunk_rows=CFG.numeric_chunk_rows,
                full_scan=CFG.full_numeric_scan,
            ))
        except Exception as exc:
            record["scan_error"] = repr(exc)
    integrity_records.append(record)
    if (len(integrity_records) % 1000) == 0:
        print(f"Validated {len(integrity_records):,}/{len(train_df):,} rows...")

integrity_df = pd.DataFrame(integrity_records)
elapsed = time.perf_counter() - started
print(f"Integrity scan completed in {elapsed / 60:.2f} minutes")


def nullable_bool(column: str) -> pd.Series:
    """Convert an object/nullable column to a warning-free boolean mask."""
    return integrity_df[column].astype("boolean").fillna(False).astype(bool)


numeric_mask = nullable_bool("numeric")
nan_mask = nullable_bool("has_nan")
inf_mask = nullable_bool("has_inf")
zero_only_mask = nullable_bool("zero_only")

valid_shape_rows = integrity_df[
    integrity_df["scan_error"].isna()
    & integrity_df["motion_missing"].eq(False)
    & integrity_df["motion_ambiguous"].eq(False)
]
shape_counts = valid_shape_rows["shape"].map(str).value_counts()
ndim_counts = valid_shape_rows["ndim"].value_counts(dropna=False)
feature_dim_counts = valid_shape_rows.loc[valid_shape_rows["ndim"].eq(2), "shape"].map(lambda shape: shape[-1]).value_counts()

EXPECTED_MOTION_NDIM = None
EXPECTED_FEATURE_DIM = None
if len(ndim_counts) == 1:
    EXPECTED_MOTION_NDIM = int(ndim_counts.index[0])
if EXPECTED_MOTION_NDIM == 2 and len(feature_dim_counts) == 1:
    EXPECTED_FEATURE_DIM = int(feature_dim_counts.index[0])

integrity_df["invalid_shape"] = (
    integrity_df["shape"].isna()
    | (integrity_df["ndim"] != EXPECTED_MOTION_NDIM)
    | integrity_df["num_values"].fillna(0).eq(0)
    | ~numeric_mask
)
integrity_df["invalid_shape"] |= integrity_df["shape"].map(
    lambda shape: shape is None or any(int(dimension) <= 0 for dimension in shape)
)
if EXPECTED_FEATURE_DIM is not None:
    integrity_df["invalid_shape"] |= integrity_df["shape"].map(
        lambda shape: shape is not None and len(shape) == 2 and shape[-1] != EXPECTED_FEATURE_DIM
    )

integrity_summary = {
    "total_examples": int(len(train_df)),
    "successfully_loaded_motion": int((integrity_df["motion_path"].notna() & integrity_df["scan_error"].isna()).sum()),
    "missing_motion": int(integrity_df["motion_missing"].sum()),
    "ambiguous_motion": int(integrity_df["motion_ambiguous"].sum()),
    "missing_tokens": int(integrity_df["token_missing"].sum()),
    "token_parse_or_load_errors": int(len(token_parse_errors)),
    "invalid_shapes": int(integrity_df["invalid_shape"].sum()),
    "scan_errors": int(integrity_df["scan_error"].notna().sum()),
    "non_numeric_samples": int((integrity_df["motion_path"].notna() & ~numeric_mask).sum()),
    "empty_arrays": int((integrity_df["motion_path"].notna() & integrity_df["num_values"].fillna(0).eq(0)).sum()),
    "nan_samples": int(nan_mask.sum()),
    "inf_samples": int(inf_mask.sum()),
    "zero_only_samples": int(zero_only_mask.sum()),
    "missing_sentence": int(integrity_df["sentence_missing"].sum()),
    "missing_gloss": int(integrity_df["gloss_missing"].sum()),
    "duplicate_ids": int(train_df[ID_COL].duplicated(keep=False).sum()),
    "metadata_mismatches": int(len(metadata_mismatches)),
    "observed_motion_ndim": EXPECTED_MOTION_NDIM,
    "observed_feature_dim": EXPECTED_FEATURE_DIM,
    "full_numeric_scan": CFG.full_numeric_scan,
}

print("Observed rank counts:", ndim_counts.to_dict())
print("Observed 2-D feature dimensions:", feature_dim_counts.to_dict())
print("Most common exact shapes:")
display(shape_counts.head(20).rename("count").to_frame())
print("Required integrity report:")
display(pd.Series(integrity_summary, name="value").to_frame())

bad_mask = (
    integrity_df["motion_missing"]
    | integrity_df["motion_ambiguous"]
    | integrity_df["token_missing"]
    | integrity_df["invalid_shape"]
    | integrity_df["scan_error"].notna()
    | nan_mask
    | inf_mask
    | zero_only_mask
    | integrity_df["sentence_missing"]
    | integrity_df["gloss_missing"]
)
if bad_mask.any():
    print("First failing rows:")
    failing_rows = integrity_df.loc[bad_mask].head(50)
    display(failing_rows)
    # Plain text is retained in Kaggle logs even when rich DataFrame output is omitted.
    print(failing_rows.to_string(index=False))

# Apply only the exact, previously reviewed quarantine list. This is intentionally safer
# than a generic "drop every bad row" rule: newly discovered corruption cannot disappear.
configured_quarantine_ids = set(map(str, CFG.approved_quarantine_ids))
known_train_ids = set(train_df[ID_COL].astype(str))
all_bad_ids = set(integrity_df.loc[bad_mask, "id"].astype(str))
unknown_quarantine_ids = configured_quarantine_ids - known_train_ids
approved_but_now_valid_ids = configured_quarantine_ids - all_bad_ids
QUARANTINED_IDS = configured_quarantine_ids & all_bad_ids
UNAPPROVED_BAD_IDS = all_bad_ids - configured_quarantine_ids

quarantine_integrity = integrity_df[integrity_df["id"].isin(QUARANTINED_IDS)].copy()
quarantine_annotations = train_df[train_df[ID_COL].astype(str).isin(QUARANTINED_IDS)][
    [ID_COL, SENTENCE_COL, GLOSS_COL]
].copy()
quarantine_audit = quarantine_annotations.merge(
    quarantine_integrity,
    left_on=ID_COL,
    right_on="id",
    how="left",
    validate="one_to_one",
)

print("Approved quarantine IDs:", sorted(QUARANTINED_IDS))
if not quarantine_audit.empty:
    display(quarantine_audit)
    print(quarantine_audit.to_string(index=False))
print("Unapproved failing IDs:", sorted(UNAPPROVED_BAD_IDS))
print("Configured quarantine IDs absent from train.csv:", sorted(unknown_quarantine_ids))
print("Configured quarantine IDs that are now valid:", sorted(approved_but_now_valid_ids))

eligible_train_df = train_df[
    ~train_df[ID_COL].astype(str).isin(QUARANTINED_IDS)
].copy()
assert len(eligible_train_df) + len(QUARANTINED_IDS) == len(train_df)
print(f"Eligible complete examples after explicit quarantine: {len(eligible_train_df):,}")

# %% [markdown]
# ## 10. Exploratory statistics from verified representations
#
# **What:** Measure sentence/gloss uniqueness and length, motion/token length percentiles,
# plot the four requested histograms, compute length correlations, and display real paired
# examples.
#
# **Why:** P95/P99—not an arbitrary 200 frames—will guide later maximum-length policy.
# Correlations expose how linguistic and temporal lengths relate.
#
# **Output shapes:** Motion remains its observed shape (normally `[T_motion, D_motion]`),
# and tokens remain `[L, T_token]`.

# %%
def word_count(text: Any) -> int:
    return len(str(text).strip().split()) if pd.notna(text) else 0


analysis_df = eligible_train_df[[ID_COL, SENTENCE_COL, GLOSS_COL]].copy()
analysis_df["sentence_words"] = analysis_df[SENTENCE_COL].map(word_count)
analysis_df["gloss_tokens"] = analysis_df[GLOSS_COL].map(word_count)

motion_length_by_id = {}
motion_shape_by_id = {}
for row in integrity_df.itertuples():
    if getattr(row, "shape", None) is not None and not row.invalid_shape:
        motion_shape_by_id[row.id] = tuple(row.shape)
        motion_length_by_id[row.id] = int(row.shape[0])

analysis_df["motion_length"] = analysis_df[ID_COL].map(motion_length_by_id)
analysis_df["token_length"] = analysis_df[ID_COL].map(
    {sample_id: int(tokens.shape[1]) for sample_id, tokens in token_by_id.items()}
)


def numeric_stats(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


text_stats = {
    "sentence_unique": int(eligible_train_df[SENTENCE_COL].nunique(dropna=True)),
    "sentence_length": numeric_stats(analysis_df["sentence_words"]),
    "gloss_unique_sequences": int(eligible_train_df[GLOSS_COL].nunique(dropna=True)),
    "gloss_vocabulary_size": int(len({token for gloss in eligible_train_df[GLOSS_COL].dropna() for token in str(gloss).split()})),
    "gloss_length": numeric_stats(analysis_df["gloss_tokens"]),
    "motion_length": numeric_stats(analysis_df["motion_length"]),
    "motion_token_length": numeric_stats(analysis_df["token_length"]),
}
print(json.dumps(text_stats, indent=2))

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
plot_specs = [
    ("sentence_words", "Sentence length (words)"),
    ("gloss_tokens", "Gloss length (tokens)"),
    ("motion_length", "Motion length (frames)"),
    ("token_length", "Motion-token sequence length"),
]
for ax, (column, title) in zip(axes.ravel(), plot_specs):
    values = analysis_df[column].dropna()
    if len(values):
        ax.hist(values, bins=50)
    ax.set_title(title)
    ax.set_xlabel(title)
    ax.set_ylabel("Examples")
plt.tight_layout()
plt.show()

correlation_columns = ["sentence_words", "gloss_tokens", "motion_length", "token_length"]
length_correlations = analysis_df[correlation_columns].corr(method="spearman")
print("Spearman length correlations:")
display(length_correlations)

for _, row in analysis_df.head(CFG.examples_to_show).iterrows():
    sample_id = str(row[ID_COL])
    print("\n" + "=" * 100)
    print("ID:", sample_id)
    print("Sentence:", row[SENTENCE_COL])
    print("Gloss:", row[GLOSS_COL])
    print("Motion shape:", motion_shape_by_id.get(sample_id))
    print("Token shape [L, T]:", tuple(token_by_id[sample_id].shape) if sample_id in token_by_id else None)

# %% [markdown]
# ## 11. Leakage-safe 80/10/10 internal split
#
# **What:** Normalize text by lowercasing, removing punctuation, and collapsing whitespace;
# then use that normalized sentence as a grouping key. `GroupShuffleSplit` first reserves
# 20% for validation/test, then divides those groups equally.
#
# **Why:** Every exact normalized duplicate is assigned as a unit, so no identical sentence
# can cross split boundaries. The official competition test set is not used as the internal
# test set and never enters the future retrieval database.
#
# **Output:** `split_df` with a `split` column and an assertion that exact leakage is zero.

# %%
def normalize_sentence(text: Any) -> str:
    lowered = str(text).lower()
    no_punctuation = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    return re.sub(r"\s+", " ", no_punctuation).strip()


split_df = eligible_train_df.copy()
split_df["normalized_sentence"] = split_df[SENTENCE_COL].map(normalize_sentence)
if split_df["normalized_sentence"].eq("").any():
    raise ValueError("At least one sentence becomes empty after normalization")

outer = GroupShuffleSplit(n_splits=1, test_size=CFG.validation_fraction + CFG.test_fraction, random_state=CFG.seed)
train_indices, holdout_indices = next(outer.split(split_df, groups=split_df["normalized_sentence"]))
holdout_df = split_df.iloc[holdout_indices]

relative_test_fraction = CFG.test_fraction / (CFG.validation_fraction + CFG.test_fraction)
inner = GroupShuffleSplit(n_splits=1, test_size=relative_test_fraction, random_state=CFG.seed + 1)
validation_rel, test_rel = next(inner.split(holdout_df, groups=holdout_df["normalized_sentence"]))
validation_indices = holdout_indices[validation_rel]
test_indices = holdout_indices[test_rel]

split_df["split"] = ""
split_df.iloc[train_indices, split_df.columns.get_loc("split")] = "train"
split_df.iloc[validation_indices, split_df.columns.get_loc("split")] = "validation"
split_df.iloc[test_indices, split_df.columns.get_loc("split")] = "test"

assert not split_df["split"].eq("").any()
split_sets = {
    name: set(group["normalized_sentence"])
    for name, group in split_df.groupby("split")
}
exact_overlap = {
    "train_validation": len(split_sets["train"] & split_sets["validation"]),
    "train_test": len(split_sets["train"] & split_sets["test"]),
    "validation_test": len(split_sets["validation"] & split_sets["test"]),
}
assert sum(exact_overlap.values()) == 0, exact_overlap

split_counts = split_df["split"].value_counts()
print("Split sample counts:")
display(split_counts.rename("samples").to_frame())
print("Split proportions:")
display((split_counts / len(split_df)).rename("fraction").to_frame())
print("Duplicate leakage (exact normalized sentences):", exact_overlap)

# %% [markdown]
# ## 12. Practical near-duplicate audit
#
# **What:** Character 3–5 gram TF-IDF finds each validation/test sentence's closest sentence
# in the training split, and each test sentence's closest validation sentence. Pairs above
# `CFG.near_duplicate_threshold` are reported for manual review.
#
# **Why:** This catches minor spelling, punctuation, or inflection variants that exact
# normalization does not merge. These are reported rather than silently reassigning data;
# the threshold is an audit setting, not a model confidence threshold.

# %%
def cross_split_near_duplicates(
    reference: pd.DataFrame,
    query: pd.DataFrame,
    reference_name: str,
    query_name: str,
) -> pd.DataFrame:
    if reference.empty or query.empty:
        return pd.DataFrame()
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=CFG.near_duplicate_max_features,
        sublinear_tf=True,
    )
    reference_text = reference["normalized_sentence"].tolist()
    query_text = query["normalized_sentence"].tolist()
    reference_matrix = vectorizer.fit_transform(reference_text)
    query_matrix = vectorizer.transform(query_text)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(reference_matrix)
    distances, indices = nn.kneighbors(query_matrix)
    similarities = 1.0 - distances[:, 0]
    rows = []
    for query_pos, (similarity, reference_pos) in enumerate(zip(similarities, indices[:, 0])):
        if similarity >= CFG.near_duplicate_threshold:
            ref_row = reference.iloc[int(reference_pos)]
            query_row = query.iloc[query_pos]
            rows.append({
                "reference_split": reference_name,
                "query_split": query_name,
                "similarity": float(similarity),
                "reference_id": str(ref_row[ID_COL]),
                "query_id": str(query_row[ID_COL]),
                "reference_sentence": ref_row[SENTENCE_COL],
                "query_sentence": query_row[SENTENCE_COL],
            })
    return pd.DataFrame(rows).sort_values("similarity", ascending=False) if rows else pd.DataFrame()


train_split = split_df[split_df["split"].eq("train")]
validation_split = split_df[split_df["split"].eq("validation")]
test_split = split_df[split_df["split"].eq("test")]
near_duplicate_frames = [
    cross_split_near_duplicates(train_split, validation_split, "train", "validation"),
    cross_split_near_duplicates(train_split, test_split, "train", "test"),
    cross_split_near_duplicates(validation_split, test_split, "validation", "test"),
]
near_duplicates = pd.concat([frame for frame in near_duplicate_frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in near_duplicate_frames) else pd.DataFrame()
print(f"Cross-split near-duplicate pairs at similarity >= {CFG.near_duplicate_threshold}: {len(near_duplicates)}")
if not near_duplicates.empty:
    display(near_duplicates.head(100))

# %% [markdown]
# ## 13. Phase 1 gate, structured report, and validated exports
#
# **What:** Combine all checks into a single readiness gate. Only a passing run writes split
# CSVs, the integrity table, manifest, and JSON report to `/kaggle/working`.
#
# **Why:** Downstream retrieval and neural models must never train on unverified targets.
# A failed gate prints every reason and raises after the report is displayed.
#
# **Expected output:** A structured report containing measured—not fabricated—values and
# `PHASE1_READY = True`. If it is false, fix the reported data/override issue and rerun.

# %%
blocking_reasons = []
unapproved_integrity = integrity_df[~integrity_df["id"].isin(QUARANTINED_IDS)].copy()
unapproved_token_errors = [
    item for item in token_parse_errors if str(item.get("id")) not in QUARANTINED_IDS
]
unapproved_metadata_mismatches = [
    item for item in metadata_mismatches if str(item.get("id")) not in QUARANTINED_IDS
]
unapproved_masks = {
    column: unapproved_integrity[column].astype("boolean").fillna(False).astype(bool)
    for column in ("numeric", "has_nan", "has_inf", "zero_only")
}
actionable_integrity = {
    "missing_motion": int(unapproved_integrity["motion_missing"].sum()),
    "ambiguous_motion": int(unapproved_integrity["motion_ambiguous"].sum()),
    "missing_tokens": int(unapproved_integrity["token_missing"].sum()),
    "token_parse_or_load_errors": int(len(unapproved_token_errors)),
    "invalid_shapes": int(unapproved_integrity["invalid_shape"].sum()),
    "scan_errors": int(unapproved_integrity["scan_error"].notna().sum()),
    "non_numeric_samples": int((unapproved_integrity["motion_path"].notna() & ~unapproved_masks["numeric"]).sum()),
    "empty_arrays": int((unapproved_integrity["motion_path"].notna() & unapproved_integrity["num_values"].fillna(0).eq(0)).sum()),
    "nan_samples": int(unapproved_masks["has_nan"].sum()),
    "inf_samples": int(unapproved_masks["has_inf"].sum()),
    "zero_only_samples": int(unapproved_masks["zero_only"].sum()),
    "missing_sentence": int(unapproved_integrity["sentence_missing"].sum()),
    "missing_gloss": int(unapproved_integrity["gloss_missing"].sum()),
}

if unknown_quarantine_ids:
    blocking_reasons.append(f"Configured quarantine IDs are absent from train.csv: {sorted(unknown_quarantine_ids)}")
if approved_but_now_valid_ids:
    blocking_reasons.append(
        "Previously quarantined IDs are now valid; remove them from the quarantine list after review: "
        f"{sorted(approved_but_now_valid_ids)}"
    )
if UNAPPROVED_BAD_IDS:
    blocking_reasons.append(f"Unapproved failing training IDs require review: {sorted(UNAPPROVED_BAD_IDS)}")
if actionable_integrity["missing_motion"]:
    blocking_reasons.append(f"{actionable_integrity['missing_motion']} unapproved training motions are missing")
if actionable_integrity["ambiguous_motion"]:
    blocking_reasons.append(f"{actionable_integrity['ambiguous_motion']} unapproved motion paths are ambiguous")
if CFG.require_motion_tokens and actionable_integrity["missing_tokens"]:
    blocking_reasons.append(f"{actionable_integrity['missing_tokens']} unapproved training token targets are missing")
if CFG.require_motion_tokens and actionable_integrity["token_parse_or_load_errors"]:
    blocking_reasons.append(f"{actionable_integrity['token_parse_or_load_errors']} unapproved token targets failed parsing/loading")
if actionable_integrity["invalid_shapes"]:
    blocking_reasons.append(f"{actionable_integrity['invalid_shapes']} unapproved motion arrays have invalid/inconsistent shapes")
if actionable_integrity["scan_errors"]:
    blocking_reasons.append(f"{actionable_integrity['scan_errors']} unapproved motion arrays could not be scanned")
if actionable_integrity["non_numeric_samples"]:
    blocking_reasons.append(f"{actionable_integrity['non_numeric_samples']} unapproved motion arrays are non-numeric")
if actionable_integrity["empty_arrays"]:
    blocking_reasons.append(f"{actionable_integrity['empty_arrays']} unapproved motion arrays are empty")
if actionable_integrity["nan_samples"]:
    blocking_reasons.append(f"{actionable_integrity['nan_samples']} unapproved motions contain NaN")
if actionable_integrity["inf_samples"]:
    blocking_reasons.append(f"{actionable_integrity['inf_samples']} unapproved motions contain Inf")
if actionable_integrity["zero_only_samples"]:
    blocking_reasons.append(f"{actionable_integrity['zero_only_samples']} unapproved motions are zero-only")
if actionable_integrity["missing_sentence"] or actionable_integrity["missing_gloss"]:
    blocking_reasons.append("Sentence or gloss annotations are missing")
if integrity_summary["duplicate_ids"]:
    blocking_reasons.append(f"{integrity_summary['duplicate_ids']} rows participate in duplicate IDs")
if unapproved_metadata_mismatches:
    blocking_reasons.append(f"{len(unapproved_metadata_mismatches)} unapproved CSV/metadata annotation mismatches were found")
if EXPECTED_MOTION_NDIM is None or EXPECTED_FEATURE_DIM is None:
    blocking_reasons.append("A single rank-2 motion representation/feature dimension was not established")
if CFG.require_rvq_checkpoint and not checkpoint_candidates:
    blocking_reasons.append(
        "No supplied RVQ/VQ-VAE checkpoint was found. In Kaggle, add the "
        "Motion-S_VAE-RVQ model input (or set CFG.rvq_checkpoint to its exact attached path)."
    )
if CFG.require_rvq_checkpoint and checkpoint_candidates and not any("error" not in item for item in checkpoint_inspection):
    blocking_reasons.append("RVQ/VQ-VAE checkpoint candidates were found but none could be inspected safely")
if not CFG.full_numeric_scan:
    blocking_reasons.append("full_numeric_scan=False; NaN/Inf/zero-only checks were not exhaustive")

PHASE1_READY = len(blocking_reasons) == 0
phase1_warnings = []
if not checkpoint_candidates:
    phase1_warnings.append(
        "RVQ-VAE checkpoint is not attached. This does not block Phase-2 retrieval export, "
        "but it must be resolved before token decoding or motion reconstruction."
    )
if QUARANTINED_IDS:
    phase1_warnings.append(
        f"{len(QUARANTINED_IDS)} audited provider-incomplete records are explicitly quarantined: "
        f"{sorted(QUARANTINED_IDS)}"
    )

dataset_report = {
    "phase": "Phase 1 - Dataset Verification and Preprocessing",
    "readiness_scope": "Phase-2 retrieval artifact export",
    "phase1_ready": PHASE1_READY,
    "blocking_reasons": blocking_reasons,
    "non_blocking_warnings": phase1_warnings,
    "dataset_root": str(DATA_ROOT),
    "train_csv": str(TRAIN_CSV),
    "external_test_csv": str(EXTERNAL_TEST_CSV) if EXTERNAL_TEST_CSV else None,
    "train_examples_original": int(len(train_df)),
    "train_examples_usable": int(len(eligible_train_df)),
    "external_test_examples": int(len(external_test_df)) if external_test_df is not None else None,
    "quarantine": {
        "policy": "Exact audited IDs only; no target substitution",
        "configured_ids": sorted(configured_quarantine_ids),
        "applied_ids": sorted(QUARANTINED_IDS),
        "unapproved_bad_ids": sorted(UNAPPROVED_BAD_IDS),
        "unknown_configured_ids": sorted(unknown_quarantine_ids),
        "configured_ids_now_valid": sorted(approved_but_now_valid_ids),
    },
    "columns": {
        "id": ID_COL,
        "sentence": SENTENCE_COL,
        "gloss": GLOSS_COL,
        "motion_tokens": TOKEN_COLS,
    },
    "raw_motion": {
        "representation": "Observed numeric feature array; no joint reshape performed",
        "ndim": EXPECTED_MOTION_NDIM,
        "feature_dimension": EXPECTED_FEATURE_DIM,
        "length_statistics": text_stats["motion_length"],
        "dtype_counts": integrity_df["dtype"].value_counts(dropna=False).to_dict(),
    },
    "motion_tokens": token_summary,
    "token_canonical_layout": "[L, T]" if token_by_id else None,
    "special_tokens": {
        "evidence": special_token_evidence,
        "conclusion": (
            "Provider declarations printed above; verify their role against decoder code"
            if special_token_evidence
            else "No dataset-provided declaration located; RVQ values are codebook IDs, and model special IDs must be added later without collision"
        ),
    },
    "rvq_vae": {
        "checkpoint_candidates": [str(path) for path in checkpoint_candidates],
        "code_config_candidates": [str(path) for path in code_config_candidates],
        "checkpoint_inspection": checkpoint_inspection,
        "codebook_tensor_evidence": codebook_tensor_evidence,
    },
    "text_statistics": text_stats,
    "integrity": integrity_summary,
    "actionable_integrity_after_quarantine": actionable_integrity,
    "splits": {
        "counts": {str(k): int(v) for k, v in split_counts.items()},
        "exact_duplicate_leakage": exact_overlap,
        "near_duplicate_threshold": CFG.near_duplicate_threshold,
        "near_duplicate_pairs": int(len(near_duplicates)),
    },
    "configuration": asdict(CFG),
}

print("\n" + "=" * 100)
print("MOTION-S PHASE 1 STRUCTURED DATASET REPORT")
print("=" * 100)
print(json.dumps(dataset_report, indent=2, default=str))

# Diagnostics are always exported, including failed runs. They are evidence for repair,
# not approved training caches or splits.
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
integrity_df.to_csv(DIAGNOSTICS_DIR / "integrity_report.csv", index=False)
if not quarantine_audit.empty:
    quarantine_audit.to_csv(DIAGNOSTICS_DIR / "quarantined_samples.csv", index=False)
with (DIAGNOSTICS_DIR / "dataset_report.json").open("w", encoding="utf-8") as handle:
    json.dump(dataset_report, handle, indent=2, default=str)
if token_parse_errors:
    pd.DataFrame(token_parse_errors).to_csv(DIAGNOSTICS_DIR / "token_errors.csv", index=False)
if metadata_mismatches:
    pd.DataFrame(metadata_mismatches).to_csv(DIAGNOSTICS_DIR / "metadata_mismatches.csv", index=False)
print("Diagnostic evidence written to:", DIAGNOSTICS_DIR)

if PHASE1_READY:
    manifest.to_csv(OUTPUT_DIR / "file_manifest.csv", index=False)
    if not near_duplicates.empty:
        near_duplicates.to_csv(DIAGNOSTICS_DIR / "near_duplicate_audit.csv", index=False)
    print("\nPHASE1_READY = True")
    print(f"Explicitly quarantined incomplete examples: {len(QUARANTINED_IDS)} -> {sorted(QUARANTINED_IDS)}")
    print("Dataset validation passed; creating the portable Phase-2 artifact package next.")
else:
    print("\nPHASE1_READY = False")
    print("Blocking reasons:")
    for reason in blocking_reasons:
        print(" -", reason)
    raise RuntimeError("Phase 1 failed. Do not proceed to retrieval or model construction.")

# %% [markdown]
# ## 14. Finalize the portable Phase-1 artifact package
#
# **What:** Reuse the already-created split DataFrames exactly as they are, convert source
# paths to dataset-root-relative references, normalize each RVQ-level cell to JSON, create a
# compact sample manifest/config/report/README, and immediately reload and verify every
# export.
#
# **Why:** A new Phase-2 notebook must work after this kernel is gone. It should attach the
# original Motion-S data plus this output package, load the fixed CSVs, and never resplit.
# Raw BVH/NPY data are not copied.
#
# **Checkpoint scope:** The fixed RVQ-VAE checkpoint is recorded if attached but is not
# required for the Phase-2 retrieval baseline, which returns existing motion/tokens. It is a
# hard prerequisite again before RVQ decoding or visualization.

# %%
print("=" * 100)
print("PHASE 1 EXPORT PRE-CHECK")
print("=" * 100)
print("Master dataframe: eligible_train_df")
print("Original annotation dataframe: train_df")
print("Persistent split source: split_df['split']")
print("Available split dataframes: train_split, validation_split, test_split")
print("ID column:", ID_COL)
print("Sentence column:", SENTENCE_COL)
print("Gloss column:", GLOSS_COL)

essential_columns = [ID_COL, SENTENCE_COL, GLOSS_COL]
missing_essential = [column for column in essential_columns if column not in eligible_train_df.columns]
if missing_essential:
    raise RuntimeError(f"Cannot export Phase 1; essential columns are missing: {missing_essential}")
if not TOKEN_COLS or any(column not in eligible_train_df.columns for column in TOKEN_COLS):
    raise RuntimeError(f"Cannot export Phase 1; verified token columns are unavailable: {TOKEN_COLS}")

bvh_column_candidates = [
    column for column in eligible_train_df.columns
    if "bvh" in normalized_name(column) and "path" in normalized_name(column)
]
if len(bvh_column_candidates) > 1:
    raise RuntimeError(f"Ambiguous BVH path columns: {bvh_column_candidates}")
BVH_COL = bvh_column_candidates[0] if bvh_column_candidates else None
print("BVH source:", BVH_COL or "resolved by exact sample ID from discovered .bvh files")
print("Motion feature source: integrity_df['motion_path'], made relative to DATA_ROOT")
print("Motion token representation:", f"{TOKEN_COLS} -> canonical [L, T], L={token_summary.get('rvq_levels')}")
print("Relevant master columns:")
print(eligible_train_df.columns.tolist())

# Reuse—not recreate—the split membership established earlier.
split_sources = {
    "train": train_split.copy(),
    "validation": validation_split.copy(),
    "test": test_split.copy(),
}
for split_name, frame in split_sources.items():
    expected_ids = set(split_df.loc[split_df["split"].eq(split_name), ID_COL].astype(str))
    actual_ids = set(frame[ID_COL].astype(str))
    assert actual_ids == expected_ids, f"{split_name} DataFrame no longer matches split_df"

train_ids = set(split_sources["train"][ID_COL].astype(str))
val_ids = set(split_sources["validation"][ID_COL].astype(str))
test_ids = set(split_sources["test"][ID_COL].astype(str))
id_overlaps = {
    "train_val": len(train_ids & val_ids),
    "train_test": len(train_ids & test_ids),
    "val_test": len(val_ids & test_ids),
}
assert id_overlaps == {"train_val": 0, "train_test": 0, "val_test": 0}, id_overlaps
assert sum(len(frame) for frame in split_sources.values()) == len(eligible_train_df)
print("Train IDs ∩ Val IDs:", id_overlaps["train_val"])
print("Train IDs ∩ Test IDs:", id_overlaps["train_test"])
print("Val IDs ∩ Test IDs:", id_overlaps["val_test"])


def normalized_sentence_set(frame: pd.DataFrame) -> set[str]:
    return set(frame[SENTENCE_COL].map(normalize_sentence))


sentence_sets = {name: normalized_sentence_set(frame) for name, frame in split_sources.items()}
sentence_overlaps = {
    "train_val": len(sentence_sets["train"] & sentence_sets["validation"]),
    "train_test": len(sentence_sets["train"] & sentence_sets["test"]),
    "val_test": len(sentence_sets["validation"] & sentence_sets["test"]),
}
print("Exact normalized sentence overlap:")
print("Train ↔ Val:", sentence_overlaps["train_val"])
print("Train ↔ Test:", sentence_overlaps["train_test"])
print("Val ↔ Test:", sentence_overlaps["val_test"])
if any(sentence_overlaps.values()):
    raise RuntimeError(f"Exact normalized sentence leakage detected: {sentence_overlaps}")


def relative_to_dataset(path_value: Any) -> Optional[str]:
    """Convert an observed source path to a portable POSIX path below DATA_ROOT."""
    if path_value is None or pd.isna(path_value) or not str(path_value).strip():
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        try:
            return path.resolve().relative_to(DATA_ROOT.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError(f"Source path is outside DATA_ROOT and is not portable: {path}") from exc
    candidate = DATA_ROOT / path
    if candidate.exists():
        return path.as_posix()
    return None


bvh_index = build_exact_artifact_index(bvh_paths)


def resolve_bvh_relative(sample_id: str, row: pd.Series) -> Optional[str]:
    if BVH_COL is not None:
        relative = relative_to_dataset(row[BVH_COL])
        if relative is not None:
            return relative
    candidates = list(dict.fromkeys(bvh_index.get(str(sample_id), [])))
    exact = [path for path in candidates if path.stem == str(sample_id)]
    if len(exact) != 1:
        return None
    return exact[0].resolve().relative_to(DATA_ROOT.resolve()).as_posix()


motion_path_by_id = {
    str(row.id): relative_to_dataset(row.motion_path)
    for row in integrity_df.itertuples()
    if row.motion_path is not None and str(row.id) not in QUARANTINED_IDS
}
motion_length_by_id_export = {
    str(row.id): int(row.shape[0])
    for row in integrity_df.itertuples()
    if row.shape is not None and str(row.id) not in QUARANTINED_IDS
}


def make_portable_split(source: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Build one lightweight split file with JSON RVQ sequences and relative paths."""
    base_columns = list(dict.fromkeys(essential_columns + ([BVH_COL] if BVH_COL else []) + TOKEN_COLS))
    exported = source[base_columns].copy()
    exported[ID_COL] = exported[ID_COL].astype(str)
    bvh_relative = []
    motion_relative = []
    motion_lengths = []
    token_lengths = []

    for _, row in source.iterrows():
        sample_id = str(row[ID_COL])
        if sample_id not in token_by_id:
            raise RuntimeError(f"Verified token map unexpectedly lacks {sample_id}")
        tokens = token_by_id[sample_id]
        if tokens.shape[0] != len(TOKEN_COLS):
            raise RuntimeError(
                f"RVQ level mismatch for {sample_id}: {tokens.shape[0]} vs {len(TOKEN_COLS)}"
            )
        for level, column in enumerate(TOKEN_COLS):
            exported.loc[row.name, column] = json.dumps(
                [int(token) for token in tokens[level]], separators=(",", ":")
            )
        bvh_relative.append(resolve_bvh_relative(sample_id, row))
        motion_relative.append(motion_path_by_id.get(sample_id))
        motion_lengths.append(motion_length_by_id_export.get(sample_id))
        token_lengths.append(int(tokens.shape[1]))

    # Replace a source BVH path with its portable representation and also expose a stable
    # canonical name for Phase 2.
    exported["bvh_relative_path"] = bvh_relative
    if BVH_COL is not None:
        exported[BVH_COL] = bvh_relative
    exported["motion_feature_relative_path"] = motion_relative
    exported["motion_length"] = motion_lengths
    exported["token_length"] = token_lengths
    exported["split"] = split_name
    return exported.reset_index(drop=True)


portable_splits = {
    name: make_portable_split(frame, name)
    for name, frame in split_sources.items()
}

SPLITS_DIR = OUTPUT_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
split_paths = {
    "train": SPLITS_DIR / "train_split.csv",
    "validation": SPLITS_DIR / "val_split.csv",
    "test": SPLITS_DIR / "test_split.csv",
}
for name, path in split_paths.items():
    portable_splits[name].to_csv(path, index=False)

# One compact row per usable sample. Token payloads stay in the split CSVs; this manifest
# holds resource references and measured lengths only.
manifest_frames = []
for split_name, frame in portable_splits.items():
    part = pd.DataFrame({
        "sample_id": frame[ID_COL].astype(str),
        "split": split_name,
        "sentence": frame[SENTENCE_COL],
        "gloss": frame[GLOSS_COL],
        "bvh_relative_path": frame["bvh_relative_path"],
        "motion_feature_relative_path": frame["motion_feature_relative_path"],
        "has_bvh": frame["bvh_relative_path"].notna(),
        "has_motion_feature": frame["motion_feature_relative_path"].notna(),
        "has_motion_tokens": frame[TOKEN_COLS].notna().all(axis=1),
        "motion_length": frame["motion_length"],
        "token_length": frame["token_length"],
    })
    manifest_frames.append(part)
sample_manifest = pd.concat(manifest_frames, ignore_index=True)
assert len(sample_manifest) == len(eligible_train_df)
assert sample_manifest["sample_id"].is_unique
assert not set(sample_manifest["sample_id"]) & QUARANTINED_IDS
# Because the source dataset is mounted during Phase 1, verify every reference now. Phase 2
# can later reconstruct the same paths below its newly discovered Motion-S root.
sample_manifest["has_bvh"] = sample_manifest["bvh_relative_path"].map(
    lambda value: bool(pd.notna(value) and (DATA_ROOT / str(value)).is_file())
)
sample_manifest["has_motion_feature"] = sample_manifest["motion_feature_relative_path"].map(
    lambda value: bool(pd.notna(value) and (DATA_ROOT / str(value)).is_file())
)
if not sample_manifest["has_bvh"].all():
    broken = sample_manifest.loc[~sample_manifest["has_bvh"], ["sample_id", "bvh_relative_path"]]
    raise RuntimeError(f"Broken BVH references remain:\n{broken.head(50).to_string(index=False)}")
if not sample_manifest["has_motion_feature"].all():
    broken = sample_manifest.loc[
        ~sample_manifest["has_motion_feature"], ["sample_id", "motion_feature_relative_path"]
    ]
    raise RuntimeError(f"Broken motion-feature references remain:\n{broken.head(50).to_string(index=False)}")
if not sample_manifest["has_motion_tokens"].all():
    broken_ids = sample_manifest.loc[~sample_manifest["has_motion_tokens"], "sample_id"].tolist()
    raise RuntimeError(f"Missing exported token fields remain: {broken_ids[:50]}")
sample_manifest_path = OUTPUT_DIR / "sample_manifest.csv"
sample_manifest.to_csv(sample_manifest_path, index=False)

split_fingerprint_payload = "\n".join(
    f"{row.sample_id},{row.split}"
    for row in sample_manifest[["sample_id", "split"]].sort_values("sample_id").itertuples(index=False)
)
split_fingerprint = hashlib.sha256(split_fingerprint_payload.encode("utf-8")).hexdigest()

per_level_observed_ranges = [
    {
        "level": int(item["level"]),
        "min_id": int(item["observed_min_id"]),
        "max_id": int(item["observed_max_id"]),
        "observed_vocabulary": int(item["observed_vocabulary"]),
    }
    for item in token_summary.get("per_level", [])
]
dataset_config = {
    "project": "Motion-S Text-to-Sign",
    "seed": int(CFG.seed),
    "phase1_output_version": "v1",
    "source_dataset_root_name": DATA_ROOT.name,
    "source_dataset_root_runtime": str(DATA_ROOT),
    "portable_paths_are_relative_to_source_root": True,
    "id_column": ID_COL,
    "sentence_column": SENTENCE_COL,
    "gloss_column": GLOSS_COL,
    "bvh_source_column": BVH_COL,
    "bvh_relative_path_column": "bvh_relative_path",
    "motion_feature_relative_path_column": "motion_feature_relative_path",
    "motion_token_columns": TOKEN_COLS,
    "motion_token_serialization": "JSON integer array per RVQ-level CSV cell",
    "train_size": int(len(portable_splits["train"])),
    "val_size": int(len(portable_splits["validation"])),
    "test_size": int(len(portable_splits["test"])),
    "original_annotated_samples": int(len(train_df)),
    "usable_samples": int(len(eligible_train_df)),
    "quarantined_sample_ids": sorted(QUARANTINED_IDS),
    "motion_feature_dim": int(EXPECTED_FEATURE_DIM) if EXPECTED_FEATURE_DIM is not None else None,
    "motion_feature_dtype": (
        str(integrity_df.loc[integrity_df["id"].isin(set(eligible_train_df[ID_COL].astype(str))), "dtype"].dropna().mode().iloc[0])
        if not integrity_df["dtype"].dropna().empty else None
    ),
    "rvq_levels": int(token_summary["rvq_levels"]) if token_summary else None,
    "codebook_size": None,
    "observed_token_ranges": per_level_observed_ranges,
    "special_token_ids": None,
    "bvh_extension": ".bvh" if bvh_paths and {path.suffix.lower() for path in bvh_paths} == {".bvh"} else None,
    "motion_feature_extension": ".npy" if array_paths and {Path(path).suffix.lower() for path in motion_path_by_id.values()} == {".npy"} else None,
    "rvq_checkpoint_attached": bool(checkpoint_candidates),
    "rvq_checkpoint_required_for_phase2_retrieval": False,
    "rvq_checkpoint_required_before_decoding": True,
    "split_fingerprint_sha256": split_fingerprint,
}
dataset_config_path = OUTPUT_DIR / "dataset_config.json"
with dataset_config_path.open("w", encoding="utf-8") as handle:
    json.dump(dataset_config, handle, indent=2)

split_report = {
    "seed": int(CFG.seed),
    "total_samples": int(len(eligible_train_df)),
    "original_annotated_samples": int(len(train_df)),
    "quarantined_samples": int(len(QUARANTINED_IDS)),
    "quarantined_sample_ids": sorted(QUARANTINED_IDS),
    "train_samples": int(len(portable_splits["train"])),
    "val_samples": int(len(portable_splits["validation"])),
    "test_samples": int(len(portable_splits["test"])),
    "train_val_id_overlap": int(id_overlaps["train_val"]),
    "train_test_id_overlap": int(id_overlaps["train_test"]),
    "val_test_id_overlap": int(id_overlaps["val_test"]),
    "train_val_sentence_overlap": int(sentence_overlaps["train_val"]),
    "train_test_sentence_overlap": int(sentence_overlaps["train_test"]),
    "val_test_sentence_overlap": int(sentence_overlaps["val_test"]),
    "split_strategy": "GroupShuffleSplit by normalized sentence; exact Phase-1 membership reused",
    "split_fingerprint_sha256": split_fingerprint,
    "split_verified": bool(
        not any(id_overlaps.values())
        and not any(sentence_overlaps.values())
        and sum(len(frame) for frame in portable_splits.values()) == len(eligible_train_df)
    ),
}
split_report_path = OUTPUT_DIR / "split_report.json"
with split_report_path.open("w", encoding="utf-8") as handle:
    json.dump(split_report, handle, indent=2)

readme_text = f"""Motion-S Phase 1 Export

Canonical files:
splits/train_split.csv
splits/val_split.csv
splits/test_split.csv
sample_manifest.csv
dataset_config.json
split_report.json

Phase 2 must:
1. Load these exact splits. NEVER recreate train/validation/test splits.
2. Build every retrieval index from splits/train_split.csv ONLY.
3. Use validation and test splits only as query/evaluation sets.
4. Attach the original Motion-S dataset separately for BVH and motion-feature files.
5. Resolve bvh_relative_path and motion_feature_relative_path below that attached dataset root.

Columns:
ID: {ID_COL}
Sentence: {SENTENCE_COL}
Gloss: {GLOSS_COL}
RVQ levels: {', '.join(TOKEN_COLS)}
RVQ cell format: JSON integer array; deserialize with json.loads(), never eval().
Canonical RVQ shape after stacking levels: [L, T] where L={len(TOKEN_COLS)}.

Integrity:
Original annotated rows: {len(train_df)}
Usable complete rows: {len(eligible_train_df)}
Explicitly quarantined IDs: {', '.join(sorted(QUARANTINED_IDS))}
No motion or token target was synthesized.

Checkpoint:
The provider RVQ-VAE checkpoint is not needed for Phase-2 nearest-neighbor retrieval.
It must be attached and verified before token decoding, reconstruction, or visualization.

Split fingerprint (SHA-256): {split_fingerprint}
"""
readme_path = OUTPUT_DIR / "README_phase2.txt"
readme_path.write_text(readme_text, encoding="utf-8")

# Reload from disk immediately. No in-memory DataFrame is trusted as export proof.
train_check = pd.read_csv(split_paths["train"], dtype={ID_COL: str}, low_memory=False)
val_check = pd.read_csv(split_paths["validation"], dtype={ID_COL: str}, low_memory=False)
test_check = pd.read_csv(split_paths["test"], dtype={ID_COL: str}, low_memory=False)
manifest_check = pd.read_csv(sample_manifest_path, dtype={"sample_id": str}, low_memory=False)
reloaded_splits = {"train": train_check, "validation": val_check, "test": test_check}

for name, checked in reloaded_splits.items():
    assert len(checked) == len(portable_splits[name]), (name, len(checked), len(portable_splits[name]))
    for column in essential_columns + TOKEN_COLS + [
        "bvh_relative_path", "motion_feature_relative_path", "motion_length", "token_length", "split"
    ]:
        assert column in checked.columns, f"{column} absent from reloaded {name} split"
    assert set(checked[ID_COL]) == set(portable_splits[name][ID_COL].astype(str))

assert len(manifest_check) == len(eligible_train_df)
assert manifest_check["sample_id"].is_unique
assert set(manifest_check["sample_id"]) == set(eligible_train_df[ID_COL].astype(str))

# Verify JSON token survival across all six levels using one disk-reloaded example.
token_test_row = train_check.iloc[0]
recovered_levels = [np.asarray(json.loads(token_test_row[column]), dtype=np.int64) for column in TOKEN_COLS]
if any(level.ndim != 1 for level in recovered_levels):
    raise RuntimeError("A reloaded RVQ level is not one-dimensional")
if len({len(level) for level in recovered_levels}) != 1:
    raise RuntimeError("Reloaded RVQ levels do not share one temporal length")
recovered_tokens = np.stack(recovered_levels, axis=0)
print("Motion-token serialization verification:")
print("Stored form:", str(token_test_row[TOKEN_COLS[0]])[:200] + "...")
print("Recovered type:", type(recovered_tokens).__name__)
print("Recovered shape [L, T]:", recovered_tokens.shape)
assert recovered_tokens.shape[0] == token_summary["rvq_levels"] == len(TOKEN_COLS)

# Verify five portable resource references from every split against the attached source.
resource_checks = []
for split_name, checked in reloaded_splits.items():
    sample_count = min(5, len(checked))
    for _, row in checked.sample(n=sample_count, random_state=CFG.seed).iterrows():
        bvh_ref = row["bvh_relative_path"]
        motion_ref = row["motion_feature_relative_path"]
        bvh_exists = pd.notna(bvh_ref) and (DATA_ROOT / str(bvh_ref)).is_file()
        motion_exists = pd.notna(motion_ref) and (DATA_ROOT / str(motion_ref)).is_file()
        tokens_available = all(pd.notna(row[column]) for column in TOKEN_COLS)
        resource_checks.append({
            "split": split_name,
            "sample_id": str(row[ID_COL]),
            "sentence": row[SENTENCE_COL],
            "gloss": row[GLOSS_COL],
            "bvh_reference": bvh_ref,
            "bvh_exists": bool(bvh_exists),
            "motion_feature_reference": motion_ref,
            "motion_feature_exists": bool(motion_exists),
            "motion_tokens_available": bool(tokens_available),
        })
resource_check_df = pd.DataFrame(resource_checks)
display(resource_check_df)
print(resource_check_df.to_string(index=False))
if not resource_check_df[["bvh_exists", "motion_feature_exists", "motion_tokens_available"]].all().all():
    raise RuntimeError("At least one sampled portable resource reference is broken")

required_export_paths = [
    split_paths["train"],
    split_paths["validation"],
    split_paths["test"],
    sample_manifest_path,
    dataset_config_path,
    split_report_path,
    readme_path,
    DIAGNOSTICS_DIR / "dataset_report.json",
    DIAGNOSTICS_DIR / "integrity_report.csv",
    DIAGNOSTICS_DIR / "token_errors.csv",
]
missing_exports = [str(path) for path in required_export_paths if not path.is_file()]
PHASE2_EXPORT_READY = bool(
    PHASE1_READY
    and split_report["split_verified"]
    and not missing_exports
    and len(manifest_check) == len(eligible_train_df)
)

print("\nFINAL PHASE-1 OUTPUT TREE")
for current_root, directories, files in os.walk(OUTPUT_DIR):
    relative = Path(current_root).relative_to(OUTPUT_DIR)
    depth = 0 if str(relative) == "." else len(relative.parts)
    label = OUTPUT_DIR.name + "/" if depth == 0 else "    " * depth + relative.name + "/"
    print(label)
    for filename in sorted(files):
        print("    " * (depth + 1) + filename)

id_status = "PASS" if not any(id_overlaps.values()) else "FAIL"
sentence_status = "PASS" if not any(sentence_overlaps.values()) else "FAIL"
integrity_status = "WARNING" if QUARANTINED_IDS and not UNAPPROVED_BAD_IDS else ("PASS" if not UNAPPROVED_BAD_IDS else "FAIL")
checkmark = "✓"
print("\n" + "=" * 50)
print("MOTION-S PHASE 1 EXPORT COMPLETE")
print("=" * 50)
print("\nTotal samples:", len(eligible_train_df))
print("\nTrain:", len(train_check))
print("\nValidation:", len(val_check))
print("\nTest:", len(test_check))
print("\nID column:", ID_COL)
print("\nSentence column:", SENTENCE_COL)
print("\nGloss column:", GLOSS_COL)
print("\nMotion feature dimension:", EXPECTED_FEATURE_DIM)
print("\nMotion token representation: JSON arrays per level; canonical [L, T]")
print("\nRVQ levels:", token_summary.get("rvq_levels"))
print("\nID leakage:", id_status)
print("\nSentence leakage:", sentence_status)
print("\nIntegrity check:", integrity_status, f"({len(QUARANTINED_IDS)} audited records quarantined)")
print("\nExport files:")
for filename in (
    "train_split.csv", "val_split.csv", "test_split.csv", "sample_manifest.csv",
    "dataset_config.json", "split_report.json", "README_phase2.txt",
):
    print(f"{checkmark} {filename}")
print("\nExisting diagnostics preserved:")
for filename in ("dataset_report.json", "integrity_report.csv", "token_errors.csv"):
    print(f"{checkmark} {filename}")
print("\nPHASE 1 READY FOR PHASE 2:", "YES" if PHASE2_EXPORT_READY else "NO")
print("=" * 50)

if not PHASE2_EXPORT_READY:
    raise RuntimeError(f"Phase-2 export verification failed. Missing files: {missing_exports}")

# %% [markdown]
# ## Phase 1 hand-off — stop here
#
# The exported package is self-contained for split membership, language annotations, RVQ
# token targets, and portable source references. Phase 2 must attach the original Motion-S
# dataset and this output, load `splits/train_split.csv` for fitting, and never recreate the
# split. No retrieval or model implementation is included in this notebook.

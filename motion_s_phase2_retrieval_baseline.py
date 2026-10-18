# %% [markdown]
# # Motion-S Text-to-Sign — Phase 2: Semantic Retrieval Baseline
#
# This notebook consumes the immutable Phase-1 artifact package and the original Motion-S
# dataset. It builds three lookup modes—exact match, TF-IDF, and Sentence-BERT—but always
# returns **one existing training sample and its unmodified resources**. Validation and test
# examples are queries only; they never enter a retrieval index.
#
# This is not a motion generator. It does not average, blend, synthesize, or decode motion.
# No Phase-3 model is included.

# %% [markdown]
# ## 2.1 Environment, configuration, and Phase-1 input contract
#
# **What:** Locate the saved Phase-1 package and original Motion-S root, load the exact CSV
# splits/config/manifest, and verify their fingerprint and disjointness.
#
# **Why:** Phase 2 must remain reproducible after the Phase-1 kernel is closed. The saved
# split is authoritative and is never recreated here.
#
# **Expected output:** Split sizes, actual column names, resource roots, six RVQ token
# columns, and zero ID/sentence overlap.

# %%
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from IPython.display import HTML, display
from matplotlib import animation
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class Phase2Config:
    seed: int = 42
    kaggle_input: str = "/kaggle/input"
    phase1_root: Optional[str] = None
    motion_s_root: Optional[str] = None
    output_dir: str = "/kaggle/working/motion_s_phase2"

    sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    sbert_model_path: Optional[str] = None
    sbert_batch_size: int = 128
    embedding_dtype: str = "float32"

    top_k: int = 5
    evaluation_batch_size: int = 128
    self_retrieval_examples: int = 100
    latency_queries: int = 100
    visualization_max_frames: int = 160


CFG2 = Phase2Config()
np.random.seed(CFG2.seed)
torch.manual_seed(CFG2.seed)
torch.cuda.manual_seed_all(CFG2.seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PHASE2_OUTPUT = Path(CFG2.output_dir)
CACHE_DIR = PHASE2_OUTPUT / "cache"
RESULTS_DIR = PHASE2_OUTPUT / "results"
FIGURES_DIR = PHASE2_OUTPUT / "figures"
for directory in (PHASE2_OUTPUT, CACHE_DIR, RESULTS_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Configuration:")
display(pd.Series(asdict(CFG2), name="value").to_frame())


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
    if CFG2.phase1_root:
        root = Path(CFG2.phase1_root)
        if not is_phase1_root(root):
            raise FileNotFoundError(f"Configured Phase-1 package is incomplete: {root}")
        return root.resolve()

    candidates = []
    working_candidate = Path("/kaggle/working/motion_s_phase1")
    if is_phase1_root(working_candidate):
        candidates.append(working_candidate)
    input_root = Path(CFG2.kaggle_input)
    if input_root.exists():
        for config_path in input_root.rglob("dataset_config.json"):
            if is_phase1_root(config_path.parent):
                candidates.append(config_path.parent)
    candidates = list(dict.fromkeys(path.resolve() for path in candidates))
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one Phase-1 artifact package. "
            f"Found: {[str(path) for path in candidates]}. Set CFG2.phase1_root explicitly."
        )
    return candidates[0]


def is_motion_s_root(path: Path) -> bool:
    return (
        (path / "Motion-Features").is_dir()
        and (path / "Train").is_dir()
        and (path / "train.csv").is_file()
    )


def locate_motion_s_root() -> Path:
    if CFG2.motion_s_root:
        root = Path(CFG2.motion_s_root)
        if not is_motion_s_root(root):
            raise FileNotFoundError(f"Configured Motion-S root is invalid: {root}")
        return root.resolve()

    # If these cells were appended to Phase 1, reuse its verified root.
    inherited = globals().get("DATA_ROOT")
    if inherited is not None and is_motion_s_root(Path(inherited)):
        return Path(inherited).resolve()

    input_root = Path(CFG2.kaggle_input)
    candidates = []
    if input_root.exists():
        for current_root, directories, files in os.walk(input_root):
            current = Path(current_root)
            if "train.csv" in files and {"Motion-Features", "Train"}.issubset(set(directories)):
                candidates.append(current.resolve())
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one original Motion-S dataset root. "
            f"Found: {[str(path) for path in candidates]}. Set CFG2.motion_s_root explicitly."
        )
    return candidates[0]


PHASE1_ROOT = locate_phase1_root()
MOTION_S_ROOT = locate_motion_s_root()
print("Phase-1 package:", PHASE1_ROOT)
print("Original Motion-S root:", MOTION_S_ROOT)

with (PHASE1_ROOT / "dataset_config.json").open(encoding="utf-8") as handle:
    dataset_config = json.load(handle)
with (PHASE1_ROOT / "split_report.json").open(encoding="utf-8") as handle:
    split_report = json.load(handle)

ID_COL = dataset_config["id_column"]
SENTENCE_COL = dataset_config["sentence_column"]
GLOSS_COL = dataset_config["gloss_column"]
TOKEN_COLS = list(dataset_config["motion_token_columns"])
BVH_REL_COL = dataset_config["bvh_relative_path_column"]
MOTION_REL_COL = dataset_config["motion_feature_relative_path_column"]

train_df = pd.read_csv(PHASE1_ROOT / "splits/train_split.csv", dtype={ID_COL: str}, low_memory=False)
val_df = pd.read_csv(PHASE1_ROOT / "splits/val_split.csv", dtype={ID_COL: str}, low_memory=False)
test_df = pd.read_csv(PHASE1_ROOT / "splits/test_split.csv", dtype={ID_COL: str}, low_memory=False)
manifest_df = pd.read_csv(PHASE1_ROOT / "sample_manifest.csv", dtype={"sample_id": str}, low_memory=False)

assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0
required_split_columns = [ID_COL, SENTENCE_COL, GLOSS_COL, BVH_REL_COL, MOTION_REL_COL] + TOKEN_COLS
for name, frame in (("train", train_df), ("validation", val_df), ("test", test_df)):
    missing = [column for column in required_split_columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{name} split lacks required Phase-1 columns: {missing}")

train_ids = set(train_df[ID_COL])
val_ids = set(val_df[ID_COL])
test_ids = set(test_df[ID_COL])
assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)
assert len(train_df) + len(val_df) + len(test_df) == len(manifest_df)

fingerprint_payload = "\n".join(
    f"{row.sample_id},{row.split}"
    for row in manifest_df[["sample_id", "split"]].sort_values("sample_id").itertuples(index=False)
)
observed_fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()
assert observed_fingerprint == split_report["split_fingerprint_sha256"]
assert split_report["split_verified"] is True

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Test samples:", len(test_df))
print("ID column:", ID_COL)
print("Sentence column:", SENTENCE_COL)
print("Gloss column:", GLOSS_COL)
print("BVH root/path:", MOTION_S_ROOT, "/", BVH_REL_COL)
print("Motion feature root/path:", MOTION_S_ROOT, "/", MOTION_REL_COL)
print("Motion token representation:", TOKEN_COLS, "JSON per level -> [L, T]")
print("Split fingerprint:", observed_fingerprint)
print("ID overlap train/val, train/test, val/test:", 0, 0, 0)

# %% [markdown]
# ## 2.2 Text normalization and exact-match baseline
#
# **What:** Keep natural sentences unchanged for TF-IDF/SBERT, while maintaining a minimal
# whitespace-normalized form for display and a lowercase/punctuation-insensitive form only
# for exact duplicate lookup.
#
# **Why:** Aggressive preprocessing can damage a pretrained encoder's semantics. Exact
# lookup runs first so repeated user inputs are distinguished from semantic retrieval.
#
# **Output:** An exact-match dictionary containing training IDs only.

# %%
def normalize_text(text: Any) -> str:
    """Safe display/model normalization: stringify, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_for_exact_match(text: Any) -> str:
    """More aggressive normalization used only for exact lookup/leakage checks."""
    text = normalize_text(text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


for frame in (train_df, val_df, test_df):
    frame["raw_sentence"] = frame[SENTENCE_COL].map(str)
    frame["normalized_sentence"] = frame[SENTENCE_COL].map(normalize_text)
    frame["exact_match_key"] = frame[SENTENCE_COL].map(normalize_for_exact_match)

sentence_overlap = {
    "train_val": len(set(train_df["exact_match_key"]) & set(val_df["exact_match_key"])),
    "train_test": len(set(train_df["exact_match_key"]) & set(test_df["exact_match_key"])),
    "val_test": len(set(val_df["exact_match_key"]) & set(test_df["exact_match_key"])),
}
assert not any(sentence_overlap.values()), sentence_overlap
print("Exact normalized sentence overlap:", sentence_overlap)

exact_lookup: dict[str, list[int]] = {}
for position, key in enumerate(train_df["exact_match_key"]):
    exact_lookup.setdefault(key, []).append(position)


def format_candidates(indices: np.ndarray, similarities: np.ndarray, method: str) -> pd.DataFrame:
    rows = []
    for rank, (position, similarity) in enumerate(zip(indices, similarities), start=1):
        row = train_df.iloc[int(position)]
        rows.append({
            "rank": rank,
            "sample_id": str(row[ID_COL]),
            "sentence": row[SENTENCE_COL],
            "gloss": row[GLOSS_COL],
            "similarity": float(similarity),
            "method": method,
            "train_position": int(position),
        })
    result = pd.DataFrame(rows)
    assert set(result["sample_id"]).issubset(train_ids)
    return result


def retrieve_exact(query: str, top_k: int = 5) -> pd.DataFrame:
    positions = exact_lookup.get(normalize_for_exact_match(query), [])[:top_k]
    if not positions:
        return pd.DataFrame(columns=[
            "rank", "sample_id", "sentence", "gloss", "similarity", "method", "train_position"
        ])
    return format_candidates(np.asarray(positions), np.ones(len(positions)), "exact")


exact_example = train_df.iloc[0][SENTENCE_COL]
exact_result = retrieve_exact(exact_example)
assert len(exact_result) >= 1 and exact_result.iloc[0]["sample_id"] in train_ids
print("Match type: Exact")
print("Sample ID:", exact_result.iloc[0]["sample_id"])
print("Sentence:", exact_result.iloc[0]["sentence"])
print("Gloss:", exact_result.iloc[0]["gloss"])

# %% [markdown]
# ## 2.3 TF-IDF lexical baseline
#
# **What:** Fit word unigram/bigram TF-IDF using training sentences only and return cosine
# Top-K neighbors.
#
# **Input:** Natural sentence strings. **Index:** training rows only.
#
# **Output:** Ranked training IDs, sentences, glosses, and lexical cosine similarity.

# %%
train_sentences = train_df[SENTENCE_COL].map(normalize_text).tolist()
tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
    norm="l2",
)
tfidf_matrix = tfidf_vectorizer.fit_transform(train_sentences)
assert tfidf_matrix.shape[0] == len(train_df)


def top_k_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    """Return stable descending Top-K positions."""
    scores = np.asarray(scores).reshape(-1)
    k = min(int(top_k), len(scores))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    candidates = np.argpartition(-scores, kth=k - 1)[:k]
    return candidates[np.argsort(-scores[candidates], kind="stable")]


def retrieve_tfidf(query: str, top_k: int = 5) -> pd.DataFrame:
    query_vector = tfidf_vectorizer.transform([normalize_text(query)])
    scores = (query_vector @ tfidf_matrix.T).toarray().ravel()
    indices = top_k_indices(scores, top_k)
    return format_candidates(indices, scores[indices], "tfidf")


tfidf_example = retrieve_tfidf("I am going to school tomorrow.", top_k=5)
display(tfidf_example)
assert len(tfidf_example) == 5
assert set(tfidf_example["sample_id"]).issubset(train_ids)
print("TF-IDF matrix shape:", tfidf_matrix.shape)
print("TF-IDF vocabulary size:", len(tfidf_vectorizer.vocabulary_))

# %% [markdown]
# ## 2.4 Offline Sentence-BERT index and deterministic cache
#
# **What:** Load `all-MiniLM-L6-v2` from an explicitly attached/local Kaggle model, encode
# training sentences, L2-normalize embeddings, and cache them with a metadata fingerprint.
#
# **Why:** The final notebook must not depend on internet access. If the package/model is
# unavailable, this section raises an actionable error instead of silently downloading or
# replacing SBERT.
#
# **Output shape:** `train_embeddings = [N_train, D_sbert]`, normally float32.

# %%
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

if importlib.util.find_spec("sentence_transformers") is None:
    raise ImportError(
        "sentence-transformers is not installed in this Kaggle image. Attach it as an "
        "offline wheel/dataset or choose a Kaggle image that already provides the package."
    )
from sentence_transformers import SentenceTransformer


def looks_like_sentence_transformer(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and ((path / "modules.json").is_file() or (path / "sentence_bert_config.json").is_file())
    )


def locate_sbert_model() -> str:
    if CFG2.sbert_model_path:
        path = Path(CFG2.sbert_model_path)
        if not looks_like_sentence_transformer(path):
            raise FileNotFoundError(f"Configured SBERT directory is invalid: {path}")
        return str(path.resolve())

    candidates = []
    input_root = Path(CFG2.kaggle_input)
    if input_root.exists():
        for config_path in input_root.rglob("config.json"):
            candidate = config_path.parent
            if looks_like_sentence_transformer(candidate):
                candidates.append(candidate.resolve())
    # A cached model is acceptable only in forced offline mode.
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    if cache_root.exists():
        for modules_path in cache_root.rglob("modules.json"):
            candidate = modules_path.parent
            if looks_like_sentence_transformer(candidate):
                candidates.append(candidate.resolve())
    candidates = list(dict.fromkeys(candidates))
    preferred = [
        path for path in candidates
        if "minilm" in str(path).lower() or "all-minilm-l6-v2" in str(path).lower()
    ]
    usable = preferred or candidates
    if len(usable) != 1:
        raise RuntimeError(
            "Could not select one offline SentenceTransformer model. Attach "
            "sentence-transformers/all-MiniLM-L6-v2 as a Kaggle model/dataset and set "
            f"CFG2.sbert_model_path if needed. Candidates: {[str(path) for path in usable]}"
        )
    return str(usable[0])


SBERT_MODEL_SOURCE = locate_sbert_model()
print("Offline SBERT model source:", SBERT_MODEL_SOURCE)
sbert_model = SentenceTransformer(SBERT_MODEL_SOURCE, device=DEVICE)

embedding_cache_path = CACHE_DIR / "train_sentence_embeddings.npy"
embedding_metadata_path = CACHE_DIR / "retrieval_index_metadata.json"
train_text_fingerprint = hashlib.sha256(
    "\n".join(f"{sample_id}\t{text}" for sample_id, text in zip(train_df[ID_COL], train_sentences)).encode("utf-8")
).hexdigest()
expected_embedding_metadata = {
    "phase1_split_fingerprint_sha256": observed_fingerprint,
    "train_text_fingerprint_sha256": train_text_fingerprint,
    "embedding_model_source": SBERT_MODEL_SOURCE,
    "embedding_model_name": CFG2.sbert_model_name,
    "normalized": True,
    "train_rows": len(train_df),
}

use_embedding_cache = False
if embedding_cache_path.is_file() and embedding_metadata_path.is_file():
    with embedding_metadata_path.open(encoding="utf-8") as handle:
        cached_metadata = json.load(handle)
    comparable = {key: cached_metadata.get(key) for key in expected_embedding_metadata}
    use_embedding_cache = comparable == expected_embedding_metadata

if use_embedding_cache:
    train_embeddings = np.load(embedding_cache_path, allow_pickle=False)
    print("Loaded verified embedding cache:", embedding_cache_path)
else:
    train_embeddings = sbert_model.encode(
        train_sentences,
        batch_size=CFG2.sbert_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(CFG2.embedding_dtype, copy=False)
    np.save(embedding_cache_path, train_embeddings, allow_pickle=False)
    metadata_to_save = {
        **expected_embedding_metadata,
        "embedding_shape": list(train_embeddings.shape),
        "embedding_dtype": str(train_embeddings.dtype),
    }
    with embedding_metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata_to_save, handle, indent=2)

assert train_embeddings.ndim == 2
assert train_embeddings.shape[0] == len(train_df)
assert np.isfinite(train_embeddings).all()
embedding_norms = np.linalg.norm(train_embeddings, axis=1)
assert np.allclose(embedding_norms, 1.0, atol=2e-4)
loaded_embeddings = np.load(embedding_cache_path, allow_pickle=False)
assert loaded_embeddings.shape == train_embeddings.shape
assert loaded_embeddings.dtype == train_embeddings.dtype

print("Embedding model:", CFG2.sbert_model_name)
print("Number of train embeddings:", train_embeddings.shape[0])
print("Embedding dimension:", train_embeddings.shape[1])
print("Data type:", train_embeddings.dtype)
print(f"Approximate memory: {train_embeddings.nbytes / 2**20:.2f} MiB")

# Lightweight deterministic metadata mapping; no BVH arrays are stored.
index_mapping_path = CACHE_DIR / "retrieval_index_mapping.csv"
train_df[[ID_COL, SENTENCE_COL, GLOSS_COL, BVH_REL_COL, MOTION_REL_COL]].to_csv(index_mapping_path, index=False)
joblib.dump(tfidf_vectorizer, CACHE_DIR / "tfidf_vectorizer.joblib")
sp.save_npz(CACHE_DIR / "tfidf_matrix.npz", tfidf_matrix)

# %% [markdown]
# ## 2.5 SBERT search and unified `MotionSRetriever`
#
# **What:** Encode a query, use the normalized dot product against training embeddings, and
# expose exact/TF-IDF/SBERT through one class.
#
# **Why:** One API prevents method-specific metadata mistakes and centralizes the mandatory
# assertion that every returned ID belongs to the training split.

# %%
def retrieve_sbert(query: str, top_k: int = 5) -> pd.DataFrame:
    query_embedding = sbert_model.encode(
        [normalize_text(query)],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(train_embeddings.dtype, copy=False)[0]
    scores = train_embeddings @ query_embedding
    indices = top_k_indices(scores, top_k)
    return format_candidates(indices, scores[indices], "sbert")


class MotionSRetriever:
    """Exact, lexical, and semantic retrieval over the immutable training split."""

    def __init__(
        self,
        train_frame: pd.DataFrame,
        id_column: str,
        sentence_column: str,
        gloss_column: str,
    ) -> None:
        self.train_df = train_frame
        self.id_column = id_column
        self.sentence_column = sentence_column
        self.gloss_column = gloss_column
        self.fitted = False

    def fit(self) -> "MotionSRetriever":
        assert len(self.train_df) == tfidf_matrix.shape[0] == train_embeddings.shape[0]
        assert set(self.train_df[self.id_column].astype(str)) == train_ids
        self.fitted = True
        return self

    def retrieve(self, query: str, method: str = "sbert", top_k: int = 5) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Call fit() before retrieval")
        if not normalize_text(query):
            raise ValueError("Query must contain non-whitespace text")
        if method == "exact":
            result = retrieve_exact(query, top_k)
        elif method == "tfidf":
            result = retrieve_tfidf(query, top_k)
        elif method == "sbert":
            result = retrieve_sbert(query, top_k)
        else:
            raise ValueError("method must be one of: exact, tfidf, sbert")
        assert set(result["sample_id"]).issubset(train_ids)
        assert set(result["sample_id"]).isdisjoint(val_ids | test_ids)
        return result


retriever = MotionSRetriever(train_df, ID_COL, SENTENCE_COL, GLOSS_COL).fit()
sbert_example = retriever.retrieve("I am going to school tomorrow.", method="sbert", top_k=5)
display(sbert_example)
assert len(sbert_example) == 5

# %% [markdown]
# ## 2.6 Motion resource lookup and complete retrieval pipeline
#
# **What:** Resolve a retrieved training ID to its sentence, gloss, BVH, 668-D motion
# feature array, and six-level RVQ tokens. `retrieve_sign` performs exact lookup first and
# otherwise uses the requested retrieval method.
#
# **Important shapes:** Motion remains `[T_motion, 668]`; tokens remain `[6, T_token]`.
# Missing or broken files raise—nothing is replaced with zeros.

# %%
train_position_by_id = {
    str(sample_id): position for position, sample_id in enumerate(train_df[ID_COL].astype(str))
}


def deserialize_motion_tokens(row: pd.Series) -> np.ndarray:
    levels = []
    for column in TOKEN_COLS:
        try:
            values = json.loads(row[column])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON tokens for {row[ID_COL]} column {column}") from exc
        array = np.asarray(values)
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"Token level {column} must be a non-empty 1-D sequence")
        numeric = array.astype(np.float64)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"Token level {column} contains invalid token IDs")
        levels.append(numeric.astype(np.int64))
    if len({len(level) for level in levels}) != 1:
        raise ValueError(f"RVQ levels have unequal lengths for sample {row[ID_COL]}")
    tokens = np.stack(levels, axis=0)
    assert tokens.shape[0] == dataset_config["rvq_levels"]
    return tokens


def resolve_existing_resource(relative_path: Any, label: str, sample_id: str) -> Path:
    if relative_path is None or pd.isna(relative_path) or not str(relative_path).strip():
        raise FileNotFoundError(f"{label} reference is missing for training sample {sample_id}")
    candidate = (MOTION_S_ROOT / str(relative_path)).resolve()
    try:
        candidate.relative_to(MOTION_S_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Unsafe {label} path outside Motion-S root: {candidate}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} file does not exist for {sample_id}: {candidate}")
    return candidate


def get_sample_resources(sample_id: str, load_motion: bool = True) -> dict[str, Any]:
    sample_id = str(sample_id)
    if sample_id not in train_position_by_id:
        raise KeyError(f"Sample {sample_id} is not in the training retrieval database")
    row = train_df.iloc[train_position_by_id[sample_id]]
    bvh_path = resolve_existing_resource(row[BVH_REL_COL], "BVH", sample_id)
    motion_path = resolve_existing_resource(row[MOTION_REL_COL], "motion feature", sample_id)
    tokens = deserialize_motion_tokens(row)
    motion = None
    motion_shape = None
    if load_motion:
        motion = np.load(motion_path, mmap_mode="r", allow_pickle=False)
        motion_shape = tuple(int(value) for value in motion.shape)
        if motion.ndim != 2 or motion.shape[1] != dataset_config["motion_feature_dim"]:
            raise ValueError(f"Unexpected motion shape for {sample_id}: {motion.shape}")
        if not np.issubdtype(motion.dtype, np.number):
            raise TypeError(f"Motion array is not numeric for {sample_id}: {motion.dtype}")
    return {
        "sample_id": sample_id,
        "sentence": row[SENTENCE_COL],
        "gloss": row[GLOSS_COL],
        "bvh_path": bvh_path,
        "motion_feature_path": motion_path,
        "motion": motion,
        "motion_shape": motion_shape,
        "motion_tokens": tokens,
        "motion_token_shape": tuple(tokens.shape),
    }


def retrieve_sign(sentence: str, method: str = "sbert", top_k: int = 5) -> dict[str, Any]:
    exact = retriever.retrieve(sentence, method="exact", top_k=top_k)
    if not exact.empty:
        candidates = exact
        match_type = "Exact"
        retrieval_method = "exact"
    else:
        candidates = retriever.retrieve(sentence, method=method, top_k=top_k)
        match_type = "Semantic retrieval" if method == "sbert" else "Lexical retrieval"
        retrieval_method = method
    if candidates.empty:
        raise RuntimeError("Retriever returned no candidates")
    top = candidates.iloc[0]
    resources = get_sample_resources(top["sample_id"], load_motion=True)
    assert resources["sample_id"] in train_ids
    return {
        "query": sentence,
        "match_type": match_type,
        "retrieval_method": retrieval_method,
        "similarity": float(top["similarity"]),
        "top_match": top.to_dict(),
        "top_k": candidates,
        "resources": resources,
    }


demo_result = retrieve_sign("I am going to school tomorrow.", method="sbert", top_k=5)
print("QUERY\n", demo_result["query"])
print("\nMATCH TYPE\n", demo_result["match_type"])
print("\nTOP MATCH\n", demo_result["resources"]["sentence"])
print("\nSIMILARITY\n", demo_result["similarity"])
print("\nSAMPLE ID\n", demo_result["resources"]["sample_id"])
print("\nGLOSS\n", demo_result["resources"]["gloss"])
print("\nBVH PATH\n", demo_result["resources"]["bvh_path"])
print("\nMOTION FEATURE SHAPE\n", demo_result["resources"]["motion_shape"])
print("\nMOTION TOKEN SHAPE\n", demo_result["resources"]["motion_token_shape"])
assert demo_result["resources"]["motion_token_shape"][0] == len(TOKEN_COLS)

# %% [markdown]
# ## 2.7 BVH visualization using the file's actual hierarchy
#
# **What:** Reuse an existing Phase-1 renderer if present. Otherwise, parse the retrieved
# BVH hierarchy, channel order, offsets, and parent links, perform forward kinematics, and
# animate the actual skeleton topology with Matplotlib.
#
# **Why:** No DWPose links or invented joint layout are used. The animation is the existing
# retrieved training motion, not generated or blended motion.

# %%
@dataclass
class ParsedBVH:
    joint_names: list[str]
    parents: np.ndarray
    offsets: np.ndarray
    channels: list[list[str]]
    channel_starts: np.ndarray
    frames: np.ndarray
    frame_time: float


def parse_bvh(path: Path) -> ParsedBVH:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        motion_line = next(i for i, line in enumerate(lines) if line.strip() == "MOTION")
    except StopIteration as exc:
        raise ValueError(f"BVH MOTION section not found: {path}") from exc

    names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    channels: list[list[str]] = []
    block_stack: list[tuple[str, Optional[int]]] = []
    pending_block: Optional[tuple[str, Optional[int]]] = None
    current_joint: Optional[int] = None

    for raw_line in lines[:motion_line]:
        line = raw_line.strip()
        if line.startswith("ROOT ") or line.startswith("JOINT "):
            name = line.split(maxsplit=1)[1]
            parent = current_joint if current_joint is not None else -1
            joint_index = len(names)
            names.append(name)
            parents.append(parent)
            offsets.append([0.0, 0.0, 0.0])
            channels.append([])
            pending_block = ("joint", joint_index)
        elif line.startswith("End Site"):
            pending_block = ("end", current_joint)
        elif line == "{":
            if pending_block is None:
                continue
            block_stack.append(pending_block)
            if pending_block[0] == "joint":
                current_joint = pending_block[1]
            pending_block = None
        elif line == "}":
            if block_stack:
                block_stack.pop()
            joint_blocks = [item for item in block_stack if item[0] == "joint"]
            current_joint = joint_blocks[-1][1] if joint_blocks else None
        elif line.startswith("OFFSET ") and block_stack and block_stack[-1][0] == "joint":
            values = [float(value) for value in line.split()[1:4]]
            offsets[int(block_stack[-1][1])] = values
        elif line.startswith("CHANNELS ") and current_joint is not None:
            parts = line.split()
            count = int(parts[1])
            channels[current_joint] = parts[2 : 2 + count]

    frames_declared = int(lines[motion_line + 1].split(":", 1)[1].strip())
    frame_time = float(lines[motion_line + 2].split(":", 1)[1].strip())
    frame_rows = [
        [float(value) for value in line.split()]
        for line in lines[motion_line + 3 :]
        if line.strip()
    ]
    frames = np.asarray(frame_rows, dtype=np.float64)
    if len(frames) != frames_declared:
        raise ValueError(f"BVH frame count mismatch: declared {frames_declared}, loaded {len(frames)}")
    expected_channels = sum(len(item) for item in channels)
    if frames.ndim != 2 or frames.shape[1] != expected_channels:
        raise ValueError(f"BVH channel mismatch: expected {expected_channels}, got {frames.shape}")
    starts = np.cumsum([0] + [len(item) for item in channels[:-1]]).astype(np.int64)
    return ParsedBVH(
        joint_names=names,
        parents=np.asarray(parents, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.float64),
        channels=channels,
        channel_starts=starts,
        frames=frames,
        frame_time=frame_time,
    )


def bvh_global_positions(parsed: ParsedBVH, frame_indices: np.ndarray) -> np.ndarray:
    """Forward kinematics using each BVH joint's declared rotation-channel order."""
    def axis_rotation(axis: str, angle_degrees: float) -> np.ndarray:
        angle = np.deg2rad(angle_degrees)
        cosine, sine = np.cos(angle), np.sin(angle)
        if axis == "x":
            return np.asarray([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]])
        if axis == "y":
            return np.asarray([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
        if axis == "z":
            return np.asarray([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
        raise ValueError(f"Unknown BVH rotation axis: {axis}")

    output = np.zeros((len(frame_indices), len(parsed.joint_names), 3), dtype=np.float64)
    for output_index, frame_index in enumerate(frame_indices):
        frame = parsed.frames[int(frame_index)]
        global_rotations = np.repeat(np.eye(3)[None, :, :], len(parsed.joint_names), axis=0)
        for joint in range(len(parsed.joint_names)):
            names = parsed.channels[joint]
            start = int(parsed.channel_starts[joint])
            values = frame[start : start + len(names)]
            local_translation = np.zeros(3, dtype=np.float64)
            rotation_axes = []
            rotation_values = []
            for channel, value in zip(names, values):
                lower = channel.lower()
                if lower.endswith("position"):
                    local_translation["xyz".index(lower[0])] = value
                elif lower.endswith("rotation"):
                    rotation_axes.append(lower[0])
                    rotation_values.append(value)
            local_rotation = np.eye(3)
            # BVH transformations are composed in the exact declared channel order.
            for axis, angle in zip(rotation_axes, rotation_values):
                local_rotation = local_rotation @ axis_rotation(axis, angle)
            parent = int(parsed.parents[joint])
            local_offset = parsed.offsets[joint] + local_translation
            if parent < 0:
                output[output_index, joint] = local_offset
                global_rotations[joint] = local_rotation
            else:
                output[output_index, joint] = (
                    output[output_index, parent] + global_rotations[parent] @ local_offset
                )
                global_rotations[joint] = global_rotations[parent] @ local_rotation
    return output


def render_bvh_motion(path: Path, max_frames: int = 160) -> HTML:
    parsed = parse_bvh(path)
    stride = max(1, math.ceil(len(parsed.frames) / max_frames))
    frame_indices = np.arange(0, len(parsed.frames), stride, dtype=np.int64)
    positions = bvh_global_positions(parsed, frame_indices)
    valid_parents = [(parent, child) for child, parent in enumerate(parsed.parents) if parent >= 0]

    mins = positions.min(axis=(0, 1))
    maxs = positions.max(axis=(0, 1))
    center = (mins + maxs) / 2
    radius = max(float((maxs - mins).max()) / 2, 1.0)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    lines = [ax.plot([], [], [], color="royalblue", linewidth=1.5)[0] for _ in valid_parents]
    points = ax.scatter([], [], [], s=5, color="darkred")
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(path.name)

    def update(frame_number: int):
        xyz = positions[frame_number]
        for line, (parent, child) in zip(lines, valid_parents):
            segment = xyz[[parent, child]]
            line.set_data(segment[:, 0], segment[:, 1])
            line.set_3d_properties(segment[:, 2])
        points._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
        return [*lines, points]

    interval_ms = parsed.frame_time * stride * 1000
    anim = animation.FuncAnimation(fig, update, frames=len(frame_indices), interval=interval_ms, blit=False)
    html = HTML(anim.to_jshtml())
    plt.close(fig)
    return html


def retrieve_and_visualize(sentence: str, method: str = "sbert") -> tuple[dict[str, Any], Any]:
    result = retrieve_sign(sentence, method=method, top_k=CFG2.top_k)
    resources = result["resources"]
    print("User query:", sentence)
    print("Retrieved sentence:", resources["sentence"])
    print("Retrieved gloss:", resources["gloss"])
    print("Similarity:", result["similarity"])
    print("Sample ID:", resources["sample_id"])

    inherited_renderer = globals().get("visualize_bvh") or globals().get("render_bvh_animation")
    if callable(inherited_renderer):
        visual = inherited_renderer(resources["bvh_path"])
    else:
        visual = render_bvh_motion(resources["bvh_path"], max_frames=CFG2.visualization_max_frames)
    display(visual)
    return result, visual


visual_demo_result, visual_demo = retrieve_and_visualize(
    "I am going to school tomorrow.", method="sbert"
)
VISUALIZATION_VERIFIED = visual_demo is not None

# %% [markdown]
# ## 2.8 Retrieval metrics and batched validation evaluation
#
# **What:** Evaluate TF-IDF and SBERT on the same validation queries using mean smoothed
# sentence BLEU-4, multiset gloss-token F1, normalized edit distance, exact gloss match,
# Top-1/3/5 exact-gloss hit, and Top-1 similarity distribution.
#
# **Why:** Gloss agreement is a linguistic proxy for selection quality—not a direct motion
# metric. No validation item is added to the index.

# %%
try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
except ImportError as exc:
    raise ImportError("NLTK is required for the declared gloss BLEU metric") from exc


def gloss_tokens(gloss: Any) -> list[str]:
    return re.findall(r"[\w']+", normalize_text(gloss).upper(), flags=re.UNICODE)


def gloss_exact(reference: Any, hypothesis: Any) -> bool:
    return gloss_tokens(reference) == gloss_tokens(hypothesis)


def gloss_token_f1(reference: Any, hypothesis: Any) -> float:
    ref = Counter(gloss_tokens(reference))
    hyp = Counter(gloss_tokens(hypothesis))
    overlap = sum((ref & hyp).values())
    precision = overlap / max(sum(hyp.values()), 1)
    recall = overlap / max(sum(ref.values()), 1)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def token_edit_distance(reference: Any, hypothesis: Any) -> int:
    ref = gloss_tokens(reference)
    hyp = gloss_tokens(hypothesis)
    previous = list(range(len(hyp) + 1))
    for i, ref_token in enumerate(ref, start=1):
        current = [i]
        for j, hyp_token in enumerate(hyp, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + int(ref_token != hyp_token),
            ))
        previous = current
    return previous[-1]


BLEU_SMOOTHER = SmoothingFunction().method1


def gloss_bleu(reference: Any, hypothesis: Any) -> float:
    ref = gloss_tokens(reference)
    hyp = gloss_tokens(hypothesis)
    if not ref or not hyp:
        return 0.0
    return float(sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=BLEU_SMOOTHER))


def get_query_topk(query_sentences: list[str], method: str, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    all_indices = []
    all_scores = []
    for start in range(0, len(query_sentences), CFG2.evaluation_batch_size):
        texts = query_sentences[start : start + CFG2.evaluation_batch_size]
        if method == "tfidf":
            query_matrix = tfidf_vectorizer.transform(texts)
            score_matrix = (query_matrix @ tfidf_matrix.T).toarray()
        elif method == "sbert":
            query_embeddings = sbert_model.encode(
                texts,
                batch_size=CFG2.sbert_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(train_embeddings.dtype, copy=False)
            score_matrix = query_embeddings @ train_embeddings.T
        else:
            raise ValueError(method)
        batch_indices = np.stack([top_k_indices(row, top_k) for row in score_matrix])
        batch_scores = np.take_along_axis(score_matrix, batch_indices, axis=1)
        all_indices.append(batch_indices)
        all_scores.append(batch_scores)
    return np.concatenate(all_indices), np.concatenate(all_scores)


def evaluate_retrieval(query_df: pd.DataFrame, method: str, split_name: str) -> tuple[pd.DataFrame, dict[str, float]]:
    queries = query_df[SENTENCE_COL].map(normalize_text).tolist()
    indices, scores = get_query_topk(queries, method=method, top_k=5)
    records = []
    for query_position, (candidate_positions, candidate_scores) in enumerate(zip(indices, scores)):
        query_row = query_df.iloc[query_position]
        candidates = train_df.iloc[candidate_positions]
        reference_gloss = query_row[GLOSS_COL]
        predicted_gloss = candidates.iloc[0][GLOSS_COL]
        exact_hits = [gloss_exact(reference_gloss, value) for value in candidates[GLOSS_COL]]
        edit = token_edit_distance(reference_gloss, predicted_gloss)
        ref_length = max(len(gloss_tokens(reference_gloss)), 1)
        records.append({
            "split": split_name,
            "method": method,
            "query_id": str(query_row[ID_COL]),
            "query": query_row[SENTENCE_COL],
            "ground_truth_gloss": reference_gloss,
            "retrieved_id": str(candidates.iloc[0][ID_COL]),
            "retrieved_sentence": candidates.iloc[0][SENTENCE_COL],
            "retrieved_gloss": predicted_gloss,
            "top1_similarity": float(candidate_scores[0]),
            "gloss_bleu": gloss_bleu(reference_gloss, predicted_gloss),
            "gloss_token_f1": gloss_token_f1(reference_gloss, predicted_gloss),
            "gloss_edit_distance": int(edit),
            "normalized_gloss_edit_distance": float(edit / ref_length),
            "exact_gloss_match": bool(exact_hits[0]),
            "top1_gloss_hit": bool(any(exact_hits[:1])),
            "top3_gloss_hit": bool(any(exact_hits[:3])),
            "top5_gloss_hit": bool(any(exact_hits[:5])),
        })
    results = pd.DataFrame(records)
    assert set(results["retrieved_id"]).issubset(train_ids)
    assert set(results["retrieved_id"]).isdisjoint(val_ids | test_ids)
    metrics = {
        "gloss_bleu": float(results["gloss_bleu"].mean()),
        "gloss_token_f1": float(results["gloss_token_f1"].mean()),
        "gloss_exact_match": float(results["exact_gloss_match"].mean()),
        "gloss_edit_distance": float(results["gloss_edit_distance"].mean()),
        "normalized_gloss_edit_distance": float(results["normalized_gloss_edit_distance"].mean()),
        "top1_gloss_hit": float(results["top1_gloss_hit"].mean()),
        "top3_gloss_hit": float(results["top3_gloss_hit"].mean()),
        "top5_gloss_hit": float(results["top5_gloss_hit"].mean()),
        "mean_top1_similarity": float(results["top1_similarity"].mean()),
    }
    return results, metrics


tfidf_val_results, tfidf_val_metrics = evaluate_retrieval(val_df, "tfidf", "validation")
sbert_val_results, sbert_val_metrics = evaluate_retrieval(val_df, "sbert", "validation")
tfidf_val_results.to_csv(RESULTS_DIR / "validation_tfidf_results.csv", index=False)
sbert_val_results.to_csv(RESULTS_DIR / "validation_sbert_results.csv", index=False)

comparison_table = pd.DataFrame({
    "Metric": [
        "Gloss BLEU", "Gloss exact match", "Gloss token F1", "Gloss edit distance",
        "Normalized edit distance", "Top-1 gloss hit", "Top-3 gloss hit",
        "Top-5 gloss hit", "Mean Top-1 similarity",
    ],
    "TF-IDF": [
        tfidf_val_metrics["gloss_bleu"], tfidf_val_metrics["gloss_exact_match"],
        tfidf_val_metrics["gloss_token_f1"], tfidf_val_metrics["gloss_edit_distance"],
        tfidf_val_metrics["normalized_gloss_edit_distance"], tfidf_val_metrics["top1_gloss_hit"],
        tfidf_val_metrics["top3_gloss_hit"], tfidf_val_metrics["top5_gloss_hit"],
        tfidf_val_metrics["mean_top1_similarity"],
    ],
    "SBERT": [
        sbert_val_metrics["gloss_bleu"], sbert_val_metrics["gloss_exact_match"],
        sbert_val_metrics["gloss_token_f1"], sbert_val_metrics["gloss_edit_distance"],
        sbert_val_metrics["normalized_gloss_edit_distance"], sbert_val_metrics["top1_gloss_hit"],
        sbert_val_metrics["top3_gloss_hit"], sbert_val_metrics["top5_gloss_hit"],
        sbert_val_metrics["mean_top1_similarity"],
    ],
})
comparison_table.to_csv(RESULTS_DIR / "validation_method_comparison.csv", index=False)
display(comparison_table)

# %% [markdown]
# ## 2.9 Self-retrieval sanity check and similarity/confidence analysis
#
# **What:** For a seeded training subset, query SBERT while masking the source row itself.
# Separately, summarize validation Top-1 similarities and relate them to gloss-token F1.
#
# **Why:** Self-retrieval is only a neighborhood sanity check. The validation analysis is
# the evidence needed for any later confidence-threshold experiment; no threshold is chosen
# here.

# %%
rng = np.random.default_rng(CFG2.seed)
self_count = min(CFG2.self_retrieval_examples, len(train_df))
self_positions = rng.choice(len(train_df), size=self_count, replace=False)
self_records = []
for position in self_positions:
    query_embedding = train_embeddings[position]
    scores = train_embeddings @ query_embedding
    scores[position] = -np.inf  # Mandatory self exclusion.
    neighbor = int(np.argmax(scores))
    assert neighbor != position
    self_records.append({
        "query_id": str(train_df.iloc[position][ID_COL]),
        "query": train_df.iloc[position][SENTENCE_COL],
        "query_gloss": train_df.iloc[position][GLOSS_COL],
        "neighbor_id": str(train_df.iloc[neighbor][ID_COL]),
        "neighbor_sentence": train_df.iloc[neighbor][SENTENCE_COL],
        "neighbor_gloss": train_df.iloc[neighbor][GLOSS_COL],
        "similarity": float(scores[neighbor]),
        "gloss_token_f1": gloss_token_f1(
            train_df.iloc[position][GLOSS_COL], train_df.iloc[neighbor][GLOSS_COL]
        ),
    })
self_retrieval_df = pd.DataFrame(self_records)
self_retrieval_df.to_csv(RESULTS_DIR / "self_retrieval_sanity.csv", index=False)
display(self_retrieval_df.head(10))
print("Self-retrieval sanity mean similarity:", self_retrieval_df["similarity"].mean())
print("Self-retrieval sanity mean gloss token F1:", self_retrieval_df["gloss_token_f1"].mean())

similarities = sbert_val_results["top1_similarity"]
similarity_stats = {
    "mean": float(similarities.mean()),
    "median": float(similarities.median()),
    "standard_deviation": float(similarities.std(ddof=1)),
    "minimum": float(similarities.min()),
    "maximum": float(similarities.max()),
    "p10": float(similarities.quantile(0.10)),
    "p25": float(similarities.quantile(0.25)),
    "p75": float(similarities.quantile(0.75)),
    "p90": float(similarities.quantile(0.90)),
}
print("SBERT validation Top-1 similarity distribution:")
display(pd.Series(similarity_stats, name="value").to_frame())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(similarities, bins=35, color="steelblue", edgecolor="white")
axes[0].set_title("Validation SBERT Top-1 similarity")
axes[0].set_xlabel("Cosine similarity")
axes[0].set_ylabel("Queries")
axes[1].scatter(
    sbert_val_results["top1_similarity"],
    sbert_val_results["gloss_token_f1"],
    alpha=0.35,
    s=18,
)
axes[1].set_title("Similarity vs gloss token F1")
axes[1].set_xlabel("SBERT Top-1 similarity")
axes[1].set_ylabel("Gloss token F1")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "sbert_confidence_analysis.png", dpi=160, bbox_inches="tight")
plt.show()

confidence_correlation = float(
    sbert_val_results[["top1_similarity", "gloss_token_f1"]].corr(method="spearman").iloc[0, 1]
)
print("Spearman similarity/F1 correlation:", confidence_correlation)

# %% [markdown]
# ## 2.10 Qualitative error analysis and custom sentences
#
# **What:** Display ten strongest and ten weakest validation retrievals by gloss-token F1,
# attach transparent heuristic failure notes, and run ten custom sentences that are not used
# in any metric.
#
# **Why:** Aggregate metrics hide negation, ambiguity, idioms, rare words, and alternative
# valid gloss formulations. Labels remain unchanged.

# %%
def probable_failure_reason(row: pd.Series) -> str:
    query = normalize_text(row["query"]).lower()
    retrieved = normalize_text(row["retrieved_sentence"]).lower()
    query_tokens = query.split()
    if any(word in query_tokens for word in ("not", "no", "never", "don't", "cannot")):
        return "Possible negation mismatch or lost negative scope"
    if len(query_tokens) <= 3:
        return "Short sentence is semantically ambiguous"
    if len(query_tokens) >= 15:
        return "Long sentence may contain several unmatched propositions"
    if "?" in row["query"]:
        return "Question intent or interrogative structure may differ"
    lexical_overlap = len(set(query_tokens) & set(retrieved.split())) / max(len(set(query_tokens)), 1)
    if lexical_overlap > 0.5 and row["gloss_token_f1"] < 0.3:
        return "High lexical overlap but different meaning/gloss realization"
    if row["top1_similarity"] < similarity_stats["p25"]:
        return "Weak semantic neighborhood, possibly rare wording or vocabulary"
    return "Alternative gloss formulation or semantic retrieval mismatch"


ranked_analysis = sbert_val_results.sort_values(
    ["gloss_token_f1", "top1_similarity"], ascending=[False, False]
).copy()
good_examples = ranked_analysis.head(10).copy()
poor_examples = ranked_analysis.tail(10).sort_values(
    ["gloss_token_f1", "top1_similarity"], ascending=[True, True]
).copy()
poor_examples["probable_failure_reason"] = poor_examples.apply(probable_failure_reason, axis=1)
analysis_columns = [
    "query", "ground_truth_gloss", "retrieved_sentence", "retrieved_gloss",
    "top1_similarity", "gloss_token_f1",
]
print("Good retrieval examples:")
display(good_examples[analysis_columns])
print("Poor retrieval examples:")
display(poor_examples[analysis_columns + ["probable_failure_reason"]])
good_examples.to_csv(RESULTS_DIR / "qualitative_good_examples.csv", index=False)
poor_examples.to_csv(RESULTS_DIR / "qualitative_poor_examples.csv", index=False)

custom_sentences = [
    "I am cooking dinner at home.",
    "Could you please help me find the hospital?",
    "I will visit my teacher next week.",
    "I do not want to go outside today.",
    "Where did you put my blue bag?",
    "My brother takes the bus to work every morning.",
    "The meeting was unexpectedly moved to Friday afternoon.",
    "Please wait here while I call my mother.",
    "Will it rain before we reach the station?",
    "The old clock forgot how to count the quiet hours.",
]
custom_records = []
for query in custom_sentences:
    result = retrieve_sign(query, method="sbert", top_k=5)
    custom_records.append({
        "input": query,
        "match_type": result["match_type"],
        "retrieved_id": result["resources"]["sample_id"],
        "retrieved_sentence": result["resources"]["sentence"],
        "retrieved_gloss": result["resources"]["gloss"],
        "similarity": result["similarity"],
    })
custom_results = pd.DataFrame(custom_records)
display(custom_results)
custom_results.to_csv(RESULTS_DIR / "custom_sentence_results.csv", index=False)

# %% [markdown]
# ## 2.11 Retrieval latency and held-out test evaluation
#
# **What:** Measure batched query-embedding and brute-force similarity time over up to 100
# validation queries. Then evaluate TF-IDF and SBERT once on the untouched internal test
# split using the same fixed training index.
#
# **Why:** With roughly ten thousand training examples, exact matrix multiplication is an
# appropriate transparent baseline; FAISS is unnecessary. Test results are not used to tune
# models, weights, or thresholds.

# %%
latency_count = min(CFG2.latency_queries, len(val_df))
latency_queries = val_df[SENTENCE_COL].map(normalize_text).iloc[:latency_count].tolist()

if DEVICE == "cuda":
    torch.cuda.synchronize()
embedding_started = time.perf_counter()
latency_embeddings = sbert_model.encode(
    latency_queries,
    batch_size=CFG2.sbert_batch_size,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=False,
).astype(train_embeddings.dtype, copy=False)
if DEVICE == "cuda":
    torch.cuda.synchronize()
embedding_elapsed = time.perf_counter() - embedding_started

search_started = time.perf_counter()
latency_scores = latency_embeddings @ train_embeddings.T
latency_top1 = np.argmax(latency_scores, axis=1)
if DEVICE == "cuda":
    torch.cuda.synchronize()
search_elapsed = time.perf_counter() - search_started
assert len(latency_top1) == latency_count

latency_metrics = {
    "queries": latency_count,
    "embedding_ms_per_query": float(1000 * embedding_elapsed / latency_count),
    "similarity_search_ms_per_query": float(1000 * search_elapsed / latency_count),
    "total_ms_per_query": float(1000 * (embedding_elapsed + search_elapsed) / latency_count),
}
print("Latency:")
display(pd.Series(latency_metrics, name="value").to_frame())

tfidf_test_results, tfidf_test_metrics = evaluate_retrieval(test_df, "tfidf", "test")
sbert_test_results, sbert_test_metrics = evaluate_retrieval(test_df, "sbert", "test")
tfidf_test_results.to_csv(RESULTS_DIR / "test_tfidf_results.csv", index=False)
sbert_test_results.to_csv(RESULTS_DIR / "test_sbert_results.csv", index=False)

# Verify several held-out queries resolve only to complete training resources.
for sample_id in sbert_val_results["retrieved_id"].head(5):
    resources = get_sample_resources(sample_id, load_motion=True)
    assert resources["sample_id"] in train_ids
    assert resources["bvh_path"].is_file()
    assert resources["motion_feature_path"].is_file()
    assert resources["motion_shape"][1] == dataset_config["motion_feature_dim"]

# %% [markdown]
# ## 2.12 Final demonstration and Phase-2 report
#
# **What:** Run the requested custom query through the complete pipeline, save all measured
# metrics/configuration, and print the final report.
#
# **Completion gate:** Phase 2 is complete only if TF-IDF and SBERT were both evaluated,
# retrieval returns training IDs exclusively, motion/BVH resources load, and the retrieved
# BVH animation object was successfully created.

# %%
query = "I am going to school tomorrow."
final_result = retrieve_sign(query, method="sbert", top_k=5)
final_resources = final_result["resources"]

print("=" * 41)
print("MOTION-S RETRIEVAL BASELINE")
print("=" * 41)
print("\nInput:\n", query)
print("\nRetrieval method:\n", final_result["retrieval_method"].upper())
print("\nRetrieved sentence:\n", final_resources["sentence"])
print("\nRetrieved gloss:\n", final_resources["gloss"])
print("\nSimilarity:\n", final_result["similarity"])
print("\nSample ID:\n", final_resources["sample_id"])
print("\nBVH:\n", final_resources["bvh_path"])
print("\nMotion feature:\n", final_resources["motion_feature_path"], final_resources["motion_shape"])
print("\nMotion tokens:\n", final_resources["motion_token_shape"])

leakage_detected = not (
    train_ids.isdisjoint(val_ids)
    and train_ids.isdisjoint(test_ids)
    and val_ids.isdisjoint(test_ids)
    and set(sbert_val_results["retrieved_id"]).issubset(train_ids)
    and set(sbert_test_results["retrieved_id"]).issubset(train_ids)
)
BVH_RETRIEVAL_VERIFIED = final_resources["bvh_path"].is_file()
MOTION_RETRIEVAL_VERIFIED = (
    final_resources["motion_feature_path"].is_file()
    and final_resources["motion_shape"][1] == dataset_config["motion_feature_dim"]
)
PHASE2_COMPLETE = bool(
    not leakage_detected
    and BVH_RETRIEVAL_VERIFIED
    and MOTION_RETRIEVAL_VERIFIED
    and VISUALIZATION_VERIFIED
    and len(tfidf_val_results) == len(val_df)
    and len(sbert_val_results) == len(val_df)
    and len(poor_examples) >= min(10, len(val_df))
)

phase2_report = {
    "phase": "Phase 2 - Semantic Retrieval Baseline",
    "phase2_complete": PHASE2_COMPLETE,
    "training_documents_indexed": int(len(train_df)),
    "sentence_embedding_model": CFG2.sbert_model_name,
    "sentence_embedding_source": SBERT_MODEL_SOURCE,
    "embedding_dimension": int(train_embeddings.shape[1]),
    "tfidf_validation": tfidf_val_metrics,
    "sbert_validation": sbert_val_metrics,
    "tfidf_test": tfidf_test_metrics,
    "sbert_test": sbert_test_metrics,
    "sbert_similarity_distribution": similarity_stats,
    "similarity_gloss_f1_spearman": confidence_correlation,
    "latency": latency_metrics,
    "bvh_retrieval_verified": BVH_RETRIEVAL_VERIFIED,
    "motion_retrieval_verified": MOTION_RETRIEVAL_VERIFIED,
    "retrieval_visualization_verified": VISUALIZATION_VERIFIED,
    "data_leakage_detected": leakage_detected,
    "phase1_split_fingerprint_sha256": observed_fingerprint,
    "configuration": asdict(CFG2),
}
with (PHASE2_OUTPUT / "phase2_report.json").open("w", encoding="utf-8") as handle:
    json.dump(phase2_report, handle, indent=2)

print("\nPHASE 2 — RETRIEVAL BASELINE COMPLETE")
print("\nTraining documents indexed:", len(train_df))
print("\nSentence embedding model:", CFG2.sbert_model_name)
print("\nEmbedding dimension:", train_embeddings.shape[1])
print("\nTF-IDF validation:")
print("    Gloss BLEU:", tfidf_val_metrics["gloss_bleu"])
print("    Gloss token F1:", tfidf_val_metrics["gloss_token_f1"])
print("    Exact gloss match:", tfidf_val_metrics["gloss_exact_match"])
print("\nSBERT validation:")
print("    Gloss BLEU:", sbert_val_metrics["gloss_bleu"])
print("    Gloss token F1:", sbert_val_metrics["gloss_token_f1"])
print("    Exact gloss match:", sbert_val_metrics["gloss_exact_match"])
print("\nMean SBERT Top-1 similarity:", similarity_stats["mean"])
print("\nMedian SBERT Top-1 similarity:", similarity_stats["median"])
print("\nAverage retrieval latency (ms/query):", latency_metrics["total_ms_per_query"])
print("\nBVH retrieval verified:", "YES" if BVH_RETRIEVAL_VERIFIED else "NO")
print("\nMotion retrieval verified:", "YES" if MOTION_RETRIEVAL_VERIFIED else "NO")
print("\nRetrieval visualization verified:", "YES" if VISUALIZATION_VERIFIED else "NO")
print("\nData leakage detected:", "YES" if leakage_detected else "NO")
print("\nPHASE 2 COMPLETE:", "YES" if PHASE2_COMPLETE else "NO")

if not PHASE2_COMPLETE:
    raise RuntimeError("Phase 2 completion criteria were not satisfied; do not begin Phase 3")

# %% [markdown]
# ## Stop — Phase 3 is intentionally not implemented
#
# The saved report and result tables contain only measured retrieval results. No T5,
# motion-token generator, diffusion model, motion blending, or generated BVH is present.

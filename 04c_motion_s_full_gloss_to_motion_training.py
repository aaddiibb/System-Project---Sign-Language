# %% [markdown]
# # Motion-S / SignVRSE — Phase 4C: Full Gloss-to-Motion Token Training
#
# This standalone Kaggle notebook continues only from completed Phases 1, 3, 4A, and 4B.
# The first run is intentionally smoke-only. Leave `run_full_training=False`, run all cells,
# inspect `diagnostics/phase4c_smoke_report.json`, then change it to `True` for the expensive run.
# Test data is never used for checkpoint selection or configuration changes.

# %%
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from torch.utils.data import DataLoader, Dataset, Sampler

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EXPECTED_FINGERPRINT = "e2acdf4d6dce91775239ad50be3866dbc6f5da94d940204b62456cdb77fc4b0d"
EXPECTED_SPLIT_SIZES = {"train": 9970, "validation": 1246, "test": 1247}
SPLIT_FILE_CANDIDATES = {
    "train": ("train_split.csv",),
    "validation": ("val_split.csv", "validation_split.csv"),
    "test": ("test_split.csv",),
}
LEVEL_ORDER = ["base_tokens", "residual_1", "residual_2", "residual_3", "residual_4", "residual_5"]
RVQ_LEVELS = 6
CODEBOOK_SIZES = [512] * RVQ_LEVELS
PAD_ID, BOS_ID, EOS_ID = 512, 513, 514
MOTION_VOCAB_SIZE = 515


@dataclass(frozen=True)
class Phase4CConfig:
    seed: int = 42
    kaggle_input: str = "/kaggle/input"
    # Dataset references may use either Kaggle's catalog-style path
    # `/kaggle/input/datasets/<owner>/<slug>` or mounted path `/kaggle/input/<slug>`.
    # `locate()` normalizes both forms and finds the actual nested artifact directory.
    phase1_root: Optional[str] = "/kaggle/input/datasets/adibraian/gpt-first-phase"
    phase3_root: Optional[str] = "/kaggle/input/datasets/adibraian/phase-3"
    phase4a_root: Optional[str] = "/kaggle/input/datasets/adibraian/phase-4"
    phase4b_root: Optional[str] = "/kaggle/input/datasets/adibraian/phase4boutput"
    motion_s_train_csv: Optional[str] = None
    output_dir: str = "/kaggle/working/motion_s_phase4"
    run_full_training: bool = False  # First run MUST remain False for the six-check smoke gate.
    supported_max_t: int = 512
    max_model_positions: int = 514
    max_generation_t: int = 512
    d_model: int = 256
    d_level: int = 64
    num_heads: int = 8
    decoder_layers: int = 4
    ffn_dimension: int = 1024
    dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.06
    max_epochs: int = 25
    early_stopping_patience: int = 4
    effective_batch_size: int = 8
    bucket_boundaries: tuple[int, ...] = (64, 96, 128, 181, 256, 384, 512)
    bucket_batch_sizes: tuple[int, ...] = (8, 8, 8, 8, 4, 2, 1)
    num_workers: int = 0
    pin_memory: bool = True
    smoke_train_samples: int = 64
    smoke_validation_samples: int = 32
    cache_encoder_outputs: bool = False


CFG = Phase4CConfig()
assert CFG.supported_max_t == 512 and CFG.max_model_positions >= CFG.supported_max_t + 1
assert (RVQ_LEVELS, CODEBOOK_SIZES, PAD_ID, BOS_ID, EOS_ID, MOTION_VOCAB_SIZE) == (
    6, [512] * 6, 512, 513, 514, 515
)
assert len(CFG.bucket_boundaries) == len(CFG.bucket_batch_sizes)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


set_seed(CFG.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    raise RuntimeError("Phase 4C requires a Kaggle GPU accelerator")

OUTPUT = Path(CFG.output_dir)
DIAGNOSTICS, RESULTS, FIGURES, BEST_MODEL_DIR = (
    OUTPUT / "diagnostics", OUTPUT / "results", OUTPUT / "figures", OUTPUT / "best_model"
)
for directory in (OUTPUT, DIAGNOSTICS, RESULTS, FIGURES, BEST_MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    if isinstance(value, torch.dtype): return str(value)
    raise TypeError(type(value).__name__)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, np.floating): return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.ndarray): return json_safe(value.tolist())
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, ensure_ascii=False, default=json_default, allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


print("Python:", sys.version.split()[0], "PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0), "memory GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
display(pd.Series(asdict(CFG), name="value").to_frame())

# %% [markdown]
# ## 4C.1 Locked-artifact verification and supported-range coverage
#
# Phase 1 supplies the immutable splits. Phase 3 supplies the frozen T5 encoder/tokenizer and saved
# gloss predictions. Phase 4A supplies token/source integrity evidence. Phase 4B may be a separate
# Kaggle output and must report a passed overfit gate. The original Motion-S CSV is located as an additional provenance check; split CSVs
# remain the actual training source.

# %%
def is_phase1(path: Path) -> bool:
    return (
        (path / "dataset_config.json").is_file()
        and (path / "sample_manifest.csv").is_file()
        and all(any((path / "splits" / filename).is_file() for filename in candidates)
                for candidates in SPLIT_FILE_CANDIDATES.values())
    )


def split_file(root: Path, split_name: str) -> Path:
    matches = [root / "splits" / filename for filename in SPLIT_FILE_CANDIDATES[split_name]
               if (root / "splits" / filename).is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one locked {split_name} split below {root}; found {matches}")
    return matches[0]


def is_phase3(path: Path) -> bool:
    return all((path / item).is_file() for item in (
        "phase3_report.json", "best_model/config.json", "tokenizer/tokenizer_config.json",
        "results/validation_predictions.csv", "results/test_predictions.csv",
    ))


def is_phase4a(path: Path) -> bool:
    return (path / "phase4a_verification_report.json").is_file()


def is_phase4b(path: Path) -> bool:
    return (path / "phase4b_report.json").is_file() and (path / "diagnostics/debug_overfit_report.json").is_file()


def input_reference_candidates(reference: str) -> list[Path]:
    """Return literal and Kaggle-mounted variants for a dataset reference."""
    literal = Path(reference)
    candidates = [literal]
    parts = literal.parts
    if "datasets" in parts:
        dataset_index = parts.index("datasets")
        if len(parts) >= dataset_index + 3:
            slug = parts[dataset_index + 2]
            candidates.append(Path(CFG.kaggle_input) / slug)
    # Also match a mounted input whose slug differs only by punctuation/case.
    input_root = Path(CFG.kaggle_input)
    requested_slug = literal.name
    canonical = lambda value: re.sub(r"[^a-z0-9]", "", value.casefold())
    if input_root.exists():
        candidates.extend(path for path in input_root.iterdir()
                          if path.is_dir() and canonical(path.name) == canonical(requested_slug))
    return list(dict.fromkeys(candidates))


def roots_below(base: Path, predicate) -> list[Path]:
    if not base.exists(): return []
    matches = []
    if predicate(base): matches.append(base.resolve())
    for current, directories, _ in os.walk(base):
        candidate = Path(current)
        if candidate == base: continue
        if predicate(candidate): matches.append(candidate.resolve()); directories[:] = []
    return matches


def locate(kind: str, override: Optional[str], predicate, working: Optional[Path] = None) -> Path:
    attempted: list[str] = []
    if override:
        override_matches = []
        for candidate in input_reference_candidates(override):
            attempted.append(str(candidate))
            override_matches.extend(roots_below(candidate, predicate))
        override_matches = list(dict.fromkeys(override_matches))
        if len(override_matches) == 1: return override_matches[0]
        if len(override_matches) > 1:
            raise RuntimeError(f"Multiple {kind} roots below {override}: {[str(x) for x in override_matches]}")
        print(f"Configured {kind} reference did not resolve; falling back to global discovery: {attempted}")
    if working is not None and predicate(working): return working.resolve()
    matches: list[Path] = []
    root = Path(CFG.kaggle_input)
    if root.exists():
        for current, directories, _ in os.walk(root):
            candidate = Path(current)
            if predicate(candidate): matches.append(candidate.resolve()); directories[:] = []
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        available = [str(path) for path in Path(CFG.kaggle_input).iterdir()] if Path(CFG.kaggle_input).exists() else []
        raise RuntimeError(
            f"Expected exactly one {kind}; found {[str(x) for x in matches]}. "
            f"Configured candidates: {attempted}. Available Kaggle inputs: {available}"
        )
    return matches[0]


PHASE1_ROOT = locate("Phase-1 root", CFG.phase1_root, is_phase1, Path("/kaggle/working/motion_s_phase1"))
PHASE3_ROOT = locate("Phase-3 root", CFG.phase3_root, is_phase3, Path("/kaggle/working/motion_s_phase3"))
PHASE4A_ROOT = locate("completed Phase-4A root", CFG.phase4a_root, is_phase4a, OUTPUT)
PHASE4B_ROOT = locate("completed Phase-4B root", CFG.phase4b_root, is_phase4b, OUTPUT)

with (PHASE1_ROOT / "dataset_config.json").open(encoding="utf-8") as handle: PHASE1_CONFIG = json.load(handle)
with (PHASE3_ROOT / "phase3_report.json").open(encoding="utf-8") as handle: PHASE3_REPORT = json.load(handle)
with (PHASE4A_ROOT / "phase4a_verification_report.json").open(encoding="utf-8") as handle: PHASE4A_REPORT = json.load(handle)
with (PHASE4B_ROOT / "phase4b_report.json").open(encoding="utf-8") as handle: PHASE4B_REPORT = json.load(handle)
with (PHASE4B_ROOT / "diagnostics/debug_overfit_report.json").open(encoding="utf-8") as handle: PHASE4B_DEBUG = json.load(handle)

ID_COL, GLOSS_COL = PHASE1_CONFIG["id_column"], PHASE1_CONFIG["gloss_column"]
SENTENCE_COL = PHASE1_CONFIG.get("sentence_column", "sentence")
TOKEN_COLS = list(PHASE1_CONFIG["motion_token_columns"])
assert PHASE1_CONFIG["split_fingerprint_sha256"] == EXPECTED_FINGERPRINT
assert PHASE4A_REPORT["phase1_split_fingerprint_sha256"] == EXPECTED_FINGERPRINT
assert PHASE4A_REPORT["ready_to_build_motion_model"] is True
assert PHASE4B_REPORT["phase1_fingerprint"] == EXPECTED_FINGERPRINT
assert PHASE4B_REPORT["phase4b_complete"] is True and PHASE4B_DEBUG["passed"] is True
representation = PHASE4A_REPORT["motion_token_representation"]
assert representation["shape_convention"] == "[L,T]"
assert representation["rvq_levels"] == RVQ_LEVELS
assert representation["codebook_sizes"] == CODEBOOK_SIZES
assert representation["level_order"] == LEVEL_ORDER == TOKEN_COLS
assert PHASE4A_REPORT["token_integrity"]["invalid_ids"] == 0

splits: dict[str, pd.DataFrame] = {}
for split_name in EXPECTED_SPLIT_SIZES:
    split_path = split_file(PHASE1_ROOT, split_name)
    frame = pd.read_csv(split_path, dtype={ID_COL: str}, low_memory=False)
    assert len(frame) == EXPECTED_SPLIT_SIZES[split_name] and frame[ID_COL].is_unique
    frame[ID_COL] = frame[ID_COL].astype(str)
    frame["motion_length"] = pd.to_numeric(frame["token_length"], errors="raise").astype(int)
    frame["long_sequence_excluded"] = frame["motion_length"] > CFG.supported_max_t
    frame["split"] = split_name
    splits[split_name] = frame
assert not (set(splits["train"][ID_COL]) & set(splits["validation"][ID_COL]))
assert not (set(splits["train"][ID_COL]) & set(splits["test"][ID_COL]))
assert not (set(splits["validation"][ID_COL]) & set(splits["test"][ID_COL]))


def locate_original_motion_s() -> Path:
    if CFG.motion_s_train_csv:
        path = Path(CFG.motion_s_train_csv)
        if not path.is_file(): raise FileNotFoundError(path)
        return path.resolve()
    candidates = []
    for path in Path(CFG.kaggle_input).rglob("train.csv"):
        if "splits" in path.parts: continue
        try:
            columns = pd.read_csv(path, nrows=0).columns.tolist()
            if ID_COL in columns and GLOSS_COL in columns and all(column in columns for column in TOKEN_COLS):
                candidates.append(path.resolve())
        except Exception:
            continue
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one original Motion-S train.csv, found {[str(x) for x in candidates]}")
    return candidates[0]


MOTION_S_TRAIN_CSV = locate_original_motion_s()
source_columns = pd.read_csv(MOTION_S_TRAIN_CSV, nrows=0).columns.tolist()
assert ID_COL in source_columns and GLOSS_COL in source_columns and all(c in source_columns for c in TOKEN_COLS)

coverage_rows, exclusion_rows = [], []
for split_name, frame in splits.items():
    supported = int((~frame["long_sequence_excluded"]).sum())
    excluded = int(frame["long_sequence_excluded"].sum())
    coverage_rows.append({
        "split": split_name, "original": len(frame), "supported": supported,
        "excluded_gt512": excluded, "coverage_percentage": 100.0 * supported / len(frame),
    })
    exclusion_rows.extend(frame.loc[frame["long_sequence_excluded"], [ID_COL, "motion_length"]].assign(
        split=split_name, long_sequence_excluded=True
    ).rename(columns={ID_COL: "sample_id"}).to_dict("records"))
coverage_df = pd.DataFrame(coverage_rows)
policy_df = pd.DataFrame(exclusion_rows, columns=["sample_id", "motion_length", "split", "long_sequence_excluded"])
policy_df.to_csv(DIAGNOSTICS / "phase4c_long_sequence_policy.csv", index=False)
COVERAGE_REPORT = {
    "supported_max_t": CFG.supported_max_t, "policy": "T<=512 supported; T>512 retained by ID and never truncated",
    "splits": {row["split"]: row for row in coverage_rows},
}
write_json(DIAGNOSTICS / "phase4c_coverage_report.json", COVERAGE_REPORT)

for row in coverage_rows:
    print(f"{row['split']} supported: {row['supported']}")
    print(f"{row['split']} >512: {row['excluded_gt512']}")
    print(f"{row['split']} coverage: {row['coverage_percentage']:.4f}%")
display(coverage_df)
print("Locked artifact verification: PASS")
print("Phase-1:", PHASE1_ROOT, "Phase-3:", PHASE3_ROOT)
print("Phase-4A:", PHASE4A_ROOT, "Phase-4B:", PHASE4B_ROOT)
print("Original Motion-S:", MOTION_S_TRAIN_CSV)

# %% [markdown]
# ## 4C.2 Length-aware data pipeline and verified architecture
#
# Targets are parsed without truncation. Adaptive bucket batches use sizes `8/8/8/8/4/2/1`
# from short to long sequences. Dynamic padding stops at each batch maximum. The frozen encoder is
# evaluated on the fly; this avoids a large variable-length cache while preserving correctness.

# %%
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
except ImportError as error:
    raise ImportError("A clean Kaggle runtime with compatible PyTorch/Transformers is required") from error

TOKENIZER_PATH, PHASE3_MODEL_PATH = PHASE3_ROOT / "tokenizer", PHASE3_ROOT / "best_model"
gloss_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True, use_fast=True)


def parse_motion_tokens(row: pd.Series) -> np.ndarray:
    levels = []
    for column in TOKEN_COLS:
        values = np.asarray(json.loads(row[column]), dtype=np.int64)
        if values.ndim != 1 or values.size == 0: raise ValueError(f"Invalid {column} for {row[ID_COL]}")
        levels.append(values)
    if len({len(level) for level in levels}) != 1: raise ValueError(f"RVQ length mismatch: {row[ID_COL]}")
    tokens = np.stack(levels, axis=0)
    if tokens.shape != (RVQ_LEVELS, int(row["motion_length"])):
        raise ValueError(f"Shape/metadata mismatch: {row[ID_COL]} {tokens.shape}")
    if tokens.min() < 0 or tokens.max() > 511: raise ValueError(f"Invalid RVQ ID: {row[ID_COL]}")
    return tokens


class GlossMotionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, conditioning_column: str = GLOSS_COL):
        self.frame = frame.reset_index(drop=True)
        self.conditioning_column = conditioning_column
        self.token_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int: return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        if index not in self.token_cache: self.token_cache[index] = parse_motion_tokens(row)
        return {
            "sample_id": str(row[ID_COL]), "sentence": str(row.get(SENTENCE_COL, "")),
            "ground_truth_gloss": str(row[GLOSS_COL]), "conditioning_gloss": str(row[self.conditioning_column]),
            "motion_tokens": self.token_cache[index], "motion_length": int(row["motion_length"]),
        }


def collate_motion(batch: list[dict[str, Any]]) -> dict[str, Any]:
    gloss_batch = gloss_tokenizer(
        [item["conditioning_gloss"] for item in batch], padding=True, truncation=False, return_tensors="pt"
    )
    lengths = torch.tensor([item["motion_length"] for item in batch], dtype=torch.long)
    if int(lengths.max()) > CFG.supported_max_t: raise ValueError("Unsupported target reached collate; no truncation allowed")
    temporal = int(lengths.max()) + 1
    decoder_tokens = torch.full((len(batch), RVQ_LEVELS, temporal), PAD_ID, dtype=torch.long)
    target_tokens = torch.full_like(decoder_tokens, PAD_ID)
    padding_mask = torch.ones((len(batch), temporal), dtype=torch.bool)
    for index, item in enumerate(batch):
        target = torch.from_numpy(item["motion_tokens"]).long(); length = target.shape[1]
        decoder_tokens[index, :, 0] = BOS_ID
        decoder_tokens[index, :, 1:length + 1] = target
        target_tokens[index, :, :length] = target
        target_tokens[index, :, length] = EOS_ID
        padding_mask[index, :length + 1] = False
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "sentence": [item["sentence"] for item in batch],
        "ground_truth_gloss": [item["ground_truth_gloss"] for item in batch],
        "conditioning_gloss": [item["conditioning_gloss"] for item in batch],
        "gloss_input_ids": gloss_batch["input_ids"], "gloss_attention_mask": gloss_batch["attention_mask"],
        "decoder_tokens": decoder_tokens, "target_tokens": target_tokens,
        "motion_padding_mask": padding_mask, "original_motion_length": lengths,
    }


def bucket_index(length: int) -> int:
    for index, upper in enumerate(CFG.bucket_boundaries):
        if length <= upper: return index
    raise ValueError(f"Unsupported length reached bucket sampler: {length}")


class AdaptiveBucketBatchSampler(Sampler[list[int]]):
    def __init__(self, lengths: list[int], shuffle: bool, seed: int):
        self.lengths, self.shuffle, self.seed, self.epoch = lengths, shuffle, seed, 0
        self.buckets: list[list[int]] = [[] for _ in CFG.bucket_boundaries]
        for index, length in enumerate(lengths): self.buckets[bucket_index(int(length))].append(index)

    def set_epoch(self, epoch: int) -> None: self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        for bucket, batch_size in zip(self.buckets, CFG.bucket_batch_sizes):
            indices = list(bucket)
            if self.shuffle: rng.shuffle(indices)
            batches.extend(indices[start:start + batch_size] for start in range(0, len(indices), batch_size))
        if self.shuffle: rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]: return iter(self._batches())
    def __len__(self) -> int: return sum(math.ceil(len(b) / s) for b, s in zip(self.buckets, CFG.bucket_batch_sizes))


def make_loader(frame: pd.DataFrame, conditioning_column: str, shuffle: bool, seed_offset: int = 0) -> tuple[DataLoader, AdaptiveBucketBatchSampler]:
    dataset = GlossMotionDataset(frame, conditioning_column)
    sampler = AdaptiveBucketBatchSampler(frame["motion_length"].astype(int).tolist(), shuffle, CFG.seed + seed_offset)
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_motion, num_workers=CFG.num_workers,
        pin_memory=CFG.pin_memory and DEVICE.type == "cuda",
    )
    return loader, sampler


def load_frozen_encoder() -> nn.Module:
    seq2seq = AutoModelForSeq2SeqLM.from_pretrained(PHASE3_MODEL_PATH, local_files_only=True)
    encoder = seq2seq.get_encoder()
    for parameter in encoder.parameters(): parameter.requires_grad = False
    encoder.eval(); del seq2seq
    return encoder


class GlossMotionTransformer(nn.Module):
    def __init__(self, gloss_encoder: nn.Module):
        super().__init__()
        self.gloss_encoder = gloss_encoder
        self.encoder_projection = nn.Linear(int(gloss_encoder.config.d_model), CFG.d_model)
        self.level_embeddings = nn.ModuleList([
            nn.Embedding(MOTION_VOCAB_SIZE, CFG.d_level, padding_idx=PAD_ID) for _ in range(RVQ_LEVELS)
        ])
        self.motion_projection = nn.Linear(RVQ_LEVELS * CFG.d_level, CFG.d_model)
        self.temporal_positions = nn.Embedding(CFG.max_model_positions, CFG.d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=CFG.d_model, nhead=CFG.num_heads, dim_feedforward=CFG.ffn_dimension,
            dropout=CFG.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=CFG.decoder_layers, norm=nn.LayerNorm(CFG.d_model))
        self.heads = nn.ModuleList([nn.Linear(CFG.d_model, MOTION_VOCAB_SIZE) for _ in range(RVQ_LEVELS)])
        self._initialize_trainable()

    def _initialize_trainable(self) -> None:
        for name, parameter in self.named_parameters():
            if name.startswith("gloss_encoder.") or not parameter.requires_grad: continue
            if parameter.ndim > 1: nn.init.xavier_uniform_(parameter)
            elif "norm" in name and name.endswith("weight"): nn.init.ones_(parameter)
            elif name.endswith("bias"): nn.init.zeros_(parameter)
        with torch.no_grad():
            for embedding in self.level_embeddings: embedding.weight[PAD_ID].zero_()

    def train(self, mode: bool = True):
        super().train(mode); self.gloss_encoder.eval(); return self

    def encode_gloss(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hidden = self.gloss_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.encoder_projection(hidden)

    def embed_motion(self, tokens: torch.Tensor) -> torch.Tensor:
        _, levels, temporal = tokens.shape
        assert levels == RVQ_LEVELS and temporal <= CFG.max_model_positions
        embeddings = torch.cat([self.level_embeddings[level](tokens[:, level]) for level in range(RVQ_LEVELS)], dim=-1)
        positions = torch.arange(temporal, device=tokens.device).unsqueeze(0)
        return self.motion_projection(embeddings) + self.temporal_positions(positions)

    def decode(self, decoder_tokens, memory, gloss_attention_mask, motion_padding_mask):
        embedded = self.embed_motion(decoder_tokens); temporal = embedded.shape[1]
        causal_mask = torch.triu(torch.ones(temporal, temporal, dtype=torch.bool, device=embedded.device), diagonal=1)
        hidden = self.decoder(
            tgt=embedded, memory=memory, tgt_mask=causal_mask,
            tgt_key_padding_mask=motion_padding_mask, memory_key_padding_mask=~gloss_attention_mask.bool(),
        )
        return torch.stack([head(hidden) for head in self.heads], dim=1)

    def forward(self, gloss_input_ids, gloss_attention_mask, decoder_tokens, motion_padding_mask):
        memory = self.encode_gloss(gloss_input_ids, gloss_attention_mask)
        return self.decode(decoder_tokens, memory, gloss_attention_mask, motion_padding_mask)


def to_device(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: value.to(DEVICE, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast_context():
    return torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True)


def make_scaler():
    enabled = AMP_DTYPE == torch.float16
    try: return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError: return torch.cuda.amp.GradScaler(enabled=enabled)


def loss_and_counts(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    predictions = logits.argmax(dim=-1); level_losses, level_correct, level_total = [], [], []
    for level in range(RVQ_LEVELS):
        level_losses.append(F.cross_entropy(
            logits[:, level].reshape(-1, MOTION_VOCAB_SIZE), targets[:, level].reshape(-1), ignore_index=PAD_ID
        ))
        valid = targets[:, level] != PAD_ID
        level_correct.append(int((predictions[:, level][valid] == targets[:, level][valid]).sum().item()))
        level_total.append(int(valid.sum().item()))
    valid = targets != PAD_ID; motion = targets < 512
    return torch.stack(level_losses).mean(), {
        "level_losses": [float(x.detach()) for x in level_losses],
        "level_correct": level_correct, "level_total": level_total,
        "correct": int((predictions[valid] == targets[valid]).sum().item()), "total": int(valid.sum().item()),
        "motion_correct": int((predictions[motion] == targets[motion]).sum().item()),
        "motion_total": int(motion.sum().item()),
    }


@torch.no_grad()
def evaluate_teacher_forced(
    model: GlossMotionTransformer, loader: DataLoader, collect_distributions: bool = False
) -> tuple[dict[str, Any], Optional[list[dict[str, Counter]]]]:
    model.eval(); level_loss_numer = np.zeros(RVQ_LEVELS); level_correct = np.zeros(RVQ_LEVELS)
    level_total = np.zeros(RVQ_LEVELS); correct = total = motion_correct = motion_total = 0
    distributions = [{"target": Counter(), "prediction": Counter()} for _ in range(RVQ_LEVELS)] if collect_distributions else None
    for raw_batch in loader:
        batch = to_device(raw_batch)
        with autocast_context():
            logits = model(batch["gloss_input_ids"], batch["gloss_attention_mask"],
                           batch["decoder_tokens"], batch["motion_padding_mask"])
            _, info = loss_and_counts(logits, batch["target_tokens"])
        for level in range(RVQ_LEVELS):
            level_loss_numer[level] += info["level_losses"][level] * info["level_total"][level]
        level_correct += info["level_correct"]; level_total += info["level_total"]
        correct += info["correct"]; total += info["total"]
        motion_correct += info["motion_correct"]; motion_total += info["motion_total"]
        if distributions is not None:
            predictions = logits.argmax(-1).detach().cpu(); targets = batch["target_tokens"].detach().cpu()
            for level in range(RVQ_LEVELS):
                mask = targets[:, level] < 512
                distributions[level]["target"].update(targets[:, level][mask].tolist())
                distributions[level]["prediction"].update(predictions[:, level][mask].tolist())
    level_losses = level_loss_numer / np.maximum(level_total, 1)
    metrics = {
        "loss": float(level_losses.mean()), "level_losses": level_losses.tolist(),
        "overall_accuracy": correct / max(total, 1), "motion_only_accuracy": motion_correct / max(motion_total, 1),
        "per_level_accuracy": (level_correct / np.maximum(level_total, 1)).tolist(),
    }
    return metrics, distributions


def finish_optimizer_step(model, optimizer, scaler, scheduler, accumulated_examples: int) -> float:
    scaler.unscale_(optimizer)
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            parameter.grad.div_(max(accumulated_examples, 1))
    grad_norm = float(torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], CFG.max_grad_norm
    ))
    if not np.isfinite(grad_norm): raise FloatingPointError(f"Non-finite gradient norm: {grad_norm}")
    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
    if scheduler is not None: scheduler.step()
    return grad_norm


def train_one_epoch(model, loader, sampler, optimizer, scaler, scheduler, epoch: int) -> dict[str, Any]:
    sampler.set_epoch(epoch); model.train(); optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(); started = time.time()
    loss_numer = 0.0; example_total = 0; level_loss_numer = np.zeros(RVQ_LEVELS)
    accumulated_examples = 0; optimizer_steps = 0; last_grad_norm = float("nan")
    for batch_index, raw_batch in enumerate(loader):
        batch = to_device(raw_batch); batch_size = len(raw_batch["sample_id"])
        with autocast_context():
            logits = model(batch["gloss_input_ids"], batch["gloss_attention_mask"],
                           batch["decoder_tokens"], batch["motion_padding_mask"])
            loss, info = loss_and_counts(logits, batch["target_tokens"])
        if not torch.isfinite(loss): raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {batch_index}")
        scaler.scale(loss * batch_size).backward()
        accumulated_examples += batch_size; example_total += batch_size
        loss_numer += float(loss.detach()) * batch_size
        level_loss_numer += np.asarray(info["level_losses"]) * batch_size
        if accumulated_examples >= CFG.effective_batch_size or batch_index + 1 == len(loader):
            last_grad_norm = finish_optimizer_step(model, optimizer, scaler, scheduler, accumulated_examples)
            optimizer_steps += 1; accumulated_examples = 0
    return {
        "train_loss": loss_numer / max(example_total, 1),
        "train_level_losses": (level_loss_numer / max(example_total, 1)).tolist(),
        "optimizer_steps": optimizer_steps, "grad_norm": last_grad_norm,
        "learning_rate": optimizer.param_groups[0]["lr"], "epoch_seconds": time.time() - started,
        "gpu_peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }


supported_frames = {
    name: frame.loc[~frame["long_sequence_excluded"]].copy().reset_index(drop=True)
    for name, frame in splits.items()
}

# %% [markdown]
# ## 4C.3 Mandatory six-check smoke gate
#
# This cell performs artifact verification, coverage accounting, the `T=512` positional test,
# batching/masking tests, one real backward update, and a short one-epoch subset pipeline. It saves
# a report and destroys the smoke model. No smoke weight is reused for full training.

# %%
SMOKE_CHECKS: dict[str, bool] = {
    "artifact_verification": True,
    "supported_range_count": sum(len(frame) for frame in supported_frames.values()) + len(policy_df) == sum(EXPECTED_SPLIT_SIZES.values()),
}

smoke_train = supported_frames["train"].sort_values(["motion_length", ID_COL]).iloc[
    np.linspace(0, len(supported_frames["train"]) - 1, CFG.smoke_train_samples).round().astype(int)
].drop_duplicates(ID_COL).reset_index(drop=True)
smoke_validation = supported_frames["validation"].sort_values(["motion_length", ID_COL]).iloc[
    np.linspace(0, len(supported_frames["validation"]) - 1, CFG.smoke_validation_samples).round().astype(int)
].drop_duplicates(ID_COL).reset_index(drop=True)
smoke_train_loader, smoke_train_sampler = make_loader(smoke_train, GLOSS_COL, True, 101)
smoke_val_loader, _ = make_loader(smoke_validation, GLOSS_COL, False, 102)

set_seed(CFG.seed + 100)
smoke_model = GlossMotionTransformer(load_frozen_encoder()).to(DEVICE)
assert all(not parameter.requires_grad for parameter in smoke_model.gloss_encoder.parameters())

# Synthetic maximum-length target: decoder tensors must be [1,6,513], while embedding capacity is 514.
synthetic = [{
    "sample_id": "synthetic_t512", "sentence": "", "ground_truth_gloss": "ME WANT WATER DRINK//",
    "conditioning_gloss": "ME WANT WATER DRINK//",
    "motion_tokens": np.tile(np.arange(512, dtype=np.int64)[None, :], (RVQ_LEVELS, 1)),
    "motion_length": 512,
}]
synthetic_batch_cpu = collate_motion(synthetic)
assert synthetic_batch_cpu["decoder_tokens"].shape == (1, RVQ_LEVELS, 513)
with torch.no_grad(), autocast_context():
    synthetic_batch = to_device(synthetic_batch_cpu)
    synthetic_logits = smoke_model(
        synthetic_batch["gloss_input_ids"], synthetic_batch["gloss_attention_mask"],
        synthetic_batch["decoder_tokens"], synthetic_batch["motion_padding_mask"],
    )
assert synthetic_logits.shape == (1, RVQ_LEVELS, 513, MOTION_VOCAB_SIZE)
SMOKE_CHECKS["t512_positional_capacity"] = True
del synthetic_batch, synthetic_batch_cpu, synthetic_logits; torch.cuda.empty_cache()

# Dynamic padding, shift, common EOS, and mask assertions on a real adaptive batch.
mask_batch = next(iter(smoke_train_loader))
for index, length in enumerate(mask_batch["original_motion_length"].tolist()):
    assert (mask_batch["decoder_tokens"][index, :, 0] == BOS_ID).all()
    assert (mask_batch["target_tokens"][index, :, length] == EOS_ID).all()
    assert not mask_batch["motion_padding_mask"][index, :length + 1].any()
    assert mask_batch["motion_padding_mask"][index, length + 1:].all()
assert mask_batch["decoder_tokens"].shape[-1] == int(mask_batch["original_motion_length"].max()) + 1
SMOKE_CHECKS["batching_and_masking"] = True

# One complete forward/backward/optimizer batch.
smoke_optimizer = torch.optim.AdamW(
    [p for p in smoke_model.parameters() if p.requires_grad], lr=CFG.learning_rate, weight_decay=CFG.weight_decay
)
smoke_scaler = make_scaler(); smoke_optimizer.zero_grad(set_to_none=True)
one_batch = to_device(mask_batch)
with autocast_context():
    one_logits = smoke_model(one_batch["gloss_input_ids"], one_batch["gloss_attention_mask"],
                             one_batch["decoder_tokens"], one_batch["motion_padding_mask"])
    one_loss, _ = loss_and_counts(one_logits, one_batch["target_tokens"])
assert torch.isfinite(one_loss)
smoke_scaler.scale(one_loss).backward(); smoke_scaler.unscale_(smoke_optimizer)
one_grad_norm = float(torch.nn.utils.clip_grad_norm_(
    [p for p in smoke_model.parameters() if p.requires_grad], CFG.max_grad_norm
))
assert np.isfinite(one_grad_norm) and one_grad_norm > 0
smoke_scaler.step(smoke_optimizer); smoke_scaler.update(); smoke_optimizer.zero_grad(set_to_none=True)
SMOKE_CHECKS["one_forward_backward_batch"] = True
del one_batch, one_logits

# One short subset epoch, including oracle validation.
smoke_epoch = train_one_epoch(
    smoke_model, smoke_train_loader, smoke_train_sampler, smoke_optimizer, smoke_scaler, None, epoch=1
)
smoke_validation_metrics, _ = evaluate_teacher_forced(smoke_model, smoke_val_loader)
SMOKE_CHECKS["one_epoch_pipeline"] = bool(
    np.isfinite(smoke_epoch["train_loss"]) and np.isfinite(smoke_validation_metrics["loss"])
)
SMOKE_PASSED = all(SMOKE_CHECKS.values())
SMOKE_REPORT = {
    "phase": "Phase 4C smoke gate", "checks": SMOKE_CHECKS, "passed": SMOKE_PASSED,
    "coverage": COVERAGE_REPORT, "synthetic_t512_decoder_positions": 513,
    "max_model_positions": CFG.max_model_positions, "batch_strategy": {
        "boundaries": CFG.bucket_boundaries, "batch_sizes": CFG.bucket_batch_sizes,
        "effective_batch_size": CFG.effective_batch_size,
    },
    "encoder_cache": {"enabled": CFG.cache_encoder_outputs,
                      "policy": "on-the-fly frozen encoding; avoids a large variable-length cache"},
    "amp_dtype": str(AMP_DTYPE), "one_batch_loss": float(one_loss.detach()),
    "one_batch_grad_norm": one_grad_norm, "smoke_epoch": smoke_epoch,
    "smoke_validation": smoke_validation_metrics,
}
write_json(DIAGNOSTICS / "phase4c_smoke_report.json", SMOKE_REPORT)
print("\nPHASE 4C SIX-CHECK SMOKE GATE:", "PASS" if SMOKE_PASSED else "FAIL")
display(pd.Series(SMOKE_CHECKS, name="passed").to_frame())
print("Full training enabled:", CFG.run_full_training)
del smoke_model, smoke_optimizer, smoke_scaler, one_loss, mask_batch
gc.collect(); torch.cuda.empty_cache()
if not SMOKE_PASSED:
    raise RuntimeError("Phase 4C smoke gate failed. Do not start full training.")

# %% [markdown]
# ## 4C.4 Autoregressive evaluation and Phase-5 token serialization
#
# Greedy inference is identical to Phase 4B: synchronized EOS, suppressed PAD/BOS, replacement of
# partial EOS by the best real codebook ID, and only `0..511` retained. At the supported boundary,
# one final EOS decision is allowed after 512 real steps; otherwise termination is marked as max-length.

# %%
@torch.no_grad()
def generate_batch(model: GlossMotionTransformer, raw_batch: dict[str, Any]) -> list[dict[str, Any]]:
    model.eval(); batch = to_device(raw_batch)
    with autocast_context(): memory = model.encode_gloss(batch["gloss_input_ids"], batch["gloss_attention_mask"])
    batch_size = len(raw_batch["sample_id"])
    decoder_tokens = torch.full((batch_size, RVQ_LEVELS, 1), BOS_ID, dtype=torch.long, device=DEVICE)
    padding_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=DEVICE)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)
    eos_reached = torch.zeros_like(finished); max_terminated = torch.zeros_like(finished)
    generated: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
    for step in range(CFG.max_generation_t + 1):
        with autocast_context():
            last = model.decode(decoder_tokens, memory, batch["gloss_attention_mask"], padding_mask)[:, :, -1].float()
        last[:, :, PAD_ID] = -torch.inf; last[:, :, BOS_ID] = -torch.inf
        choices = last.argmax(-1); all_eos = (choices == EOS_ID).all(1) & ~finished
        eos_reached |= all_eos; finished |= all_eos
        if step == CFG.max_generation_t:
            max_terminated |= ~finished; finished[:] = True
            break
        real_choices = last[:, :, :512].argmax(-1)
        choices = torch.where((choices == EOS_ID) & ~all_eos[:, None], real_choices, choices)
        append_tokens = torch.full_like(choices, PAD_ID); active = ~finished
        append_tokens[active] = choices[active]
        for index in range(batch_size):
            if bool(active[index]): generated[index].append(choices[index].detach().cpu())
        decoder_tokens = torch.cat([decoder_tokens, append_tokens.unsqueeze(-1)], dim=-1)
        padding_mask = torch.cat([padding_mask, finished.unsqueeze(-1)], dim=-1)
        if bool(finished.all()): break
    outputs = []
    for index in range(batch_size):
        tokens = (torch.stack(generated[index], dim=1).numpy().astype(np.int64)
                  if generated[index] else np.empty((RVQ_LEVELS, 0), dtype=np.int64))
        assert tokens.shape[0] == RVQ_LEVELS and tokens.shape[1] <= CFG.max_generation_t
        if tokens.size: assert tokens.min() >= 0 and tokens.max() <= 511
        outputs.append({
            "tokens": tokens, "eos_reached": bool(eos_reached[index].item()),
            "max_length_termination": bool(max_terminated[index].item()),
        })
    return outputs


@torch.no_grad()
def gloss_to_motion_tokens(model: GlossMotionTransformer, gloss: str) -> dict[str, Any]:
    """Generate one `[6,T]` real-codebook array from an already-produced gloss string."""
    encoded = gloss_tokenizer([str(gloss)], padding=True, truncation=False, return_tensors="pt")
    output = generate_batch(model, {
        "sample_id": ["inference"], "conditioning_gloss": [str(gloss)],
        "gloss_input_ids": encoded["input_ids"], "gloss_attention_mask": encoded["attention_mask"],
    })[0]
    assert output["tokens"].shape[0] == RVQ_LEVELS
    return output


try:
    from rapidfuzz.distance import Levenshtein as _RapidLevenshtein
    def edit_distance(reference: list[int], prediction: list[int]) -> int:
        return int(_RapidLevenshtein.distance(reference, prediction))
except ImportError:
    def edit_distance(reference: list[int], prediction: list[int]) -> int:
        previous = list(range(len(prediction) + 1))
        for i, ref in enumerate(reference, 1):
            current = [i]
            for j, pred in enumerate(prediction, 1):
                current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + int(ref != pred)))
            previous = current
        return previous[-1]


def spearman(x: pd.Series, y: pd.Series) -> float:
    valid = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(valid) < 2 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2: return float("nan")
    return float(valid["x"].rank(method="average").corr(valid["y"].rank(method="average")))


def autoregressive_evaluate(
    model: GlossMotionTransformer, loader: DataLoader, label: str
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, np.ndarray]]:
    rows, token_map = [], {}; per_level_correct = np.zeros(RVQ_LEVELS); per_level_total = np.zeros(RVQ_LEVELS)
    micro_correct = micro_total = invalid = generated_total = 0
    for batch_number, raw_batch in enumerate(loader, 1):
        outputs = generate_batch(model, raw_batch)
        for index, output in enumerate(outputs):
            sample_id = str(raw_batch["sample_id"][index]); reference_length = int(raw_batch["original_motion_length"][index])
            reference = raw_batch["target_tokens"][index, :, :reference_length].numpy()
            prediction = output["tokens"]; token_map[sample_id] = prediction
            generated_length = prediction.shape[1]; aligned = min(reference_length, generated_length)
            level_accuracies, normalized_edits = [], []
            for level in range(RVQ_LEVELS):
                correct = int((reference[level, :aligned] == prediction[level, :aligned]).sum()) if aligned else 0
                per_level_correct[level] += correct; per_level_total[level] += aligned
                level_accuracies.append(correct / max(aligned, 1))
                normalized_edits.append(edit_distance(reference[level].tolist(), prediction[level].tolist()) /
                                        max(reference_length, generated_length, 1))
            all_correct = int((reference[:, :aligned] == prediction[:, :aligned]).sum()) if aligned else 0
            micro_correct += all_correct; micro_total += RVQ_LEVELS * aligned
            invalid_here = int(((prediction < 0) | (prediction > 511)).sum())
            invalid += invalid_here; generated_total += prediction.size
            exact = reference.shape == prediction.shape and np.array_equal(reference, prediction)
            rows.append({
                "sample_id": sample_id, "sentence": raw_batch["sentence"][index],
                "ground_truth_gloss": raw_batch["ground_truth_gloss"][index],
                "conditioning_gloss": raw_batch["conditioning_gloss"][index],
                "reference_length": reference_length, "generated_length": generated_length,
                "length_error": generated_length - reference_length,
                "length_absolute_error": abs(generated_length - reference_length),
                "relative_length_error": abs(generated_length - reference_length) / max(reference_length, 1),
                "eos_reached": output["eos_reached"],
                "max_length_termination": output["max_length_termination"],
                "aligned_token_accuracy": all_correct / max(RVQ_LEVELS * aligned, 1),
                **{f"level_{level}_aligned_accuracy": level_accuracies[level] for level in range(RVQ_LEVELS)},
                "mean_normalized_edit_distance": float(np.mean(normalized_edits)),
                "exact_complete_sequence_match": exact, "invalid_id_count": invalid_here,
            })
        if batch_number % 25 == 0: print(f"{label}: generated {len(rows)} samples")
    frame = pd.DataFrame(rows)
    metrics = {
        "samples": len(frame), "aligned_token_accuracy": float(frame["aligned_token_accuracy"].mean()),
        "micro_aligned_token_accuracy": micro_correct / max(micro_total, 1),
        "per_level_prefix_accuracy": (per_level_correct / np.maximum(per_level_total, 1)).tolist(),
        "normalized_token_edit_distance": float(frame["mean_normalized_edit_distance"].mean()),
        "length_mae": float(frame["length_absolute_error"].mean()),
        "length_rmse": float(np.sqrt(np.mean(np.square(frame["length_error"])))),
        "mean_signed_length_error": float(frame["length_error"].mean()),
        "relative_length_error": float(frame["relative_length_error"].mean()),
        "length_spearman": spearman(frame["reference_length"], frame["generated_length"]),
        "eos_success_rate": float(frame["eos_reached"].mean()),
        "max_length_termination_rate": float(frame["max_length_termination"].mean()),
        "invalid_id_rate": invalid / max(generated_total, 1),
        "exact_complete_sequence_match": float(frame["exact_complete_sequence_match"].mean()),
    }
    return metrics, frame, token_map


def save_token_npz(path: Path, ordered_ids: list[str], token_map: dict[str, np.ndarray]) -> None:
    sequences = [token_map[str(sample_id)] for sample_id in ordered_ids]
    lengths = np.asarray([array.shape[1] for array in sequences], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    packed = np.concatenate(sequences, axis=1) if sequences else np.empty((RVQ_LEVELS, 0), dtype=np.int64)
    np.savez_compressed(path, sample_ids=np.asarray(ordered_ids, dtype=str), lengths=lengths,
                        offsets=offsets, tokens_flat=packed)


def validate_token_npz(path: Path, seed: int) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        sample_ids = data["sample_ids"].astype(str); lengths = data["lengths"]
        offsets = data["offsets"]; packed = data["tokens_flat"]
        assert packed.ndim == 2 and packed.shape[0] == RVQ_LEVELS
        assert len(sample_ids) == len(lengths) and len(offsets) == len(sample_ids) + 1
        assert int(offsets[-1]) == packed.shape[1] and np.array_equal(np.diff(offsets), lengths)
        rng = np.random.default_rng(seed); count = min(20, len(sample_ids))
        checked = rng.choice(len(sample_ids), size=count, replace=False)
        for index in checked:
            sequence = packed[:, offsets[index]:offsets[index + 1]]
            assert sequence.shape == (RVQ_LEVELS, int(lengths[index]))
            assert np.issubdtype(sequence.dtype, np.integer)
            if sequence.size: assert sequence.min() >= 0 and sequence.max() <= 511
    return {"path": str(path), "samples": len(sample_ids), "random_samples_checked": count, "passed": True}


def counter_entropy(counter: Counter) -> float:
    counts = np.asarray(list(counter.values()), dtype=np.float64)
    if counts.size == 0: return 0.0
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities + 1e-15)).sum())


def collapse_rows(distributions: list[dict[str, Counter]]) -> list[dict[str, Any]]:
    rows = []
    for level, pair in enumerate(distributions):
        for kind in ("ground_truth", "prediction"):
            counter = pair["target" if kind == "ground_truth" else "prediction"]
            top_id, top_count = counter.most_common(1)[0] if counter else (-1, 0)
            total = sum(counter.values()); unique_real = sum(1 for token in counter if 0 <= token <= 511)
            rows.append({
                "rvq_level": level, "distribution": kind, "unique_ids": unique_real,
                "codebook_usage_percentage": 100.0 * unique_real / 512,
                "top_token_id": top_id, "top_token_frequency": top_count / max(total, 1),
                "entropy_bits": counter_entropy(counter),
                "possible_token_collapse": kind == "prediction" and (unique_real < 8 or top_count / max(total, 1) > 0.50),
            })
    return rows


TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*|[/?.!,;:]", flags=re.UNICODE)


def lexical_tokens(text: Any) -> list[str]:
    return TOKEN_PATTERN.findall(re.sub(r"\s+", " ", str(text)).strip().casefold())


TRAIN_GLOSS_VOCAB = set(token for gloss in splits["train"][GLOSS_COL].astype(str) for token in lexical_tokens(gloss))


def predicted_gloss_anomaly(text: Any) -> str:
    value = str(text); tokens = lexical_tokens(value); flags = []
    if not value.strip(): flags.append("empty")
    if any(token not in TRAIN_GLOSS_VOCAB for token in tokens): flags.append("oov_like")
    if any(tokens[i:i + 4] == [tokens[i]] * 4 for i in range(max(len(tokens) - 3, 0))): flags.append("repeated_token_run")
    if re.search(r"[^\w\s/?.,'!:\-]", value, flags=re.UNICODE): flags.append("strange_formatting")
    return "|".join(flags)


def prepare_phase3_conditioning(split_name: str) -> pd.DataFrame:
    predictions = pd.read_csv(PHASE3_ROOT / "results" / f"{split_name}_predictions.csv", dtype={"sample_id": str})
    predictions["sample_id"] = predictions["sample_id"].astype(str)
    required = {"sample_id", "predicted_gloss", "lexical_token_f1", "lexical_exact_match"}
    if not required.issubset(predictions.columns): raise ValueError(f"Missing Phase-3 columns: {required - set(predictions.columns)}")
    if predictions["sample_id"].duplicated().any(): raise ValueError(f"Duplicate Phase-3 {split_name} IDs")
    base = supported_frames[split_name].copy()
    selected_columns = ["sample_id", "predicted_gloss", "lexical_token_f1", "lexical_exact_match"]
    for optional in ("generated_audit_flags", "audit_flags"):
        if optional in predictions.columns: selected_columns.append(optional)
    merged = base.merge(predictions[selected_columns], left_on=ID_COL, right_on="sample_id", how="left", validate="one_to_one")
    if merged["predicted_gloss"].isna().any():
        raise ValueError(f"Missing Phase-3 {split_name} joins: {merged.loc[merged['predicted_gloss'].isna(), ID_COL].tolist()[:10]}")
    merged["conditioning_gloss"] = merged["predicted_gloss"].astype(str)
    computed = merged["predicted_gloss"].map(predicted_gloss_anomaly)
    if "generated_audit_flags" in merged:
        existing = merged["generated_audit_flags"].fillna("").astype(str)
        merged["phase3_anomaly_flags"] = ["|".join(filter(None, [a, b])) for a, b in zip(existing, computed)]
    else:
        merged["phase3_anomaly_flags"] = computed
    return merged


def write_metric_csv(path: Path, metrics: dict[str, Any]) -> None:
    flattened = {}
    for key, value in metrics.items():
        if isinstance(value, list):
            for index, item in enumerate(value): flattened[f"{key}_L{index}"] = item
        else: flattened[key] = value
    pd.DataFrame([flattened]).to_csv(path, index=False)


def subset_metric_rows(error_frame: pd.DataFrame, split_name: str) -> list[dict[str, Any]]:
    exact = error_frame["phase3_exact_match"].astype(bool)
    f1 = error_frame["phase3_gloss_f1"]
    anomaly = error_frame["phase3_anomaly_flags"].fillna("").ne("")
    definitions = {
        "exact_gloss": exact,
        "high_f1_non_exact": (~exact) & (f1 >= 0.80),
        "medium_f1": (~exact) & (f1 >= 0.40) & (f1 < 0.80),
        "low_f1": f1 < 0.40,
        "normal_prediction": ~anomaly,
        "anomalous_prediction": anomaly,
    }
    rows = []
    for name, mask in definitions.items():
        subset = error_frame.loc[mask]
        rows.append({
            "split": split_name, "subset": name, "n": len(subset),
            "mean_phase3_f1": float(subset["phase3_gloss_f1"].mean()) if len(subset) else None,
            "oracle_token_accuracy": float(subset["oracle_phase4_token_accuracy"].mean()) if len(subset) else None,
            "end2end_token_accuracy": float(subset["end2end_phase4_token_accuracy"].mean()) if len(subset) else None,
            "accuracy_drop": float(subset["accuracy_drop"].mean()) if len(subset) else None,
            "end2end_edit_distance": float(subset["end2end_edit_distance"].mean()) if len(subset) else None,
            "end2end_length_mae": float(subset["end2end_length_absolute_error"].mean()) if len(subset) else None,
            "end2end_eos_success": float(subset["end2end_eos_reached"].mean()) if len(subset) else None,
            "end2end_invalid_ids": int(subset["end2end_invalid_id_count"].sum()) if len(subset) else 0,
        })
    return rows


def build_error_propagation(
    split_name: str, phase3_frame: pd.DataFrame, oracle: pd.DataFrame, end2end: pd.DataFrame,
    oracle_tokens: dict[str, np.ndarray], end2end_tokens: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    phase3_columns = [ID_COL, "predicted_gloss", "lexical_token_f1", "lexical_exact_match", "phase3_anomaly_flags"]
    base = phase3_frame[phase3_columns].rename(columns={
        ID_COL: "sample_id", "lexical_token_f1": "phase3_gloss_f1", "lexical_exact_match": "phase3_exact_match"
    })
    oracle_part = oracle[["sample_id", "aligned_token_accuracy", "mean_normalized_edit_distance",
                          "generated_length", "length_absolute_error"]].rename(columns={
        "aligned_token_accuracy": "oracle_phase4_token_accuracy",
        "mean_normalized_edit_distance": "oracle_edit_distance",
        "generated_length": "oracle_generated_length", "length_absolute_error": "oracle_length_absolute_error",
    })
    end_part = end2end[["sample_id", "aligned_token_accuracy", "mean_normalized_edit_distance",
                        "generated_length", "length_absolute_error", "eos_reached", "invalid_id_count"]].rename(columns={
        "aligned_token_accuracy": "end2end_phase4_token_accuracy",
        "mean_normalized_edit_distance": "end2end_edit_distance",
        "generated_length": "end2end_generated_length", "length_absolute_error": "end2end_length_absolute_error",
        "eos_reached": "end2end_eos_reached", "invalid_id_count": "end2end_invalid_id_count",
    })
    merged = base.merge(oracle_part, on="sample_id", validate="one_to_one").merge(end_part, on="sample_id", validate="one_to_one")
    merged["accuracy_drop"] = merged["oracle_phase4_token_accuracy"] - merged["end2end_phase4_token_accuracy"]
    merged["length_difference"] = merged["end2end_generated_length"] - merged["oracle_generated_length"]
    merged["oracle_end2end_tokens_identical"] = [
        np.array_equal(oracle_tokens[sample_id], end2end_tokens[sample_id]) for sample_id in merged["sample_id"]
    ]
    summary = {
        "split": split_name, "samples": len(merged),
        "spearman_phase3_f1_vs_end2end_accuracy": spearman(merged["phase3_gloss_f1"], merged["end2end_phase4_token_accuracy"]),
        "spearman_phase3_f1_vs_phase4_degradation": spearman(merged["phase3_gloss_f1"], merged["accuracy_drop"]),
        "exact_gloss_output_identity_rate": float(merged.loc[merged["phase3_exact_match"].astype(bool),
                                                               "oracle_end2end_tokens_identical"].mean()),
    }
    return merged, summary

# %% [markdown]
# ## 4C.5 Full training, validation-only selection, frozen evaluation, and handoff
#
# This function is called only when `run_full_training=True`. The checkpoint rule is fixed in
# advance: maximize oracle-validation motion-only accuracy; break ties using lower validation loss.
# After selection, the checkpoint/architecture/greedy decoder are frozen before any test evaluation.

# %%
def run_full_phase4c() -> dict[str, Any]:
    write_json(OUTPUT / "phase4c_config.json", {
        "config": asdict(CFG), "selection_rule": "maximize oracle validation motion-only accuracy; lower loss breaks ties",
        "phase1_fingerprint": EXPECTED_FINGERPRINT,
        "resolved_paths": {"phase1": PHASE1_ROOT, "phase3": PHASE3_ROOT,
                           "phase4a": PHASE4A_ROOT, "phase4b": PHASE4B_ROOT,
                           "original_motion_s": MOTION_S_TRAIN_CSV},
    })
    train_frame, validation_frame, test_frame = (
        supported_frames["train"], supported_frames["validation"], supported_frames["test"]
    )
    train_loader, train_sampler = make_loader(train_frame, GLOSS_COL, True, 201)
    validation_loader, _ = make_loader(validation_frame, GLOSS_COL, False, 202)
    test_loader, _ = make_loader(test_frame, GLOSS_COL, False, 203)

    set_seed(CFG.seed)
    model = GlossMotionTransformer(load_frozen_encoder()).to(DEVICE)
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=CFG.learning_rate, weight_decay=CFG.weight_decay
    )
    estimated_steps_per_epoch = math.ceil(len(train_frame) / CFG.effective_batch_size)
    total_steps = estimated_steps_per_epoch * CFG.max_epochs
    warmup_steps = max(1, round(total_steps * CFG.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = make_scaler()

    probe_frame = validation_frame.sort_values(["motion_length", ID_COL]).iloc[
        np.linspace(0, len(validation_frame) - 1, 3).round().astype(int)
    ].reset_index(drop=True)
    probe_loader, _ = make_loader(probe_frame, GLOSS_COL, False, 204)
    probe_batch = next(iter(probe_loader))

    best_accuracy, best_loss, best_epoch = -1.0, float("inf"), -1
    best_probe_outputs: Optional[list[dict[str, Any]]] = None
    epochs_without_improvement = 0; history = []
    for epoch in range(1, CFG.max_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, train_sampler, optimizer, scaler, scheduler, epoch)
        validation_metrics, _ = evaluate_teacher_forced(model, validation_loader)
        row = {"epoch": epoch, **train_metrics,
               "validation_loss": validation_metrics["loss"],
               "validation_overall_accuracy": validation_metrics["overall_accuracy"],
               "validation_motion_only_accuracy": validation_metrics["motion_only_accuracy"]}
        for level in range(RVQ_LEVELS):
            row[f"train_level_{level}_loss"] = train_metrics["train_level_losses"][level]
            row[f"validation_level_{level}_loss"] = validation_metrics["level_losses"][level]
            row[f"validation_level_{level}_accuracy"] = validation_metrics["per_level_accuracy"][level]
        history.append(row)
        print(
            f"Epoch {epoch:02d} | train_loss={train_metrics['train_loss']:.6f} "
            f"train_levels={[round(x, 5) for x in train_metrics['train_level_losses']]} | "
            f"val_loss={validation_metrics['loss']:.6f} "
            f"val_overall={validation_metrics['overall_accuracy']:.6f} "
            f"val_motion={validation_metrics['motion_only_accuracy']:.6f} "
            f"val_levels={[round(x, 5) for x in validation_metrics['per_level_accuracy']]} | "
            f"lr={train_metrics['learning_rate']:.3e} seconds={train_metrics['epoch_seconds']:.1f} "
            f"peak_GiB={train_metrics['gpu_peak_memory_gib']:.3f}"
        )
        accuracy, loss = validation_metrics["motion_only_accuracy"], validation_metrics["loss"]
        improved = accuracy > best_accuracy + 1e-12 or (abs(accuracy - best_accuracy) <= 1e-12 and loss < best_loss)
        if improved:
            best_accuracy, best_loss, best_epoch = accuracy, loss, epoch; epochs_without_improvement = 0
            motion_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
                            if not name.startswith("gloss_encoder.")}
            torch.save({"motion_state_dict": motion_state, "best_epoch": epoch,
                        "validation_motion_only_accuracy": accuracy, "validation_loss": loss},
                       BEST_MODEL_DIR / "model.pt")
            best_probe_outputs = generate_batch(model, probe_batch)
            print(f"Epoch {epoch:02d}: NEW BEST val_motion={accuracy:.6f} val_loss={loss:.6f}")
            del motion_state
        else:
            epochs_without_improvement += 1
            print(f"Epoch {epoch:02d}: train={train_metrics['train_loss']:.6f} val={loss:.6f} "
                  f"motion={accuracy:.6f} patience={epochs_without_improvement}/{CFG.early_stopping_patience}")
        if epochs_without_improvement >= CFG.early_stopping_patience:
            print("Early stopping triggered using oracle validation only."); break

    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT / "training_history.csv", index=False)
    assert best_epoch > 0 and (BEST_MODEL_DIR / "model.pt").is_file() and best_probe_outputs is not None

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o", label="train")
    axes[0].plot(history_df["epoch"], history_df["validation_loss"], marker="o", label="oracle validation")
    axes[0].set(xlabel="epoch", ylabel="loss", title="Full-training loss"); axes[0].legend()
    axes[1].plot(history_df["epoch"], history_df["validation_motion_only_accuracy"], marker="o")
    axes[1].set(xlabel="epoch", ylabel="accuracy", title="Oracle-validation motion-only accuracy")
    fig.tight_layout(); fig.savefig(FIGURES / "full_training_curves.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 5))
    for level in range(RVQ_LEVELS):
        axis.plot(history_df["epoch"], history_df[f"validation_level_{level}_accuracy"], marker="o", label=f"L{level}")
    axis.set(xlabel="epoch", ylabel="accuracy", title="Per-level oracle-validation accuracy"); axis.legend(ncol=3)
    fig.tight_layout(); fig.savefig(FIGURES / "per_level_validation_accuracy.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    model_config = {
        "d_model": CFG.d_model, "d_level": CFG.d_level, "num_heads": CFG.num_heads,
        "decoder_layers": CFG.decoder_layers, "ffn_dimension": CFG.ffn_dimension, "dropout": CFG.dropout,
        "max_model_positions": CFG.max_model_positions, "gloss_encoder_frozen": True,
        "parameters_total": total_parameters, "parameters_trainable": trainable_parameters,
    }
    token_config = {
        "shape": "[6,T]", "rvq_levels": RVQ_LEVELS, "level_order": LEVEL_ORDER,
        "codebook_sizes": CODEBOOK_SIZES, "valid_ids": [0, 511], "PAD_ID": PAD_ID,
        "BOS_ID": BOS_ID, "EOS_ID": EOS_ID, "motion_vocab_size": MOTION_VOCAB_SIZE,
        "supported_max_t": CFG.supported_max_t,
    }
    dependency_manifest = {
        "phase1_fingerprint": EXPECTED_FINGERPRINT, "phase3_encoder_path": str(PHASE3_MODEL_PATH),
        "phase3_tokenizer_path": str(TOKENIZER_PATH),
        "phase3_model_config_sha256": sha256_file(PHASE3_MODEL_PATH / "config.json"),
        "phase3_tokenizer_config_sha256": sha256_file(TOKENIZER_PATH / "tokenizer_config.json"),
        "checkpoint_contents": "all trainable Phase-4 state except frozen gloss_encoder.*",
    }
    write_json(BEST_MODEL_DIR / "model_config.json", model_config)
    write_json(BEST_MODEL_DIR / "token_config.json", token_config)
    write_json(BEST_MODEL_DIR / "dependency_manifest.json", dependency_manifest)

    # Freeze selection before test: destroy training state and reconstruct from immutable dependencies.
    del model, optimizer, scheduler, scaler; gc.collect(); torch.cuda.empty_cache()
    frozen_model = GlossMotionTransformer(load_frozen_encoder()).to(DEVICE)
    checkpoint = torch.load(BEST_MODEL_DIR / "model.pt", map_location="cpu", weights_only=True)
    incompatible = frozen_model.load_state_dict(checkpoint["motion_state_dict"], strict=False)
    expected_missing = [name for name in frozen_model.state_dict() if name.startswith("gloss_encoder.")]
    assert sorted(incompatible.missing_keys) == sorted(expected_missing) and not incompatible.unexpected_keys
    reloaded_probe_outputs = generate_batch(frozen_model, probe_batch)
    checkpoint_reload_verified = all(
        before["eos_reached"] == after["eos_reached"]
        and before["max_length_termination"] == after["max_length_termination"]
        and np.array_equal(before["tokens"], after["tokens"])
        for before, after in zip(best_probe_outputs, reloaded_probe_outputs)
    )
    RELOAD_REPORT = {"best_epoch": best_epoch, "probe_sample_ids": probe_batch["sample_id"],
                     "identical": checkpoint_reload_verified, "checkpoint_reload_verified": checkpoint_reload_verified}
    write_json(DIAGNOSTICS / "checkpoint_reload_report.json", RELOAD_REPORT)
    if not checkpoint_reload_verified: raise RuntimeError("Fresh checkpoint reload changed deterministic generation")

    # Phase-3 saved predictions are joined after selection and never influence training/checkpoint choice.
    validation_phase3 = prepare_phase3_conditioning("validation")
    test_phase3 = prepare_phase3_conditioning("test")
    validation_end_loader, _ = make_loader(validation_phase3, "conditioning_gloss", False, 301)
    test_end_loader, _ = make_loader(test_phase3, "conditioning_gloss", False, 302)

    oracle_val_teacher, distributions = evaluate_teacher_forced(frozen_model, validation_loader, True)
    oracle_test_teacher, _ = evaluate_teacher_forced(frozen_model, test_loader)
    end_val_teacher, _ = evaluate_teacher_forced(frozen_model, validation_end_loader)
    end_test_teacher, _ = evaluate_teacher_forced(frozen_model, test_end_loader)
    collapse_df = pd.DataFrame(collapse_rows(distributions or []))
    collapse_df.to_csv(DIAGNOSTICS / "token_collapse_report.csv", index=False)

    oracle_val_ar, oracle_val_predictions, oracle_val_tokens = autoregressive_evaluate(
        frozen_model, validation_loader, "validation/oracle"
    )
    end_val_ar, end_val_predictions, end_val_tokens = autoregressive_evaluate(
        frozen_model, validation_end_loader, "validation/end-to-end"
    )
    oracle_test_ar, oracle_test_predictions, oracle_test_tokens = autoregressive_evaluate(
        frozen_model, test_loader, "test/oracle"
    )
    end_test_ar, end_test_predictions, end_test_tokens = autoregressive_evaluate(
        frozen_model, test_end_loader, "test/end-to-end"
    )
    evaluation_bundle = {
        "validation_oracle": (oracle_val_teacher, oracle_val_ar, oracle_val_predictions, oracle_val_tokens, validation_frame),
        "validation_end2end": (end_val_teacher, end_val_ar, end_val_predictions, end_val_tokens, validation_phase3),
        "test_oracle": (oracle_test_teacher, oracle_test_ar, oracle_test_predictions, oracle_test_tokens, test_frame),
        "test_end2end": (end_test_teacher, end_test_ar, end_test_predictions, end_test_tokens, test_phase3),
    }
    npz_reports = []
    for offset, (name, (teacher, autoregressive, predictions, tokens, frame)) in enumerate(evaluation_bundle.items()):
        predictions.to_csv(RESULTS / f"{name}_predictions.csv", index=False)
        write_metric_csv(RESULTS / f"{name}_metrics.csv", {
            **autoregressive, "teacher_forced_loss": teacher["loss"],
            "teacher_forced_overall_accuracy": teacher["overall_accuracy"],
            "teacher_forced_motion_only_accuracy": teacher["motion_only_accuracy"],
            "teacher_forced_per_level_accuracy": teacher["per_level_accuracy"],
        })
        npz_path = RESULTS / f"{name}_tokens.npz"
        ordered_ids = frame[ID_COL].astype(str).tolist()
        save_token_npz(npz_path, ordered_ids, tokens)
        npz_reports.append(validate_token_npz(npz_path, CFG.seed + offset))
    phase5_artifacts_verified = all(report["passed"] for report in npz_reports)
    write_json(DIAGNOSTICS / "phase5_npz_validation_report.json", {
        "format": "sample_ids[N], lengths[N], offsets[N+1], tokens_flat[6,sum(lengths)]",
        "reports": npz_reports, "passed": phase5_artifacts_verified,
    })

    val_error, val_error_summary = build_error_propagation(
        "validation", validation_phase3, oracle_val_predictions, end_val_predictions, oracle_val_tokens, end_val_tokens
    )
    test_error, test_error_summary = build_error_propagation(
        "test", test_phase3, oracle_test_predictions, end_test_predictions, oracle_test_tokens, end_test_tokens
    )
    error_df = pd.concat([val_error.assign(split="validation"), test_error.assign(split="test")], ignore_index=True)
    error_df.to_csv(DIAGNOSTICS / "phase3_error_propagation.csv", index=False)
    subset_df = pd.DataFrame(subset_metric_rows(val_error, "validation") + subset_metric_rows(test_error, "test"))
    subset_df.to_csv(DIAGNOSTICS / "phase3_quality_subset_metrics.csv", index=False)

    comparison_rows = []
    for split_name, oracle_metrics, end_metrics in (
        ("validation", oracle_val_ar, end_val_ar), ("test", oracle_test_ar, end_test_ar)
    ):
        for metric in ("aligned_token_accuracy", "normalized_token_edit_distance", "length_mae",
                       "eos_success_rate", "exact_complete_sequence_match"):
            comparison_rows.append({"split": split_name, "metric": metric,
                                    "oracle": oracle_metrics[metric], "phase3_predicted_gloss": end_metrics[metric],
                                    "oracle_minus_end2end": oracle_metrics[metric] - end_metrics[metric]})
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(RESULTS / "oracle_vs_end2end_comparison.csv", index=False)

    # Qualitative token-level examples only; no claim about visual motion quality.
    ranked = test_error.sort_values("end2end_phase4_token_accuracy").reset_index(drop=True)
    chosen_indices = np.linspace(0, len(ranked) - 1, min(12, len(ranked))).round().astype(int)
    qualitative = ranked.iloc[chosen_indices].copy()
    qualitative["quality_band"] = pd.cut(
        qualitative["end2end_phase4_token_accuracy"], bins=[-0.001, .25, .50, .75, 1.0],
        labels=["poor", "medium", "good", "excellent"]
    ).astype(str)
    oracle_qual_columns = ["sample_id", "sentence", "ground_truth_gloss", "reference_length"] + [
        f"level_{i}_aligned_accuracy" for i in range(RVQ_LEVELS)
    ]
    oracle_qual = oracle_test_predictions[oracle_qual_columns].rename(columns={
        f"level_{i}_aligned_accuracy": f"oracle_level_{i}_accuracy" for i in range(RVQ_LEVELS)
    })
    end_qual = end_test_predictions[["sample_id"] + [f"level_{i}_aligned_accuracy" for i in range(RVQ_LEVELS)]].rename(
        columns={f"level_{i}_aligned_accuracy": f"end2end_level_{i}_accuracy" for i in range(RVQ_LEVELS)}
    )
    qualitative = qualitative.merge(oracle_qual, on="sample_id", how="left", validate="one_to_one").merge(
        end_qual, on="sample_id", how="left", validate="one_to_one"
    )
    qualitative.to_csv(DIAGNOSTICS / "qualitative_token_examples.csv", index=False)
    display(qualitative[["quality_band", "sample_id", "sentence", "ground_truth_gloss", "predicted_gloss",
                         "reference_length", "oracle_generated_length", "end2end_generated_length",
                         "oracle_phase4_token_accuracy", "end2end_phase4_token_accuracy",
                         "oracle_edit_distance", "end2end_edit_distance"]])

    validity_rows = []
    for name, (_, metrics, predictions, _, _) in evaluation_bundle.items():
        validity_rows.append({"evaluation": name, "samples": len(predictions),
                              "eos_success_rate": metrics["eos_success_rate"],
                              "max_length_termination_rate": metrics["max_length_termination_rate"],
                              "invalid_id_rate": metrics["invalid_id_rate"]})
    validity_df = pd.DataFrame(validity_rows)
    validity_df.to_csv(DIAGNOSTICS / "generated_token_validity.csv", index=False)

    # Required figures.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, split_name in zip(axes, ("validation", "test")):
        plotted = comparison_df[(comparison_df["split"] == split_name) & comparison_df["metric"].isin(
            ["aligned_token_accuracy", "eos_success_rate", "exact_complete_sequence_match"]
        )].set_index("metric")[["oracle", "phase3_predicted_gloss"]]
        plotted.plot(kind="bar", ax=axis); axis.set(title=f"{split_name}: oracle vs end-to-end", ylabel="score")
    fig.tight_layout(); fig.savefig(FIGURES / "oracle_vs_end2end.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(oracle_test_predictions["reference_length"], oracle_test_predictions["generated_length"], alpha=.35, s=12)
    axis.plot([0, 512], [0, 512], linestyle="--", color="black"); axis.set(
        xlabel="ground-truth length", ylabel="generated length", title="Oracle test length prediction"
    )
    fig.tight_layout(); fig.savefig(FIGURES / "length_prediction.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(test_error["phase3_gloss_f1"], test_error["end2end_phase4_token_accuracy"], alpha=.35, s=12)
    axis.set(xlabel="Phase-3 lexical token F1", ylabel="end-to-end Phase-4 token accuracy",
             title="Upstream error propagation")
    fig.tight_layout(); fig.savefig(FIGURES / "phase3_f1_vs_phase4_accuracy.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, column, title in zip(axes, ("unique_ids", "top_token_frequency"), ("Unique IDs", "Top-token frequency")):
        collapse_df.pivot(index="rvq_level", columns="distribution", values=column).plot(kind="bar", ax=axis)
        axis.set_title(title)
    fig.tight_layout(); fig.savefig(FIGURES / "token_usage_comparison.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    error_summary = {"validation": val_error_summary, "test": test_error_summary}
    all_invalid_zero = bool((validity_df["invalid_id_rate"] == 0).all())
    ready_for_phase5 = bool(checkpoint_reload_verified and phase5_artifacts_verified and all_invalid_zero)
    report = {
        "phase": "Phase 4C - Full Gloss to Motion Token Training", "phase4c_complete": ready_for_phase5,
        "phase1_fingerprint": EXPECTED_FINGERPRINT, "original_split_sizes": EXPECTED_SPLIT_SIZES,
        "supported_max_t": CFG.supported_max_t,
        "coverage": {
            f"{row['split']}_{key}": row[value] for row in coverage_rows
            for key, value in (("supported", "supported"), ("excluded_gt512", "excluded_gt512"))
        },
        "model": model_config, "best_epoch": best_epoch,
        "best_validation_selection_metric": "oracle_validation_motion_only_accuracy_then_lower_loss",
        "oracle_validation": {"teacher_forced": oracle_val_teacher, "autoregressive": oracle_val_ar},
        "oracle_test": {"teacher_forced": oracle_test_teacher, "autoregressive": oracle_test_ar,
                        "label": "SUPPORTED-RANGE ORACLE-GLOSS TEST PERFORMANCE"},
        "end2end_validation": {"teacher_forced": end_val_teacher, "autoregressive": end_val_ar},
        "end2end_test": {"teacher_forced": end_test_teacher, "autoregressive": end_test_ar},
        "phase3_error_propagation": error_summary,
        "token_validity": {"all_invalid_rates_zero": all_invalid_zero,
                           "eos_success_rates": dict(zip(validity_df["evaluation"], validity_df["eos_success_rate"]))},
        "checkpoint_reload_verified": checkpoint_reload_verified,
        "phase5_artifacts_verified": phase5_artifacts_verified, "ready_for_phase5": ready_for_phase5,
    }
    write_json(OUTPUT / "phase4c_report.json", report)

    readme = f"""PHASE 5 INPUT CONTRACT — MOTION-S / SIGNVRSE

Best Phase-4 checkpoint: {BEST_MODEL_DIR / 'model.pt'}
Architecture: frozen Phase-3 T5-small encoder; 512->256 projection; six Embedding(515,64) tables;
Linear(384->256); four-layer causal Transformer decoder; d_model=256, 8 heads, FFN=1024;
six independent Linear(256->515) heads. Dropout was 0.1 during full training.
Special IDs (model only): PAD=512, BOS=513, EOS=514.
RVQ level order: {LEVEL_ORDER}
Each real codebook size: 512. Valid IDs sent to RVQ-VAE: 0..511 only.
Canonical generated sequence shape: [6,T]. Supported maximum T: 512.
T>512 samples were not truncated; they are listed in diagnostics/phase4c_long_sequence_policy.csv.
Phase-1 fingerprint: {EXPECTED_FINGERPRINT}
Frozen encoder dependency: {PHASE3_MODEL_PATH}
Tokenizer dependency: {TOKENIZER_PATH}

Reload: instantiate GlossMotionTransformer(load_frozen_encoder()), load best_model/model.pt,
then load motion_state_dict with strict=False; only gloss_encoder.* keys may be missing.
gloss_to_motion_tokens(): after reload, call
output = gloss_to_motion_tokens(model, "ME WANT WATER DRINK//"); tokens = output["tokens"].
Pass an already-produced gloss WITHOUT a 'text to gloss:' prefix. The function starts all six levels
with BOS, suppresses PAD/BOS, uses synchronized EOS, and returns only IDs 0..511.

NPZ format: sample_ids[N], lengths[N], offsets[N+1], tokens_flat[6,sum(lengths)].
For sample i: tokens_flat[:, offsets[i]:offsets[i+1]] has shape [6,lengths[i]].
Oracle Phase-5 test decoding input: results/test_oracle_tokens.npz
End-to-end Phase-5 test decoding input: results/test_end2end_tokens.npz

Official Motion-S RVQ-VAE decoder/checkpoint: not embedded here. Use the exact provider artifact
identified by the Motion-S competition/data source and verify its six 512-entry codebooks before decoding.
Never pass PAD/BOS/EOS to that decoder.
"""
    (OUTPUT / "README_phase5.txt").write_text(readme, encoding="utf-8")

    inventory_rows = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "artifact_inventory.csv":
            inventory_rows.append({"relative_path": str(path.relative_to(OUTPUT)), "bytes": path.stat().st_size,
                                   "sha256": sha256_file(path)})
    inventory_rows.append({"relative_path": "artifact_inventory.csv", "bytes": None, "sha256": None})
    pd.DataFrame(inventory_rows).to_csv(OUTPUT / "artifact_inventory.csv", index=False)

    print("\n============================================================")
    print("PHASE 4C COMPLETE")
    print("============================================================")
    print("Best epoch:", best_epoch)
    print("Training samples used:", len(train_frame))
    print("Training >512 excluded:", COVERAGE_REPORT["splits"]["train"]["excluded_gt512"])
    print("Validation coverage:", f"{COVERAGE_REPORT['splits']['validation']['coverage_percentage']:.4f}%")
    print("Test coverage:", f"{COVERAGE_REPORT['splits']['test']['coverage_percentage']:.4f}%")
    print("Oracle validation token accuracy:", oracle_val_ar["aligned_token_accuracy"])
    print("Oracle test token accuracy:", oracle_test_ar["aligned_token_accuracy"])
    print("End-to-end test token accuracy:", end_test_ar["aligned_token_accuracy"])
    print("Oracle -> End-to-end degradation:", oracle_test_ar["aligned_token_accuracy"] - end_test_ar["aligned_token_accuracy"])
    for level, accuracy in enumerate(oracle_test_ar["per_level_prefix_accuracy"]): print(f"Per-level test accuracy L{level}:", accuracy)
    print("Test token edit distance:", oracle_test_ar["normalized_token_edit_distance"])
    print("Length MAE:", oracle_test_ar["length_mae"])
    print("EOS success:", oracle_test_ar["eos_success_rate"])
    print("Invalid token rate:", oracle_test_ar["invalid_id_rate"])
    print("Checkpoint reload:", "PASS" if checkpoint_reload_verified else "FAIL")
    print("Phase-5 token files:", "PASS" if phase5_artifacts_verified else "FAIL")
    print("READY FOR PHASE 5:", "YES" if ready_for_phase5 else "NO")
    print("============================================================")
    return report


if CFG.run_full_training:
    PHASE4C_REPORT = run_full_phase4c()
else:
    PHASE4C_REPORT = None
    print("\nINTENTIONAL STOP AFTER SMOKE GATE.")
    print("Inspect diagnostics/phase4c_smoke_report.json.")
    print("Only after the six checks pass, set Phase4CConfig.run_full_training = True and restart/run all.")

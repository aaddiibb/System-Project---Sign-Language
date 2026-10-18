# %% [markdown]
# # Motion-S Phase 4B — Gloss→Motion-Token Model / 32-Sample Overfit Gate
#
# This continuation consumes a **successful Phase-4A report**. It does not repeat the token
# audit and it never starts full training. It builds the frozen-T5-encoder + causal temporal
# decoder baseline, runs a deterministic 32-training-sample memorization test, evaluates actual
# greedy autoregressive generation, and verifies checkpoint reload identity.

# %% [markdown]
# ## 4B.0 Reproducibility, fixed representation, and configuration
#
# Inputs are the Phase-1 package, Phase-3 model/tokenizer, and Phase-4A report. Output is added
# under `/kaggle/working/motion_s_phase4` without deleting Phase-4A artifacts.

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from torch.utils.data import DataLoader, Dataset

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EXPECTED_FINGERPRINT = "e2acdf4d6dce91775239ad50be3866dbc6f5da94d940204b62456cdb77fc4b0d"
EXPECTED_SPLIT_SIZES = {"train": 9970, "validation": 1246, "test": 1247}
LEVEL_ORDER = ["base_tokens", "residual_1", "residual_2", "residual_3", "residual_4", "residual_5"]
RVQ_LEVELS = 6
CODEBOOK_SIZES = [512] * RVQ_LEVELS
PAD_ID, BOS_ID, EOS_ID = 512, 513, 514
MOTION_VOCAB_SIZE = 515

# Locked from the first Phase-4B run. The retry must use these exact 32 examples;
# none may be removed after inspecting the failed run.
LOCKED_DEBUG_SAMPLE_IDS = [
    "1223901", "1566383", "2374784", "262348", "263350", "296739", "308612", "3151456",
    "318836", "32358", "3620380", "3721935", "3735631", "3735700", "436659", "4850031",
    "49065", "503825", "51518", "5160579", "5418244", "5610323", "6088083", "6125970",
    "6142110", "6201741", "6272475", "684683", "767302", "803092", "847141", "8532486",
]
MICRO_DEBUG_SAMPLE_IDS = ["263350", "3721935", "1566383", "8532486"]


@dataclass(frozen=True)
class Phase4BConfig:
    seed: int = 42
    kaggle_input: str = "/kaggle/input"
    phase1_root: Optional[str] = None
    phase3_root: Optional[str] = None
    phase4a_root: Optional[str] = None
    output_dir: str = "/kaggle/working/motion_s_phase4"
    debug_samples: int = 32
    debug_max_motion_length: int = 181
    batch_size: int = 8
    epochs: int = 180
    micro_epochs: int = 300
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    metrics_every_epochs: int = 10
    d_model: int = 256
    d_level: int = 64
    num_heads: int = 8
    decoder_layers: int = 4
    ffn_dimension: int = 1024
    # Memorization diagnostic only. Reconsider regularization before Phase 4C.
    dropout: float = 0.0
    positional_capacity: int = 512
    max_generate_length: int = 256
    num_workers: int = 0
    gate_min_loss_reduction: float = 0.90
    gate_min_teacher_accuracy: float = 0.90
    gate_min_motion_only_accuracy: float = 0.90
    gate_min_autoregressive_aligned_accuracy: float = 0.70
    gate_min_eos_success_rate: float = 0.80


CFG = Phase4BConfig()
assert PAD_ID == 512 and BOS_ID == 513 and EOS_ID == 514
assert PAD_ID > 511 and BOS_ID > 511 and EOS_ID > 511
assert len({PAD_ID, BOS_ID, EOS_ID}) == 3
assert MOTION_VOCAB_SIZE == 515 and RVQ_LEVELS == len(LEVEL_ORDER) == len(CODEBOOK_SIZES)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(CFG.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT = Path(CFG.output_dir)
DIAGNOSTICS, FIGURES, DEBUG_MODEL_DIR = OUTPUT / "diagnostics", OUTPUT / "figures", OUTPUT / "debug_model"
for directory in (OUTPUT, DIAGNOSTICS, FIGURES):
    directory.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=json_default, allow_nan=False)


print("Python:", sys.version.split()[0], "PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), "memory GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
else:
    raise RuntimeError("Phase 4B requires a Kaggle GPU accelerator")
display(pd.Series(asdict(CFG), name="value").to_frame())

# %% [markdown]
# ## 4B.1 Locate verified inputs and enforce the Phase-4A gate
#
# Phase 4B trusts no inferred representation. The prior report must say ready, contain the locked
# fingerprint, `[L,T]`, six levels, and six 512-entry codebooks. Only the train split is loaded.

# %%
def is_phase1(path: Path) -> bool:
    return all(p.is_file() for p in (path / "dataset_config.json", path / "sample_manifest.csv", path / "splits" / "train_split.csv"))


def is_phase3(path: Path) -> bool:
    return all(p.is_file() for p in (
        path / "phase3_report.json", path / "phase3_config.json", path / "best_model" / "config.json",
        path / "tokenizer" / "tokenizer_config.json",
    ))


def is_phase4a(path: Path) -> bool:
    return (path / "phase4a_verification_report.json").is_file()


def locate(kind: str, override: Optional[str], predicate, working: Optional[Path] = None) -> Path:
    if override:
        path = Path(override)
        if not predicate(path): raise FileNotFoundError(f"Invalid {kind}: {path}")
        return path.resolve()
    if working is not None and predicate(working):
        return working.resolve()
    candidates = []
    root = Path(CFG.kaggle_input)
    if root.exists():
        for current, directories, _ in os.walk(root):
            path = Path(current)
            if predicate(path):
                candidates.append(path.resolve()); directories[:] = []
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one {kind}, found {[str(x) for x in candidates]}; set its config path")
    return candidates[0]


PHASE1_ROOT = locate("Phase-1 root", CFG.phase1_root, is_phase1, Path("/kaggle/working/motion_s_phase1"))
PHASE3_ROOT = locate("Phase-3 root", CFG.phase3_root, is_phase3, Path("/kaggle/working/motion_s_phase3"))
PHASE4A_ROOT = locate("Phase-4A root", CFG.phase4a_root, is_phase4a, OUTPUT)
with (PHASE4A_ROOT / "phase4a_verification_report.json").open(encoding="utf-8") as handle:
    phase4a = json.load(handle)
representation = phase4a["motion_token_representation"]
assert phase4a["ready_to_build_motion_model"] is True
assert phase4a["phase1_split_fingerprint_sha256"] == EXPECTED_FINGERPRINT
assert representation["shape_convention"] == "[L,T]"
assert representation["rvq_levels"] == RVQ_LEVELS
assert representation["codebook_sizes"] == CODEBOOK_SIZES
assert representation["level_order"] == LEVEL_ORDER
assert phase4a["token_integrity"]["invalid_ids"] == 0
assert phase4a["token_integrity"]["missing"] == 0
print("Phase 4A prerequisite: PASS")
print("Phase-1:", PHASE1_ROOT, "Phase-3:", PHASE3_ROOT, "Phase-4A:", PHASE4A_ROOT)

with (PHASE1_ROOT / "dataset_config.json").open(encoding="utf-8") as handle:
    phase1_config = json.load(handle)
ID_COL, GLOSS_COL = phase1_config["id_column"], phase1_config["gloss_column"]
TOKEN_COLS = list(phase1_config["motion_token_columns"])
assert TOKEN_COLS == LEVEL_ORDER
train_df = pd.read_csv(PHASE1_ROOT / "splits" / "train_split.csv", dtype={ID_COL: str}, low_memory=False)
manifest_df = pd.read_csv(PHASE1_ROOT / "sample_manifest.csv", dtype={"sample_id": str}, low_memory=False)
assert len(train_df) == EXPECTED_SPLIT_SIZES["train"] and train_df[ID_COL].is_unique
assert phase1_config["split_fingerprint_sha256"] == EXPECTED_FINGERPRINT

# %% [markdown]
# ## 4B.2 Long-sequence report and deterministic, length-diverse debug set
#
# No sequence is truncated. Counts use all 12,463 manifest rows. Phase 4B selects exactly 32
# train samples with `T≤181`, stratified across eight motion-length quantile bins and spread by
# gloss length inside each bin.

# %%
all_lengths = pd.to_numeric(manifest_df["token_length"], errors="raise").astype(int)
thresholds = [181, 256, 320, 384, 512]
LONG_SEQUENCE_REPORT = {
    "total_samples": len(all_lengths),
    "length_statistics": representation["sequence_length_statistics"],
    "counts": {
        f"T_gt_{threshold}": {"count": int((all_lengths > threshold).sum()), "percentage": float(100 * (all_lengths > threshold).mean())}
        for threshold in thresholds
    },
    "phase4b_policy": "debug selection only uses T<=181; no target is truncated",
}
write_json(DIAGNOSTICS / "long_sequence_report.json", LONG_SEQUENCE_REPORT)
display(pd.DataFrame(LONG_SEQUENCE_REPORT["counts"]).T)


def parse_motion_tokens(row: pd.Series) -> np.ndarray:
    levels = []
    for column in TOKEN_COLS:
        values = np.asarray(json.loads(row[column]), dtype=np.int64)
        if values.ndim != 1 or values.size == 0: raise ValueError(f"Bad {column} for {row[ID_COL]}")
        levels.append(values)
    if len({len(level) for level in levels}) != 1: raise ValueError(f"Level length mismatch: {row[ID_COL]}")
    tokens = np.stack(levels, axis=0)
    assert tokens.shape[0] == RVQ_LEVELS and tokens.min() >= 0 and tokens.max() <= 511
    assert not np.isin(tokens, [PAD_ID, BOS_ID, EOS_ID]).any()
    return tokens


train_df["motion_length_verified"] = train_df["token_length"].astype(int)
train_df["gloss_token_count"] = train_df[GLOSS_COL].map(lambda x: len(re.sub(r"\s+", " ", str(x)).strip().split()))
locked_lookup = train_df.assign(_phase4b_id=train_df[ID_COL].astype(str)).set_index("_phase4b_id", drop=False)
missing_locked_ids = sorted(set(LOCKED_DEBUG_SAMPLE_IDS) - set(locked_lookup.index))
assert not missing_locked_ids, f"Locked Phase-4B samples missing from Phase-1 train split: {missing_locked_ids}"
debug_df = locked_lookup.loc[LOCKED_DEBUG_SAMPLE_IDS].copy().reset_index(drop=True)
assert len(debug_df) == 32 and debug_df[ID_COL].is_unique
assert debug_df[ID_COL].astype(str).tolist() == LOCKED_DEBUG_SAMPLE_IDS
assert debug_df["motion_length_verified"].max() <= 181
debug_df["motion_tokens"] = [parse_motion_tokens(row) for _, row in debug_df.iterrows()]
assert all(tokens.shape == (RVQ_LEVELS, length) for tokens, length in zip(debug_df["motion_tokens"], debug_df["motion_length_verified"]))
debug_manifest = debug_df[[ID_COL, GLOSS_COL, "gloss_token_count", "motion_length_verified"]].rename(columns={
    ID_COL: "sample_id", GLOSS_COL: "gloss", "motion_length_verified": "motion_length"
})
debug_manifest.to_csv(DIAGNOSTICS / "debug_sample_manifest.csv", index=False)
display(debug_manifest)

duplicate_glosses = debug_manifest[debug_manifest.duplicated("gloss", keep=False)].sort_values("gloss")
CONDITIONING_REPORT = {
    "locked_manifest": True,
    "locked_sample_ids": LOCKED_DEBUG_SAMPLE_IDS,
    "duplicate_gloss_row_count": int(len(duplicate_glosses)),
    "duplicate_gloss_group_count": int(duplicate_glosses["gloss"].nunique()),
    "note": (
        "Duplicate gloss strings are retained. Identical conditioning with different targets makes exact "
        "deterministic memorization ambiguous for those rows, but does not justify changing the 32-sample gate."
    ),
}
write_json(DIAGNOSTICS / "debug_conditioning_report.json", CONDITIONING_REPORT)

# %% [markdown]
# ## 4B.3 Phase-3 tokenizer/encoder and dynamically padded dataset
#
# Ground-truth gloss enters without the English→Gloss prefix. The tokenizer is loaded from
# Phase 3 and the T5 decoder is discarded. Encoder output `[B,Tg,512]` is frozen; only its
# projection and the motion model train.

# %%
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError as error:
    raise ImportError("Kaggle image must provide transformers") from error

TOKENIZER_PATH, PHASE3_MODEL_PATH = PHASE3_ROOT / "tokenizer", PHASE3_ROOT / "best_model"
gloss_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True, use_fast=True)


def load_frozen_encoder() -> nn.Module:
    seq2seq = AutoModelForSeq2SeqLM.from_pretrained(PHASE3_MODEL_PATH, local_files_only=True)
    encoder = seq2seq.get_encoder()
    for parameter in encoder.parameters(): parameter.requires_grad = False
    encoder.eval()
    del seq2seq
    return encoder


class GlossMotionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame): self.frame = frame.reset_index(drop=True)
    def __len__(self): return len(self.frame)
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        return {"sample_id": str(row[ID_COL]), "gloss": str(row[GLOSS_COL]), "motion_tokens": row["motion_tokens"]}


def collate_debug(batch: list[dict[str, Any]]) -> dict[str, Any]:
    gloss_batch = gloss_tokenizer(
        [item["gloss"] for item in batch], padding=True, truncation=False, return_tensors="pt"
    )
    original_lengths = torch.tensor([item["motion_tokens"].shape[1] for item in batch], dtype=torch.long)
    temporal = int(original_lengths.max().item()) + 1
    decoder_tokens = torch.full((len(batch), RVQ_LEVELS, temporal), PAD_ID, dtype=torch.long)
    target_tokens = torch.full_like(decoder_tokens, PAD_ID)
    motion_padding_mask = torch.ones((len(batch), temporal), dtype=torch.bool)  # True means ignored.
    for batch_index, item in enumerate(batch):
        target = torch.from_numpy(item["motion_tokens"]).long()
        length = target.shape[1]
        decoder_tokens[batch_index, :, 0] = BOS_ID
        decoder_tokens[batch_index, :, 1:length + 1] = target
        target_tokens[batch_index, :, :length] = target
        target_tokens[batch_index, :, length] = EOS_ID
        motion_padding_mask[batch_index, :length + 1] = False
    return {
        "sample_id": [item["sample_id"] for item in batch], "gloss": [item["gloss"] for item in batch],
        "gloss_input_ids": gloss_batch["input_ids"], "gloss_attention_mask": gloss_batch["attention_mask"],
        "decoder_tokens": decoder_tokens, "target_tokens": target_tokens,
        "motion_padding_mask": motion_padding_mask, "original_motion_length": original_lengths,
    }


generator = torch.Generator().manual_seed(CFG.seed)
debug_loader = DataLoader(
    GlossMotionDataset(debug_df), batch_size=CFG.batch_size, shuffle=True, collate_fn=collate_debug,
    num_workers=CFG.num_workers, generator=generator,
)
debug_eval_loader = DataLoader(
    GlossMotionDataset(debug_df), batch_size=CFG.batch_size, shuffle=False, collate_fn=collate_debug,
    num_workers=CFG.num_workers,
)
unit_batch = next(iter(debug_eval_loader))
for index, length in enumerate(unit_batch["original_motion_length"].tolist()):
    reference = torch.from_numpy(debug_df.iloc[index]["motion_tokens"])
    assert torch.equal(unit_batch["decoder_tokens"][index, :, 0], torch.full((6,), BOS_ID))
    assert torch.equal(unit_batch["decoder_tokens"][index, :, 1:length + 1], reference)
    assert torch.equal(unit_batch["target_tokens"][index, :, :length], reference)
    assert torch.equal(unit_batch["target_tokens"][index, :, length], torch.full((6,), EOS_ID))
    assert not unit_batch["motion_padding_mask"][index, :length + 1].any()
    assert unit_batch["motion_padding_mask"][index, length + 1:].all()
assert not torch.isin(unit_batch["target_tokens"][unit_batch["target_tokens"] < 512], torch.tensor([512, 513, 514])).any()
print("Target shift, synchronized BOS/EOS, and PAD-mask unit tests: PASS")

# %% [markdown]
# ## 4B.4 Causal temporal model with six independent levels
#
# Separate `Embedding(515,64)` tables are concatenated in verified level order and projected to
# `[B,T,256]`. A learned 512-position embedding is added. PyTorch's batch-first decoder receives
# an explicit boolean future mask, motion padding mask, and inverted T5 attention mask. Six
# independent heads return logits ordered `[B,6,T,515]`.

# %%
class GlossMotionTransformer(nn.Module):
    def __init__(self, gloss_encoder: nn.Module):
        super().__init__()
        self.gloss_encoder = gloss_encoder
        encoder_dim = int(gloss_encoder.config.d_model)
        self.encoder_projection = nn.Linear(encoder_dim, CFG.d_model)
        self.level_embeddings = nn.ModuleList([
            nn.Embedding(MOTION_VOCAB_SIZE, CFG.d_level, padding_idx=PAD_ID) for _ in range(RVQ_LEVELS)
        ])
        self.motion_projection = nn.Linear(RVQ_LEVELS * CFG.d_level, CFG.d_model)
        self.temporal_positions = nn.Embedding(CFG.positional_capacity, CFG.d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=CFG.d_model, nhead=CFG.num_heads, dim_feedforward=CFG.ffn_dimension,
            dropout=CFG.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=CFG.decoder_layers, norm=nn.LayerNorm(CFG.d_model))
        self.heads = nn.ModuleList([nn.Linear(CFG.d_model, MOTION_VOCAB_SIZE) for _ in range(RVQ_LEVELS)])
        self._initialize_trainable()

    def _initialize_trainable(self):
        for name, parameter in self.named_parameters():
            if name.startswith("gloss_encoder.") or not parameter.requires_grad: continue
            if parameter.ndim > 1: nn.init.xavier_uniform_(parameter)
            elif "norm" in name and name.endswith("weight"): nn.init.ones_(parameter)
            elif name.endswith("bias"): nn.init.zeros_(parameter)
        with torch.no_grad():
            for embedding in self.level_embeddings: embedding.weight[PAD_ID].zero_()

    def train(self, mode: bool = True):
        super().train(mode); self.gloss_encoder.eval(); return self

    def encode_gloss(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            hidden = self.gloss_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return hidden, self.encoder_projection(hidden)

    def embed_motion(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, levels, temporal = tokens.shape
        assert levels == RVQ_LEVELS and temporal <= CFG.positional_capacity
        embedded = torch.cat([self.level_embeddings[level](tokens[:, level]) for level in range(RVQ_LEVELS)], dim=-1)
        positions = torch.arange(temporal, device=tokens.device).unsqueeze(0)
        return self.motion_projection(embedded) + self.temporal_positions(positions)

    def decode(self, decoder_tokens, memory, gloss_attention_mask, motion_padding_mask):
        embedded = self.embed_motion(decoder_tokens)
        temporal = embedded.shape[1]
        causal_mask = torch.triu(torch.ones(temporal, temporal, dtype=torch.bool, device=embedded.device), diagonal=1)
        hidden = self.decoder(
            tgt=embedded, memory=memory, tgt_mask=causal_mask,
            tgt_key_padding_mask=motion_padding_mask,
            memory_key_padding_mask=~gloss_attention_mask.bool(),
        )
        logits = torch.stack([head(hidden) for head in self.heads], dim=1)
        return logits, embedded, hidden

    def forward(self, gloss_input_ids, gloss_attention_mask, decoder_tokens, motion_padding_mask, return_shapes=False):
        encoder_hidden, memory = self.encode_gloss(gloss_input_ids, gloss_attention_mask)
        logits, embedded, decoder_hidden = self.decode(
            decoder_tokens, memory, gloss_attention_mask, motion_padding_mask
        )
        if return_shapes:
            return logits, {
                "encoder_hidden": encoder_hidden, "encoder_projected": memory,
                "decoder_embeddings": embedded, "decoder_hidden": decoder_hidden,
            }
        return logits


shape_model = GlossMotionTransformer(load_frozen_encoder()).to(DEVICE)
shape_model.train()
assert all(not parameter.requires_grad for parameter in shape_model.gloss_encoder.parameters())
assert all(any(parameter.requires_grad for parameter in head.parameters()) for head in shape_model.heads)
total_parameters = sum(parameter.numel() for parameter in shape_model.parameters())
trainable_parameters = sum(parameter.numel() for parameter in shape_model.parameters() if parameter.requires_grad)
print("Parameters total/trainable:", f"{total_parameters:,}", f"{trainable_parameters:,}")


def to_device(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: (value.to(DEVICE, non_blocking=True) if torch.is_tensor(value) else value) for key, value in batch.items()}


shape_batch = to_device(unit_batch)
shape_model.eval()
with torch.no_grad():
    shape_logits, intermediates = shape_model(
        shape_batch["gloss_input_ids"], shape_batch["gloss_attention_mask"],
        shape_batch["decoder_tokens"], shape_batch["motion_padding_mask"], return_shapes=True,
    )
MODEL_SHAPE_REPORT = {
    "gloss_input_ids": list(shape_batch["gloss_input_ids"].shape),
    "gloss_attention_mask": list(shape_batch["gloss_attention_mask"].shape),
    "encoder_hidden": list(intermediates["encoder_hidden"].shape),
    "encoder_projected": list(intermediates["encoder_projected"].shape),
    "decoder_input_tokens": list(shape_batch["decoder_tokens"].shape),
    "decoder_embeddings": list(intermediates["decoder_embeddings"].shape),
    "decoder_hidden": list(intermediates["decoder_hidden"].shape),
    "logits": list(shape_logits.shape), "targets": list(shape_batch["target_tokens"].shape),
    "padding_mask": list(shape_batch["motion_padding_mask"].shape),
    "ordering": {"tokens": "[B,L,T]", "hidden": "[B,T,D]", "logits": "[B,L,T,V]"},
}
assert shape_logits.shape[:3] == shape_batch["target_tokens"].shape and shape_logits.shape[-1] == 515
write_json(DIAGNOSTICS / "model_shape_report.json", MODEL_SHAPE_REPORT)
write_json(DIAGNOSTICS / "special_token_config.json", {
    "pad_id": PAD_ID, "bos_id": BOS_ID, "eos_id": EOS_ID, "motion_vocab_size": MOTION_VOCAB_SIZE,
    "rvq_decoder_policy": "remove PAD/BOS/EOS; only IDs 0..511 may enter the fixed RVQ decoder",
})
display(pd.Series(MODEL_SHAPE_REPORT).to_frame("actual_shape"))

# Causal leakage test: changing future teacher tokens cannot change earlier logits.
probe = {key: (value[:2].clone() if torch.is_tensor(value) else value[:2]) for key, value in shape_batch.items()}
cut = min(5, probe["decoder_tokens"].shape[-1] - 2)
mutated = probe["decoder_tokens"].clone()
future_valid = ~probe["motion_padding_mask"][:, cut + 1:]
replacement = torch.randint(0, 512, mutated[:, :, cut + 1:].shape, device=DEVICE)
mutated[:, :, cut + 1:] = torch.where(future_valid[:, None, :], replacement, mutated[:, :, cut + 1:])
shape_model.eval()
with torch.no_grad():
    original_logits = shape_model(probe["gloss_input_ids"], probe["gloss_attention_mask"], probe["decoder_tokens"], probe["motion_padding_mask"])
    mutated_logits = shape_model(probe["gloss_input_ids"], probe["gloss_attention_mask"], mutated, probe["motion_padding_mask"])
assert torch.allclose(original_logits[:, :, :cut + 1], mutated_logits[:, :, :cut + 1], atol=2e-5, rtol=2e-5)
assert not torch.allclose(
    intermediates["encoder_projected"][0].mean(0), intermediates["encoder_projected"][1].mean(0)
), "Frozen encoder/projection did not distinguish two glosses"
print("Causal-mask and distinct-gloss-representation tests: PASS")
del shape_model, shape_logits, original_logits, mutated_logits, intermediates
gc.collect(); torch.cuda.empty_cache()

# %% [markdown]
# ## 4B.5 Equal-weight six-head loss and teacher-forced metrics
#
# Cross entropy is computed separately for every level and averaged. PAD is ignored. Accuracy
# is reported with EOS included and again for motion IDs only (`0..511`).

# %%
def loss_and_metrics(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    level_losses = []
    per_level_correct, per_level_total = [], []
    predictions = logits.argmax(dim=-1)
    for level in range(RVQ_LEVELS):
        level_loss = F.cross_entropy(
            logits[:, level].reshape(-1, MOTION_VOCAB_SIZE), targets[:, level].reshape(-1), ignore_index=PAD_ID
        )
        level_losses.append(level_loss)
        valid = targets[:, level] != PAD_ID
        per_level_correct.append(int((predictions[:, level][valid] == targets[:, level][valid]).sum().item()))
        per_level_total.append(int(valid.sum().item()))
    loss = torch.stack(level_losses).mean()
    valid_all = targets != PAD_ID
    motion_only = targets < 512
    metrics = {
        "level_losses": [float(value.detach().item()) for value in level_losses],
        "per_level_correct": per_level_correct, "per_level_total": per_level_total,
        "correct": int((predictions[valid_all] == targets[valid_all]).sum().item()),
        "total": int(valid_all.sum().item()),
        "motion_correct": int((predictions[motion_only] == targets[motion_only]).sum().item()),
        "motion_total": int(motion_only.sum().item()),
        "first_step_correct": int((predictions[:, :, 0] == targets[:, :, 0]).sum().item()),
        "first_step_total": int(targets.shape[0] * targets.shape[1]),
    }
    return loss, metrics


def autocast_context():
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def make_scaler():
    try: return torch.amp.GradScaler("cuda", enabled=True)
    except TypeError: return torch.cuda.amp.GradScaler(enabled=True)


@torch.no_grad()
def evaluate_teacher_forced(current_model: nn.Module, evaluation_loader: DataLoader) -> dict[str, Any]:
    current_model.eval()
    loss_sum = 0.0; batches = 0
    level_loss_sums = np.zeros(RVQ_LEVELS); level_correct = np.zeros(RVQ_LEVELS); level_total = np.zeros(RVQ_LEVELS)
    correct = total = motion_correct = motion_total = first_step_correct = first_step_total = 0
    for raw_batch in evaluation_loader:
        batch = to_device(raw_batch)
        with autocast_context():
            logits = current_model(batch["gloss_input_ids"], batch["gloss_attention_mask"], batch["decoder_tokens"], batch["motion_padding_mask"])
            loss, info = loss_and_metrics(logits, batch["target_tokens"])
        loss_sum += float(loss.item()); batches += 1; level_loss_sums += info["level_losses"]
        level_correct += info["per_level_correct"]; level_total += info["per_level_total"]
        correct += info["correct"]; total += info["total"]
        motion_correct += info["motion_correct"]; motion_total += info["motion_total"]
        first_step_correct += info["first_step_correct"]; first_step_total += info["first_step_total"]
    return {
        "loss": loss_sum / batches, "level_losses": (level_loss_sums / batches).tolist(),
        "overall_accuracy": correct / total, "motion_only_accuracy": motion_correct / motion_total,
        "per_level_accuracy": (level_correct / level_total).tolist(),
        "first_step_accuracy": first_step_correct / first_step_total,
    }


@torch.no_grad()
def generate_batch(current_model: GlossMotionTransformer, raw_batch: dict[str, Any]) -> list[dict[str, Any]]:
    current_model.eval(); batch = to_device(raw_batch)
    with autocast_context():
        _, memory = current_model.encode_gloss(batch["gloss_input_ids"], batch["gloss_attention_mask"])
    batch_size = len(raw_batch["sample_id"])
    decoder_tokens = torch.full((batch_size, RVQ_LEVELS, 1), BOS_ID, dtype=torch.long, device=DEVICE)
    padding_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=DEVICE)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)
    eos_reached = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)
    generated: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
    for _ in range(CFG.max_generate_length):
        with autocast_context():
            logits, _, _ = current_model.decode(decoder_tokens, memory, batch["gloss_attention_mask"], padding_mask)
        last = logits[:, :, -1].float(); last[:, :, PAD_ID] = -torch.inf; last[:, :, BOS_ID] = -torch.inf
        choices = last.argmax(dim=-1)
        all_eos = (choices == EOS_ID).all(dim=1) & ~finished
        real_code_choices = last[:, :, :512].argmax(dim=-1)
        choices = torch.where((choices == EOS_ID) & ~all_eos[:, None], real_code_choices, choices)
        append_tokens = torch.full_like(choices, PAD_ID)
        active_tokens = (~finished) & (~all_eos)
        append_tokens[active_tokens] = choices[active_tokens]
        for index in range(batch_size):
            if bool(active_tokens[index]): generated[index].append(choices[index].detach().cpu())
        eos_reached |= all_eos; finished |= all_eos
        decoder_tokens = torch.cat([decoder_tokens, append_tokens.unsqueeze(-1)], dim=-1)
        padding_mask = torch.cat([padding_mask, finished.unsqueeze(-1)], dim=-1)
        if bool(finished.all()): break
    outputs = []
    for index in range(batch_size):
        array = (torch.stack(generated[index], dim=1).numpy().astype(np.int64)
                 if generated[index] else np.empty((RVQ_LEVELS, 0), dtype=np.int64))
        if array.size: assert array.min() >= 0 and array.max() <= 511
        outputs.append({"tokens": array, "eos_reached": bool(eos_reached[index].item())})
    return outputs


@torch.no_grad()
def compact_generation_metrics(current_model: GlossMotionTransformer, evaluation_loader: DataLoader) -> dict[str, float]:
    aligned_scores, eos_flags, length_errors = [], [], []
    invalid = generated_total = 0
    for raw_batch in evaluation_loader:
        for index, output in enumerate(generate_batch(current_model, raw_batch)):
            reference_length = int(raw_batch["original_motion_length"][index])
            reference = raw_batch["target_tokens"][index, :, :reference_length].numpy()
            prediction = output["tokens"]; aligned = min(reference_length, prediction.shape[1])
            correct = int((reference[:, :aligned] == prediction[:, :aligned]).sum()) if aligned else 0
            aligned_scores.append(correct / max(RVQ_LEVELS * aligned, 1))
            eos_flags.append(output["eos_reached"])
            length_errors.append(abs(prediction.shape[1] - reference_length))
            invalid += int(((prediction < 0) | (prediction > 511)).sum())
            generated_total += prediction.size
    return {
        "aligned_token_accuracy": float(np.mean(aligned_scores)),
        "eos_success_rate": float(np.mean(eos_flags)),
        "length_mae": float(np.mean(length_errors)),
        "invalid_id_rate": invalid / max(generated_total, 1),
    }


def train_memorization_run(
    run_model: GlossMotionTransformer,
    training_loader: DataLoader,
    evaluation_loader: DataLoader,
    epochs: int,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], torch.optim.Optimizer, Any]:
    initial = evaluate_teacher_forced(run_model, evaluation_loader)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in run_model.parameters() if parameter.requires_grad],
        lr=CFG.learning_rate, weight_decay=CFG.weight_decay,
    )
    assert optimizer.param_groups and all(
        parameter.requires_grad for group in optimizer.param_groups for parameter in group["params"]
    )
    scaler = make_scaler(); run_history = [{"epoch": 0, **initial, "seconds": 0.0}]
    for epoch in range(1, epochs + 1):
        started = time.time(); run_model.train(); running_loss = 0.0; batches = 0
        for raw_batch in training_loader:
            batch = to_device(raw_batch); optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                logits = run_model(
                    batch["gloss_input_ids"], batch["gloss_attention_mask"],
                    batch["decoder_tokens"], batch["motion_padding_mask"],
                )
                loss, _ = loss_and_metrics(logits, batch["target_tokens"])
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                [p for p in run_model.parameters() if p.requires_grad], CFG.max_grad_norm
            ))
            scaler.step(optimizer); scaler.update(); running_loss += float(loss.item()); batches += 1
        if epoch == 1 or epoch % CFG.metrics_every_epochs == 0 or epoch == epochs:
            measured = evaluate_teacher_forced(run_model, evaluation_loader)
            row = {"epoch": epoch, **measured, "train_batch_loss": running_loss / batches,
                   "grad_norm": gradient_norm, "seconds": time.time() - started}
            run_history.append(row)
            print(f"{label} epoch {epoch:03d}: loss={measured['loss']:.4f} "
                  f"overall={measured['overall_accuracy']:.4f} first={measured['first_step_accuracy']:.4f}")
    return initial, evaluate_teacher_forced(run_model, evaluation_loader), run_history, optimizer, scaler

# %% [markdown]
# ## 4B.6 Four-sample diagnostic, then memorize the locked 32 samples
#
# The first failed run was still improving at epoch 60, while greedy decoding had collapsed onto
# modal trajectories. A fresh four-sample run now proves end-to-end conditional memorization before
# spending a longer run on the exact same 32 samples. This is diagnostic training: dropout is zero.

# %%
micro_df = debug_df.set_index(debug_df[ID_COL].astype(str), drop=False).loc[MICRO_DEBUG_SAMPLE_IDS].reset_index(drop=True)
assert micro_df[ID_COL].astype(str).tolist() == MICRO_DEBUG_SAMPLE_IDS
micro_manifest = micro_df[[ID_COL, GLOSS_COL, "motion_length_verified"]].rename(
    columns={ID_COL: "sample_id", GLOSS_COL: "gloss", "motion_length_verified": "motion_length"}
)
micro_manifest.to_csv(DIAGNOSTICS / "micro_debug_sample_manifest.csv", index=False)
micro_generator = torch.Generator().manual_seed(CFG.seed + 1)
micro_loader = DataLoader(
    GlossMotionDataset(micro_df), batch_size=4, shuffle=True, collate_fn=collate_debug,
    num_workers=CFG.num_workers, generator=micro_generator,
)
micro_eval_loader = DataLoader(
    GlossMotionDataset(micro_df), batch_size=4, shuffle=False, collate_fn=collate_debug,
    num_workers=CFG.num_workers,
)
set_seed(CFG.seed + 1)
micro_model = GlossMotionTransformer(load_frozen_encoder()).to(DEVICE)
micro_initial, micro_final, micro_history, micro_optimizer, micro_scaler = train_memorization_run(
    micro_model, micro_loader, micro_eval_loader, CFG.micro_epochs, "Micro-4",
)
micro_loss_reduction = 1.0 - micro_final["loss"] / micro_initial["loss"]
micro_autoregressive = compact_generation_metrics(micro_model, micro_eval_loader)
micro_gate = bool(
    micro_loss_reduction >= 0.95
    and micro_final["overall_accuracy"] >= 0.98
    and micro_final["motion_only_accuracy"] >= 0.98
    and micro_final["first_step_accuracy"] >= 0.95
    and micro_autoregressive["aligned_token_accuracy"] >= 0.70
    and micro_autoregressive["eos_success_rate"] >= 0.75
    and micro_autoregressive["invalid_id_rate"] == 0.0
)
pd.DataFrame(micro_history).to_csv(DIAGNOSTICS / "micro_overfit_history.csv", index=False)
MICRO_REPORT = {
    "sample_ids": MICRO_DEBUG_SAMPLE_IDS, "epochs": CFG.micro_epochs,
    "initial_teacher_forced": micro_initial, "final_teacher_forced": micro_final,
    "loss_reduction": micro_loss_reduction, "autoregressive": micro_autoregressive,
    "thresholds": {"loss_reduction": 0.95, "teacher_accuracy": 0.98,
                   "first_step_accuracy": 0.95, "autoregressive_aligned_accuracy": 0.70,
                   "eos_success_rate": 0.75},
    "passed": micro_gate,
}
write_json(DIAGNOSTICS / "micro_overfit_report.json", MICRO_REPORT)
print("Four-sample diagnostic:", "PASS" if micro_gate else "FAIL", MICRO_REPORT)
del micro_model, micro_optimizer, micro_scaler
gc.collect(); torch.cuda.empty_cache()
if not micro_gate:
    raise RuntimeError(
        "PHASE 4B MICRO GATE FAILED. The four-sample pipeline cannot autoregressively memorize; "
        "inspect micro_overfit_report.json before retrying the locked 32 samples."
    )

# Fresh initialization: the four-sample weights never leak into the locked 32-sample run.
set_seed(CFG.seed)
model = GlossMotionTransformer(load_frozen_encoder()).to(DEVICE)
initial_teacher, final_teacher, history, optimizer, scaler = train_memorization_run(
    model, debug_loader, debug_eval_loader, CFG.epochs, "Locked-32",
)
loss_reduction = 1.0 - final_teacher["loss"] / initial_teacher["loss"]

history_df = pd.DataFrame([{key: value for key, value in row.items() if key not in {"level_losses", "per_level_accuracy"}}
                           | {f"level_{i}_loss": row["level_losses"][i] for i in range(6)}
                           | {f"level_{i}_accuracy": row["per_level_accuracy"][i] for i in range(6)} for row in history])
history_df.to_csv(DIAGNOSTICS / "debug_overfit_history.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history_df["epoch"], history_df["loss"], marker="o"); axes[0].set(title="32-sample loss", xlabel="epoch", ylabel="loss")
axes[1].plot(history_df["epoch"], history_df["overall_accuracy"], marker="o", label="overall")
axes[1].plot(history_df["epoch"], history_df["motion_only_accuracy"], marker="o", label="motion-only")
axes[1].set(title="Teacher-forced accuracy", xlabel="epoch", ylabel="accuracy"); axes[1].legend()
fig.tight_layout(); fig.savefig(FIGURES / "debug_overfit_loss.png", dpi=160, bbox_inches="tight"); plt.show()
accuracy_fig, axis = plt.subplots(figsize=(8, 4))
for level in range(6): axis.plot(history_df["epoch"], history_df[f"level_{level}_accuracy"], marker="o", label=f"L{level}")
axis.set(title="Per-RVQ-level teacher-forced accuracy", xlabel="epoch", ylabel="accuracy"); axis.legend(ncol=3)
accuracy_fig.tight_layout(); accuracy_fig.savefig(FIGURES / "debug_overfit_accuracy.png", dpi=160, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 4B.7 Synchronized-EOS greedy generation and autoregressive metrics
#
# PAD/BOS logits are suppressed. Generation stops only when all six heads choose EOS together.
# Partial EOS choices are replaced by each affected head's best real codebook ID. Returned arrays
# therefore contain only `0..511` and remain `[6,T]`.

# %%
# `generate_batch` is defined above so the four-sample gate can test true greedy decoding
# before the locked 32-sample run begins.
def edit_distance(reference: list[int], prediction: list[int]) -> int:
    previous = list(range(len(prediction) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, pred in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + int(ref != pred)))
        previous = current
    return previous[-1]


generation_rows = []
all_aligned_correct = all_aligned_total = invalid_generated = generated_total = 0
per_level_aligned_correct = np.zeros(6); per_level_aligned_total = np.zeros(6)
for raw_batch in debug_eval_loader:
    outputs = generate_batch(model, raw_batch)
    for index, output in enumerate(outputs):
        sample_id = raw_batch["sample_id"][index]
        reference_length = int(raw_batch["original_motion_length"][index])
        reference = raw_batch["target_tokens"][index, :, :reference_length].numpy()
        prediction = output["tokens"]; generated_length = prediction.shape[1]; aligned = min(reference_length, generated_length)
        level_accuracies = []
        for level in range(6):
            correct = int((reference[level, :aligned] == prediction[level, :aligned]).sum()) if aligned else 0
            per_level_aligned_correct[level] += correct; per_level_aligned_total[level] += aligned
            level_accuracies.append(correct / aligned if aligned else 0.0)
        correct_all = int((reference[:, :aligned] == prediction[:, :aligned]).sum()) if aligned else 0
        all_aligned_correct += correct_all; all_aligned_total += RVQ_LEVELS * aligned
        invalid = int(((prediction < 0) | (prediction > 511)).sum())
        invalid_generated += invalid; generated_total += prediction.size
        normalized_edits = [edit_distance(reference[level].tolist(), prediction[level].tolist()) /
                            max(reference_length, generated_length, 1) for level in range(6)]
        exact = reference.shape == prediction.shape and np.array_equal(reference, prediction)
        generation_rows.append({
            "sample_id": sample_id, "gloss": raw_batch["gloss"][index],
            "reference_length": reference_length, "generated_length": generated_length,
            "length_absolute_error": abs(generated_length - reference_length),
            "eos_reached": output["eos_reached"], "aligned_token_accuracy": correct_all / max(6 * aligned, 1),
            **{f"level_{level}_aligned_accuracy": level_accuracies[level] for level in range(6)},
            "mean_normalized_edit_distance": float(np.mean(normalized_edits)),
            "exact_full_sequence_match": exact, "invalid_id_count": invalid,
            "reference_preview": json.dumps(reference[:, :16].tolist()),
            "generated_preview": json.dumps(prediction[:, :16].tolist()),
        })

generation_df = pd.DataFrame(generation_rows)
generation_df.to_csv(DIAGNOSTICS / "debug_overfit_predictions.csv", index=False)
prefix_counts = generation_df["generated_preview"].value_counts()
AUTOREGRESSIVE_METRICS = {
    "aligned_token_accuracy": float(generation_df["aligned_token_accuracy"].mean()),
    "micro_aligned_token_accuracy": all_aligned_correct / max(all_aligned_total, 1),
    "per_level_prefix_accuracy": (per_level_aligned_correct / np.maximum(per_level_aligned_total, 1)).tolist(),
    "length_mae": float(generation_df["length_absolute_error"].mean()),
    "normalized_token_edit_distance": float(generation_df["mean_normalized_edit_distance"].mean()),
    "exact_sequence_match": float(generation_df["exact_full_sequence_match"].mean()),
    "eos_success_rate": float(generation_df["eos_reached"].mean()),
    "unique_generated_16_token_prefixes": int(generation_df["generated_preview"].nunique()),
    "modal_generated_prefix_rate": float(prefix_counts.iloc[0] / len(generation_df)),
}
INVALID_ID_RATE = invalid_generated / max(generated_total, 1)
display(generation_df[["sample_id", "reference_length", "generated_length", "eos_reached", "aligned_token_accuracy",
                       "mean_normalized_edit_distance", "exact_full_sequence_match", "invalid_id_count"]])
print("Autoregressive metrics:", AUTOREGRESSIVE_METRICS, "invalid rate:", INVALID_ID_RATE)

# %% [markdown]
# ## 4B.8 Gate, compact checkpoint, and deterministic reload test
#
# The frozen encoder is not duplicated in `model.pt`; the checkpoint stores all trainable motion
# components plus SHA-256 hashes identifying the Phase-3 model/config. Reload reconstructs the
# same frozen encoder locally, loads the motion state, and must reproduce one greedy output byte
# for byte. A checkpoint is created only after the metric portion of the gate passes.

# %%
metric_gate = bool(
    loss_reduction >= CFG.gate_min_loss_reduction
    and final_teacher["overall_accuracy"] >= CFG.gate_min_teacher_accuracy
    and final_teacher["motion_only_accuracy"] >= CFG.gate_min_motion_only_accuracy
    and AUTOREGRESSIVE_METRICS["aligned_token_accuracy"] >= CFG.gate_min_autoregressive_aligned_accuracy
    and AUTOREGRESSIVE_METRICS["eos_success_rate"] >= CFG.gate_min_eos_success_rate
    and INVALID_ID_RATE == 0.0
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


MODEL_CONFIG = {
    "rvq_levels": 6, "codebook_sizes": CODEBOOK_SIZES, "PAD_ID": PAD_ID, "BOS_ID": BOS_ID,
    "EOS_ID": EOS_ID, "motion_vocab_size": MOTION_VOCAB_SIZE, "level_order": LEVEL_ORDER,
    "d_model": CFG.d_model, "d_level": CFG.d_level, "num_heads": CFG.num_heads,
    "decoder_layers": CFG.decoder_layers, "ffn_dimension": CFG.ffn_dimension,
    "dropout": CFG.dropout, "positional_capacity": CFG.positional_capacity,
    "gloss_encoder": str(PHASE3_MODEL_PATH), "gloss_tokenizer": str(TOKENIZER_PATH),
    "gloss_encoder_frozen": True, "phase1_fingerprint": EXPECTED_FINGERPRINT,
    "checkpoint_contents": "all state_dict entries except frozen gloss_encoder.*; reload Phase-3 encoder first",
    "phase3_model_config_sha256": sha256_file(PHASE3_MODEL_PATH / "config.json"),
    "phase3_tokenizer_config_sha256": sha256_file(TOKENIZER_PATH / "tokenizer_config.json"),
}
checkpoint_reload_verified = False
reload_details: dict[str, Any] = {"attempted": False}
if metric_gate:
    DEBUG_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    reload_probe_batch = next(iter(DataLoader(
        GlossMotionDataset(debug_df.iloc[:1]), batch_size=1, shuffle=False, collate_fn=collate_debug
    )))
    before = generate_batch(model, reload_probe_batch)[0]
    motion_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
                    if not name.startswith("gloss_encoder.")}
    checkpoint = {"motion_state_dict": motion_state}
    torch.save(checkpoint, DEBUG_MODEL_DIR / "model.pt")
    write_json(DEBUG_MODEL_DIR / "model_config.json", MODEL_CONFIG)
    write_json(DEBUG_MODEL_DIR / "debug_model_config.json", MODEL_CONFIG)
    del model, optimizer, scaler, motion_state, checkpoint; gc.collect(); torch.cuda.empty_cache()
    reloaded_model = GlossMotionTransformer(load_frozen_encoder()).to(DEVICE)
    saved = torch.load(DEBUG_MODEL_DIR / "model.pt", map_location="cpu", weights_only=True)
    incompatible = reloaded_model.load_state_dict(saved["motion_state_dict"], strict=False)
    expected_missing = [name for name in reloaded_model.state_dict() if name.startswith("gloss_encoder.")]
    assert sorted(incompatible.missing_keys) == sorted(expected_missing) and not incompatible.unexpected_keys
    after = generate_batch(reloaded_model, reload_probe_batch)[0]
    checkpoint_reload_verified = bool(
        before["eos_reached"] == after["eos_reached"] and np.array_equal(before["tokens"], after["tokens"])
    )
    reload_details = {"attempted": True, "identical": checkpoint_reload_verified,
                      "probe_sample_id": reload_probe_batch["sample_id"][0],
                      "generated_shape": list(after["tokens"].shape)}
    model = reloaded_model

OVERFIT_GATE_PASSED = bool(metric_gate and checkpoint_reload_verified)
DEBUG_OVERFIT_REPORT = {
    "debug_samples": 32, "initial_loss": initial_teacher["loss"], "final_loss": final_teacher["loss"],
    "loss_reduction": loss_reduction, "teacher_forced": final_teacher,
    "retry_configuration": {"locked_manifest": True, "epochs": CFG.epochs, "dropout": CFG.dropout,
                            "micro_gate_passed": micro_gate},
    "autoregressive": AUTOREGRESSIVE_METRICS, "generated_token_validity": {"invalid_id_rate": INVALID_ID_RATE},
    "thresholds": {
        "loss_reduction": CFG.gate_min_loss_reduction, "teacher_accuracy": CFG.gate_min_teacher_accuracy,
        "motion_only_accuracy": CFG.gate_min_motion_only_accuracy,
        "autoregressive_aligned_accuracy": CFG.gate_min_autoregressive_aligned_accuracy,
        "eos_success_rate": CFG.gate_min_eos_success_rate,
    },
    "metric_gate_passed": metric_gate, "checkpoint_reload": reload_details,
    "passed": OVERFIT_GATE_PASSED,
}
write_json(DIAGNOSTICS / "debug_overfit_report.json", DEBUG_OVERFIT_REPORT)

PHASE4B_REPORT = {
    "phase": "Phase 4B - Motion Model Debug / Overfit", "phase4b_complete": OVERFIT_GATE_PASSED,
    "phase1_fingerprint": EXPECTED_FINGERPRINT,
    "representation": {"shape": "[L,T]", "rvq_levels": 6, "codebook_sizes": CODEBOOK_SIZES, "level_order": LEVEL_ORDER},
    "special_tokens": {"pad": PAD_ID, "bos": BOS_ID, "eos": EOS_ID},
    "model": {"gloss_encoder": str(PHASE3_MODEL_PATH), "encoder_frozen": True,
              "d_model": CFG.d_model, "d_level": CFG.d_level, "heads": CFG.num_heads,
              "decoder_layers": CFG.decoder_layers, "parameters_total": total_parameters,
              "parameters_trainable": trainable_parameters},
    "debug_samples": 32, "initial_loss": initial_teacher["loss"], "final_loss": final_teacher["loss"],
    "loss_reduction": loss_reduction,
    "teacher_forced": {"overall_accuracy": final_teacher["overall_accuracy"],
                       "motion_only_accuracy": final_teacher["motion_only_accuracy"],
                       "per_level_accuracy": final_teacher["per_level_accuracy"],
                       "first_step_accuracy": final_teacher["first_step_accuracy"]},
    "autoregressive": AUTOREGRESSIVE_METRICS,
    "generated_token_validity": {"invalid_id_rate": INVALID_ID_RATE},
    "checkpoint_reload_verified": checkpoint_reload_verified,
    "ready_for_full_training": OVERFIT_GATE_PASSED,
}
write_json(OUTPUT / "phase4b_report.json", PHASE4B_REPORT)

print("\n============================================================")
print("PHASE 4B — MOTION MODEL OVERFIT GATE")
print("============================================================")
print("Representation: [6,T]")
print("Codebooks: 6 × 512")
print("Special IDs: PAD = 512, BOS = 513, EOS = 514")
print("Debug samples: 32")
print("Initial loss:", initial_teacher["loss"])
print("Final loss:", final_teacher["loss"])
print("Loss reduction:", loss_reduction)
print("Teacher-forced overall accuracy:", final_teacher["overall_accuracy"])
print("Motion-only accuracy:", final_teacher["motion_only_accuracy"])
print("First-step accuracy:", final_teacher["first_step_accuracy"])
for level, accuracy in enumerate(final_teacher["per_level_accuracy"]): print(f"Per-level accuracy L{level}:", accuracy)
print("Autoregressive aligned accuracy:", AUTOREGRESSIVE_METRICS["aligned_token_accuracy"])
print("EOS success:", AUTOREGRESSIVE_METRICS["eos_success_rate"])
print("Length MAE:", AUTOREGRESSIVE_METRICS["length_mae"])
print("Invalid generated IDs:", INVALID_ID_RATE)
print("Checkpoint reload:", "PASS" if checkpoint_reload_verified else "FAIL / NOT ATTEMPTED")
print("OVERFIT GATE:", "PASS" if OVERFIT_GATE_PASSED else "FAIL")
print("READY FOR PHASE 4C FULL TRAINING:", "YES" if OVERFIT_GATE_PASSED else "NO")
print("============================================================")

if not OVERFIT_GATE_PASSED:
    raise RuntimeError("PHASE 4B OVERFIT GATE FAILED. Inspect saved diagnostics; do not start Phase 4C.")
print("Phase 4B passed. STOP here and inspect reports before authorizing Phase 4C.")

# %% [markdown]
# # Motion-S Final End-to-End Demo
# ## English Text → Gloss → Motion Length → RVQ Tokens → 668-D Motion
#
# **Final demo / inference only.** This notebook performs no training, fine-tuning,
# checkpoint modification, or hyperparameter selection. Feature-space plots are
# diagnostics and do not claim linguistic sign correctness.

# %%
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT = Path("/kaggle/working/motion_s_final_demo")
OUTPUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU inference")
print("Training performed: NO")

# %% [markdown]
# ## 2 — Automatic artifact discovery
#
# The filesystem is scanned once. Recorded hashes are used whenever available;
# duplicate non-identical final outputs are rejected.

# %%
SEARCH_BASES = [Path("/kaggle/input"), Path("/kaggle/working")]
ALL_FILES: list[Path] = []
ALL_DIRS: list[Path] = []
for base in SEARCH_BASES:
    if not base.exists():
        continue
    for current, directories, files in os.walk(base):
        directory = Path(current)
        if directory.resolve() == OUTPUT.resolve():
            directories[:] = []
            continue
        ALL_DIRS.append(directory)
        ALL_FILES.extend(directory / name for name in files)

def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def canonical(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())

def named(name: str) -> list[Path]:
    return sorted({path.resolve() for path in ALL_FILES if path.name == name}, key=str)

phase8_roots = []
for report_path in named("phase8_report.json"):
    root = report_path.parent
    if all((root / relative).is_file() for relative in (
        "final_pipeline/generate_motion.py", "final_pipeline/model_manifest.json",
        "final_pipeline/system_contract.json", "final_model/best_motion_generator.pt")):
        phase8_roots.append(root.resolve())
if not phase8_roots:
    raise RuntimeError("No complete motion_s_phase8 output was found under /kaggle/input or /kaggle/working")
report_hashes = {root: sha256(root / "phase8_report.json") for root in phase8_roots}
if len(set(report_hashes.values())) > 1:
    raise RuntimeError(f"Multiple non-identical Phase-8 outputs found: {report_hashes}")
preferred = [root for root in phase8_roots if "motionsphase8" in canonical(root)]
PHASE8_ROOT = sorted(preferred or phase8_roots, key=lambda path: (0 if str(path).startswith("/kaggle/working") else 1, len(path.parts), str(path)))[0]

GENERATE_MODULE = PHASE8_ROOT / "final_pipeline/generate_motion.py"
MANIFEST_PATH = PHASE8_ROOT / "final_pipeline/model_manifest.json"
CONTRACT_PATH = PHASE8_ROOT / "final_pipeline/system_contract.json"
REPORT_PATH = PHASE8_ROOT / "phase8_report.json"
GENERATOR_CKPT = PHASE8_ROOT / "final_model/best_motion_generator.pt"
MANIFEST = read_json(MANIFEST_PATH)
CONTRACT = read_json(CONTRACT_PATH)
PHASE8_REPORT = read_json(REPORT_PATH)

def resolve_hashed_file(label: str, filename: str, expected_hash: str | None, preferred_path: object = None) -> Path:
    candidates = []
    if preferred_path:
        candidate = Path(str(preferred_path))
        if candidate.is_file():
            candidates.append(candidate.resolve())
    candidates.extend(named(filename))
    candidates = sorted(set(candidates), key=str)
    if expected_hash:
        candidates = [path for path in candidates if sha256(path) == expected_hash]
    if not candidates:
        raise RuntimeError(f"Could not resolve {label} ({filename}, expected SHA256={expected_hash})")
    hashes = {path: sha256(path) for path in candidates}
    if len(set(hashes.values())) != 1:
        raise RuntimeError(f"Ambiguous non-identical {label}: {hashes}")
    return sorted(candidates, key=lambda path: (0 if PHASE8_ROOT in path.parents else 1, len(path.parts), str(path)))[0]

length_entry = MANIFEST["phase6_length_checkpoint"]
rvq_entry = MANIFEST["rvq_checkpoint"]
generator_entry = MANIFEST["final_generator"]
normal_entry = MANIFEST["normalization"]
LENGTH_CKPT = resolve_hashed_file("Phase-6 length checkpoint", "best_model.pt", length_entry.get("sha256"), length_entry.get("path"))
RVQ_CKPT = resolve_hashed_file("official RVQ checkpoint", "rvq_vae_best.pth", rvq_entry.get("sha256"), rvq_entry.get("path"))
GENERATOR_CKPT = resolve_hashed_file("Phase-8 final generator", "best_motion_generator.pt", generator_entry.get("sha256"), GENERATOR_CKPT)
MOTION_MEAN_PATH = resolve_hashed_file("motion mean", "motion_mean_668.npy", normal_entry.get("mean_sha256"), normal_entry.get("mean"))
MOTION_STD_PATH = resolve_hashed_file("motion std", "motion_std_668.npy", normal_entry.get("std_sha256"), normal_entry.get("std"))

# Resolve the Phase-3 model by its recorded config hash, then use its sibling tokenizer.
phase3_entry = MANIFEST["phase3_model"]
phase3_configs = [path for path in named("config.json") if sha256(path) == phase3_entry.get("config_sha256")]
phase3_model_dirs = [path.parent for path in phase3_configs if (path.parent / "model.safetensors").is_file() or (path.parent / "pytorch_model.bin").is_file()]
if not phase3_model_dirs:
    raise RuntimeError("No Phase-3 model matched the manifest config hash")
phase3_weight_hashes = {}
for directory in phase3_model_dirs:
    weight = directory / "model.safetensors" if (directory / "model.safetensors").is_file() else directory / "pytorch_model.bin"
    phase3_weight_hashes[directory] = sha256(weight)
if len(set(phase3_weight_hashes.values())) != 1:
    raise RuntimeError(f"Ambiguous Phase-3 models share the config but have different weights: {phase3_weight_hashes}")
PHASE3_MODEL = sorted(set(phase3_model_dirs), key=lambda path: (0 if "phase3" in canonical(path) else 1, len(path.parts), str(path)))[0]
phase3_root = PHASE3_MODEL.parent
TOKENIZER_PATH = phase3_root / "tokenizer"
if not (TOKENIZER_PATH / "tokenizer_config.json").is_file():
    tokenizer_candidates = [path.parent for path in named("tokenizer_config.json") if "tokenizer" in canonical(path.parent)]
    if len(tokenizer_candidates) != 1:
        raise RuntimeError(f"Phase-3 tokenizer resolution failed: {tokenizer_candidates}")
    TOKENIZER_PATH = tokenizer_candidates[0]
PHASE3_REPORT_PATH = phase3_root / "phase3_report.json"

LENGTH_CONFIG = LENGTH_CKPT.parent / "config.json"
if not LENGTH_CONFIG.is_file():
    raise RuntimeError(f"Missing Phase-6 length config beside {LENGTH_CKPT}")

# Exact unchanged Phase-4/7 architecture metadata is required for safe restoration.
motion_config_candidates = []
for path in named("model_config.json"):
    try:
        payload = read_json(path)
        if {"d_model", "d_level", "num_heads", "ffn_dimension", "decoder_layers", "max_model_positions"}.issubset(payload):
            motion_config_candidates.append(path)
    except Exception:
        pass
if not motion_config_candidates:
    raise RuntimeError("Missing the unchanged Phase-4 best_model/model_config.json required to restore the generator")
motion_config_hashes = {path: sha256(path) for path in motion_config_candidates}
if len(set(motion_config_hashes.values())) != 1:
    raise RuntimeError(f"Ambiguous generator architecture configs: {motion_config_hashes}")
MOTION_CONFIG_PATH = motion_config_candidates[0]

# Phase-1 metadata and held-out split are optional for the custom demo, required only for GT comparison.
DATASET_CONFIG_PATH = None
for path in named("dataset_config.json"):
    try:
        if read_json(path).get("split_fingerprint_sha256") == MANIFEST.get("phase1_fingerprint"):
            DATASET_CONFIG_PATH = path; break
    except Exception:
        pass
TEST_SPLIT_PATH = None
if DATASET_CONFIG_PATH:
    for candidate_name in ("test_split.csv", "test.csv"):
        candidate = DATASET_CONFIG_PATH.parent / "splits" / candidate_name
        if candidate.is_file(): TEST_SPLIT_PATH = candidate; break

OFFICIAL_TRAIN_CSV = None
if DATASET_CONFIG_PATH:
    discovery_config = read_json(DATASET_CONFIG_PATH)
    required_columns = {discovery_config["id_column"], discovery_config["gloss_column"], *discovery_config.get("motion_token_columns", [])}
    train_candidates = []
    for path in named("train.csv"):
        try:
            if required_columns.issubset(pd.read_csv(path, nrows=0).columns): train_candidates.append(path)
        except Exception:
            pass
    if train_candidates:
        train_hashes = {path: sha256(path) for path in train_candidates}
        if len(set(train_hashes.values())) != 1: raise RuntimeError(f"Ambiguous official Motion-S train.csv files: {train_hashes}")
        OFFICIAL_TRAIN_CSV = train_candidates[0]

FEATURE_DIRS = [directory for directory in ALL_DIRS if canonical(directory.name) == "motionfeatures" and any(directory.glob("*.npy"))]
FEATURE_DIR = sorted(set(FEATURE_DIRS), key=lambda path: (len(path.parts), str(path)))[0] if FEATURE_DIRS else None

artifact_rows = [
    ("Phase-8 root", PHASE8_ROOT), ("generate_motion.py", GENERATE_MODULE),
    ("model_manifest.json", MANIFEST_PATH), ("system_contract.json", CONTRACT_PATH),
    ("phase8_report.json", REPORT_PATH), ("best_motion_generator.pt", GENERATOR_CKPT),
    ("Phase-3 model", PHASE3_MODEL), ("Phase-3 tokenizer", TOKENIZER_PATH),
    ("Phase-6 length predictor", LENGTH_CKPT), ("RVQ checkpoint", RVQ_CKPT),
    ("motion_mean_668.npy", MOTION_MEAN_PATH), ("motion_std_668.npy", MOTION_STD_PATH),
    ("Official Motion-S train.csv", OFFICIAL_TRAIN_CSV), ("Phase-1 test split", TEST_SPLIT_PATH), ("Motion-Features", FEATURE_DIR),
]
artifact_table = pd.DataFrame([{"Artifact": label, "Resolved path": str(path) if path else "UNAVAILABLE", "Exists": bool(path and path.exists())} for label, path in artifact_rows])
display(artifact_table)

# %% [markdown]
# ## 3 — Final system metadata

# %%
final_info = {
    "Final generator variant": PHASE8_REPORT.get("final_generator", {}).get("source_variant", generator_entry.get("source_variant")),
    "lambda_velocity": PHASE8_REPORT.get("final_generator", {}).get("lambda_velocity"),
    "lambda_persistence": PHASE8_REPORT.get("final_generator", {}).get("lambda_persistence"),
    "Motion generator checkpoint": str(GENERATOR_CKPT),
    "Phase-3 model": str(PHASE3_MODEL),
    "Phase-6 length checkpoint": str(LENGTH_CKPT),
    "RVQ checkpoint": str(RVQ_CKPT),
    "Generator contract": MANIFEST.get("contracts", {}).get("generator"),
    "Decoder contract": MANIFEST.get("contracts", {}).get("decoder"),
    "RVQ real ID range": MANIFEST.get("token_ids", {}).get("real", [0, 511]),
    "Motion feature dimension": 668,
    "Temporal expansion factor": 4,
}
display(pd.Series(final_info, name="Value").to_frame())
if str(final_info["Final generator variant"]).casefold() != "dynamics":
    raise RuntimeError(f"Expected frozen Phase-8 DYNAMICS generator, found {final_info['Final generator variant']}")
if abs(float(final_info["lambda_velocity"]) - .1) > 1e-12 or abs(float(final_info["lambda_persistence"])) > 1e-12:
    raise RuntimeError("Phase-8 loss weights do not match the final locked DYNAMICS system")

# %% [markdown]
# ## 4 — Load the frozen final inference pipeline
#
# The saved API is imported first. Its four exact frozen inference functions are
# installed from relocated, hash-verified artifacts. No optimizer or training code is used.

# %%
try:
    module_spec = importlib.util.spec_from_file_location("motion_s_final_api", GENERATE_MODULE)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Could not create import spec for {GENERATE_MODULE}")
    final_api = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(final_api)
except Exception as error:
    print("generate_motion.py import failed:", repr(error))
    print("Manifest:", MANIFEST_PATH)
    print("System contract:", CONTRACT_PATH)
    raise RuntimeError("Final API cannot be safely imported; no reconstruction or training was attempted") from error

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PAD_ID, BOS_ID, EOS_ID = 512, 513, 514
MODEL_CONFIG = read_json(MOTION_CONFIG_PATH)
LENGTH_CONFIG_PAYLOAD = read_json(LENGTH_CONFIG)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True, use_fast=True)
phase3_model = AutoModelForSeq2SeqLM.from_pretrained(PHASE3_MODEL, local_files_only=True).to(DEVICE).eval()
for parameter in phase3_model.parameters(): parameter.requires_grad = False
gloss_encoder = phase3_model.get_encoder()

class MotionModel(nn.Module):
    def __init__(self, encoder):
        super().__init__(); self.gloss_encoder = encoder
        d_model, d_level = int(MODEL_CONFIG["d_model"]), int(MODEL_CONFIG["d_level"])
        self.encoder_projection = nn.Linear(int(encoder.config.d_model), d_model)
        self.level_embeddings = nn.ModuleList([nn.Embedding(515, d_level, padding_idx=PAD_ID) for _ in range(6)])
        self.motion_projection = nn.Linear(6 * d_level, d_model)
        self.temporal_positions = nn.Embedding(int(MODEL_CONFIG["max_model_positions"]), d_model)
        layer = nn.TransformerDecoderLayer(d_model, int(MODEL_CONFIG["num_heads"]), int(MODEL_CONFIG["ffn_dimension"]), float(MODEL_CONFIG["dropout"]), activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, int(MODEL_CONFIG["decoder_layers"]), norm=nn.LayerNorm(d_model))
        self.heads = nn.ModuleList([nn.Linear(d_model, 515) for _ in range(6)])
    def encode(self, ids, mask):
        return self.encoder_projection(self.gloss_encoder(input_ids=ids, attention_mask=mask).last_hidden_state)
    def decode(self, tokens, memory, mask, padding):
        embedded = torch.cat([self.level_embeddings[q](tokens[:, q]) for q in range(6)], -1)
        embedded = self.motion_projection(embedded) + self.temporal_positions(torch.arange(tokens.shape[-1], device=tokens.device)[None])
        causal = torch.triu(torch.ones(tokens.shape[-1], tokens.shape[-1], dtype=torch.bool, device=tokens.device), 1)
        hidden = self.decoder(embedded, memory, tgt_mask=causal, tgt_key_padding_mask=padding, memory_key_padding_mask=~mask.bool())
        return torch.stack([head(hidden) for head in self.heads], 1)

motion_generator = MotionModel(gloss_encoder).to(DEVICE)
generator_payload = torch.load(GENERATOR_CKPT, map_location="cpu", weights_only=True)
generator_state = generator_payload.get("motion_state_dict", generator_payload)
load_result = motion_generator.load_state_dict(generator_state, strict=False)
expected_missing = sorted(name for name in motion_generator.state_dict() if name.startswith("gloss_encoder."))
if sorted(load_result.missing_keys) != expected_missing or load_result.unexpected_keys:
    raise RuntimeError(f"Final generator checkpoint contract mismatch: {load_result}")
motion_generator.eval()
for parameter in motion_generator.parameters(): parameter.requires_grad = False

class LengthHead(nn.Module):
    def __init__(self):
        super().__init__(); dimension = int(gloss_encoder.config.d_model)
        self.network = nn.Sequential(nn.Linear(dimension, 256), nn.GELU(), nn.Dropout(.1), nn.Linear(256, 1))
    def forward(self, values): return self.network(values).squeeze(-1)

length_head = LengthHead().to(DEVICE)
length_payload = torch.load(LENGTH_CKPT, map_location="cpu", weights_only=True)
length_head.load_state_dict(length_payload["length_head_state_dict"], strict=True)
length_head.eval()
for parameter in length_head.parameters(): parameter.requires_grad = False
length_bounds = list(map(int, LENGTH_CONFIG_PAYLOAD["training_bounds"]))
if length_bounds != [24, 469]: raise RuntimeError(f"Unexpected length bounds: {length_bounds}")
calibration = LENGTH_CONFIG_PAYLOAD["calibration"]

class MotionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 512, 3, 2, 1, output_padding=1), nn.ReLU(), nn.BatchNorm1d(512),
            nn.ConvTranspose1d(512, 512, 3, 2, 1, output_padding=1), nn.ReLU(), nn.BatchNorm1d(512),
            nn.ConvTranspose1d(512, 512, 3, 1, 1), nn.ReLU(), nn.BatchNorm1d(512),
            nn.ConvTranspose1d(512, 668, 3, 1, 1))
    def forward(self, values): return self.decoder(values)

rvq_payload = torch.load(RVQ_CKPT, map_location="cpu", weights_only=True)
for state_key in ("model_state_dict", "state_dict"):
    if isinstance(rvq_payload, dict) and state_key in rvq_payload:
        rvq_payload = rvq_payload[state_key]; break
rvq_state = {key: value for key, value in rvq_payload.items() if isinstance(value, torch.Tensor)}
motion_decoder = MotionDecoder()
motion_decoder.load_state_dict({key[8:]: value for key, value in rvq_state.items() if key.startswith("decoder.")}, strict=True)
motion_decoder = motion_decoder.to(DEVICE).eval()
for parameter in motion_decoder.parameters(): parameter.requires_grad = False
codebooks = torch.stack([rvq_state[f"rvq.quantizers.{level}.embedding"].float() for level in range(6)]).to(DEVICE)
motion_mean = np.load(MOTION_MEAN_PATH, allow_pickle=False).astype(np.float32)
motion_std = np.load(MOTION_STD_PATH, allow_pickle=False).astype(np.float32)
safe_motion_std = motion_std.copy(); safe_motion_std[safe_motion_std < 1e-6] = 1.0
mean_tensor = torch.from_numpy(motion_mean).to(DEVICE)
std_tensor = torch.from_numpy(safe_motion_std).to(DEVICE)

phase3_decoding = read_json(PHASE3_REPORT_PATH).get("selected_decoding") if PHASE3_REPORT_PATH.is_file() else None
phase3_beams = 4 if phase3_decoding == "beam_4" else 1

@torch.inference_mode()
def text_to_gloss(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if not cleaned: raise ValueError("English text must not be empty")
    encoded = tokenizer("text to gloss: " + cleaned, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
    output = phase3_model.generate(**encoded, max_length=128, num_beams=phase3_beams, do_sample=False, early_stopping=phase3_beams > 1)
    gloss = re.sub(r"\s+", " ", tokenizer.decode(output[0], skip_special_tokens=True)).strip()
    if not gloss: raise RuntimeError("Phase-3 returned an empty gloss")
    return gloss

@torch.inference_mode()
def predict_motion_length(glosses: list[str]) -> np.ndarray:
    encoded = tokenizer(list(map(str, glosses)), padding=True, truncation=False, return_tensors="pt")
    ids, mask = encoded["input_ids"].to(DEVICE), encoded["attention_mask"].to(DEVICE)
    hidden = gloss_encoder(input_ids=ids, attention_mask=mask).last_hidden_state.float()
    pooled = (hidden * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
    values = torch.expm1(length_head(pooled)).cpu().numpy()
    if calibration.get("selected"):
        values = float(calibration["slope"]) * values + float(calibration["intercept"])
    return np.clip(np.rint(values), length_bounds[0], length_bounds[1]).astype(int)

@torch.inference_mode()
def generate_rvq_tokens(gloss: str, length: int) -> np.ndarray:
    length = int(length)
    if not length_bounds[0] <= length <= length_bounds[1]: raise ValueError(f"Length {length} outside {length_bounds}")
    encoded = tokenizer([str(gloss)], padding=True, truncation=False, return_tensors="pt")
    ids, mask = encoded["input_ids"].to(DEVICE), encoded["attention_mask"].to(DEVICE)
    memory = motion_generator.encode(ids, mask)
    history = torch.full((1, 6, 1), BOS_ID, dtype=torch.long, device=DEVICE)
    padding = torch.zeros((1, 1), dtype=torch.bool, device=DEVICE)
    steps = []
    for _ in range(length):
        choice = motion_generator.decode(history, memory, mask, padding)[:, :, -1, :512].argmax(-1)
        steps.append(choice.cpu()); history = torch.cat([history, choice.unsqueeze(-1)], -1)
        padding = torch.cat([padding, torch.zeros((1, 1), dtype=torch.bool, device=DEVICE)], 1)
    return torch.stack(steps, -1).squeeze(0).numpy().astype(np.int64)

@torch.inference_mode()
def decode_motion_s_tokens(tokens: np.ndarray) -> np.ndarray:
    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.ndim != 2 or tokens.shape[0] != 6 or tokens.min() < 0 or tokens.max() > 511:
        raise ValueError(f"Invalid RVQ token tensor: shape={tokens.shape}")
    ids = torch.as_tensor(tokens, dtype=torch.long, device=DEVICE)
    latent = torch.zeros((256, tokens.shape[1]), device=DEVICE)
    for level in range(6): latent += codebooks[level][:, ids[level]]
    motion = (motion_decoder(latent[None]).squeeze(0).T * std_tensor + mean_tensor).cpu().numpy().astype(np.float32)
    if motion.shape != (4 * tokens.shape[1], 668) or not np.isfinite(motion).all():
        raise RuntimeError(f"Decoder contract failed: shape={motion.shape}")
    return motion

final_api.install_pipeline(text_to_gloss, predict_motion_length, generate_rvq_tokens, decode_motion_s_tokens)
generate_motion = final_api.generate_motion
print("Final generate_motion() pipeline loaded: PASS")

def extract_result(payload: object) -> dict:
    if not isinstance(payload, dict): raise TypeError(f"Expected dictionary intermediates, received {type(payload).__name__}")
    def get(*aliases):
        for alias in aliases:
            if alias in payload: return payload[alias]
        raise KeyError(f"Missing aliases {aliases}; available keys={list(payload)}")
    return {"text": get("text", "input", "english_text"), "gloss": get("predicted_gloss", "gloss"),
            "length": int(get("predicted_token_length", "token_length", "length")),
            "tokens": np.asarray(get("rvq_tokens", "tokens")), "motion": np.asarray(get("motion", "continuous_motion"))}

# %% [markdown]
# ## 5 — Custom English demo

# %%
custom_text = "I want to drink water"
custom = extract_result(generate_motion(custom_text, return_intermediates=True))
custom_tokens, custom_motion, custom_length = custom["tokens"], custom["motion"], custom["length"]
custom_valid_tokens = bool(custom_tokens.ndim == 2 and custom_tokens.shape[0] == 6 and custom_tokens.min() >= 0 and custom_tokens.max() <= 511)
custom_finite = bool(np.isfinite(custom_motion).all())
print("=" * 60); print("CUSTOM TEXT DEMO"); print("=" * 60)
print("Input English:", custom_text); print("Predicted Gloss:", custom["gloss"])
print("Predicted Motion-Token Length:", custom_length); print("RVQ Token Shape:", custom_tokens.shape)
print("RVQ Token Min/Max:", int(custom_tokens.min()), int(custom_tokens.max()))
print("Continuous Motion Shape:", custom_motion.shape); print("Expected motion frames:", 4 * custom_length)
print("Actual motion frames:", len(custom_motion)); print("All RVQ IDs Valid [0..511]:", custom_valid_tokens)
print("Motion Finite:", custom_finite); print("NaN count:", int(np.isnan(custom_motion).sum())); print("Inf count:", int(np.isinf(custom_motion).sum())); print("=" * 60)
assert custom_tokens.ndim == 2 and custom_tokens.shape[0] == 6
assert custom_tokens.min() >= 0 and custom_tokens.max() <= 511
assert custom_motion.ndim == 2 and custom_motion.shape[1] == 668
assert custom_motion.shape[0] == 4 * custom_length
assert custom_motion.dtype == np.float32 and custom_finite

# %% [markdown]
# ## 6 — First RVQ token steps

# %%
level_names = ["base", "residual_1", "residual_2", "residual_3", "residual_4", "residual_5"]
steps_to_show = min(15, custom_tokens.shape[1])
token_preview = pd.DataFrame(custom_tokens[:, :steps_to_show].T, index=[f"t{i}" for i in range(steps_to_show)], columns=level_names)
display(token_preview)
print("Number of token steps:", custom_tokens.shape[1])
display(pd.DataFrame({"RVQ level": level_names, "Unique IDs used": [np.unique(custom_tokens[level]).size for level in range(6)]}))

# %% [markdown]
# ## 7 — Generated feature trajectories

# %%
custom_variance = np.var(custom_motion, axis=0, dtype=np.float64)
custom_top6 = np.argsort(custom_variance)[-6:][::-1]
fig, axis = plt.subplots(figsize=(14, 6))
for feature in custom_top6: axis.plot(custom_motion[:, feature], label=f"Feature {feature}")
axis.set(title=f'Generated Motion Feature Trajectories\nInput: "{custom_text}"', xlabel="Frame", ylabel="Feature value")
axis.grid(alpha=.25); axis.legend(ncol=2); fig.tight_layout(); fig.savefig(OUTPUT / "custom_feature_trajectories.png", dpi=180); plt.show()

# %% [markdown]
# ## 8 — Generated motion activity

# %%
custom_active = motion_std >= 1e-6
custom_normalized = (custom_motion[:, custom_active] - motion_mean[custom_active]) / motion_std[custom_active]
custom_velocity = np.diff(custom_normalized.astype(np.float64), axis=0)
custom_velocity_magnitude = np.sqrt(np.mean(np.square(custom_velocity), axis=1))
fig, axis = plt.subplots(figsize=(14, 4.5)); axis.plot(np.arange(1, len(custom_motion)), custom_velocity_magnitude)
axis.set(title="Generated Motion Activity Over Time", xlabel="Frame", ylabel="RMS feature velocity (diagnostic)")
axis.grid(alpha=.25); fig.tight_layout(); fig.savefig(OUTPUT / "custom_motion_activity.png", dpi=180); plt.show()
print("Mean velocity magnitude:", float(custom_velocity_magnitude.mean()))
print("Median velocity magnitude:", float(np.median(custom_velocity_magnitude)))
print("Maximum velocity magnitude:", float(custom_velocity_magnitude.max()))
print("Mean feature temporal variance:", float(custom_variance.mean()))

# %% [markdown]
# ## 9 — Compact high-variance feature heatmap

# %%
custom_top20 = np.argsort(custom_variance)[-20:][::-1]
fig, axis = plt.subplots(figsize=(14, 6)); image = axis.imshow(custom_motion[:, custom_top20].T, aspect="auto", interpolation="nearest", cmap="coolwarm")
axis.set(title="Generated Motion Feature Activity", xlabel="Frame", ylabel="Selected feature index", yticks=np.arange(20), yticklabels=custom_top20)
fig.colorbar(image, ax=axis, label="Feature value"); fig.tight_layout(); fig.savefig(OUTPUT / "custom_motion_heatmap.png", dpi=180); plt.show()

# %% [markdown]
# ## 10 — Select one real locked held-out test sample

# %%
heldout_available = False
heldout_error = ""
try:
    if DATASET_CONFIG_PATH is None or TEST_SPLIT_PATH is None or FEATURE_DIR is None:
        raise FileNotFoundError("Phase-1 test split, dataset config, or Motion-Features directory is unavailable")
    dataset_config = read_json(DATASET_CONFIG_PATH)
    id_column = dataset_config["id_column"]; sentence_column = dataset_config.get("sentence_column", "sentence"); gloss_column = dataset_config.get("gloss_column", "gloss")
    heldout_frame = pd.read_csv(TEST_SPLIT_PATH, dtype={id_column: str}, low_memory=False); heldout_frame[id_column] = heldout_frame[id_column].astype(str)
    if (sentence_column not in heldout_frame or gloss_column not in heldout_frame) and OFFICIAL_TRAIN_CSV is not None:
        official_columns = pd.read_csv(OFFICIAL_TRAIN_CSV, nrows=0).columns
        enrichment_columns = [id_column] + [column for column in (sentence_column, gloss_column) if column in official_columns and column not in heldout_frame]
        official_text = pd.read_csv(OFFICIAL_TRAIN_CSV, usecols=enrichment_columns, dtype={id_column: str}, low_memory=False)
        heldout_frame = heldout_frame.merge(official_text, on=id_column, how="left", validate="one_to_one")
    if sentence_column not in heldout_frame:
        raise RuntimeError(f"English sentence column {sentence_column!r} is absent from both the locked split and official train.csv")
    feature_paths = {path.stem: path for path in FEATURE_DIR.glob("*.npy")}
    candidates = []
    for index, row in heldout_frame.sort_values(id_column, kind="stable").iterrows():
        sample_id = str(row[id_column]); sentence = str(row.get(sentence_column, "")).strip(); path = feature_paths.get(sample_id)
        if path is None or not sentence or sentence.casefold() == "nan": continue
        shape = np.load(path, mmap_mode="r", allow_pickle=False).shape
        token_length = row.get("motion_token_length", row.get("token_length", math.ceil(shape[0] / 4)))
        if int(token_length) <= 512 and len(shape) == 2 and shape[1] == 668:
            candidates.append((index, path, shape)); break
    if not candidates: raise RuntimeError("No deterministic supported held-out sample has sentence text and a [T,668] feature file")
    heldout_index, heldout_feature_path, heldout_shape = candidates[0]
    heldout_row = heldout_frame.loc[heldout_index]; heldout_id = str(heldout_row[id_column]); heldout_sentence = str(heldout_row[sentence_column]).strip()
    heldout_gt_gloss = str(heldout_row.get(gloss_column, "")).strip(); heldout_gt_motion = np.load(heldout_feature_path, allow_pickle=False).astype(np.float32)
    heldout_available = True
    print("Sample ID:", heldout_id); print("English sentence:", heldout_sentence)
    print("Ground-truth gloss:", heldout_gt_gloss if heldout_gt_gloss and heldout_gt_gloss.casefold() != "nan" else "UNAVAILABLE")
    print("Ground-truth motion shape:", heldout_gt_motion.shape)
except Exception as error:
    heldout_error = repr(error); print("Ground-truth comparison unavailable"); print("Diagnostic:", heldout_error)

# %% [markdown]
# ## 11 — Run the final pipeline on the held-out sentence

# %%
heldout = None
if heldout_available:
    heldout = extract_result(generate_motion(heldout_sentence, return_intermediates=True))
    heldout_tokens, heldout_motion = heldout["tokens"], heldout["motion"]
    heldout_token_valid = bool(heldout_tokens.shape == (6, heldout["length"]) and heldout_tokens.min() >= 0 and heldout_tokens.max() <= 511)
    heldout_finite = bool(np.isfinite(heldout_motion).all())
    print("=" * 60); print("HELD-OUT TEST EXAMPLE"); print("=" * 60)
    print("Sample ID:", heldout_id); print("English:", heldout_sentence); print("Ground-Truth Gloss:", heldout_gt_gloss)
    print("Predicted Gloss:", heldout["gloss"]); print("Predicted Token Length:", heldout["length"])
    print("Generated RVQ Shape:", heldout_tokens.shape); print("Ground-Truth Motion Shape:", heldout_gt_motion.shape)
    print("Generated Motion Shape:", heldout_motion.shape); print("Token Validity:", heldout_token_valid)
    print("Generated Motion Finite:", heldout_finite); print("=" * 60)
else:
    print("Ground-truth comparison unavailable")

# %% [markdown]
# ## 12 — Ground-truth versus generated duration

# %%
if heldout_available:
    gt_frames, generated_frames = len(heldout_gt_motion), len(heldout_motion); frame_difference = generated_frames - gt_frames
    duration_table = pd.DataFrame({"Metric": ["Ground-truth frames", "Generated frames", "Frame difference", "Absolute duration error", "Relative duration error", "Generated token length (frames / 4)"],
                                  "Value": [gt_frames, generated_frames, frame_difference, abs(frame_difference), abs(frame_difference) / gt_frames, generated_frames / 4]})
    display(duration_table)
else: print("Ground-truth comparison unavailable")

# %% [markdown]
# ## 13 — Native-duration ground truth versus generated features

# %%
if heldout_available:
    heldout_top4 = np.argsort(np.var(heldout_gt_motion, axis=0, dtype=np.float64))[-4:][::-1]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    for axis, feature in zip(axes.ravel(), heldout_top4):
        axis.plot(np.arange(len(heldout_gt_motion)), heldout_gt_motion[:, feature], label="Ground Truth")
        axis.plot(np.arange(len(heldout_motion)), heldout_motion[:, feature], label="Generated", alpha=.85)
        axis.set(title=f"Sample {heldout_id} — Feature {feature}", xlabel="Native frame index"); axis.grid(alpha=.2); axis.legend()
    fig.suptitle("Native durations differ; curves use their own frame axes", y=1.01); fig.tight_layout(); fig.savefig(OUTPUT / "heldout_gt_vs_generated_features.png", dpi=180, bbox_inches="tight"); plt.show()
else: print("Ground-truth comparison unavailable")

# %% [markdown]
# ## 14 — Time-normalized visualization
#
# Generated motion is linearly interpolated along **time only**. This removes the
# duration error for visualization and is not the raw generation.

# %%
def resample_time(values: np.ndarray, frames: int) -> np.ndarray:
    if len(values) == frames: return values.copy()
    old_axis = np.linspace(0.0, 1.0, len(values)); new_axis = np.linspace(0.0, 1.0, frames)
    return np.stack([np.interp(new_axis, old_axis, values[:, feature]) for feature in range(values.shape[1])], axis=1).astype(np.float32)

if heldout_available:
    heldout_aligned = resample_time(heldout_motion, len(heldout_gt_motion))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for axis, feature in zip(axes.ravel(), heldout_top4):
        axis.plot(heldout_gt_motion[:, feature], label="Ground Truth")
        axis.plot(heldout_aligned[:, feature], label="Generated (time-normalized)", alpha=.85)
        axis.set(title=f"Sample {heldout_id} — Feature {feature}", xlabel="Ground-truth frame index"); axis.grid(alpha=.2); axis.legend()
    fig.suptitle("Temporal-resampling visualization — duration error removed", y=1.01); fig.tight_layout(); fig.savefig(OUTPUT / "heldout_time_normalized_features.png", dpi=180, bbox_inches="tight"); plt.show()
else: print("Ground-truth comparison unavailable")

# %% [markdown]
# ## 15 — Simple held-out numerical comparison

# %%
heldout_metrics = {}
if heldout_available:
    difference = heldout_aligned.astype(np.float64) - heldout_gt_motion.astype(np.float64)
    active = motion_std >= 1e-6
    gt_normalized = (heldout_gt_motion[:, active] - motion_mean[active]) / motion_std[active]
    generated_normalized = (heldout_aligned[:, active] - motion_mean[active]) / motion_std[active]
    flat_gt, flat_generated = heldout_gt_motion.ravel().astype(np.float64), heldout_aligned.ravel().astype(np.float64)
    correlation = float(np.corrcoef(flat_gt, flat_generated)[0, 1]) if np.std(flat_gt) > 0 and np.std(flat_generated) > 0 else float("nan")
    velocity_mae = float(np.mean(np.abs(np.diff(generated_normalized, axis=0) - np.diff(gt_normalized, axis=0))))
    heldout_metrics = {"Native absolute duration error (frames)": abs(len(heldout_motion) - len(heldout_gt_motion)), "Time-normalized raw MAE": float(np.mean(np.abs(difference))),
                       "Time-normalized raw RMSE": float(np.sqrt(np.mean(np.square(difference)))), "Flattened correlation": correlation,
                       "Normalized active-feature MAE": float(np.mean(np.abs(generated_normalized - gt_normalized))), "Aligned normalized velocity MAE": velocity_mae}
    display(pd.Series(heldout_metrics, name="Value").to_frame())
    print("MPJPE: NOT CALCULATED")
else: print("Ground-truth comparison unavailable")

# %% [markdown]
# ## 16 — Predicted gloss comparison

# %%
if heldout_available:
    print("Ground Truth Gloss:", heldout_gt_gloss if heldout_gt_gloss and heldout_gt_gloss.casefold() != "nan" else "UNAVAILABLE")
    print("Predicted Gloss:", heldout["gloss"])
    print("Qualitative inspection only; no new gloss benchmark was calculated.")
else: print("Ground-truth comparison unavailable")

# %% [markdown]
# ## 17 — Additional custom inference demonstrations

# %%
additional_rows = []
for sentence in ("Thank you", "Where are you going?", "I am going home"):
    output = extract_result(generate_motion(sentence, return_intermediates=True))
    additional_rows.append({"Input": sentence, "Predicted gloss": output["gloss"], "Predicted token length": output["length"],
                            "RVQ shape": str(tuple(output["tokens"].shape)), "Motion shape": str(tuple(output["motion"].shape)), "Finite": bool(np.isfinite(output["motion"]).all())})
display(pd.DataFrame(additional_rows))
print("Demonstrations only — semantic sign-language correctness is not claimed.")

# %% [markdown]
# ## 18 — Final end-to-end pipeline

# %%
print(r"""
English Sentence
      |
      v
Phase-3 Text -> Gloss
      |
      v
Predicted Gloss
      |
      v
Phase-6 Length Predictor
      |
      v
Predicted Motion Length T
      |
      v
Phase-8 Dynamics Motion Generator
      |
      v
RVQ Tokens [6,T]
      |
      v
Verified Motion-S RVQ Decoder
      |
      v
Continuous Motion [4T,668]
""")
print("Final generator: DYNAMICS"); print("lambda_velocity: 0.1"); print("lambda_persistence: 0.0")

# %% [markdown]
# ## 19 — Previously measured Phase-8 research results

# %%
reference_results = pd.DataFrame({"Metric": ["Length MAE tokens", "Duration MAE frames", "Normalized token edit distance", "Normalized feature MAE", "Velocity MAE", "Collapse rate", "Repetition score", "Invalid token rate", "NaN / Inf"],
                                  "Previously measured Phase-8 test result": [19.785197103781176, 79.07723250201126, .8526705552201046, .8919008301086994, .0893096392996735, .4682220434432824, .4665980782406539, 0, 0]})
display(reference_results.style.set_caption("Previously measured Phase-8 test results"))

# %% [markdown]
# ## 20 — Scientific limitations

# %%
print("=" * 60); print("SCIENTIFIC LIMITATIONS"); print("=" * 60)
print("Continuous representation:\n[T,668]\n")
print("668-D semantic inverse:\nUNRESOLVED\n")
print("Feature -> skeleton conversion:\nNOT VERIFIED\n")
print("Generated BVH:\nNOT VERIFIED\n")
print("MPJPE:\nNOT CALCULATED\n")
print("Human sign-language evaluation:\nNOT PERFORMED\n")
print("Semantic sign correctness:\nNOT CLAIMED\n")
print("Current system status:\nRESEARCH PROTOTYPE"); print("=" * 60)

# %% [markdown]
# ## 21 — Final system status

# %%
demo_pass = bool(custom_valid_tokens and custom_finite and custom_motion.shape == (4 * custom_length, 668))
invalid_generated_ids = int(np.count_nonzero((custom_tokens < 0) | (custom_tokens > 511)))
nan_inf_count = int(np.count_nonzero(~np.isfinite(custom_motion)))
print("=" * 60); print("MOTION-S FINAL SYSTEM DEMO"); print("=" * 60)
print("English -> Gloss:\nPASS\n"); print("Gloss -> Motion Length:\nPASS\n"); print("Gloss + Length -> RVQ Tokens:\nPASS\n"); print("RVQ Tokens -> Continuous Motion:\nPASS\n")
print("RVQ shape:\n[6,T]\n"); print("Motion shape:\n[4T,668]\n"); print("RVQ ID range:\n0..511\n"); print("Final model:\nPhase-8 DYNAMICS\n")
print(f"Invalid generated IDs:\n{invalid_generated_ids}\n"); print(f"NaN / Inf:\n{nan_inf_count}\n"); print("generate_motion():\n" + ("PASS" if demo_pass else "FAIL") + "\n")
print("Training performed in demo notebook:\nNO\n"); print("Project experimental phases:\nCOMPLETE"); print("=" * 60)

# %% [markdown]
# ## 22 — Save final demo artifacts

# %%
np.save(OUTPUT / "custom_demo_tokens.npy", custom_tokens, allow_pickle=False)
np.save(OUTPUT / "custom_demo_motion.npy", custom_motion, allow_pickle=False)
heldout_metadata = {"available": heldout_available, "diagnostic": heldout_error}
if heldout_available:
    np.save(OUTPUT / "heldout_demo_motion.npy", heldout_motion, allow_pickle=False)
    heldout_metadata.update({"sample_id": heldout_id, "sentence": heldout_sentence, "ground_truth_gloss": heldout_gt_gloss,
                             "predicted_gloss": heldout["gloss"], "ground_truth_frames": len(heldout_gt_motion),
                             "generated_frames": len(heldout_motion), "generated_token_length": heldout["length"], "metrics": heldout_metrics})
with (OUTPUT / "heldout_demo_metadata.json").open("w", encoding="utf-8") as handle:
    json.dump(heldout_metadata, handle, indent=2, ensure_ascii=False)
demo_summary = {"notebook_type": "FINAL DEMO / INFERENCE ONLY", "training_performed": False,
                "custom_sentence": custom_text, "predicted_gloss": custom["gloss"], "predicted_length": custom_length,
                "token_shape": list(custom_tokens.shape), "motion_shape": list(custom_motion.shape),
                "token_validity": custom_valid_tokens, "motion_finite": custom_finite,
                "heldout_sample_id": heldout_id if heldout_available else None, "heldout_sentence": heldout_sentence if heldout_available else None,
                "heldout_ground_truth_gloss": heldout_gt_gloss if heldout_available else None,
                "heldout_predicted_gloss": heldout["gloss"] if heldout_available else None,
                "heldout_gt_frames": len(heldout_gt_motion) if heldout_available else None,
                "heldout_generated_frames": len(heldout_motion) if heldout_available else None,
                "final_phase8_model": "DYNAMICS", "lambda_velocity": .1, "lambda_persistence": 0.,
                "semantic_sign_correctness_claimed": False, "demo_passed": demo_pass}
with (OUTPUT / "demo_summary.json").open("w", encoding="utf-8") as handle:
    json.dump(demo_summary, handle, indent=2, ensure_ascii=False)
print("Saved final demo outputs to:", OUTPUT)
display(pd.DataFrame([{"Artifact": path.name, "Bytes": path.stat().st_size} for path in sorted(OUTPUT.iterdir()) if path.is_file()]))

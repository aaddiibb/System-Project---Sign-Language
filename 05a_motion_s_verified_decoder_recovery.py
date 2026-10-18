# %% [markdown]
# # Motion-S / SignVRSE — Phase 5A: Verified RVQ Decoder Recovery
#
# This notebook reconstructs and verifies the provider-compatible Candidate-B decoder, computes
# empirical 668-feature normalization over every valid Motion-Features file, and validates twenty
# ground-truth-token reconstructions. It does not decode Phase-4 predictions, train any model,
# construct a skeleton, export BVH, or calculate joint-space metrics.

# %%
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import display

TOKEN_COLUMNS = ["base_tokens", "residual_1", "residual_2", "residual_3", "residual_4", "residual_5"]
RVQ_LEVELS = 6
CODEBOOK_SIZE = 512
LATENT_DIM = 256
HIDDEN_DIM = 512
MOTION_DIM = 668
TEMPORAL_FACTOR = 4
VALID_MIN, VALID_MAX = 0, 511


@dataclass(frozen=True)
class Phase5ARecoveryConfig:
    seed: int = 42
    kaggle_input: str = "/kaggle/input"
    output_dir: str = "/kaggle/working/motion_s_phase5"
    reconstruction_samples: int = 20
    near_zero_std: float = 1e-6
    sanity_max_mean_mae: float = 1.0
    sanity_max_mean_mse: float = 4.0
    sanity_min_mean_correlation: float = 0.98


CFG = Phase5ARecoveryConfig()
INPUT = Path(CFG.kaggle_input)
OUTPUT = Path(CFG.output_dir)
DIAGNOSTICS = OUTPUT / "diagnostics"
OUTPUT.mkdir(parents=True, exist_ok=True)
DIAGNOSTICS.mkdir(parents=True, exist_ok=True)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(child) for child in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, torch.Tensor): return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, ensure_ascii=False, allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def canonical_id(value: Any) -> str:
    if isinstance(value, (int, np.integer)): return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit(): return text[:-2]
    return text


print("Python:", sys.version.split()[0], "PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
display(pd.Series(asdict(CFG), name="value").to_frame())

# %% [markdown]
# ## 5A.1 — Locate official inputs

# %%
def motion_s_roots() -> list[tuple[Path, Path]]:
    matches: list[tuple[Path, Path]] = []
    if not INPUT.exists(): return matches
    for current, directories, files in os.walk(INPUT):
        if "train.csv" not in files: continue
        directory_map = {name.casefold().replace("-", "").replace("_", ""): name for name in directories}
        if "motionfeatures" in directory_map:
            root = Path(current).resolve()
            motion_dir = (root / directory_map["motionfeatures"]).resolve()
            matches.append((root, motion_dir))
    return list(dict.fromkeys(matches))


dataset_matches = motion_s_roots()
if len(dataset_matches) != 1:
    raise RuntimeError(f"Expected exactly one official Motion-S dataset root; found {[(str(a), str(b)) for a, b in dataset_matches]}")
MOTION_S_ROOT, MOTION_FEATURES = dataset_matches[0]
TRAIN_CSV = MOTION_S_ROOT / "train.csv"

checkpoint_candidates = sorted(path.resolve() for path in INPUT.rglob("rvq_vae_best.pth") if path.is_file())
print("Official checkpoint candidates:")
for candidate in checkpoint_candidates: print(" -", candidate)
if len(checkpoint_candidates) != 1:
    raise RuntimeError(f"Expected exactly one rvq_vae_best.pth; found {len(checkpoint_candidates)}")
CHECKPOINT_PATH = checkpoint_candidates[0]

print("Resolved train.csv:", TRAIN_CSV)
print("Resolved Motion-Features:", MOTION_FEATURES)
print("Resolved checkpoint:", CHECKPOINT_PATH)
write_json(OUTPUT / "phase5a_config.json", {
    **asdict(CFG),
    "resolved_train_csv": TRAIN_CSV,
    "resolved_motion_features": MOTION_FEATURES,
    "resolved_checkpoint": CHECKPOINT_PATH,
})

# %% [markdown]
# ## 5A.2 — Load and verify checkpoint configuration

# %%
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
if not isinstance(checkpoint, dict): raise RuntimeError("Checkpoint top level is not a dictionary")
if "model_state_dict" not in checkpoint: raise RuntimeError("Checkpoint has no model_state_dict")
state = checkpoint["model_state_dict"]
if not isinstance(state, dict) or not state: raise RuntimeError("model_state_dict is empty or invalid")
checkpoint_config = checkpoint.get("config", checkpoint.get("model_config"))
if not isinstance(checkpoint_config, dict): raise RuntimeError("Checkpoint has no dictionary config/model_config")

expected_config = {
    "input_dim": MOTION_DIM,
    "output_dim": MOTION_DIM,
    "hidden_dim": HIDDEN_DIM,
    "latent_dim": LATENT_DIM,
    "num_quantizers": RVQ_LEVELS,
    "num_embeddings": CODEBOOK_SIZE,
    "downsampling_ratio": TEMPORAL_FACTOR,
}
config_mismatches = {}
for key, expected in expected_config.items():
    observed = checkpoint_config.get(key)
    if observed != expected: config_mismatches[key] = {"expected": expected, "observed": observed}
if config_mismatches: raise RuntimeError(f"Checkpoint config mismatch: {config_mismatches}")

print("Checkpoint config:")
display(pd.Series(checkpoint_config, name="value").to_frame())
print("State-dict entries:", len(state))
print("Checkpoint configuration: PASS")

# %% [markdown]
# ## 5A.3–5A.4 — Reconstruct Candidate B and load six codebooks

# %%
class VerifiedMotionSDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.ConvTranspose1d(512, 668, kernel_size=3, stride=1, padding=1, output_padding=0),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


decoder_state = {key[len("decoder."):]: value for key, value in state.items() if key.startswith("decoder.")}
if not decoder_state: raise RuntimeError("No decoder.* tensors found in checkpoint")
verified_decoder = VerifiedMotionSDecoder()
strict_result = verified_decoder.load_state_dict(decoder_state, strict=True)
if strict_result.missing_keys or strict_result.unexpected_keys:
    raise RuntimeError(f"Strict decoder load failed: {strict_result}")
print("Decoder strict load: All keys matched successfully")

codebook_list = []
for level in range(RVQ_LEVELS):
    key = f"rvq.quantizers.{level}.embedding"
    if key not in state: raise RuntimeError(f"Missing codebook tensor {key}")
    tensor = state[key].detach().float()
    if tuple(tensor.shape) != (LATENT_DIM, CODEBOOK_SIZE):
        raise RuntimeError(f"{key}: expected [256,512], got {tuple(tensor.shape)}")
    codebook_list.append(tensor)
    print(f"Codebook L{level}: {list(tensor.shape)} PASS")
codebooks_cpu = torch.stack(codebook_list, dim=0).contiguous()


def validate_tokens(tokens_6t: np.ndarray) -> np.ndarray:
    tokens = np.asarray(tokens_6t)
    if tokens.ndim != 2 or tokens.shape[0] != RVQ_LEVELS or tokens.shape[1] <= 0:
        raise ValueError(f"Expected non-empty [6,T] tokens, got {tokens.shape}")
    if not np.issubdtype(tokens.dtype, np.integer): raise TypeError(f"Token dtype must be integer, got {tokens.dtype}")
    if int(tokens.min()) < VALID_MIN or int(tokens.max()) > VALID_MAX:
        raise ValueError(f"Invalid token IDs [{tokens.min()},{tokens.max()}]")
    return tokens.astype(np.int64, copy=False)


def rvq_tokens_to_latent_cpu(tokens_6t: np.ndarray) -> torch.Tensor:
    tokens = validate_tokens(tokens_6t)
    ids = torch.from_numpy(tokens)
    latent = torch.zeros((LATENT_DIM, tokens.shape[1]), dtype=torch.float32)
    for level in range(RVQ_LEVELS): latent += codebooks_cpu[level][:, ids[level]]
    return latent.unsqueeze(0)

# %% [markdown]
# ## 5A.5 — Compute empirical normalization over every valid Motion-Features file

# %%
train_frame = pd.read_csv(TRAIN_CSV)
id_column = next((column for column in ("id", "sample_id") if column in train_frame.columns), None)
if id_column is None: raise RuntimeError(f"No ID column in {TRAIN_CSV}; columns={train_frame.columns.tolist()}")
train_frame["_sample_id"] = train_frame[id_column].map(canonical_id)
if train_frame["_sample_id"].duplicated().any(): raise RuntimeError("Duplicate IDs in official train.csv")

feature_files = sorted(MOTION_FEATURES.glob("*.npy"), key=lambda path: path.name)
if not feature_files: raise RuntimeError(f"No .npy files found in {MOTION_FEATURES}")
feature_index = {path.stem: path for path in feature_files}
if len(feature_index) != len(feature_files): raise RuntimeError("Duplicate Motion-Features filename stems")

feature_sum = np.zeros(MOTION_DIM, dtype=np.float64)
feature_square_sum = np.zeros(MOTION_DIM, dtype=np.float64)
total_frames = 0
normalization_rows = []
valid_feature_ids = set()
for number, path in enumerate(feature_files, 1):
    status, reason, frames = "valid", "", 0
    try:
        motion = np.load(path, allow_pickle=False)
        if motion.ndim != 2 or motion.shape[1] != MOTION_DIM or motion.shape[0] <= 0:
            raise ValueError(f"expected [T,668], got {motion.shape}")
        if not np.issubdtype(motion.dtype, np.number): raise TypeError(f"non-numeric dtype {motion.dtype}")
        values = np.asarray(motion, dtype=np.float64)
        if not np.isfinite(values).all(): raise FloatingPointError("contains NaN/Inf")
        frames = int(values.shape[0])
        feature_sum += values.sum(axis=0, dtype=np.float64)
        feature_square_sum += np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)
        total_frames += frames
        valid_feature_ids.add(path.stem)
    except Exception as error:
        status, reason = "invalid", f"{type(error).__name__}: {error}"
    normalization_rows.append({"sample_id": path.stem, "path": str(path), "status": status,
                               "frames": frames, "reason": reason})
    if number % 500 == 0 or number == len(feature_files):
        print(f"Normalization scan: {number}/{len(feature_files)} files; valid={len(valid_feature_ids)}")

if total_frames <= 0: raise RuntimeError("No valid frames available for normalization")
motion_mean_64 = feature_sum / total_frames
variance_64 = feature_square_sum / total_frames - np.square(motion_mean_64)
minimum_variance = float(variance_64.min())
if minimum_variance < -1e-7:
    raise RuntimeError(f"Numerical variance failure: minimum variance={minimum_variance}")
variance_64 = np.maximum(variance_64, 0.0)
motion_std_64 = np.sqrt(variance_64)
motion_mean = motion_mean_64.astype(np.float32)
motion_std = motion_std_64.astype(np.float32)
if motion_mean.shape != (MOTION_DIM,) or motion_std.shape != (MOTION_DIM,): raise RuntimeError("Normalization shape failure")
if not np.isfinite(motion_mean).all() or not np.isfinite(motion_std).all(): raise RuntimeError("Non-finite normalization")
near_constant = motion_std < CFG.near_zero_std
safe_std = motion_std.copy(); safe_std[near_constant] = 1.0

normalization_audit = pd.DataFrame(normalization_rows)
normalization_audit.to_csv(DIAGNOSTICS / "normalization_file_audit.csv", index=False)
train_ids = set(train_frame["_sample_id"])
missing_feature_ids = sorted(train_ids - set(feature_index))
invalid_feature_ids = sorted(set(feature_index) - valid_feature_ids)
orphan_feature_ids = sorted(set(feature_index) - train_ids)
normalization_report = {
    "files_discovered": len(feature_files),
    "valid_files_used": len(valid_feature_ids),
    "invalid_files": len(invalid_feature_ids),
    "invalid_file_ids": invalid_feature_ids,
    "train_rows_without_feature_file": len(missing_feature_ids),
    "missing_feature_ids": missing_feature_ids,
    "feature_files_not_in_train_csv": len(orphan_feature_ids),
    "orphan_feature_ids": orphan_feature_ids,
    "total_frames": total_frames,
    "feature_dim": MOTION_DIM,
    "near_constant_dimensions": int(near_constant.sum()),
    "near_constant_indices": np.flatnonzero(near_constant).astype(int).tolist(),
    "accumulator_dtype": "float64",
    "saved_dtype": "float32",
}
write_json(DIAGNOSTICS / "normalization_computation_report.json", normalization_report)
np.save(DIAGNOSTICS / "motion_mean_668.npy", motion_mean, allow_pickle=False)
np.save(DIAGNOSTICS / "motion_std_668.npy", motion_std, allow_pickle=False)
print("Normalization files used:", len(valid_feature_ids))
print("Total frames:", total_frames)
print("Near-constant dimensions:", int(near_constant.sum()))
if invalid_feature_ids: print("Invalid Motion-Features IDs:", invalid_feature_ids)
if missing_feature_ids: print("Train IDs missing Motion-Features:", missing_feature_ids)
if orphan_feature_ids: print("Motion-Features IDs absent from train.csv:", orphan_feature_ids)

# %% [markdown]
# ## 5A.6 — Final verified decoder API

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
verified_decoder = verified_decoder.to(DEVICE).eval()
codebooks = codebooks_cpu.to(DEVICE)
mean_tensor = torch.from_numpy(motion_mean).to(DEVICE)
safe_std_tensor = torch.from_numpy(safe_std).to(DEVICE)


@torch.inference_mode()
def rvq_tokens_to_latent(tokens_6t: np.ndarray) -> torch.Tensor:
    tokens = validate_tokens(tokens_6t)
    ids = torch.as_tensor(tokens, dtype=torch.long, device=DEVICE)
    latent = torch.zeros((LATENT_DIM, tokens.shape[1]), dtype=torch.float32, device=DEVICE)
    for level in range(RVQ_LEVELS): latent += codebooks[level][:, ids[level]]
    return latent.unsqueeze(0)


@torch.inference_mode()
def decode_motion_s_tokens(tokens_6t: np.ndarray, trim_to_frames: int | None = None) -> np.ndarray:
    tokens = validate_tokens(tokens_6t)
    latent = rvq_tokens_to_latent(tokens)
    normalized = verified_decoder(latent)
    expected_shape = (1, MOTION_DIM, TEMPORAL_FACTOR * tokens.shape[1])
    if tuple(normalized.shape) != expected_shape:
        raise RuntimeError(f"Expected decoder output {expected_shape}, got {tuple(normalized.shape)}")
    motion = normalized.squeeze(0).transpose(0, 1) * safe_std_tensor + mean_tensor
    if trim_to_frames is not None:
        if not isinstance(trim_to_frames, (int, np.integer)) or int(trim_to_frames) <= 0:
            raise ValueError("trim_to_frames must be a positive integer or None")
        if int(trim_to_frames) > motion.shape[0]:
            raise ValueError(f"Cannot trim {motion.shape[0]} decoded frames to longer reference {trim_to_frames}")
        motion = motion[:int(trim_to_frames)]
    output = motion.detach().cpu().numpy().astype(np.float32, copy=False)
    if output.ndim != 2 or output.shape[1] != MOTION_DIM: raise RuntimeError(f"Invalid output shape {output.shape}")
    if not np.isfinite(output).all(): raise FloatingPointError("Decoded motion contains NaN/Inf")
    return output

# %% [markdown]
# ## 5A.7 — Deterministic 20-sample ground-truth reconstruction test

# %%
def parse_token_string(value: Any) -> np.ndarray:
    if not isinstance(value, str) or not value.strip(): raise ValueError("empty/non-string token sequence")
    pieces = value.strip().split()
    try:
        array = np.asarray([int(piece) for piece in pieces], dtype=np.int64)
    except ValueError as error:
        raise ValueError("non-integer token value") from error
    if array.size <= 0 or int(array.min()) < VALID_MIN or int(array.max()) > VALID_MAX:
        raise ValueError("token IDs outside 0..511")
    return array


candidate_rows = []
candidate_rejections = []
for row_index, row in train_frame.iterrows():
    sample_id = row["_sample_id"]
    if sample_id not in valid_feature_ids:
        candidate_rejections.append({"sample_id": sample_id, "reason": "missing_or_invalid_motion_feature"})
        continue
    try:
        levels = [parse_token_string(row[column]) for column in TOKEN_COLUMNS]
        lengths = [len(level) for level in levels]
        if len(set(lengths)) != 1: raise ValueError(f"unequal token lengths {lengths}")
        tokens = np.stack(levels, axis=0)
        validate_tokens(tokens)
        reference_frames = int(np.load(feature_index[sample_id], mmap_mode="r", allow_pickle=False).shape[0])
        if TEMPORAL_FACTOR * tokens.shape[1] < reference_frames:
            raise ValueError(f"decoded frames {TEMPORAL_FACTOR * tokens.shape[1]} < reference {reference_frames}")
        candidate_rows.append({"row_index": int(row_index), "sample_id": sample_id,
                               "token_length": int(tokens.shape[1]), "reference_frames": reference_frames})
    except Exception as error:
        candidate_rejections.append({"sample_id": sample_id, "reason": f"{type(error).__name__}: {error}"})

if len(candidate_rows) < CFG.reconstruction_samples:
    raise RuntimeError(f"Only {len(candidate_rows)} valid reconstruction candidates; need {CFG.reconstruction_samples}")
rng = np.random.default_rng(CFG.seed)
selected_indices = sorted(rng.choice(len(candidate_rows), CFG.reconstruction_samples, replace=False).tolist())
selected_candidates = [candidate_rows[index] for index in selected_indices]
pd.DataFrame(candidate_rejections).to_csv(DIAGNOSTICS / "reconstruction_candidate_rejections.csv", index=False)


def flat_correlation(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2: return float("nan")
    x -= x.mean(); y -= y.mean()
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denominator <= 0 or not np.isfinite(denominator): return float("nan")
    return float(np.dot(x, y) / denominator)


reconstruction_rows = []
for selected in selected_candidates:
    row = train_frame.loc[selected["row_index"]]
    sample_id = selected["sample_id"]
    tokens = np.stack([parse_token_string(row[column]) for column in TOKEN_COLUMNS], axis=0)
    reference = np.load(feature_index[sample_id], allow_pickle=False).astype(np.float32, copy=False)
    decoded_before_trim = decode_motion_s_tokens(tokens, trim_to_frames=None)
    decoded = decode_motion_s_tokens(tokens, trim_to_frames=int(reference.shape[0]))
    difference = decoded.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(np.square(difference)))
    reconstruction_rows.append({
        "sample_id": sample_id,
        "token_length": int(tokens.shape[1]),
        "decoded_frames_before_trim": int(decoded_before_trim.shape[0]),
        "reference_frames": int(reference.shape[0]),
        "frame_difference": int(decoded_before_trim.shape[0] - reference.shape[0]),
        "mae": float(np.mean(np.abs(difference))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "flattened_correlation": flat_correlation(decoded, reference),
        "finite": bool(np.isfinite(decoded).all()),
    })
    print("Reconstructed", sample_id, "tokens", tokens.shape, "motion", decoded.shape)

reconstruction_frame = pd.DataFrame(reconstruction_rows)
reconstruction_frame.to_csv(DIAGNOSTICS / "ground_truth_reconstruction_20.csv", index=False)
mean_mae = float(reconstruction_frame["mae"].mean())
mean_mse = float(reconstruction_frame["mse"].mean())
mean_rmse = float(reconstruction_frame["rmse"].mean())
mean_correlation = float(reconstruction_frame["flattened_correlation"].mean())
nan_failures = int(reconstruction_frame.isna().any(axis=1).sum())
inf_failures = int((~np.isfinite(reconstruction_frame.select_dtypes(include=[np.number]).to_numpy())).any(axis=1).sum())
all_finite = bool(reconstruction_frame["finite"].all() and nan_failures == 0 and inf_failures == 0)
quality_sanity_pass = bool(
    mean_mae <= CFG.sanity_max_mean_mae
    and mean_mse <= CFG.sanity_max_mean_mse
    and mean_correlation >= CFG.sanity_min_mean_correlation
)
print("20-sample mean MAE:", mean_mae)
print("20-sample mean MSE:", mean_mse)
print("20-sample mean RMSE:", mean_rmse)
print("20-sample mean correlation:", mean_correlation)
if not quality_sanity_pass:
    raise RuntimeError(
        "Candidate-B reconstruction is far outside the conservative sanity range: "
        f"MAE<={CFG.sanity_max_mean_mae}, MSE<={CFG.sanity_max_mean_mse}, "
        f"correlation>={CFG.sanity_min_mean_correlation}."
    )
if not all_finite: raise RuntimeError("20-sample reconstruction contains NaN/Inf")

# %% [markdown]
# ## 5A.9–5A.10 — Save decoder contract and reload required artifacts

# %%
MEAN_PATH = DIAGNOSTICS / "motion_mean_668.npy"
STD_PATH = DIAGNOSTICS / "motion_std_668.npy"
CONTRACT_PATH = DIAGNOSTICS / "decoder_contract.json"

decoder_layers = [
    {"type": "ConvTranspose1d", "in": 256, "out": 512, "kernel": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "ConvTranspose1d", "in": 512, "out": 512, "kernel": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "ConvTranspose1d", "in": 512, "out": 512, "kernel": 3, "stride": 1, "padding": 1, "output_padding": 0},
    {"type": "ConvTranspose1d", "in": 512, "out": 668, "kernel": 3, "stride": 1, "padding": 1, "output_padding": 0},
]
decoder_contract = {
    "phase": "Phase 5A - Verified RVQ Decoder Recovery",
    "phase5a_complete": True,
    "checkpoint": str(CHECKPOINT_PATH),
    "checkpoint_sha256": sha256_file(CHECKPOINT_PATH),
    "checkpoint_config": expected_config,
    "token_representation": {
        "shape": "[6,T]", "levels": RVQ_LEVELS, "num_quantizers": RVQ_LEVELS,
        "valid_ids": [VALID_MIN, VALID_MAX], "codebook_size": CODEBOOK_SIZE,
        "num_embeddings": CODEBOOK_SIZE, "latent_dim": LATENT_DIM,
    },
    "motion_representation": {
        "shape": "[T,668]", "feature_dim": MOTION_DIM, "motion_dim": MOTION_DIM,
        "temporal_upsampling_factor": TEMPORAL_FACTOR, "temporal_expansion": TEMPORAL_FACTOR,
    },
    "decoder": {
        "architecture": "Candidate B", "strict_load": True, "hidden_dim": HIDDEN_DIM,
        "layers": decoder_layers,
    },
    "codebooks": {"verified": True, "count": RVQ_LEVELS, "shape_each": [LATENT_DIM, CODEBOOK_SIZE]},
    "normalization": {
        "type": "empirically recovered per-feature z-score",
        "official_statistics_confirmed": False,
        "mean_file": str(MEAN_PATH), "std_file": str(STD_PATH), "feature_dim": MOTION_DIM,
        "files_used": len(valid_feature_ids), "total_frames": total_frames,
        "near_constant_dimensions": int(near_constant.sum()),
    },
    "reconstruction_test": {
        "samples": CFG.reconstruction_samples,
        "mean_mae": mean_mae, "mean_mse": mean_mse, "mean_rmse": mean_rmse,
        "mean_correlation": mean_correlation, "nan_failures": nan_failures,
        "inf_failures": inf_failures, "all_finite": all_finite,
        "quality_sanity_pass": quality_sanity_pass,
    },
    "ground_truth_trim_policy": "Trim only padded decoder tail to known original frame count.",
    "generated_motion_trim_policy": "Do not trim generated motion.",
    "ready_for_phase5b": True,
}
write_json(CONTRACT_PATH, decoder_contract)

with CONTRACT_PATH.open(encoding="utf-8") as handle: reloaded_contract = json.load(handle)
reloaded_mean = np.load(MEAN_PATH, allow_pickle=False)
reloaded_std = np.load(STD_PATH, allow_pickle=False)
artifact_verification = {
    "decoder_contract_path": str(CONTRACT_PATH.resolve()),
    "motion_mean_path": str(MEAN_PATH.resolve()),
    "motion_std_path": str(STD_PATH.resolve()),
    "contract_parses": isinstance(reloaded_contract, dict),
    "contract_ready_for_phase5b": reloaded_contract.get("ready_for_phase5b") is True,
    "mean_shape": list(reloaded_mean.shape),
    "std_shape": list(reloaded_std.shape),
    "mean_dtype": str(reloaded_mean.dtype),
    "std_dtype": str(reloaded_std.dtype),
    "mean_finite": bool(np.isfinite(reloaded_mean).all()),
    "std_finite": bool(np.isfinite(reloaded_std).all()),
}
artifact_reload_pass = bool(
    artifact_verification["contract_parses"]
    and artifact_verification["contract_ready_for_phase5b"]
    and reloaded_mean.shape == (MOTION_DIM,)
    and reloaded_std.shape == (MOTION_DIM,)
    and reloaded_mean.dtype == np.float32
    and reloaded_std.dtype == np.float32
    and artifact_verification["mean_finite"]
    and artifact_verification["std_finite"]
)
artifact_verification["passed"] = artifact_reload_pass
write_json(DIAGNOSTICS / "phase5a_artifact_verification.json", artifact_verification)
if not artifact_reload_pass: raise RuntimeError(f"Saved artifact reload verification failed: {artifact_verification}")
print("Saved artifact reload verification: PASS")
print("Decoder contract:", CONTRACT_PATH.resolve())
print("Motion mean:", MEAN_PATH.resolve())
print("Motion std:", STD_PATH.resolve())

# %% [markdown]
# ## 5A.11–5A.12 — Final report and Phase-5B handoff

# %%
phase5a_report = {
    "phase": "Phase 5A - Verified RVQ Decoder Recovery",
    "phase5a_complete": True,
    "checkpoint": str(CHECKPOINT_PATH),
    "checkpoint_sha256": sha256_file(CHECKPOINT_PATH),
    "checkpoint_config_verified": True,
    "strict_decoder_load": True,
    "codebook_verification": {"passed": True, "count": RVQ_LEVELS, "shape": [LATENT_DIM, CODEBOOK_SIZE]},
    "normalization_files_used": len(valid_feature_ids),
    "normalization_invalid_files": len(invalid_feature_ids),
    "total_frames_used": total_frames,
    "near_constant_feature_count": int(near_constant.sum()),
    "reconstruction_test": decoder_contract["reconstruction_test"],
    "all_finite": all_finite,
    "artifact_verification": artifact_verification,
    "ready_for_phase5b": artifact_reload_pass and quality_sanity_pass and all_finite,
}
write_json(OUTPUT / "phase5a_report.json", phase5a_report)

readme = f"""PHASE 5B INPUT CONTRACT — MOTION-S / SIGNVRSE

Required artifacts:
{CONTRACT_PATH}
{MEAN_PATH}
{STD_PATH}

Reconstruct the verified Candidate-B decoder and strict-load decoder weights from:
{CHECKPOINT_PATH}

RVQ token input: [6,T]
Valid token IDs: 0..511
Latent after six codebook lookups and summation: [256,T]
Decoded normalized motion: [4T,668]
Final inverse-normalized continuous motion: [4T,668]

Normalization is an empirically recovered per-feature z-score computed from every valid
Motion-Features file. It is not claimed to be official provider normalization.

Ground-truth reconstruction may trim only the decoder's padded tail when the original frame count
is known. Do not trim Phase-4 generated sequences.
"""
(OUTPUT / "README_phase5b.txt").write_text(readme, encoding="utf-8")

ready_for_phase5b = bool(phase5a_report["ready_for_phase5b"])
if not ready_for_phase5b: raise RuntimeError("PHASE 5A COMPLETION GATE FAILED")

print("\n============================================================")
print("PHASE 5A COMPLETE")
print("============================================================")
print("Checkpoint:", CHECKPOINT_PATH)
print("Decoder strict load: PASS")
print(f"Codebooks: {RVQ_LEVELS} / {RVQ_LEVELS} verified")
print("Codebook shape: [256,512]")
print("Motion feature dimension: 668")
print("Temporal expansion: 4x")
print("Normalization files used:", len(valid_feature_ids))
print("Total frames for statistics:", total_frames)
print("Near-constant dimensions:", int(near_constant.sum()))
print("20-sample mean MAE:", mean_mae)
print("20-sample mean MSE:", mean_mse)
print("20-sample mean correlation:", mean_correlation)
print("Saved:", CONTRACT_PATH)
print("Saved:", MEAN_PATH)
print("Saved:", STD_PATH)
print("Artifact reload verification: PASS")
print("READY FOR PHASE 5B: YES")
print("============================================================")
print("STOP. Do not begin Phase 5B automatically.")

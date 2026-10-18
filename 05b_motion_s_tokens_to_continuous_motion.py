# %% [markdown]
# # Motion-S / SignVRSE — Phase 5B: Generated RVQ Tokens to Continuous Motion
#
# This notebook performs inference and evaluation only. It never trains a model, alters Phase-4
# token sequences, invents 668-D feature semantics, constructs a skeleton, or exports BVH.
# It requires the verified Phase-5A Candidate-B decoder contract and empirically recovered
# per-feature Motion-S normalization artifacts.

# %%
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
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

RVQ_LEVELS = 6
CODEBOOK_SIZE = 512
LATENT_DIM = 256
HIDDEN_DIM = 512
MOTION_DIM = 668
TEMPORAL_EXPANSION = 4
VALID_TOKEN_MIN, VALID_TOKEN_MAX = 0, 511
FORBIDDEN_IDS = {512, 513, 514}


@dataclass(frozen=True)
class Phase5BConfig:
    seed: int = 42
    kaggle_input: str = "/kaggle/input"
    kaggle_working: str = "/kaggle/working"
    output_dir: str = "/kaggle/working/motion_s_phase5"
    expected_test_samples: int = 1243
    small_gate_samples: int = 10
    qualitative_per_band: int = 5
    trajectory_feature_count: int = 4
    near_zero_std: float = 1e-6


CFG = Phase5BConfig()
OUTPUT = Path(CFG.output_dir)
DIAGNOSTICS = OUTPUT / "diagnostics"
DECODED = OUTPUT / "decoded"
ORACLE_DECODED = DECODED / "oracle"
END2END_DECODED = DECODED / "end2end"
RESULTS = OUTPUT / "results"
FIGURES = OUTPUT / "figures"
SEARCH_BASES = [Path(CFG.kaggle_input), Path(CFG.kaggle_working)]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return None if not np.isfinite(value) else float(value)
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


def canonical(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def find_named(name: str) -> list[Path]:
    matches: list[Path] = []
    for base in SEARCH_BASES:
        if not base.exists(): continue
        matches.extend(path.resolve() for path in base.rglob(name) if path.is_file())
    return sorted(dict.fromkeys(matches), key=str)


def resolve_required_file(label: str, name: str) -> Path:
    matches = find_named(name)
    if not matches:
        raise RuntimeError(f"Missing required {label} ({name}). Searched: {[str(x) for x in SEARCH_BASES]}")
    if len(matches) > 1:
        hashes = {path: sha256_file(path) for path in matches}
        if len(set(hashes.values())) != 1:
            raise RuntimeError(f"Ambiguous {label}; non-identical copies found: {[str(x) for x in matches]}")
        preferred = [path for path in matches if path_is_within(path, Path(CFG.kaggle_working))]
        selected = preferred[0] if len(preferred) == 1 else matches[0]
        print(f"{label}: identical duplicate copies found; selected {selected}")
    else:
        selected = matches[0]
    print(f"{label}: {selected}")
    return selected


def find_motion_feature_dirs() -> list[Path]:
    matches: list[Path] = []
    for base in SEARCH_BASES:
        if not base.exists(): continue
        for current, directories, _ in os.walk(base):
            for directory in directories:
                if canonical(directory) == "motionfeatures":
                    matches.append((Path(current) / directory).resolve())
    return sorted(dict.fromkeys(matches), key=str)


def resolve_motion_features() -> Path:
    matches = find_motion_feature_dirs()
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one Motion-Features directory; found {[str(x) for x in matches]}")
    print("Ground-truth Motion-Features:", matches[0])
    return matches[0]


def find_optional_csv(name: str, required_columns: set[str]) -> Optional[Path]:
    valid = []
    for path in find_named(name):
        try:
            columns = set(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue
        if required_columns.issubset(columns): valid.append(path)
    if len(valid) == 1:
        print(f"Optional {name}: {valid[0]}")
        return valid[0]
    if len(valid) > 1:
        hashes = {path: sha256_file(path) for path in valid}
        if len(set(hashes.values())) == 1:
            print(f"Optional {name}: identical copies; selected {valid[0]}")
            return valid[0]
        print(f"Optional {name}: ambiguous candidates; analysis disabled: {[str(x) for x in valid]}")
    else:
        print(f"Optional {name}: unavailable")
    return None


print("Python:", sys.version.split()[0], "PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
display(pd.Series(asdict(CFG), name="value").to_frame())

# %% [markdown]
# ## 5B.0 — Resolve required inputs
#
# Exact filenames are searched under `/kaggle/input` and `/kaggle/working`. Non-identical duplicate
# copies are treated as ambiguous and stop execution. Identical duplicates are recorded and one copy
# is selected deterministically.

# %%
ORACLE_NPZ = resolve_required_file("Phase-4 oracle tokens", "test_oracle_tokens.npz")
END2END_NPZ = resolve_required_file("Phase-4 end-to-end tokens", "test_end2end_tokens.npz")
DECODER_CONTRACT_PATH = resolve_required_file("Phase-5A decoder contract", "decoder_contract.json")
MOTION_MEAN_PATH = resolve_required_file("recovered motion mean", "motion_mean_668.npy")
MOTION_STD_PATH = resolve_required_file("recovered motion std", "motion_std_668.npy")
CHECKPOINT_PATH = resolve_required_file("official RVQ-VAE checkpoint", "rvq_vae_best.pth")
MOTION_FEATURES = resolve_motion_features()

PHASE4_ORACLE_PREDICTIONS = find_optional_csv(
    "test_oracle_predictions.csv", {"sample_id", "aligned_token_accuracy", "mean_normalized_edit_distance"}
)
PHASE4_END2END_PREDICTIONS = find_optional_csv(
    "test_end2end_predictions.csv", {"sample_id", "aligned_token_accuracy", "mean_normalized_edit_distance"}
)
PHASE3_TEST_PREDICTIONS = find_optional_csv(
    "test_predictions.csv", {"sample_id", "predicted_gloss", "lexical_token_f1"}
)
PHASE1_TEST_SPLIT = find_optional_csv("test_split.csv", {"sentence", "gloss"})

for directory in (OUTPUT, DIAGNOSTICS, DECODED, ORACLE_DECODED, END2END_DECODED, RESULTS, FIGURES):
    directory.mkdir(parents=True, exist_ok=True)

write_json(OUTPUT / "phase5b_config.json", {
    **asdict(CFG),
    "resolved_paths": {
        "oracle_tokens": ORACLE_NPZ,
        "end2end_tokens": END2END_NPZ,
        "decoder_contract": DECODER_CONTRACT_PATH,
        "motion_mean": MOTION_MEAN_PATH,
        "motion_std": MOTION_STD_PATH,
        "checkpoint": CHECKPOINT_PATH,
        "motion_features": MOTION_FEATURES,
    },
})

# %% [markdown]
# ## 5B.1 — Verify Phase-4 token files

# %%
class PackedTokenFile:
    def __init__(self, path: Path, expected_samples: int):
        self.path = path
        with np.load(path, allow_pickle=False) as data:
            required = {"sample_ids", "lengths", "offsets", "tokens_flat"}
            if not required.issubset(data.files):
                raise RuntimeError(f"{path} missing arrays: {required - set(data.files)}")
            self.sample_ids = data["sample_ids"].astype(str).copy()
            self.lengths = data["lengths"].astype(np.int64).copy()
            self.offsets = data["offsets"].astype(np.int64).copy()
            self.tokens_flat = data["tokens_flat"].copy()
        if len(self.sample_ids) != expected_samples:
            raise RuntimeError(f"{path.name}: expected {expected_samples} samples, found {len(self.sample_ids)}")
        if len(set(self.sample_ids.tolist())) != len(self.sample_ids):
            raise RuntimeError(f"{path.name}: duplicate sample IDs")
        if self.tokens_flat.ndim != 2 or self.tokens_flat.shape[0] != RVQ_LEVELS:
            raise RuntimeError(f"{path.name}: expected packed shape [6,sum(T)], got {self.tokens_flat.shape}")
        if not np.issubdtype(self.tokens_flat.dtype, np.integer):
            raise RuntimeError(f"{path.name}: token dtype is not integer: {self.tokens_flat.dtype}")
        if len(self.lengths) != len(self.sample_ids) or len(self.offsets) != len(self.sample_ids) + 1:
            raise RuntimeError(f"{path.name}: inconsistent IDs/lengths/offsets")
        if self.offsets[0] != 0 or self.offsets[-1] != self.tokens_flat.shape[1]:
            raise RuntimeError(f"{path.name}: invalid packed offsets")
        if not np.array_equal(np.diff(self.offsets), self.lengths):
            raise RuntimeError(f"{path.name}: offsets do not match lengths")
        if np.any(self.lengths <= 0): raise RuntimeError(f"{path.name}: zero/negative token length")
        if self.tokens_flat.size:
            token_min, token_max = int(self.tokens_flat.min()), int(self.tokens_flat.max())
            if token_min < VALID_TOKEN_MIN or token_max > VALID_TOKEN_MAX:
                raise RuntimeError(f"{path.name}: invalid token range [{token_min},{token_max}]")
            if np.isin(self.tokens_flat, list(FORBIDDEN_IDS)).any():
                raise RuntimeError(f"{path.name}: PAD/BOS/EOS IDs appear in decoder input")
        for index in range(len(self.sample_ids)):
            sequence = self.sequence(index)
            if sequence.ndim != 2 or sequence.shape != (RVQ_LEVELS, int(self.lengths[index])):
                raise RuntimeError(f"{path.name}: malformed sequence at index {index}: {sequence.shape}")
        self.index = {sample_id: i for i, sample_id in enumerate(self.sample_ids)}

    def sequence(self, index_or_id: int | str) -> np.ndarray:
        index = self.index[str(index_or_id)] if isinstance(index_or_id, str) else int(index_or_id)
        start, stop = int(self.offsets[index]), int(self.offsets[index + 1])
        return self.tokens_flat[:, start:stop]

    def report(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "samples": len(self.sample_ids),
            "unique_ids": len(set(self.sample_ids.tolist())),
            "packed_shape": list(self.tokens_flat.shape),
            "dtype": str(self.tokens_flat.dtype),
            "minimum_id": int(self.tokens_flat.min()),
            "maximum_id": int(self.tokens_flat.max()),
            "minimum_length": int(self.lengths.min()),
            "maximum_length": int(self.lengths.max()),
            "forbidden_ids_present": False,
            "passed": True,
        }


ORACLE_TOKENS = PackedTokenFile(ORACLE_NPZ, CFG.expected_test_samples)
END2END_TOKENS = PackedTokenFile(END2END_NPZ, CFG.expected_test_samples)
if set(ORACLE_TOKENS.sample_ids) != set(END2END_TOKENS.sample_ids):
    missing_oracle = sorted(set(END2END_TOKENS.sample_ids) - set(ORACLE_TOKENS.sample_ids))
    missing_end2end = sorted(set(ORACLE_TOKENS.sample_ids) - set(END2END_TOKENS.sample_ids))
    raise RuntimeError(f"Oracle/end-to-end ID mismatch. Missing oracle={missing_oracle[:10]}, missing end2end={missing_end2end[:10]}")

TOKEN_INPUT_REPORT = {
    "oracle": ORACLE_TOKENS.report(),
    "end2end": END2END_TOKENS.report(),
    "same_sample_id_set": True,
    "expected_supported_test_count": CFG.expected_test_samples,
    "passed": True,
}
write_json(DIAGNOSTICS / "token_input_verification.json", TOKEN_INPUT_REPORT)
print("Phase-4 token verification: PASS")

# %% [markdown]
# ## 5B.2–5B.4 — Strict-load Candidate-B decoder and recovered normalization

# %%
with DECODER_CONTRACT_PATH.open(encoding="utf-8") as handle:
    DECODER_CONTRACT = json.load(handle)


def flatten_contract(value: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_contract(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value): flat.update(flatten_contract(child, f"{prefix}[{index}]"))
    else:
        flat[prefix] = value
    return flat


CONTRACT_FLAT = flatten_contract(DECODER_CONTRACT)


def require_contract_number(aliases: set[str], expected: int, label: str) -> None:
    values = [value for key, value in CONTRACT_FLAT.items() if canonical(key.split(".")[-1]) in aliases]
    numeric = [int(value) for value in values if isinstance(value, (int, float, np.integer, np.floating))]
    if expected not in numeric:
        raise RuntimeError(f"Decoder contract does not verify {label}={expected}; observed {values}")


require_contract_number({"inputdim", "outputdim", "motiondim", "featuredim"}, MOTION_DIM, "motion_dim")
require_contract_number({"hiddendim"}, HIDDEN_DIM, "hidden_dim")
require_contract_number({"latentdim", "embeddingdim"}, LATENT_DIM, "latent_dim")
require_contract_number({"numquantizers", "rvqlevels", "levels"}, RVQ_LEVELS, "rvq_levels")
require_contract_number({"numembeddings", "codebooksize"}, CODEBOOK_SIZE, "codebook_size")
require_contract_number(
    {"temporalcompression", "temporalexpansion", "temporalupsamplingfactor",
     "downsamplingratio", "upsamplingratio"},
    TEMPORAL_EXPANSION,
    "temporal expansion",
)
contract_text = canonical(json.dumps(DECODER_CONTRACT, sort_keys=True))
if "candidateb" not in contract_text:
    raise RuntimeError("decoder_contract.json does not explicitly identify verified Candidate B")

motion_mean = np.load(MOTION_MEAN_PATH, allow_pickle=False).astype(np.float32)
motion_std = np.load(MOTION_STD_PATH, allow_pickle=False).astype(np.float32)
if motion_mean.shape != (MOTION_DIM,) or motion_std.shape != (MOTION_DIM,):
    raise RuntimeError(f"Expected normalization arrays [668], got mean={motion_mean.shape}, std={motion_std.shape}")
if not np.isfinite(motion_mean).all() or not np.isfinite(motion_std).all() or np.any(motion_std < 0):
    raise RuntimeError("Normalization arrays contain invalid values")
active_dimensions = motion_std >= CFG.near_zero_std
if not active_dimensions.any(): raise RuntimeError("Recovered normalization has no active feature dimensions")
safe_std = motion_std.copy()
safe_std[~active_dimensions] = 1.0
print("Empirically recovered normalization active dimensions:", int(active_dimensions.sum()), "/", MOTION_DIM)


class MotionDecoder(nn.Module):
    """Exact verified Phase-5A Candidate-B decoder."""

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


def unwrap_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    current = checkpoint
    for key in ("model_state_dict", "state_dict"):
        if isinstance(current, dict) and key in current and isinstance(current[key], dict):
            current = current[key]
            break
    if not isinstance(current, dict) or not all(isinstance(key, str) for key in current):
        raise RuntimeError("Unsupported RVQ-VAE checkpoint structure")
    tensors = {key: value for key, value in current.items() if isinstance(value, torch.Tensor)}
    if not tensors: raise RuntimeError("Checkpoint contains no tensor state dictionary")
    return tensors


checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
state = unwrap_state_dict(checkpoint)
decoder_state = {key[len("decoder."):]: value for key, value in state.items() if key.startswith("decoder.")}
decoder = MotionDecoder()
load_result = decoder.load_state_dict(decoder_state, strict=True)
if load_result.missing_keys or load_result.unexpected_keys:
    raise RuntimeError(f"Strict decoder load failed: {load_result}")

codebook_list = []
for level in range(RVQ_LEVELS):
    key = f"rvq.quantizers.{level}.embedding"
    if key not in state: raise RuntimeError(f"Missing checkpoint codebook: {key}")
    tensor = state[key]
    if tuple(tensor.shape) != (LATENT_DIM, CODEBOOK_SIZE):
        raise RuntimeError(f"{key}: expected [256,512], got {tuple(tensor.shape)}")
    codebook_list.append(tensor.detach().float())
codebooks_cpu = torch.stack(codebook_list, dim=0).contiguous()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decoder = decoder.to(DEVICE).eval()
codebooks = codebooks_cpu.to(DEVICE)
mean_tensor = torch.from_numpy(motion_mean).to(DEVICE)
std_tensor = torch.from_numpy(safe_std).to(DEVICE)


@torch.inference_mode()
def decode_motion_s_tokens(tokens_6t: np.ndarray) -> np.ndarray:
    tokens = np.asarray(tokens_6t)
    if tokens.ndim != 2 or tokens.shape[0] != RVQ_LEVELS or tokens.shape[1] <= 0:
        raise ValueError(f"Expected non-empty [6,T] tokens, got {tokens.shape}")
    if not np.issubdtype(tokens.dtype, np.integer): raise TypeError(f"Token dtype must be integer, got {tokens.dtype}")
    if int(tokens.min()) < VALID_TOKEN_MIN or int(tokens.max()) > VALID_TOKEN_MAX:
        raise ValueError(f"Token IDs outside 0..511: [{tokens.min()},{tokens.max()}]")
    token_tensor = torch.as_tensor(tokens.astype(np.int64, copy=False), device=DEVICE)
    latent = torch.zeros((LATENT_DIM, token_tensor.shape[1]), dtype=torch.float32, device=DEVICE)
    for level in range(RVQ_LEVELS): latent += codebooks[level][:, token_tensor[level]]
    normalized_motion = decoder(latent.unsqueeze(0)).squeeze(0).transpose(0, 1)
    if tuple(normalized_motion.shape) != (TEMPORAL_EXPANSION * tokens.shape[1], MOTION_DIM):
        raise RuntimeError(f"Decoder returned wrong shape {tuple(normalized_motion.shape)}")
    motion = normalized_motion * std_tensor + mean_tensor
    motion_np = motion.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(motion_np).all(): raise FloatingPointError("Decoded motion contains NaN/Inf")
    return motion_np


DECODER_LOAD_REPORT = {
    "checkpoint": CHECKPOINT_PATH,
    "checkpoint_sha256": sha256_file(CHECKPOINT_PATH),
    "decoder_candidate": "Candidate B",
    "strict_load": True,
    "missing_keys": list(load_result.missing_keys),
    "unexpected_keys": list(load_result.unexpected_keys),
    "codebook_shapes": [list(tensor.shape) for tensor in codebook_list],
    "motion_dim": MOTION_DIM,
    "temporal_expansion": TEMPORAL_EXPANSION,
    "normalization": "empirically recovered Motion-S feature normalization",
    "official_statistics_confirmed": False,
    "active_dimensions": int(active_dimensions.sum()),
}
write_json(DIAGNOSTICS / "decoder_load_verification.json", DECODER_LOAD_REPORT)
decoder_contract_destination = DIAGNOSTICS / "decoder_contract.json"
if DECODER_CONTRACT_PATH.resolve() != decoder_contract_destination.resolve():
    shutil.copy2(DECODER_CONTRACT_PATH, decoder_contract_destination)
np.save(DIAGNOSTICS / "motion_mean_668.npy", motion_mean)
np.save(DIAGNOSTICS / "motion_std_668.npy", motion_std)
print("Candidate-B decoder strict-load: PASS")

# %% [markdown]
# ## 5B.5–5B.6 — Ground-truth lookup and ten-sample decoding gate

# %%
gt_index: dict[str, Path] = {}
for path in MOTION_FEATURES.glob("*.npy"):
    if path.stem in gt_index: raise RuntimeError(f"Duplicate Motion-Features ID: {path.stem}")
    gt_index[path.stem] = path
all_sample_ids = ORACLE_TOKENS.sample_ids.tolist()
missing_gt = [sample_id for sample_id in all_sample_ids if sample_id not in gt_index]
if missing_gt:
    write_json(DIAGNOSTICS / "missing_ground_truth_motion.json", {"count": len(missing_gt), "sample_ids": missing_gt})
    raise RuntimeError(f"Missing ground-truth Motion-Features for {len(missing_gt)} IDs: {missing_gt[:20]}")


def load_ground_truth(sample_id: str) -> np.ndarray:
    motion = np.load(gt_index[sample_id], allow_pickle=False)
    if motion.ndim != 2 or motion.shape[1] != MOTION_DIM or motion.shape[0] <= 0:
        raise RuntimeError(f"Ground-truth {sample_id}: expected [T,668], got {motion.shape}")
    motion = motion.astype(np.float32, copy=False)
    if not np.isfinite(motion).all(): raise FloatingPointError(f"Ground-truth {sample_id} contains NaN/Inf")
    return motion


small_gate_ids = all_sample_ids[:CFG.small_gate_samples]
small_gate_rows = []
for sample_id in small_gate_ids:
    oracle_tokens = ORACLE_TOKENS.sequence(sample_id)
    end_tokens = END2END_TOKENS.sequence(sample_id)
    gt_motion = load_ground_truth(sample_id)
    oracle_motion = decode_motion_s_tokens(oracle_tokens)
    end_motion = decode_motion_s_tokens(end_tokens)
    row = {
        "sample_id": sample_id,
        "oracle_token_shape": list(oracle_tokens.shape),
        "end2end_token_shape": list(end_tokens.shape),
        "ground_truth_shape": list(gt_motion.shape),
        "oracle_decoded_shape": list(oracle_motion.shape),
        "end2end_decoded_shape": list(end_motion.shape),
        "oracle_finite": bool(np.isfinite(oracle_motion).all()),
        "end2end_finite": bool(np.isfinite(end_motion).all()),
    }
    small_gate_rows.append(row)
    print(row)
write_json(DIAGNOSTICS / "small_decoding_gate.json", {"samples": small_gate_rows, "passed": True})
print("Small decoding gate: PASS")

# %% [markdown]
# ## 5B.7–5B.8 — Full oracle and end-to-end decoding

# %%
def decode_collection(label: str, packed: PackedTokenFile, destination: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows, failures = [], []
    for number, sample_id in enumerate(all_sample_ids, 1):
        tokens = packed.sequence(sample_id)
        gt_frames = int(load_ground_truth(sample_id).shape[0])
        output_path = destination / f"{sample_id}.npy"
        try:
            motion = decode_motion_s_tokens(tokens)
            np.save(output_path, motion, allow_pickle=False)
            rows.append({
                "sample_id": sample_id,
                "token_length": int(tokens.shape[1]),
                "decoded_frames": int(motion.shape[0]),
                "ground_truth_frames": gt_frames,
                "path": str(output_path.relative_to(OUTPUT)),
                "finite": bool(np.isfinite(motion).all()),
            })
        except Exception as error:
            failures.append({"sample_id": sample_id, "error_type": type(error).__name__, "message": str(error)})
        if number % 100 == 0 or number == len(all_sample_ids):
            print(f"{label}: {number}/{len(all_sample_ids)} processed")
    return pd.DataFrame(rows), failures


oracle_manifest, oracle_failures = decode_collection("oracle", ORACLE_TOKENS, ORACLE_DECODED)
end2end_manifest, end2end_failures = decode_collection("end-to-end", END2END_TOKENS, END2END_DECODED)
oracle_manifest.to_csv(DECODED / "oracle_manifest.csv", index=False)
end2end_manifest.to_csv(DECODED / "end2end_manifest.csv", index=False)

decoder_reliability = {
    "oracle_requested": CFG.expected_test_samples,
    "oracle_decoded_successfully": len(oracle_manifest),
    "end2end_requested": CFG.expected_test_samples,
    "end2end_decoded_successfully": len(end2end_manifest),
    "invalid_token_count": 0,
    "nan_count": int((~oracle_manifest.get("finite", pd.Series(dtype=bool))).sum()) +
                 int((~end2end_manifest.get("finite", pd.Series(dtype=bool))).sum()),
    "inf_count": 0,
    "shape_failure_count": len(oracle_failures) + len(end2end_failures),
    "wrong_feature_dimension_count": 0,
    "oracle_failures": oracle_failures,
    "end2end_failures": end2end_failures,
}
write_json(DIAGNOSTICS / "decoder_reliability.json", decoder_reliability)
if oracle_failures or end2end_failures:
    raise RuntimeError(
        f"Full decoding failed: oracle={len(oracle_failures)}, end-to-end={len(end2end_failures)}. "
        "Inspect diagnostics/decoder_reliability.json."
    )
if len(oracle_manifest) != CFG.expected_test_samples or len(end2end_manifest) != CFG.expected_test_samples:
    raise RuntimeError("Decoded coverage does not equal the required 1243 samples")
print("Full oracle and end-to-end decoding: PASS")

# %% [markdown]
# ## 5B.9–5B.16 — Duration, feature, normalized, velocity, and stability metrics
#
# Prefix metrics preserve the original time axes and are sensitive to temporal drift. Time-normalized
# metrics linearly resample only along time and therefore remove duration error; duration is reported
# separately. MPJPE and joint metrics are deliberately not calculated.

# %%
def safe_flat_correlation(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2: return float("nan")
    x_centered = x - x.mean(); y_centered = y - y.mean()
    denominator = float(np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)))
    if denominator <= 0 or not np.isfinite(denominator): return float("nan")
    return float(np.dot(x_centered, y_centered) / denominator)


def spearman(first: pd.Series, second: pd.Series) -> float:
    frame = pd.DataFrame({"first": pd.to_numeric(first, errors="coerce"),
                          "second": pd.to_numeric(second, errors="coerce")}).dropna()
    if len(frame) < 2 or frame["first"].nunique() < 2 or frame["second"].nunique() < 2:
        return float("nan")
    return float(frame["first"].rank(method="average").corr(frame["second"].rank(method="average")))


def linear_resample_time(motion: np.ndarray, target_frames: int) -> np.ndarray:
    if target_frames <= 0: raise ValueError("target_frames must be positive")
    if motion.shape[0] == target_frames: return motion.astype(np.float32, copy=True)
    if motion.shape[0] == 1: return np.repeat(motion.astype(np.float32), target_frames, axis=0)
    tensor = torch.from_numpy(np.asarray(motion, dtype=np.float32).T).unsqueeze(0)
    resized = F.interpolate(tensor, size=target_frames, mode="linear", align_corners=True)
    return resized.squeeze(0).T.numpy().astype(np.float32, copy=False)


def normalized_active(motion: np.ndarray) -> np.ndarray:
    return ((motion[:, active_dimensions] - motion_mean[active_dimensions]) /
            motion_std[active_dimensions]).astype(np.float32, copy=False)


def dynamic_statistics(motion: np.ndarray) -> tuple[float, float]:
    z_motion = normalized_active(motion)
    feature_variance = float(np.mean(np.var(z_motion, axis=0, dtype=np.float64)))
    if len(z_motion) < 2: return 0.0, feature_variance
    velocity = np.diff(z_motion, axis=0)
    velocity_magnitude = np.sqrt(np.mean(np.square(velocity, dtype=np.float64), axis=1))
    return float(np.mean(velocity_magnitude)), feature_variance


def extreme_statistics(motion: np.ndarray) -> tuple[float, float, float]:
    z_motion = normalized_active(motion)
    return float(motion.min()), float(motion.max()), float(np.max(np.abs(z_motion)))


gt_dynamic_rows, gt_extreme_rows = [], []
feature_sum = np.zeros(MOTION_DIM, dtype=np.float64)
feature_square_sum = np.zeros(MOTION_DIM, dtype=np.float64)
feature_frame_count = 0
for number, sample_id in enumerate(all_sample_ids, 1):
    gt = load_ground_truth(sample_id)
    velocity_stat, variance_stat = dynamic_statistics(gt)
    minimum, maximum, max_normalized = extreme_statistics(gt)
    gt_dynamic_rows.append({"sample_id": sample_id, "gt_velocity_magnitude": velocity_stat,
                            "gt_feature_temporal_variance": variance_stat})
    gt_extreme_rows.append({"sample_id": sample_id, "gt_minimum_feature_value": minimum,
                            "gt_maximum_feature_value": maximum,
                            "gt_maximum_normalized_absolute_value": max_normalized})
    feature_sum += gt.sum(axis=0, dtype=np.float64)
    feature_square_sum += np.square(gt, dtype=np.float64).sum(axis=0)
    feature_frame_count += gt.shape[0]
    if number % 250 == 0: print(f"Ground-truth statistics: {number}/{len(all_sample_ids)}")

gt_dynamic = pd.DataFrame(gt_dynamic_rows)
gt_extreme = pd.DataFrame(gt_extreme_rows)
static_velocity_threshold = float(np.percentile(gt_dynamic["gt_velocity_magnitude"], 1.0))
static_variance_threshold = float(np.percentile(gt_dynamic["gt_feature_temporal_variance"], 1.0))
extreme_min_threshold = float(np.percentile(gt_extreme["gt_minimum_feature_value"], 0.1))
extreme_max_threshold = float(np.percentile(gt_extreme["gt_maximum_feature_value"], 99.9))
extreme_z_threshold = float(np.percentile(gt_extreme["gt_maximum_normalized_absolute_value"], 99.9))
feature_mean_global = feature_sum / max(feature_frame_count, 1)
feature_variance_global = feature_square_sum / max(feature_frame_count, 1) - np.square(feature_mean_global)
feature_variance_global = np.maximum(feature_variance_global, 0.0)
high_variance_features = np.flatnonzero(active_dimensions)[
    np.argsort(feature_variance_global[active_dimensions])[-CFG.trajectory_feature_count:][::-1]
].astype(int).tolist()

write_json(DIAGNOSTICS / "empirical_motion_thresholds.json", {
    "basis": "ground-truth supported test Motion-Features distribution",
    "static_velocity_threshold_gt_percentile_1": static_velocity_threshold,
    "static_variance_threshold_gt_percentile_1": static_variance_threshold,
    "extreme_minimum_threshold_gt_sample_min_percentile_0_1": extreme_min_threshold,
    "extreme_maximum_threshold_gt_sample_max_percentile_99_9": extreme_max_threshold,
    "extreme_normalized_magnitude_threshold_gt_percentile_99_9": extreme_z_threshold,
    "high_variance_feature_indices": high_variance_features,
})


def calculate_sample_metrics(sample_id: str, predicted: np.ndarray, ground_truth: np.ndarray) -> dict[str, Any]:
    gt_frames, pred_frames = int(ground_truth.shape[0]), int(predicted.shape[0])
    overlap = min(gt_frames, pred_frames)
    prefix_pred, prefix_gt = predicted[:overlap], ground_truth[:overlap]
    prefix_difference = prefix_pred.astype(np.float64) - prefix_gt.astype(np.float64)
    prefix_mse = float(np.mean(np.square(prefix_difference)))

    resampled = linear_resample_time(predicted, gt_frames)
    resampled_difference = resampled.astype(np.float64) - ground_truth.astype(np.float64)
    time_mse = float(np.mean(np.square(resampled_difference)))

    z_pred = normalized_active(resampled)
    z_gt = normalized_active(ground_truth)
    z_difference = z_pred.astype(np.float64) - z_gt.astype(np.float64)
    normalized_mse = float(np.mean(np.square(z_difference)))

    if gt_frames >= 2:
        velocity_difference = np.diff(z_pred, axis=0).astype(np.float64) - np.diff(z_gt, axis=0).astype(np.float64)
        velocity_mse = float(np.mean(np.square(velocity_difference)))
        velocity_mae = float(np.mean(np.abs(velocity_difference)))
    else:
        velocity_mse = velocity_mae = float("nan")
    if gt_frames >= 3:
        acceleration_difference = (np.diff(z_pred, n=2, axis=0).astype(np.float64) -
                                   np.diff(z_gt, n=2, axis=0).astype(np.float64))
        acceleration_mae = float(np.mean(np.abs(acceleration_difference)))
    else:
        acceleration_mae = float("nan")

    predicted_velocity, predicted_variance = dynamic_statistics(predicted)
    minimum, maximum, max_normalized = extreme_statistics(predicted)
    frame_error = pred_frames - gt_frames
    return {
        "sample_id": sample_id,
        "gt_frames": gt_frames,
        "pred_frames": pred_frames,
        "frame_error": frame_error,
        "absolute_frame_error": abs(frame_error),
        "relative_frame_error": abs(frame_error) / max(gt_frames, 1),
        "predicted_to_gt_frame_ratio": pred_frames / max(gt_frames, 1),
        "raw_prefix_MAE": float(np.mean(np.abs(prefix_difference))),
        "raw_prefix_MSE": prefix_mse,
        "raw_prefix_RMSE": float(np.sqrt(prefix_mse)),
        "flattened_prefix_correlation": safe_flat_correlation(prefix_pred, prefix_gt),
        "time_normalized_raw_MAE": float(np.mean(np.abs(resampled_difference))),
        "time_normalized_raw_MSE": time_mse,
        "time_normalized_RMSE": float(np.sqrt(time_mse)),
        "time_normalized_correlation": safe_flat_correlation(resampled, ground_truth),
        "normalized_feature_MAE": float(np.mean(np.abs(z_difference))),
        "normalized_feature_MSE": normalized_mse,
        "normalized_feature_RMSE": float(np.sqrt(normalized_mse)),
        "velocity_MAE": velocity_mae,
        "velocity_MSE": velocity_mse,
        "acceleration_MAE": acceleration_mae,
        "mean_velocity_magnitude": predicted_velocity,
        "feature_temporal_variance": predicted_variance,
        "minimum_feature_value": minimum,
        "maximum_feature_value": maximum,
        "maximum_normalized_absolute_feature_value": max_normalized,
    }


def evaluate_decoded_variant(label: str, directory: Path) -> pd.DataFrame:
    rows = []
    for number, sample_id in enumerate(all_sample_ids, 1):
        predicted = np.load(directory / f"{sample_id}.npy", allow_pickle=False)
        ground_truth = load_ground_truth(sample_id)
        rows.append(calculate_sample_metrics(sample_id, predicted, ground_truth))
        if number % 100 == 0 or number == len(all_sample_ids):
            print(f"{label} metrics: {number}/{len(all_sample_ids)}")
    return pd.DataFrame(rows)


oracle_metrics = evaluate_decoded_variant("oracle", ORACLE_DECODED)
end2end_metrics = evaluate_decoded_variant("end-to-end", END2END_DECODED)
oracle_metrics.to_csv(RESULTS / "oracle_motion_metrics.csv", index=False)
end2end_metrics.to_csv(RESULTS / "end2end_motion_metrics.csv", index=False)


def summarize_metrics(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    mean_columns = [
        "absolute_frame_error", "frame_error", "relative_frame_error", "predicted_to_gt_frame_ratio",
        "raw_prefix_MAE", "raw_prefix_MSE", "raw_prefix_RMSE", "flattened_prefix_correlation",
        "time_normalized_raw_MAE", "time_normalized_raw_MSE", "time_normalized_RMSE",
        "time_normalized_correlation", "normalized_feature_MAE", "normalized_feature_MSE",
        "normalized_feature_RMSE", "velocity_MAE", "velocity_MSE", "acceleration_MAE",
    ]
    summary = {"label": label, "samples": len(frame), "active_dimensions": int(active_dimensions.sum())}
    for column in mean_columns: summary[f"mean_{column}"] = float(pd.to_numeric(frame[column], errors="coerce").mean())
    summary.update({
        "duration_mae_frames": float(frame["absolute_frame_error"].mean()),
        "duration_rmse_frames": float(np.sqrt(np.mean(np.square(frame["frame_error"].astype(np.float64))))),
        "duration_mean_signed_error": float(frame["frame_error"].mean()),
        "duration_median_absolute_error": float(frame["absolute_frame_error"].median()),
        "duration_spearman": spearman(frame["gt_frames"], frame["pred_frames"]),
    })
    return summary


oracle_summary = summarize_metrics(oracle_metrics, "oracle")
end2end_summary = summarize_metrics(end2end_metrics, "end-to-end")
write_json(RESULTS / "oracle_motion_summary.json", oracle_summary)
write_json(RESULTS / "end2end_motion_summary.json", end2end_summary)

duration_metrics = pd.DataFrame([
    {
        "variant": label,
        "MAE": summary["duration_mae_frames"],
        "RMSE": summary["duration_rmse_frames"],
        "mean_signed_error": summary["duration_mean_signed_error"],
        "median_absolute_error": summary["duration_median_absolute_error"],
        "Spearman_correlation": summary["duration_spearman"],
    }
    for label, summary in (("oracle", oracle_summary), ("end-to-end", end2end_summary))
])
duration_metrics.to_csv(RESULTS / "duration_metrics.csv", index=False)


def save_length_scatter(frame: pd.DataFrame, path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(frame["gt_frames"], frame["pred_frames"], alpha=0.35, s=13)
    upper = float(max(frame["gt_frames"].max(), frame["pred_frames"].max()))
    axis.plot([0, upper], [0, upper], linestyle="--", color="black", linewidth=1)
    axis.set(xlabel="Ground-truth frames", ylabel="Generated frames", title=title)
    figure.tight_layout(); figure.savefig(path, dpi=170, bbox_inches="tight"); plt.close(figure)


save_length_scatter(oracle_metrics, FIGURES / "oracle_length_scatter.png", "Oracle duration")
save_length_scatter(end2end_metrics, FIGURES / "end2end_length_scatter.png", "End-to-end duration")

# %% [markdown]
# ## 5B.17–5B.20 — Oracle/end-to-end comparison and Phase-4 quality join

# %%
comparison_spec = [
    ("duration MAE", "duration_mae_frames", False),
    ("duration signed bias", "duration_mean_signed_error", False),
    ("prefix feature MAE", "mean_raw_prefix_MAE", False),
    ("prefix correlation", "mean_flattened_prefix_correlation", True),
    ("time-normalized MAE", "mean_time_normalized_raw_MAE", False),
    ("time-normalized correlation", "mean_time_normalized_correlation", True),
    ("normalized feature MAE", "mean_normalized_feature_MAE", False),
    ("velocity MAE", "mean_velocity_MAE", False),
]
comparison_rows = []
for label, key, higher_is_better in comparison_spec:
    oracle_value, end_value = float(oracle_summary[key]), float(end2end_summary[key])
    degradation = oracle_value - end_value if higher_is_better else end_value - oracle_value
    comparison_rows.append({
        "Metric": label,
        "Oracle": oracle_value,
        "End-to-End": end_value,
        "End-to-End minus Oracle": end_value - oracle_value,
        "degradation_positive_is_worse": degradation,
    })
oracle_vs_end2end = pd.DataFrame(comparison_rows)
oracle_vs_end2end.to_csv(RESULTS / "oracle_vs_end2end_motion.csv", index=False)

figure, axes = plt.subplots(2, 4, figsize=(17, 8))
for axis, row in zip(axes.flat, comparison_rows):
    values = [row["Oracle"], row["End-to-End"]]
    bars = axis.bar(["Oracle", "End-to-End"], values, color=["#3b82f6", "#f97316"])
    axis.set_title(row["Metric"], fontsize=10)
    axis.tick_params(axis="x", labelrotation=15)
    for bar, value in zip(bars, values): axis.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.4g}", ha="center", va="bottom", fontsize=8)
figure.suptitle("Oracle vs end-to-end continuous-motion metrics")
figure.tight_layout(); figure.savefig(FIGURES / "oracle_vs_end2end_metrics.png", dpi=170, bbox_inches="tight"); plt.close(figure)


def load_prediction_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"sample_id": str})
    frame["sample_id"] = frame["sample_id"].astype(str)
    if frame["sample_id"].duplicated().any(): raise RuntimeError(f"Duplicate IDs in {path}")
    return frame


token_motion_join_available = PHASE4_ORACLE_PREDICTIONS is not None and PHASE4_END2END_PREDICTIONS is not None
token_motion_correlations: list[dict[str, Any]] = []
token_vs_motion = None
if token_motion_join_available:
    oracle_token_frame = load_prediction_csv(PHASE4_ORACLE_PREDICTIONS)
    end_token_frame = load_prediction_csv(PHASE4_END2END_PREDICTIONS)
    oracle_selected = oracle_token_frame[[
        "sample_id", "aligned_token_accuracy", "mean_normalized_edit_distance",
        "generated_length", "reference_length", "length_error",
    ]].rename(columns={column: f"oracle_{column}" for column in oracle_token_frame.columns if column != "sample_id"})
    end_selected_columns = [
        "sample_id", "aligned_token_accuracy", "mean_normalized_edit_distance",
        "generated_length", "reference_length", "length_error",
    ]
    for optional in ("sentence", "ground_truth_gloss", "conditioning_gloss"):
        if optional in end_token_frame.columns: end_selected_columns.append(optional)
    end_selected = end_token_frame[end_selected_columns].rename(columns={
        column: f"end2end_{column}" for column in end_selected_columns if column != "sample_id"
    })
    oracle_motion_selected = oracle_metrics[[
        "sample_id", "frame_error", "time_normalized_raw_MAE", "normalized_feature_MAE", "velocity_MAE"
    ]].rename(columns={column: f"oracle_motion_{column}" for column in oracle_metrics.columns if column != "sample_id"})
    end_motion_selected = end2end_metrics[[
        "sample_id", "frame_error", "time_normalized_raw_MAE", "normalized_feature_MAE", "velocity_MAE"
    ]].rename(columns={column: f"end2end_motion_{column}" for column in end2end_metrics.columns if column != "sample_id"})
    token_vs_motion = oracle_selected.merge(end_selected, on="sample_id", validate="one_to_one").merge(
        oracle_motion_selected, on="sample_id", validate="one_to_one"
    ).merge(end_motion_selected, on="sample_id", validate="one_to_one")
    if PHASE3_TEST_PREDICTIONS is not None:
        phase3_frame = load_prediction_csv(PHASE3_TEST_PREDICTIONS)
        phase3_columns = [column for column in ("sample_id", "predicted_gloss", "lexical_token_f1") if column in phase3_frame]
        token_vs_motion = token_vs_motion.merge(phase3_frame[phase3_columns], on="sample_id", how="left", validate="one_to_one")
    if len(token_vs_motion) != CFG.expected_test_samples:
        raise RuntimeError(f"Phase-4 token-quality join coverage is {len(token_vs_motion)}, expected 1243")
    token_vs_motion.to_csv(RESULTS / "token_vs_motion_metrics.csv", index=False)

    correlation_pairs = [
        ("oracle token accuracy vs time-normalized motion error", "oracle_aligned_token_accuracy", "oracle_motion_time_normalized_raw_MAE"),
        ("end-to-end token accuracy vs time-normalized motion error", "end2end_aligned_token_accuracy", "end2end_motion_time_normalized_raw_MAE"),
        ("oracle edit distance vs motion error", "oracle_mean_normalized_edit_distance", "oracle_motion_time_normalized_raw_MAE"),
        ("end-to-end edit distance vs motion error", "end2end_mean_normalized_edit_distance", "end2end_motion_time_normalized_raw_MAE"),
        ("oracle token-length error vs frame-length error", "oracle_length_error", "oracle_motion_frame_error"),
        ("end-to-end token-length error vs frame-length error", "end2end_length_error", "end2end_motion_frame_error"),
    ]
    if "lexical_token_f1" in token_vs_motion:
        correlation_pairs.append(("Phase-3 gloss F1 vs end-to-end motion error", "lexical_token_f1", "end2end_motion_time_normalized_raw_MAE"))
    for label, first, second in correlation_pairs:
        token_motion_correlations.append({"analysis": label, "first_column": first, "second_column": second,
                                          "Spearman_correlation": spearman(token_vs_motion[first], token_vs_motion[second])})
    pd.DataFrame(token_motion_correlations).to_csv(RESULTS / "token_vs_motion_correlations.csv", index=False)
else:
    print("Phase-4 per-sample token metrics unavailable; token-vs-motion analysis marked unavailable.")

# %% [markdown]
# ## 5B.21–5B.24 — Collapse/extreme checks and qualitative feature trajectories

# %%
collapse_frames, extreme_frames = [], []
for variant, frame in (("oracle", oracle_metrics), ("end-to-end", end2end_metrics)):
    collapse = frame[["sample_id", "mean_velocity_magnitude", "feature_temporal_variance"]].copy()
    collapse.insert(0, "variant", variant)
    collapse = collapse.merge(gt_dynamic, on="sample_id", validate="one_to_one")
    collapse["velocity_below_gt_p01"] = collapse["mean_velocity_magnitude"] < static_velocity_threshold
    collapse["variance_below_gt_p01"] = collapse["feature_temporal_variance"] < static_variance_threshold
    collapse["static_motion"] = collapse["velocity_below_gt_p01"] | collapse["variance_below_gt_p01"]
    collapse_frames.append(collapse)

    extreme = frame[[
        "sample_id", "minimum_feature_value", "maximum_feature_value", "maximum_normalized_absolute_feature_value"
    ]].copy()
    extreme.insert(0, "variant", variant)
    extreme = extreme.merge(gt_extreme, on="sample_id", validate="one_to_one")
    extreme["below_gt_sample_min_p001"] = extreme["minimum_feature_value"] < extreme_min_threshold
    extreme["above_gt_sample_max_p999"] = extreme["maximum_feature_value"] > extreme_max_threshold
    extreme["normalized_magnitude_above_gt_p999"] = (
        extreme["maximum_normalized_absolute_feature_value"] > extreme_z_threshold
    )
    extreme["extreme_motion"] = extreme[[
        "below_gt_sample_min_p001", "above_gt_sample_max_p999", "normalized_magnitude_above_gt_p999"
    ]].any(axis=1)
    extreme_frames.append(extreme)

motion_collapse_flags = pd.concat(collapse_frames, ignore_index=True)
extreme_motion_flags = pd.concat(extreme_frames, ignore_index=True)
motion_collapse_flags.to_csv(DIAGNOSTICS / "motion_collapse_flags.csv", index=False)
extreme_motion_flags.to_csv(DIAGNOSTICS / "extreme_motion_flags.csv", index=False)

qualitative = oracle_metrics[["sample_id", "gt_frames", "pred_frames", "normalized_feature_MAE"]].rename(columns={
    "pred_frames": "oracle_frames", "normalized_feature_MAE": "oracle_normalized_MAE"
}).merge(
    end2end_metrics[["sample_id", "pred_frames", "normalized_feature_MAE"]].rename(columns={
        "pred_frames": "end2end_frames", "normalized_feature_MAE": "end2end_normalized_MAE"
    }), on="sample_id", validate="one_to_one"
).sort_values("end2end_normalized_MAE").reset_index(drop=True)

band_count = min(CFG.qualitative_per_band, max(len(qualitative) // 3, 1))
low_indices = np.arange(band_count)
middle_start = max((len(qualitative) - band_count) // 2, band_count)
middle_indices = np.arange(middle_start, min(middle_start + band_count, len(qualitative)))
high_indices = np.arange(max(len(qualitative) - band_count, 0), len(qualitative))
selected_parts = []
for label, indices in (("low-error", low_indices), ("medium-error", middle_indices), ("high-error", high_indices)):
    part = qualitative.iloc[indices].copy(); part.insert(0, "error_band", label); selected_parts.append(part)
qualitative_cases = pd.concat(selected_parts, ignore_index=True).drop_duplicates("sample_id")

if token_vs_motion is not None:
    metadata_candidates = [
        "end2end_sentence", "end2end_ground_truth_gloss", "end2end_conditioning_gloss",
        "oracle_aligned_token_accuracy", "end2end_aligned_token_accuracy",
    ]
    if "end2end_conditioning_gloss" not in token_vs_motion and "predicted_gloss" in token_vs_motion:
        metadata_candidates.append("predicted_gloss")
    metadata_columns = ["sample_id"] + [column for column in metadata_candidates if column in token_vs_motion]
    qualitative_cases = qualitative_cases.merge(token_vs_motion[metadata_columns], on="sample_id", how="left", validate="one_to_one")
elif PHASE1_TEST_SPLIT is not None:
    split = pd.read_csv(PHASE1_TEST_SPLIT, dtype=str)
    id_column = next((column for column in ("sample_id", "id") if column in split), None)
    if id_column is not None:
        split = split.rename(columns={id_column: "sample_id", "gloss": "ground_truth_gloss"})
        qualitative_cases = qualitative_cases.merge(split[[
            column for column in ("sample_id", "sentence", "ground_truth_gloss") if column in split
        ]], on="sample_id", how="left", validate="one_to_one")
    if PHASE3_TEST_PREDICTIONS is not None:
        phase3_frame = load_prediction_csv(PHASE3_TEST_PREDICTIONS)
        qualitative_cases = qualitative_cases.merge(phase3_frame[[
            column for column in ("sample_id", "predicted_gloss", "lexical_token_f1") if column in phase3_frame
        ]], on="sample_id", how="left", validate="one_to_one")

rename_metadata = {
    "end2end_sentence": "sentence",
    "end2end_ground_truth_gloss": "ground_truth_gloss",
    "end2end_conditioning_gloss": "phase3_predicted_gloss",
    "predicted_gloss": "phase3_predicted_gloss",
    "end2end_aligned_token_accuracy": "token_accuracy",
}
qualitative_cases = qualitative_cases.rename(columns={key: value for key, value in rename_metadata.items() if key in qualitative_cases})
for column in ("sentence", "ground_truth_gloss", "phase3_predicted_gloss", "token_accuracy"):
    if column not in qualitative_cases: qualitative_cases[column] = pd.NA
qualitative_columns = [
    "error_band", "sample_id", "sentence", "ground_truth_gloss", "phase3_predicted_gloss",
    "gt_frames", "oracle_frames", "end2end_frames", "oracle_normalized_MAE",
    "end2end_normalized_MAE", "token_accuracy",
]
qualitative_cases[qualitative_columns].to_csv(RESULTS / "qualitative_case_manifest.csv", index=False)

for sample_id in qualitative_cases["sample_id"].astype(str):
    gt = load_ground_truth(sample_id)
    oracle = np.load(ORACLE_DECODED / f"{sample_id}.npy", allow_pickle=False)
    end2end = np.load(END2END_DECODED / f"{sample_id}.npy", allow_pickle=False)
    figure, axes = plt.subplots(len(high_variance_features), 1, figsize=(11, 2.4 * len(high_variance_features)), sharex=False)
    axes = np.atleast_1d(axes)
    for axis, feature in zip(axes, high_variance_features):
        axis.plot(np.arange(len(gt)), gt[:, feature], label="ground truth", linewidth=1.4)
        axis.plot(np.arange(len(oracle)), oracle[:, feature], label="oracle", linewidth=1.0, alpha=0.85)
        axis.plot(np.arange(len(end2end)), end2end[:, feature], label="end-to-end", linewidth=1.0, alpha=0.85)
        axis.set_ylabel(f"feature {feature}")
        axis.grid(alpha=0.2)
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Frame index (untrimmed)")
    figure.suptitle(f"Diagnostic feature trajectories — {sample_id}")
    figure.tight_layout(); figure.savefig(FIGURES / f"feature_trajectory_{sample_id}.png", dpi=160, bbox_inches="tight"); plt.close(figure)

# %% [markdown]
# ## 5B.25–5B.28 — Reliability report, Phase-5C handoff, and completion gate

# %%
oracle_collapse_count = int(motion_collapse_flags.query("variant == 'oracle'")["static_motion"].sum())
end2end_collapse_count = int(motion_collapse_flags.query("variant == 'end-to-end'")["static_motion"].sum())
oracle_extreme_count = int(extreme_motion_flags.query("variant == 'oracle'")["extreme_motion"].sum())
end2end_extreme_count = int(extreme_motion_flags.query("variant == 'end-to-end'")["extreme_motion"].sum())

ready_for_phase5c = bool(
    decoder_reliability["oracle_decoded_successfully"] == CFG.expected_test_samples
    and decoder_reliability["end2end_decoded_successfully"] == CFG.expected_test_samples
    and decoder_reliability["invalid_token_count"] == 0
    and decoder_reliability["nan_count"] == 0
    and decoder_reliability["inf_count"] == 0
    and decoder_reliability["shape_failure_count"] == 0
    and (oracle_manifest["decoded_frames"] == TEMPORAL_EXPANSION * oracle_manifest["token_length"]).all()
    and (end2end_manifest["decoded_frames"] == TEMPORAL_EXPANSION * end2end_manifest["token_length"]).all()
)

phase5b_report = {
    "phase": "Phase 5B - Generated RVQ Tokens to Continuous Motion",
    "phase5b_complete": ready_for_phase5c,
    "decoder": {
        "verified_phase5a_decoder": True,
        "candidate": "Candidate B",
        "strict_checkpoint_load": True,
        "rvq_levels": RVQ_LEVELS,
        "codebook_size": CODEBOOK_SIZE,
        "latent_dim": LATENT_DIM,
        "motion_dim": MOTION_DIM,
        "temporal_upsampling": TEMPORAL_EXPANSION,
    },
    "normalization": {
        "type": "empirically recovered per-feature z-score",
        "official_statistics_confirmed": False,
        "active_dimensions": int(active_dimensions.sum()),
        "near_zero_dimensions": int((~active_dimensions).sum()),
    },
    "coverage": {
        "oracle_requested": CFG.expected_test_samples,
        "oracle_decoded": len(oracle_manifest),
        "end2end_requested": CFG.expected_test_samples,
        "end2end_decoded": len(end2end_manifest),
    },
    "oracle": {
        "duration_mae_frames": oracle_summary["duration_mae_frames"],
        "duration_mean_signed_error": oracle_summary["duration_mean_signed_error"],
        "prefix_raw_mae": oracle_summary["mean_raw_prefix_MAE"],
        "time_normalized_raw_mae": oracle_summary["mean_time_normalized_raw_MAE"],
        "normalized_feature_mae": oracle_summary["mean_normalized_feature_MAE"],
        "velocity_mae": oracle_summary["mean_velocity_MAE"],
        "mean_correlation": oracle_summary["mean_time_normalized_correlation"],
    },
    "end2end": {
        "duration_mae_frames": end2end_summary["duration_mae_frames"],
        "duration_mean_signed_error": end2end_summary["duration_mean_signed_error"],
        "prefix_raw_mae": end2end_summary["mean_raw_prefix_MAE"],
        "time_normalized_raw_mae": end2end_summary["mean_time_normalized_raw_MAE"],
        "normalized_feature_mae": end2end_summary["mean_normalized_feature_MAE"],
        "velocity_mae": end2end_summary["mean_velocity_MAE"],
        "mean_correlation": end2end_summary["mean_time_normalized_correlation"],
    },
    "oracle_vs_end2end": comparison_rows,
    "token_vs_motion_analysis_available": token_motion_join_available,
    "token_vs_motion_correlations": token_motion_correlations,
    "decoder_failures": {
        "invalid_tokens": decoder_reliability["invalid_token_count"],
        "nan": decoder_reliability["nan_count"],
        "inf": decoder_reliability["inf_count"],
        "shape_failures": decoder_reliability["shape_failure_count"],
    },
    "motion_collapse_samples": {"oracle": oracle_collapse_count, "end2end": end2end_collapse_count},
    "extreme_motion_samples": {"oracle": oracle_extreme_count, "end2end": end2end_extreme_count},
    "ready_for_phase5c": ready_for_phase5c,
}
write_json(OUTPUT / "phase5b_report.json", phase5b_report)

readme = f"""PHASE 5C INPUT CONTRACT — MOTION-S / SIGNVRSE

1. Verified RVQ decoder architecture
Candidate B strict-loaded from the official rvq_vae_best.pth:
ConvTranspose1d 256->512 (k=3,s=2,p=1,op=1), ReLU, BatchNorm1d;
ConvTranspose1d 512->512 (k=3,s=2,p=1,op=1), ReLU, BatchNorm1d;
ConvTranspose1d 512->512 (k=3,s=1,p=1,op=0), ReLU, BatchNorm1d;
ConvTranspose1d 512->668 (k=3,s=1,p=1,op=0), no final activation.

2. Token shape
[6,T], integer IDs 0..511. One selected [256]-vector from each codebook is summed.

3. Continuous motion output
[4*T,668], float32. Generated durations are untrimmed.

4. Normalization
Decoder output is inverse transformed with empirically recovered Motion-S feature normalization:
motion = normalized_motion * safe_std + mean. For std < 1e-6, safe_std=1.0.
These are not claimed to be official provider statistics.

5. Decoded locations
Oracle: {ORACLE_DECODED}
End-to-end: {END2END_DECODED}

6. Ground-truth Motion-Features
{MOTION_FEATURES}

7. Qualitative sample IDs
{', '.join(qualitative_cases['sample_id'].astype(str).tolist())}

8. Quantitative findings
Oracle duration MAE: {oracle_summary['duration_mae_frames']:.6f} frames
End-to-end duration MAE: {end2end_summary['duration_mae_frames']:.6f} frames
Oracle normalized feature MAE: {oracle_summary['mean_normalized_feature_MAE']:.6f}
End-to-end normalized feature MAE: {end2end_summary['mean_normalized_feature_MAE']:.6f}
Oracle velocity MAE: {oracle_summary['mean_velocity_MAE']:.6f}
End-to-end velocity MAE: {end2end_summary['mean_velocity_MAE']:.6f}

9. Collapse / instability findings
Motion-collapse flags — oracle: {oracle_collapse_count}, end-to-end: {end2end_collapse_count}
Extreme-motion flags — oracle: {oracle_extreme_count}, end-to-end: {end2end_extreme_count}
Thresholds are empirical ground-truth test-distribution percentiles; inspect diagnostics CSVs.

10. Explicit unresolved item
The semantic layout / joint conversion of the 668-dimensional Motion-S representation must be
verified before skeleton/BVH visualization.
"""
(OUTPUT / "README_phase5c.txt").write_text(readme, encoding="utf-8")

inventory_rows = []
for path in sorted(OUTPUT.rglob("*")):
    if path.is_file() and path.name != "artifact_inventory.csv":
        inventory_rows.append({
            "relative_path": str(path.relative_to(OUTPUT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
pd.DataFrame(inventory_rows).to_csv(OUTPUT / "artifact_inventory.csv", index=False)

if not ready_for_phase5c:
    raise RuntimeError("PHASE 5B COMPLETION GATE FAILED. Inspect phase5b_report.json and diagnostics.")

print("\n============================================================")
print("PHASE 5B COMPLETE")
print("============================================================")
print("Verified decoder: PASS")
print(f"Oracle samples decoded: {len(oracle_manifest)} / {CFG.expected_test_samples}")
print(f"End-to-end samples decoded: {len(end2end_manifest)} / {CFG.expected_test_samples}")
print("Invalid token sequences:", decoder_reliability["invalid_token_count"])
print("NaN/Inf failures:", decoder_reliability["nan_count"] + decoder_reliability["inf_count"])
print("Motion feature dimension:", MOTION_DIM)
print("Temporal expansion: 4x")
print("\nORACLE:")
print("Duration MAE:", oracle_summary["duration_mae_frames"])
print("Mean signed duration error:", oracle_summary["duration_mean_signed_error"])
print("Time-normalized raw MAE:", oracle_summary["mean_time_normalized_raw_MAE"])
print("Normalized feature MAE:", oracle_summary["mean_normalized_feature_MAE"])
print("Velocity MAE:", oracle_summary["mean_velocity_MAE"])
print("\nEND-TO-END:")
print("Duration MAE:", end2end_summary["duration_mae_frames"])
print("Mean signed duration error:", end2end_summary["duration_mean_signed_error"])
print("Time-normalized raw MAE:", end2end_summary["mean_time_normalized_raw_MAE"])
print("Normalized feature MAE:", end2end_summary["mean_normalized_feature_MAE"])
print("Velocity MAE:", end2end_summary["mean_velocity_MAE"])
degradation_row = oracle_vs_end2end.loc[oracle_vs_end2end["Metric"] == "normalized feature MAE"].iloc[0]
print("\nOracle -> End-to-End degradation:", degradation_row["degradation_positive_is_worse"])
print("Motion-collapse flags:", {"oracle": oracle_collapse_count, "end2end": end2end_collapse_count})
print("Extreme-motion flags:", {"oracle": oracle_extreme_count, "end2end": end2end_extreme_count})
print("READY FOR PHASE 5C: YES")
print("============================================================")
print("STOP. Do not begin Phase 5C automatically.")

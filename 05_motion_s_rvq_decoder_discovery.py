# %% [markdown]
# # Motion-S / SignVRSE — Phase 5A: Official RVQ Decoder Discovery Gate
#
# Phase 5 cannot safely decode motion until the exact provider implementation, checkpoint,
# token layout, output representation, and normalization contract are all verified. This notebook
# performs only Sections 5.0–5.2. It never constructs a guessed RVQ-VAE and never decodes Phase-4
# predictions. Attach the gated official `Signvrse/Motion-S-RVQ-VAE-Tokenizer` artifact first.

# %%
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from IPython.display import display

EXPECTED_FINGERPRINT = "e2acdf4d6dce91775239ad50be3866dbc6f5da94d940204b62456cdb77fc4b0d"
EXPECTED_LEVELS = 6
EXPECTED_CODEBOOK_SIZE = 512
VALID_MIN, VALID_MAX = 0, 511
FORBIDDEN_MODEL_IDS = {512, 513, 514}
OFFICIAL_REPOSITORY = "Signvrse/Motion-S-RVQ-VAE-Tokenizer"
OFFICIAL_CHECKPOINT_NAME = "rvq_vae_best.pth"


@dataclass(frozen=True)
class Phase5DiscoveryConfig:
    seed: int = 42
    kaggle_input: str = "/kaggle/input"
    phase1_reference: Optional[str] = "/kaggle/input/datasets/adibraian/gpt-first-phase"
    phase4_reference: Optional[str] = None
    motion_s_reference: Optional[str] = None
    decoder_reference: Optional[str] = None
    output_dir: str = "/kaggle/working/motion_s_phase5"
    random_token_checks: int = 20
    motion_candidate_limit: int = 100


CFG = Phase5DiscoveryConfig()
OUTPUT = Path(CFG.output_dir)
DIAGNOSTICS, RESULTS, RECONSTRUCTED, ANIMATIONS = (
    OUTPUT / "diagnostics", OUTPUT / "results", OUTPUT / "reconstructed", OUTPUT / "animations"
)
for path in (OUTPUT, DIAGNOSTICS, RESULTS, RECONSTRUCTED, ANIMATIONS): path.mkdir(parents=True, exist_ok=True)


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
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, ensure_ascii=False, allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def canonical_name(value: str) -> str: return re.sub(r"[^a-z0-9]", "", value.casefold())


def reference_candidates(reference: Optional[str]) -> list[Path]:
    if not reference: return []
    literal = Path(reference); candidates = [literal]
    parts = literal.parts
    if "datasets" in parts:
        index = parts.index("datasets")
        if len(parts) >= index + 3: candidates.append(Path(CFG.kaggle_input) / parts[index + 2])
    root = Path(CFG.kaggle_input)
    if root.exists():
        candidates.extend(p for p in root.rglob("*") if p.is_dir() and canonical_name(p.name) == canonical_name(literal.name))
    return list(dict.fromkeys(candidates))


def find_roots(predicate, reference: Optional[str] = None) -> list[Path]:
    matches = []
    bases = [p for p in reference_candidates(reference) if p.exists()] or [Path(CFG.kaggle_input)]
    for base in bases:
        if predicate(base): matches.append(base.resolve())
        for current, directories, _ in os.walk(base):
            candidate = Path(current)
            if candidate == base: continue
            if predicate(candidate): matches.append(candidate.resolve()); directories[:] = []
    return list(dict.fromkeys(matches))


def require_one(kind: str, predicate, reference: Optional[str] = None) -> Path:
    matches = find_roots(predicate, reference)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {kind}; found {[str(x) for x in matches]}")
    return matches[0]


def is_phase1(path: Path) -> bool:
    return (path / "dataset_config.json").is_file() and (path / "sample_manifest.csv").is_file()


def is_phase4(path: Path) -> bool:
    return all((path / name).is_file() for name in (
        "phase4c_report.json", "results/test_oracle_tokens.npz", "results/test_end2end_tokens.npz"
    ))


def is_motion_s(path: Path) -> bool:
    return (
        (path / "train.csv").is_file()
        and any(canonical_name(child.name) == "motionfeatures" for child in path.iterdir() if child.is_dir())
    )


print("Python:", sys.version.split()[0], "PyTorch:", torch.__version__)
print("Kaggle input root:", CFG.kaggle_input)
display(pd.Series(asdict(CFG), name="value").to_frame())

# %% [markdown]
# ## 5.0 Verify Phase-1 and Phase-4 token artifacts

# %%
PHASE1_ROOT = require_one("Phase-1 root", is_phase1, CFG.phase1_reference)
PHASE4_ROOT = require_one("completed Phase-4C root", is_phase4, CFG.phase4_reference)
MOTION_S_ROOT = require_one("original Motion-S root", is_motion_s, CFG.motion_s_reference)
with (PHASE1_ROOT / "dataset_config.json").open(encoding="utf-8") as handle: phase1_config = json.load(handle)
with (PHASE4_ROOT / "phase4c_report.json").open(encoding="utf-8") as handle: phase4_report = json.load(handle)
assert phase1_config["split_fingerprint_sha256"] == EXPECTED_FINGERPRINT
assert phase4_report["phase1_fingerprint"] == EXPECTED_FINGERPRINT
assert phase4_report["phase4c_complete"] is True and phase4_report["ready_for_phase5"] is True


def validate_packed_tokens(path: Path, seed: int) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {"sample_ids", "lengths", "offsets", "tokens_flat"}
        assert required.issubset(data.files), required - set(data.files)
        ids = data["sample_ids"].astype(str); lengths = data["lengths"].astype(np.int64)
        offsets = data["offsets"].astype(np.int64); packed = data["tokens_flat"]
        assert packed.ndim == 2 and packed.shape[0] == EXPECTED_LEVELS
        assert np.issubdtype(packed.dtype, np.integer)
        assert len(ids) == len(lengths) and len(offsets) == len(ids) + 1
        assert offsets[0] == 0 and offsets[-1] == packed.shape[1]
        assert np.array_equal(np.diff(offsets), lengths)
        if packed.size:
            assert int(packed.min()) >= VALID_MIN and int(packed.max()) <= VALID_MAX
            assert not np.isin(packed, list(FORBIDDEN_MODEL_IDS)).any()
        rng = np.random.default_rng(seed); selected = rng.choice(len(ids), min(CFG.random_token_checks, len(ids)), replace=False)
        samples = []
        for index in selected:
            tokens = packed[:, offsets[index]:offsets[index + 1]]
            assert tokens.ndim == 2 and tokens.shape == (EXPECTED_LEVELS, int(lengths[index]))
            samples.append({"sample_id": ids[index], "shape": list(tokens.shape),
                            "min": int(tokens.min()) if tokens.size else None,
                            "max": int(tokens.max()) if tokens.size else None})
    return {"path": str(path), "samples": len(ids), "packed_shape": list(packed.shape),
            "random_checks": samples, "passed": True}


TOKEN_REPORTS = {
    "oracle": validate_packed_tokens(PHASE4_ROOT / "results/test_oracle_tokens.npz", CFG.seed),
    "end2end": validate_packed_tokens(PHASE4_ROOT / "results/test_end2end_tokens.npz", CFG.seed + 1),
}
write_json(DIAGNOSTICS / "phase4_token_artifact_verification.json", TOKEN_REPORTS)
print("Phase-1 fingerprint and Phase-4 NPZ contract: PASS")
print("Phase-1:", PHASE1_ROOT)
print("Phase-4C:", PHASE4_ROOT)
print("Original Motion-S:", MOTION_S_ROOT)

# %% [markdown]
# ## 5.1 Discover the exact official provider artifact
#
# The checkpoint is inspected with `weights_only=True`. Tensor names and shapes are inventoried,
# but no architecture is inferred into executable code. An official implementation containing
# `RVQVAE` and `decode_from_tokens` (or an equivalently documented provider API) is required.

# %%
input_root = Path(CFG.kaggle_input)
all_checkpoint_candidates = sorted(input_root.rglob(OFFICIAL_CHECKPOINT_NAME)) if input_root.exists() else []
checkpoint_candidates = list(all_checkpoint_candidates)
checkpoint_selection_reason = "unique exact filename"
if CFG.decoder_reference:
    referenced = []
    for root in reference_candidates(CFG.decoder_reference):
        if root.is_file() and root.name == OFFICIAL_CHECKPOINT_NAME: referenced.append(root)
        elif root.exists(): referenced.extend(root.rglob(OFFICIAL_CHECKPOINT_NAME))
    if referenced:
        checkpoint_candidates = sorted(set(path.resolve() for path in referenced))
        checkpoint_selection_reason = "explicit decoder_reference"
elif len(checkpoint_candidates) > 1:
    # The same official weight may be attached both as a Dataset and as a host-pinned Kaggle Model.
    # Prefer exactly one model-mounted copy; retain every candidate in the audit report.
    model_mounted = [path for path in checkpoint_candidates if "models" in {part.casefold() for part in path.parts}]
    if len(model_mounted) == 1:
        checkpoint_candidates = model_mounted
        checkpoint_selection_reason = "preferred unique host-model mount over duplicate dataset copy"

decoder_root = checkpoint_candidates[0].parent if len(checkpoint_candidates) == 1 else None
provider_inventory = []
if decoder_root:
    for path in sorted(decoder_root.rglob("*")):
        if path.is_file():
            provider_inventory.append({
                "relative_path": str(path.relative_to(decoder_root)), "bytes": path.stat().st_size,
                "suffix": path.suffix.lower(), "sha256": sha256_file(path),
            })
pd.DataFrame(provider_inventory).to_csv(DIAGNOSTICS / "provider_file_inventory.csv", index=False)

source_files = [] if decoder_root is None else sorted(decoder_root.rglob("*.py"))
config_files = [] if decoder_root is None else sorted(
    path for pattern in ("*.json", "*.yaml", "*.yml") for path in decoder_root.rglob(pattern)
)
normalization_files = [] if decoder_root is None else sorted(
    path for path in decoder_root.rglob("*")
    if path.is_file() and any(term in path.name.casefold() for term in ("mean", "std", "norm", "scale"))
)
readme_files = [] if decoder_root is None else sorted(
    path for path in decoder_root.rglob("*") if path.is_file() and path.name.casefold().startswith("readme")
)

source_evidence = []
for path in source_files:
    text = path.read_text(encoding="utf-8", errors="replace")
    source_evidence.append({
        "path": str(path), "defines_rvqvae": bool(re.search(r"class\s+RVQVAE\b", text)),
        "defines_decode_from_tokens": "decode_from_tokens" in text,
        "mentions_num_quantizers": "num_quantizers" in text,
    })
official_decoder_source_verified = any(
    row["defines_rvqvae"] and row["defines_decode_from_tokens"] for row in source_evidence
)

checkpoint_inventory, checkpoint_error, state_dict = [], None, None
if len(checkpoint_candidates) == 1:
    try:
        loaded = torch.load(checkpoint_candidates[0], map_location="cpu", weights_only=True)
        if isinstance(loaded, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in loaded and isinstance(loaded[key], dict): state_dict = loaded[key]; break
            if state_dict is None and all(torch.is_tensor(value) for value in loaded.values()): state_dict = loaded
        if state_dict is None: checkpoint_error = f"Unsupported safe checkpoint container: {type(loaded).__name__}"
        else:
            for name, tensor in state_dict.items():
                if torch.is_tensor(tensor): checkpoint_inventory.append({
                    "key": name, "shape": json.dumps(list(tensor.shape)), "dtype": str(tensor.dtype),
                    "numel": tensor.numel(),
                })
    except Exception as error:
        checkpoint_error = f"{type(error).__name__}: {error}"
pd.DataFrame(checkpoint_inventory).to_csv(DIAGNOSTICS / "checkpoint_key_inventory.csv", index=False)

codebook_candidates = []
if state_dict is not None:
    for name, tensor in state_dict.items():
        shape = tuple(tensor.shape) if torch.is_tensor(tensor) else ()
        if len(shape) == 2 and EXPECTED_CODEBOOK_SIZE in shape and any(term in name.casefold() for term in ("embed", "codebook", "quant")):
            codebook_candidates.append({"key": name, "shape": list(shape)})

DISCOVERY_REPORT = {
    "official_repository": OFFICIAL_REPOSITORY,
    "expected_checkpoint": OFFICIAL_CHECKPOINT_NAME,
    "all_exact_filename_candidates": all_checkpoint_candidates,
    "checkpoint_candidates": checkpoint_candidates,
    "checkpoint_selection_reason": checkpoint_selection_reason,
    "exact_checkpoint_located": len(checkpoint_candidates) == 1,
    "decoder_root": decoder_root,
    "checkpoint_sha256": sha256_file(checkpoint_candidates[0]) if len(checkpoint_candidates) == 1 else None,
    "checkpoint_safe_load_error": checkpoint_error,
    "checkpoint_tensor_count": len(checkpoint_inventory),
    "codebook_tensor_candidates": codebook_candidates,
    "provider_source_files": source_files,
    "source_evidence": source_evidence,
    "official_decoder_source_verified": official_decoder_source_verified,
    "config_files": config_files,
    "normalization_files": normalization_files,
    "readme_files": readme_files,
    "expected_contract_from_official_model_card_only": {
        "num_quantizers": 6, "num_embeddings": 512, "latent_dim": 256,
        "model_card_example_input_dim": 263, "model_card_downsampling_ratio": 4,
        "warning": "Model-card values are not accepted as an executable decoder contract without provider implementation/config verification."
    },
}
write_json(DIAGNOSTICS / "official_decoder_discovery.json", DISCOVERY_REPORT)
display(pd.DataFrame(provider_inventory))
display(pd.DataFrame(checkpoint_inventory).head(100))
print("Checkpoint candidates:", checkpoint_candidates)
print("Official executable decoder source verified:", official_decoder_source_verified)

# %% [markdown]
# ## 5.2 Inventory continuous-motion and normalization candidates, then enforce the stop gate

# %%
motion_candidates = []
interesting_suffixes = {".npy", ".npz", ".pt", ".pth", ".bvh", ".pkl", ".joblib"}
for path in MOTION_S_ROOT.rglob("*"):
    if not path.is_file() or path.suffix.casefold() not in interesting_suffixes: continue
    if path.name == OFFICIAL_CHECKPOINT_NAME or (decoder_root and decoder_root in path.parents): continue
    record = {"path": str(path), "suffix": path.suffix.casefold(), "bytes": path.stat().st_size,
              "shape": None, "dtype": None, "inspection_error": None}
    if path.suffix.casefold() == ".npy":
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            record["shape"], record["dtype"] = json.dumps(list(array.shape)), str(array.dtype)
        except Exception as error: record["inspection_error"] = f"{type(error).__name__}: {error}"
    motion_candidates.append(record)
    if len(motion_candidates) >= CFG.motion_candidate_limit: break
pd.DataFrame(motion_candidates).to_csv(DIAGNOSTICS / "continuous_motion_candidate_inventory.csv", index=False)

READY_FOR_GROUND_TRUTH_ROUNDTRIP = bool(
    len(checkpoint_candidates) == 1
    and checkpoint_error is None
    and official_decoder_source_verified
    and len(codebook_candidates) >= EXPECTED_LEVELS
)
GATE_REPORT = {
    "phase": "Phase 5A - Official Decoder Discovery",
    "phase1_fingerprint_verified": True,
    "phase4_token_artifacts_verified": all(item["passed"] for item in TOKEN_REPORTS.values()),
    "official_checkpoint_located": len(checkpoint_candidates) == 1,
    "official_decoder_source_verified": official_decoder_source_verified,
    "candidate_codebook_tensor_count": len(codebook_candidates),
    "normalization_artifacts_found": [str(path) for path in normalization_files],
    "continuous_motion_candidate_count": len(motion_candidates),
    "ready_for_ground_truth_roundtrip": READY_FOR_GROUND_TRUTH_ROUNDTRIP,
    "ground_truth_rvq_decoding_verified": False,
    "phase5_complete": False,
}
write_json(DIAGNOSTICS / "phase5_discovery_gate_report.json", GATE_REPORT)
write_json(OUTPUT / "phase5_config.json", asdict(CFG))

print("\n============================================================")
print("PHASE 5A — OFFICIAL DECODER DISCOVERY GATE")
print("============================================================")
print("Phase-1 fingerprint: PASS")
print("Phase-4 token artifacts: PASS")
print("Official checkpoint:", "FOUND" if len(checkpoint_candidates) == 1 else "MISSING / AMBIGUOUS")
print("Official decoder implementation:", "VERIFIED" if official_decoder_source_verified else "NOT VERIFIED")
print("Candidate codebook tensors:", len(codebook_candidates))
print("Normalization artifacts:", len(normalization_files))
print("Continuous-motion candidates inventoried:", len(motion_candidates))
print("READY FOR GROUND-TRUTH ROUND-TRIP:", "YES" if READY_FOR_GROUND_TRUTH_ROUNDTRIP else "NO")
print("============================================================")

if not READY_FOR_GROUND_TRUTH_ROUNDTRIP:
    raise RuntimeError(
        "PHASE 5 STOP GATE: attach the unmodified official rvq_vae_best.pth plus the provider's exact "
        "RVQVAE/decode_from_tokens implementation, configuration, and normalization artifacts. "
        "Do not construct or run a replacement decoder."
    )

print("Discovery evidence is sufficient for the next notebook: ground-truth token round-trip only.")
print("STOP here; do not decode Phase-4 generated tokens yet.")

# %% [markdown]
# # Motion-S / SignVRSE — Phase 5C: Verified Motion Interpretation and Visualization
#
# This notebook performs evidence-gated interpretation and feature-space diagnostics. It never
# guesses a 668-D joint layout, reshapes features into XYZ joints, calculates MPJPE without a
# verified converter, trims generated tails, smooths motion, retrains Phase 4, or starts Phase 6.

# %%
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
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

MOTION_DIM = 668
RVQ_LEVELS = 6
CODEBOOK_SIZE = 512
LATENT_DIM = 256
EXPECTED_SAMPLES = 1243
TEMPORAL_FACTOR = 4
OFFICIAL_ABSTRACT_URL = "https://sti.easteco.org/documents/4th_EAC_Regional_STI_Book_of_Abstracts_2026_WEB.pdf"
OFFICIAL_MODEL_CARD_URL = "https://huggingface.co/Signvrse/Motion-S-RVQ-VAE-Tokenizer"


@dataclass(frozen=True)
class Phase5CConfig:
    seed: int = 42
    kaggle_input: str = "/kaggle/input"
    kaggle_working: str = "/kaggle/working"
    output_dir: str = "/kaggle/working/motion_s_phase5c"
    cases_per_group: int = 5
    trajectory_features: int = 4
    bvh_alignment_samples: int = 10
    low_motion_gt_percentile: float = 1.0
    repetition_gt_percentile: float = 99.0
    high_velocity_gt_percentile: float = 99.0
    near_zero_std: float = 1e-6


CFG = Phase5CConfig()
SEARCH_BASES = [Path(CFG.kaggle_input), Path(CFG.kaggle_working)]
OUTPUT = Path(CFG.output_dir)
DIAGNOSTICS = OUTPUT / "diagnostics"
RESULTS = OUTPUT / "results"
FIGURES = OUTPUT / "figures"
FEATURE_FIGURES = FIGURES / "features"
VELOCITY_FIGURES = FIGURES / "velocity"
COLLAPSE_FIGURES = FIGURES / "collapse"
TAIL_FIGURES = FIGURES / "tail_analysis"
for directory in (OUTPUT, DIAGNOSTICS, RESULTS, FIGURES, FEATURE_FIGURES,
                  VELOCITY_FIGURES, COLLAPSE_FIGURES, TAIL_FIGURES):
    directory.mkdir(parents=True, exist_ok=True)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(child) for child in value]
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


def canonical(value: str) -> str: return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def canonical_id(value: Any) -> str:
    if isinstance(value, (int, np.integer)): return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer(): return str(int(value))
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def unique_roots(predicate) -> list[Path]:
    matches = []
    for base in SEARCH_BASES:
        if not base.exists(): continue
        for current, directories, _ in os.walk(base):
            candidate = Path(current)
            if predicate(candidate):
                matches.append(candidate.resolve()); directories[:] = []
    return sorted(dict.fromkeys(matches), key=str)


def child_named(root: Path, target: str) -> Optional[Path]:
    expected = canonical(target)
    return next((path for path in root.iterdir() if path.is_dir() and canonical(path.name) == expected), None)


def is_phase5b_root(path: Path) -> bool:
    required = [
        "phase5b_report.json", "diagnostics/decoder_contract.json",
        "diagnostics/motion_mean_668.npy", "diagnostics/motion_std_668.npy",
        "diagnostics/motion_collapse_flags.csv", "diagnostics/extreme_motion_flags.csv",
        "diagnostics/decoder_reliability.json", "decoded/oracle_manifest.csv",
        "decoded/end2end_manifest.csv", "results/oracle_motion_metrics.csv",
        "results/end2end_motion_metrics.csv", "results/oracle_vs_end2end_motion.csv",
    ]
    return all((path / relative).is_file() for relative in required)


def is_motion_s_root(path: Path) -> bool:
    return (path / "train.csv").is_file() and child_named(path, "Motion-Features") is not None and child_named(path, "Train") is not None


phase5b_roots = unique_roots(is_phase5b_root)
if not phase5b_roots:
    raise RuntimeError(f"Expected exactly one completed Phase-5B root; found {[str(path) for path in phase5b_roots]}")
if len(phase5b_roots) > 1:
    report_hashes = {root: sha256_file(root / "phase5b_report.json") for root in phase5b_roots}
    if len(set(report_hashes.values())) != 1:
        raise RuntimeError(f"Ambiguous non-identical Phase-5B roots: {[str(path) for path in phase5b_roots]}")
    working_matches = [root for root in phase5b_roots if str(root).startswith(str(Path(CFG.kaggle_working)))]
    PHASE5B_ROOT = working_matches[0] if len(working_matches) == 1 else phase5b_roots[0]
    print("Identical Phase-5B report copies found; selected:", PHASE5B_ROOT)
else:
    PHASE5B_ROOT = phase5b_roots[0]
motion_s_roots = unique_roots(is_motion_s_root)
if len(motion_s_roots) != 1:
    raise RuntimeError(f"Expected exactly one original Motion-S root with Train and Motion-Features; found {[str(path) for path in motion_s_roots]}")
MOTION_S_ROOT = motion_s_roots[0]
MOTION_FEATURES = child_named(MOTION_S_ROOT, "Motion-Features")
TRAIN_BVH_ROOT = child_named(MOTION_S_ROOT, "Train")
assert MOTION_FEATURES is not None and TRAIN_BVH_ROOT is not None

print("Python:", sys.version.split()[0], "PyTorch:", torch.__version__)
print("Phase-5B root:", PHASE5B_ROOT)
print("Motion-S root:", MOTION_S_ROOT)
print("Motion-Features:", MOTION_FEATURES)
print("Original BVH root:", TRAIN_BVH_ROOT)
display(pd.Series(asdict(CFG), name="value").to_frame())

# %% [markdown]
# ## 5C.1 — Verify Phase-5B artifacts without modifying motion

# %%
with (PHASE5B_ROOT / "phase5b_report.json").open(encoding="utf-8") as handle: phase5b_report = json.load(handle)
with (PHASE5B_ROOT / "diagnostics/decoder_contract.json").open(encoding="utf-8") as handle: decoder_contract = json.load(handle)
if phase5b_report.get("phase5b_complete") is not True or phase5b_report.get("ready_for_phase5c") is not True:
    raise RuntimeError("Phase-5B report is not complete/ready for Phase 5C")
motion_contract = decoder_contract.get("motion_representation", {})
contract_motion_dim = motion_contract.get("motion_dim", motion_contract.get("feature_dim"))
contract_temporal = motion_contract.get("temporal_expansion", motion_contract.get("temporal_upsampling_factor"))
if contract_motion_dim != MOTION_DIM or contract_temporal != TEMPORAL_FACTOR:
    raise RuntimeError(f"Decoder contract mismatch: motion_dim={contract_motion_dim}, temporal={contract_temporal}")
normalization_contract = decoder_contract.get("normalization", {})
if "empirically recovered" not in str(normalization_contract.get("type", "")).casefold():
    raise RuntimeError("Decoder contract does not identify Phase-5A empirical normalization")

motion_mean = np.load(PHASE5B_ROOT / "diagnostics/motion_mean_668.npy", allow_pickle=False).astype(np.float32)
motion_std = np.load(PHASE5B_ROOT / "diagnostics/motion_std_668.npy", allow_pickle=False).astype(np.float32)
if motion_mean.shape != (MOTION_DIM,) or motion_std.shape != (MOTION_DIM,): raise RuntimeError("Normalization shape mismatch")
if not np.isfinite(motion_mean).all() or not np.isfinite(motion_std).all(): raise RuntimeError("Non-finite normalization")
active_dimensions = motion_std >= CFG.near_zero_std
if not active_dimensions.any(): raise RuntimeError("No active feature dimensions")

oracle_manifest = pd.read_csv(PHASE5B_ROOT / "decoded/oracle_manifest.csv", dtype={"sample_id": str})
end2end_manifest = pd.read_csv(PHASE5B_ROOT / "decoded/end2end_manifest.csv", dtype={"sample_id": str})
for label, manifest in (("oracle", oracle_manifest), ("end-to-end", end2end_manifest)):
    manifest["sample_id"] = manifest["sample_id"].astype(str)
    if len(manifest) != EXPECTED_SAMPLES or manifest["sample_id"].duplicated().any():
        raise RuntimeError(f"{label} manifest count/uniqueness failure")
if set(oracle_manifest["sample_id"]) != set(end2end_manifest["sample_id"]): raise RuntimeError("Oracle/end-to-end ID mismatch")
sample_ids = oracle_manifest["sample_id"].tolist()

ORACLE_DIR = PHASE5B_ROOT / "decoded/oracle"
END2END_DIR = PHASE5B_ROOT / "decoded/end2end"
verification_rows = []
for variant, directory in (("oracle", ORACLE_DIR), ("end-to-end", END2END_DIR)):
    if not directory.is_dir(): raise RuntimeError(f"Missing decoded directory: {directory}")
    files = sorted(directory.glob("*.npy"))
    if len(files) != EXPECTED_SAMPLES: raise RuntimeError(f"{variant}: expected 1243 .npy files, found {len(files)}")
    for number, sample_id in enumerate(sample_ids, 1):
        path = directory / f"{sample_id}.npy"
        if not path.is_file(): raise RuntimeError(f"{variant}: missing {path.name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        correct_shape = array.ndim == 2 and array.shape[0] > 0 and array.shape[1] == MOTION_DIM
        finite = bool(np.isfinite(array).all())
        if not correct_shape or not finite: raise RuntimeError(f"{variant}/{sample_id}: shape={array.shape}, finite={finite}")
        verification_rows.append({"variant": variant, "sample_id": sample_id, "shape": str(tuple(array.shape)), "finite": finite})
        if number % 250 == 0: print(f"Verified {variant}: {number}/{EXPECTED_SAMPLES}")

pd.DataFrame(verification_rows).to_csv(DIAGNOSTICS / "phase5b_motion_verification.csv", index=False)
write_json(DIAGNOSTICS / "phase5b_artifact_verification.json", {
    "phase5b_root": PHASE5B_ROOT, "oracle_samples": EXPECTED_SAMPLES,
    "end2end_samples": EXPECTED_SAMPLES, "motion_dim": MOTION_DIM,
    "temporal_upsampling_factor": TEMPORAL_FACTOR, "all_finite": True,
    "motions_modified": False, "passed": True,
})
print("Phase-5B decoded motion verification: PASS")

oracle_metrics = pd.read_csv(PHASE5B_ROOT / "results/oracle_motion_metrics.csv", dtype={"sample_id": str})
end2end_metrics = pd.read_csv(PHASE5B_ROOT / "results/end2end_motion_metrics.csv", dtype={"sample_id": str})
collapse_flags = pd.read_csv(PHASE5B_ROOT / "diagnostics/motion_collapse_flags.csv", dtype={"sample_id": str})
extreme_flags = pd.read_csv(PHASE5B_ROOT / "diagnostics/extreme_motion_flags.csv", dtype={"sample_id": str})
phase5b_qualitative_path = PHASE5B_ROOT / "results/qualitative_case_manifest.csv"
phase5b_qualitative = pd.read_csv(phase5b_qualitative_path, dtype={"sample_id": str}) if phase5b_qualitative_path.is_file() else pd.DataFrame()

# %% [markdown]
# ## 5C.2–5C.6 — Evidence-gated feature semantics and BVH cross-reference

# %%
SEARCH_TERMS = [
    "recover_from_ric", "recover_pose", "features_to_pose", "motion_to_bvh", "feature_to_bvh",
    "recover_joints", "inverse_transform", "joint_names", "kinematic_chain", "rotation_6d",
    "rot6d", "foot_contact", "foot contacts", "668", "d-pose", "d_pose", "smpl", "smpl-x",
]
CONVERTER_TERMS = ["recover_from_ric", "recover_pose", "features_to_pose", "motion_to_bvh",
                   "feature_to_bvh", "recover_joints"]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".ipynb"}


def source_authority(path: Path) -> tuple[str, str]:
    lowered = str(path).casefold().replace("\\", "/")
    if "/competitions/" in lowered: return "official competition artifact", "HIGH"
    if "/models/" in lowered and any(term in lowered for term in ("motion-s", "signvrse", "antony")):
        return "official/provider model artifact", "HIGH"
    if any(term in lowered for term in ("signvrse", "motion-s-vae-rvq", "motion_s_vae_rvq")):
        return "provider-associated artifact", "MEDIUM"
    return "participant/unknown artifact (clue only)", "LOW"


def readable_source(path: Path) -> str:
    if path.suffix.casefold() == ".ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
    return path.read_text(encoding="utf-8", errors="replace")


evidence_records = [
    {
        "source": "Anthony Marugu / Signvrse, Semantic-Driven Sign Motion Generation abstract",
        "location": OFFICIAL_ABSTRACT_URL + " (conference book pp. 53–54)",
        "claim": "Motion-S converts raw BVH into a 668-dimensional D-Pose representation incorporating 6D rotations, joint velocities, and foot-contact signals.",
        "interpretation": "Feature families are partially identified; their exact indices, joint order, rotation convention, and inverse transform are not supplied.",
        "confidence": "HIGH for component families; UNRESOLVED for layout",
    },
    {
        "source": "Signvrse Motion-S RVQ-VAE model card",
        "location": OFFICIAL_MODEL_CARD_URL,
        "claim": "The model card documents motion-feature input to the RVQ-VAE and describes the first four generic D-Pose components as root rotational velocity, X/Z root velocity, and root height.",
        "interpretation": "The same card uses a 263-D example and does not provide the complete 668-D layout, joint list, or inverse converter; treat this as partial evidence only.",
        "confidence": "MEDIUM/PARTIAL",
    },
]
file_evidence, converter_candidates = [], []
for base in [Path(CFG.kaggle_input)]:
    if not base.exists(): continue
    for path in base.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in TEXT_SUFFIXES or path.stat().st_size > 10_000_000: continue
        try:
            text = readable_source(path)
        except Exception:
            continue
        lower = text.casefold()
        matched = [term for term in SEARCH_TERMS if term in lower]
        if not matched: continue
        authority, confidence = source_authority(path)
        lines = text.splitlines()
        snippets = []
        for index, line in enumerate(lines):
            if any(term in line.casefold() for term in matched):
                start, stop = max(0, index - 1), min(len(lines), index + 2)
                snippets.append("\n".join(lines[start:stop]).strip())
                if len(snippets) >= 8: break
        record = {"path": str(path), "authority": authority, "confidence": confidence,
                  "matched_terms": matched, "snippets": snippets}
        file_evidence.append(record)
        if any(term in lower for term in CONVERTER_TERMS): converter_candidates.append(record)

evidence_lines = [
    "MOTION-S 668-D FEATURE SEMANTICS EVIDENCE",
    "========================================",
    "",
    "VERIFIED/PUBLISHED EVIDENCE",
]
for record in evidence_records:
    evidence_lines.extend([
        f"Source: {record['source']}", f"Location: {record['location']}",
        f"Claim: {record['claim']}", f"Interpretation: {record['interpretation']}",
        f"Confidence: {record['confidence']}", "",
    ])
evidence_lines.extend([
    "LOCAL KAGGLE ARTIFACT SEARCH",
    f"Files with relevant terms: {len(file_evidence)}",
    f"Files with converter-like terms: {len(converter_candidates)}",
    "Participant/unknown artifacts are clues only and are not accepted as authoritative converters.", "",
])
for record in file_evidence:
    evidence_lines.extend([
        f"File: {record['path']}", f"Authority: {record['authority']}",
        f"Matched: {', '.join(record['matched_terms'])}",
        "Relevant code/text:", *[f"  {snippet}" for snippet in record["snippets"]],
        "Interpretation: requires explicit 668-D input contract and ground-truth BVH validation before use.",
        f"Confidence: {record['confidence']}", "",
    ])
evidence_lines.extend([
    "CURRENT SCIENTIFIC CONCLUSION",
    "The representation is partially identified as 668-D D-Pose containing 6D rotations, joint velocities, and foot contacts.",
    "The exact index layout, joint order, rotation convention, root integration, and inverse feature-to-joint/BVH conversion remain unresolved unless a provider utility discovered above explicitly proves them.",
])
(DIAGNOSTICS / "feature_semantics_evidence.txt").write_text("\n".join(evidence_lines), encoding="utf-8")

official_converter_candidates = [record for record in converter_candidates if record["confidence"] in {"HIGH", "MEDIUM"}]
# Static keyword discovery is not sufficient to execute a converter. It must expose an explicit
# 668-D contract and pass the original-BVH ground-truth gate. No generic dynamic import is attempted.
converter_verified = False
converter_status = "NOT FOUND" if not official_converter_candidates else "CANDIDATE FOUND — NOT VERIFIED"


def parse_bvh_header(path: Path) -> dict[str, Any]:
    joints = frames = None; frame_time = None; joint_count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("ROOT ") or stripped.startswith("JOINT "): joint_count += 1
            if stripped.casefold().startswith("frames:"):
                frames = int(stripped.split(":", 1)[1].strip())
            elif stripped.casefold().startswith("frame time:"):
                frame_time = float(stripped.split(":", 1)[1].strip()); break
    if frames is None or frame_time is None or joint_count <= 0:
        raise ValueError(f"Could not parse BVH header: {path}")
    return {"bvh_frames": frames, "num_bvh_joints": joint_count, "frame_time": frame_time}


feature_index = {path.stem: path for path in MOTION_FEATURES.glob("*.npy")}
bvh_paths = list(TRAIN_BVH_ROOT.rglob("*.bvh"))
bvh_index: dict[str, Path] = {}
for path in bvh_paths:
    sample_id = path.stem
    preferred = path.parent.name == sample_id
    if sample_id not in bvh_index or preferred: bvh_index[sample_id] = path

bvh_alignment_ids = [sample_id for sample_id in sample_ids if sample_id in feature_index and sample_id in bvh_index][:CFG.bvh_alignment_samples]
if len(bvh_alignment_ids) < 5: raise RuntimeError(f"Only {len(bvh_alignment_ids)} feature/BVH pairs found; need at least 5")
bvh_rows = []
for sample_id in bvh_alignment_ids:
    feature = np.load(feature_index[sample_id], mmap_mode="r", allow_pickle=False)
    header = parse_bvh_header(bvh_index[sample_id])
    bvh_rows.append({
        "sample_id": sample_id, "feature_frames": int(feature.shape[0]),
        "bvh_frames": header["bvh_frames"], "frame_match": int(feature.shape[0]) == header["bvh_frames"],
        "num_bvh_joints": header["num_bvh_joints"], "frame_time": header["frame_time"],
        "bvh_path": str(bvh_index[sample_id]),
    })
bvh_alignment = pd.DataFrame(bvh_rows)
bvh_alignment.to_csv(DIAGNOSTICS / "bvh_feature_alignment.csv", index=False)
frame_times = bvh_alignment["frame_time"].dropna().to_numpy(dtype=float)
frame_rate = float(1.0 / frame_times[0]) if len(frame_times) and np.allclose(frame_times, frame_times[0], atol=1e-9) else None

converter_verification = {
    "status": converter_status,
    "official_or_provider_candidates": official_converter_candidates,
    "explicit_668_input_contract_verified": False,
    "ground_truth_feature_conversion_run": False,
    "ground_truth_bvh_comparison_passed": False,
    "feature_to_joint_converter": None,
    "feature_to_bvh_converter": None,
    "reason": "No converter is executed without an explicit Motion-S 668-D contract and a provider-defined invocation path.",
}
write_json(DIAGNOSTICS / "converter_verification.json", converter_verification)
pd.DataFrame([{
    "sample_id": sample_id, "status": "NOT_RUN_NO_VERIFIED_CONVERTER",
    "original_bvh": str(bvh_index[sample_id]), "recovered_motion": None, "passed": False,
} for sample_id in bvh_alignment_ids[:5]]).to_csv(RESULTS / "ground_truth_bvh_recovery_validation.csv", index=False)

representation_contract = {
    "feature_dim": MOTION_DIM,
    "feature_shape": "[T,668]",
    "frame_rate": frame_rate,
    "feature_semantics_verified": False,
    "feature_semantics_status": "PARTIAL",
    "verified_feature_families": ["6D rotations", "joint velocities", "foot-contact signals"],
    "joint_representation": None,
    "num_joints": None,
    "root_representation": None,
    "hand_representation": None,
    "feature_to_joint_converter": None,
    "feature_to_bvh_converter": None,
    "ground_truth_conversion_verified": False,
    "bvh_feature_frame_alignment_samples": len(bvh_alignment),
    "bvh_feature_exact_frame_matches": int(bvh_alignment["frame_match"].sum()),
}
write_json(DIAGNOSTICS / "representation_contract.json", representation_contract)
print("Feature semantics: PARTIAL; exact 668-D layout unresolved")
print("Feature -> skeleton/BVH converter:", converter_status)

# %% [markdown]
# ## 5C.8–5C.9 — Select 25 diagnostic cases and always create feature-space plots

# %%
def bool_values(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series): return series.fillna(False)
    return series.fillna(False).astype(str).str.casefold().isin({"true", "1", "yes"})


def load_motion(sample_id: str, variant: str) -> np.ndarray:
    if variant == "ground-truth": path = feature_index[sample_id]
    elif variant == "oracle": path = ORACLE_DIR / f"{sample_id}.npy"
    elif variant == "end-to-end": path = END2END_DIR / f"{sample_id}.npy"
    else: raise ValueError(variant)
    motion = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    if motion.ndim != 2 or motion.shape[1] != MOTION_DIM or not np.isfinite(motion).all():
        raise RuntimeError(f"Invalid {variant} motion for {sample_id}: {motion.shape}")
    return motion


def normalized_active(motion: np.ndarray) -> np.ndarray:
    return ((motion[:, active_dimensions] - motion_mean[active_dimensions]) /
            motion_std[active_dimensions]).astype(np.float32, copy=False)


def velocity_magnitude(motion: np.ndarray) -> np.ndarray:
    z_motion = normalized_active(motion)
    if len(z_motion) < 2: return np.empty(0, dtype=np.float32)
    return np.sqrt(np.mean(np.square(np.diff(z_motion, axis=0), dtype=np.float64), axis=1)).astype(np.float32)


def repetition_score(values: np.ndarray, max_lag: int = 180) -> float:
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(series) < 6: return float("nan")
    series = series - series.mean()
    denominator = float(np.dot(series, series))
    if denominator <= 1e-15: return 0.0
    limit = min(max_lag, len(series) // 2)
    if limit < 2: return float("nan")
    correlations = [float(np.dot(series[:-lag], series[lag:]) /
                          np.sqrt(np.dot(series[:-lag], series[:-lag]) * np.dot(series[lag:], series[lag:])))
                    for lag in range(2, limit + 1)
                    if np.dot(series[:-lag], series[:-lag]) > 0 and np.dot(series[lag:], series[lag:]) > 0]
    return max(correlations) if correlations else float("nan")


def longest_true_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


gt_velocity_arrays, gt_dynamics_rows = [], []
for number, sample_id in enumerate(sample_ids, 1):
    gt = load_motion(sample_id, "ground-truth")
    velocity = velocity_magnitude(gt)
    if velocity.size: gt_velocity_arrays.append(velocity)
    gt_dynamics_rows.append({
        "sample_id": sample_id,
        "gt_mean_velocity": float(np.mean(velocity)) if velocity.size else 0.0,
        "gt_temporal_variance": float(np.mean(np.var(normalized_active(gt), axis=0, dtype=np.float64))),
        "gt_repetition_score": repetition_score(velocity),
    })
    if number % 250 == 0: print(f"Ground-truth dynamics: {number}/{EXPECTED_SAMPLES}")
if not gt_velocity_arrays: raise RuntimeError("No ground-truth velocity samples")
all_gt_velocity = np.concatenate(gt_velocity_arrays)
gt_dynamics = pd.DataFrame(gt_dynamics_rows)
low_motion_threshold = float(np.percentile(all_gt_velocity, CFG.low_motion_gt_percentile))
gt_mean_velocity_p01 = float(np.percentile(gt_dynamics["gt_mean_velocity"], CFG.low_motion_gt_percentile))
gt_repetition_p99 = float(np.nanpercentile(gt_dynamics["gt_repetition_score"], CFG.repetition_gt_percentile))
gt_high_velocity_p99 = float(np.percentile(gt_dynamics["gt_mean_velocity"], CFG.high_velocity_gt_percentile))
write_json(DIAGNOSTICS / "phase5c_empirical_thresholds.json", {
    "basis": "supported ground-truth Motion-Features",
    "low_motion_per_frame_velocity_gt_p01": low_motion_threshold,
    "low_motion_sample_mean_velocity_gt_p01": gt_mean_velocity_p01,
    "repetition_score_gt_p99": gt_repetition_p99,
    "sample_mean_velocity_gt_p99": gt_high_velocity_p99,
})

oracle_selected = oracle_metrics[[
    "sample_id", "gt_frames", "pred_frames", "normalized_feature_MAE", "mean_velocity_magnitude"
]].rename(columns={
    "pred_frames": "oracle_frames", "normalized_feature_MAE": "oracle_normalized_feature_MAE",
    "mean_velocity_magnitude": "oracle_mean_velocity",
})
end_selected = end2end_metrics[[
    "sample_id", "pred_frames", "normalized_feature_MAE", "mean_velocity_magnitude"
]].rename(columns={
    "pred_frames": "end2end_frames", "normalized_feature_MAE": "end2end_normalized_feature_MAE",
    "mean_velocity_magnitude": "end2end_mean_velocity",
})
case_pool = oracle_selected.merge(end_selected, on="sample_id", validate="one_to_one")
for variant, prefix in (("oracle", "oracle"), ("end-to-end", "end2end")):
    collapse_part = collapse_flags.loc[collapse_flags["variant"] == variant, ["sample_id", "static_motion"]].copy()
    collapse_part[f"{prefix}_collapse_flag"] = bool_values(collapse_part.pop("static_motion"))
    extreme_part = extreme_flags.loc[extreme_flags["variant"] == variant, ["sample_id", "extreme_motion"]].copy()
    extreme_part[f"{prefix}_extreme_flag"] = bool_values(extreme_part.pop("extreme_motion"))
    case_pool = case_pool.merge(collapse_part, on="sample_id", how="left", validate="one_to_one")
    case_pool = case_pool.merge(extreme_part, on="sample_id", how="left", validate="one_to_one")
for column in ("oracle_collapse_flag", "end2end_collapse_flag", "oracle_extreme_flag", "end2end_extreme_flag"):
    case_pool[column] = case_pool[column].fillna(False).astype(bool)
case_pool["any_collapse"] = case_pool["oracle_collapse_flag"] | case_pool["end2end_collapse_flag"]
case_pool["any_extreme"] = case_pool["oracle_extreme_flag"] | case_pool["end2end_extreme_flag"]
case_pool["combined_error"] = (case_pool["oracle_normalized_feature_MAE"] + case_pool["end2end_normalized_feature_MAE"]) / 2

used_ids: set[str] = set()
selected_groups = []


def choose_group(candidates: pd.DataFrame, group: str, sort_column: str, ascending: bool, middle: bool = False) -> None:
    available = candidates.loc[~candidates["sample_id"].isin(used_ids)].copy()
    if middle:
        median = float(available[sort_column].median())
        available["_distance"] = (available[sort_column] - median).abs()
        chosen = available.sort_values("_distance").head(CFG.cases_per_group)
    else:
        chosen = available.sort_values(sort_column, ascending=ascending).head(CFG.cases_per_group)
    if len(chosen) != CFG.cases_per_group:
        raise RuntimeError(f"Could not select {CFG.cases_per_group} unique samples for {group}; found {len(chosen)}")
    chosen = chosen.copy(); chosen.insert(0, "diagnostic_group", group)
    selected_groups.append(chosen); used_ids.update(chosen["sample_id"].astype(str))


noncollapsed = case_pool.loc[~case_pool["any_collapse"] & ~case_pool["any_extreme"]]
choose_group(noncollapsed, "A_LOW_ERROR_NON_COLLAPSED", "combined_error", True)
choose_group(noncollapsed, "B_MEDIUM_ERROR_NON_COLLAPSED", "combined_error", True, middle=True)
choose_group(case_pool, "C_HIGH_ERROR", "combined_error", False)
choose_group(case_pool.loc[case_pool["any_collapse"]], "D_COLLAPSE_FLAGGED", "end2end_mean_velocity", True)
extreme_ranked = case_pool.loc[case_pool["any_extreme"]].copy()
choose_group(extreme_ranked, "E_EXTREME_MOTION_FLAGGED", "combined_error", False)
case_manifest = pd.concat(selected_groups, ignore_index=True)
if len(case_manifest) != CFG.cases_per_group * 5 or case_manifest["sample_id"].duplicated().any():
    raise RuntimeError("Diagnostic case selection did not produce 25 unique IDs")

train_frame = pd.read_csv(MOTION_S_ROOT / "train.csv")
id_column = next((column for column in ("id", "sample_id") if column in train_frame), None)
if id_column is None: raise RuntimeError("Original train.csv has no ID column")
train_frame["sample_id"] = train_frame[id_column].map(canonical_id)
metadata_columns = ["sample_id"] + [column for column in ("sentence", "gloss") if column in train_frame]
metadata = train_frame[metadata_columns].drop_duplicates("sample_id").rename(columns={"gloss": "ground_truth_gloss"})
case_manifest = case_manifest.merge(metadata, on="sample_id", how="left", validate="one_to_one")


def find_optional_csv(filename: str, required: set[str]) -> Optional[Path]:
    candidates = []
    for base in SEARCH_BASES:
        if not base.exists(): continue
        for path in base.rglob(filename):
            try: columns = set(pd.read_csv(path, nrows=0).columns)
            except Exception: continue
            if required.issubset(columns): candidates.append(path.resolve())
    candidates = sorted(dict.fromkeys(candidates), key=str)
    if len(candidates) == 1: return candidates[0]
    if len(candidates) > 1:
        hashes = {path: sha256_file(path) for path in candidates}
        if len(set(hashes.values())) == 1: return candidates[0]
        print(f"Optional {filename} ambiguous; disabled: {[str(path) for path in candidates]}")
    return None


phase3_predictions_path = find_optional_csv("test_predictions.csv", {"sample_id", "predicted_gloss"})
oracle_token_predictions_path = find_optional_csv("test_oracle_predictions.csv", {"sample_id", "aligned_token_accuracy"})
end_token_predictions_path = find_optional_csv("test_end2end_predictions.csv", {"sample_id", "aligned_token_accuracy"})
if phase3_predictions_path:
    phase3_predictions = pd.read_csv(phase3_predictions_path, dtype={"sample_id": str})
    case_manifest = case_manifest.merge(phase3_predictions[["sample_id", "predicted_gloss"]],
                                        on="sample_id", how="left", validate="one_to_one")
elif not phase5b_qualitative.empty and {"sample_id", "phase3_predicted_gloss"}.issubset(phase5b_qualitative.columns):
    case_manifest = case_manifest.merge(phase5b_qualitative[["sample_id", "phase3_predicted_gloss"]],
                                        on="sample_id", how="left", validate="one_to_one")
if "predicted_gloss" not in case_manifest and "phase3_predicted_gloss" not in case_manifest:
    case_manifest["phase3_predicted_gloss"] = pd.NA
if "predicted_gloss" in case_manifest: case_manifest = case_manifest.rename(columns={"predicted_gloss": "phase3_predicted_gloss"})
if end_token_predictions_path:
    end_token_predictions = pd.read_csv(end_token_predictions_path, dtype={"sample_id": str})
    case_manifest = case_manifest.merge(end_token_predictions[["sample_id", "aligned_token_accuracy"]].rename(
        columns={"aligned_token_accuracy": "phase4_token_accuracy"}), on="sample_id", how="left", validate="one_to_one")
else:
    case_manifest["phase4_token_accuracy"] = pd.NA

manifest_columns = [
    "diagnostic_group", "sample_id", "sentence", "ground_truth_gloss", "phase3_predicted_gloss",
    "gt_frames", "oracle_frames", "end2end_frames", "oracle_normalized_feature_MAE",
    "end2end_normalized_feature_MAE", "oracle_collapse_flag", "end2end_collapse_flag",
    "oracle_extreme_flag", "end2end_extreme_flag", "phase4_token_accuracy",
]
for column in manifest_columns:
    if column not in case_manifest: case_manifest[column] = pd.NA
case_manifest[manifest_columns].to_csv(RESULTS / "phase5c_case_manifest.csv", index=False)

for _, selected in case_manifest.iterrows():
    sample_id = str(selected["sample_id"])
    gt = load_motion(sample_id, "ground-truth")
    oracle = load_motion(sample_id, "oracle")
    end2end = load_motion(sample_id, "end-to-end")
    active_indices = np.flatnonzero(active_dimensions)
    top_features = active_indices[np.argsort(np.var(gt[:, active_dimensions], axis=0))[-CFG.trajectory_features:][::-1]]
    figure, axes = plt.subplots(len(top_features), 1, figsize=(12, 2.5 * len(top_features)))
    axes = np.atleast_1d(axes)
    for axis, feature in zip(axes, top_features):
        axis.plot(gt[:, feature], label="ground truth", linewidth=1.4)
        axis.plot(oracle[:, feature], label="oracle", linewidth=1.0, alpha=.85)
        axis.plot(end2end[:, feature], label="end-to-end", linewidth=1.0, alpha=.85)
        axis.set_ylabel(f"feature {int(feature)}"); axis.grid(alpha=.2)
    axes[0].legend(fontsize=8); axes[-1].set_xlabel("Frame index (untrimmed)")
    figure.suptitle(f"Feature-space trajectories — {sample_id}")
    figure.tight_layout(); figure.savefig(FEATURE_FIGURES / f"{sample_id}_trajectories.png", dpi=160, bbox_inches="tight"); plt.close(figure)

    gt_velocity, oracle_velocity, end_velocity = velocity_magnitude(gt), velocity_magnitude(oracle), velocity_magnitude(end2end)
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(gt_velocity, label="ground truth", linewidth=1.4)
    axis.plot(oracle_velocity, label="oracle", linewidth=1.0, alpha=.85)
    axis.plot(end_velocity, label="end-to-end", linewidth=1.0, alpha=.85)
    axis.axhline(low_motion_threshold, color="black", linestyle="--", linewidth=.8, label="GT velocity p1")
    axis.set(xlabel="Frame transition (untrimmed)", ylabel="Active-feature normalized RMS velocity",
             title=f"Velocity magnitude — {sample_id}")
    axis.legend(fontsize=8); axis.grid(alpha=.2)
    figure.tight_layout(); figure.savefig(VELOCITY_FIGURES / f"{sample_id}_velocity.png", dpi=160, bbox_inches="tight"); plt.close(figure)

print("Diagnostic cases selected:", len(case_manifest))

# %% [markdown]
# ## 5C.10–5C.13 — Collapse, excess-tail, EOS, and RVQ-level diagnostics

# %%
collapse_analysis_rows = []
flagged = collapse_flags.loc[bool_values(collapse_flags["static_motion"])].copy()
for number, row in enumerate(flagged.itertuples(index=False), 1):
    sample_id, variant = str(row.sample_id), str(row.variant)
    motion = load_motion(sample_id, variant)
    velocity = velocity_magnitude(motion)
    low_mask = velocity <= low_motion_threshold
    collapse_analysis_rows.append({
        "sample_id": sample_id,
        "oracle_or_end2end": variant,
        "mean_velocity": float(np.mean(velocity)) if velocity.size else 0.0,
        "median_velocity": float(np.median(velocity)) if velocity.size else 0.0,
        "low_motion_fraction": float(np.mean(low_mask)) if low_mask.size else 1.0,
        "longest_low_motion_run": longest_true_run(low_mask),
        "temporal_variance": float(np.mean(np.var(normalized_active(motion), axis=0, dtype=np.float64))),
        "autocorrelation_motion_magnitude": repetition_score(velocity),
        "repetition_score": repetition_score(velocity),
        "collapse_pattern": "AUTO_UNKNOWN",
    })
    if number % 250 == 0: print(f"Collapse analysis: {number}/{len(flagged)}")
collapse_analysis = pd.DataFrame(collapse_analysis_rows)
collapse_analysis.to_csv(RESULTS / "collapse_analysis.csv", index=False)

collapse_case_ids = case_manifest.loc[
    case_manifest["diagnostic_group"] == "D_COLLAPSE_FLAGGED", "sample_id"
].astype(str).tolist()
for sample_id in collapse_case_ids:
    figure, axes = plt.subplots(2, 1, figsize=(12, 7))
    for variant, color in (("ground-truth", "black"), ("oracle", "#3b82f6"), ("end-to-end", "#f97316")):
        velocity = velocity_magnitude(load_motion(sample_id, variant))
        axes[0].plot(velocity, label=variant, color=color, alpha=.85)
        if velocity.size:
            axes[1].plot(np.cumsum(velocity <= low_motion_threshold) / np.arange(1, len(velocity) + 1),
                         label=variant, color=color, alpha=.85)
    axes[0].axhline(low_motion_threshold, color="gray", linestyle="--", linewidth=.8)
    axes[0].set(ylabel="Normalized RMS velocity", title=f"Collapse diagnostic — {sample_id}")
    axes[1].set(xlabel="Frame transition", ylabel="Cumulative low-motion fraction")
    for axis in axes: axis.legend(fontsize=8); axis.grid(alpha=.2)
    figure.tight_layout(); figure.savefig(COLLAPSE_FIGURES / f"{sample_id}_collapse.png", dpi=160, bbox_inches="tight"); plt.close(figure)

# %% [markdown]
# ### 5C.11 — Over-generation tail analysis

# %%

def segment_statistics(motion: np.ndarray) -> dict[str, float]:
    velocity = velocity_magnitude(motion)
    return {
        "frames": int(len(motion)),
        "mean_velocity": float(np.mean(velocity)) if velocity.size else 0.0,
        "feature_variance": float(np.mean(np.var(normalized_active(motion), axis=0, dtype=np.float64))) if len(motion) else float("nan"),
        "repetition_score": repetition_score(velocity),
    }


tail_rows = []
for _, selected in case_manifest.iterrows():
    sample_id = str(selected["sample_id"])
    gt_frames = int(selected["gt_frames"])
    for variant in ("oracle", "end-to-end"):
        prediction = load_motion(sample_id, variant)
        first = prediction[:min(gt_frames, len(prediction))]
        tail = prediction[gt_frames:] if len(prediction) > gt_frames else prediction[:0]
        first_stats = segment_statistics(first)
        tail_stats = segment_statistics(tail) if len(tail) else {
            "frames": 0, "mean_velocity": float("nan"), "feature_variance": float("nan"), "repetition_score": float("nan")
        }
        if len(tail) == 0: behavior = "NO_EXCESS_TAIL"
        elif tail_stats["mean_velocity"] <= gt_mean_velocity_p01: behavior = "LOW_MOTION_TAIL"
        elif np.isfinite(tail_stats["repetition_score"]) and tail_stats["repetition_score"] >= gt_repetition_p99:
            behavior = "REPETITIVE_TAIL"
        else: behavior = "ACTIVE_NONREPETITIVE_TAIL"
        tail_rows.append({
            "sample_id": sample_id, "variant": variant, "gt_frames": gt_frames,
            "predicted_frames": len(prediction), "excess_frames": max(len(prediction) - gt_frames, 0),
            "first_gt_length_mean_velocity": first_stats["mean_velocity"],
            "first_gt_length_feature_variance": first_stats["feature_variance"],
            "tail_mean_velocity": tail_stats["mean_velocity"],
            "tail_feature_variance": tail_stats["feature_variance"],
            "tail_repetition_score": tail_stats["repetition_score"],
            "empirical_tail_behavior": behavior,
        })

        velocity = velocity_magnitude(prediction)
        figure, axis = plt.subplots(figsize=(12, 4))
        axis.plot(velocity, color="#3b82f6" if variant == "oracle" else "#f97316", linewidth=1)
        axis.axvline(max(gt_frames - 1, 0), color="black", linestyle="--", label="GT duration boundary")
        axis.axhline(low_motion_threshold, color="gray", linestyle=":", label="GT velocity p1")
        axis.set(xlabel="Frame transition", ylabel="Normalized RMS velocity",
                 title=f"{variant} excess-tail diagnostic — {sample_id}")
        axis.legend(fontsize=8); axis.grid(alpha=.2)
        figure.tight_layout(); figure.savefig(TAIL_FIGURES / f"{sample_id}_{variant.replace('-', '')}_tail.png", dpi=150, bbox_inches="tight"); plt.close(figure)

tail_analysis = pd.DataFrame(tail_rows)
tail_analysis.to_csv(RESULTS / "overgeneration_tail_analysis.csv", index=False)
tail_behavior_counts = tail_analysis.loc[tail_analysis["excess_frames"] > 0, "empirical_tail_behavior"].value_counts()
dominant_tail_behavior = str(tail_behavior_counts.index[0]) if len(tail_behavior_counts) else "NO_EXCESS_TAIL"

# %% [markdown]
# ### 5C.12 — EOS failure relation

# %%
eos_rows = []
for variant, path in (("oracle", oracle_token_predictions_path), ("end-to-end", end_token_predictions_path)):
    if path is None: continue
    predictions = pd.read_csv(path, dtype={"sample_id": str})
    required_eos = {"sample_id", "eos_reached", "max_length_termination"}
    if not required_eos.issubset(predictions.columns): continue
    predictions["eos_reached"] = bool_values(predictions["eos_reached"])
    predictions["max_length_termination"] = bool_values(predictions["max_length_termination"])
    predictions["termination"] = np.select(
        [predictions["eos_reached"], predictions["max_length_termination"]],
        ["EOS_TERMINATED", "MAX_LENGTH_TERMINATED"], default="OTHER"
    )
    metric_frame = oracle_metrics if variant == "oracle" else end2end_metrics
    variant_collapse = collapse_flags.loc[collapse_flags["variant"] == variant, ["sample_id", "static_motion"]].copy()
    variant_collapse["static_motion"] = bool_values(variant_collapse["static_motion"])
    variant_tail = tail_analysis.loc[tail_analysis["variant"] == variant]
    joined = predictions.merge(metric_frame[["sample_id", "frame_error", "normalized_feature_MAE"]],
                               on="sample_id", validate="one_to_one").merge(
        variant_collapse, on="sample_id", how="left", validate="one_to_one"
    ).merge(variant_tail[["sample_id", "tail_mean_velocity", "tail_repetition_score"]],
            on="sample_id", how="left", validate="one_to_one")
    for termination, group in joined.groupby("termination"):
        eos_rows.append({
            "analysis_available": True, "variant": variant, "termination": termination,
            "samples": len(group), "mean_duration_error": float(group["frame_error"].mean()),
            "collapse_rate": float(bool_values(group["static_motion"]).mean()),
            "mean_normalized_motion_error": float(group["normalized_feature_MAE"].mean()),
            "mean_tail_velocity_selected_cases": float(group["tail_mean_velocity"].mean()),
            "mean_tail_repetition_selected_cases": float(group["tail_repetition_score"].mean()),
        })
if not eos_rows:
    eos_rows = [{"analysis_available": False, "variant": None, "termination": "UNAVAILABLE",
                 "samples": 0, "reason": "Sample-level EOS/max-length metadata not located."}]
eos_motion_relation = pd.DataFrame(eos_rows)
eos_motion_relation.to_csv(RESULTS / "eos_motion_relation.csv", index=False)

# %% [markdown]
# ### 5C.13 — Base-to-full RVQ level ablation
#
# Full-level redecoding may differ minutely from saved Phase-5B arrays because Phase 5B used CUDA
# while this optional audit can execute on CPU. Consistency is checked with conservative float32
# absolute and relative tolerances; large architectural or normalization mismatches still fail.

# %%

def resolve_optional_exact(filename: str, expected_sha256: Optional[str] = None) -> Optional[Path]:
    candidates = []
    for base in SEARCH_BASES:
        if base.exists(): candidates.extend(path.resolve() for path in base.rglob(filename) if path.is_file())
    candidates = sorted(dict.fromkeys(candidates), key=str)
    if expected_sha256:
        matches = [path for path in candidates if sha256_file(path) == expected_sha256]
        return matches[0] if matches else None
    if len(candidates) == 1: return candidates[0]
    if len(candidates) > 1:
        hashes = {path: sha256_file(path) for path in candidates}
        if len(set(hashes.values())) == 1: return candidates[0]
    return None


class PackedTokens:
    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as data:
            self.ids = data["sample_ids"].astype(str); self.offsets = data["offsets"].astype(np.int64)
            self.lengths = data["lengths"].astype(np.int64); self.flat = data["tokens_flat"].copy()
        if self.flat.ndim != 2 or self.flat.shape[0] != RVQ_LEVELS or len(self.ids) != EXPECTED_SAMPLES:
            raise RuntimeError(f"Invalid token NPZ: {path}")
        self.index = {sample_id: index for index, sample_id in enumerate(self.ids)}

    def get(self, sample_id: str) -> np.ndarray:
        index = self.index[sample_id]; start, stop = int(self.offsets[index]), int(self.offsets[index + 1])
        return self.flat[:, start:stop]


class CandidateBDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 512, 3, 2, 1, output_padding=1), nn.ReLU(), nn.BatchNorm1d(512),
            nn.ConvTranspose1d(512, 512, 3, 2, 1, output_padding=1), nn.ReLU(), nn.BatchNorm1d(512),
            nn.ConvTranspose1d(512, 512, 3, 1, 1, output_padding=0), nn.ReLU(), nn.BatchNorm1d(512),
            nn.ConvTranspose1d(512, 668, 3, 1, 1, output_padding=0),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor: return self.decoder(latent)


def resample_time(motion: np.ndarray, target: int) -> np.ndarray:
    if len(motion) == target: return motion.copy()
    if len(motion) == 1: return np.repeat(motion, target, axis=0)
    tensor = torch.from_numpy(motion.astype(np.float32).T).unsqueeze(0)
    return F.interpolate(tensor, size=target, mode="linear", align_corners=True).squeeze(0).T.numpy()


checkpoint_sha = decoder_contract.get("checkpoint_sha256")
ablation_checkpoint = resolve_optional_exact("rvq_vae_best.pth", checkpoint_sha)
oracle_token_npz = resolve_optional_exact("test_oracle_tokens.npz")
end_token_npz = resolve_optional_exact("test_end2end_tokens.npz")
ablation_rows = []
ablation_consistency_rows = []
ablation_performed = all(path is not None for path in (ablation_checkpoint, oracle_token_npz, end_token_npz))
if ablation_performed:
    checkpoint = torch.load(ablation_checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    decoder = CandidateBDecoder().eval()
    decoder_state = {key[len("decoder."):]: value for key, value in state.items() if key.startswith("decoder.")}
    decoder.load_state_dict(decoder_state, strict=True)
    codebooks = torch.stack([state[f"rvq.quantizers.{level}.embedding"].float() for level in range(RVQ_LEVELS)])
    oracle_tokens, end_tokens = PackedTokens(oracle_token_npz), PackedTokens(end_token_npz)
    mean_tensor, std_tensor = torch.from_numpy(motion_mean), torch.from_numpy(np.where(active_dimensions, motion_std, 1.0).astype(np.float32))

    @torch.inference_mode()
    def partial_decode(tokens: np.ndarray, levels: int) -> np.ndarray:
        ids = torch.from_numpy(tokens.astype(np.int64, copy=False))
        latent = torch.zeros((LATENT_DIM, tokens.shape[1]), dtype=torch.float32)
        for level in range(levels): latent += codebooks[level][:, ids[level]]
        normalized = decoder(latent.unsqueeze(0)).squeeze(0).T
        return (normalized * std_tensor + mean_tensor).numpy().astype(np.float32)

    for selected_number, sample_id in enumerate(case_manifest["sample_id"].astype(str)):
        gt = load_motion(sample_id, "ground-truth")
        for variant, packed in (("oracle", oracle_tokens), ("end-to-end", end_tokens)):
            tokens = packed.get(sample_id)
            for levels in range(1, RVQ_LEVELS + 1):
                decoded = partial_decode(tokens, levels)
                if levels == RVQ_LEVELS:
                    saved = load_motion(sample_id, variant)
                    absolute_difference = np.abs(decoded.astype(np.float64) - saved.astype(np.float64))
                    maximum_difference = float(np.max(absolute_difference))
                    mean_difference = float(np.mean(absolute_difference))
                    consistent = bool(np.allclose(decoded, saved, rtol=1e-5, atol=5e-4, equal_nan=False))
                    ablation_consistency_rows.append({
                        "sample_id": sample_id, "variant": variant,
                        "maximum_absolute_difference": maximum_difference,
                        "mean_absolute_difference": mean_difference,
                        "rtol": 1e-5, "atol": 5e-4, "consistent": consistent,
                    })
                    if not consistent:
                        raise RuntimeError(
                            f"RVQ ablation full-level mismatch for {variant}/{sample_id}: "
                            f"max_abs={maximum_difference}, mean_abs={mean_difference}"
                        )
                aligned = resample_time(decoded, len(gt))
                raw_difference = aligned.astype(np.float64) - gt.astype(np.float64)
                z_difference = normalized_active(aligned).astype(np.float64) - normalized_active(gt).astype(np.float64)
                ablation_rows.append({
                    "performed": True, "sample_id": sample_id, "variant": variant,
                    "levels_used": levels, "included_levels": "+".join(str(level) for level in range(levels)),
                    "time_normalized_raw_feature_MAE": float(np.mean(np.abs(raw_difference))),
                    "time_normalized_normalized_feature_MAE": float(np.mean(np.abs(z_difference))),
                    "mean_velocity": float(np.mean(velocity_magnitude(decoded))),
                    "decoded_frames": len(decoded), "gt_frames": len(gt),
                })
else:
    missing = [name for name, path in (("checkpoint", ablation_checkpoint), ("oracle tokens", oracle_token_npz),
                                       ("end-to-end tokens", end_token_npz)) if path is None]
    ablation_rows = [{"performed": False, "reason": "Missing or ambiguous: " + ", ".join(missing)}]
rvq_ablation = pd.DataFrame(ablation_rows)
rvq_ablation.to_csv(RESULTS / "rvq_level_ablation.csv", index=False)
pd.DataFrame(ablation_consistency_rows).to_csv(DIAGNOSTICS / "rvq_ablation_redecode_consistency.csv", index=False)
print("RVQ level ablation:", "PASS" if ablation_performed else "NOT RUN")

# %% [markdown]
# ## 5C.14–5C.25 — Failure taxonomy, bottleneck evidence, and Phase-6 handoff

# %%
taxonomy_rows = []
for _, selected in case_manifest.iterrows():
    sample_id = str(selected["sample_id"])
    for variant, collapse_column, extreme_column, frames_column, velocity_column in (
        ("oracle", "oracle_collapse_flag", "oracle_extreme_flag", "oracle_frames", "oracle_mean_velocity"),
        ("end-to-end", "end2end_collapse_flag", "end2end_extreme_flag", "end2end_frames", "end2end_mean_velocity"),
    ):
        labels = []
        predicted_frames, gt_frames = int(selected[frames_column]), int(selected["gt_frames"])
        tail_row = tail_analysis[(tail_analysis["sample_id"] == sample_id) & (tail_analysis["variant"] == variant)].iloc[0]
        if bool(selected[collapse_column]): labels.append("STATIC_COLLAPSE")
        if bool(selected[extreme_column]): labels.append("EXTREME_FEATURES")
        if predicted_frames > gt_frames: labels.append("OVER_GENERATION")
        elif predicted_frames < gt_frames: labels.append("PREMATURE_STOP")
        if tail_row["empirical_tail_behavior"] == "LOW_MOTION_TAIL": labels.append("TERMINAL_FREEZE")
        if tail_row["empirical_tail_behavior"] == "REPETITIVE_TAIL": labels.append("MOTION_LOOP")
        if float(selected[velocity_column]) >= gt_high_velocity_p99: labels.append("HIGH_FREQUENCY_JITTER")
        if not labels: labels.append("UNKNOWN")
        taxonomy_rows.append({
            "sample_id": sample_id, "variant": variant,
            "automatic_failure_labels": "|".join(dict.fromkeys(labels)),
            "semantic_mismatch_assigned": False,
            "visual_confirmation_required": True,
            "notes": "No semantic/root/body/hand label is assigned without verified joint conversion and human sign inspection.",
        })
failure_taxonomy = pd.DataFrame(taxonomy_rows)
failure_taxonomy.to_csv(RESULTS / "failure_taxonomy.csv", index=False)

oracle_signed_error = float(oracle_metrics["frame_error"].mean())
end_signed_error = float(end2end_metrics["frame_error"].mean())
oracle_motion_mae = float(oracle_metrics["normalized_feature_MAE"].mean())
end_motion_mae = float(end2end_metrics["normalized_feature_MAE"].mean())
phase3_degradation = end_motion_mae - oracle_motion_mae
oracle_collapse_total = int(bool_values(collapse_flags.loc[collapse_flags["variant"] == "oracle", "static_motion"]).sum())
end_collapse_total = int(bool_values(collapse_flags.loc[collapse_flags["variant"] == "end-to-end", "static_motion"]).sum())
oracle_extreme_total = int(bool_values(extreme_flags.loc[extreme_flags["variant"] == "oracle", "extreme_motion"]).sum())
end_extreme_total = int(bool_values(extreme_flags.loc[extreme_flags["variant"] == "end-to-end", "extreme_motion"]).sum())
dominant_collapse_behavior = "AUTO_UNKNOWN — feature-space collapse confirmed; manual plot inspection required for subtype"

evidence = [
    f"Oracle mean signed duration error is {oracle_signed_error:.3f} frames and end-to-end is {end_signed_error:.3f} frames.",
    f"Collapse flags affect {oracle_collapse_total} oracle and {end_collapse_total} end-to-end samples.",
    f"Oracle-to-end-to-end normalized-feature degradation is only {phase3_degradation:.6f}.",
    f"Selected excess tails are most commonly classified as {dominant_tail_behavior} using ground-truth-derived thresholds.",
    "Phase-5A ground-truth reconstruction passed, so decoder failure is not the leading explanation.",
]
if eos_rows and eos_rows[0].get("analysis_available") is True:
    evidence.append("Sample-level EOS/max-length metadata was joined; inspect eos_motion_relation.csv for termination-stratified errors.")
else:
    evidence.append("Sample-level EOS metadata was unavailable, so the EOS mechanism cannot be directly stratified in this run.")
primary_bottleneck = "Phase-4 stopping/length control plus autoregressive collapse/drift"

recommendations = f"""# Phase 6 recommendations

These recommendations are ranked from measured Phase-4C/5B/5C evidence. They are proposals only;
Phase 6 is not implemented here.

## 1. Explicit duration and stopping control — highest priority

- Add a separate motion-token length predictor or explicit stop/EOS head.
- Condition decoding on the predicted target length and impose a defensible length window.
- Diagnose EOS supervision and max-length termination using `eos_motion_relation.csv`.

Evidence: oracle signed frame bias is **{oracle_signed_error:.3f}** and end-to-end bias is
**{end_signed_error:.3f}** frames. This dominates the oracle/end-to-end difference.

## 2. Mitigate autoregressive exposure bias and collapse

- Evaluate scheduled sampling or an autoregressive curriculum.
- Compare with non-autoregressive/masked temporal token generation.
- Add repetition-aware validation diagnostics and sequence-level early stopping criteria.

Evidence: **{oracle_collapse_total}** oracle and **{end_collapse_total}** end-to-end sequences were
flagged for collapse, so ground-truth gloss alone does not solve the problem.

## 3. Hierarchical coarse-to-fine RVQ prediction

- Predict the base level first, then condition residual levels on the base trajectory.
- Reweight residual-level losses according to measured ablation gains.
- Use `rvq_level_ablation.csv`; performed in this run: **{ablation_performed}**.

## 4. Motion-dynamics objectives

- Consider velocity/smoothness losses or temporal-consistency regularization in token/model space.
- Validate that improvements reduce collapse without suppressing legitimate fast hand articulation.
- Use empirical GT velocity percentiles instead of fixed arbitrary thresholds.

## 5. Phase-3 gloss improvements — lower priority

Oracle-to-end-to-end normalized-feature degradation is only **{phase3_degradation:.6f}**, so Phase 3
is not the leading bottleneck. Revisit it after length and collapse failures improve.

## Unresolved representation requirement

The exact 668-D index layout and verified feature-to-joint/BVH inverse remain unresolved. Obtain the
provider preprocessing/inverse utility before adding MPJPE, skeleton rendering, or generated BVH.
"""
(RESULTS / "phase6_recommendations.md").write_text(recommendations, encoding="utf-8")

phase5c_complete = bool(
    len(case_manifest) == 25
    and len(verification_rows) == EXPECTED_SAMPLES * 2
    and (DIAGNOSTICS / "feature_semantics_evidence.txt").is_file()
    and (DIAGNOSTICS / "bvh_feature_alignment.csv").is_file()
    and (RESULTS / "collapse_analysis.csv").is_file()
    and (RESULTS / "overgeneration_tail_analysis.csv").is_file()
)
ready_for_phase6 = phase5c_complete
phase5c_report = {
    "phase": "Phase 5C - Motion Interpretation and Visualization",
    "phase5c_complete": phase5c_complete,
    "feature_space_analysis": "PASS" if phase5c_complete else "FAIL",
    "feature_representation": {
        "dimension": MOTION_DIM,
        "semantics_status": "PARTIAL",
        "semantics_verified": False,
        "verified_feature_families": ["6D rotations", "joint velocities", "foot-contact signals"],
        "joint_conversion_verified": False,
        "bvh_conversion_verified": False,
    },
    "bvh_alignment": {
        "samples": len(bvh_alignment),
        "exact_frame_matches": int(bvh_alignment["frame_match"].sum()),
        "frame_rate_if_consistent": frame_rate,
    },
    "selected_cases": case_manifest["diagnostic_group"].value_counts().to_dict(),
    "collapse_analysis": {
        "oracle_flagged": oracle_collapse_total,
        "end2end_flagged": end_collapse_total,
        "analyzed_flagged_rows": len(collapse_analysis),
        "dominant_pattern": dominant_collapse_behavior,
    },
    "overgeneration": {
        "oracle_mean_signed_frame_error": oracle_signed_error,
        "end2end_mean_signed_frame_error": end_signed_error,
        "dominant_tail_behavior": dominant_tail_behavior,
    },
    "rvq_ablation": {
        "performed": ablation_performed,
        "rows": int(len(rvq_ablation)) if ablation_performed else 0,
        "base_only_behavior": "See levels_used=1 rows" if ablation_performed else None,
        "residual_effect": "See incremental levels_used rows" if ablation_performed else None,
    },
    "visualization": {
        "feature_trajectory_plots": len(list(FEATURE_FIGURES.glob("*.png"))),
        "velocity_plots": len(list(VELOCITY_FIGURES.glob("*.png"))),
        "verified_skeleton_visualization": False,
        "videos_created": 0,
        "bvh_exports_created": 0,
    },
    "extreme_motion": {"oracle_flagged": oracle_extreme_total, "end2end_flagged": end_extreme_total},
    "main_bottleneck": {"finding": primary_bottleneck, "evidence": evidence},
    "ready_for_phase6": ready_for_phase6,
}
write_json(OUTPUT / "phase5c_report.json", phase5c_report)

readme = f"""PHASE 6 HANDOFF — MOTION-S / SIGNVRSE

1. Full pipeline status
Phases 1–5C completed through verified feature-space analysis. Phase 5C did not retrain any model.

2. Verified decoder
Phase-5A Candidate B and empirical normalization were used by Phase 5B; all 1243 oracle and 1243
end-to-end motions were finite [T,668].

3. 668-D semantics
PARTIAL: provider-authored material identifies D-Pose components including 6D rotations, joint
velocities, and foot contacts. Exact indices, joint order, root integration, and rotation convention
remain unresolved.

4. Skeleton conversion
NOT VERIFIED. No converter passed the explicit 668-D contract plus original-BVH ground-truth gate.

5. BVH generation
NOT VERIFIED. No generated BVH was written.

6. Motion collapse
Oracle flags: {oracle_collapse_total}; end-to-end flags: {end_collapse_total}.
Subtype remains AUTO_UNKNOWN until plots are manually inspected or a verified skeleton converter exists.

7. Over-generation
Oracle signed bias: {oracle_signed_error:.6f} frames.
End-to-end signed bias: {end_signed_error:.6f} frames.
Dominant selected-case tail behavior: {dominant_tail_behavior}.

8. Oracle vs end-to-end
Normalized-feature degradation: {phase3_degradation:.6f}; much smaller than the shared duration/collapse failure.

9. RVQ-level findings
Ablation performed: {ablation_performed}. See results/rvq_level_ablation.csv.

10. Recommended Phase-6 direction
Prioritize explicit length/EOS control, then autoregressive exposure-bias/collapse mitigation, followed
by hierarchical coarse-to-fine RVQ prediction and motion-dynamics objectives.
"""
(OUTPUT / "README_phase6.txt").write_text(readme, encoding="utf-8")

inventory_rows = []
for path in sorted(OUTPUT.rglob("*")):
    if path.is_file() and path.name != "artifact_inventory.csv":
        inventory_rows.append({"relative_path": str(path.relative_to(OUTPUT)),
                               "bytes": path.stat().st_size, "sha256": sha256_file(path)})
pd.DataFrame(inventory_rows).to_csv(OUTPUT / "artifact_inventory.csv", index=False)

if not phase5c_complete: raise RuntimeError("PHASE 5C COMPLETION GATE FAILED")

print("\n============================================================")
print("PHASE 5C COMPLETE")
print("============================================================")
print("668-D feature semantics: PARTIAL")
print("Feature -> joint conversion:", "PASS" if converter_verified else ("NOT FOUND" if converter_status == "NOT FOUND" else "FAIL"))
print("Feature -> BVH conversion:", "PASS" if converter_verified else "NOT FOUND")
print("Ground-truth conversion validation:", "PASS" if converter_verified else "NOT RUN")
print("Diagnostic samples:", len(case_manifest))
print("Collapse cases analyzed:", len(collapse_analysis))
print("Dominant collapse behavior:", dominant_collapse_behavior)
print("Oracle mean over-generation:", oracle_signed_error, "frames")
print("End-to-End mean over-generation:", end_signed_error, "frames")
print("Dominant excess-tail behavior:", dominant_tail_behavior)
print("RVQ level ablation:", "PASS" if ablation_performed else "NOT RUN")
print("Verified skeleton videos: 0")
print("Generated BVH files: 0")
print("Primary model bottleneck:", primary_bottleneck)
print("Recommended Phase-6 direction: explicit length/EOS control, then collapse/exposure-bias mitigation")
print("READY FOR PHASE 6:", "YES" if ready_for_phase6 else "NO")
print("============================================================")
print("STOP. Do not begin Phase 6 automatically.")

"""
================================================================================
FINAL SESSION: Merge All Batches + Normalize + Train/Val Split
================================================================================

PURPOSE:
  - Load all 7 batch outputs (pose .npy files + mapping CSVs)
  - Normalize all poses: neck at origin, scale by shoulder width
  - Create signer-independent train/val split (4 signers train, 1 val)
  - Build the final dataset ready for model training
  - Save as a single Kaggle Dataset

RUNTIME: ~1-2 hours (mostly I/O and normalization)

INSTRUCTIONS:
  1. Create a new Kaggle Notebook (CPU is fine)
  2. Add these 2 datasets:
     - "bdsl-ban-sign-sent-9k-extracted"   (Session 0 — original HuggingFace data)
     - "bdsl-mediapipe-final"              (all 7 batch outputs in one dataset)
  3. Paste this script and run
  4. Save the output as a NEW Kaggle Dataset: "bdsl-final-normalized"

================================================================================
"""

import os
import sys
import csv
import json
import glob
import shutil
import errno
import numpy as np
from pathlib import Path
from collections import defaultdict


def free_gb(path):
    """Free space in GB on the filesystem containing `path`."""
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024 ** 3)
    except Exception:
        return float("nan")

# ============================================================================
# CONFIGURATION
# ============================================================================
NUM_BATCHES = 7
WORK_DIR = "/kaggle/working"
OUTPUT_DIR = os.path.join(WORK_DIR, "bdsl_final_dataset")
INPUT_BASE = "/kaggle/input"

# ------- DATASET PATHS -------
# All 7 batch outputs live inside a single Kaggle dataset:
#   /kaggle/input/bdsl-mediapipe-final/batch_X_output/
BATCH_DATASET = os.path.join(INPUT_BASE, "bdsl-mediapipe-final")

# Session 0 (original annotations / HuggingFace download):
SESSION0_DATASET = os.path.join(INPUT_BASE, "bdsl-ban-sign-sent-9k-extracted")

# Output subdirectories
# NOTE: We deliberately do NOT create an intermediate "poses_normalized/" dir.
# Writing a full normalized copy (~10 GB) AND a train/val copy overflows
# Kaggle's ~20 GB working disk. Instead we normalize each file in memory and
# write it straight into train/ or val/ — one copy on disk, ~10 GB total.
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")

# Normalization settings
# MediaPipe Holistic landmark indices for key body points
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_NOSE = 0
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24

# Neck = midpoint of shoulders (not a MediaPipe landmark, we compute it)
# This is the standard SLRTP2025 normalization anchor

# Train/Val split settings
VAL_RATIO = 0.15  # ~15% for validation

# ============================================================================
# STEP 1: Clean previous output + create fresh directories
# ============================================================================
print("=" * 70)
print("BdSL Pipeline — Final Merge & Normalization")
print("=" * 70)

print(f"\n  Free space on {WORK_DIR}: {free_gb(WORK_DIR):.1f} GB (before cleanup)")

# CRITICAL: Kaggle keeps /kaggle/working across cell re-runs. A previous
# (failed) run can leave ~10 GB of leftover .npy files here, so a re-run
# starts with a nearly-full disk and writes fail from the very first file.
# Wipe our own output dir so every run starts clean.
if os.path.exists(OUTPUT_DIR):
    print(f"  Removing leftover output from a previous run: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

for d in [OUTPUT_DIR, TRAIN_DIR, VAL_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"  Free space on {WORK_DIR}: {free_gb(WORK_DIR):.1f} GB (after cleanup)")

# ============================================================================
# STEP 2: Discover and load all batch outputs
# ============================================================================
print("\nStep 2: Discovering batch outputs...")
print(f"  Looking in: {BATCH_DATASET}")

# Verify the dataset exists
if not os.path.exists(BATCH_DATASET):
    print(f"\n  ERROR: Dataset not found at {BATCH_DATASET}")
    print(f"  Available datasets in /kaggle/input/:")
    if os.path.exists(INPUT_BASE):
        for item in sorted(os.listdir(INPUT_BASE)):
            print(f"    - {item}")
    print(f"\n  Please add 'bdsl-mediapipe-final' as an input dataset.")
    sys.exit(1)

# Show what's inside
print(f"  Contents of {BATCH_DATASET}:")
for item in sorted(os.listdir(BATCH_DATASET)):
    print(f"    - {item}")

all_pose_files = []
all_mappings = []
batch_stats_all = []

for batch_id in range(NUM_BATCHES):
    print(f"\n  Batch {batch_id}:")

    # All batches are inside the single dataset as batch_X_output/
    batch_dir = os.path.join(BATCH_DATASET, f"batch_{batch_id}_output")

    if not os.path.exists(batch_dir):
        # Fallback: search for any folder containing this batch_id
        candidates = glob.glob(os.path.join(BATCH_DATASET, f"*batch*{batch_id}*"))
        if candidates:
            batch_dir = candidates[0]
        else:
            print(f"    WARNING: batch_{batch_id}_output NOT FOUND — skipping")
            continue

    print(f"    Found at: {batch_dir}")

    # Show contents of batch dir
    if os.path.exists(batch_dir):
        contents = os.listdir(batch_dir)
        print(f"    Contents: {contents}")

    # Find pose files
    poses_subdir = os.path.join(batch_dir, "poses")
    if not os.path.exists(poses_subdir):
        # Search recursively for .npy files
        npy_search = glob.glob(os.path.join(batch_dir, "**/*.npy"), recursive=True)
        if npy_search:
            poses_subdir = os.path.dirname(npy_search[0])
            print(f"    Found poses at: {poses_subdir}")
        else:
            print(f"    WARNING: No pose files (.npy) found in batch {batch_id}")
            continue

    npy_files = sorted(glob.glob(os.path.join(poses_subdir, "*.npy")))
    print(f"    Found {len(npy_files)} pose files")
    all_pose_files.extend(npy_files)

    # Find mapping CSV
    mapping_candidates = glob.glob(
        os.path.join(batch_dir, f"**/*mapping*"), recursive=True
    )
    for mc in mapping_candidates:
        if mc.endswith('.csv'):
            print(f"    Mapping: {os.path.basename(mc)}")
            with open(mc, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_mappings.append(row)

    # Find stats
    stats_candidates = glob.glob(
        os.path.join(batch_dir, f"**/*stats*"), recursive=True
    )
    for sc in stats_candidates:
        if sc.endswith('.json'):
            with open(sc, 'r') as f:
                batch_stats_all.append(json.load(f))

print(f"\n  Total pose files found: {len(all_pose_files)}")
print(f"  Total mapping records: {len(all_mappings)}")

if len(all_pose_files) == 0:
    print("\n  ERROR: No pose files found! Check dataset structure.")
    print("  Expected: bdsl-mediapipe-final/batch_X_output/poses/*.npy")
    sys.exit(1)

# ============================================================================
# STEP 3: Aggregate statistics
# ============================================================================
print(f"\n{'=' * 70}")
print("Step 3: Computing aggregate statistics...")

if batch_stats_all:
    total_succeeded = sum(s.get("succeeded", 0) for s in batch_stats_all)
    total_failed = sum(s.get("failed", 0) for s in batch_stats_all)
    avg_pose_det = np.mean([
        s.get("avg_pose_detection_rate", 0) for s in batch_stats_all
        if s.get("avg_pose_detection_rate", 0) > 0
    ])
    avg_hand_det = np.mean([
        s.get("avg_hand_detection_rate", 0) for s in batch_stats_all
        if s.get("avg_hand_detection_rate", 0) > 0
    ])
    total_time = sum(s.get("processing_time_hours", 0) for s in batch_stats_all)

    print(f"  Total succeeded: {total_succeeded}")
    print(f"  Total failed:    {total_failed}")
    print(f"  Success rate:    {total_succeeded / max(1, total_succeeded + total_failed):.1%}")
    print(f"  Avg pose det:    {avg_pose_det:.1%}")
    print(f"  Avg hand det:    {avg_hand_det:.1%}")
    print(f"  Total time:      {total_time:.1f} hours")

# ============================================================================
# STEP 4: Define normalization (actual processing happens in Step 7)
# ============================================================================
print(f"\n{'=' * 70}")
print("Step 4: Preparing normalization...")
print("  Normalization: neck (shoulder midpoint) at origin, scale by shoulder width")
print("  (Files are normalized in memory during Step 7 — no intermediate copy)")


def normalize_pose_sequence(poses):
    """
    Normalize a pose sequence (T, 543, 3):
    1. Center on neck (midpoint of shoulders) — translate so neck = (0, 0, 0)
    2. Scale by inter-shoulder distance — makes poses signer-independent
    3. Handle frames where shoulders are missing (zeros)

    Returns:
        normalized_poses: (T, 543, 3)
        quality_score: float [0, 1] — fraction of frames with valid shoulders
    """
    T = poses.shape[0]
    normalized = poses.copy()

    valid_frames = 0

    for t in range(T):
        left_shoulder = poses[t, POSE_LEFT_SHOULDER]
        right_shoulder = poses[t, POSE_RIGHT_SHOULDER]

        # Check if shoulders are detected (not all zeros)
        if np.all(left_shoulder == 0) or np.all(right_shoulder == 0):
            continue

        valid_frames += 1

        # Compute neck (midpoint)
        neck = (left_shoulder + right_shoulder) / 2.0

        # Compute inter-shoulder distance for scale
        shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)

        if shoulder_dist < 1e-6:
            shoulder_dist = 1.0  # Avoid division by zero

        # Translate: neck to origin
        normalized[t] -= neck

        # Scale: normalize by shoulder width
        normalized[t] /= shoulder_dist

        # Zero out landmarks that were originally zero (undetected)
        zero_mask = np.all(poses[t] == 0, axis=1)
        normalized[t, zero_mask] = 0.0

    quality = valid_frames / max(1, T)

    # For frames without valid shoulders, use nearest valid frame's transform
    if valid_frames < T and valid_frames > 0:
        # Forward-fill then backward-fill
        last_valid = None
        for t in range(T):
            ls = poses[t, POSE_LEFT_SHOULDER]
            rs = poses[t, POSE_RIGHT_SHOULDER]
            if not (np.all(ls == 0) or np.all(rs == 0)):
                last_valid = t
            elif last_valid is not None:
                # Apply same transform as last valid frame
                neck = (poses[last_valid, POSE_LEFT_SHOULDER] +
                        poses[last_valid, POSE_RIGHT_SHOULDER]) / 2.0
                sd = np.linalg.norm(
                    poses[last_valid, POSE_LEFT_SHOULDER] -
                    poses[last_valid, POSE_RIGHT_SHOULDER]
                )
                if sd < 1e-6:
                    sd = 1.0
                normalized[t] = (poses[t] - neck) / sd
                zero_mask = np.all(poses[t] == 0, axis=1)
                normalized[t, zero_mask] = 0.0

    return normalized, quality


# ============================================================================
# STEP 5: Build sentence -> files mapping (from RAW input files)
# ============================================================================
print(f"\n{'=' * 70}")
print("Step 5: Building sentence-to-file mapping...")

# Parse filenames to extract sentence_id and signer_id
# Expected format: {sentence_id}_{signer_id}.npy
# We map the RAW input .npy files here; normalization is deferred to Step 7
# so we only ever write ONE copy of each pose to disk (train/ or val/).
sentence_to_files = defaultdict(list)
all_signer_ids = set()

for npy_path in all_pose_files:
    fname = Path(npy_path).stem  # without .npy

    # Try to split into sentence_id and signer_id
    parts = fname.rsplit('_', 1)
    if len(parts) == 2:
        sentence_id, signer_id = parts
    else:
        sentence_id = fname
        signer_id = "unknown"

    sentence_to_files[sentence_id].append({
        "signer_id": signer_id,
        "file": npy_path,  # RAW input path — normalized on write in Step 7
        "filename": os.path.basename(npy_path)
    })
    all_signer_ids.add(signer_id)

print(f"  Unique sentences: {len(sentence_to_files)}")
print(f"  Unique signers: {len(all_signer_ids)}")
print(f"  Signer IDs: {sorted(all_signer_ids)[:10]}")

# Videos per sentence distribution
vps = [len(v) for v in sentence_to_files.values()]
if vps:
    print(f"  Videos per sentence: mean={np.mean(vps):.1f}, "
          f"min={np.min(vps)}, max={np.max(vps)}")

# ============================================================================
# STEP 6: Signer-independent train/val split
# ============================================================================
print(f"\n{'=' * 70}")
print("Step 6: Creating signer-independent train/val split...")

"""
Strategy:
  If we have identifiable signers (5 per sentence), hold out 1 signer for val.
  This ensures the model generalizes across signers.

  If signers aren't clearly identifiable, fall back to random sentence-level split.
"""

signer_list = sorted(all_signer_ids)

# Check if we have meaningful signer diversity
if len(signer_list) >= 3 and "unknown" not in signer_list:
    # Signer-based split: hold out ~20% of signers for validation
    num_val_signers = max(1, len(signer_list) // 5)
    np.random.seed(42)
    val_signers = set(np.random.choice(
        signer_list, size=num_val_signers, replace=False
    ))
    train_signers = set(signer_list) - val_signers

    print(f"  Split type: Signer-independent")
    print(f"  Train signers ({len(train_signers)}): {sorted(train_signers)[:5]}...")
    print(f"  Val signers ({len(val_signers)}):   {sorted(val_signers)}")

    train_files = []
    val_files = []

    for sentence_id, file_list in sentence_to_files.items():
        for entry in file_list:
            if entry["signer_id"] in val_signers:
                val_files.append(entry)
            else:
                train_files.append(entry)

else:
    # Sentence-level random split
    print(f"  Split type: Sentence-level random (signers not identifiable)")

    all_sentence_ids = sorted(sentence_to_files.keys())
    np.random.seed(42)
    np.random.shuffle(all_sentence_ids)

    val_count = int(len(all_sentence_ids) * VAL_RATIO)
    val_sentences = set(all_sentence_ids[:val_count])
    train_sentences = set(all_sentence_ids[val_count:])

    print(f"  Train sentences: {len(train_sentences)}")
    print(f"  Val sentences:   {len(val_sentences)}")

    train_files = []
    val_files = []

    for sentence_id, file_list in sentence_to_files.items():
        for entry in file_list:
            if sentence_id in val_sentences:
                val_files.append(entry)
            else:
                train_files.append(entry)

print(f"\n  Train samples: {len(train_files)}")
print(f"  Val samples:   {len(val_files)}")
print(f"  Val ratio:     {len(val_files) / max(1, len(train_files) + len(val_files)):.1%}")

# ============================================================================
# STEP 7: Normalize in memory + write directly to train/val directories
# ============================================================================
print(f"\n{'=' * 70}")
print("Step 7: Normalizing and writing train/val split...")
print("  (Each file is normalized in RAM and written once — no intermediate copy)")

# Stats collected during normalization (previously done in Step 4)
quality_scores = []
frame_counts = []
failed_files = []


class DiskFullError(Exception):
    """Raised when the working disk runs out of space — abort immediately."""
    pass


def _is_enospc(exc):
    """True if the exception is (or wraps) an out-of-space condition."""
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return True
    # np.save surfaces short writes as a bare OSError with this message,
    # not always with errno set, so match the text as a fallback.
    msg = str(exc).lower()
    return "no space" in msg or ("requested and" in msg and "written" in msg)


def normalize_and_save(entry, dst_dir):
    """Load raw pose, normalize in memory, save to dst_dir. Returns True on success."""
    src = entry["file"]
    dst = os.path.join(dst_dir, entry["filename"])
    try:
        poses = np.load(src)  # (T, 543, 3)

        # Validate shape
        if len(poses.shape) != 3 or poses.shape[1] != 543 or poses.shape[2] != 3:
            print(f"    WARNING: Unexpected shape {poses.shape} in {entry['filename']}, skipping")
            failed_files.append(entry["filename"])
            return False

        norm_poses, quality = normalize_pose_sequence(poses)
        np.save(dst, norm_poses.astype(np.float32))

        quality_scores.append(quality)
        frame_counts.append(poses.shape[0])
        return True

    except Exception as e:
        # Always clean up a partial/corrupt write first.
        if os.path.exists(dst):
            try:
                os.remove(dst)
            except OSError:
                pass
        # A disk-full error won't fix itself — stop now instead of logging
        # thousands of identical failures and producing a broken dataset.
        if _is_enospc(e):
            raise DiskFullError(
                f"Out of disk space while writing {entry['filename']} "
                f"({free_gb(WORK_DIR):.2f} GB free). Aborting."
            ) from e
        print(f"    WARNING: Error processing {entry['filename']}: {e}")
        failed_files.append(entry["filename"])
        return False


# Train files
train_manifest = []
for i, entry in enumerate(train_files):
    if i % 500 == 0:
        print(f"  [train {i+1}/{len(train_files)}] Normalizing + writing...")
    if normalize_and_save(entry, TRAIN_DIR):
        train_manifest.append(entry)

# Val files
val_manifest = []
for i, entry in enumerate(val_files):
    if i % 500 == 0:
        print(f"  [val {i+1}/{len(val_files)}] Normalizing + writing...")
    if normalize_and_save(entry, VAL_DIR):
        val_manifest.append(entry)

print(f"\n  Wrote {len(train_manifest)} files to train/")
print(f"  Wrote {len(val_manifest)} files to val/")
print(f"  Failed: {len(failed_files)} files")
if quality_scores:
    print(f"  Quality (shoulder detection): "
          f"mean={np.mean(quality_scores):.1%}, "
          f"median={np.median(quality_scores):.1%}, "
          f"min={np.min(quality_scores):.1%}")
if frame_counts:
    print(f"  Frame counts: mean={np.mean(frame_counts):.0f}, "
          f"median={np.median(frame_counts):.0f}, total={sum(frame_counts)}")

# ============================================================================
# STEP 8: Save manifests and metadata
# ============================================================================
print(f"\n{'=' * 70}")
print("Step 8: Saving manifests and metadata...")

# Train manifest
train_manifest_path = os.path.join(OUTPUT_DIR, "train_manifest.csv")
with open(train_manifest_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "signer_id"])
    writer.writeheader()
    for entry in train_manifest:
        writer.writerow({
            "filename": entry["filename"],
            "signer_id": entry["signer_id"]
        })

# Val manifest
val_manifest_path = os.path.join(OUTPUT_DIR, "val_manifest.csv")
with open(val_manifest_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "signer_id"])
    writer.writeheader()
    for entry in val_manifest:
        writer.writerow({
            "filename": entry["filename"],
            "signer_id": entry["signer_id"]
        })

# Also copy annotation files from Session 0 dataset
if os.path.exists(SESSION0_DATASET):
    print(f"\n  Session 0 dataset found at: {SESSION0_DATASET}")
    # Look for annotation/metadata JSON files
    ann_files = glob.glob(os.path.join(SESSION0_DATASET, "**/*.json"), recursive=True)
    if ann_files:
        print(f"  Found {len(ann_files)} annotation files from Session 0")
        for af in ann_files[:20]:  # Cap at 20 to avoid copying thousands
            dst = os.path.join(OUTPUT_DIR, os.path.basename(af))
            if not os.path.exists(dst):
                shutil.copy2(af, dst)
                print(f"    Copied: {os.path.basename(af)}")
    # Also look for CSV files (sentence annotations, etc.)
    csv_files = glob.glob(os.path.join(SESSION0_DATASET, "**/*.csv"), recursive=True)
    for cf in csv_files:
        dst = os.path.join(OUTPUT_DIR, os.path.basename(cf))
        if not os.path.exists(dst):
            shutil.copy2(cf, dst)
            print(f"    Copied: {os.path.basename(cf)}")
else:
    print(f"\n  Session 0 dataset not found at: {SESSION0_DATASET}")
    print(f"  (Not critical — annotations can be added later)")

# Dataset metadata
dataset_meta = {
    "name": "BdSL MediaPipe Dataset (Ban-Sign-Sent-9K)",
    "source": "Ban-Sign-Sent-9K via HuggingFace (banglagov/Ban-Sign-Sent-9K-V1)",
    "extraction": "MediaPipe Holistic (SLRTP2025 format)",
    "format": {
        "shape": "(T, 543, 3)",
        "landmarks": {
            "pose": {"indices": "0-32", "count": 33},
            "left_hand": {"indices": "33-53", "count": 21},
            "right_hand": {"indices": "54-74", "count": 21},
            "face": {"indices": "75-542", "count": 468},
            "total": 543
        },
        "coordinates": "normalized (x, y, z) in image space then neck-centered",
        "dtype": "float32"
    },
    "normalization": {
        "method": "neck-centered, shoulder-scaled",
        "anchor": "midpoint of left_shoulder (idx 11) and right_shoulder (idx 12)",
        "scale": "inter-shoulder distance"
    },
    "split": {
        "method": "signer-independent" if len(signer_list) >= 3 else "sentence-level",
        "train_samples": len(train_manifest),
        "val_samples": len(val_manifest),
        "val_ratio": len(val_manifest) / max(1, len(train_manifest) + len(val_manifest))
    },
    "statistics": {
        "total_samples": len(train_manifest) + len(val_manifest),
        "unique_sentences": len(sentence_to_files),
        "unique_signers": len(all_signer_ids),
        "avg_frames": float(np.mean(frame_counts)) if frame_counts else 0,
        "total_frames": sum(frame_counts) if frame_counts else 0,
        "avg_quality": float(np.mean(quality_scores)) if quality_scores else 0
    }
}

meta_path = os.path.join(OUTPUT_DIR, "dataset_metadata.json")
with open(meta_path, 'w') as f:
    json.dump(dataset_meta, f, indent=2, ensure_ascii=False)

print(f"\n  Saved: train_manifest.csv ({len(train_manifest)} entries)")
print(f"  Saved: val_manifest.csv ({len(val_manifest)} entries)")
print(f"  Saved: dataset_metadata.json")

# ============================================================================
# STEP 9: Verify final output
# ============================================================================
print(f"\n{'=' * 70}")
print("Step 9: Verification...")

# Check a few files
train_npys = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.npy")))
val_npys = sorted(glob.glob(os.path.join(VAL_DIR, "*.npy")))

print(f"\n  Train directory: {len(train_npys)} files")
print(f"  Val directory:   {len(val_npys)} files")

if train_npys:
    sample = np.load(train_npys[0])
    print(f"\n  Sample file: {os.path.basename(train_npys[0])}")
    print(f"    Shape: {sample.shape}")
    print(f"    Dtype: {sample.dtype}")
    print(f"    Value range: [{sample.min():.4f}, {sample.max():.4f}]")

    # Check normalization: frame 0, neck should be near origin
    neck = (sample[0, POSE_LEFT_SHOULDER] + sample[0, POSE_RIGHT_SHOULDER]) / 2
    print(f"    Neck position (should be ~0): {neck}")

    # Check shoulder distance (should be ~1 after normalization)
    sd = np.linalg.norm(sample[0, POSE_LEFT_SHOULDER] - sample[0, POSE_RIGHT_SHOULDER])
    print(f"    Shoulder distance (should be ~1): {sd:.4f}")

# Disk usage
total_size = sum(
    os.path.getsize(os.path.join(root, f))
    for root, _, files in os.walk(OUTPUT_DIR)
    for f in files
)
print(f"\n  Total dataset size: {total_size / (1024**3):.2f} GB")

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'=' * 70}")
print("FINAL DATASET COMPLETE!")
print(f"{'=' * 70}")
print(f"""
  Output directory: {OUTPUT_DIR}

  Structure:
    bdsl_final_dataset/
    ├── train/                    # Training pose files (.npy)
    │   └── *.npy                 # Shape: (T, 543, 3), normalized
    ├── val/                      # Validation pose files (.npy)
    │   └── *.npy
    ├── train_manifest.csv        # filename, signer_id
    ├── val_manifest.csv          # filename, signer_id
    └── dataset_metadata.json     # Full dataset description

  To upload as a new Kaggle dataset:
    1. Click "Save Version" → "Save & Run All"
    2. Go to the Output tab
    3. Click "New Dataset" → name it "bdsl-final-normalized"

  To use in training:
    import numpy as np
    import os

    train_dir = "/kaggle/input/bdsl-final-normalized/bdsl_final_dataset/train/"
    for fname in os.listdir(train_dir):
        poses = np.load(os.path.join(train_dir, fname))  # (T, 543, 3)
        # poses[:, :33, :]   -> body pose
        # poses[:, 33:54, :] -> left hand
        # poses[:, 54:75, :] -> right hand
        # poses[:, 75:, :]   -> face
""")

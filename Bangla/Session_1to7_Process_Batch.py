"""
================================================================================
SESSIONS 1-7: MediaPipe Holistic Extraction — Batch Processing
================================================================================

PURPOSE:
  - Process one batch of ~1,373 videos with MediaPipe Holistic
  - Extract 543 landmarks (33 pose + 21×2 hands + 468 face) per frame
  - Save as numpy arrays with shape (T, 543, 3)
  - Handle failures gracefully with checkpointing

RUNTIME: ~8-10 hours per batch (within Kaggle's 12-hour limit)

INSTRUCTIONS:
  1. Create a new Kaggle Notebook (CPU accelerator — GPU not needed)
  2. Add the dataset you created in Session 0:
     "bdsl-ban-sign-sent-9k-extracted" → should mount at
     /kaggle/input/bdsl-ban-sign-sent-9k-extracted/
  3. *** CHANGE BATCH_ID below to match your session number (0-6) ***
  4. Paste this entire script and run
  5. After completion, download the output from /kaggle/working/batch_X_output/
  6. Upload it as a new Kaggle Dataset: "bdsl-batch-X-mediapipe"
  7. Repeat for each batch (Sessions 1 through 7 = BATCH_ID 0 through 6)

================================================================================
"""

import os
import sys
import subprocess
import csv
import json
import time
import traceback
import urllib.request
from pathlib import Path

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  ⚠️  CHANGE THIS FOR EACH SESSION  ⚠️                                    ║
# ║                                                                          ║
# ║  Session 1 → BATCH_ID = 0                                               ║
# ║  Session 2 → BATCH_ID = 1                                               ║
# ║  Session 3 → BATCH_ID = 2                                               ║
# ║  Session 4 → BATCH_ID = 3                                               ║
# ║  Session 5 → BATCH_ID = 4                                               ║
# ║  Session 6 → BATCH_ID = 5                                               ║
# ║  Session 7 → BATCH_ID = 6                                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝
BATCH_ID = 0  # ← CHANGE THIS (0 to 6)

# ============================================================================
# CONFIGURATION
# ============================================================================
# Input path — the batch CSV and videos are located automatically anywhere
# under this folder (Kaggle may nest the upload, e.g. .../bdsl_dataset/...)
INPUT_ROOT = "/kaggle/input"

# Output paths
WORK_DIR = "/kaggle/working"
OUTPUT_DIR = os.path.join(WORK_DIR, f"batch_{BATCH_ID}_output")
POSES_DIR = os.path.join(OUTPUT_DIR, "poses")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")
LOG_FILE = os.path.join(OUTPUT_DIR, "processing_log.csv")
STATS_FILE = os.path.join(OUTPUT_DIR, "batch_stats.json")

# Processing parameters
TARGET_FPS = 25          # Target frame rate for extraction
MIN_FRAMES = 5           # Skip videos shorter than this
MAX_FRAMES = 1500        # Truncate videos longer than this (60s at 25fps)

# MediaPipe settings (Tasks API HolisticLandmarker)
MP_MIN_DETECTION_CONF = 0.5
MP_MIN_TRACKING_CONF = 0.5   # used for the landmark-confidence thresholds
MP_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
                "holistic_landmarker/float16/latest/holistic_landmarker.task")
MP_MODEL_PATH = os.path.join(WORK_DIR, "holistic_landmarker.task")

# Checkpointing — save progress every N videos
CHECKPOINT_INTERVAL = 50

# ============================================================================
# STEP 1: Install MediaPipe
# ============================================================================
print("=" * 70)
print(f"BdSL MediaPipe Pipeline — Batch {BATCH_ID}")
print("=" * 70)

print("\nStep 1: Installing MediaPipe...")
# mediapipe >= 0.10.30 dropped the legacy mp.solutions.holistic API, and the
# last release that had it (0.10.21) has no wheel for Kaggle's Python 3.13.
# So we use the Tasks API HolisticLandmarker, which outputs the same landmarks.
# Its own opencv-contrib-python dependency provides cv2 — don't add opencv-python-headless.
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "mediapipe==0.10.35"
])

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HolisticLandmarker, HolisticLandmarkerOptions, RunningMode
)

if not os.path.exists(MP_MODEL_PATH):
    print(f"  Downloading model: {MP_MODEL_URL}")
    urllib.request.urlretrieve(MP_MODEL_URL, MP_MODEL_PATH)

print(f"  Python: {sys.version.split()[0]}  NumPy: {np.__version__}")
print(f"  MediaPipe version: {mp.__version__}")
print(f"  OpenCV version: {cv2.__version__}")

# ============================================================================
# STEP 2: Setup directories and load batch
# ============================================================================
print(f"\nStep 2: Loading batch {BATCH_ID}...")

os.makedirs(POSES_DIR, exist_ok=True)

# One walk over /kaggle/input: find the batch CSV and index every mp4 by filename
batch_csv_path = None
video_index = {}
for root, dirs, files in os.walk(INPUT_ROOT):
    for f in files:
        if f == f"batch_{BATCH_ID}.csv" and batch_csv_path is None:
            batch_csv_path = os.path.join(root, f)
        elif f.endswith(".mp4"):
            video_index[f] = os.path.join(root, f)

if batch_csv_path is None:
    print(f"\n  Contents of {INPUT_ROOT}:")
    for root, dirs, files in os.walk(INPUT_ROOT):
        depth = root.replace(INPUT_ROOT, '').count(os.sep)
        if depth < 4:
            indent = '  ' * (depth + 2)
            print(f"{indent}{os.path.basename(root)}/")
            for f in sorted(files)[:10]:
                print(f"{indent}  {f}")
    raise FileNotFoundError(
        f"Cannot find batch_{BATCH_ID}.csv under {INPUT_ROOT}. "
        f"Did you add the bdsl-ban-sign-sent-9k-extracted dataset as input?"
    )
print(f"  Batch CSV: {batch_csv_path}")
print(f"  Indexed {len(video_index)} .mp4 files under {INPUT_ROOT}")

# Load batch
video_records = []
with open(batch_csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        video_records.append(row)

print(f"  Loaded {len(video_records)} videos from batch {BATCH_ID}")

# Pre-flight: fail fast if the videos aren't actually in the input dataset
missing = [r["filename"] for r in video_records if r["filename"] not in video_index]
print(f"  Videos found: {len(video_records) - len(missing)}/{len(video_records)}")
if len(missing) == len(video_records):
    raise FileNotFoundError(
        "None of this batch's videos exist under /kaggle/input. The Session 0 "
        "dataset probably contains only metadata — the videos/ folder must be uploaded too."
    )
if missing:
    print(f"  WARNING: {len(missing)} videos missing, e.g. {missing[:3]}")

# Load checkpoint if resuming
processed_set = set()
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, 'r') as f:
        checkpoint = json.load(f)
    processed_set = set(checkpoint.get("processed", []))
    print(f"  Resuming: {len(processed_set)} already processed")

# ============================================================================
# STEP 3: MediaPipe extraction functions
# ============================================================================
print("\nStep 3: Initializing MediaPipe Holistic...")



def create_holistic():
    # VIDEO mode tracks across frames; timestamps must strictly increase,
    # so a fresh landmarker is created per video.
    return HolisticLandmarker.create_from_options(HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MP_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        min_face_detection_confidence=MP_MIN_DETECTION_CONF,
        min_pose_detection_confidence=MP_MIN_DETECTION_CONF,
        min_face_landmarks_confidence=MP_MIN_TRACKING_CONF,
        min_pose_landmarks_confidence=MP_MIN_TRACKING_CONF,
        min_hand_landmarks_confidence=MP_MIN_TRACKING_CONF,
        output_face_blendshapes=False,
        output_segmentation_mask=False,
    ))


def extract_landmarks_from_video(video_path, target_fps=TARGET_FPS):
    """
    Extract MediaPipe Holistic landmarks from a video file.

    Returns:
        poses: np.ndarray of shape (T, 543, 3) or None if failed
        metadata: dict with extraction info
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, {"error": "Cannot open video"}

    # Get video properties
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Many Ban-Sign-Sent-9K clips are VP8 with variable frame rate, for which
    # OpenCV can report fps/frame-count of 0 or nonsense. So sample by each
    # frame's timestamp (CAP_PROP_POS_MSEC), not by a fixed frame interval.
    sample_step_ms = 1000.0 / target_fps
    next_sample_ms = 0.0
    fps_is_sane = 1.0 <= orig_fps <= 120.0

    # Initialize MediaPipe Holistic
    with create_holistic() as holistic:

        all_landmarks = []
        last_ts = -1
        frame_idx = 0
        frames_read = 0
        frames_with_pose = 0
        frames_with_hands = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Timestamp of this frame; fall back to index/fps if the container has none
            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if t_ms <= 0 and frame_idx > 0:
                t_ms = frame_idx * 1000.0 / (orig_fps if fps_is_sane else target_fps)
            frame_idx += 1

            # Sample frames at target FPS (frames closer than 1/target_fps are skipped)
            if t_ms + 1e-3 < next_sample_ms:
                continue
            next_sample_ms += sample_step_ms
            if next_sample_ms <= t_ms:          # gap in VFR stream — resync
                next_sample_ms = t_ms + sample_step_ms
            frames_read += 1

            # Enforce max frames
            if frames_read > MAX_FRAMES:
                break

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Process with MediaPipe (timestamp must be strictly increasing)
            ts = max(int(t_ms), last_ts + 1)
            last_ts = ts
            results = holistic.detect_for_video(mp_image, ts)

            # Build 543-landmark array for this frame
            # Layout: [33 pose, 21 left_hand, 21 right_hand, 468 face]
            frame_landmarks = np.zeros((543, 3), dtype=np.float32)

            # 1. Pose landmarks (33)
            if results.pose_landmarks:
                frames_with_pose += 1
                for i, lm in enumerate(results.pose_landmarks[:33]):
                    frame_landmarks[i] = [lm.x, lm.y, lm.z]

            # 2. Left hand landmarks (21) — indices 33-53
            if results.left_hand_landmarks or results.right_hand_landmarks:
                frames_with_hands += 1
            if results.left_hand_landmarks:
                for i, lm in enumerate(results.left_hand_landmarks[:21]):
                    frame_landmarks[33 + i] = [lm.x, lm.y, lm.z]

            # 3. Right hand landmarks (21) — indices 54-74
            if results.right_hand_landmarks:
                for i, lm in enumerate(results.right_hand_landmarks[:21]):
                    frame_landmarks[54 + i] = [lm.x, lm.y, lm.z]

            # 4. Face landmarks (468) — indices 75-542
            if results.face_landmarks:
                for i, lm in enumerate(results.face_landmarks[:468]):
                    frame_landmarks[75 + i] = [lm.x, lm.y, lm.z]

            all_landmarks.append(frame_landmarks)

    cap.release()

    if len(all_landmarks) < MIN_FRAMES:
        return None, {
            "error": f"Too few frames: {len(all_landmarks)} < {MIN_FRAMES}",
            "frames_read": frames_read
        }

    poses = np.array(all_landmarks, dtype=np.float32)  # (T, 543, 3)

    metadata = {
        "original_fps": orig_fps,
        "target_fps": target_fps,
        "original_frames": total_frames,
        "extracted_frames": len(all_landmarks),
        "width": width,
        "height": height,
        "frames_with_pose": frames_with_pose,
        "frames_with_hands": frames_with_hands,
        "pose_detection_rate": frames_with_pose / max(1, len(all_landmarks)),
        "hand_detection_rate": frames_with_hands / max(1, len(all_landmarks))
    }

    return poses, metadata


def interpolate_missing_landmarks(poses):
    """
    Fill in frames where landmarks are all zeros (detection failed)
    using linear interpolation from neighboring frames.

    Input/Output: np.ndarray of shape (T, 543, 3)
    """
    T = poses.shape[0]
    if T < 2:
        return poses

    for landmark_idx in range(543):
        # Check which frames have valid data (not all zeros)
        valid = np.any(poses[:, landmark_idx, :] != 0, axis=1)

        if not np.any(valid):
            continue  # This landmark never detected

        if np.all(valid):
            continue  # No gaps

        # Get indices of valid frames
        valid_indices = np.where(valid)[0]

        # Interpolate each coordinate
        for coord in range(3):
            values = poses[:, landmark_idx, coord]
            valid_values = values[valid_indices]

            # Interpolate
            interpolated = np.interp(
                np.arange(T),
                valid_indices,
                valid_values
            )

            # Only fill the gaps (keep original valid data)
            missing = ~valid
            poses[missing, landmark_idx, coord] = interpolated[missing]

    return poses


# ============================================================================
# STEP 4: Process all videos in batch
# ============================================================================
print(f"\nStep 4: Processing {len(video_records)} videos...")
print(f"  Target FPS: {TARGET_FPS}")
print(f"  Max frames per video: {MAX_FRAMES}")
print(f"  Checkpoint every {CHECKPOINT_INTERVAL} videos\n")

# Initialize log
log_records = []
stats = {
    "batch_id": BATCH_ID,
    "total_videos": len(video_records),
    "processed": 0,
    "succeeded": 0,
    "failed": 0,
    "skipped": 0,
    "total_frames": 0,
    "avg_frames": 0,
    "avg_pose_detection_rate": 0,
    "avg_hand_detection_rate": 0,
    "processing_time_seconds": 0
}

detection_rates_pose = []
detection_rates_hand = []
frame_counts = []

start_time = time.time()

for idx, record in enumerate(video_records):
    video_idx = record.get("video_idx", str(idx))
    video_path = record["video_path"]
    sentence_id = record.get("sentence_id", "unknown")
    signer_id = record.get("signer_id", "unknown")

    # Skip if already processed
    if video_idx in processed_set:
        stats["skipped"] += 1
        continue

    # Paths in the CSV point to Session 0's /kaggle/working — resolve via the index
    if not os.path.exists(video_path):
        fname = record.get("filename", os.path.basename(video_path))
        if fname in video_index:
            video_path = video_index[fname]
        else:
            log_records.append({
                "video_idx": video_idx,
                "sentence_id": sentence_id,
                "signer_id": signer_id,
                "status": "FAILED",
                "error": "File not found",
                "frames": 0
            })
            stats["failed"] += 1
            continue

    # Progress display
    elapsed = time.time() - start_time
    videos_done = idx + 1 - stats["skipped"]
    if videos_done > 0:
        rate = elapsed / videos_done
        remaining = rate * (len(video_records) - idx - 1)
        eta_hours = remaining / 3600
    else:
        eta_hours = 0

    if idx % 10 == 0:
        print(
            f"  [{idx+1}/{len(video_records)}] "
            f"Processing: {os.path.basename(video_path)} "
            f"(ETA: {eta_hours:.1f}h)"
        )

    # Extract landmarks
    try:
        poses, metadata = extract_landmarks_from_video(video_path)

        if poses is None:
            log_records.append({
                "video_idx": video_idx,
                "sentence_id": sentence_id,
                "signer_id": signer_id,
                "status": "FAILED",
                "error": metadata.get("error", "Unknown"),
                "frames": 0
            })
            stats["failed"] += 1
            continue

        # Interpolate missing landmarks
        poses = interpolate_missing_landmarks(poses)

        # Save as .npy
        output_name = f"{sentence_id}_{signer_id}.npy"
        # Sanitize filename
        output_name = "".join(
            c if (c.isalnum() or c in '._-') else '_'
            for c in output_name
        )
        output_path = os.path.join(POSES_DIR, output_name)
        np.save(output_path, poses)

        # Log success
        log_records.append({
            "video_idx": video_idx,
            "sentence_id": sentence_id,
            "signer_id": signer_id,
            "status": "OK",
            "error": "",
            "frames": poses.shape[0],
            "output_file": output_name,
            "pose_det_rate": f"{metadata['pose_detection_rate']:.3f}",
            "hand_det_rate": f"{metadata['hand_detection_rate']:.3f}",
            "orig_fps": metadata["original_fps"],
            "resolution": f"{metadata['width']}x{metadata['height']}"
        })

        stats["succeeded"] += 1
        stats["total_frames"] += poses.shape[0]
        detection_rates_pose.append(metadata["pose_detection_rate"])
        detection_rates_hand.append(metadata["hand_detection_rate"])
        frame_counts.append(poses.shape[0])

    except Exception as e:
        log_records.append({
            "video_idx": video_idx,
            "sentence_id": sentence_id,
            "signer_id": signer_id,
            "status": "ERROR",
            "error": str(e)[:200],
            "frames": 0
        })
        stats["failed"] += 1
        if idx < 5:  # Print full traceback for first few errors
            traceback.print_exc()

    stats["processed"] = stats["succeeded"] + stats["failed"]

    # Checkpoint
    if (idx + 1) % CHECKPOINT_INTERVAL == 0:
        processed_set.add(video_idx)
        checkpoint_data = {
            "batch_id": BATCH_ID,
            "processed": list(processed_set),
            "last_index": idx,
            "stats": stats
        }
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint_data, f)
        print(f"    ✓ Checkpoint saved ({stats['succeeded']} OK, {stats['failed']} failed)")

    processed_set.add(video_idx)

# ============================================================================
# STEP 5: Save final results
# ============================================================================
elapsed_total = time.time() - start_time

print(f"\n{'=' * 70}")
print(f"Step 5: Saving results...")
print(f"{'=' * 70}")

# Update final stats
if detection_rates_pose:
    stats["avg_pose_detection_rate"] = float(np.mean(detection_rates_pose))
if detection_rates_hand:
    stats["avg_hand_detection_rate"] = float(np.mean(detection_rates_hand))
if frame_counts:
    stats["avg_frames"] = float(np.mean(frame_counts))
    stats["median_frames"] = float(np.median(frame_counts))
    stats["min_frames"] = int(np.min(frame_counts))
    stats["max_frames"] = int(np.max(frame_counts))

stats["processing_time_seconds"] = elapsed_total
stats["processing_time_hours"] = elapsed_total / 3600

# Save stats
with open(STATS_FILE, 'w') as f:
    json.dump(stats, f, indent=2)

# Save processing log
log_path = LOG_FILE
with open(log_path, 'w', newline='', encoding='utf-8') as f:
    if log_records:
        writer = csv.DictWriter(f, fieldnames=log_records[0].keys())
        writer.writeheader()
        writer.writerows(log_records)

# Also save a mapping file (sentence_id → pose file)
mapping_path = os.path.join(OUTPUT_DIR, f"batch_{BATCH_ID}_mapping.csv")
with open(mapping_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["sentence_id", "signer_id", "pose_file", "num_frames"])
    for rec in log_records:
        if rec.get("status") == "OK":
            writer.writerow([
                rec["sentence_id"],
                rec["signer_id"],
                rec.get("output_file", ""),
                rec.get("frames", 0)
            ])

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'=' * 70}")
print(f"BATCH {BATCH_ID} COMPLETE!")
print(f"{'=' * 70}")
print(f"""
  Results:
    Total videos:      {stats['total_videos']}
    Processed:         {stats['processed']}
    Succeeded:         {stats['succeeded']}
    Failed:            {stats['failed']}
    Skipped (cached):  {stats['skipped']}

  Quality:
    Avg pose detection:  {stats.get('avg_pose_detection_rate', 0):.1%}
    Avg hand detection:  {stats.get('avg_hand_detection_rate', 0):.1%}
    Avg frames/video:    {stats.get('avg_frames', 0):.0f}
    Median frames:       {stats.get('median_frames', 0):.0f}

  Timing:
    Total time:        {stats['processing_time_hours']:.1f} hours
    Per video:         {elapsed_total / max(1, stats['processed']):.1f} seconds

  Output directory:    {OUTPUT_DIR}
  Pose files:          {POSES_DIR}/*.npy
  Processing log:      {LOG_FILE}
  Batch mapping:       {mapping_path}

NEXT STEPS:
  1. Download {OUTPUT_DIR}/ from Kaggle
  2. Upload as a new Kaggle Dataset: "bdsl-batch-{BATCH_ID}-mediapipe"
  3. Run the next batch (BATCH_ID = {BATCH_ID + 1})
  4. After all 7 batches, run Session_Final_Merge.py
""")

# Count output files
npy_files = [f for f in os.listdir(POSES_DIR) if f.endswith('.npy')]
print(f"  Total .npy files saved: {len(npy_files)}")

# Show sample shapes
if npy_files:
    sample = np.load(os.path.join(POSES_DIR, npy_files[0]))
    print(f"  Sample shape: {sample.shape}  (expected: (T, 543, 3))")
    print(f"  Sample dtype: {sample.dtype}")

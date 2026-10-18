"""
================================================================================
SESSION 0: Download Ban-Sign-Sent-9K + Build Metadata + Split into Batches
================================================================================

PURPOSE:
  - Download Ban-Sign-Sent-9K dataset from HuggingFace
  - Extract the multi-part zip archive
  - Build a master CSV mapping (video_path, sentence_id, signer_id, sentence_bn, sentence_en)
  - Split ~9,610 videos into 7 processing batches (~1,373 each)
  - Save everything as a Kaggle Dataset for subsequent sessions

RUNTIME: ~1-2 hours (mostly download + extraction)
STORAGE: ~15-20 GB after extraction

INSTRUCTIONS:
  1. Create a new Kaggle Notebook (CPU is fine, no GPU needed)
  2. Paste this entire script into a single code cell
  3. Run it
  4. After completion, go to /kaggle/working/bdsl_dataset/
  5. Click "New Dataset" → upload the entire bdsl_dataset folder
  6. Name it: "bdsl-ban-sign-sent-9k-extracted"
  7. This dataset will be used by Sessions 1-7
================================================================================
"""

import os
import sys
import subprocess
import glob
import json
import csv
import shutil
from pathlib import Path
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================
NUM_BATCHES = 7
WORK_DIR = "/kaggle/working"
OUTPUT_DIR = os.path.join(WORK_DIR, "bdsl_dataset")
VIDEO_DIR = os.path.join(OUTPUT_DIR, "videos")
META_DIR = os.path.join(OUTPUT_DIR, "metadata")
BATCH_DIR = os.path.join(OUTPUT_DIR, "batches")

# HuggingFace dataset info (public, no token needed)
# https://huggingface.co/datasets/banglagov/Ban-Sign-Sent-9K-V1
# Repo layout:
#   Bangla_Sign_Sentence.xlsx               — Sentence ID / Natural Sentence / Sign Sentence (gloss)
#   Sign_Videos.z01 ... .z09 + .zip         — spanned zip (~20 GB), 9,610 mp4s
#   inside: Sign_Videos/<sentence_id>/REC_CS_S<sentence_id>_U<signer>_<date>_<ts>.mp4
HF_REPO = "banglagov/Ban-Sign-Sent-9K-V1"
ANNOTATION_XLSX = "Bangla_Sign_Sentence.xlsx"

# Large raw archives go to scratch disk (not /kaggle/working, which is size-limited)
DOWNLOAD_DIR = "/kaggle/tmp/hf_download" if os.path.isdir("/kaggle") else os.path.join(WORK_DIR, "hf_download")

# ============================================================================
# STEP 1: Install dependencies
# ============================================================================
print("=" * 70)
print("STEP 1: Installing dependencies...")
print("=" * 70)

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "huggingface_hub", "openpyxl", "pandas"
])

# 7-Zip is needed to extract the spanned .z01/.zip archive (Python zipfile can't)
if shutil.which("7z") is None:
    print("  Installing p7zip-full...")
    subprocess.run("apt-get install -y -qq p7zip-full > /dev/null 2>&1 || "
                   "sudo apt-get install -y -qq p7zip-full > /dev/null 2>&1", shell=True)

from huggingface_hub import list_repo_files

# ============================================================================
# STEP 2: Create directory structure
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Creating directory structure...")
print("=" * 70)

for d in [OUTPUT_DIR, VIDEO_DIR, META_DIR, BATCH_DIR]:
    os.makedirs(d, exist_ok=True)
    print(f"  Created: {d}")

# ============================================================================
# STEP 3: Download dataset from HuggingFace
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Downloading Ban-Sign-Sent-9K from HuggingFace...")
print("=" * 70)

# List all files in the repo to understand the structure
print("  Listing repository files...")
try:
    repo_files = list_repo_files(HF_REPO, repo_type="dataset")
    print(f"  Found {len(repo_files)} files in repository")
    for f in repo_files[:20]:
        print(f"    - {f}")
    if len(repo_files) > 20:
        print(f"    ... and {len(repo_files) - 20} more")
except Exception as e:
    print(f"  Warning: Could not list files: {e}")
    repo_files = []

# Download all files (skip anything already fully downloaded — resumable)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

print(f"\n  Downloading dataset files to {DOWNLOAD_DIR} (~20 GB)...")
from huggingface_hub import snapshot_download
snapshot_download(
    HF_REPO,
    repo_type="dataset",
    local_dir=DOWNLOAD_DIR,
    max_workers=4,
)
print("  Download complete.")

# ============================================================================
# STEP 4: Extract the spanned zip (Sign_Videos.z01 ... .z09 + Sign_Videos.zip)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: Extracting archives...")
print("=" * 70)

main_zip = os.path.join(DOWNLOAD_DIR, "Sign_Videos.zip")
parts = sorted(glob.glob(os.path.join(DOWNLOAD_DIR, "Sign_Videos.z[0-9][0-9]")))
print(f"  Found {len(parts)} .zNN parts + {os.path.basename(main_zip)}")
if not os.path.exists(main_zip) or len(parts) != 9:
    raise RuntimeError("Archive incomplete — expected Sign_Videos.z01..z09 + Sign_Videos.zip. Re-run to resume download.")

already = glob.glob(os.path.join(VIDEO_DIR, "Sign_Videos", "*", "*.mp4"))
if len(already) >= 9610:
    print(f"  Already extracted ({len(already)} videos), skipping.")
elif shutil.which("7z"):
    # 7z reads the .z01..z09 parts automatically when given the .zip
    subprocess.run(["7z", "x", "-y", "-bso0", "-bsp1", main_zip, f"-o{VIDEO_DIR}"], check=True)
else:
    # Fallback: Info-ZIP — merge spans into a single zip, then unzip
    merged = os.path.join(DOWNLOAD_DIR, "Sign_Videos_merged.zip")
    subprocess.run(["zip", "-s", "0", main_zip, "--out", merged], check=True)
    subprocess.run(["unzip", "-o", "-q", merged, "-d", VIDEO_DIR], check=True)
    os.remove(merged)

# Free scratch space once videos are out
for p in parts + [main_zip]:
    os.remove(p)
print("  Removed archive parts to free disk.")

# ============================================================================
# STEP 5: Load sentence annotations (xlsx)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: Loading annotations...")
print("=" * 70)

import pandas as pd

ann = pd.read_excel(os.path.join(DOWNLOAD_DIR, ANNOTATION_XLSX), sheet_name=0)
ann = ann.rename(columns={
    "Sentence ID": "sentence_id",
    "Natural Sentence": "sentence_bn",
    "Sign Sentence": "gloss_bn",
})[["sentence_id", "sentence_bn", "gloss_bn"]]
ann["sentence_id"] = ann["sentence_id"].astype(int).astype(str)
ann["sentence_bn"] = ann["sentence_bn"].astype(str).str.strip()
ann["gloss_bn"] = ann["gloss_bn"].astype(str).str.strip()
print(f"  {len(ann)} sentences")
print(ann.head().to_string())

ann.to_csv(os.path.join(META_DIR, "sentences.csv"), index=False, encoding="utf-8")
shutil.copy(os.path.join(DOWNLOAD_DIR, ANNOTATION_XLSX), META_DIR)
sentence_lookup = ann.set_index("sentence_id").to_dict("index")

# ============================================================================
# STEP 6: Discover videos and build master CSV
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: Building master CSV...")
print("=" * 70)

import re

# Sign_Videos/<sentence_id>/{REC,UL}_CS_S<sentence_id>_U<signer>_<date>_<ts>.mp4
NAME_RE = re.compile(r"_CS_S(?P<sid>\d+)_U(?P<uid>\d+)_")

all_videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "**", "*.mp4"), recursive=True))
print(f"  Found {len(all_videos)} video files")

master_records = []
unmatched, no_annotation = [], 0
for video_path in all_videos:
    rel_path = os.path.relpath(video_path, VIDEO_DIR)
    m = NAME_RE.search(Path(video_path).name)
    if not m:
        unmatched.append(rel_path)
        continue
    sentence_id = m.group("sid")
    signer_id = f"U{m.group('uid')}"   # no underscore: downstream uses {sentence_id}_{signer_id}.npy
    info = sentence_lookup.get(sentence_id)
    if info is None:
        no_annotation += 1
    master_records.append({
        "video_path": video_path,
        "rel_path": rel_path,
        "sentence_id": sentence_id,
        "signer_id": signer_id,
        "filename": Path(video_path).name,
        "sentence_bn": info["sentence_bn"] if info else "",
        "gloss_bn": info["gloss_bn"] if info else "",
    })

print(f"  Built {len(master_records)} records")
print(f"  Unparseable filenames: {len(unmatched)} {unmatched[:5]}")
print(f"  Videos without annotation: {no_annotation}")

sentence_counts = defaultdict(int)
for r in master_records:
    sentence_counts[r["sentence_id"]] += 1
count_dist = defaultdict(int)
for cnt in sentence_counts.values():
    count_dist[cnt] += 1
print(f"  Unique sentence IDs: {len(sentence_counts)}")
print(f"  Unique signers: {sorted({r['signer_id'] for r in master_records})}")
print(f"  Videos per sentence distribution:")
for cnt, num_sentences in sorted(count_dist.items()):
    print(f"    {cnt} videos: {num_sentences} sentences")

# Save master CSV
master_csv_path = os.path.join(META_DIR, "master_videos.csv")
with open(master_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=[
        "video_idx", "video_path", "rel_path", "sentence_id",
        "signer_id", "filename", "sentence_bn", "gloss_bn"
    ])
    writer.writeheader()
    for i, rec in enumerate(master_records):
        rec["video_idx"] = i
        writer.writerow(rec)

print(f"\n  Saved master CSV: {master_csv_path}")
print(f"  Total records: {len(master_records)}")

# ============================================================================
# STEP 7: Select best signer per sentence (use all 5 for robustness)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: Selecting videos for processing...")
print("=" * 70)

"""
Strategy: Use ALL 5 signers per sentence for signer-independent training.
This gives us ~9,610 videos total.
For the model to generalize across signers, we need variety.
Later during training, we can do signer-based train/val split.
"""

# Use all videos
videos_to_process = master_records.copy()
print(f"  Using ALL {len(videos_to_process)} videos (all signers)")

# ============================================================================
# STEP 8: Split into 7 batches
# ============================================================================
print("\n" + "=" * 70)
print(f"STEP 8: Splitting into {NUM_BATCHES} processing batches...")
print("=" * 70)

import math

batch_size = math.ceil(len(videos_to_process) / NUM_BATCHES)
print(f"  Total videos: {len(videos_to_process)}")
print(f"  Target batch size: {batch_size}")

for batch_id in range(NUM_BATCHES):
    start = batch_id * batch_size
    end = min(start + batch_size, len(videos_to_process))
    batch = videos_to_process[start:end]

    batch_file = os.path.join(BATCH_DIR, f"batch_{batch_id}.csv")
    with open(batch_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video_idx", "video_path", "rel_path", "sentence_id",
            "signer_id", "filename", "sentence_bn", "gloss_bn"
        ])
        writer.writeheader()
        writer.writerows(batch)

    print(f"  Batch {batch_id}: {len(batch)} videos → {batch_file}")

# ============================================================================
# STEP 9: Save processing config
# ============================================================================
print("\n" + "=" * 70)
print("STEP 9: Saving processing config...")
print("=" * 70)

config = {
    "dataset": "Ban-Sign-Sent-9K",
    "hf_repo": HF_REPO,
    "total_videos": len(videos_to_process),
    "num_batches": NUM_BATCHES,
    "batch_size": batch_size,
    "extraction_method": "MediaPipe Holistic (SLRTP2025 format)",
    "output_format": "numpy (T, 543, 3)",
    "landmarks": {
        "pose": "33 landmarks (indices 0-32)",
        "left_hand": "21 landmarks (indices 33-53)",
        "right_hand": "21 landmarks (indices 54-74)",
        "face": "468 landmarks (indices 75-542)",
        "total": 543
    },
    "normalization": "neck (landmark 0 midpoint of shoulders) at origin",
    "target_fps": 25
}

config_path = os.path.join(META_DIR, "processing_config.json")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"  Config saved to: {config_path}")
print(f"\n  Config summary:")
print(json.dumps(config, indent=4))

# ============================================================================
# STEP 10: Verify and summarize
# ============================================================================
print("\n" + "=" * 70)
print("STEP 10: Verification & Summary")
print("=" * 70)

print(f"\n  Output directory: {OUTPUT_DIR}")
print(f"\n  Directory structure:")
for root, dirs, files in os.walk(OUTPUT_DIR):
    depth = root.replace(OUTPUT_DIR, '').count(os.sep)
    if depth < 2:
        indent = '  ' * (depth + 2)
        print(f"{indent}{os.path.basename(root)}/")
        for f in sorted(files):
            size_mb = os.path.getsize(os.path.join(root, f)) / (1024**2)
            print(f"{indent}  {f} ({size_mb:.1f} MB)")

# Disk usage
total_size = sum(
    os.path.getsize(os.path.join(root, f))
    for root, _, files in os.walk(OUTPUT_DIR)
    for f in files
)
print(f"\n  Total output size: {total_size / (1024**3):.2f} GB")

# Free disk space
import shutil as sh
total, used, free = sh.disk_usage(WORK_DIR)
print(f"  Free disk space: {free / (1024**3):.2f} GB")

print("\n" + "=" * 70)
print("SESSION 0 COMPLETE!")
print("=" * 70)
print("""
NEXT STEPS:
  1. Check the output in /kaggle/working/bdsl_dataset/
  2. Verify video files exist in bdsl_dataset/videos/
  3. Check batch files in bdsl_dataset/batches/ (should be batch_0.csv through batch_6.csv)
  4. CREATE A NEW KAGGLE DATASET from the bdsl_dataset/ folder:
     - Click "New Dataset" in Kaggle
     - Upload the entire bdsl_dataset/ folder
     - Name it: "bdsl-ban-sign-sent-9k-extracted"
  5. Proceed to Session 1 with the batch processing script

IMPORTANT:
  - If no videos were found, check the download directory structure manually
  - The dataset structure may vary — adjust VIDEO_DIR path if needed
  - Each batch CSV contains video paths that Session 1-7 scripts will read
""")

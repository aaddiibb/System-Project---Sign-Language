"""Clean demo-run (1).ipynb and embed the exact offline Kiseki runtime."""
from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
NOTEBOOK = ROOT / "demo-run (1).ipynb"
KISEKI_PACKAGE = ROOT / ".tmp-kiseki" / "kiseki"
KISEKI_COMMIT = "07fd2bcfadb8477e985d33d4945bf86eedcff440"

if not NOTEBOOK.is_file():
    raise FileNotFoundError(NOTEBOOK)
if not (KISEKI_PACKAGE / "sample.bvh").is_file():
    raise FileNotFoundError("Exact Kiseki package source is unavailable")

archive_buffer = io.BytesIO()
with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for path in sorted(KISEKI_PACKAGE.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            archive.write(path, Path("kiseki") / path.relative_to(KISEKI_PACKAGE))
embedded_archive = base64.b64encode(archive_buffer.getvalue()).decode("ascii")

heldout_source = '''heldout_available = False
heldout_error = ""
try:
    if FEATURE_DIR is None:
        raise FileNotFoundError("Motion-Features directory is unavailable")

    # Preferred source: the locked Phase-1 test split. If it was not attached,
    # Phase-8's frozen final test result IDs are an exact record of that split.
    if DATASET_CONFIG_PATH is not None and TEST_SPLIT_PATH is not None:
        dataset_config = read_json(DATASET_CONFIG_PATH)
        id_column = dataset_config["id_column"]
        sentence_column = dataset_config.get("sentence_column", "sentence")
        gloss_column = dataset_config.get("gloss_column", "gloss")
        heldout_frame = pd.read_csv(TEST_SPLIT_PATH, dtype={id_column: str}, low_memory=False)
        heldout_frame[id_column] = heldout_frame[id_column].astype(str)
        test_id_source = "locked Phase-1 test split"
    else:
        frozen_test_results = PHASE8_ROOT / "results/final_end2end_metrics.csv"
        if not frozen_test_results.is_file():
            raise FileNotFoundError("Neither the Phase-1 test split nor Phase-8 frozen test metrics are available")
        heldout_frame = pd.read_csv(frozen_test_results, dtype={"sample_id": str}, low_memory=False)
        if "sample_id" not in heldout_frame:
            raise RuntimeError(f"No sample_id column in {frozen_test_results}")
        heldout_frame = heldout_frame[["sample_id"]].drop_duplicates().reset_index(drop=True)
        heldout_frame["sample_id"] = heldout_frame["sample_id"].astype(str)
        id_column, sentence_column, gloss_column = "sample_id", "sentence", "gloss"
        test_id_source = "Phase-8 frozen final-test IDs"

    # Resolve official train.csv even when dataset_config.json was not attached.
    if OFFICIAL_TRAIN_CSV is None:
        official_candidates = []
        for candidate in named("train.csv"):
            try:
                columns = set(pd.read_csv(candidate, nrows=0).columns)
                if {"sentence", "gloss"}.issubset(columns) and ({"id", "sample_id"} & columns):
                    official_candidates.append(candidate)
            except Exception:
                pass
        if not official_candidates:
            raise FileNotFoundError("Official Motion-S train.csv could not be resolved")
        OFFICIAL_TRAIN_CSV = sorted(official_candidates, key=lambda path: (0 if "competitions" in str(path) else 1, len(path.parts), str(path)))[0]

    official_columns = list(pd.read_csv(OFFICIAL_TRAIN_CSV, nrows=0).columns)
    official_id = "id" if "id" in official_columns else "sample_id"
    enrichment = pd.read_csv(
        OFFICIAL_TRAIN_CSV,
        usecols=[official_id, "sentence", "gloss"],
        dtype={official_id: str},
        low_memory=False,
    ).rename(columns={official_id: id_column})
    enrichment[id_column] = enrichment[id_column].astype(str)
    for column in (sentence_column, gloss_column):
        if column in heldout_frame:
            heldout_frame = heldout_frame.drop(columns=column)
    heldout_frame = heldout_frame.merge(enrichment, on=id_column, how="left", validate="one_to_one")

    feature_paths = {path.stem: path for path in FEATURE_DIR.glob("*.npy")}
    selected = None
    for index, row in heldout_frame.sort_values(id_column, kind="stable").iterrows():
        sample_id = str(row[id_column])
        sentence = str(row.get(sentence_column, "")).strip()
        path = feature_paths.get(sample_id)
        if path is None or not sentence or sentence.casefold() == "nan":
            continue
        shape = np.load(path, mmap_mode="r", allow_pickle=False).shape
        if len(shape) == 2 and shape[1] == 668 and math.ceil(shape[0] / 4) <= 512:
            selected = (index, path, shape)
            break
    if selected is None:
        raise RuntimeError("No deterministic supported held-out sample has text and a [T,668] feature file")

    heldout_index, heldout_feature_path, heldout_shape = selected
    heldout_row = heldout_frame.loc[heldout_index]
    heldout_id = str(heldout_row[id_column])
    heldout_sentence = str(heldout_row[sentence_column]).strip()
    heldout_gt_gloss = str(heldout_row.get(gloss_column, "")).strip()
    heldout_gt_motion = np.load(heldout_feature_path, allow_pickle=False).astype(np.float32)
    heldout_available = True
    print("Test-ID source:", test_id_source)
    print("Sample ID:", heldout_id)
    print("English sentence:", heldout_sentence)
    print("Ground-truth gloss:", heldout_gt_gloss if heldout_gt_gloss and heldout_gt_gloss.casefold() != "nan" else "UNAVAILABLE")
    print("Ground-truth motion shape:", heldout_gt_motion.shape)
except Exception as error:
    heldout_error = repr(error)
    print("Ground-truth comparison unavailable")
    print("Diagnostic:", heldout_error)
'''

visualization_source = f'''# Offline 55-joint visualization of the generated [T,668] motion.
# Kiseki commit: {KISEKI_COMMIT} (the exact commit used successfully by
# the official Motion-S dataset-exploration notebook).
import base64
import importlib
import io
import sys
import zipfile
from pathlib import Path
from IPython.display import Video, display

KISEKI_COMMIT = "{KISEKI_COMMIT}"
KISEKI_ARCHIVE_B64 = """{embedded_archive}"""
vendor_root = Path("/kaggle/working/motion_s_final_demo/_offline_vendor")
package_root = vendor_root / "kiseki"

if not (package_root / "sample.bvh").is_file():
    vendor_root.mkdir(parents=True, exist_ok=True)
    payload = base64.b64decode(KISEKI_ARCHIVE_B64)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(vendor_root)

if str(vendor_root) not in sys.path:
    sys.path.insert(0, str(vendor_root))
importlib.invalidate_caches()

from kiseki import visualize

demo_dir = Path("/kaggle/working/motion_s_final_demo")
motion_path = demo_dir / "custom_demo_motion.npy"
video_path = demo_dir / "generated_motion_3d.mp4"

# The animation cell intentionally runs before the final artifact inventory.
# Use the inference result already present in memory and persist it here.
demo_dir.mkdir(parents=True, exist_ok=True)
if "custom_motion" in globals():
    motion_for_view = np.asarray(custom_motion, dtype=np.float32)
    np.save(motion_path, motion_for_view, allow_pickle=False)
elif motion_path.is_file():
    motion_for_view = np.load(motion_path, allow_pickle=False)
else:
    raise FileNotFoundError(
        "No generated motion is available. Run the Custom English demo cell before this visualization cell."
    )

assert motion_for_view.ndim == 2 and motion_for_view.shape[1] == 668
assert np.isfinite(motion_for_view).all()

# display=False is intentional: Kiseki then saves the MP4 and frame grid.
# The saved MP4 is embedded immediately afterward without rendering twice.
result_path = visualize(
    motion_path,
    output_path=video_path,
    fixed_view="front",
    fps=30,
    tracking=True,
    title="Generated Motion-S Motion",
    trails="fingertips",
    trail_length=30,
    save_grid=True,
    grid_frames=9,
    display=False,
)

if result_path is None or not Path(result_path).is_file():
    raise RuntimeError("Kiseki did not create the expected MP4")
print("Offline Kiseki commit:", KISEKI_COMMIT)
print("Reconstructed generated motion:", motion_for_view.shape)
print("Saved 55-joint animation:", result_path)
display(Video(str(result_path), embed=True, width=900))
'''

notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
cells = notebook["cells"]
if len(cells) < 44:
    raise RuntimeError(f"Unexpected notebook structure: {len(cells)} cells")

# Remove previously appended visualization cells by content. This makes the
# cleanup idempotent while preserving the original 22-section demo.
drop_markers = ("generated_motion_feature_animation", "pip install -q", "Offline 55-joint generated-motion animation", "KISEKI_ARCHIVE_B64")
cells = [cell for cell in cells if not any(marker in "".join(cell.get("source", [])) for marker in drop_markers)]
if len(cells) != 44:
    raise RuntimeError(f"Expected 44 core cells after cleanup, found {len(cells)}")
cells[19]["source"] = heldout_source.splitlines(keepends=True)

visualization_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Offline 55-joint generated-motion animation\n",
            "\n",
            "This uses the exact Kiseki runtime exercised by the official Motion-S exploration notebook. "
            "It is embedded in this notebook, so Kaggle Internet is not required.\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": visualization_source.splitlines(keepends=True),
    },
]
# Place visualization before the final artifact-save section so its MP4 and
# frame-grid PNG are included in the output inventory printed by the last cell.
cells = cells[:42] + visualization_cells + cells[42:]

# Remove stale outputs and the previous error trace. The cleaned notebook is now
# small, reproducible, and ready for Run All.
for cell in cells:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

notebook["cells"] = cells
NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Updated {NOTEBOOK} with {len(cells)} cells; embedded archive bytes={len(archive_buffer.getvalue())}")

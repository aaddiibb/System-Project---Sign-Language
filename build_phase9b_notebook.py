"""Build the Phase-9b video-comparison Kaggle notebook with the offline Kiseki package embedded."""
from __future__ import annotations
import base64
import io
import json
import zipfile
from pathlib import Path
ROOT=Path(__file__).parent;SOURCE=ROOT/"09b_motion_s_video_comparison.py";TARGET=ROOT/"09b_motion_s_video_comparison.ipynb";KISEKI=next((p for p in (ROOT/".tmp-kiseki"/"kiseki",ROOT/"Outputs"/"Last final demo run"/"motion_s_final_demo"/"_offline_vendor"/"kiseki") if (p/"sample.bvh").is_file()),ROOT/".tmp-kiseki"/"kiseki")
def kiseki_archive():
    if not (KISEKI/"sample.bvh").is_file():raise FileNotFoundError("Vendored Kiseki package not found in .tmp-kiseki/ or Outputs/Last final demo run/motion_s_final_demo/_offline_vendor/")
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in sorted(KISEKI.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:archive.write(path,Path("kiseki")/path.relative_to(KISEKI))
    return base64.b64encode(buffer.getvalue()).decode("ascii")
def markdown(lines):return [x[2:] if x.startswith("# ") else x[1:] if x.startswith("#") else x for x in lines]
def build():
    text=SOURCE.read_text(encoding="utf-8")
    if text.count('"__KISEKI_ARCHIVE_B64__"')!=1:raise RuntimeError("Kiseki placeholder missing")
    lines=text.replace('"__KISEKI_ARCHIVE_B64__"','"'+kiseki_archive()+'"').splitlines(keepends=True);cells=[];kind=None;buffer=[]
    def flush():
        nonlocal buffer
        if kind is None:return
        cell={"cell_type":kind,"metadata":{},"source":markdown(buffer) if kind=="markdown" else buffer}
        if kind=="code":cell.update({"execution_count":None,"outputs":[]})
        cells.append(cell);buffer=[]
    for line in lines:
        marker=line.rstrip("\r\n")
        if marker=="# %% [markdown]":flush();kind="markdown"
        elif marker=="# %%":flush();kind="code"
        else:buffer.append(line)
    flush();notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"},"kaggle":{"accelerator":"none","dataSources":[],"isGpuEnabled":False,"isInternetEnabled":False,"language":"python","sourceType":"notebook"}},"nbformat":4,"nbformat_minor":5};TARGET.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding="utf-8");print("Wrote",TARGET,"with",len(cells),"cells")
if __name__=="__main__":build()

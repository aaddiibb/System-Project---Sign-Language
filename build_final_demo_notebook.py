"""Build the Motion-S final demo/inference Kaggle notebook."""
from __future__ import annotations

import json
from pathlib import Path

SOURCE = Path(__file__).with_name("motion_s_final_end_to_end_demo.py")
TARGET = Path(__file__).with_name("motion_s_final_end_to_end_demo.ipynb")

def markdown(lines: list[str]) -> list[str]:
    return [line[2:] if line.startswith("# ") else line[1:] if line.startswith("#") else line for line in lines]

def build() -> None:
    cells = []
    kind = None
    buffer: list[str] = []
    def flush() -> None:
        nonlocal buffer
        if kind is None: return
        cell = {"cell_type": kind, "metadata": {}, "source": markdown(buffer) if kind == "markdown" else buffer}
        if kind == "code": cell.update({"execution_count": None, "outputs": []})
        cells.append(cell); buffer = []
    for line in SOURCE.read_text(encoding="utf-8").splitlines(keepends=True):
        marker = line.rstrip("\r\n")
        if marker == "# %% [markdown]": flush(); kind = "markdown"
        elif marker == "# %%": flush(); kind = "code"
        else: buffer.append(line)
    flush()
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "kaggle": {"accelerator": "gpu", "dataSources": [], "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {TARGET} with {len(cells)} cells")

if __name__ == "__main__": build()

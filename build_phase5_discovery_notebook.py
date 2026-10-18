"""Build the Phase-5A official-decoder discovery notebook."""

from __future__ import annotations

import json
from pathlib import Path

SOURCE = Path(__file__).with_name("05_motion_s_rvq_decoder_discovery.py")
TARGET = Path(__file__).with_name("05_motion_s_rvq_decoder_discovery.ipynb")


def markdown_source(lines: list[str]) -> list[str]:
    output = []
    for line in lines:
        if line.startswith("# "): output.append(line[2:])
        elif line.startswith("#"): output.append(line[1:])
        else: output.append(line)
    return output


def build() -> None:
    lines = SOURCE.read_text(encoding="utf-8").splitlines(keepends=True)
    cells, kind, buffer = [], None, []

    def flush() -> None:
        nonlocal buffer
        if kind is None: return
        source = markdown_source(buffer) if kind == "markdown" else buffer
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == "code": cell.update({"execution_count": None, "outputs": []})
        cells.append(cell); buffer = []

    for line in lines:
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
            "kaggle": {"accelerator": "gpu", "dataSources": [], "dockerImageVersionId": None,
                       "isGpuEnabled": True, "isInternetEnabled": False, "language": "python",
                       "sourceType": "notebook"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {TARGET} with {len(cells)} cells")


if __name__ == "__main__": build()

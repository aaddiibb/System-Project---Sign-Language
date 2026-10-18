"""Build the Kaggle notebook from the percent-format Python source."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "cv_mammqa_16_kaggle.py"
TARGET = ROOT / "CV_MAMMQA_Provided_Context_16.ipynb"


def split_cells(text: str) -> list[dict]:
    cells: list[dict] = []
    kind = None
    lines: list[str] = []

    def flush() -> None:
        nonlocal lines, kind
        if kind is None:
            return
        if kind == "markdown":
            cooked = []
            for line in lines:
                if line.startswith("# "):
                    cooked.append(line[2:])
                elif line.rstrip() == "#":
                    cooked.append("")
                else:
                    cooked.append(line)
            source = "\n".join(cooked).rstrip() + "\n"
            cells.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)})
        else:
            source = "\n".join(lines).rstrip() + "\n"
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": source.splitlines(True),
                }
            )
        lines = []

    for line in text.splitlines():
        if line == "# %%":
            flush()
            kind = "code"
        elif line == "# %% [markdown]":
            flush()
            kind = "markdown"
        else:
            lines.append(line)
    flush()
    return cells


notebook = {
    "cells": split_cells(SOURCE.read_text(encoding="utf-8")),
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "kaggle": {"accelerator": "gpu", "dataSources": [], "dockerImageVersionId": None},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {TARGET} with {len(notebook['cells'])} cells")

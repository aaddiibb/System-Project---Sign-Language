"""Static integrity checks for the generated CV-MAMMQA Kaggle notebook."""
from __future__ import annotations

import ast
import builtins
import json
import re
import symtable
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "cv_mammqa_16_kaggle.py"
NOTEBOOK_PATH = ROOT / "CV_MAMMQA_Provided_Context_16.ipynb"
source = SOURCE_PATH.read_text(encoding="utf-8")
tree = ast.parse(source, filename=str(SOURCE_PATH))
compile(tree, str(SOURCE_PATH), "exec")

notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
assert notebook["nbformat"] == 4
code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
for index, cell in enumerate(code_cells):
    compile("".join(cell["source"]), f"notebook-code-cell-{index}", "exec")

module_table = symtable.symtable(source, str(SOURCE_PATH), "exec")
module_defined = {
    symbol.get_name()
    for symbol in module_table.get_symbols()
    if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace() or symbol.is_parameter()
}
allowed = module_defined | set(dir(builtins)) | {"__file__", "__name__", "__conditional_annotations__"}
undefined = set()


def visit_table(table) -> None:
    for symbol in table.get_symbols():
        if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in allowed:
            undefined.add(symbol.get_name())
    for child in table.get_children():
        visit_table(child)


visit_table(module_table)
assert not undefined, f"Potential undefined global symbols: {sorted(undefined)}"

for banned_import in ["faiss", "sentence_transformers", "networkx", "rank_bm25"]:
    assert not re.search(rf"(?m)^\s*(?:from|import)\s+{re.escape(banned_import)}\b", source)
assert "EVALUATION_ONLY_STORE" not in source
assert source.index("PREDICTIONS_FROZEN = True") < source.index("EVALUATION_VIEW = load_evaluation_after_freeze")

functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
baseline = functions["aggregate_mammqa_baseline"]
assert [argument.arg for argument in baseline.args.args] == ["item", "stage2"]
baseline_calls = {node.func.id for node in ast.walk(baseline) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
assert not ({"extract_claims", "verify_claim", "detect_dispute", "repair_candidate"} & baseline_calls)

for forbidden in ["TODO", "FIXME", "NotImplementedError"]:
    assert forbidden not in source

print(
    json.dumps(
        {
            "source_compiles": True,
            "notebook_code_cells_compile": len(code_cells),
            "undefined_global_symbols": [],
            "gold_loaded_only_after_freeze": True,
            "baseline_cv_contamination_check": "PASS",
            "forbidden_retrieval_imports": "ABSENT",
        },
        indent=2,
    )
)

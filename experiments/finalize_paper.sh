#!/bin/bash
# Run after the EKS sweep completes and the cluster is destroyed.
# Aggregates cluster logs → figures → recompiles paper.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$REPO_ROOT"
source venv/bin/activate

echo "== 1) Aggregate per-episode JSONs → summary =="
python experiments/aggregate_cluster_results.py --cluster eks --out results/cluster_summary.json

echo ""
echo "== 2) Regenerate cluster-sourced figures =="
python experiments/generate_cluster_figures.py

echo ""
echo "== 3) Remove DRAFT header + submission note from main.tex =="
python3 <<PYEOF
import pathlib, re
p = pathlib.Path("paper/main.tex")
t = p.read_text()
# Strip the long DRAFT STATUS comment block (between the two ==== fences)
t = re.sub(
    r"% =+\n% DRAFT STATUS.*?% =+\n", "",
    t, flags=re.DOTALL,
)
p.write_text(t)
print("  paper/main.tex cleaned")

# Also strip the [Submission note] placeholder in experiments.tex now that
# cluster numbers are live.
q = pathlib.Path("paper/sections/experiments.tex")
s = q.read_text()
s = re.sub(
    r"\\textit\{\[Submission note:.*?\]\}\n?",
    "",
    s, flags=re.DOTALL,
)
q.write_text(s)
print("  paper/sections/experiments.tex cleaned")
PYEOF

echo ""
echo "== 4) Recompile paper =="
cd paper
pdflatex -interaction=nonstopmode main.tex > /tmp/finalize-compile.log 2>&1
bibtex main > /tmp/finalize-bibtex.log 2>&1
pdflatex -interaction=nonstopmode main.tex > /tmp/finalize-compile2.log 2>&1
pdflatex -interaction=nonstopmode main.tex > /tmp/finalize-compile3.log 2>&1

if grep -qE "^!|Error|Undefined" /tmp/finalize-compile3.log; then
    echo "  compile had warnings/errors — see /tmp/finalize-compile3.log"
else
    echo "  compile clean"
fi
grep "Output written" /tmp/finalize-compile3.log | head -1
echo ""
echo "Done. Open paper/main.pdf."

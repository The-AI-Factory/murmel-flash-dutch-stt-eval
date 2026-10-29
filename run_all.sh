#!/usr/bin/env bash
# Prepare the open Dutch test sets once, then benchmark every model in models.txt, each in its own uv virtualenv
# (the toolkits pin conflicting versions), then write results/RESULTS.md.
#   ./run_all.sh                         # everything
#   MAX_CLIPS=100 ./run_all.sh           # quick check: 100 evenly spaced clips per set
#   SETS="fleurs mls" ./run_all.sh       # some sets
#   MODELS=models.txt ./run_all.sh       # another model list
# Needs: uv (https://docs.astral.sh/uv), ffmpeg, an NVIDIA GPU for representative speed (CPU works, slowly).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON_VERSION:-3.11}
SETS=${SETS:-"fleurs common_voice mls voxpopuli"}
MAX_CLIPS=${MAX_CLIPS:-0}
MODELS=${MODELS:-models.txt}
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH" UV_SYSTEM_PYTHON=0
export MPLBACKEND=Agg   # notebook hosts (Colab) set an inline backend that the toolkits cannot load

venv() {   # venv <name> <requirements...>: create once, reuse afterwards
  local d=venvs/$1; shift
  if [ ! -x "$d/bin/python" ]; then
    uv venv -q --python "$PY" "$d"
    uv pip install -q --python "$d/bin/python" -r envs/base.txt
    for r in "$@"; do
      if [ "$r" = envs/murmel.txt ] && [ -n "${MURMEL_PACKAGE:-}" ]; then
        uv pip install -q --python "$d/bin/python" "$MURMEL_PACKAGE"
      else
        uv pip install -q --python "$d/bin/python" -r "$r"
      fi
    done
  fi
  echo "$d/bin/python"
}

BASE=$(venv base)
$BASE -m bench.sets --sets $SETS

grep -vE '^\s*(#|$)' "$MODELS" | while read -r backend model _; do
  py=$(venv "$backend" "envs/$backend.txt" < /dev/null)
  echo "=== $backend $model"
  $py -m bench.run --backend "$backend" --model "$model" --sets $SETS --max-clips "$MAX_CLIPS" < /dev/null \
    || echo "!!! $model failed, continuing"
done
$BASE -m bench.report
$BASE -m bench.plots

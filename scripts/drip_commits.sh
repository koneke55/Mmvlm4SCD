#!/usr/bin/env bash
# scripts/drip_commits.sh
#
# Optional helper that produces N "experiment progress" commits, one every
# INTERVAL_SECONDS, and pushes after each. Off by default; gated by env
# variables. Always runs on a non-main BRANCH so the main history is not
# polluted by paced micro-commits.
#
# Usage:
#   N=50 INTERVAL_SECONDS=180 BRANCH=experiment-log scripts/drip_commits.sh
#   nohup env N=50 INTERVAL_SECONDS=180 BRANCH=experiment-log \
#     scripts/drip_commits.sh > experiments/logs/drip.log 2>&1 < /dev/null &
#
# Each iteration:
#   1. runs the smoke experiment
#   2. appends one timestamped row to experiments/results/drip_log.csv
#   3. commits + pushes that single-line change
set -euo pipefail

N="${N:-50}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-180}"
BRANCH="${BRANCH:-experiment-log}"
LOG_CSV="experiments/results/drip_log.csv"

REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$REPO_ROOT"

if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

if [[ "$BRANCH" == "main" ]]; then
  echo "Refusing to drip-commit on main; pick another branch." >&2
  exit 2
fi

git fetch origin >/dev/null 2>&1 || true
git switch -c "$BRANCH" 2>/dev/null || git switch "$BRANCH"
mkdir -p "$(dirname "$LOG_CSV")" experiments/logs
[[ -f "$LOG_CSV" ]] || echo "iter,timestamp_utc,seed,val_auroc,val_c_index" > "$LOG_CSV"

extract_metric() {
  local key="$1"
  PYTHONPATH=src "$PY" - "$key" <<'PY' 2>/dev/null || echo "nan"
import json, sys, glob
key = sys.argv[1]
paths = sorted(glob.glob("experiments/results/*/test_metrics.json"))
if not paths:
    print("nan"); sys.exit(0)
m = json.load(open(paths[-1]))
print(f"{m.get(key, 'nan')}")
PY
}

for i in $(seq 1 "$N"); do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  seed="$RANDOM"
  PYTHONPATH=src "$PY" src/scripts/run_full_experiment.py \
      --config configs/default.yaml --smoke \
      > "experiments/logs/drip_iter_${i}.log" 2>&1 || true

  auroc="$(extract_metric auroc_ovr)"
  cidx="$(extract_metric c_index)"

  echo "${i},${ts},${seed},${auroc},${cidx}" >> "$LOG_CSV"
  git add "$LOG_CSV"
  git commit -m "experiment(drip): iter ${i}/${N} auroc=${auroc} c_index=${cidx}" >/dev/null
  git push -u origin "$BRANCH" >/dev/null 2>&1 || \
    echo "[$(date -u +%H:%M:%S)] push failed at iter ${i}; continuing"
  echo "[$(date -u +%H:%M:%S)] iter ${i}/${N} auroc=${auroc} c_index=${cidx}"
  if [[ "$i" -lt "$N" ]]; then sleep "$INTERVAL_SECONDS"; fi
done
echo "Drip cycle finished: $N commits pushed to $BRANCH."

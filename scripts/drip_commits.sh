#!/usr/bin/env bash
# scripts/drip_commits.sh
#
# Optional helper that produces N "experiment progress" commits, one every
# INTERVAL_SECONDS, and pushes after each. This is *purposefully gated*:
# it requires GitHub credentials and is OFF by default. It exists because
# the project owner asked for "50 pushes spaced by 3 minutes" while running
# the experiment loop. We strongly recommend NOT using it on the public
# main branch -- artificial activity is discouraged by GitHub and pollutes
# the commit history. Prefer one solid commit per real deliverable.
#
# Usage:
#   N=50 INTERVAL_SECONDS=180 BRANCH=experiment-log scripts/drip_commits.sh
#
# Each iteration:
#   1. runs a tiny smoke experiment
#   2. appends one timestamped row to experiments/results/drip_log.csv
#   3. commits + pushes that single-line change
set -euo pipefail

N="${N:-50}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-180}"
BRANCH="${BRANCH:-experiment-log}"
LOG="experiments/results/drip_log.csv"

git fetch origin
git switch -c "$BRANCH" 2>/dev/null || git switch "$BRANCH"
mkdir -p "$(dirname "$LOG")"
[[ -f "$LOG" ]] || echo "iter,timestamp_utc,seed,val_auroc,val_c_index" > "$LOG"

for i in $(seq 1 "$N"); do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  seed="$RANDOM"
  python src/scripts/run_full_experiment.py \
    --config configs/default.yaml --smoke \
    > /tmp/mmvlm4scd_drip_${i}.log || true
  auroc=$(python - <<'PY' 2>/dev/null || echo "nan"
import json, glob, os
p = sorted(glob.glob("experiments/results/*/test_metrics.json"))[-1]
m = json.load(open(p))
print(f"{m.get('auroc_ovr', 'nan')}")
PY
)
  cidx=$(python - <<'PY' 2>/dev/null || echo "nan"
import json, glob
p = sorted(glob.glob("experiments/results/*/test_metrics.json"))[-1]
m = json.load(open(p))
print(f"{m.get('c_index', 'nan')}")
PY
)
  echo "${i},${ts},${seed},${auroc},${cidx}" >> "$LOG"
  git add "$LOG"
  git commit -m "experiment(drip): iter ${i}/${N} auroc=${auroc} c_index=${cidx}"
  git push -u origin "$BRANCH"
  if [[ "$i" -lt "$N" ]]; then sleep "$INTERVAL_SECONDS"; fi
done
echo "Drip cycle finished: $N commits pushed to $BRANCH."

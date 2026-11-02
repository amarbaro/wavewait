#!/usr/bin/env bash
# Wall time and peak RSS of `wavewait app` per run dir. usage: tools/perf-receipt.sh RUNDIR...
set -euo pipefail
ww="$(cd "$(dirname "$0")/.." && pwd)/build/wavewait"
[ -x "$ww" ] || { echo "FAIL perf: build/wavewait missing"; exit 1; }
for r in "$@"; do
  s=$(du -sh "$r" | cut -f1); m=$( { /usr/bin/time -f '%e s %M KB' "$ww" app "$r" --allow-mism >/dev/null; } 2>&1 | tail -1)
  echo "$m  $s  $r"
done

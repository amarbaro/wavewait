#!/usr/bin/env bash
# Re-walk rocprofv3 ATT run dirs and check every traced dispatch against AMD's own decoder.
# usage: tools/att-audit.sh RUNDIR...    (OLD=path/to/an/older/wavewait adds its cycles for comparison)
# One line per dispatch: run, dispatch, kernel, new cycles, old cycles, oracle verdict. Exit 1 if any oracle fails.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
ww="$here/build/wavewait"
[ -x "$ww" ] || { echo "FAIL audit: build/wavewait missing (run check.sh)"; exit 1; }
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
bad=0
for run in "$@"; do
  app=$("$ww" app "$run" 2>/dev/null) || { echo "FAIL audit: $run: wavewait app exit $?"; bad=1; continue; }
  while read -r kernel cycles n; do
    [ "$cycles" -gt 0 ] || continue
    old=-
    [ -n "${OLD:-}" ] && old=$("$OLD" app "$run" --dispatch "$n" --json 2>/dev/null | awk -F'[:,]' '/"grand"/{print $2+0}')
    code=$(ls "$run"/ui_output_*_dispatch_"$n"/code.json 2>/dev/null | head -1)
    verdict="no code.json"
    if [ -n "$code" ]; then
      "$ww" app "$run" --dispatch "$n" --json > "$tmp/w.json"
      verdict=$(python3 "$here/tools/amd_oracle.py" "$tmp/w.json" "$code" 2>&1 | tr '\n' ' ') && verdict="ok $verdict" || { verdict="FAIL $verdict"; bad=1; }
    fi
    echo "$run dispatch $n $kernel new $cycles old $old $verdict"
  done < <(awk 'NR>1 && NF>=7 && $2 ~ /^[0-9]+$/ {print $1, $2, $NF}' <<<"$app")
done
exit $bad

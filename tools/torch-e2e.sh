#!/usr/bin/env bash
# torch-e2e.sh [PYTHON]: the recipe of docs/pytorch.md on the GPU, on tools/torch_probe.py: rank the process, capture the middle
# dispatch of two kernels, walk them. PYTHON = the python of the torch venv. Fails unless the walk has waves, no mismatch, no open
# wave, and the traced dispatch ran warm: under 1.3x the kernel's median, below its first dispatch.
set -euo pipefail
cd "$(dirname "$0")/.."
py=${1:-$HOME/Projects/mojo/mojo-baro/amdhq/.venv/bin/python}
out=$PWD/.work/pt/e2e; rm -rf "$out"; mkdir -p "$out"
job=("$py" "$PWD/tools/torch_probe.py" 1500)
preload="--preload librocprofiler-sdk.so"
gpu-wait run --timeout 60 --vram 2 --label torch-rank -- /opt/rocm/bin/rocprofv3 --kernel-trace --output-format csv $preload -d "$out/rank" -- "${job[@]}" > "$out/rank.log" 2>&1 || { echo "FAIL rank: capture, log $out/rank.log"; exit 1; }
build/wavewait rank "$out/rank" --top 0 > "$out/rank.txt" || { echo "FAIL rank: wavewait rank"; exit 1; }
for k in Cijk_Ailk reduce_kernel; do
  mid=$(awk -v k="$k" 'index($0, k) { print $5; exit }' "$out/rank.txt")
  [ -n "$mid" ] || { echo "FAIL rank: no kernel matches $k"; exit 1; }
  ATT_SLIM=1 ATT_TIMEOUT=60 ATT_KERNEL=$k ATT_EXTRA="--kernel-trace --output-format csv $preload --kernel-iteration-range [$mid-$mid]" tools/att-capture.sh "$out/$k" -- "${job[@]}" | grep -v '^E2' || { echo "FAIL att: $k, log $out/$k.log"; exit 1; }
  python3 - "$out/$k" "$k" <<'P'
import csv, glob, json, statistics, subprocess, sys
run, k = sys.argv[1:3]
rows = list(csv.DictReader(open(glob.glob(run + "/*/*_kernel_trace.csv")[0])))
atts = glob.glob(run + "/*/*.att")
ids = {int(a.rsplit("_", 1)[1].split(".")[0]) for a in atts}
dur = lambda x: int(x["End_Timestamp"]) - int(x["Start_Timestamp"])
for d in sorted(ids):
    x = next(r for r in rows if int(r["Dispatch_Id"]) == d)
    same = [dur(r) for r in rows if r["Kernel_Name"] == x["Kernel_Name"]]
    app = subprocess.run(["build/wavewait", "app", run], capture_output=True, text=True, check=True).stdout
    row = next(l.split() for l in app.splitlines() if l.split() and l.split()[-1] == str(d))
    w = json.loads(subprocess.check_output(["build/wavewait", "app", run, "--dispatch", str(d), "--json"]))
    med = statistics.median(same)
    print(f"{k} dispatch {d}: {w['waves']} waves, {w['grand']} cycles, open {row[-3]}, mism {row[-2]}, {dur(x) / 1e3:.0f} us, median {med / 1e3:.0f}, first {same[0] / 1e3:.0f}")
    assert w["waves"] > 0 and row[-3] == "0" and row[-2] == "0", "FAIL walk: no waves, or open or mismatched"
    assert dur(x) < 1.3 * med, "FAIL warm: the traced dispatch ran slow"
P
done
echo "ok torch-e2e: $("$py" -c 'import torch; print(torch.__version__)' 2>/dev/null)"

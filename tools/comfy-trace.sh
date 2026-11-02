#!/usr/bin/env bash
# comfy-trace.sh OUT [KERNEL_REGEX ITERATIONS | --no-job | --bare]: one ComfyUI TRELLIS.2 mesh under rocprofv3, as a gpu-wait job; the server dies with it.
# No kernel: --kernel-trace of the whole server, then `wavewait rank OUT/kt`. A kernel: also ATT of those dispatches, OUT/att, with ITERATIONS as
# --kernel-iteration-range takes them ([750,1734]), and frames.tsv (tools/comfy_frames.py, its prompt thread's Python stack). cpu.tsv: tools/comfy_cpu.py.
# --no-job: start and stop the server only (does the kernel trace reach the disk?). --bare: no rocprofv3, the job's true stage times. OUT gets server.log, api.json, stages.tsv (tools/comfy_stages.py), mesh.glb.
# rocprofv3 writes its output on SIGINT, then the server lingers half finalized, so it gets SIGKILL as soon as the tool is done: on 2026-10-30 one lingered
# 3 min with its queues live and a GPU reset followed (cause not proven). A new GPU reset in the kernel log kills the job, with file OUT/reset-abort. Env: IMAGE (a file in ComfyUI's input/,
# default assetgen_bench_slat.png), ASSET_STEPS (default 0.75), KEEP_WAVES=1 (keep AMD's per-wave JSON for tools/att-audit.sh), COMFY, FLOW, WORKFLOW (comfy_workflow.py), MIN_FREE_GB (default 15: under it the job is killed).
set -uo pipefail
out=${1:?usage: comfy-trace.sh OUT [KERNEL_REGEX ITERATIONS | --no-job | --bare]}
regex=${2:-}; iter=${3:-}
root=$(cd "$(dirname "$0")/.." && pwd)
comfy=${COMFY:-$HOME/Projects/imports/ComfyUI-v0.43.0}
flow=${FLOW:-$HOME/Projects/imports/comfy-workflows/templates/3d_pixal3d_trellis2_image_to_model.json}
wf=${WORKFLOW:-$HOME/Projects/homeless-sim/tools/comfy_workflow.py}
image=${IMAGE:-assetgen_bench_slat.png}; steps=${ASSET_STEPS:-0.75}; floor=${MIN_FREE_GB:-15}
if [ -z "${COMFY_TRACE_INNER:-}" ]; then
  exec gpu-wait run --vram 20 --timeout 1200 --label comfy-trace -- env -u HSA_OVERRIDE_GFX_VERSION COMFY_TRACE_INNER=1 KEEP_WAVES="${KEEP_WAVES:-}" IMAGE="$image" ASSET_STEPS="$steps" MIN_FREE_GB="$floor" COMFY="$comfy" FLOW="$flow" WORKFLOW="$wf" "$0" "$@"
fi
set -m  # a job-controlled server keeps Python's SIGINT handler: a plain background job starts with SIGINT ignored
[ ! -e "$out" ] || { echo "FAIL trace: $out exists"; exit 1; }
mkdir -p "$out"; out=$(cd "$out" && pwd)
ss -ltn | grep -q ':8188 ' && { echo "FAIL trace: port 8188 is taken"; exit 1; }
free_gb() { df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9; }
since=$(date '+%F %T')
resets() { journalctl -k --since "$since" --no-pager 2> /dev/null | grep -c 'GPU reset begin'; }
cd "$comfy" || { echo "FAIL trace: no $comfy"; exit 1; }
server=(.venv/bin/python main.py --listen 127.0.0.1 --port 8188 --use-pytorch-cross-attention --reserve-vram 0.5 --disable-pinned-memory)
if [ "$regex" = --bare ]; then
  "${server[@]}" > "$out/server.log" 2>&1 &
  log=$out/server.log
elif [ -n "$regex" ] && [ "$regex" != --no-job ]; then
  ATT_INNER=1 ATT_TIMEOUT=1100 ATT_KERNEL="$regex" ATT_EXTRA="--kernel-trace --memory-copy-trace --output-format csv --preload librocprofiler-sdk.so --kernel-iteration-range $iter" \
    "$root/tools/att-capture.sh" "$out/att" -- .venv/bin/python "$root/tools/comfy_frames.py" "$out/frames.tsv" main.py "${server[@]:2}" > "$out/capture.txt" 2>&1 &
  log=$out/att.log
else
  /opt/rocm/bin/rocprofv3 --kernel-trace --memory-copy-trace --output-format csv --preload librocprofiler-sdk.so -d "$out/kt" -- "${server[@]}" > "$out/server.log" 2>&1 &
  log=$out/server.log
fi
job=$!
pid=
cleanup() { [ -z "$pid" ] || kill -KILL "$pid" 2> /dev/null; kill -KILL -- "-$job" 2> /dev/null; }
trap cleanup EXIT
for _ in $(seq 240); do curl -s 127.0.0.1:8188/system_stats > /dev/null && break; kill -0 "$job" 2> /dev/null || { echo "FAIL trace: server died at start, log $log"; exit 1; }; sleep 1; done
pid=$(ss -ltnp | grep ':8188 ' | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
[ -n "$pid" ] || { echo "FAIL trace: no server on 8188, log $log"; exit 1; }
echo "server $pid up, $(free_gb) GB free"
.venv/bin/python -I "$root/tools/comfy_cpu.py" "$pid" "$out/cpu.tsv" &
( while kill -0 "$pid" 2> /dev/null; do
    if [ "$(free_gb)" -lt "$floor" ]; then echo "FAIL disk: under $floor GB free, killing $pid"; kill -KILL "$pid"; touch "$out/disk-abort"; exit 1; fi
    if [ "$(resets)" != 0 ]; then echo "FAIL reset: GPU reset in the kernel log, killing $pid"; kill -KILL "$pid"; touch "$out/reset-abort"; exit 1; fi
    sleep 5
  done ) &
rc=0
if [ "$regex" != --no-job ]; then
  # ComfyUI sends `executing` only to the prompt's client: a private copy of the workflow script names one
  sed "s/{'prompt':api}/{'prompt':api,'client_id':'comfy_stages'}/" "$wf" > "$out/workflow.py"
  grep -q "'client_id':'comfy_stages'" "$out/workflow.py" || { echo "FAIL trace: $wf has no {'prompt':api} to add a client id to"; exit 1; }
  .venv/bin/python -I "$root/tools/comfy_stages.py" "$out/stages.tsv" "$out/api.json" &
  stages=$!
  sleep 1
  ASSET_STEPS=$steps timeout 700 .venv/bin/python -I "$out/workflow.py" "$flow" "$image" "$out/api.json" "assetgen/ww_trace" 2>&1 | grep -E 'success|FAIL|error' | cut -c1-300
  rc=${PIPESTATUS[0]}
  kill "$stages" 2> /dev/null
  echo "RUN rc=$rc $(grep -o 'Prompt executed in [0-9.]* [a-z]*' "$log" | tail -1)"
  glb=$(ls -t output/assetgen/ww_trace_*.glb 2> /dev/null | head -1)
  [ -z "$glb" ] || mv "$glb" "$out/mesh.glb"
fi
[ ! -e "$out/disk-abort" ] && [ ! -e "$out/reset-abort" ] || exit 1
if [ "$regex" = --bare ]; then
  kill -KILL "$pid"; pid=
  [ "$(resets)" = 0 ] || { touch "$out/reset-abort"; echo "FAIL reset: GPU reset in the kernel log"; exit 1; }
  echo "ok comfy-trace: bare run, workflow rc=$rc"
  exit "$rc"
fi
kill -INT "$pid"
for _ in $(seq 300); do grep -q 'found chained signal handler' "$log" && break; kill -0 "$pid" 2> /dev/null || break; sleep 1; done
grep -q 'found chained signal handler' "$log" || echo "WARN trace: no end of rocprofv3's finalization in 300 s, killing anyway"
sleep 2
kill -KILL "$pid" 2> /dev/null; pid=
wait "$job" 2> /dev/null; jobrc=$?
[ -n "${KEEP_WAVES:-}" ] || find "$out" \( -name 'se*_sm*_wv*.json' -o -name 'wstates*.json' -o -name 'realtime.json' \) -delete  # att-capture stops before ATT_SLIM on a killed server
[ "$(resets)" = 0 ] || { touch "$out/reset-abort"; echo "FAIL reset: GPU reset in the kernel log"; exit 1; }
n=$(find "$out" -name '*_kernel_trace.csv' | wc -l)
[ "$n" -gt 0 ] || { echo "FAIL trace: no kernel trace after a clean stop (server exit $jobrc), log $log"; exit 1; }
echo "ok comfy-trace: $n kernel traces, $(find "$out" -name '*.att' | wc -l) att, server exit $jobrc, workflow rc=$rc, $(du -sh "$out" | cut -f1), $(free_gb) GB free"
exit "$rc"

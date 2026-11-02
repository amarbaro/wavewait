#!/usr/bin/env bash
# att-capture.sh RUNDIR -- CMD [ARGS...]: a rocprofv3 --att capture (the input of `wavewait app RUNDIR`) that cannot hang the card.
# Defaults: CU 3 and shader-engine mask 0x1 (the only CU that emits instruction tokens on a 7900 XTX), the decoder library next
# to AMD's docs, a 256 MB buffer per engine, 300 s. Env overrides: ATT_CU ATT_SE_MASK ATT_DECODER_DIR ATT_BUFFER (bytes, 0x ok)
# ATT_TIMEOUT ATT_KERNEL (a kernel-name regex: --kernel-include-regex) ATT_ROCPROF (the rocprofv3 to run) ATT_EXTRA (more rocprofv3 flags).
# A buffer above 256 MB needs ATT_BUFFER_OK=1: rocprofv3 fails to create the ATT packet at 1 GB, then deadlocks in its signal
# handler holding the GPU for minutes. Here a watcher kills the whole process group within about 1 s of "failed to create ATT
# packet" or "Resource deadlock" in the log and exits `FAIL att: ...`. After a capture the biggest trace is printed against
# the buffer (WARN from 90 %: probably cut). ATT_SLIM=1 then deletes AMD's decoded per-wave JSON (wavewait needs only code.json and
# filenames.json; 245 MB per wave of a 100 ms conv), which leaves tools/amd_oracle.py its per-pc check only. Runs under `gpu-wait run` when it is installed (ATT_NO_GPU_WAIT=1: never).
set -euo pipefail
usage() { echo "usage: att-capture.sh RUNDIR -- CMD [ARGS...]"; exit 2; }
[ $# -ge 3 ] && [ "$2" = -- ] || usage
run=$1; shift 2
cap=$((0x10000000))
buf=$((${ATT_BUFFER:-$cap}))
tmo=${ATT_TIMEOUT:-300}
if [ "$buf" -gt "$cap" ] && [ "${ATT_BUFFER_OK:-}" != 1 ]; then
  echo "FAIL att: buffer $buf bytes is above the 256 MB cap (rocprofv3 fails to create the ATT packet and then hangs the GPU); ATT_BUFFER_OK=1 overrides"; exit 1
fi
if [ -z "${ATT_INNER:-}" ] && [ -z "${ATT_NO_GPU_WAIT:-}" ] && command -v gpu-wait > /dev/null; then  # the capture itself is the GPU job
  fwd=(); for v in "${!ATT_@}"; do fwd+=("$v=${!v}"); done  # gpu-wait's job does not inherit this shell's env
  exec gpu-wait run --timeout $((tmo + 30)) --vram 2 --label att-capture -- env ATT_INNER=1 "${fwd[@]}" "$0" "$run" -- "$@"
fi
rocprof=${ATT_ROCPROF:-$(command -v rocprofv3 || echo /opt/rocm/bin/rocprofv3)}
[ -x "$rocprof" ] || { echo "FAIL att: no rocprofv3 ($rocprof)"; exit 1; }
dec=${ATT_DECODER_DIR:-}
if [ -z "$dec" ]; then
  for d in "$HOME/Projects/AMD/docs/sqtt/decoder-bin" /opt/rocm/lib /opt/rocm/lib/rocprofiler-sdk; do
    [ -e "$d/librocprof-trace-decoder.so" ] && { dec=$d; break; }
  done
fi
[ -n "$dec" ] && [ -e "$dec/librocprof-trace-decoder.so" ] || { echo "FAIL att: librocprof-trace-decoder.so not found (set ATT_DECODER_DIR)"; exit 1; }
if [ -d "$run" ] && [ -n "$(ls -A "$run")" ]; then echo "FAIL att: $run is not empty (a second capture would mix its dispatches with the first)"; exit 1; fi
mkdir -p "$(dirname "$run")"
log=$run.log; rm -f "$log"
args=(--att --att-target-cu "${ATT_CU:-3}" --att-shader-engine-mask "${ATT_SE_MASK:-0x1}" --att-buffer-size "$(printf '0x%x' "$buf")" --att-library-path "$dec")
[ -z "${ATT_KERNEL:-}" ] || args+=(--kernel-include-regex "$ATT_KERNEL")
read -ra extra <<<"${ATT_EXTRA:-}"
args+=("${extra[@]}")
set -m  # the capture is its own process group: one kill takes rocprofv3, its tool processes and the program
timeout -s KILL "$tmo" "$rocprof" "${args[@]}" -d "$run" -- "$@" > "$log" 2>&1 &
pid=$!
bad='failed to create ATT packet|Resource deadlock'
while kill -0 "$pid" 2> /dev/null; do
  if grep -qE "$bad" "$log"; then
    kill -KILL -- "-$pid" 2> /dev/null || true
    { wait "$pid"; } 2> /dev/null || true
    echo "FAIL att: rocprofv3 logged '$(grep -m1 -oE "$bad" "$log")' (buffer $buf bytes): killed its process group, log $log"; exit 1
  fi
  sleep 0.5
done
rc=0; { wait "$pid"; } 2> /dev/null || rc=$?
if grep -qE "$bad" "$log"; then echo "FAIL att: rocprofv3 logged '$(grep -m1 -oE "$bad" "$log")' (buffer $buf bytes), log $log"; exit 1; fi
[ "$rc" != 137 ] || { echo "FAIL att: killed after $tmo s (ATT_TIMEOUT), log $log"; exit 1; }
[ "$rc" = 0 ] || { echo "FAIL att: rocprofv3 exit $rc, log $log"; exit 1; }
big=0; n=0
for f in "$run"/*/*_shader_engine_*.att; do
  [ -s "$f" ] || continue
  n=$((n + 1)); s=$(stat -c %s "$f"); [ "$s" -le "$big" ] || big=$s
done
[ "$n" -gt 0 ] || { echo "FAIL att: no .att file under $run (no workgroup ran on CU ${ATT_CU:-3}? the kernel needs 192+ workgroups), log $log"; exit 1; }
pct=$((big * 100 / buf))
echo "att: $n traces under $run, the biggest $((big / 1024)) KiB of a $((buf / 1024)) KiB buffer ($pct%)"
[ -z "${ATT_SLIM:-}" ] || find "$run" \( -name 'se*_sm*_wv*.json' -o -name 'wstates*.json' -o -name 'realtime.json' \) -delete
[ "$pct" -lt 90 ] || echo "WARN att: the biggest trace fills $pct% of the buffer: probably truncated, wavewait will warn of open waves (a bigger ATT_BUFFER, up to 256 MB, or ATT_KERNEL to trace fewer dispatches)"

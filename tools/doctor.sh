#!/usr/bin/env bash
# doctor.sh: is this machine ready to capture? One line per need, with the fix. Exit 1 if any fails.
set -uo pipefail
cd "$(dirname "$0")/.."
bad=0
need() { if (eval "$2") >/dev/null 2>&1; then echo "ok   $1"; else echo "FAIL $1: $3"; bad=1; fi; }
rp=${ATT_ROCPROF:-$(command -v rocprofv3 || echo /opt/rocm/bin/rocprofv3)}
dirs=${ATT_DECODER_DIR:-$HOME/Projects/AMD/docs/sqtt/decoder-bin /opt/rocm/lib /opt/rocm/lib/rocprofiler-sdk}
need "wavewait built" '[ -x build/wavewait ]' "pixi run check"
need "llvm-objdump" '[ -x /usr/bin/llvm-objdump ]' "install the system llvm package"
need "rocprofv3" "[ -x '$rp' ]" "install ROCm, or set ATT_ROCPROF"
need "decoder library" "for d in $dirs; do [ -e \$d/librocprof-trace-decoder.so ] && exit 0; done; exit 1" "get rocprof-trace-decoder, set ATT_DECODER_DIR to its dir"
need "/dev/kfd access" '[ -r /dev/kfd ] && [ -w /dev/kfd ]' "add your user to the render group"
need "gfx1100 GPU" "grep -qs 'gfx_target_version 110000' /sys/class/kfd/kfd/topology/nodes/*/properties" "wavewait reads gfx1100 traces only (RX 7900 XTX)"
exit $bad

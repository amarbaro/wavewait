#!/usr/bin/env bash
# from-att.sh RUNDIR OUT [CODE_OBJECT_ID]: a rocprofv3 --att capture as build/wavewait input (no tinygrad at capture).
# RUNDIR = rocprofv3 -d dir; OUT gets blob_N.bin (the .att files, unchanged: they are raw RDNA3 SQTT) and
# pcmap.tsv from the kernel's code object (default id 2; 1 is usually the runtime's fill kernel).
set -euo pipefail
run=${1:?usage: from-att.sh RUNDIR OUT [CODE_OBJECT_ID]} out=${2:?OUT} co=${3:-2}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
n=0
for f in "$run"/*/*_shader_engine_*.att; do [ -s "$f" ] || continue; cp "$f" "$out/blob_$n.bin"; n=$((n + 1)); done
[ "$n" -gt 0 ] || { echo "FAIL att: no .att files under $run"; exit 1; }
nd=$(ls "$run"/*/*_shader_engine_*.att | sed -E 's/.*_([0-9]+)\.att$/\1/' | sort -u | wc -l)
[ "$nd" -le 1 ] || echo "WARN from-att: $nd dispatches merged into one table against one code object; use: wavewait app $run --dispatch N" >&2
obj=$(ls "$run"/*/*_code_object_id_"$co".out 2>/dev/null | head -1)
[ -n "$obj" ] || { echo "FAIL code object: no *_code_object_id_$co.out under $run"; exit 1; }
python3 -c "import sys; sys.path.insert(0, '$here'); from export import pc_map
with open('$out/pcmap.tsv', 'w') as f:
  for pc, (size, text, word) in sorted(pc_map(open('$obj', 'rb').read()).items()): f.write(f'{pc}\t{size}\t{word}\t{text}\n')"
echo "ok from-att: $n blobs, pcmap from $(basename "$obj") -> $out"

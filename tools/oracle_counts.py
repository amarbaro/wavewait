"""oracle_counts.py BLOB CODEOBJ: executions per pc from tinygrad's map_insts (the pc walk oracle), one "pc count"
line per pc. Wave packets are kept for CU 3 only (the traced WGP; the SE reports every WGP's waves) and renamed to
CU 0, the one tinygrad's walk keeps."""
import collections
import sys
import tinygrad.renderer.amd.sqtt as sqtt

decode = sqtt.decode


def traced_only(data):
  for p in decode(data):
    if hasattr(p, 'wgp'):
      if p.cu != 3: continue
      p.wgp, p.sa = 0, 0
    yield p


sqtt.decode = traced_only
n = collections.Counter(info.pc for _, info in sqtt.map_insts(open(sys.argv[1], 'rb').read(), open(sys.argv[2], 'rb').read(), 'gfx1100') if info)
for pc in sorted(n): print(pc, n[pc])

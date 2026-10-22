# Probe I: instruction-cache fetch. Two 256 B-window crossings in one kernel.
# Region 1: short VALU run after the s_load wait, then cross 0x200 with no further DRAM wait.
#   Hypothesis H0 (256 B window, fetch on demand, no prefetch): 475-cycle stall at the first
#   instruction in 0x200.. (as in probe_bank, pc 0x200).
# Region 2: an s_load wait immediately before crossing 0x300, giving the front end ~600 cycles
#   to fetch the next window while the wave is stalled on DRAM.
#   H0 predicts the stall disappears (0) if the fetch of 0x300.. completed during the wait;
#   if it does not, H0 predicts another ~475 stall.
# Run: PYTHONPATH=. DEV=MOCK+AMD python3 .work/sqtt/probes/icache.py  (emulator smoke)
# Capture: SQTT_WGP_PROBE=3 tools/sqtt-capture $PWD/probes/icache.py
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from tinygrad.renderer.amd.dsl import s, v, NULL

def icache(A: UOp) -> UOp:
  A = A.flatten()
  threads, wg = UOp.special(32, "lidx0"), UOp.special(1, "gidx0")
  w = s_waitcnt_lgkmcnt(sdst=NULL, simm16=0)
  # SQTT_PF=1..3 arm: set the ISA prefetch distance (64 B lines ahead of PC) first
  pf = getenv("SQTT_PF", 0)
  va = lambda: v_add_f32_e32(v[52], v[52], v[2])
  def fill_to(insts, boundary):  # VALUs until the next instruction starts at byte offset `boundary` of the code
    while sum(i.size() for i in insts) < boundary: insts.append(va())
    assert sum(i.size() for i in insts) == boundary
    return insts
  insts = ([s_set_inst_prefetch_distance(pf)] if pf else []) + [s_load_b64(s[0:1], s[0:1], soffset=NULL), w,
                                                              v_lshlrev_b32_e32(v[1], 2, v[0])]
  fill_to(insts, 0x100)
  insts += [va()]                                               # region 1: first instruction of the next 256 B
  insts += [s_load_b64(s[4:5], s[0:1], soffset=NULL), w]        # region 2: DRAM wait, then cross 0x200
  fill_to(insts, 0x200)
  insts += [va(), global_store_b32(addr=v[1], data=v[52], saddr=s[0:1]), s_endpgm()]
  sink = UOp.sink(A.base, threads, wg, arg=KernelInfo("probe_icache"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=(x, dtypes.void)) for x in insts))))

if __name__ == "__main__":
  a = Tensor.empty(32, dtype=dtypes.int32).contiguous().realize()
  Tensor.custom_kernel(a, fxn=icache)[0].realize()

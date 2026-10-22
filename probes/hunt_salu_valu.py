# Probe: SALU and branch consuming a VALU-written SGPR with no s_delay_alu.
# v_cmp writes VCC, s_cbranch_vccnz reads it; v_readfirstlane writes s2, s_add reads it.
# Rule A: no issue stall (delta 1 after the producer). Rule B: wait for the VALU writeback.
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from tinygrad.renderer.amd.dsl import s, v, NULL, VCC_LO

def salu_valu(A:UOp) -> UOp:
  A = A.flatten()
  threads, wg = UOp.special(32, "lidx0"), UOp.special(1, "gidx0")
  w = s_waitcnt_lgkmcnt(sdst=NULL, simm16=0)
  insts = [s_load_b64(s[0:1], s[0:1], soffset=NULL), w, v_lshlrev_b32_e32(v[1], 2, v[0])]
  insts += [v_readfirstlane_b32_e32(s[2], v[1]), s_add_u32(s[3], s[2], 1)]          # VALU -> SGPR -> SALU
  insts += [s_nop(0)] * 4
  insts += [v_readfirstlane_b32_e32(s[4], v[1]), s_nop(0), s_add_u32(s[5], s[4], 1)]  # one cycle apart
  insts += [s_nop(0)] * 4
  insts += [v_cmp_gt_u32_e32(16, v[0]), s_cbranch_vccz(simm16=0)]                    # VALU -> VCC -> branch
  insts += [s_nop(0)] * 4
  insts += [v_mov_b32_e32(v[2], s[3]), global_store_b32(addr=v[1], data=v[2], saddr=s[0:1]), s_endpgm()]
  sink = UOp.sink(A.base, threads, wg, arg=KernelInfo("probe_salu_valu"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=(x, dtypes.void)) for x in insts))))

if __name__ == "__main__":
  a = Tensor.empty(32, dtype=dtypes.int32).contiguous().realize()
  Tensor.custom_kernel(a, fxn=salu_valu)[0].realize()

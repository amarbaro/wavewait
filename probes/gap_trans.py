# Probe: transcendental unit. Independent v_exp/v_rcp back to back, interleaved with plain VALU, then dependent chains.
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from tinygrad.renderer.amd.dsl import s, v, NULL

def gap_trans(A:UOp) -> UOp:
  A = A.flatten()
  threads, wg = UOp.special(32, "lidx0"), UOp.special(1, "gidx0")
  w = s_waitcnt_lgkmcnt(sdst=NULL, simm16=0)
  insts = [s_load_b64(s[0:1], s[0:1], soffset=NULL), w, v_lshlrev_b32_e32(v[1], 2, v[0]), v_cvt_f32_u32_e32(v[2], v[0])]
  insts += [v_exp_f32_e32(v[10+i], v[2]) for i in range(4)]                                   # 4 independent trans
  insts += [v_rcp_f32_e32(v[14], v[2]), v_add_f32_e32(v[15], v[2], v[2]), v_rcp_f32_e32(v[16], v[2]), v_add_f32_e32(v[17], v[2], v[2])]
  insts += [v_exp_f32_e32(v[20], v[2]), v_exp_f32_e32(v[21], v[20]), v_exp_f32_e32(v[22], v[21])]       # trans -> trans chain
  insts += [v_rcp_f32_e32(v[23], v[2]), v_add_f32_e32(v[24], v[23], v[2])]                            # trans -> VALU
  insts += [v_add_f32_e32(v[25], v[2], v[2]), v_sqrt_f32_e32(v[26], v[25])]                           # VALU -> trans
  insts += [v_add_f32_e32(v[3], v[24], v[26]), global_store_b32(addr=v[1], data=v[3], saddr=s[0:1]), s_endpgm()]
  sink = UOp.sink(A.base, threads, wg, arg=KernelInfo("probe_gap_trans"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=(x, dtypes.void)) for x in insts))))

if __name__ == "__main__":
  a = Tensor.empty(32, dtype=dtypes.int32).contiguous().realize()
  Tensor.custom_kernel(a, fxn=gap_trans)[0].realize()

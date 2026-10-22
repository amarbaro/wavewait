# Probe: WMMA. 4 independent v_wmma back to back, then a dependent chain of 3 (D -> C), then a VALU reading the result.
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from tinygrad.renderer.amd.dsl import s, v, NULL

def gap_wmma(A:UOp) -> UOp:
  A = A.flatten()
  threads, wg = UOp.special(32, "lidx0"), UOp.special(1, "gidx0")
  w = s_waitcnt_lgkmcnt(sdst=NULL, simm16=0)
  insts = [s_load_b64(s[0:1], s[0:1], soffset=NULL), w, v_lshlrev_b32_e32(v[1], 2, v[0])]
  insts += [v_mov_b32_e32(v[i], 0) for i in range(8, 24)]
  insts += [v_wmma_f32_16x16x16_f16(v[32+8*i:39+8*i], v[8:15], v[16:23], v[32+8*i:39+8*i]) for i in range(4)]   # independent
  insts += [v_nop_e32()] * 2
  insts += [v_wmma_f32_16x16x16_f16(v[64:71], v[8:15], v[16:23], v[64:71]) for _ in range(3)]                   # D feeds C
  insts += [v_add_f32_e32(v[2], v[64], v[1]), global_store_b32(addr=v[1], data=v[2], saddr=s[0:1]), s_endpgm()]
  sink = UOp.sink(A.base, threads, wg, arg=KernelInfo("probe_gap_wmma"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=(x, dtypes.void)) for x in insts))))

if __name__ == "__main__":
  a = Tensor.empty(32, dtype=dtypes.int32).contiguous().realize()
  Tensor.custom_kernel(a, fxn=gap_wmma)[0].realize()

# Probe: EXEC=0. v_cmpx false on every lane, then VALU, LDS store, LDS load, global store and a wait; then EXEC restored.
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.dtype import AddrSpace
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from tinygrad.renderer.amd.dsl import s, v, NULL, EXEC_LO

def gap_exec0(A:UOp) -> UOp:
  A = A.flatten()
  threads, wg = UOp.special(32, "lidx0"), UOp.special(1, "gidx0")
  lds = UOp.placeholder((512,), dtypes.uint8, 0, AddrSpace.LOCAL)
  w = s_waitcnt_lgkmcnt(sdst=NULL, simm16=0)
  insts = [s_load_b64(s[0:1], s[0:1], soffset=NULL), w, v_lshlrev_b32_e32(v[1], 2, v[0]), s_mov_b32(s[4], EXEC_LO)]
  insts += [v_cmpx_gt_u32_e32(0, v[0])]                              # 0 > lane never true: EXEC = 0
  insts += [v_add_nc_u32_e32(v[2], 1, v[0]), ds_store_b32(addr=v[1], data0=v[2]), ds_load_b32(vdst=v[3], addr=v[1]), w]
  insts += [global_store_b32(addr=v[1], data=v[2], saddr=s[0:1]), s_mov_b32(EXEC_LO, s[4])]
  insts += [v_add_nc_u32_e32(v[2], 1, v[0]), global_store_b32(addr=v[1], data=v[2], saddr=s[0:1]), s_endpgm()]
  sink = UOp.sink(A.base, lds, threads, wg, arg=KernelInfo("probe_gap_exec0"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=(x, dtypes.void)) for x in insts))))

if __name__ == "__main__":
  a = Tensor.empty(32, dtype=dtypes.int32).contiguous().realize()
  Tensor.custom_kernel(a, fxn=gap_exec0)[0].realize()

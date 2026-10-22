# Probe: LDS bank conflicts. b32 loads with lane stride 1, 2, 4, 8, 16, 32 dwords (conflict degree 1..32), one wait each.
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.dtype import AddrSpace
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from tinygrad.renderer.amd.dsl import s, v, NULL

def gap_bank(A:UOp) -> UOp:
  A = A.flatten()
  threads, wg = UOp.special(32, "lidx0"), UOp.special(1, "gidx0")
  lds = UOp.placeholder((32*32*4,), dtypes.uint8, 0, AddrSpace.LOCAL)
  w = s_waitcnt_lgkmcnt(sdst=NULL, simm16=0)
  insts = [s_load_b64(s[0:1], s[0:1], soffset=NULL), w, v_lshlrev_b32_e32(v[1], 2, v[0])]
  for k, sh in enumerate((2, 3, 4, 5, 6, 7)):                       # byte stride 4 << k
    insts += [v_lshlrev_b32_e32(v[4], sh, v[0]), ds_load_b32(vdst=v[10+k], addr=v[4]), w]
  for k, sh in enumerate((2, 4, 7)):                                 # stores with stride 1, 4, 32 dwords
    insts += [v_lshlrev_b32_e32(v[4], sh, v[0]), ds_store_b32(addr=v[4], data0=v[1]), w]
  insts += [v_add_nc_u32_e32(v[2], v[10], v[15]), global_store_b32(addr=v[1], data=v[2], saddr=s[0:1]), s_endpgm()]
  sink = UOp.sink(A.base, lds, threads, wg, arg=KernelInfo("probe_gap_bank"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=(x, dtypes.void)) for x in insts))))

if __name__ == "__main__":
  a = Tensor.empty(32, dtype=dtypes.int32).contiguous().realize()
  Tensor.custom_kernel(a, fxn=gap_bank)[0].realize()

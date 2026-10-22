# Probe: wide LDS loads. 4 back-to-back ds_load_b128 (and b64) then a wait, one wave per SIMD pair slot.
# Rule A: every load occupies the LDS server 1 cycle. Rule B: occupancy grows with width (b128 = 4).
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.dtype import AddrSpace
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from tinygrad.renderer.amd.dsl import s, v, NULL

def ds_wide(A:UOp) -> UOp:
  A = A.flatten()
  threads, wg = UOp.special(32, "lidx0"), UOp.special(1, "gidx0")
  lds = UOp.placeholder((8192,), dtypes.uint8, 0, AddrSpace.LOCAL)
  w = s_waitcnt_lgkmcnt(sdst=NULL, simm16=0)
  insts = [s_load_b64(s[0:1], s[0:1], soffset=NULL), w, v_lshlrev_b32_e32(v[1], 4, v[0]), v_lshlrev_b32_e32(v[20], 2, v[0])]
  insts += [ds_load_b128(vdst=v[2+4*i:5+4*i], addr=v[1], offset0=0) for i in range(4)] + [w]
  insts += [ds_load_b64(vdst=v[2+2*i:3+2*i], addr=v[1], offset0=0) for i in range(4)] + [w]
  insts += [ds_load_b32(vdst=v[2+i], addr=v[20]) for i in range(4)] + [w]
  insts += [global_store_b32(addr=v[20], data=v[2], saddr=s[0:1]), s_endpgm()]
  sink = UOp.sink(A.base, lds, threads, wg, arg=KernelInfo("probe_ds_wide"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=(x, dtypes.void)) for x in insts))))

if __name__ == "__main__":
  a = Tensor.empty(32, dtype=dtypes.int32).contiguous().realize()
  Tensor.custom_kernel(a, fxn=ds_wide)[0].realize()

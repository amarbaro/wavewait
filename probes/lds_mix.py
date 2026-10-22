# Probe A: LDS service order. 4 waves (2 per SIMD) issue back-to-back stores and loads of mixed width,
# so requests from both SIMDs and both slots tie at the LDS unit. Run: tools/sqtt-capture $PWD/probes/lds_mix.py
import functools
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.dtype import AddrSpace
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from tinygrad.renderer.amd.dsl import s, v, NULL

def lds_mix(A:UOp) -> UOp:
  A = A.flatten()
  threads, wg = UOp.special(128, "lidx0"), UOp.special(1, "gidx0")
  lds = UOp.placeholder((2048,), dtypes.uint8, 0, AddrSpace.LOCAL)
  w = s_waitcnt_lgkmcnt(sdst=NULL, simm16=0)
  insts = [
    s_load_b64(s[0:1], s[0:1], soffset=NULL), w,
    v_lshlrev_b32_e32(v[1], 2, v[0]),   # 4 B per lane: conflict-free b32 accesses
    v_add_nc_u32_e32(v[15], 512, v[1]), v_add_nc_u32_e32(v[16], 1024, v[1]),
    v_mov_b32_e32(v[2], 1), v_mov_b32_e32(v[3], 2),
    # burst 1: store, load, store, load
    ds_store_b32(addr=v[1], data0=v[2]), ds_load_b32(vdst=v[6], addr=v[1]),
    ds_store_b32(addr=v[15], data0=v[3]), ds_load_b32(vdst=v[7], addr=v[15]), w,
    # burst 2: three stores then three loads
    ds_store_b32(addr=v[1], data0=v[2]), ds_store_b32(addr=v[15], data0=v[3]), ds_store_b32(addr=v[16], data0=v[2]),
    ds_load_b32(vdst=v[6], addr=v[1]), ds_load_b32(vdst=v[7], addr=v[15]), ds_load_b32(vdst=v[8], addr=v[16]), w,
    # burst 3: loads first, then stores
    ds_load_b32(vdst=v[6], addr=v[1]), ds_load_b32(vdst=v[7], addr=v[15]),
    ds_store_b32(addr=v[16], data0=v[2]), ds_store_b32(addr=v[1], data0=v[3]), w,
    v_add_nc_u32_e32(v[6], v[6], v[7]),
    global_store_b32(addr=v[1], data=v[6], saddr=s[0:1]),
    s_endpgm(),
  ]
  sink = UOp.sink(A.base, lds, threads, wg, arg=KernelInfo("probe_lds_mix"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=(x, dtypes.void)) for x in insts))))

if __name__ == "__main__":
  a = Tensor.empty(128, dtype=dtypes.int32).contiguous().realize()
  Tensor.custom_kernel(a, fxn=lds_mix)[0].realize()

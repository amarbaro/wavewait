# Probe: gemm's layout. Long DRAM wait (global load + vmcnt) inside line 2, a few instructions, then cross into line 4.
# Single wave, so no other wave can warm the cache. Stall at the crossing -> gemm was warmed by other waves;
# no stall -> the prefetch rule is wrong (prefetch started before the wait).
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from tinygrad.renderer.amd.dsl import s, v, NULL

def gemmlike(A:UOp) -> UOp:
  A = A.flatten()
  threads, wg = UOp.special(32, "lidx0"), UOp.special(1, "gidx0")
  w = s_waitcnt_lgkmcnt(sdst=NULL, simm16=0)
  va = lambda: v_add_nc_u32_e32(v[10], 1, v[10])
  def fill_to(insts, boundary):
    while sum(i.size() for i in insts) < boundary: insts.append(va())
    assert sum(i.size() for i in insts) == boundary
  insts = [s_load_b64(s[0:1], s[0:1], soffset=NULL), w, v_lshlrev_b32_e32(v[1], 2, v[0])]
  fill_to(insts, 0x88)                                                   # into line 2, like gemm's waits at 0x88/0x98
  insts += [global_load_b32(vdst=v[5], addr=v[1], saddr=s[0:1]), s_waitcnt_vmcnt(sdst=NULL, simm16=0)]
  fill_to(insts, 0x100)
  insts += [va(), global_store_b32(addr=v[1], data=v[10], saddr=s[0:1]), s_endpgm()]
  sink = UOp.sink(A.base, threads, wg, arg=KernelInfo("probe_icache_gemmlike"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=(x, dtypes.void)) for x in insts))))

if __name__ == "__main__":
  a = Tensor.empty(32, dtype=dtypes.int32).contiguous().realize()
  Tensor.custom_kernel(a, fxn=gemmlike)[0].realize()

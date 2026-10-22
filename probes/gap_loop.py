# Probe: icache in a loop. A 3-instruction loop body spread so each pass crosses into a new 256 B block and back.
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from tinygrad.renderer.amd.dsl import s, v, NULL
from extra.gemm.amd_asm_matmul import Kernel

def gap_loop(A:UOp) -> UOp:
  A = A.flatten()
  threads, wg = UOp.special(32, "lidx0"), UOp.special(1, "gidx0")
  k = Kernel()
  k.emit(s_load_b64(s[0:1], s[0:1], soffset=NULL)); k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(v_lshlrev_b32_e32(v[1], 2, v[0])); k.emit(s_mov_b32(s[3], 0))
  k.label("top")
  k.emit(s_add_u32(s[3], s[3], 1))
  for _ in range(120): k.emit(v_add_nc_u32_e32(v[10], 1, v[10]))   # 480 B of body: the loop spans two blocks
  k.emit(s_cmp_lt_u32(s[3], 4)); k.emit(s_cbranch_scc1(), target="top")
  k.emit(global_store_b32(addr=v[1], data=v[10], saddr=s[0:1])); k.emit(s_endpgm())
  sink = UOp.sink(A.base, threads, wg, arg=KernelInfo("probe_gap_loop"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=(x, dtypes.void)) for x in k.finalize()))))

if __name__ == "__main__":
  a = Tensor.empty(32, dtype=dtypes.int32).contiguous().realize()
  Tensor.custom_kernel(a, fxn=gap_loop)[0].realize()

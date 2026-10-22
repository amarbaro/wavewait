from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from tinygrad.renderer.amd.dsl import s, v, NULL

def k(A: UOp) -> UOp:
  A = A.flatten()
  threads, wg = UOp.special(32, "lidx0"), UOp.special(1, "gidx0")
  insts = [
    s_nop(simm16=0), s_nop(simm16=5), s_sleep(simm16=1),
    v_mov_b32_e32(v[0], 1), s_nop(simm16=0), s_endpgm(),
  ]
  sink = UOp.sink(A.base, threads, wg, arg=KernelInfo("hunt_nop_sleep"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=(x, dtypes.void)) for x in insts))))

if __name__ == "__main__":
  a = Tensor.empty(32, dtype=dtypes.int32).contiguous().realize()
  Tensor.custom_kernel(a, fxn=k)[0].realize()

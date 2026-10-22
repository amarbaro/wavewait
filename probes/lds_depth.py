# Probe D: LDS request queue depth and same-port cross-wave issue. 4 waves; waves on simd 2 (odd wave id) exit at once,
# so only simd 0 (slots 0 and 1) issues. Part 1: 12 back-to-back loads, then 12 back-to-back stores (issue stall = queue depth).
# Part 2: both simd-0 waves issue one store within a cycle of each other (the sync s0w1 anomaly, wait 8 instead of 3).
# Run: SQTT_WGP_PROBE=3 tools/sqtt-capture $PWD/probes/lds_depth.py
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.dtype import AddrSpace
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from tinygrad.renderer.amd.dsl import s, v, NULL
from extra.gemm.amd_asm_matmul import Kernel

def lds_depth(A:UOp) -> UOp:
  A = A.flatten()
  threads, wg = UOp.special(128, "lidx0"), UOp.special(1, "gidx0")
  lds = UOp.placeholder((4096,), dtypes.uint8, 0, AddrSpace.LOCAL)
  w = s_waitcnt_lgkmcnt(sdst=NULL, simm16=0)
  k = Kernel()
  k.emit(s_load_b64(s[0:1], s[0:1], soffset=NULL)); k.emit(w)
  k.emit(v_lshrrev_b32_e32(v[1], 5, v[0]))            # wave id
  k.emit(v_and_b32_e32(v[2], 1, v[1]))                 # odd wave id -> simd 2
  k.emit(v_lshlrev_b32_e32(v[3], 2, v[0]))             # byte address
  k.emit(v_mov_b32_e32(v[4], 7))
  k.emit(v_cmpx_eq_u32_e32(0, v[2]))                   # keep only even waves (simd 0)
  k.emit(s_cbranch_execz(), target="end")
  for i in range(12): k.emit(ds_load_b32(vdst=v[8 + i], addr=v[3], offset0=16 * i))   # part 1a: 12 loads back to back
  k.emit(w)
  for i in range(12): k.emit(ds_store_b32(addr=v[3], data0=v[4], offset0=16 * i))     # part 1b: 12 stores back to back
  k.emit(w)
  k.emit(v_add_f32_e32(v[5], v[8], v[9]))              # VALU so both waves re-align on the mem-after-VALU rule
  k.emit(ds_store_b32(addr=v[3], data0=v[5]))          # part 2: one store per wave, waves 0 and 2 issue ~1 cycle apart
  k.emit(w)
  k.emit(global_store_b32(addr=v[3], data=v[5], saddr=s[0:1]))
  k.label("end")
  k.emit(s_endpgm())
  insts = k.finalize()
  sink = UOp.sink(A.base, lds, threads, wg, arg=KernelInfo("probe_lds_depth"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=(x, dtypes.void)) for x in insts))))

if __name__ == "__main__":
  a = Tensor.empty(128, dtype=dtypes.int32).contiguous().realize()
  Tensor.custom_kernel(a, fxn=lds_depth)[0].realize()

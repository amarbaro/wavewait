# Probe B: barrier release. 4 waves (2 per SIMD). Each wave spins a scalar loop whose count depends on its wave id,
# so waves reach each s_barrier at different, measured times. Six barriers, each isolating one question (see
# .work/fable-notes.md, round 4). Run: SQTT_WGP_PROBE=3 tools/sqtt-capture $PWD/probes/barrier.py
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.dtype import AddrSpace
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from tinygrad.renderer.amd.dsl import s, v, NULL
from extra.gemm.amd_asm_matmul import Kernel

def barrier(A:UOp) -> UOp:
  A = A.flatten()
  threads, wg = UOp.special(128, "lidx0"), UOp.special(1, "gidx0")
  lds = UOp.placeholder((4096,), dtypes.uint8, 0, AddrSpace.LOCAL)
  w = s_waitcnt_lgkmcnt(sdst=NULL, simm16=0)
  k = Kernel()
  def spin(name, count_sgpr):  # loop max(1, count) times: s_add, s_cmp, s_cbranch (3 SALU per iteration)
    k.emit(s_mov_b32(s[3], 0))
    k.label(name)
    k.emit(s_add_u32(s[3], s[3], 1))
    k.emit(s_cmp_lt_u32(s[3], count_sgpr))
    k.emit(s_cbranch_scc1(), target=name)
  k.emit(s_load_b64(s[0:1], s[0:1], soffset=NULL)); k.emit(w)
  k.emit(v_lshrrev_b32_e32(v[1], 5, v[0]))            # v1 = wave id 0..3
  k.emit(v_readfirstlane_b32_e32(s[2], v[1]))          # s2 = wave id
  k.emit(v_lshlrev_b32_e32(v[14], 2, v[0]))            # byte address per lane
  k.emit(v_mov_b32_e32(v[2], 1.0)); k.emit(v_mov_b32_e32(v[3], 2.0))
  k.emit(s_lshl_b32(s[4], s[2], 4))                    # s4 = 16*wid
  k.emit(s_sub_i32(s[5], 48, s[4]))                    # s5 = 16*(3-wid)
  # B1: staggered arrival, wave 3 last (SALU loop then s_barrier)
  spin("l1", s[4]); k.emit(s_barrier())
  # B2: staggered arrival, wave 0 last
  spin("l2", s[5]); k.emit(s_barrier())
  # B3: no work in between: all waves arrive together
  k.emit(s_barrier())
  # B4: wave 3 last, its last pre-barrier instruction is a ds_store with no waitcnt (memory outstanding at arrival)
  spin("l4", s[4]); k.emit(ds_store_b32(addr=v[14], data0=v[2])); k.emit(s_barrier())
  k.emit(w)
  # B5: wave 3 last, arrival preceded by a 3-deep dependent FP VALU chain (arrival = issue or completion?)
  spin("l5", s[4]); k.emit(v_add_f32_e32(v[4], v[2], v[3])); k.emit(v_add_f32_e32(v[5], v[4], v[3])); k.emit(v_add_f32_e32(v[6], v[5], v[3]))
  k.emit(s_barrier())
  # B6: two barriers back to back
  k.emit(s_barrier())
  k.emit(v_add_f32_e32(v[7], v[6], v[2]))
  k.emit(global_store_b32(addr=v[14], data=v[7], saddr=s[0:1]))
  k.emit(s_endpgm())
  insts = k.finalize()
  sink = UOp.sink(A.base, lds, threads, wg, arg=KernelInfo("probe_barrier"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=(x, dtypes.void)) for x in insts))))

if __name__ == "__main__":
  a = Tensor.empty(128, dtype=dtypes.int32).contiguous().realize()
  Tensor.custom_kernel(a, fxn=barrier)[0].realize()

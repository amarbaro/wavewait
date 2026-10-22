# Probe C: VGPR bank penalty (R2c) and per-op VALU latency. One wave (local 32), no cross-wave pipe effects.
# Bank = vgpr index mod 4. Groups G1..G13 described in .work/fable-notes.md (round 4).
# Run: SQTT_WGP_PROBE=3 tools/sqtt-capture $PWD/probes/bank.py
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from tinygrad.renderer.amd.dsl import s, v, NULL

def bank(A:UOp) -> UOp:
  A = A.flatten()
  threads, wg = UOp.special(32, "lidx0"), UOp.special(1, "gidx0")
  w = s_waitcnt_lgkmcnt(sdst=NULL, simm16=0)
  add = v_add_f32_e32
  insts = [
    s_load_b64(s[0:1], s[0:1], soffset=NULL), w,
    v_lshlrev_b32_e32(v[1], 2, v[0]),
    # constants: v2 b2, v3 b3, v4 b0, v5 b1, v6 b2, v7 b3, v8 b0, v9 b1
    v_mov_b32_e32(v[2], 1.0), v_mov_b32_e32(v[3], 2.0), v_mov_b32_e32(v[4], 0.5), v_mov_b32_e32(v[5], 4.0),
    v_mov_b32_e32(v[6], 1.0), v_mov_b32_e32(v[7], 2.0), v_mov_b32_e32(v[8], 0.5), v_mov_b32_e32(v[9], 4.0),
    # G1 control: non-forwarded source in a different bank than the producer's dest
    add(v[10], v[2], v[3]), add(v[11], v[10], v[5]), add(v[12], v[11], v[6]),          # v10 b2, v5 b1 | v11 b3, v6 b2
    # G2 penalty: non-forwarded source (src1) in the producer's dest bank
    add(v[13], v[2], v[3]), add(v[14], v[13], v[5]), add(v[15], v[14], v[4]),          # v13 b1, v5 b1 | v14 b2, v4 b0
    # G3 source shares bank with the consumer's OWN dest, not with the producer's
    add(v[16], v[2], v[3]), add(v[17], v[16], v[5]), add(v[18], v[17], v[6]),          # v16 b0, v5 b1 = b(v17) | v17 b1, v6 b2
    # G4 forwarded twice, then a penalised third op (penalty on c, visible on d)
    add(v[19], v[2], v[3]), add(v[20], v[19], v[19]), add(v[21], v[20], v[8]), add(v[22], v[21], v[5]),  # v20 b0, v8 b0 | v21 b1, v5 b1 -> d also penalised
    # G5 same-bank source in src0 position
    add(v[23], v[2], v[3]), add(v[24], v[7], v[23]), add(v[25], v[24], v[4]),          # v23 b3, v7 b3 | v24 b0, v4 b0 -> c also penalised
    # G6 shift producer (LAT 9): consumer same-bank; strict R2c says no penalty (not both-limited)
    v_lshlrev_b32_e32(v[26], v[2], v[3]), add(v[27], v[26], v[6]), add(v[28], v[27], v[4]),  # v26 b2, v6 b2 | v27 b3, v4 b0
    # G7 fma consumer with two non-forwarded same-bank sources
    add(v[29], v[2], v[3]), v_fma_f32(v[30], v[29], v[5], v[9]), add(v[31], v[30], v[4]),   # v29 b1, v5 b1, v9 b1 | v30 b2, v4 b0
    # G8 v_mov with no VGPR read (LAT 6?) feeding a dependent add
    v_mov_b32_e32(v[32], 3.0), add(v[33], v[32], v[5]), add(v[34], v[33], v[4]),        # v32 b0, v5 b1 | v33 b1, v4 b0
    # G9 v_mov from a VGPR (LAT 8?) feeding a dependent add
    v_mov_b32_e32(v[35], v[2]), add(v[36], v[35], v[5]), add(v[37], v[36], v[4]),       # v35 b3, v5 b1 | v36 b0, v4 b0 -> c penalised
    # G10 VOPD reading two same-bank registers, no dependency
    v_dual_add_f32(VOPDOp.V_DUAL_ADD_F32, vdstx=v[38], srcx0=v[2], vsrcx1=v[6], vdsty=v[39], srcy0=v[3], vsrcy1=v[7]),
    # G11 VOPD consumer: X half has forwarded v40 plus same-bank v4
    add(v[40], v[2], v[3]), v_dual_add_f32(VOPDOp.V_DUAL_ADD_F32, vdstx=v[41], srcx0=v[40], vsrcx1=v[4], vdsty=v[42], srcy0=v[5], vsrcy1=v[9]), add(v[43], v[41], v[6]),  # v40 b0, v4 b0 | v41 b1, v6 b2
    # G12 int dependent chain (issue spacing 5 or 1?)
    v_add_nc_u32_e32(v[44], v[0], v[1]), v_add_nc_u32_e32(v[45], v[44], v[1]), v_add_nc_u32_e32(v[46], v[45], v[1]),
    # G13 v_cmpx then a VALU op (exec 7 or 8?)
    v_cmpx_gt_u32_e32(64, v[0]), add(v[47], v[2], v[3]), add(v[48], v[47], v[5]),
    # G14 int producer, fp consumer and fp producer, int consumer
    v_add_nc_u32_e32(v[49], v[0], v[1]), add(v[50], v[49], v[5]), v_add_nc_u32_e32(v[51], v[50], v[1]),
    v_add_f32_e32(v[52], v[12], v[15]), v_add_f32_e32(v[52], v[52], v[18]), v_add_f32_e32(v[52], v[52], v[22]),
    v_add_f32_e32(v[52], v[52], v[25]), v_add_f32_e32(v[52], v[52], v[28]), v_add_f32_e32(v[52], v[52], v[31]),
    v_add_f32_e32(v[52], v[52], v[34]), v_add_f32_e32(v[52], v[52], v[37]), v_add_f32_e32(v[52], v[52], v[39]),
    v_add_f32_e32(v[52], v[52], v[43]), v_add_f32_e32(v[52], v[52], v[46]), v_add_f32_e32(v[52], v[52], v[48]), v_add_f32_e32(v[52], v[52], v[51]),
    global_store_b32(addr=v[1], data=v[52], saddr=s[0:1]),
    s_endpgm(),
  ]
  sink = UOp.sink(A.base, threads, wg, arg=KernelInfo("probe_bank"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=(x, dtypes.void)) for x in insts))))

if __name__ == "__main__":
  a = Tensor.empty(32, dtype=dtypes.int32).contiguous().realize()
  Tensor.custom_kernel(a, fxn=bank)[0].realize()

# status

- 2026-10-21: run_hsaco + stalls work on the 7900 XTX for a hipcc saxpy (-mcumode), fixture committed.
- 2026-10-26: rocprofv3 ATT capture (`tools/att-capture.sh`, hang-proof) for any HIP program: WGP mode,
  any grid, scalar args. App mode, diff of two run dirs, `--by-class`, `--loops`, budget gate.
- 2026-10-29: builds on Mojo 1.2.0. Mojo kernels captured through `examples/dump_*.mojo` + `.co`.
- 2026-10-30: AMD's decoder as oracle (`tools/amd_oracle.py`); traced SIMD found from the trace (was SIMD 0: wrong
  on loops); `--exposed`, occupancy + `wavewait occ`, `divergent` hint; walk 2x faster.
- 2026-10-30: `tools/att-audit.sh` re-walked 139 dispatches in 30 ATT runs on this machine: 133 agree with AMD's
  decoder; 4 changed under the SIMD fix (2 kinestride `smooth`, doc corrected in kinestride `5284961`).
- 2026-10-30 (later): app mode takes the traced SIMD and the dispatch's waves from rocprofv3's filenames.json;
  the oracle checks issues per wave against AMD's per-wave files (code.json maps only part of a Mojo kernel).
  Exact per wave on kinestride smooth u0/u1, ssm_conv, app_div.
- 2026-10-30 (fix): SQTT NEW_PC (0b0100001) is 64 bits as AMD's decoder reads it, not tinygrad's 72; a
  TS_DELTA_OR_MARK with rt set adds no delta (AMD gfx11parser.h). att6/att11 mega now agree with AMD: every
  packet time of all 4 waves equals AMD's issue + stall (att6 3.7-4.1 M per wave), 132.3 M / 109.0 M cycles.
- 2026-10-30 (att4): a wave's walk stops at its first packet that fits no instruction, as AMD's stitcher does.
  On att4 mega the GPU preempted the waves (CWSR): the save trap handler's packets (pcs outside the code) came
  3.2 M cycles after the last real issue. att-audit 136/136 ok; att4 4127 139.9 M -> 127.0 M cycles.
- Open: tinygrad path stays CU-mode, pointer-args, 1-D (use ATT instead). `wavewait sim` blocked (docs/W7-blocked.md).
- 2026-10-30 (use): on mojo-baro mega11 (att11) wavewait's --exposed ranking found the delta-scan grid barrier
  (64 of 96 blocks idle); mojo-baro 1d89111 moved the z projection there: +1.7% tok/s (Mojo 1.1), +2.7% (1.2),
  ids 20/20. Open there, not here: Mojo 1.2 codegen makes the kq megakernel 13% slower.
- 2026-10-30 (PyTorch): SQTT capture of a torch ROCm process works, warm (`docs/pytorch.md`).
  `--preload librocprofiler-sdk.so` solves the wheel's second SDK; `--kernel-iteration-range` traces a later dispatch.
  `wavewait rank` ranks a process from its kernel trace. The walk follows multi-kernel code objects (entry from code.json) and
  MIOpen's Sp3 kernels (s_version, s_call_b64, computed s_setpc_b64). `att-capture.sh` forwards ATT_* through gpu-wait.
  `tools/torch-e2e.sh` passed on torch 2.13.0+rocm7.2. Open: 9 of 44 torch dispatches differ from AMD on
  `s_waitcnt` rows after `ds_store_b128` (2.4 cycles per hit on the matmul, issues and hits exact).
- 2026-10-30 (TRELLIS.2): one mesh traced end to end (`tools/comfy-trace.sh`, `tools/comfy_join.py`, `wavewait rank --stages`, busy as a union).
  120.5 s bare, GPU busy 51.4 s, samplers 33 %, mesh post-processing 53 %. A GPU reset came during the server preflight, cause not proven.

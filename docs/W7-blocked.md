# W7 what-if simulator: blocked

`wavewait sim` exits 2. It needs a cycle model that predicts per-instruction deltas from the code alone.
The candidate is the per-SIMD scheduler for tinygrad's RDNA3 emulator (bounty work, skill `rdna3-sqtt-emu`,
section "model needed"): fitted constants failed off-fit (sync 16/30, gemm 18/59, plus 7/12 exact deltas), so a
cycle-stepped scheduler is required and does not exist yet.

Unblocks when: that scheduler reproduces the SRAM fixtures' deltas exactly on at least one kernel it was not
fitted on. Then `wavewait sim CODEOBJ` = run the scheduler, emit the same Row record as `analyze`, and
`wavewait diff` against a real capture scores it (exact-match rate per kernel, no threshold claimed).

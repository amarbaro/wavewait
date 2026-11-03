# wavewait

<!-- badges -->
[![check](https://github.com/amarbaro/wavewait/actions/workflows/check.yml/badge.svg?branch=main)](https://github.com/amarbaro/wavewait/actions/workflows/check.yml) [![latest release](https://img.shields.io/github/v/release/amarbaro/wavewait)](https://github.com/amarbaro/wavewait/releases/latest) [![license](https://img.shields.io/github/license/amarbaro/wavewait)](https://github.com/amarbaro/wavewait/blob/main/LICENSE)
<!-- /badges -->

Cycle-level stall profile of any gfx1100 kernel (hipcc, Mojo, hand asm) from a real SQTT thread trace
on the 7900 XTX, captured through rocprofv3 ATT (any HIP program) or tinygrad's KFD runtime (one workgroup).
Shows, per instruction, how many cycles each wave waited before it issued.

`tools/doctor.sh` says what this machine still needs to capture, with the fix.

## Use: rocprofv3 ATT

ROCm's thread trace writes raw RDNA3 SQTT for any HIP program as it is (no ELF swap, any grid,
WGP mode). Point it at WGP 3 (the only one that emits instruction tokens here) and launch enough workgroups
that CU 3 gets some (`examples/saxpy.hip` launches 192):

```bash
hipcc --offload-arch=gfx1100 -O3 examples/saxpy.hip -o .work/att/saxpy
rocprofv3 --att --att-target-cu 3 --att-shader-engine-mask 0x1 \
    --att-library-path <dir of AMD's rocprof-trace-decoder library> -d .work/att/run -- .work/att/saxpy
tools/from-att.sh .work/att/run .work/att/saxpy_att      # .att files unchanged + pcmap of code object 2
build/wavewait .work/att/saxpy_att --causes --top 10
```
`tools/att-capture.sh RUNDIR -- CMD...` is that capture without the traps: CU 3 / SE mask 0x1 / the decoder path as defaults
(`ATT_CU`, `ATT_SE_MASK`, `ATT_DECODER_DIR`, `ATT_KERNEL` regex), a 256 MB buffer (`ATT_BUFFER`; more needs `ATT_BUFFER_OK=1`:
at 1 GB rocprofv3 fails to create the ATT packet and then deadlocks in its signal handler, holding the GPU for minutes), `timeout -s KILL`
(`ATT_TIMEOUT`, 300 s) and a watcher that kills the whole process group within about a second of `failed to create ATT packet` or
`Resource deadlock` in the log and exits `FAIL att: ...`. Afterwards it prints the biggest trace against the buffer and warns
(`WARN`) from 90 %: probably cut. It runs itself under `gpu-wait run` when that is installed. Then `build/wavewait app RUNDIR`.
Checked: the saxpy table has the same 28 instructions in the same order as the tinygrad capture
(`fixtures/saxpy_cu`), 8 waves per instruction instead of 4 (two workgroups on CU 3). A single-workgroup
launch traces nothing: it runs on whichever CU the dispatcher picks.

A walk with `mism` > 0 prints `FAIL walk` and exits 3: its causes look valid and are not (`--allow-mism` shows them, with `WARN`).

## Capture through tinygrad: one workgroup

No ROCm profiler needed and one workgroup is enough, but CU mode, pointer args and a 1-D grid only (see Limits);
`fixtures/saxpy_cu`, the check's golden, comes from here.

```
                    run_hsaco.py                       wavewait (Mojo)
k.hsaco ──► tinygrad custom kernel ──► sqtt-capture ──► profile.pkl ──► per-pc table
           (ELF swapped in, hidden     (SQTT,                       (llvm-objdump pc map,
            kernargs filled)            WGP 3 pinned)                  waves walked by packet)
```


```bash
# 1. build a plain code object in CU mode (see Limits)
hipcc --offload-arch=gfx1100 --cuda-device-only --no-gpu-bundle-output -O3 -mcumode -c k.hip -o k.hsaco
# 2. once per boot / session (TINYGRAD = your tinygrad checkout): fixed clocks, tinygrad patch for instruction tokens
#    (card1 = your 7900 XTX's drm card; tools/sqtt-capture finds it and prints the exact line)
echo profile_standard | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
git -C $TINYGRAD apply $PWD/wgp-probe.patch
# 3. capture: one workgroup is enough for a stall profile
SQTT_WGP_PROBE=3 SQTT_HSACO=$PWD/k.hsaco SQTT_BUFS=1024,1024,1024 SQTT_GLOBAL=1 SQTT_LOCAL=256 \
  tools/sqtt-capture $PWD/tools/run_hsaco.py OUTDIR    # runs under gpu-wait when installed
# 4. table (Mojo; reads the pickle through tools/export.py, unpacked into PKL.raw/)
mojo build -I src src/wavewait.mojo -o build/wavewait
build/wavewait OUTDIR/profile_run_hsaco.pkl --causes --top 10      # --kernel PREFIX for other programs
# after: revert the patch, restore the clocks
git -C $TINYGRAD checkout tinygrad/runtime/ops_amd.py
echo auto | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
```

Reading the table: in the raw table a wait's cost shows on the instruction that issues AFTER it (an
`s_waitcnt vmcnt(0)` that waited 670 cycles shows as a 670 delta on the next instruction); `--causes` fixes that.

## Alignment

Each packet's class (VALU, SALU, SMEM, branch, memory, barrier, IMMEDIATE) must fit the instruction it is
assigned to; an instruction no packet fits is untraced and skipped (seen: `s_delay_alu` always, `s_clause`
sometimes traced as IMMEDIATE, `s_endpgm` never). Packets that fit nothing print a WARN. Disassembly uses the
system LLVM 23: ROCm 7.2's objdump prints some gfx11 VOP3 as `.long` and shifts every later pc.

## Subcommands and outputs

| command | what | check.sh |
|---|---|---|
| `wavewait DIR [--top N] [--causes]` | per-instruction stall table, causes. `--top N` cuts an app-dispatch or `--causes` table to the N costliest rows (charged cycles, costliest first) and ends on the cause summary; the plain table of a capture dir stays whole, `--top N` there sizes the ranking below it; `--top 0` prints every row | golden table, cycles conserved, `app --top 3` prints 3 rows |
| `wavewait rank RUNDIR [--top N]` | every kernel of a whole process from rocprofv3 `--kernel-trace --output-format csv`: share, ms, dispatches, avg us, `mid` (the iteration for a warm `--kernel-iteration-range` capture); PyTorch recipe: `docs/pytorch.md` | total against python's csv sum, quoted C++ names |
| `wavewait diff A B [--json]` | two captures aligned by instruction text (LCS): per-row delta, cause shift, +/- rows, total cycles | 28/28 aligned, 1 removed row flagged |
| `wavewait diff RUN_A RUN_B --kernel K [--top N] [--by-class]` | the same for ONE dispatch of two rocprofv3 ATT run dirs (`--kernel PREFIX` or `--dispatch N`, one selector for both sides). Rows are the N (default 20, `--top 0` = all) that moved the most total cycles (`dcycles`), then the class table, the cause table (cycles A, B, shift) and `total cycles A -> B`. Above 4M cells (`WAVEWAIT_ALIGN_CELLS`) the alignment is Hirschberg's, linear in memory: a 14,000-row kernel diffs in 10 s and 88 MB where the table took 1.2 GB | run dir against itself moves 0 cycles, both alignments agree |
| `wavewait DIR --json` | rows + totals, stable keys, for scripts and agents | round-trips 28 rows |
| `wavewait DIR --by-class` | charged cycles, share and issues per wave per instruction class: vmem load (`global_/buffer_/tbuffer_/flat_/scratch_load*`), vmem store (stores and atomics), smem (`s_load*`, `s_buffer_load*`), lds (`ds_*`), `s_waitcnt` split by counter (vmcnt, lgkmcnt, expcnt, vscnt, depctr, mixed = more than one counter), valu (`v_*`), salu (every other `s_*`, so `s_nop`, `s_clause` and `s_set_inst_prefetch_distance` count here), branch (`s_branch`, `s_cbranch*`), other; `--json` carries the same as `classes`. Cycles are the charged ones and sum to the table total | classes sum to 12897, issues = awk count |
| `wavewait DIR --by-line` | costs per source line (DWARF, build with `-g`) | every pc mapped, addr2line agrees |
| `wavewait DIR --hints` | fix hints from the measured gfx1100 timing (hoist, empty wait, VOPD, LDS bank) and `divergent`: the waves took a branch a different number of times | probe fires all 4, saxpy golden; app_div back edge taken 0..2, silent on uniform kernels |
| `wavewait DIR --exposed [--top N]` | rows by exposed cycles: the traced SIMD's idle cycles, each shared by the waves alive then, so a wait 8 waves sat in together counts once and cycles another wave issued in count not at all; then occupancy (mean, max waves alive) and the waves per SIMD the code object allows | rows + tail = SIMD idle; one wave slot (`WAVEWAIT_ONLY`) gives exposed = cycles - 1 per issue exactly |
| `wavewait occ CODEOBJ` | per kernel: waves per SIMD it allows and the limit (vgpr, lds, waves), LLVM's rules | 16/16 kernels equal hipcc's `; Occupancy:` (fixtures/occ) |
| `wavewait budget FILE DIR` rule `exposed N` | max exposed cycles | own value passes, -1 fails |
| `wavewait app RUNDIR` | every dispatch of a rocprofv3 ATT run, one row per kernel | 2 kernels, saxpy row = its table (38342) |
| `wavewait app RUNDIR --dispatch N` / `--kernel NAME` | the full table (and every flag below) of ONE dispatch, walked against its own code object | equals its app row |
| `wavewait DIR --loops` | charged cycles per innermost loop (backward branch), poll loops flagged `spin` | cycles conserved on the megakernel fixture |
| `WAVEWAIT_PACKETS=W wavewait ...` | debug: wave W's first 400 packets, the pc each lands on, MISS where the walk lost alignment | - |
| `wavewait budget FILE DIR` | `cycles N` / `cause NAME PCT` rules, exit 1 naming each broken rule | own numbers pass, -1 cycle and -10% fail |
| `wavewait DIR --html OUT` | self-contained per-wave timeline page | 4 lanes x 112 bars |
| `wavewait sim` | blocked: needs the per-SIMD scheduler (`docs/W7-blocked.md`) | exits 2 |

Peak RSS on the 11 MB trace: 24.0 MB table, 24.1 MB `--html`; app mode on `fixtures/app` 45.2 MB.
A 37 MB trace walks in 294 ms at 48.4 MiB (was 579 ms: a List allocated per packet).

## Limits

- The traced SIMD is found from the trace (the one where every instruction packet's wave slot is alive); AMD's
  decoder put `examples/diverge.hip` on SIMD 3. Earlier versions always walked SIMD 0.
- `--exposed` sees one SIMD's issue stream: a cycle counts as hidden when another wave on that SIMD issued. AMD's
  rocprof-compute-viewer ranks pipes instead (WMMA > VALU > VMEM/LDS > SALU); the two numbers are not the same.
- Divergence is wave level only: SQTT on gfx11 has no EXEC mask. `divergent` counts taken branches (op 3); a
  not-taken branch sends op 4, which the walk does not count.
- **tinygrad capture (`tools/sqtt-capture`) only:** CU mode (hipcc defaults to WGP mode on gfx1100; the capture
  pins dispatch to one CU, so a WGP-mode workgroup never launches and the run times out; build with `-mcumode`),
  one kernel per code object, pointer args only (filled with 1.0f), 1-D grid. The rocprofv3 ATT capture has none
  of these: use it for real programs.
- ATT traces only workgroups that land on CU 3: launch enough of them (a single workgroup traces nothing).
- tinygrad's emulator does not decode `buffer_gl0_inv` (hipcc WGP-mode output), so no emulator replay yet.

## Mojo analyzer

`src/wavewait.mojo` (binary `build/wavewait`) decodes raw RDNA3 SQTT itself (tinygrad's nibble state machine, tables generated into
`src/sqtt_table.mojo` by `tools/gen_table.py`), no tinygrad at analysis time. `tools/export.py` turns a
pickle into `blob_N.bin` + `pcmap.tsv`. Checked byte-identical against the retired Python analyzer (`tools/stalls.py`, deleted; its saxpy table is frozen in `fixtures/golden/`) on the saxpy
fixture and 8 probe captures (loops and taken branches included). `--kernel PREFIX` picks another program
(e.g. `probe` for the probe captures).

### `--causes`

Moves each wait's cost onto the `s_waitcnt` / `s_barrier` that held the wave (the next instruction keeps 1
cycle), labels every row (wave start, mem, lds/smem, barrier, depctr, valu dep, lds queue, salu dep, vmem
issue, fetch, issue wait, issue) from the measured gfx1100 timing in the `rdna3-sqtt-emu` skill, and sums the kernel
by cause. Total cycles are conserved (check.sh). saxpy: 27% wave start, 25% lds/smem, 21% mem, 21% mem+lds.

The decoder streams (no packet list): 11 MB trace 72 -> 23 MB peak, of which 11 MB is the trace itself.

## Also here

- `probes/`: 14 hand-written SRAM probe kernels (LDS, banks, barriers, WMMA, icache, s_nop) that measured
  RDNA3 timing rules used by our tinygrad emulator work. Run with `tools/sqtt-capture $PWD/probes/<probe>.py`.
- `wgp-probe.patch`: tinygrad patch that traces WGP 3 (the unit that emits instruction tokens on a 7900 XTX under KFD).
- `tools/att-capture.sh`: the rocprofv3 ATT capture above, guarded (buffer cap, hang watcher, truncation warning).
- `tools/sqtt-capture`: captures a tinygrad preset or script on the GPU into a trimmed profile pickle.

## Check

`./check.sh` (CPU only): `tools/amd_oracle.py` checks every ATT fixture against AMD's own decoder (`code.json`): hits
equal waves on every pc, each `s_waitcnt` within 1 cycle per wave. stall table of `fixtures/saxpy_cu.pkl` is aligned (28 instructions x 4 waves), Mojo == golden, `--causes` conserves cycles.

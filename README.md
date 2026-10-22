# wavewait

Cycle-level stall profile of any gfx1100 kernel (hipcc, Mojo, hand asm) from a real SQTT thread trace
on the 7900 XTX, through tinygrad's KFD runtime. Shows, per instruction, how many cycles each wave
waited before it issued.

```
                    run_hsaco.py                       wavewait (Mojo)
k.hsaco ──► tinygrad custom kernel ──► sqtt-capture ──► profile.pkl ──► per-pc table
           (ELF swapped in, hidden     (SQTT,                       (llvm-objdump pc map,
            kernargs filled)            WGP 3 pinned)                  waves walked by packet)
```

## Use

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

## Capture without tinygrad: rocprofv3 ATT

ROCm's thread trace writes the same raw RDNA3 SQTT, for any HIP program as it is (no ELF swap, any grid,
WGP mode). Point it at WGP 3 (the only one that emits instruction tokens here) and launch enough workgroups
that CU 3 gets some (`examples/saxpy.hip` launches 192):

```bash
hipcc --offload-arch=gfx1100 -O3 examples/saxpy.hip -o .work/att/saxpy
rocprofv3 --att --att-target-cu 3 --att-shader-engine-mask 0x1 \
    --att-library-path <dir of AMD's rocprof-trace-decoder library> -d .work/att/run -- .work/att/saxpy
tools/from-att.sh .work/att/run .work/att/saxpy_att      # .att files unchanged + pcmap of code object 2
build/wavewait .work/att/saxpy_att --causes --top 10
```
Checked: the saxpy table has the same 28 instructions in the same order as the tinygrad capture
(`fixtures/saxpy_cu`), 8 waves per instruction instead of 4 (two workgroups on CU 3). A single-workgroup
launch traces nothing: it runs on whichever CU the dispatcher picks.

## Subcommands and outputs

| command | what | check.sh |
|---|---|---|
| `wavewait DIR [--top N] [--causes]` | per-instruction stall table, causes | golden table, cycles conserved |
| `wavewait diff A B [--json]` | two captures aligned by instruction text (LCS): per-row delta, cause shift, +/- rows | 28/28 aligned, 1 removed row flagged |
| `wavewait DIR --json` | rows + totals, stable keys, for scripts and agents | round-trips 28 rows |
| `wavewait DIR --by-line` | costs per source line (DWARF, build with `-g`) | every pc mapped, addr2line agrees |
| `wavewait DIR --hints` | fix hints from the measured gfx1100 timing (hoist, empty wait, VOPD, LDS bank) | probe fires all 4, saxpy golden |
| `wavewait app RUNDIR` | every dispatch of a rocprofv3 ATT run, one row per kernel | 2 kernels, saxpy row = its table (38342) |
| `wavewait budget FILE DIR` | `cycles N` / `cause NAME PCT` rules, exit 1 naming each broken rule | own numbers pass, -1 cycle and -10% fail |
| `wavewait DIR --html OUT` | self-contained per-wave timeline page | 4 lanes x 112 bars |
| `wavewait sim` | blocked: needs the per-SIMD scheduler (`docs/W7-blocked.md`) | exits 2 |

Peak RSS on the 11 MB trace: 24.0 MB table, 24.1 MB `--html`; app mode on `fixtures/app` 45.2 MB.

## Limits

- **CU mode only under capture.** hipcc defaults to WGP mode on gfx1100; the capture pins dispatch to one
  CU, so a WGP-mode workgroup never launches and the run times out. Build with `-mcumode`.
- One kernel per code object, pointer args only (filled with 1.0f), 1-D grid.
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
- `tools/sqtt-capture`: captures a tinygrad preset or script on the GPU into a trimmed profile pickle.

## Check

`./check.sh` (CPU only): stall table of `fixtures/saxpy_cu.pkl` is aligned (28 instructions x 4 waves), Mojo == golden, `--causes` conserves cycles.

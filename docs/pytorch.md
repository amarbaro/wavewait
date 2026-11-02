# PyTorch ROCm jobs through wavewait

Question: where does GPU time go inside a long PyTorch job on the 7900 XTX?
Verdict: SQTT capture of a torch process works, warm, with no change to the job. Tested on `tools/torch_probe.py`
and one ComfyUI TRELLIS.2 mesh (torch 2.13.0+rocm7.2, ROCm 7.2.4 rocprofv3).

Check: `tools/torch-e2e.sh [PYTHON]` runs the recipe below on the GPU and fails unless the walk is clean and the traced
dispatch ran warm. `./check.sh` covers the CPU side (`fixtures/app_torch`, `fixtures/rank`).

## Recipe

```bash
cd ~/Projects/mojo/wavewait
PY=/path/to/the/job/venv/bin/python
JOB="$PWD/tools/torch_probe.py 1500"      # the job; absolute paths

# 1. rank: every dispatch on every CU, +4 % wall time, about 390 bytes per dispatch
gpu-wait run --timeout 60 --vram 2 -- /opt/rocm/bin/rocprofv3 --kernel-trace --output-format csv \
    --preload librocprofiler-sdk.so -d $PWD/.work/pt/rank -- $PY $JOB
build/wavewait rank .work/pt/rank --top 10

# 2. capture the kernel you picked at its `mid` iteration (750 here): warm, one dispatch
ATT_SLIM=1 ATT_KERNEL=Cijk_Ailk ATT_EXTRA="--preload librocprofiler-sdk.so --kernel-iteration-range [750-750]" \
    tools/att-capture.sh $PWD/.work/pt/warm -- $PY $JOB

# 3. walk it
build/wavewait app .work/pt/warm                                # the row: cycles, waves, cause, open, mism, dispatch
build/wavewait app .work/pt/warm --dispatch N --causes --top 10
```

`ATT_KERNEL` is a regex on the kernel name and matches torch's C++ names (`reduce_kernel` matched two).
`--kernel-iteration-range` counts per kernel from 1 and is inclusive: `[800-802]` traced the 800th, 801st and 802nd
dispatch. Without `ATT_KERNEL` it applies to every kernel, so `[800-801]` gave 20 traces (160 MB) for 10 kernels.

A time window is the other warm capture: `ATT_EXTRA="--preload librocprofiler-sdk.so -P 2:0.02:1"` starts 2 s after process
start, lasts 20 ms and traces the first dispatch of each kernel inside the window. Dispatch ids restart at 1, and the kernel
trace covers only the window (180 of 18005 dispatches). `-P` needs the window's time from the process start, which
`wavewait rank` does not print.

## The `--preload` line

`rocprofv3` on a torch wheel aborts at `import torch` without it:
`api registration failed with error code 16: Configuration request occurred outside of valid rocprofiler configuration period`.
The wheel ships its own `librocprofiler-sdk.so`, and `libtorch_cpu.so` needs that unversioned name. The preloaded SDK has the
soname `librocprofiler-sdk.so.1`, so the loader does not match it, takes the wheel's copy through RPATH, and two SDKs run.
Preloading the bare name makes it a known name of the system copy: one SDK, nothing in the venv changes.

## The four known gaps

| Gap | State | Evidence |
|---|---|---|
| Only the first dispatch, cold | closed | the 800th of 1500 matmuls traced at t = 1.1 s, 640 us against 1377 us cold; AMD's decoder agrees on the walk |
| Many unknown kernels | closed | `wavewait rank`: 12 kernels, 18005 dispatches in 40 ms; the names are MIOpen, rocBLAS and torch's own |
| Code objects loaded at run time | closed | 14 objects mapped from `code.json`; 15 of 16 cold dispatches had waves, none mismatched. No `-g` in them (0 `.debug_*` sections): `--by-line` stays out |
| One traced CU | open by hardware, covered | `rank` counts every CU. The dispatches with 0 waves were small (a one-workgroup fill, a `Cijk_D`). A traced kernel is a sample: 2 to 1330 waves of a dispatch |

## Why warm matters

The same matmul, cold (dispatch 5) then warm (dispatch 9606): cycles per wave stay within 1 % (1.03 M, 1.02 M), wall time halves
(1377 us to 640 us), and the top cause moves from `lds/smem` to `barrier`. A cold capture names the wrong wait.

## Traps

- **Disk.** rocprofv3 decodes every trace to JSON: 854 MB for a 1.2 s job, 245 MB per wave of one conv. Trace one kernel at one
  iteration, use `ATT_SLIM=1`, and check `df` first. `ATT_SLIM` runs after the decode, so the space is needed during the run.
- **Buffer.** 256 MB per engine. One cold conv (180 M cycles, 2 waves) took 98 MB. `att-capture.sh` warns from 90 %.
- **`--att-consecutive-kernels 4`:** SIGSEGV in `rocprofiler_set_api_table` at start (one run). Use the iteration range.
- **`--pmc`:** known-broken on this card (`~/Brain/m.ledger/gpu.md`, 2026-10-12).
- **Attach (`rocprofv3 -p PID`):** not tested. Never on a live render: `~/Brain/m.ledger/comfyui.md` 2026-10-15.
- **MIOpen's first run of a new conv shape** benchmarks its candidates: `naive_conv_ab_nonpacked` took 91 % of the kernel time of
  the first capture and 0 dispatches in every later run, because the result is cached on disk. A first run ranks that search,
  not the job: rank a repeat run.

## Audit against AMD's decoder

`tools/att-audit.sh` on 44 torch dispatches of 4 captures: 35 ok, 9 fail, each on `s_waitcnt` rows only. Issue counts and hit counts
are exact on all 44 (AMD's instructions less one per wave). Read on the matmul (7 of the 9): AMD charges a `ds_store_b128`
5 cycles and wavewait 1, so the rest lands on the next wait, 2.4 cycles per hit, with dispatch totals within 0.12 %. The other 2 (a
`Cijk` kernel's `lgkmcnt(7)`, a normal-distribution kernel's `vmcnt(0)`) were not read. The audit keeps reporting all 9.

## Not tested

- A repeat job on one server process lands on later iterations than the first.

## ComfyUI

`tools/comfy-trace.sh OUT [KERNEL_REGEX ITERATIONS | --bare]` runs one mesh under rocprofv3 as a gpu-wait job; `tools/comfy_join.py BARE TRACED`
joins a bare run and a traced run per node. `wavewait rank RUN --stages OUT/stages.tsv` adds busy, idle and top kernel per node.

- Take wall time from a `--bare` run: tracing added 14 % on 2.5 M dispatches (120.5 s to 137.7 s), nearly all of it in one node.
- A kernel trace is 2 GB for that job (about 800 bytes per dispatch); `rank` takes 8 s and 5 GB of RAM.
- rocprofv3 finalizes on SIGINT and the server then lingers: kill it by PID. One lingered 3 min and a GPU reset followed (cause not proven).
- `--kernel-iteration-range` counts per kernel id, and one name can have several: `attn_fwd` iteration 1734 landed at position 1783.
- wavewait's `valu dep` is a label on the row, not a proven dependency: on `gemm_wmma` a `v_mov_b32 v67, 0` carried it. Read it as VALU stall.
- ComfyUI sends `executing` only to the client that queued the prompt: `comfy-trace.sh` runs a private copy of the workflow script with a client id.

# fixtures/probe

A probe capture for the `wavewait --hints` rules: the kernel (`examples/probe.hip`) is built to
carry the four patterns the rules table looks for, so every rule fires on it.

    blob_0.bin   the raw RDNA3 SQTT of the probe dispatch (8 waves)
    blob_1.bin   the overflow buffer of the last wave (the tail packets)
    pcmap.tsv    pc, size, dword, text (from the code object the GPU ran)

The capture is a `rocprofv3 --att` run of the `-O2 -mcumode` build of `examples/probe.hip`
(192 workgroups of 256 threads, so that CU 3 gets work: a single workgroup does not reach the
only compute unit that emits instruction tokens, see `README.md`):

    rocprofv3 --att --att-target-cu 3 \
        --att-shader-engine-mask 0x1 --att-library-path <rocprof-trace-decoder lib dir> \
        -d .work/att/run5 -- .work/probe_g
    tools/from-att.sh .work/att/run5 fixtures/probe

What each rule sees here:

    0x1914  hoist      the wave start: the scalar load of the kernel arguments, waited on right
                       after it (3237.6 cycles, the front end fetch of the first instruction)
    0x195c  hoist      the global load, waited on at 0x1974 (534.6 cycles)
    0x1978  lds bank   ds_store_b32 v6, v5: v6 = the lane id shifted by 4, so the byte address
                       steps by 16 bytes per lane (4 lanes per bank)
    0x198c  hoist      the second global load, waited on at 0x1994 (389.3 cycles)
    0x1998  idle wait  an s_waitcnt with nothing in flight (the second inline wait of the probe)
    0x199c  lds bank   ds_load_b32 v4, v6: the same 16 byte stride
    0x19a4  vopd       two independent 32 bit VALU ops in two slots (the compiler left them
                       unpacked; a VOPD slot carries two of them)

The saxpy capture stays quiet on the rules it does not expect: `fixtures/golden/saxpy_cu.hints.txt`
lists the 5 hints it does have (4 hoist, 1 vopd) and no other rule fires on it.

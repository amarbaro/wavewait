# fixtures/saxpy_g

The same ATT saxpy capture as `../saxpy_att` (dispatch 1 of the run), but the kernel was
compiled with `-g`, so the capture ships the code object it was built from:

    blob_0.bin   the raw RDNA3 SQTT of the saxpy dispatch
    pcmap.tsv    pc, size, dword, text (as in ../saxpy_att)
    codeobj.out  the gfx1100 code object with its .debug_line section

`wavewait fixtures/saxpy_g --by-line` maps every traced pc to its DWARF source position.
The ground truth (`llvm-addr2line -e codeobj.out`) agrees with the tool for all 28 rows:

    0x1900-0x1920  amd_hip_runtime.h:264   (the inlined __hip_get_block_dim_x: the prologue
                                            that loads the kernel arguments)
    0x192c         examples/saxpy.hip:5
    0x1934-0x195c  examples/saxpy.hip:6
    0x1964-0x196c  examples/saxpy.hip:8
    0x1974-0x1978  examples/saxpy.hip:6
    0x1980         amd_device_functions.h:691  (the __syncthreads() macro)
    0x1984-0x1988  amd_device_functions.h:692  (the barrier)
    0x198c-0x19bc  examples/saxpy.hip:8

19 of the 28 traced rows land in `examples/saxpy.hip`; the other 9 are the kernel prologue
and the barrier, which the compiler attributes to the headers they were inlined from. That
is the compiler's attribution, not a limit of the tool: `llvm-addr2line` prints the same
file and line for every one of those pcs.

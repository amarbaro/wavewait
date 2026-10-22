# Run a foreign gfx1100 code object (hipcc / Mojo .hsaco, one kernel, buffer args only) through tinygrad,
# so PROFILE / VIZ=-2 / SQTT see it like a tinygrad kernel.
#   SQTT_HSACO=k.hsaco SQTT_BUFS=4096,4096,4096 SQTT_GLOBAL=16 SQTT_LOCAL=256 tools/sqtt-capture $PWD/tools/run_hsaco.py
# (SQTT_* so sqtt-capture forwards them; plain python3 works the same)
# SQTT_INTS: int32 args after the pointers (Mojo kernels: k, n). SQTT_CUMODE=1: see below.
# SQTT_BUFS: byte size of each pointer arg in kernel order (filled with 1.0f). GLOBAL/LOCAL: workgroups / threads (x only).
# How: a one-instruction custom kernel carries the args; its assembled ELF is replaced by the hsaco bytes.
# tinygrad reads the kernel descriptor (VGPRs, LDS, kernarg size) from the hsaco itself; hidden args are filled here.
import dataclasses, os
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.runtime.autogen.amd.rdna3.ins import s_endpgm
import tinygrad.renderer.amd.elf as amd_elf
import tinygrad.runtime.ops_amd as ops_amd

lib = open(os.environ["SQTT_HSACO"], "rb").read()
sizes = [int(x) for x in os.environ["SQTT_BUFS"].split(",")]
gsz, lsz = int(os.environ.get("SQTT_GLOBAL", "1")), int(os.environ.get("SQTT_LOCAL", "64"))
ints = [int(x) for x in os.environ.get("SQTT_INTS", "").split(",") if x]  # int32 args after the pointers
name = "hsaco_" + os.path.basename(os.environ["SQTT_HSACO"]).split(".")[0]

_assemble = amd_elf.assemble_linear
def assemble_linear(prg, lin, arch):
  return lib if prg.src[0].arg.function_name == name else _assemble(prg, lin, arch)
amd_elf.assemble_linear = assemble_linear

# hipcc kernels read blockDim/gridDim from code object v5 hidden kernargs right after the pointers
_pack = ops_amd.pack_args
def pack_args(args, size):
  if len(args) == len(sizes) and ints:
    args = args + [(8 * len(sizes) + 4 * i, UOp.const(v, dtypes.int32)) for i, v in enumerate(ints)]
  elif len(args) == len(sizes) and size >= 8 * len(sizes) + 24:
    o, u32, u16 = 8 * len(sizes), dtypes.uint32, dtypes.uint16
    args = args + [(o, UOp.const(gsz, u32)), (o+4, UOp.const(1, u32)), (o+8, UOp.const(1, u32)),
                   (o+12, UOp.const(lsz, u16)), (o+14, UOp.const(1, u16)), (o+16, UOp.const(1, u16))]
  return _pack(args, size)
ops_amd.pack_args = pack_args

# SQTT_CUMODE=1: run a WGP-mode kernel in CU mode. The capture pins dispatch to one CU, where a WGP-mode
# workgroup never launches. Same results for kernels without LDS (WGP mode only widens LDS sharing).
_image = ops_amd._amd_program_image
def amd_program_image(dev, l):
  data, image = _image(dev, l)
  if l == lib and os.environ.get("SQTT_CUMODE") == "1":
    assert data.group_segment_size == 0, "SQTT_CUMODE needs a kernel without LDS"
    data = dataclasses.replace(data, rsrc1=data.rsrc1 & ~(1 << 29))  # COMPUTE_PGM_RSRC1.WGP_MODE
  return data, image
ops_amd._amd_program_image = amd_program_image

def kernel(*bufs:UOp) -> UOp:
  sink = UOp.sink(*[b.flatten().base for b in bufs], UOp.special(lsz, "lidx0"), UOp.special(gsz, "gidx0"), arg=KernelInfo(name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(UOp(Ops.INS, arg=(s_endpgm(), dtypes.void)),))))

if __name__ == "__main__":
  bufs = [Tensor.ones(n // 4, dtype=dtypes.float32).contiguous().realize() for n in sizes]
  out = Tensor.custom_kernel(*bufs, fxn=kernel)[int(os.environ.get("SQTT_OUT", "0"))].realize()  # SQTT_OUT: which arg to print
  print(name, "out[:4] =", out[:4].tolist(), "min/max =", out.min().item(), out.max().item())

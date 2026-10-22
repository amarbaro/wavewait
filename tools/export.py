"""export.py PKL OUTDIR [--kernel PREFIX]: the inputs build/wavewait reads, out of a tinygrad profile pickle.
OUTDIR/blob_N.bin = raw SQTT of the kernel per shader engine, OUTDIR/pcmap.tsv = pc, size, first dword, text.
build/wavewait calls export() itself when given a .pkl. Python only because unpickling needs tinygrad's classes."""
import os, pickle, re, subprocess, sys, tempfile

# system LLVM: ROCm 7.2's objdump prints some gfx11 VOP3 (v_mul_lo_u32...) as .long + a bogus v_cndmask
LLVM = "/usr/bin/"
KERNEL = "hsaco"  # program name prefix

def pc_map(lib:bytes) -> dict[int, tuple[int, str, int]]:
  """pc -> (size, text, first dword) of every instruction in .text, s_delay_alu and s_code_end padding included."""
  with tempfile.NamedTemporaryFile(suffix=".o") as f:
    f.write(lib); f.flush()
    r = subprocess.run([LLVM + "llvm-objdump", "-d", "--mcpu=gfx1100", f.name], capture_output=True, text=True)
  if r.returncode: return pc_map_raw(lib)  # tinygrad's own minimal ELF, objdump rejects it
  out = r.stdout
  m = {}
  for line in out.splitlines():
    if (r := re.match(r"\s+(\S.*?)\s*//\s*([0-9A-F]+):((?: [0-9A-F]{8})+)", line)):
      m[int(r[2], 16)] = (4 * len(r[3].split()), r[1], int(r[3].split()[0], 16))
  return m


def pc_map_raw(lib:bytes) -> dict[int, tuple[int, str, int]]:
  """same map from .text bytes through llvm-mc, for ELFs llvm-objdump cannot read"""
  from tinygrad.runtime.support.elf import elf_loader
  image, sections, _ = elf_loader(lib)
  t = next(sh for sh in sections if sh.name == ".text")
  code = bytes(image[t.header.sh_addr:t.header.sh_addr + t.header.sh_size])
  hexs = " ".join(f"0x{b:02x}" for b in code)
  out = subprocess.run([LLVM + "llvm-mc", "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1100", "--disassemble", "-show-encoding"],
                       input=hexs, capture_output=True, text=True, check=True).stdout
  m, pc = {}, t.header.sh_addr
  for line in out.splitlines():
    if (r := re.match(r"\s+(\S.*?)\s*;\s*encoding: \[(.*)\]", line)):
      enc = [int(x, 16) for x in r[2].split(",")]
      m[pc] = (len(enc), r[1], int.from_bytes(bytes(enc[:4]), "little"))
      pc += len(enc)
  return m

def export(pkl:str, out:str, kernel:str=KERNEL) -> int:
  data = pickle.load(open(pkl, "rb"))
  prg = next(e for e in data if type(e).__name__ == "ProfileProgramEvent" and e.lib and e.name.startswith(kernel))
  os.makedirs(out, exist_ok=True)
  n = 0
  for n, ev in enumerate((e for e in data if type(e).__name__ == "ProfileSQTTEvent" and e.kern == prg.tag), 1):
    open(f"{out}/blob_{n - 1}.bin", "wb").write(ev.blob)
  with open(f"{out}/pcmap.tsv", "w") as f:
    for pc, (size, text, word) in sorted(pc_map(prg.lib).items()): f.write(f"{pc}\t{size}\t{word}\t{text}\n")
  return n

if __name__ == "__main__":
  a = sys.argv
  export(a[1], a[2], a[a.index("--kernel") + 1] if "--kernel" in a else KERNEL)

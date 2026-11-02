# The trace core: the SQTT decoder, the code map of a capture and the one walk that fills the
# row record (src/row.mojo). Every output (table, json, by-line, hints, diff, app, budget) reads
# that record; this module only produces it.
from std.collections import Dict, List
from std.os import stat, getenv
from std.os.path import exists
from std.sys import exit
from std import subprocess
from std.bit import count_trailing_zeros
from row import Analysis, Row, XL
from lines import LineMap, load_dwarf
from sqtt_table import STATE, NIBS, DLO, DMASK, SPECIAL, KIND


comptime WAVESTART = 1
comptime WAVEEND = 2
comptime INST = 3
comptime JUMP = 3
comptime IMM_MASK = 6
comptime TRACED_CU = 3  # the WGP both capture recipes target (SQTT_WGP_PROBE=3, --att-target-cu 3)

comptime TL_CAP = 64  # Gantt lanes kept in the timeline (the wave count itself is reported)

def bits(reg: UInt64, hi: Int, lo: Int) -> Int:
    return Int((reg >> UInt64(lo)) & ((UInt64(1) << UInt64(hi - lo + 1)) - 1))

struct Decoder(Movable):
    """Streaming port of tinygrad.renderer.amd.sqtt.decode for layout 3: next() stops on each packet the walk reads."""

    var data: List[UInt8]
    var state: Array[Int, 256]
    var nibs: Array[Int, 25]
    var dlo: Array[Int, 25]
    var dmask: Array[Int, 25]
    var special: Array[Int, 25]
    var kinds: Array[Int, 25]
    var reg: UInt64
    var pos: Int
    var nib_off: Int
    var nib_count: Int
    var time: Int
    var kind: Int

    def __init__(out self, var data: List[UInt8]):
        self.data = data^
        self.state = materialize[STATE]()  # comptime tables, indexed at runtime
        self.nibs = materialize[NIBS]()
        self.dlo = materialize[DLO]()
        self.dmask = materialize[DMASK]()
        self.special = materialize[SPECIAL]()
        self.kinds = materialize[KIND]()
        self.reg = 0
        self.pos = 0
        self.nib_off = 0
        self.nib_count = 16
        self.time = 0
        self.kind = 0

    def rewind(mut self):
        self.reg, self.pos, self.nib_off, self.nib_count, self.time, self.kind = 0, 0, 0, 16, 0, 0

    def next(mut self) -> Bool:
        var n = len(self.data)
        while self.pos + ((self.nib_count + self.nib_off + 1) >> 1) <= n:
            var need = self.nib_count - self.nib_off
            if self.nib_off:
                self.reg = (self.reg >> 4) | (UInt64(self.data[self.pos] >> 4) << 60)
                self.pos += 1
            var byte_count = need >> 1
            if byte_count:
                var read_bytes = min(byte_count, 8)
                var chunk: UInt64 = 0
                for i in range(read_bytes):
                    chunk |= UInt64(self.data[self.pos + i]) << UInt64(8 * i)
                # a shift by 64 is undefined in Mojo, Python's gives 0
                self.reg = chunk if read_bytes == 8 else (self.reg >> UInt64(read_bytes * 8)) | (chunk << UInt64(64 - read_bytes * 8))
                self.pos += byte_count
            self.nib_off = need & 1
            if self.nib_off:
                self.reg = (self.reg >> 4) | (UInt64(self.data[self.pos] & 0xF) << 60)
            var op = self.state[Int(self.reg & 0xFF)]
            self.nib_count = self.nibs[op]
            var delta = Int((self.reg >> UInt64(self.dlo[op])) & UInt64(self.dmask[op]))
            if self.special[op] == 1 and bits(self.reg, 9, 9) == 1:
                delta = 0  # TS_DELTA_OR_MARK marker
            elif self.special[op] == 2:
                delta += 4  # TS_DELTA_SHORT
            self.time += delta
            self.kind = self.kinds[op]
            if self.kind:
                return True
        return False

    def wave(self) -> Int:
        if self.kind == WAVESTART:
            return bits(self.reg, 17, 13)
        if self.kind == WAVEEND:
            return bits(self.reg, 19, 15)
        if self.kind == INST:
            return bits(self.reg, 12, 8)
        return bits(self.reg, 11, 7)

    def simd(self) -> Int:
        return bits(self.reg, 9, 8) if self.kind == WAVESTART else bits(self.reg, 10, 9)

    def unit(self) -> Int:
        """The (cu, simd) of a WAVESTART/WAVEEND as one number, cu = wgp | sa << 3: the SE reports the
        waves of all its WGPs, only TRACED_CU's carry instruction tokens."""
        if self.kind == WAVESTART:
            return (bits(self.reg, 12, 10) | bits(self.reg, 7, 7) << 3) * 4 + bits(self.reg, 9, 8)
        return (bits(self.reg, 13, 11) | bits(self.reg, 8, 8) << 3) * 4 + bits(self.reg, 10, 9)


comptime PC_GET = 1  # s_getpc_b64: the sgpr pair holds the next pc
comptime PC_ADD = 2  # s_add_u32 of a literal to it
comptime PC_CALL = 3  # s_call_b64: a jump, and the sgpr pair holds the return pc
comptime PC_SET = 4  # s_setpc_b64: a jump to the sgpr pair


struct Pcmap(Movable):
    """The code map of one capture: pc -> (size, first dword, text)."""

    var size: Dict[Int, Int]
    var word: Dict[Int, Int]
    var text: Dict[Int, String]
    var first: Int
    var symbol: String  # the kernel the code object was built from ("" when unknown)
    var regop: Dict[Int, Int]  # pc -> sgpr << 3 | PC_*: the instructions that compute a jump target
    var addend: Dict[Int, Int]  # pc -> the literal of its s_add_u32

    def __init__(out self):
        self.size = Dict[Int, Int]()
        self.word = Dict[Int, Int]()
        self.text = Dict[Int, String]()
        self.first = -1
        self.symbol = ""
        self.regop = Dict[Int, Int]()
        self.addend = Dict[Int, Int]()

    def add(mut self, pc: Int, size: Int, word: Int, text: String) raises:
        self.size[pc] = size
        self.word[pc] = word
        self.text[pc] = text
        self.first = pc if self.first < 0 else min(self.first, pc)
        var op = PC_GET if text.startswith("s_getpc_b64 s[") else (PC_CALL if text.startswith("s_call_b64 s[") else (PC_SET if text.startswith("s_setpc_b64 s[") else 0))
        if op:
            var at = text.find("[") + 1
            self.regop[pc] = atol(text[byte=at:text.find(":", at)]) << 3 | op
        elif text.startswith("s_add_u32 s"):  # s_add_u32 s82, s82, 0x47e4: the low half of a pc-relative address
            var f = text.split(" ")
            if len(f) == 4 and f[1] == f[2].replace(",", "") + "," and f[3].startswith("0x"):
                self.regop[pc] = atol(f[2].replace(",", "")[byte=1:]) << 3 | PC_ADD
                self.addend[pc] = atol(f[3][byte=2:], 16)



struct Table(Movable):
    """Per-pc cycle totals of one capture: as issued (total/peak) and with the waits charged (ctotal/cpeak)."""

    var order: List[Int]  # first-seen order, the tie order of the Python table
    var first: Int
    var total: List[Int]  # all six flat by (pc - first) // 4: the walk adds to them once per issue
    var count: List[Int]
    var peak: List[Int]
    var ctotal: List[Int]
    var cpeak: List[Int]
    var exposed: List[Int]  # SIMD idle cycles charged here, in XL units
    var bmin: Dict[Int, Int]  # taken branch pc -> min / max times one ended wave took it (not-taken: op 4, not counted)
    var bmax: Dict[Int, Int]

    def __init__(out self, first: Int, n: Int):
        self.order = List[Int]()
        self.first = first
        self.total = List[Int](length=n, fill=0)
        self.count = List[Int](length=n, fill=0)
        self.peak = List[Int](length=n, fill=0)
        self.ctotal = List[Int](length=n, fill=0)
        self.cpeak = List[Int](length=n, fill=0)
        self.exposed = List[Int](length=n, fill=0)
        self.bmin = Dict[Int, Int]()
        self.bmax = Dict[Int, Int]()

    def i(self, a: Int) -> Int:
        return (a - self.first) >> 2

    def add(mut self, a: Int, d: Int, c: Int):
        var i = self.i(a)
        if self.count[i] == 0:
            self.order.append(a)
            self.peak[i] = d
            self.cpeak[i] = c
        self.total[i] += d
        self.count[i] += 1
        self.peak[i] = max(self.peak[i], d)
        self.ctotal[i] += c
        self.cpeak[i] = max(self.cpeak[i], c)

    def charge(mut self, a: Int, extra: Int, own: Int):
        """Add held cycles to the wait's sample already counted (own = what the wait itself took to issue)."""
        var i = self.i(a)
        self.ctotal[i] += extra
        self.cpeak[i] = max(self.cpeak[i], own + extra)


def pclass(kind: Int, op: Int) -> String:
    """Instruction class a packet can belong to (the CLASS table of the retired Python analyzer)."""
    if kind == INST:
        if op == 0 or op == 0x72:
            return "s"
        if op == 1:
            return "k"
        if op >= 3 and op <= 5:
            return "b"
        if op == 9:
            return "m"
        if op == 0x13:
            return "r"
        if op >= 0x1C and op <= 0x36:  # FLAT, GLOBAL, LDS, then BUF 0x2f-0x36 (buffer_gl*_inv of every grid barrier)
            return "g"
        return "v"  # VALU reported as INST: transcendental, 64-bit, cmpx
    return "i" if kind >= 5 else "v"  # IMMEDIATE, IMMEDIATE_MASK / VALUINST


def starts_any(t: String, prefixes: List[String]) -> Bool:
    for p in prefixes:
        if t.startswith(p):
            return True
    return False


def fits(c: String, t: String) -> Bool:
    var mem: List[String] = ["global_", "flat_", "ds_", "buffer_", "scratch_"]
    if c == "v":
        return t.startswith("v_")
    if c == "s":
        var not_salu: List[String] = ["s_load", "s_buffer_load", "s_dcache", "s_cbranch", "s_branch", "s_barrier", "s_waitcnt", "s_clause", "s_code_end"]
        return t.startswith("s_") and not starts_any(t, not_salu)
    if c == "k":
        return starts_any(t, ["s_load", "s_buffer_load", "s_dcache"])
    if c == "b":
        return starts_any(t, ["s_cbranch", "s_branch", "s_call", "s_setpc"])
    if c == "r":
        return t.startswith("s_barrier")
    if c == "m":
        return t.startswith("s_sendmsg")
    if c == "g":
        return starts_any(t, mem)
    return starts_any(t, ["s_waitcnt", "s_clause", "s_nop", "s_sleep", "s_endpgm", "s_set", "s_inst", "s_icache", "s_version"]) or starts_any(t, mem)


comptime CLASSES = "vskbrmgi"  # the classes pclass returns; a pc's mask holds one bit each, so the walk never compares text
comptime HAS = 1 << 14
comptime DELAY = 1 << 15


@always_inline
def at(ref m: List[UInt16], first: Int, a: Int) -> Int:
    """The mask of pc a, 0 off the map."""
    var i = (a - first) >> 2
    return Int(m.unsafe_ptr()[i]) if 0 <= i and i < len(m) and a & 3 == 0 else 0


def masks(ref pm: Pcmap) raises -> List[UInt16]:
    """Flat by (pc - first) // 4: which packet classes fit the instruction there, built once per code map."""
    var top = pm.first
    var lo = pm.first
    for a in pm.text.keys():
        top = max(top, a)
    var m = List[UInt16](length=(top - lo) // 4 + 1, fill=0)
    for it in pm.text.items():
        var v = HAS | (DELAY if it.value.startswith("s_delay_alu") else 0)
        for i in range(8):
            if fits(String(CLASSES[byte=i]), it.value):
                v |= 1 << i
        m[(it.key - lo) // 4] = UInt16(v)
    return m^


def lit(t: String) raises -> Int:
    """The last operand of an instruction as a number: 0x2d248, lit(0xffffffff), -1."""
    var parts = t.split(" ")
    var v = String(parts[len(parts) - 1].replace("lit(", "").replace(")", ""))
    return atol(v[byte=2:], 16) if v.startswith("0x") else atol(v)


def far_target(ref pm: Pcmap, a: Int) raises -> Int:
    """Where the s_setpc_b64 at a jumps when it ends LLVM's long branch (s_getpc_b64; s_add_u32 lo; s_addc_u32 hi;
    s_setpc_b64): the pc after s_getpc_b64 plus the 64-bit literal. -1 for any other s_setpc (a return)."""
    if not (pm.text.get(a, "").startswith("s_setpc_b64") and pm.text.get(a - 20, "").startswith("s_getpc_b64")
            and pm.text.get(a - 16, "").startswith("s_add_u32") and pm.text.get(a - 8, "").startswith("s_addc_u32")):
        return -1
    var off = (lit(pm.text[a - 8]) & 0xFFFFFFFF) << 32 | (lit(pm.text[a - 16]) & 0xFFFFFFFF)
    return a - 16 + (off - (1 << 64) if off >> 63 else off)


def is_wait(t: String) -> Bool:
    return t.startswith("s_waitcnt") or t.startswith("s_barrier")


def cause(t: String, avg_x10: Int, is_first: Bool, pc: Int, size: Int) -> String:
    """Cause of an instruction's cost after waits are charged to themselves (gfx1100 timing in rdna3-sqtt-emu)."""
    if is_first:
        return "wave start"  # front-end fetch: ~416 cycles on WGP 0, ~739 on WGP 3
    if t.startswith("s_barrier"):
        return "barrier"
    if t.startswith("s_waitcnt_depctr"):
        return "depctr"
    if t.startswith("s_waitcnt"):
        var vm = "vmcnt" in t or "vscnt" in t
        var lgkm = "lgkmcnt" in t
        return "mem+lds/smem" if vm and lgkm else ("mem" if vm else "lds/smem")
    if avg_x10 >= 1500 and (pc % 128 == 0 or pc // 128 != (pc + size - 1) // 128):
        return "fetch"  # first use of a 128 B code line (Mojo gemv8: 83% of cycles cold, 3% warm)
    if avg_x10 >= 4000:
        return "issue wait"  # long wait with no memory or fetch cause: waves queued for the SIMD
    if avg_x10 <= 80:
        return "issue"
    if t.startswith("v_"):
        return "valu dep"
    if t.startswith("ds_"):
        return "lds queue"
    if t.startswith("s_"):
        return "salu dep"
    return "vmem issue"

def read_pcmap(path: String) raises -> Pcmap:
    """Read pcmap.tsv: pc, size, first dword, text (one instruction per line)."""
    var pm = Pcmap()
    for line in open(path, "r").read().split("\n"):  # splitlines() also splits at tabs
        if not line:
            continue
        var f = line.split("\t", 3)
        pm.add(atol(f[0], 10), atol(f[1], 10), atol(f[2], 10), String(f[3]))
    return pm^

def analyze(dir: String, tl_cap: Int, codeobj: String = "") raises -> Analysis:
    """Analyse a capture directory: the blobs and the pcmap.tsv next to them."""
    var pm = read_pcmap(dir + "/pcmap.tsv")
    var paths = List[String]()
    var b = 0
    while exists(dir + "/blob_" + String(b) + ".bin"):
        paths.append(dir + "/blob_" + String(b) + ".bin")
        b += 1
    var an = analyze_paths(paths^, pm^, dir, tl_cap, codeobj, True)
    an.codeobj = dir + "/codeobj.out" if exists(dir + "/codeobj.out") else ""
    return an^


def pcmap_of(codeobj: String) raises -> Pcmap:
    """The code map of a code object, built in Mojo: llvm-objdump disassembly, pc -> (size, dword, text).

    The system LLVM prints the gfx11 VOP3 forms correctly (ROCm's objdump prints some of them as
    .long and shifts every later pc), the same choice tools/export.py makes."""
    var pm = Pcmap()
    var out = subprocess.run("/usr/bin/llvm-objdump -d --mcpu=gfx1100 '" + codeobj + "'")
    for line in out.split("\n"):
        var cut = line.find("//")
        if cut < 0:
            if pm.symbol == "" and line.find("<") >= 0:  # the symbol header of the disassembly
                var open = line.find("<")
                var close = line.find(">", open)
                if close > open:
                    pm.symbol = String(String(line[byte=open + 1:close]))
            continue
        var text = String(line[byte=0:cut].strip())
        var tail = String(line[byte=cut + 2:].strip())  # "000000001900: BF850002"
        var colon = tail.find(":")
        if colon < 0:
            continue
        var pc = atol(String(tail[byte=0:colon].strip()), 16)
        var size = 0
        var word = 0
        for w in tail[byte=colon + 1:].strip().split(" "):
            if not w:
                continue
            if w.startswith("<"):  # a branch's target symbol follows its dwords
                break
            size += 4
            if size == 4:
                word = atol(String(w), 16)  # the first dword of the instruction
        if size:
            pm.add(pc, size, word, text)
    return pm^


def occupancy(codeobj: String) raises -> List[Tuple[String, Int, String]]:
    """Per kernel of a gfx1100 code object: (name, waves per SIMD it allows, what limits it: vgpr, lds or waves).
    LLVM's rules (AMDGPUBaseInfo getNumWavesPerEUWithNumVGPRs, AMDGPUSubtarget getOccupancyWithWorkGroupSizes),
    checked against hipcc's `; Occupancy:` comment (fixtures/occ)."""
    var out = List[Tuple[String, Int, String]]()
    var blocks = subprocess.run("/usr/bin/llvm-readelf --notes '" + codeobj + "'").split("\n  - ")
    for i in range(1, len(blocks)):
        var kv = Dict[String, String]()
        for line in blocks[i].split("\n"):
            var c = line.find(":")
            if line.startswith("    .") and not line.startswith("     ") and c > 0:
                kv[String(line[byte=4:c])] = String(line[byte=c + 1 :].strip())
        if ".vgpr_count" not in kv:
            continue
        var wave = atol(kv.get(".wavefront_size", "32"))
        var gran = 24 if wave == 32 else 12  # 1536 physical VGPRs (wave32), 768 (wave64)
        var vg = atol(kv[".vgpr_count"])
        var by_v = 16 if vg < gran else min(max((1536 if wave == 32 else 768) // ((vg + gran - 1) // gran * gran), 1), 16)
        var lds = atol(kv.get(".group_segment_fixed_size", "0"))
        var by_l = 16
        if lds > 0:
            var wgp = kv.get(".workgroup_processor_mode", "0") == "1"  # WGP: 128 KB over 4 SIMDs; CU: 64 KB over 2
            var wgs = (131072 if wgp else 65536) // lds
            var wg_waves = (atol(kv.get(".max_flat_workgroup_size", "1024")) + wave - 1) // wave
            by_l = 1 if wgs == 0 else min(16, (wgs * wg_waves + (3 if wgp else 1)) // (4 if wgp else 2))
        var n = min(by_v, by_l)
        out.append((kv.get(".name", ""), n, "waves" if n == 16 else ("vgpr" if by_v <= by_l else "lds")))
    return out^


def analyze_paths(paths: List[String], pm: Pcmap, label: String, tl_cap: Int, codeobj: String = "", strict: Bool = True, simd: Int = -1, starts: List[Int] = List[Int](), entry: Int = -1) raises -> Analysis:
    """The one walk of the trace: decode every blob, fill the row record and the per-wave timeline.

    paths are the raw SQTT files (blob_N.bin of a capture dir, the .att of a dispatch). strict
    makes an empty trace a failure; the app mode reads dispatches without instruction packets. simd: the traced SIMD
    when known (AMD's decoder names it in rocprofv3's filenames.json), else found from the trace; starts: AMD's waves
    of the dispatch (slot << 40 | start time), the only ones walked when given; entry: the kernel's first pc (the lowest pc
    of the code object when unknown or not in the map)."""
    var start_pc = entry if entry in pm.text else pm.first
    var fit = masks(pm)
    var f0 = pm.first
    var tab = Table(f0, len(fit))
    var start = -1  # pc of the first traced instruction
    var mismatches = 0
    var an = Analysis()
    an.dir = label
    an.kernel = pm.symbol
    var tl_lane = Dict[Int, Int]()  # wave -> timeline lane (-1: past the cap)
    var tl_last = Dict[Int, Int]()  # wave -> index of its last timeline entry
    var dump_w = atol(getenv("WAVEWAIT_PACKETS", "-1"), 10)  # debug: print this wave's packets and the pc each lands on
    var only_w = atol(getenv("WAVEWAIT_ONLY", "-1"), 10)  # check: walk one wave slot (never two waves alive at once)
    var keep = Dict[Int, Bool]()
    for k in starts:
        keep[k] = True
    for path in paths:
        var p = Decoder(open(path, "r").read_bytes(stat(path).st_size))  # unsized read_bytes() peaks at ~5x the file
        # pass 1: skip blobs without instruction packets, and find the SIMD the instruction packets belong to: the
        # one where every packet's wave slot is alive (slots repeat across the SIMDs of TRACED_CU, so a wrong SIMD
        # walks other waves' packets; rocprofv3 ATT traced SIMD 3 on examples/diverge.hip)
        var has_inst = False
        var alive = List[Int](length=4, fill=0)  # wave slot bitmask per SIMD of TRACED_CU
        var cand = 0xF
        var seen = 0  # packets judged; 1024 decide (the SIMDs still left then are indistinguishable: take the lowest)
        while (not has_inst or (simd < 0 and cand & (cand - 1) and seen < 1024)) and p.next():
            if (p.kind == WAVESTART or p.kind == WAVEEND) and p.unit() // 4 == TRACED_CU:
                alive[p.simd()] = alive[p.simd()] | 1 << p.wave() if p.kind == WAVESTART else alive[p.simd()] & ~(1 << p.wave())
            elif p.kind >= INST and not (p.kind == INST and bits(p.reg, 19, 13) >= 0x50 and bits(p.reg, 19, 13) <= 0x66):
                has_inst = has_inst or p.kind == INST
                seen += 1 if has_inst else 0
                var m = bits(p.reg, 23, 8) if p.kind == IMM_MASK else 1 << p.wave()
                var keep = cand
                for sd in range(4):
                    if alive[sd] & m != m:
                        keep &= ~(1 << sd)
                cand = keep if keep else cand  # a packet no SIMD explains (a wave started before the trace) decides nothing
        if not has_inst:
            continue
        # the traced (cu, simd): AMD's when given (slot liveness cannot tell two SIMDs apart that run alike: kinestride's
        # smooth kernel walked SIMD 1 for SIMD 3), else the search's, the lowest when still ambiguous
        var tu = TRACED_CU * 4 + (simd if simd >= 0 else Int(count_trailing_zeros(cand)))
        p.rewind()
        var pc = List[Int](length=32, fill=-1)  # by wave slot, -1: no wave walked there
        var alive_n = 0
        var last = List[Int](length=32, fill=0)
        var prev = List[Int](length=32, fill=-1)  # pc of the wave's previous traced instruction
        var prev_d = List[Int](length=32, fill=0)  # and the delta it issued with
        var regs = Dict[Int, Int]()  # wave << 8 | sgpr -> the pc it holds: no packet names the target of a call or an s_setpc
        # exposed cycles: every cycle the SIMD issued nothing is shared by the k waves alive then; g is that
        # share summed over time (XL units), a wave takes g - g_last when it issues, charged where the cycles go
        var g = 0
        var g_last = List[Int](length=32, fill=0)  # by wave slot (5 bits): no hashing per issue
        var t_prev = -1
        var issued = False  # an instruction issued at t_prev (its cycle is not idle)
        var idle_n = 0
        var occ = 0
        var span = 0
        var occ_max = 0
        var bcount = Dict[Int, Int]()  # branch pc << 5 | wave slot -> times the wave alive in that slot took it
        var ended = 0  # traced waves ended so far
        var lost = 0  # wave slots whose packets stopped matching the code (a trap handler: CWSR save on preemption)
        while p.next():
            var w = p.wave()
            var traced = p.kind >= INST or ((p.kind == WAVESTART or p.kind == WAVEEND) and p.unit() == tu)
            if traced and p.time != t_prev:
                var k = alive_n
                if t_prev >= 0 and k > 0:
                    var idle = max(0, p.time - t_prev - (1 if issued else 0))
                    idle_n += idle
                    occ += (p.time - t_prev) * k
                    span += p.time - t_prev
                    occ_max = max(occ_max, k)
                    g += idle * (XL // k)
                t_prev = p.time
                issued = False
            if p.kind == WAVESTART and p.unit() == tu and (only_w < 0 or w == only_w) and (not starts or (w << 40 | p.time) in keep):
                g_last[w] = g
                alive_n += 1 if pc[w] < 0 else 0
                pc[w] = start_pc
                last[w] = p.time
                prev[w] = -1
                prev_d[w] = 0
                an.waves += 1
                if an.lanes < tl_cap:
                    tl_lane[w] = an.lanes
                    an.lanes += 1
                else:
                    tl_lane[w] = -1
                tl_last[w] = -1
            elif p.kind >= INST:
                # IMMEDIATE_MASK: one immediate for every wave in its mask; a bit mask, not a list (no allocation per packet)
                var tmask = bits(p.reg, 23, 8) if p.kind == IMM_MASK else 1 << w
                while tmask:
                    var w = count_trailing_zeros(tmask)
                    tmask &= tmask - 1
                    if pc[w] < 0 or lost >> w & 1:
                        continue
                    var opv = bits(p.reg, 19, 13)
                    if p.kind == INST and opv >= 0x50 and opv <= 0x66:
                        continue  # OTHER_ ops: memory ops of the other SIMD
                    var a = pc[w]
                    while at(fit, f0, a) & DELAY:  # never emits a packet
                        a += pm.size[a]
                    if not at(fit, f0, a) & HAS:  # walked off the code map (wrong code object): drop the wave
                        mismatches += 1
                        pc[w] = -1
                        alive_n -= 1
                        continue
                    var c = pclass(p.kind, opv)
                    var cb = 1 << String(CLASSES).find(c)
                    var bb = a
                    var found = False
                    for _ in range(3):  # skip instructions this packet cannot belong to (untraced)
                        var bm = at(fit, f0, bb)
                        if not bm & HAS:
                            break
                        if bm & cb:
                            a = bb
                            found = True
                            break
                        bb += pm.size[bb]
                        while at(fit, f0, bb) & DELAY:
                            bb += pm.size[bb]
                    if not found and not at(fit, f0, a) & cb:
                        mismatches += 1
                        if an.first_mismatch == "":
                            an.first_mismatch = "wave " + String(w) + " pc " + hex(a) + " packet class '" + c + "' op " + hex(opv) + " at instruction: " + pm.text[a]
                        lost |= 1 << w  # as AMD's stitcher: the rest of the wave is not this code, walk none of it
                        continue
                    if pm.text[a].startswith("s_code_end"):  # packet after s_endpgm (late immediate, slot reuse): drop the wave
                        mismatches += 1
                        pc[w] = -1
                        alive_n -= 1
                        continue
                    if w == dump_w:
                        print("PKT t=" + String(p.time) + " kind " + String(p.kind) + " op " + hex(opv) + " class " + c + (" ok " if found else " MISS ") + hex(a) + " " + pm.text[a])
                    start = a if start < 0 else start
                    var d = p.time - last[w]
                    var ch = d  # charged delta: the wait keeps the cycles it held
                    if prev[w] >= 0 and is_wait(pm.text[prev[w]]) and d > 1:
                        tab.charge(prev[w], d - 1, prev_d[w])  # the wait held the wave; the instruction itself issued in 1
                        ch = 1
                        var li = tl_last.get(w, -1)
                        if li >= 0:
                            an.tl_cyc[li] = an.tl_cyc[li] + (d - 1)
                    tab.add(a, d, ch)
                    var x = g - g_last[w]
                    g_last[w] = g
                    var xa = prev[w] if ch < d else a
                    tab.exposed[tab.i(xa)] += x
                    issued = True
                    var lane = tl_lane.get(w, -1)
                    if lane >= 0:
                        an.tl_wave.append(lane)
                        an.tl_pc.append(a)
                        an.tl_cyc.append(ch)
                        tl_last[w] = len(an.tl_cyc) - 1
                    last[w] = p.time
                    prev[w] = a
                    prev_d[w] = ch
                    var rop = pm.regop.get(a, 0)
                    var rkey = w << 8 | rop >> 3
                    if rop & 7 == PC_GET or rop & 7 == PC_CALL:
                        regs[rkey] = a + pm.size[a]
                    elif rop & 7 == PC_ADD and rkey in regs:
                        regs[rkey] = (regs[rkey] + pm.addend[a]) & 0xFFFFFFFF
                    if rop & 7 == PC_CALL:  # hand-written Sp3 kernels (MIOpen's convs) call subroutines
                        var off = pm.word[a] & 0xFFFF
                        pc[w] = a + pm.size[a] + 4 * (off - 0x10000 if off & 0x8000 else off)
                    elif p.kind == INST and opv == JUMP:  # taken branch: simm16 dwords after the next pc
                        if a not in tab.bmin:  # a wave that ended before this branch was first taken took it 0 times
                            tab.bmin[a] = 0 if ended else -1
                            tab.bmax[a] = 0 if ended else -1
                        bcount[a << 5 | w] = bcount.get(a << 5 | w, 0) + 1
                        var off = pm.word[a] & 0xFFFF
                        pc[w] = a + pm.size[a] + 4 * (off - 0x10000 if off & 0x8000 else off)
                    elif p.kind == INST and opv == 5 and far_target(pm, a) >= 0:  # s_setpc_b64 of a long branch
                        pc[w] = far_target(pm, a)
                    elif rop & 7 == PC_SET and rkey in regs:
                        pc[w] = regs[rkey]
                    else:
                        pc[w] = a + pm.size[a]
            elif p.kind == WAVEEND and p.unit() == tu and pc[w] >= 0:
                an.xtail += g - g_last[w]  # idle after its last traced issue (s_endpgm is never traced)
                ended += 1
                for b in tab.bmin.keys():  # wave-level divergence: did this wave take each branch as often as the others
                    var v = bcount.pop(b << 5 | w, 0)
                    tab.bmin[b] = v if tab.bmin[b] < 0 else min(tab.bmin[b], v)
                    tab.bmax[b] = max(tab.bmax[b], v)
                pc[w] = -1
                alive_n -= 1
                lost &= ~(1 << w)
        an.open_waves += alive_n
        an.idle += idle_n
        an.occ += occ
        an.span += span
        an.occ_max = max(an.occ_max, occ_max)
    if not tab.order:
        if strict:
            print("FAIL trace: no instruction packets for the traced SIMD (capture with SQTT_WGP_PROBE=3)")
            exit(1)
        return an^
    for a in tab.order:
        var r = Row(a, pm.size[a], pm.text[a])
        r.waves = tab.count[tab.i(a)]
        r.sum = tab.total[tab.i(a)]
        r.mx = tab.peak[tab.i(a)]
        r.csum = tab.ctotal[tab.i(a)]
        r.cmx = tab.cpeak[tab.i(a)]
        r.xsum = tab.exposed[tab.i(a)]
        r.bmin = tab.bmin.get(a, -1)
        r.bmax = tab.bmax.get(a, -1)
        an.xgrand += r.xsum
        r.cause = cause(r.text, tab.ctotal[tab.i(a)] * 10 // tab.count[tab.i(a)], a == start, a, pm.size[a])
        an.rows.append(r)
        an.idx[a] = len(an.rows) - 1
        an.by_cause[r.cause] = an.by_cause.get(r.cause, 0) + tab.ctotal[tab.i(a)]
        an.grand += tab.ctotal[tab.i(a)]
    an.mismatches = mismatches
    # the DWARF source positions, when the caller wants them and the capture ships the code
    # object it was built from (llvm-dwarfdump is a child process: it costs its own memory)
    if codeobj != "" and exists(codeobj):
        var lm = load_dwarf(codeobj)
        for i in range(len(an.rows)):
            var hit = lm.lookup(an.rows[i].pc)
            an.rows[i].line = hit[0]
            an.rows[i].file = hit[1]
    return an^

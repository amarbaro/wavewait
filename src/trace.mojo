# The trace core: the SQTT decoder, the code map of a capture and the one walk that fills the
# row record (src/row.mojo). Every output (table, json, by-line, hints, diff, app, budget) reads
# that record; this module only produces it.
from std.collections import Dict, List
from std.os import stat
from std.os.path import exists
from std.sys import exit
from std import subprocess
from row import Analysis, Row
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
            if self.special[op] == 1 and bits(self.reg, 9, 9) == 1 and bits(self.reg, 8, 8) == 0:
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


struct Pcmap(Movable):
    """The code map of one capture: pc -> (size, first dword, text)."""

    var size: Dict[Int, Int]
    var word: Dict[Int, Int]
    var text: Dict[Int, String]
    var first: Int
    var symbol: String  # the kernel the code object was built from ("" when unknown)

    def __init__(out self):
        self.size = Dict[Int, Int]()
        self.word = Dict[Int, Int]()
        self.text = Dict[Int, String]()
        self.first = -1
        self.symbol = ""

    def add(mut self, pc: Int, size: Int, word: Int, text: String):
        self.size[pc] = size
        self.word[pc] = word
        self.text[pc] = text
        self.first = pc if self.first < 0 else min(self.first, pc)



struct Table(Movable):
    """Per-pc cycle totals of one capture: as issued (total/peak) and with the waits charged (ctotal/cpeak)."""

    var order: List[Int]  # first-seen order, the tie order of the Python table
    var total: Dict[Int, Int]
    var count: Dict[Int, Int]
    var peak: Dict[Int, Int]
    var ctotal: Dict[Int, Int]
    var cpeak: Dict[Int, Int]

    def __init__(out self):
        self.order = List[Int]()
        self.total = Dict[Int, Int]()
        self.count = Dict[Int, Int]()
        self.peak = Dict[Int, Int]()
        self.ctotal = Dict[Int, Int]()
        self.cpeak = Dict[Int, Int]()

    def add(mut self, a: Int, d: Int, c: Int) raises:
        if a not in self.total:
            self.order.append(a)
            self.total[a] = 0
            self.count[a] = 0
            self.peak[a] = d
            self.ctotal[a] = 0
            self.cpeak[a] = c
        self.total[a] = self.total[a] + d
        self.count[a] = self.count[a] + 1
        self.peak[a] = max(self.peak[a], d)
        self.ctotal[a] = self.ctotal[a] + c
        self.cpeak[a] = max(self.cpeak[a], c)

    def charge(mut self, a: Int, extra: Int, own: Int) raises:
        """Add held cycles to the wait's sample already counted (own = what the wait itself took to issue)."""
        self.ctotal[a] = self.ctotal[a] + extra
        self.cpeak[a] = max(self.cpeak[a], own + extra)

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
        if op >= 0x1C and op <= 0x2E:
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
    return starts_any(t, ["s_waitcnt", "s_clause", "s_nop", "s_sleep", "s_endpgm", "s_set", "s_inst", "s_icache"]) or starts_any(t, mem)


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
    return analyze_paths(paths^, pm^, dir, tl_cap, codeobj, True)


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


def analyze_paths(paths: List[String], pm: Pcmap, label: String, tl_cap: Int, codeobj: String = "", strict: Bool = True) raises -> Analysis:
    """The one walk of the trace: decode every blob, fill the row record and the per-wave timeline.

    paths are the raw SQTT files (blob_N.bin of a capture dir, the .att of a dispatch). strict
    makes an empty trace a failure; the app mode reads dispatches without instruction packets."""
    var tab = Table()
    var start = -1  # pc of the first traced instruction
    var mismatches = 0
    var an = Analysis()
    an.dir = label
    var tl_lane = Dict[Int, Int]()  # wave -> timeline lane (-1: past the cap)
    var tl_last = Dict[Int, Int]()  # wave -> index of its last timeline entry
    for path in paths:
        var p = Decoder(open(path, "r").read_bytes(stat(path).st_size))  # unsized read_bytes() peaks at ~5x the file
        var has_inst = False  # pass 1: skip blobs without instruction packets
        while not has_inst and p.next():
            has_inst = p.kind == INST
        if not has_inst:
            continue
        p.rewind()
        var pc = Dict[Int, Int]()
        var last = Dict[Int, Int]()
        var prev = Dict[Int, Int]()  # pc of the wave's previous traced instruction
        var prev_d = Dict[Int, Int]()  # and the delta it issued with
        while p.next():
            var w = p.wave()
            if p.kind == WAVESTART and p.unit() == TRACED_CU * 4:
                pc[w] = pm.first
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
                var targets = List[Int]()  # IMMEDIATE_MASK: one immediate for every wave in its mask
                if p.kind == IMM_MASK:
                    for b in range(16):
                        if bits(p.reg, 8 + b, 8 + b):
                            targets.append(b)
                else:
                    targets.append(w)
                for w in targets:
                    if w not in pc:
                        continue
                    var opv = bits(p.reg, 19, 13)
                    if p.kind == INST and opv >= 0x50 and opv <= 0x66:
                        continue  # OTHER_ ops: memory ops of the other SIMD
                    var a = pc[w]
                    while a in pm.text and pm.text[a].startswith("s_delay_alu"):  # never emits a packet
                        a += pm.size[a]
                    if a not in pm.text:  # walked off the code map (wrong code object): drop the wave
                        mismatches += 1
                        _ = pc.pop(w)
                        continue
                    var c = pclass(p.kind, opv)
                    var bb = a
                    var found = False
                    for _ in range(3):  # skip instructions this packet cannot belong to (untraced)
                        if bb not in pm.text:
                            break
                        if fits(c, pm.text[bb]):
                            a = bb
                            found = True
                            break
                        bb += pm.size[bb]
                        while bb in pm.text and pm.text[bb].startswith("s_delay_alu"):
                            bb += pm.size[bb]
                    if not found and not fits(c, pm.text[a]):
                        mismatches += 1
                    if pm.text[a].startswith("s_code_end"):
                        print("FAIL walk: wave", w, "ran past s_endpgm")
                        exit(1)
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
                    var lane = tl_lane.get(w, -1)
                    if lane >= 0:
                        an.tl_wave.append(lane)
                        an.tl_pc.append(a)
                        an.tl_cyc.append(ch)
                        tl_last[w] = len(an.tl_cyc) - 1
                    last[w] = p.time
                    prev[w] = a
                    prev_d[w] = ch
                    if p.kind == INST and opv == JUMP:  # taken branch: simm16 dwords after the next pc
                        var off = pm.word[a] & 0xFFFF
                        pc[w] = a + pm.size[a] + 4 * (off - 0x10000 if off & 0x8000 else off)
                    else:
                        pc[w] = a + pm.size[a]
            elif p.kind == WAVEEND and p.unit() == TRACED_CU * 4 and w in pc:
                _ = pc.pop(w)
    if not tab.order:
        if strict:
            print("FAIL trace: no instruction packets for the traced SIMD (capture with SQTT_WGP_PROBE=3)")
            exit(1)
        return an^
    for a in tab.order:
        var r = Row(a, pm.size[a], pm.text[a])
        r.waves = tab.count[a]
        r.sum = tab.total[a]
        r.mx = tab.peak[a]
        r.csum = tab.ctotal[a]
        r.cmx = tab.cpeak[a]
        r.cause = cause(r.text, tab.ctotal[a] * 10 // tab.count[a], a == start, a, pm.size[a])
        an.rows.append(r)
        an.idx[a] = len(an.rows) - 1
        an.by_cause[r.cause] = an.by_cause.get(r.cause, 0) + tab.ctotal[a]
        an.grand += tab.ctotal[a]
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

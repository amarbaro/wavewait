# The one internal row record: every output (table, json, diff, hints, html) reads it.
# A row is one traced instruction, aggregated over the waves that issued it.
# sum/mx are the cycles as issued (the plain table); csum/cmx are the same cycles with the
# waits charged to the s_waitcnt that held them (the --causes view). The cause is read off
# the charged average, the way the retired Python analyzer labelled it.
from std.collections import Dict, List

comptime XL = 720720  # exposed-cycle unit: lcm(1..16), so an idle cycle shared by k <= 16 waves splits exactly

struct Row(ImplicitlyCopyable):
    var pc: Int
    var size: Int
    var text: String
    var waves: Int
    var sum: Int
    var mx: Int
    var csum: Int
    var cmx: Int
    var cause: String
    var line: Int
    var file: String  # the DWARF source file, "" when the capture has no debug info
    var bmin: Int  # a taken branch: min / max times one wave took it (-1: never taken); they differ when the waves
    var bmax: Int  # disagree on the path (wave-level divergence; lane-level EXEC is not in the trace)
    var xsum: Int  # exposed: SIMD idle cycles this instruction cost, shared among the waves waiting then (XL units)

    def __init__(out self, pc: Int, size: Int, text: String):
        self.pc = pc
        self.size = size
        self.text = text
        self.waves = 0
        self.sum = 0
        self.mx = 0
        self.csum = 0
        self.cmx = 0
        self.cause = ""
        self.line = 0
        self.file = ""
        self.xsum = 0
        self.bmin = -1
        self.bmax = -1


# One analysed capture: the rows, the cause totals, and the per-wave timeline (the first
# `lanes` waves, one lane each, with the charged cycles of every instruction they issued).
struct Analysis(Movable):
    var rows: List[Row]
    var idx: Dict[Int, Int]  # pc -> row index
    var by_cause: Dict[String, Int]
    var grand: Int
    var mismatches: Int
    var waves: Int
    var first_mismatch: String  # where the walk first lost alignment: wave, pc, packet class, instruction
    var open_waves: Int  # waves that started and never sent WAVEEND (trace buffer full, or cut short)
    var dir: String
    var tl_wave: List[Int]
    var tl_pc: List[Int]
    var tl_cyc: List[Int]
    var lanes: Int
    var idle: Int  # cycles the traced SIMD issued nothing while a wave was alive
    var xgrand: Int  # the rows' exposed cycles, XL units; xgrand + xtail == idle * XL
    var xtail: Int  # idle after a wave's last traced issue, XL units
    var occ: Int  # wave-cycles: alive waves summed over the span
    var span: Int
    var occ_max: Int
    var kernel: String  # the kernel symbol and the code object it came from ("" when the capture has none)
    var codeobj: String

    def __init__(out self):
        self.rows = List[Row]()
        self.idx = Dict[Int, Int]()
        self.by_cause = Dict[String, Int]()
        self.grand = 0
        self.mismatches = 0
        self.waves = 0
        self.open_waves = 0
        self.first_mismatch = ""
        self.dir = ""
        self.tl_wave = List[Int]()
        self.tl_pc = List[Int]()
        self.tl_cyc = List[Int]()
        self.lanes = 0
        self.idle = 0
        self.xgrand = 0
        self.xtail = 0
        self.occ = 0
        self.span = 0
        self.occ_max = 0
        self.kernel = ""
        self.codeobj = ""

    def row_of(self, pc: Int) -> Int:
        """Index of the row with that pc, or -1."""
        return self.idx.get(pc, -1)


def pcs_of(ref an: Analysis) -> List[Int]:
    """The row pcs in pc order."""
    var pcs = List[Int]()
    for r in an.rows:
        pcs.append(r.pc)
    sort(pcs)
    return pcs^


def tenths(total: Int, n: Int) -> Int:
    """total / n in tenths, rounded half to even like Python's %.1f on these exact values."""
    if n <= 0:
        return 0
    var q = (total * 10) // n
    var r = (total * 10) % n
    if 2 * r > n or (2 * r == n and q % 2 == 1):
        q += 1
    return q


def fmt1(total: Int, n: Int) -> String:
    """Format total / n with one decimal."""
    var q = tenths(total, n)
    return String(q // 10) + "." + String(q % 10)


def fmt1x10(v: Int) -> String:
    """Format a tenths value with one decimal (negative aware)."""
    if v < 0:
        return "-" + fmt1x10(-v)
    return String(v // 10) + "." + String(v % 10)


def sgn10(v: Int) -> String:
    """A signed tenths value: "+495.6" / "-673.5" / "0.0"."""
    return "+" + fmt1x10(v) if v > 0 else fmt1x10(v)


def xcyc(x: Int) -> Int:
    """XL units to whole cycles, rounded."""
    return (x + XL // 2) // XL


def pad(s: String, w: Int) -> String:
    return " " * (w - s.byte_length()) + s


def lpad(s: String, w: Int) -> String:
    """Left aligned in w columns."""
    return s + " " * max(1, w - s.byte_length())  # at least one space: a long name never runs into the next column


def sgni(v: Int) -> String:
    """A signed integer: "+7501" / "-745" / "0"."""
    if v > 0:
        return "+" + String(v)
    return String(v)


def jstr(s: String) -> String:
    """A JSON string literal (backslash and quote escaped)."""
    return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\""

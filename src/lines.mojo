# The DWARF source positions of a code object: `llvm-dwarfdump --debug-line` parsed once into a
# pc-sorted table and binary searched per row. A capture without a code object has no positions.
from std.collections import Dict, List
from std import subprocess


struct LineMap(Movable):
    var addrs: List[Int]
    var lines: List[Int]
    var files: List[String]

    def __init__(out self):
        self.addrs = List[Int]()
        self.lines = List[Int]()
        self.files = List[String]()

    def lookup(self, pc: Int) -> Tuple[Int, String]:
        """The (line, file) of the row that covers pc: the last row with an address <= pc.
        (0, "") when the pc has no position."""
        var lo = 0
        var hi = len(self.addrs) - 1
        var best = -1
        while lo <= hi:
            var mid = (lo + hi) // 2
            if self.addrs[mid] <= pc:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best < 0:
            return (0, "")
        return (self.lines[best], self.files[best])


def fields(s: String) -> List[String]:
    """The whitespace separated fields of a line."""
    var out = List[String]()
    for f in s.split(" "):
        if f.byte_length() > 0:
            out.append(String(f))
    return out^


def parse_dwarf(text: String) raises -> LineMap:
    """The line table of an `llvm-dwarfdump --debug-line` dump: the file names, then the rows
    (address, line, file) in the order dwarfdump prints them (the address rises)."""
    var lm = LineMap()
    var names = Dict[Int, String]()
    var idx = -1
    for ln in text.split("\n"):
        var t = String(ln.strip())
        if t.startswith("file_names["):
            var a = t.find("[")
            var b = t.find("]")
            idx = atol(String(t[byte=a + 1 : b]).strip(), 10)
        elif t.startswith("name:") and idx >= 0:
            var a = t.find("\"")
            var b = t.rfind("\"")
            if a >= 0 and b > a:
                names[idx] = String(t[byte=a + 1 : b])
        elif t.startswith("0x"):
            var f = fields(t)
            if len(f) >= 4 and f[0].startswith("0x"):
                lm.addrs.append(atol(f[0], 16))
                lm.lines.append(atol(f[1], 10))
                lm.files.append(names.get(atol(f[3], 10), ""))
    return lm^


def load_dwarf(codeobj: String) raises -> LineMap:
    """The line table of a code object, read with llvm-dwarfdump."""
    return parse_dwarf(subprocess.run("llvm-dwarfdump --debug-line " + codeobj))

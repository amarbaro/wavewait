# wavewait table|diff|app|budget DIR: per-instruction issue cost from raw RDNA3 SQTT.
# DIR comes from tools/export.py (a tinygrad profile) or tools/from-att.sh (a rocprofv3 --att run):
# blob_N.bin (raw SQTT per shader engine) and pcmap.tsv (pc, size, dword, text).
# The decoder is tinygrad's nibble state machine, tables generated into sqtt_table.mojo.
# One walk of the trace fills the row record (src/row.mojo); every output reads it:
# table (default), --causes, --json, --by-line, --html, diff, app, budget.
from std.collections import Dict, List
from std.os import stat
from std.os.path import exists
from std.sys import argv, exit
from std.os import getenv
from std.python import Python
from row import Analysis, Row, fmt1, fmt1x10, jstr, lpad, pad, pcs_of, sgn10, tenths
from trace import TL_CAP, analyze
from app import app_main
from diff import diff_json, diff_rows
from hints import print_hints
from html import write_html
from lines import LineMap, load_dwarf


def unpack(pkl: String, kernel: String) raises -> String:
    """Export PKL to PKL.raw/ with tools/export.py; tinygrad must be importable to unpickle its events."""
    var os = Python.import_module("os")
    var repo = String(os.path.dirname(os.path.dirname(os.path.realpath(String(argv()[0])))))
    Python.add_to_path(repo + "/tools")
    Python.add_to_path(getenv("TINYGRAD", String(os.path.expanduser("~/tinygrad"))))
    var out = pkl + ".raw"
    _ = Python.import_module("export").export(pkl, out, kernel)
    return out


def print_table(ref an: Analysis, top: Int, causes: Bool) raises:
    """The table of the retired Python analyzer, byte for byte (fixtures/golden)."""
    var by_pc = List[Int]()
    for r in an.rows:
        by_pc.append(r.pc)
    sort(by_pc)
    print("    pc    avg   max   n  " + ("cause         " if causes else "") + "instruction")
    for a in by_pc:
        var r = an.rows[an.row_of(a)]
        var s = r.sum if not causes else r.csum
        var m = r.mx if not causes else r.cmx
        var row = pad(hex(a), 6) + " " + pad(fmt1(s, r.waves), 6) + " " + pad(String(m), 5) + " " + pad(String(r.waves), 3) + "  "
        if causes:
            row += r.cause + " " * (14 - r.cause.byte_length())
        print(row + r.text)
    if an.mismatches:
        print("WARN", an.mismatches, "packets matched no instruction class (alignment suspect)")
    # stable insertion sort by average, descending: ties keep first-seen order
    var ranked = List[Row]()
    for r in an.rows:
        ranked.append(r)
    for i in range(1, len(ranked)):
        var j = i
        while j > 0 and ranked[j].sum * ranked[j - 1].waves > ranked[j - 1].sum * ranked[j].waves:
            var t = ranked[j]
            ranked[j] = ranked[j - 1]
            ranked[j - 1] = t
            j -= 1
    print("\ntop", top, "by average delta:")
    for i in range(min(top, len(ranked))):
        var r = ranked[i]
        print(pad(hex(r.pc), 6), pad(fmt1(r.sum, r.waves), 6), "", r.text)
    if causes:
        print("\ncycles by cause (all traced waves):")
        var keys = List[String]()
        for k in an.by_cause.keys():
            keys.append(k)
        for i in range(1, len(keys)):  # largest first
            var j = i
            while j > 0 and an.by_cause[keys[j]] > an.by_cause[keys[j - 1]]:
                keys.swap_elements(j, j - 1)
                j -= 1
        for k in keys:
            print(pad(String(an.by_cause[k]), 8), pad(String(an.by_cause[k] * 100 // an.grand), 3) + "%", "", k)

def print_by_line(ref an: Analysis) raises:
    """The costs rolled up per source line, biggest first (the DWARF positions of the capture)."""
    var cyc = Dict[String, Int]()
    var cnt = Dict[String, Int]()
    var txt = Dict[String, String]()
    for r in an.rows:
        var key = r.file + ":" + String(r.line) if r.line > 0 else "(no debug info)"
        cyc[key] = cyc.get(key, 0) + r.csum
        cnt[key] = cnt.get(key, 0) + 1
        var t = txt.get(key, "")
        txt[key] = t + (" | " if t != "" else "") + r.text
    var keys = List[String]()
    for k in cyc.keys():
        keys.append(k)
    for i in range(1, len(keys)):  # largest first
        var j = i
        while j > 0 and cyc[keys[j]] > cyc[keys[j - 1]]:
            keys.swap_elements(j, j - 1)
            j -= 1
    print("   cycles   n  " + pad("source line", 34) + " instructions")
    var total = 0
    for k in keys:
        total += cyc[k]
        print(pad(String(cyc[k]), 8) + " " + pad(String(cnt[k]), 3) + " " + pad(k, 34) + " " + txt[k])
    print("")
    print(String(total), "cycles over", String(len(keys)), "source lines (the table total is", String(an.grand), ")")

def print_json(ref an: Analysis) raises:
    """The rows as JSON: one object per row (pc order) plus the totals, stable key order."""
    var by_pc = pcs_of(an)
    print("{")
    print("  \"dir\": " + jstr(an.dir) + ",")
    print("  \"waves\": " + String(an.waves) + ",")
    print("  \"grand\": " + String(an.grand) + ",")
    print("  \"rows\": [")
    for i in range(len(by_pc)):
        var r = an.rows[an.idx[by_pc[i]]]
        var c = "," if i < len(by_pc) - 1 else ""
        print("    {\"pc\": " + String(r.pc) + ", \"text\": " + jstr(r.text) + ", \"waves\": " + String(r.waves) + ", \"avg\": " + fmt1x10(tenths(r.sum, r.waves)) + ", \"sum\": " + String(r.sum) + ", \"max\": " + String(r.mx) + ", \"cavg\": " + fmt1x10(tenths(r.csum, r.waves)) + ", \"csum\": " + String(r.csum) + ", \"cmax\": " + String(r.cmx) + ", \"cause\": " + jstr(r.cause) + ", \"file\": " + jstr(r.file) + ", \"line\": " + String(r.line) + "}" + c)
    print("  ],")
    print("  \"causes\": {")
    var keys = List[String]()
    for k in an.by_cause.keys():
        keys.append(k)
    for i in range(1, len(keys)):  # largest first
        var j = i
        while j > 0 and an.by_cause[keys[j]] > an.by_cause[keys[j - 1]]:
            keys.swap_elements(j, j - 1)
            j -= 1
    for i in range(len(keys)):
        var c = "," if i < len(keys) - 1 else ""
        print("    " + jstr(keys[i]) + ": " + String(an.by_cause[keys[i]]) + c)
    print("  }")
    print("}")

def diff_main(args: List[String]) raises:
    """The aligned diff of two captures (wavewait diff A B)."""
    if len(args) < 4:
        print("usage: wavewait diff A B")
        exit(1)
    var a = analyze(args[2], 0)
    var b = analyze(args[3], 0)
    if flag(args, "--json"):
        diff_json(a, b)
    else:
        diff_rows(a, b)

def budget_main(args: List[String]) raises:
    """Budget gate (wavewait budget FILE DIR): exit 1 on any rule the capture breaks. FILE lines: `cycles N` (max total)
    or `cause NAME PCT` (max share of one cause, NAME may hold spaces); `#` starts a comment."""
    if len(args) < 4:
        print("usage: wavewait budget FILE DIR")
        exit(1)
    var an = analyze(args[3], 0)
    var broken = 0
    for raw in open(args[2], "r").read().split("\n"):
        var line = String(raw.split("#")[0].strip())
        if line == "":
            continue
        var f = line.split()
        var limit = atol(String(f[len(f) - 1]))
        var have = an.grand
        var rule = String(f[0])
        if rule == "cause":
            var name = String(" ").join(f[1 : len(f) - 1])
            have = an.by_cause.get(String(name), 0) * 100 // max(an.grand, 1)
        elif rule != "cycles":
            raise Error("budget: unknown rule " + line)
        if have > limit:
            print("OVER " + line + " (have " + String(have) + ")")
            broken += 1
    if broken:
        exit(1)
    print("ok budget: " + String(an.grand) + " cycles within " + args[2])


def flag(args: List[String], name: String) -> Bool:
    for a in args:
        if a == name:
            return True
    return False


def opt(args: List[String], name: String, dflt: String) -> String:
    for i in range(len(args)):
        if args[i] == name and i + 1 < len(args):
            return args[i + 1]
    return dflt


def main() raises:
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    if len(args) < 2:
        print("usage: wavewait [table|diff|app|budget] DIR [options]")
        exit(1)
    var cmd = args[1]
    if cmd == "diff":
        diff_main(args)
        return
    if cmd == "app":
        app_main(args)
        return
    if cmd == "budget":
        budget_main(args)
        return
    if cmd == "sim":
        print("sim: blocked, the per-SIMD scheduler does not exist yet (docs/W7-blocked.md)")
        exit(2)
    var dir = String(args[1])
    if cmd == "table":
        if len(args) < 3:
            print("usage: wavewait table DIR [options]")
            exit(1)
        dir = String(args[2])
    var kernel = opt(args, "--kernel", "hsaco")
    if dir.endswith(".pkl"):
        dir = unpack(dir, kernel)
    var top = 10
    if flag(args, "--top"):
        top = atol(opt(args, "--top", "10"), 10)
    var causes = flag(args, "--causes")
    # the source positions only cost a dwarfdump child when the output shows them
    var want_lines = flag(args, "--by-line") or flag(args, "--json")
    var html = opt(args, "--html", "")
    var an = analyze(dir, TL_CAP if html != "" else 0, dir + "/codeobj.out" if want_lines else "")
    if html != "":
        write_html(an, html)
    elif flag(args, "--hints"):
        print_hints(an)
    elif flag(args, "--by-line"):
        print_by_line(an)
    elif flag(args, "--json"):
        print_json(an)
    else:
        print_table(an, top, causes)

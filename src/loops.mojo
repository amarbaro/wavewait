# --loops: the charged cycles rolled up per innermost loop of the kernel. A loop is a backward
# branch (s_cbranch_* / s_branch with a negative simm16) and the executed rows between its target
# and the branch. A short loop that polls memory (a glc load or an atomic, or s_sleep) is flagged
# spin: in a persistent kernel those cycles are blocks waiting at a grid barrier, not work.
from std.collections import Dict, List
from row import Analysis, lpad, pad

comptime SPIN_MAX = 16  # executed instructions in a body still read as a poll loop


def branch_target(pc: Int, size: Int, text: String) raises -> Int:
    """The target pc of a branch row, or -1 when the row is not a branch. objdump prints the simm16 unsigned."""
    if not (text.startswith("s_cbranch") or text.startswith("s_branch")):
        return -1
    var f = text.split()
    if len(f) < 2:
        return -1
    var imm = atol(String(f[len(f) - 1]), 10)
    if imm >= 0x8000:
        imm -= 0x10000
    return pc + size + 4 * imm


def print_loops(ref an: Analysis, top: Int) raises:
    var pcs = List[Int]()
    for r in an.rows:
        pcs.append(r.pc)
    sort(pcs)
    var lo = List[Int]()  # loop i spans [lo[i], hi[i]] (target .. backward branch)
    var hi = List[Int]()
    for a in pcs:
        ref r = an.rows[an.row_of(a)]
        var t = branch_target(r.pc, r.size, r.text)
        if t >= 0 and t <= r.pc:
            lo.append(t)
            hi.append(r.pc)
    var cyc = List[Int]()
    var n = List[Int]()
    var spin = List[Bool]()
    var causes = List[Dict[String, Int]]()
    for _ in range(len(lo)):
        cyc.append(0)
        n.append(0)
        spin.append(False)
        causes.append(Dict[String, Int]())
    var outside = 0
    for a in pcs:
        ref r = an.rows[an.row_of(a)]
        var best = -1  # the innermost enclosing loop: the shortest span
        for i in range(len(lo)):
            if lo[i] <= a and a <= hi[i] and (best < 0 or hi[i] - lo[i] < hi[best] - lo[best]):
                best = i
        if best < 0:
            outside += r.csum
            continue
        cyc[best] += r.csum
        n[best] += 1
        causes[best][r.cause] = causes[best].get(r.cause, 0) + r.csum
        if " glc" in r.text or "atomic" in r.text or r.text.startswith("s_sleep"):
            spin[best] = True
    var order = List[Int]()
    for i in range(len(lo)):
        if n[i] > 0:
            order.append(i)
    for i in range(1, len(order)):  # largest first
        var j = i
        while j > 0 and cyc[order[j]] > cyc[order[j - 1]]:
            order.swap_elements(j, j - 1)
            j -= 1
    var g = max(an.grand, 1)
    print("     cycles    %  insts  range                 kind  top cause")
    var spin_total = 0
    for i in order:
        var isspin = spin[i] and n[i] <= SPIN_MAX
        if isspin:
            spin_total += cyc[i]
    var shown = 0
    for i in order:
        if shown >= top:
            break
        shown += 1
        var isspin = spin[i] and n[i] <= SPIN_MAX
        var tc = ""
        var tv = -1
        for k in causes[i].keys():
            if causes[i][k] > tv:
                tv = causes[i][k]
                tc = k
        print(pad(String(cyc[i]), 11) + pad(String(cyc[i] * 100 // g), 5) + pad(String(n[i]), 7) + "  " + lpad(hex(lo[i]) + ".." + hex(hi[i]), 20) + pad("spin" if isspin else "loop", 6) + "  " + tc)
    print(pad(String(outside), 11) + pad(String(outside * 100 // g), 5) + "         (outside every loop)")
    print("")
    print(String(len(order)), "loops;", String(spin_total), "cycles (" + String(spin_total * 100 // g) + "%) in spin loops, of", String(an.grand))

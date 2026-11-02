# The diff of two captures: align the rows by instruction text, show what moved.
# The alignment is the longest common subsequence of the two text sequences, so a row that
# exists in one capture only shows up as - (removed) or + (inserted).
# A pair above ALIGN_CELLS is aligned by Hirschberg's recursion (the same LCS length, memory linear in the rows):
# a kernel with 14,000 rows would need a 730 MB table. A pair under it keeps the table, whose tie-breaking the
# goldens froze. Only the row record is needed here, so this module never imports the analyzer.
from std.collections import Dict, List
from std.os import getenv
from classes import N_CLASSES, ClassTotals, class_names
from row import Analysis, fmt1x10, jstr, lpad, pad, pcs_of, sgn10, sgni, tenths

comptime ALIGN_CELLS = 4000000  # WAVEWAIT_ALIGN_CELLS overrides it (the checks force either path)


struct DRow(ImplicitlyCopyable):
    """One aligned row of a diff: the mark (-, + or space), the pc, the two averages in tenths
    (-1 when that side has no row), the delta and the instruction."""

    var mark: String
    var pc: Int
    var av: Int
    var bv: Int
    var delta: Int
    var cause: String
    var text: String
    var ca: Int  # charged cycles of the row in a and in b (0 when that side has no row)
    var cb: Int

    def __init__(out self):
        self.ca = 0
        self.cb = 0
        self.mark = " "
        self.pc = -1
        self.av = -1
        self.bv = -1
        self.delta = 0
        self.cause = ""
        self.text = ""


def intern(a: List[String], b: List[String], mut ia: List[Int], mut ib: List[Int]) raises:
    """The two key sequences as integer ids (equal text, equal id)."""
    var ids = Dict[String, Int]()
    for t in a:
        if t not in ids:
            ids[t] = len(ids)
        ia.append(ids[t])
    for t in b:
        if t not in ids:
            ids[t] = len(ids)
        ib.append(ids[t])


def lcs_row(a: List[Int], alo: Int, ahi: Int, b: List[Int], blo: Int, bhi: Int, rev: Bool) -> List[Int32]:
    """LCS lengths of a[alo:ahi] against every prefix of b[blo:bhi] (every suffix of it when `rev`): m + 1 values."""
    var m = bhi - blo
    var row = List[Int32]()
    for _ in range(m + 1):
        row.append(Int32(0))
    for ii in range(ahi - alo):
        var x = a[alo + ii] if not rev else a[ahi - 1 - ii]
        var diag = Int32(0)
        for j in range(1, m + 1):
            var up = row[j]
            var y = b[blo + j - 1] if not rev else b[bhi - j]
            if x == y:
                row[j] = diag + Int32(1)
            elif row[j - 1] > up:
                row[j] = row[j - 1]
            diag = up
    return row^


def hirschberg(a: List[Int], alo: Int, ahi: Int, b: List[Int], blo: Int, bhi: Int, mut out: List[Int]):
    """Append the aligned pairs of a[alo:ahi] and b[blo:bhi] (the layout of align())."""
    if ahi == alo:
        for j in range(blo, bhi):
            out.append(-1)
            out.append(j)
        return
    if bhi == blo:
        for i in range(alo, ahi):
            out.append(i)
            out.append(-1)
        return
    if ahi - alo == 1:
        var hit = -1
        for j in range(blo, bhi):
            if b[j] == a[alo]:
                hit = j
                break
        if hit < 0:
            out.append(alo)
            out.append(-1)
        for j in range(blo, bhi):
            if j == hit:
                out.append(alo)
                out.append(j)
            else:
                out.append(-1)
                out.append(j)
        return
    var mid = (alo + ahi) // 2
    var f = lcs_row(a, alo, mid, b, blo, bhi, False)
    var g = lcs_row(a, mid, ahi, b, blo, bhi, True)
    var m = bhi - blo
    var best = -1
    var cut = 0
    for k in range(m + 1):
        var v = Int(f[k]) + Int(g[m - k])
        if v > best:
            best = v
            cut = k
    hirschberg(a, alo, mid, b, blo, blo + cut, out)
    hirschberg(a, mid, ahi, b, blo + cut, bhi, out)


def align(a: List[String], b: List[String]) raises -> List[Int]:
    """Align two key sequences. Two entries per aligned pair: the index in a and the index in b
    (-1 when that side has no match). A row of a with no match is removed (-), a row of b with
    no match is inserted (+)."""
    var n = len(a)
    var m = len(b)
    var out = List[Int]()
    var ia = List[Int]()
    var ib = List[Int]()
    intern(a, b, ia, ib)
    if (n + 1) * (m + 1) > atol(getenv("WAVEWAIT_ALIGN_CELLS", String(ALIGN_CELLS)), 10):
        hirschberg(ia, 0, n, ib, 0, m, out)
        return out^
    var w = m + 1
    var dp = List[Int32]()
    for _ in range((n + 1) * w):
        dp.append(Int32(0))
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if ia[i] == ib[j]:
                dp[i * w + j] = dp[(i + 1) * w + j + 1] + Int32(1)
            else:
                dp[i * w + j] = max(dp[(i + 1) * w + j], dp[i * w + j + 1])
    var i = 0
    var j = 0
    while i < n or j < m:
        if i < n and j < m and ia[i] == ib[j]:
            out.append(i)
            out.append(j)
            i += 1
            j += 1
        elif j < m and (i >= n or dp[i * w + j + 1] > dp[(i + 1) * w + j]):
            out.append(-1)  # the row of b has no match in a: inserted
            out.append(j)
            j += 1
        else:
            out.append(i)  # the row of a has no match in b: removed
            out.append(-1)
            i += 1
    return out^


def diff_pairs(ref a: Analysis, ref b: Analysis) raises -> List[DRow]:
    """The aligned rows of two captures: one entry per aligned pair, removed or inserted."""
    var pa = pcs_of(a)
    var pb = pcs_of(b)
    var ka = List[String]()
    var kb = List[String]()
    for pc in pa:
        ka.append(a.rows[a.idx[pc]].text)
    for pc in pb:
        kb.append(b.rows[b.idx[pc]].text)
    var al = align(ka, kb)
    var out = List[DRow]()
    for k in range(0, len(al), 2):
        var x = al[k]
        var y = al[k + 1]
        var d = DRow()
        if x >= 0:
            var ra = a.rows[a.idx[pa[x]]]
            d.pc = ra.pc
            d.ca = ra.csum
            d.av = tenths(ra.csum, ra.waves)
            d.cause = ra.cause
            d.text = ra.text
        if y >= 0:
            var rb = b.rows[b.idx[pb[y]]]
            if d.pc < 0:
                d.pc = rb.pc
            d.cb = rb.csum
            d.bv = tenths(rb.csum, rb.waves)
            d.cause = rb.cause
            d.text = rb.text
        if x >= 0 and y >= 0:
            d.delta = d.bv - d.av
        elif x >= 0:
            d.mark = "-"
            d.delta = -d.av
        else:
            d.mark = "+"
            d.delta = d.bv
        out.append(d)
    return out^


def diff_counts(rows: List[DRow]) raises -> Tuple[Int, Int, Int]:
    """The aligned, inserted and removed counts of a diff."""
    var aligned = 0
    var inserted = 0
    var removed = 0
    for d in rows:
        if d.mark == "-":
            removed += 1
        elif d.mark == "+":
            inserted += 1
        else:
            aligned += 1
    return (aligned, inserted, removed)


def shift_of(ref a: Analysis, ref b: Analysis, cause: String) raises -> Int:
    """Cycles moved between the captures by one cause."""
    return b.by_cause.get(cause, 0) - a.by_cause.get(cause, 0)


def diff_causes(ref a: Analysis, ref b: Analysis) raises -> List[String]:
    """The causes of both captures, the biggest movers first."""
    var causes = List[String]()
    for c in a.by_cause.keys():
        causes.append(c)
    for c in b.by_cause.keys():
        if c not in causes:
            causes.append(c)
    for i in range(1, len(causes)):
        var j = i
        while j > 0 and abs(shift_of(a, b, causes[j])) > abs(shift_of(a, b, causes[j - 1])):
            causes.swap_elements(j, j - 1)
            j -= 1
    return causes^


def cut_rows(rows: List[DRow], top: Int) -> List[DRow]:
    """The `top` rows that moved the most total cycles (|cb - ca|, biggest first, ties in alignment order);
    all rows, in alignment order, when top <= 0."""
    if top <= 0 or top >= len(rows):
        return rows.copy()
    var keys = List[Int]()
    for i in range(len(rows)):
        keys.append(abs(rows[i].cb - rows[i].ca) * 16777216 + (16777215 - i))
    sort(keys)
    var out = List[DRow]()
    for k in range(top):
        out.append(rows[16777215 - keys[len(keys) - 1 - k] % 16777216])
    return out^


def pct10(delta: Int, base: Int) -> String:
    """The change as a signed percent with one decimal."""
    var t = tenths(abs(delta) * 100, max(base, 1))
    return ("-" if delta < 0 else "+") + fmt1x10(t) + "%"


def diff_rows(ref a: Analysis, ref b: Analysis, top: Int = 0, by_class: Bool = False) raises:
    """Print the aligned diff of two captures: per-row cycle delta and the per-cause shift. With top > 0 only the
    rows that moved the most total cycles, with a dcycles column (the total over all waves); the class table
    (by_class) and the grand totals follow the causes."""
    var rows = diff_pairs(a, b)
    var cnt = diff_counts(rows)
    var cut = top > 0
    var shown = cut_rows(rows, top)
    print("diff", a.dir, "->", b.dir)
    print(" " + pad("pc", 6) + " " + pad("avgA", 8) + " " + pad("avgB", 8) + " " + pad("delta", 8) + (" " + pad("dcycles", 11) if cut else "") + "  " + lpad("cause", 14) + " instruction")
    for d in shown:
        var sa = "-" if d.av < 0 else fmt1x10(d.av)
        var sb = "-" if d.bv < 0 else fmt1x10(d.bv)
        var sd = sgn10(d.delta)
        if d.mark == "-":
            sd = "-" + sa
        elif d.mark == "+":
            sd = "+" + sb
        print(d.mark + pad(hex(d.pc), 6) + " " + pad(sa, 8) + " " + pad(sb, 8) + " " + pad(sd, 8) + (" " + pad(sgni(d.cb - d.ca), 11) if cut else "") + "  " + lpad(d.cause, 14) + " " + d.text)
    print("")
    print(String(cnt[0]), "rows aligned,", String(cnt[1]), "inserted,", String(cnt[2]), "removed")
    if cut and top < len(rows):
        print("(" + String(top) + " of " + String(len(rows)) + " rows shown, by total cycles moved; --top 0 prints all)")
    if by_class:
        var ta = ClassTotals(a)
        var tb = ClassTotals(b)
        var names = class_names()
        print("")
        print("cycles by class:")
        print("  " + lpad("class", 17) + pad("cyclesA", 11) + pad("cyclesB", 11) + pad("shift", 11) + pad("per wave A", 12) + pad("per wave B", 12))
        for c in range(N_CLASSES):
            print("  " + lpad(names[c], 17) + pad(String(ta.cycles[c]), 11) + pad(String(tb.cycles[c]), 11) + pad(sgni(tb.cycles[c] - ta.cycles[c]), 11) + pad(String(ta.issues[c] // max(ta.waves, 1)), 12) + pad(String(tb.issues[c] // max(tb.waves, 1)), 12))
    print("")
    print("cycles by cause shift:")
    print("  " + lpad("cause", 14) + pad("cyclesA", 11) + pad("cyclesB", 11) + pad("shift", 11))
    for c in diff_causes(a, b):
        print("  " + lpad(c, 14) + pad(String(a.by_cause.get(c, 0)), 11) + pad(String(b.by_cause.get(c, 0)), 11) + pad(sgni(shift_of(a, b, c)), 11))
    print("")
    print("total cycles", a.grand, "->", b.grand, "(" + sgni(b.grand - a.grand) + ", " + pct10(b.grand - a.grand, a.grand) + ")")
    print("(averages are charged: a wait carries the cycles it held)")


def diff_json(ref a: Analysis, ref b: Analysis, top: Int = 0) raises:
    """Print the diff as JSON: one object per row plus the totals, stable key order."""
    var rows = cut_rows(diff_pairs(a, b), top)
    var cnt = diff_counts(diff_pairs(a, b))
    print("{")
    print("  \"a\": " + jstr(a.dir) + ",")
    print("  \"b\": " + jstr(b.dir) + ",")
    print("  \"grand\": {\"a\": " + String(a.grand) + ", \"b\": " + String(b.grand) + "},")
    print("  \"aligned\": " + String(cnt[0]) + ", \"inserted\": " + String(cnt[1]) + ", \"removed\": " + String(cnt[2]) + ",")
    print("  \"rows\": [")
    for i in range(len(rows)):
        var d = rows[i]
        var av = "null" if d.av < 0 else fmt1x10(d.av)
        var bv = "null" if d.bv < 0 else fmt1x10(d.bv)
        var c = "," if i < len(rows) - 1 else ""
        print("    {\"mark\": " + jstr(d.mark) + ", \"pc\": " + String(d.pc) + ", \"avgA\": " + av + ", \"avgB\": " + bv + ", \"delta\": " + fmt1x10(d.delta) + ", \"dcycles\": " + String(d.cb - d.ca) + ", \"cause\": " + jstr(d.cause) + ", \"text\": " + jstr(d.text) + "}" + c)
    print("  ],")
    print("  \"causes\": {")
    var causes = diff_causes(a, b)
    for i in range(len(causes)):
        var c = causes[i]
        var cm = "," if i < len(causes) - 1 else ""
        print("    " + jstr(c) + ": {\"a\": " + String(a.by_cause.get(c, 0)) + ", \"b\": " + String(b.by_cause.get(c, 0)) + ", \"shift\": " + String(shift_of(a, b, c)) + "}" + cm)
    print("  },")
    var ta = ClassTotals(a)
    var tb = ClassTotals(b)
    var names = class_names()
    print("  \"classes\": {")
    for c in range(N_CLASSES):
        print("    " + jstr(names[c]) + ": {\"a\": " + String(ta.cycles[c]) + ", \"b\": " + String(tb.cycles[c]) + ", \"shift\": " + String(tb.cycles[c] - ta.cycles[c]) + ", \"per_wave_a\": " + String(ta.issues[c] // max(ta.waves, 1)) + ", \"per_wave_b\": " + String(tb.issues[c] // max(tb.waves, 1)) + "}" + ("," if c < N_CLASSES - 1 else ""))
    print("  }")
    print("}")

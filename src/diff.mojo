# The diff of two captures: align the rows by instruction text, show what moved.
# The alignment is the longest common subsequence of the two text sequences, so a row that
# exists in one capture only shows up as - (removed) or + (inserted).
# Only the row record is needed here, so this module never imports the analyzer.
from std.collections import List
from row import Analysis, fmt1x10, jstr, lpad, pad, pcs_of, sgn10, sgni, tenths

comptime ALIGN_CELLS = 4000000  # a bigger pair falls back to a positional alignment


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

    def __init__(out self):
        self.mark = " "
        self.pc = -1
        self.av = -1
        self.bv = -1
        self.delta = 0
        self.cause = ""
        self.text = ""


def align(a: List[String], b: List[String]) -> List[Int]:
    """Align two key sequences. Two entries per aligned pair: the index in a and the index in b
    (-1 when that side has no match). A row of a with no match is removed (-), a row of b with
    no match is inserted (+)."""
    var n = len(a)
    var m = len(b)
    var out = List[Int]()
    if (n + 1) * (m + 1) > ALIGN_CELLS:
        # too big for the table: align what sits at the same place, the rest is removed/inserted
        for i in range(max(n, m)):
            out.append(i if i < n else -1)
            out.append(i if i < n and i < m and a[i] == b[i] else -1)
        return out^
    var w = m + 1
    var dp = List[Int32]()
    for _ in range((n + 1) * w):
        dp.append(Int32(0))
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if a[i] == b[j]:
                dp[i * w + j] = dp[(i + 1) * w + j + 1] + Int32(1)
            else:
                dp[i * w + j] = max(dp[(i + 1) * w + j], dp[i * w + j + 1])
    var i = 0
    var j = 0
    while i < n or j < m:
        if i < n and j < m and a[i] == b[j]:
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
            d.av = tenths(ra.csum, ra.waves)
            d.cause = ra.cause
            d.text = ra.text
        if y >= 0:
            var rb = b.rows[b.idx[pb[y]]]
            if d.pc < 0:
                d.pc = rb.pc
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


def diff_rows(ref a: Analysis, ref b: Analysis) raises:
    """Print the aligned diff of two captures: per-row cycle delta and the per-cause shift."""
    var rows = diff_pairs(a, b)
    var cnt = diff_counts(rows)
    print("diff", a.dir, "->", b.dir)
    print(" " + pad("pc", 6) + " " + pad("avgA", 8) + " " + pad("avgB", 8) + " " + pad("delta", 8) + "  " + lpad("cause", 14) + " instruction")
    for d in rows:
        var sa = "-" if d.av < 0 else fmt1x10(d.av)
        var sb = "-" if d.bv < 0 else fmt1x10(d.bv)
        var sd = sgn10(d.delta)
        if d.mark == "-":
            sd = "-" + sa
        elif d.mark == "+":
            sd = "+" + sb
        print(d.mark + pad(hex(d.pc), 6) + " " + pad(sa, 8) + " " + pad(sb, 8) + " " + pad(sd, 8) + "  " + lpad(d.cause, 14) + " " + d.text)
    print("")
    print(String(cnt[0]), "rows aligned,", String(cnt[1]), "inserted,", String(cnt[2]), "removed")
    print("")
    print("cycles by cause shift:")
    for c in diff_causes(a, b):
        print("  " + lpad(c, 14) + " " + sgni(shift_of(a, b, c)))
    print("(averages are charged: a wait carries the cycles it held)")


def diff_json(ref a: Analysis, ref b: Analysis) raises:
    """Print the diff as JSON: one object per row plus the totals, stable key order."""
    var rows = diff_pairs(a, b)
    var cnt = diff_counts(rows)
    print("{")
    print("  \"a\": " + jstr(a.dir) + ",")
    print("  \"b\": " + jstr(b.dir) + ",")
    print("  \"aligned\": " + String(cnt[0]) + ", \"inserted\": " + String(cnt[1]) + ", \"removed\": " + String(cnt[2]) + ",")
    print("  \"rows\": [")
    for i in range(len(rows)):
        var d = rows[i]
        var av = "null" if d.av < 0 else fmt1x10(d.av)
        var bv = "null" if d.bv < 0 else fmt1x10(d.bv)
        var c = "," if i < len(rows) - 1 else ""
        print("    {\"mark\": " + jstr(d.mark) + ", \"pc\": " + String(d.pc) + ", \"avgA\": " + av + ", \"avgB\": " + bv + ", \"delta\": " + fmt1x10(d.delta) + ", \"cause\": " + jstr(d.cause) + ", \"text\": " + jstr(d.text) + "}" + c)
    print("  ],")
    print("  \"causes\": {")
    var causes = diff_causes(a, b)
    for i in range(len(causes)):
        var c = causes[i]
        var cm = "," if i < len(causes) - 1 else ""
        print("    " + jstr(c) + ": {\"a\": " + String(a.by_cause.get(c, 0)) + ", \"b\": " + String(b.by_cause.get(c, 0)) + ", \"shift\": " + String(shift_of(a, b, c)) + "}" + cm)
    print("  }")
    print("}")

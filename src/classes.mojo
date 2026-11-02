# The cost of a capture rolled up per instruction class (wavewait DIR --by-class, the "classes" key of --json).
# The class is read off the instruction text, so it needs nothing the row record does not hold.
# Cycles are the charged ones (waits on their s_waitcnt), so the classes sum to the table total.
from std.collections import List
from row import Analysis, fmt1x10, jstr, lpad, pad, tenths

comptime N_CLASSES = 14


def class_names() -> List[String]:
    var n = List[String]()
    n.append("vmem load")
    n.append("vmem store")
    n.append("smem")
    n.append("lds")
    n.append("waitcnt vmcnt")
    n.append("waitcnt lgkmcnt")
    n.append("waitcnt expcnt")
    n.append("waitcnt vscnt")
    n.append("waitcnt depctr")
    n.append("waitcnt mixed")
    n.append("valu")
    n.append("salu")
    n.append("branch")
    n.append("other")
    return n^


def has_any(m: String, prefixes: List[String]) -> Bool:
    for p in prefixes:
        if m.startswith(p):
            return True
    return False


def class_of(text: String) -> Int:
    """The class index of an instruction text (index into class_names())."""
    var sp = text.find(" ")
    var m = String(text[byte=0:sp]) if sp >= 0 else text
    var mem = List[String]()
    if m.startswith("s_waitcnt"):
        if m == "s_waitcnt_vscnt":
            return 7
        if m == "s_waitcnt_depctr":
            return 8
        var vm = m == "s_waitcnt_vmcnt" or text.find("vmcnt(") >= 0
        var lg = m == "s_waitcnt_lgkmcnt" or text.find("lgkmcnt(") >= 0
        var ex = m == "s_waitcnt_expcnt" or text.find("expcnt(") >= 0
        if Int(vm) + Int(lg) + Int(ex) != 1:
            return 9
        return 4 if vm else (5 if lg else 6)
    if m.startswith("s_load") or m.startswith("s_buffer_load"):
        return 2
    if m == "s_branch" or m.startswith("s_cbranch"):
        return 12
    if m.startswith("s_"):
        return 11
    if m.startswith("v_"):
        return 10
    if m.startswith("ds_"):
        return 3
    mem.append("global_load")
    mem.append("buffer_load")
    mem.append("tbuffer_load")
    mem.append("flat_load")
    mem.append("scratch_load")
    if has_any(m, mem):
        return 0
    var st = List[String]()
    st.append("global_store")
    st.append("buffer_store")
    st.append("tbuffer_store")
    st.append("flat_store")
    st.append("scratch_store")
    st.append("global_atomic")
    st.append("buffer_atomic")
    st.append("flat_atomic")
    if has_any(m, st):
        return 1
    return 13


struct ClassTotals(Movable):
    var cycles: List[Int]  # charged cycles per class
    var issues: List[Int]  # instructions issued per class, summed over the waves
    var waves: Int
    var grand: Int

    def __init__(out self, ref an: Analysis):
        self.cycles = List[Int]()
        self.issues = List[Int]()
        for _ in range(N_CLASSES):
            self.cycles.append(0)
            self.issues.append(0)
        self.waves = an.waves
        self.grand = an.grand
        for r in an.rows:
            var c = class_of(r.text)
            self.cycles[c] += r.csum
            self.issues[c] += r.waves


def print_classes(ref an: Analysis) raises:
    """One row per class: charged cycles, share of the total, instructions issued per wave."""
    var t = ClassTotals(an)
    var names = class_names()
    print(lpad("class", 17) + pad("cycles", 11) + pad("share", 7) + pad("issues", 11) + pad("per wave", 11))
    var sum = 0
    for i in range(N_CLASSES):
        sum += t.cycles[i]
        print(lpad(names[i], 17) + pad(String(t.cycles[i]), 11) + pad(fmt1x10(tenths(t.cycles[i] * 100, t.grand)) + "%", 7) + pad(String(t.issues[i]), 11) + pad(String(t.issues[i] // max(t.waves, 1)), 11))
    print("")
    print(String(sum), "cycles in", String(N_CLASSES), "classes (the table total is", String(t.grand) + ");", String(t.waves), "waves")


def json_classes(ref an: Analysis) -> String:
    """The "classes" object of the --json output, one line per class, stable order."""
    var t = ClassTotals(an)
    var names = class_names()
    var s = String("  \"classes\": {\n")
    for i in range(N_CLASSES):
        s += "    " + jstr(names[i]) + ": {\"cycles\": " + String(t.cycles[i]) + ", \"share\": " + fmt1x10(tenths(t.cycles[i] * 100, t.grand)) + ", \"issues\": " + String(t.issues[i]) + ", \"per_wave\": " + String(t.issues[i] // max(t.waves, 1)) + "}" + ("," if i < N_CLASSES - 1 else "") + "\n"
    s += "  }"
    return s

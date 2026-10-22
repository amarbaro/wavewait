# The optimisation hints of a capture: the rules table (data) applied in one loop over the row
# record. Every rule names the penalty it costs, taken from the rdna3-sqtt-emu timing table
# (the VALU/LDS/memory rows).
from std.collections import Dict, List
from row import Analysis, Row, fmt1x10, lpad, pad, tenths


struct Rule(ImplicitlyCopyable):
    var name: String
    var penalty: String
    var min_cycles: Int  # the exposure below which the rule stays silent (0: always)

    def __init__(out self, name: String, penalty: String, min_cycles: Int = 0):
        self.name = name
        self.penalty = penalty
        self.min_cycles = min_cycles


struct Hint(ImplicitlyCopyable):
    var pc: Int
    var rule: String
    var cycles: Int  # the tenths of the charged average of the row the hint points at
    var penalty: String
    var text: String

    def __init__(out self, pc: Int, rule: String, cycles: Int, penalty: String, text: String):
        self.pc = pc
        self.rule = rule
        self.cycles = cycles
        self.penalty = penalty
        self.text = text


def rules() -> List[Rule]:
    """The rules table: the name, the measured penalty and the exposure each rule needs.

    The penalties are the measured numbers of the rdna3-sqtt-emu timing table: the lgkmcnt ready
    latency (+28 for a load b32, +30 for a store b32), the scalar op latency (exec = issue + 4),
    the VOPD slot (two 32-bit VALU ops) and the LDS banks (32 of 32 bit, one wave-wide load per
    cycle, so a bank conflict adds a cycle per conflict group). The hoist rule stays silent below
    56 cycles: twice the lgkmcnt ready latency, under it the load latency was hidden."""
    return [
        Rule("hoist", "+28 (lgkmcnt ready +28) up to the memory latency (vmcnt)", 56),
        Rule("idle wait", "+4 (a scalar op: exec = issue + 4)"),
        Rule("vopd", "1 cycle per pair (a VOPD slot carries two 32-bit VALU ops)"),
        Rule("lds bank", "+1 cycle per conflict group (32 banks of 32 bit)"),
    ]


def operands(t: String) -> List[String]:
    """The operands of an instruction text (the opcode split off the first one)."""
    var out = List[String]()
    var sp = t.find(" ")
    var rest = t if sp < 0 else String(t[byte=sp + 1 : t.byte_length()])
    for p in rest.split(","):
        out.append(String(p.strip()))
    return out^


def is_num(s: String) -> Bool:
    """Is the text a decimal number?"""
    if s.byte_length() == 0:
        return False
    for c in s:
        if c < "0" or c > "9":
            return False
    return True


def reads(t: String, reg: String) -> Bool:
    """Does the instruction text read the register (as a source, not as its destination)?"""
    var ops = operands(t)
    for i in range(1, len(ops)):
        if ops[i] == reg:
            return True
    return False


def is_load(t: String) -> Bool:
    """A memory load (the global, the LDS or the scalar cache)."""
    return t.startswith("global_load") or t.startswith("buffer_load") or t.startswith("ds_load") or t.startswith("s_load")


def is_lds(t: String) -> Bool:
    """An LDS access."""
    return t.startswith("ds_load") or t.startswith("ds_store")


def is_vmem(t: String) -> Bool:
    """A global memory op (the vmcnt counter)."""
    return t.startswith("global_load") or t.startswith("global_store") or t.startswith("buffer_")


def is_valu32(t: String) -> Bool:
    """A 32 bit VALU op (the ones a VOPD slot can carry two of)."""
    if not t.startswith("v_") or t.startswith("v_pk_"):
        return False
    if t.find("_b64") >= 0 or t.find("_u64") >= 0 or t.find("_i64") >= 0 or t.find("_f64") >= 0 or t.find("_e64") >= 0:
        return False
    return t.find("_b32") >= 0 or t.find("_e32") >= 0 or t.find("_u32") >= 0 or t.find("_i32") >= 0 or t.find("_f32") >= 0


def is_carry(t: String) -> Bool:
    """A carry op (the next one reads vcc: the pair cannot be packed)."""
    return t.find("_co_") >= 0


def is_shift32(t: String) -> Bool:
    """A 32 bit shift left (it scales the lane stride of the address it feeds)."""
    return t.startswith("v_lshlrev") and (t.find("_b32") >= 0 or t.find("_e32") >= 0)


def wait_value(t: String, cnt: String) raises -> Int:
    """The value of the counter in a wait (999 when the wait does not name it)."""
    var at = t.find(cnt + "(")
    if at < 0:
        return 999
    var close = t.find(")", at)
    var num = String(t[byte=at + cnt.byte_length() + 1 : close])
    return atol(num, 10) if is_num(num) else 999


def idle_wait(t: String, vmem: Int, lgkm: Int) -> Bool:
    """Does the wait have nothing to wait for?"""
    var waited = False
    if t.find("vmcnt(") >= 0:
        waited = True
        if vmem > 0:
            return False
    if t.find("lgkmcnt(") >= 0:
        waited = True
        if lgkm > 0:
            return False
    return waited


def trim(lst: List[Int], n: Int) -> List[Int]:
    """The last n entries of the list (a wait leaves at most n ops in flight)."""
    var out = List[Int]()
    var start = 0
    if len(lst) > n:
        start = len(lst) - n
    for i in range(start, len(lst)):
        out.append(lst[i])
    return out^


def pc_order(ref an: Analysis) -> List[Int]:
    """The row indices by pc (the static program order the rules scan).

    The rows are in the order the trace showed them first; a capture with more than one blob
    (the overflow buffer of the last wave) can put a row out of its program place, so the rules
    scan the pc order, the same order the table prints."""
    var idx = List[Int]()
    for i in range(len(an.rows)):
        idx.append(i)
    for i in range(1, len(idx)):
        var v = idx[i]
        var j = i
        while j > 0 and an.rows[idx[j - 1]].pc > an.rows[idx[j]].pc:
            idx[j] = idx[j - 1]
            j -= 1
        idx[j] = v
    return idx^


def hints(ref an: Analysis) raises -> List[Hint]:
    """The optimisation hints of the capture: the rules table applied in one loop over the rows."""
    var out = List[Hint]()
    var rl = rules()
    var stride = Dict[String, Int]()  # register -> the byte stride of the last shift on it
    var vmem = List[Int]()  # the row indices of the global memory ops in flight
    var lgkm = List[Int]()  # the row indices of the LDS and the scalar memory ops in flight
    var order = pc_order(an)
    for oi in range(len(order)):
        var i = order[oi]
        var r = an.rows[i]
        var t = r.text
        var ops = operands(t)
        var nxt = "" if oi + 1 >= len(order) else an.rows[order[oi + 1]].text
        # vopd: two consecutive 32 bit VALU ops with no dependency between them
        if nxt != "" and is_valu32(t) and is_valu32(nxt) and not is_carry(t) and not is_carry(nxt):
            if not reads(nxt, ops[0]):
                out.append(Hint(r.pc, rl[2].name, tenths(r.csum, r.waves), rl[2].penalty, t + " | " + nxt))
        # lds bank: an LDS access whose stride puts several lanes in one bank
        if is_lds(t) and len(ops) > 1:
            var base = ops[1] if t.startswith("ds_load") else ops[0]
            var st = stride.get(base, 0)
            if st >= 8:
                var lanes = min(32, st // 4)
                out.append(Hint(r.pc, rl[3].name, tenths(r.csum, r.waves), rl[3].penalty, t + " (stride " + String(st) + " B: " + String(lanes) + " lanes per bank)"))
        # the waits: the exposure of the loads in flight, and the idle check
        if t.startswith("s_waitcnt") and not t.startswith("s_waitcnt_depctr"):
            var v0 = len(vmem)
            var l0 = len(lgkm)
            var exposure = tenths(r.csum, r.waves)
            if t.find("vmcnt(") >= 0:
                if v0 > 0:
                    var li = vmem[v0 - 1]
                    if exposure > rl[0].min_cycles * 10:
                        out.append(Hint(an.rows[li].pc, rl[0].name, exposure, rl[0].penalty, an.rows[li].text))
                vmem = trim(vmem, wait_value(t, "vmcnt"))
            if t.find("lgkmcnt(") >= 0:
                if l0 > 0:
                    var li = lgkm[l0 - 1]
                    if exposure > rl[0].min_cycles * 10:
                        out.append(Hint(an.rows[li].pc, rl[0].name, exposure, rl[0].penalty, an.rows[li].text))
                lgkm = trim(lgkm, wait_value(t, "lgkmcnt"))
            if idle_wait(t, v0, l0):
                out.append(Hint(r.pc, rl[1].name, tenths(r.csum, r.waves), rl[1].penalty, t))
        # the state the next rows need
        if is_vmem(t):
            vmem.append(i)
        if is_lds(t) or t.startswith("s_load"):
            lgkm.append(i)
        if is_shift32(t) and len(ops) > 1 and is_num(ops[1]):
            var k = atol(ops[1], 10)
            stride[ops[0]] = stride.get(ops[0], 1) * (1 << k)
    # the hints by pc (a stable insertion sort, so two hints on one row keep their order)
    for i in range(1, len(out)):
        var h = out[i]
        var j = i
        while j > 0 and out[j - 1].pc > h.pc:
            out[j] = out[j - 1]
            j -= 1
        out[j] = h
    return out^


def print_hints(ref an: Analysis) raises:
    """The hints of the capture: the rule, the measured cost, the penalty and the instruction."""
    var hs = hints(an)
    print("    pc  rule         cycles  " + lpad("penalty", 60) + "instruction")
    for h in hs:
        print(pad(hex(h.pc), 6) + " " + pad(h.rule, 12) + " " + pad(fmt1x10(h.cycles), 6) + "  " + lpad(h.penalty, 60) + " " + h.text)
    print("")
    print(String(len(hs)), "hints")

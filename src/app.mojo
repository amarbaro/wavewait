# The app mode: one summary row per dispatch of a rocprofv3 --att run dir. The code map of a
# dispatch is built in Mojo from its code object (llvm-objdump), so the run dir is read directly
# and tools/from-att.sh is not needed on this path.
from std.collections import Dict, List
from std.os import listdir
from std.os.path import exists, isdir
from std.sys import exit
from row import Analysis, lpad, pad
from trace import Pcmap, analyze_paths, pcmap_of


struct Dispatch(Copyable):
    var key: String  # the "agent_index" key of the dispatch
    var objfile: String  # its code object ("" when the capture names none)
    var name: String  # the kernel name
    var rows: List[Tuple[String, Int]]  # code.json rows when it names no code object
    var entry: Int  # the kernel's first pc in its code object, -1 when the capture names none
    var atts: String  # its raw SQTT files, one path per line
    var simd: Int  # the SIMD AMD's decoder found the waves on (filenames.json), -1 when unknown
    var starts: List[Int]  # and its waves, slot << 40 | start time: the trace also holds other dispatches' waves

    def __init__(out self):
        self.key = ""
        self.objfile = ""
        self.name = ""
        self.rows = List[Tuple[String, Int]]()
        self.entry = -1
        self.atts = ""
        self.simd = -1
        self.starts = List[Int]()


def att_key(name: String) -> String:
    """The "agent_index" key of a raw SQTT file: <pid>_<agent>_shader_engine_<se>_<n>.att."""
    var parts = name.split("_")
    if len(parts) < 6:
        return name
    return parts[1] + "_" + parts[5].split(".")[0]


def dir_key(name: String) -> String:
    """The "agent_index" key of a dispatch directory: ui_output_agent_<agent>_dispatch_<n>."""
    var parts = name.split("_")
    if len(parts) < 6:
        return name
    return parts[3] + "_" + parts[5]


def dispatch_n(key: String) raises -> Int:
    """The dispatch index of an "agent_index" key."""
    var parts = key.split("_")
    return atol(String(parts[len(parts) - 1]), 10)


def obj_id(name: String) raises -> Int:
    """The id of a code object file: <pid>_gfx1100_code_object_id_<k>.out."""
    var parts = name.split("_")
    if len(parts) < 2:
        return 0
    return atol(String(parts[len(parts) - 1].split(".")[0]), 10)


def json_fields(text: String, start: Int) -> List[String]:
    """The comma-separated fields of the JSON array at `start`; the commas inside strings do not split."""
    var out = List[String]()
    var depth = 0
    var in_str = False
    var esc = False
    var begin = start + 1  # past the '['
    var i = start
    while i < text.byte_length():
        var c = text[byte=i:i + 1]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == "\"":
                in_str = False
        elif c == "\"":
            in_str = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                out.append(String(text[byte=begin:i]))
                return out^
        elif c == "," and depth == 1:
            out.append(String(text[byte=begin:i]))
            begin = i + 1
        i += 1
    return out^


def code_json(path: String) raises -> Tuple[String, Int, Int]:
    """The kernel name, the code object id and the entry pc of a rocprofv3 code.json ("", 0 and -1 when it has no code).
    The entry is the label row's address: a torch code object holds hundreds of kernels, the dispatch ran only this one."""
    var text = open(path, "r").read()
    var i = text.find("\"code\":")
    if i < 0:
        return ("", 0, -1)
    i += 7
    while i < text.byte_length() and text[byte=i:i + 1] == " ":
        i += 1
    if text[byte=i:i + 4] == "null":
        return ("", 0, -1)
    var arr = text.find("[", i)
    if arr < 0:
        return ("", 0, -1)
    var row = text.find("[", arr + 1)  # the first instruction row of the array
    if row < 0:
        return ("", 0, -1)
    var f = json_fields(text, row)
    if len(f) < 5:
        return ("", 0, -1)
    var name = String(String(f[0]).strip())
    if name.startswith("\""):
        var quoted = String(name[byte=1:name.byte_length() - 1])
        name = quoted
    var entry = -1
    if name.startswith("; "):  # the label row; without it the first row is an instruction, not a name
        var label = String(name[byte=2:])
        name = label
        if len(f) > 5:
            entry = atol(String(String(f[5]).strip()), 10)
    else:
        name = ""
    return (name, atol(String(String(f[4]).strip()), 10), entry)


def dispatches(rundir: String) raises -> List[Dispatch]:
    """The dispatches of a rocprofv3 --att run dir: the raw SQTT files and the code object of each."""
    var atts = Dict[String, String]()  # the "agent_index" key -> the raw SQTT files
    var dirs = Dict[String, String]()  # the key -> the dispatch directory
    var objs = Dict[Int, String]()  # the code object id -> its file
    var order = List[String]()
    var scan: List[String] = [rundir]  # the traces sit in a host subdir, or in the run dir itself (rocprofv3 -o NAME)
    for entry in listdir(rundir):
        var path = rundir + "/" + String(entry)
        if entry.find("ui_output_agent_") == 0:
            dirs[dir_key(entry)] = path
        elif isdir(path):
            scan.append(path)
    for path in scan:
        for f in listdir(path):
            var name = String(f)
            if name.endswith(".att"):
                var key = att_key(name)
                if key not in atts:
                    atts[key] = ""
                    order.append(key)
                atts[key] = atts[key] + path + "/" + name + "\n"
            elif name.endswith(".out") and name.find("_code_object_id_") >= 0:
                objs[obj_id(name)] = path + "/" + name
    for i in range(1, len(order)):  # numeric dispatch order: the code objects load in that order
        var j = i
        while j > 0 and dispatch_n(order[j]) < dispatch_n(order[j - 1]):
            order.swap_elements(j, j - 1)
            j -= 1
    var ds = List[Dispatch]()
    var claimed = Dict[Int, Bool]()
    for key in order:
        var d = Dispatch()
        d.key = key
        var dirpath = dirs.get(key, "")
        if dirpath != "" and exists(dirpath + "/code.json"):
            var info = code_json(dirpath + "/code.json")
            d.name = info[0]
            d.entry = info[2]
            var cid = info[1]
            if cid != 0 and cid in objs:
                d.objfile = objs[cid]
                claimed[cid] = True
            elif cid == 0:
                d.rows = code_rows(dirpath + "/code.json")
        d.atts = atts.get(key, "")
        if dirpath != "" and exists(dirpath + "/filenames.json"):  # "wave_filenames":{"<se>":{"<simd>":{...
            var t = open(dirpath + "/filenames.json", "r").read()
            var at = t.find("\"wave_filenames\":{\"")
            if at >= 0:
                var q = t.find("\":{\"", at + 19)
                var e = t.find("\"", q + 4)
                if q > 0 and e > q:
                    d.simd = atol(t[byte=q + 4 : e])
            var sl = t.find("_sl")  # ..._sl<slot>_wv0.json",<begin>,<end>]
            while sl >= 0:
                var u = t.find("_", sl + 3)
                var c = t.find(",", u)
                var c2 = t.find(",", c + 1)
                d.starts.append(atol(t[byte=sl + 3 : u]) << 40 | atol(t[byte=c + 1 : c2]))
                sl = t.find("_sl", c2)
        ds.append(d^)
    for i in range(len(ds)):
        if len(ds[i].rows):
            resolve(ds[i], objs, claimed)
    # the code objects no dispatch claimed belong to the dispatches whose capture has no code
    # (the runtime's own kernels): hand them out in dispatch order
    var free = List[Int]()
    for id in objs.keys():
        if id not in claimed:
            free.append(id)
    sort(free)
    var k = 0
    for i in range(len(ds)):
        if ds[i].objfile == "" and not len(ds[i].rows) and k < len(free):
            ds[i].objfile = objs[free[k]]
            k += 1
    return ds^


def code_rows(path: String) raises -> List[Tuple[String, Int]]:
    """The instruction rows of a code.json with no code object id: (text, vaddr). Empty when it has an id or a label."""
    var rows = List[Tuple[String, Int]]()
    var text = open(path, "r").read()
    var i = text.find("\"code\":")
    if i < 0:
        return rows^
    var at = text.find("[", i)
    if at < 0 or String(String(text[byte=i + 7:at]).strip()).byte_length() != 0:
        return rows^
    var end = text.find("]]", at)  # the array's close; a row starts with a quote, the "[" of s[0:1] does not
    at = text.find("[\"", at + 1)
    while at >= 0 and at < end:
        var f = json_fields(text, at)
        if len(f) < 6 or atol(String(String(f[4]).strip()), 10) != 0 or String(String(f[0]).strip()).startswith("\"; "):
            return List[Tuple[String, Int]]()
        var q = String(String(f[0]).strip())
        rows.append((String(q[byte=1:q.byte_length() - 1]), atol(String(String(f[5]).strip()), 10)))
        at = text.find("[\"", at + 1)
    return rows^


def holds(ref pm: Pcmap, ref rows: List[Tuple[String, Int]]) raises -> Bool:
    """The rows sit in this object at the same page offsets, with the same text."""
    for pc in pm.text.keys():
        if (pc ^ rows[0][1]) & 4095:
            continue
        var hit = True
        for r in rows:
            var at = pc + r[1] - rows[0][1]
            if at not in pm.text or pm.text[at] != r[0]:
                hit = False
                break
        if hit:
            return True
    return False


def resolve(mut d: Dispatch, ref objs: Dict[Int, String], mut claimed: Dict[Int, Bool]) raises:
    """A dispatch whose code.json has rows but no object id: the object holding those rows whose walk stays aligned.
    Several left are the same code here, so the name says so instead of guessing."""
    var fits = List[Int]()
    for id in sorted_keys(objs):
        d.objfile = objs[id]
        var pm = pcmap_of(d.objfile)
        if pm.first < 0 or not holds(pm, d.rows):
            continue
        var atts = d.atts
        var an = analyze_dispatch(d, atts)
        if an.waves and not an.mismatches:
            fits.append(id)
    d.objfile = ""
    if len(fits):
        d.objfile = objs[fits[0]]
        claimed[fits[0]] = True
        if len(fits) > 1:
            d.name = pcmap_of(d.objfile).symbol + "(1of" + String(len(fits)) + ")"


def sorted_keys(ref objs: Dict[Int, String]) -> List[Int]:
    var ks = List[Int]()
    for id in objs.keys():
        ks.append(id)
    sort(ks)
    return ks^


def analyze_dispatch(ref d: Dispatch, atts: String) raises -> Analysis:
    """The analysis of one dispatch: its code map (built from its code object) and its raw SQTT."""
    if d.objfile == "":
        return Analysis()
    var pm = pcmap_of(d.objfile)
    if pm.first < 0:
        return Analysis()
    var label = d.name if d.name != "" else pm.symbol
    var paths = List[String]()
    for p in atts.split("\n"):
        if p:
            paths.append(String(p))
    var an = analyze_paths(paths^, pm^, label, 0, "", False, d.simd, d.starts, d.entry)
    return an^


def top_cause(ref an: Analysis) raises -> String:
    """The cause with the most charged cycles, or "-" when the dispatch has no instructions."""
    var best = ""
    var top = 0
    for c in an.by_cause.keys():
        if an.by_cause[c] > top:
            top = an.by_cause[c]
            best = c
    return best if best != "" else "-"


def app_one(args: List[String], tl_cap: Int, want_lines: Bool) raises -> Analysis:
    """The full analysis of ONE dispatch of the run dir args[2], picked by --dispatch N / --kernel NAME."""
    var want_n = -1
    var want_k = ""
    for i in range(len(args) - 1):
        if args[i] == "--dispatch":
            want_n = atol(args[i + 1], 10)
        elif args[i] == "--kernel":
            want_k = args[i + 1]
    return app_pick(args[2], want_n, want_k, tl_cap, want_lines)


def app_pick(rundir: String, want_n: Int, want_k: String, tl_cap: Int, want_lines: Bool) raises -> Analysis:
    """The full analysis of ONE dispatch of a run dir, picked by its index (want_n, -1 = any) or a kernel-name
    prefix (want_k, "" = any) that must pick exactly one dispatch, walked against its own code object."""
    var picked = List[Int]()
    var ds = dispatches(rundir)
    for i in range(len(ds)):
        if want_n >= 0 and dispatch_n(ds[i].key) != want_n:
            continue
        if want_k != "":
            var nm = ds[i].name
            if nm == "" and ds[i].objfile != "":
                nm = pcmap_of(ds[i].objfile).symbol
            if not nm.startswith(want_k):
                continue
        picked.append(i)
    if len(picked) != 1:
        print("FAIL app: " + String(len(picked)) + " dispatches match (want exactly 1; add --dispatch N)")
        exit(1)
    ref d = ds[picked[0]]
    if d.objfile == "":
        print("FAIL app: dispatch " + d.key + " has no code object in " + rundir)
        exit(1)
    var pm = pcmap_of(d.objfile)
    var label = d.name if d.name != "" else pm.symbol
    var paths = List[String]()
    for p in d.atts.split("\n"):
        if p:
            paths.append(String(p))
    var an = analyze_paths(paths^, pm^, label, tl_cap, d.objfile if want_lines else "", True, d.simd, d.starts, d.entry)
    an.codeobj = d.objfile
    return an^


def refuse(ref an: Analysis, args: List[String]):
    """A desynced walk prints plausible, wrong causes: no result from it unless --allow-mism asks for one."""
    if an.mismatches and "--allow-mism" not in args:
        print("FAIL walk:", an.mismatches, "packets matched no instruction in", an.dir + ": the numbers are not trusted (--allow-mism shows them)")
        if an.first_mismatch != "":
            print("FAIL first: " + an.first_mismatch)
        exit(3)


def app_main(args: List[String]) raises:
    """One summary row per dispatch of a rocprofv3 --att run dir (wavewait app RUNDIR)."""
    if len(args) < 3:
        print("usage: wavewait app RUNDIR")
        exit(1)
    var rundir = args[2]
    var ds = dispatches(rundir)
    if not len(ds):
        print("FAIL app: no dispatch in", rundir)
        exit(1)
    print(lpad("kernel", 32) + pad("cycles", 9) + pad("waves", 7) + "  top cause" + "        open  mism  dispatch")
    var total = 0
    var bad = 0
    for d in ds:
        var an = analyze_dispatch(d, d.atts)
        total += an.grand
        bad += 1 if an.mismatches else 0
        var name = an.dir if an.dir != "" else "?"
        var tc = top_cause(an)
        print(lpad(name, 32) + pad(String(an.grand), 9) + pad(String(an.waves), 7) + "  " + tc + " " * max(1, 16 - tc.byte_length()) + pad(String(an.open_waves), 4) + pad(String(an.mismatches), 6) + "  " + String(dispatch_n(d.key)))
    print("")
    print(String(len(ds)), "dispatches,", String(total), "cycles")
    if bad and "--allow-mism" not in args:
        print("FAIL walk:", bad, "dispatches with mism > 0: their rows are not trusted (--allow-mism accepts them)")
        exit(3)

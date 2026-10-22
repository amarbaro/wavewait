# The app mode: one summary row per dispatch of a rocprofv3 --att run dir. The code map of a
# dispatch is built in Mojo from its code object (llvm-objdump), so the run dir is read directly
# and tools/from-att.sh is not needed on this path.
from std.collections import Dict, List
from std.os import listdir
from std.os.path import exists, isdir
from std.sys import exit
from row import Analysis, lpad, pad
from trace import Pcmap, analyze_paths, pcmap_of


struct Dispatch(ImplicitlyCopyable):
    var key: String  # the "agent_index" key of the dispatch
    var objfile: String  # its code object ("" when the capture names none)
    var name: String  # the kernel name
    var atts: String  # its raw SQTT files, one path per line

    def __init__(out self):
        self.key = ""
        self.objfile = ""
        self.name = ""
        self.atts = ""


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


def code_json(path: String) raises -> Tuple[String, Int]:
    """The kernel name and the code object id of a rocprofv3 code.json ("" and 0 when it has no code)."""
    var text = open(path, "r").read()
    var i = text.find("\"code\":")
    if i < 0:
        return ("", 0)
    i += 7
    while i < text.byte_length() and text[byte=i:i + 1] == " ":
        i += 1
    if text[byte=i:i + 4] == "null":
        return ("", 0)
    var arr = text.find("[", i)
    if arr < 0:
        return ("", 0)
    var row = text.find("[", arr + 1)  # the first instruction row of the array
    if row < 0:
        return ("", 0)
    var f = json_fields(text, row)
    if len(f) < 5:
        return ("", 0)
    var name = String(String(f[0]).strip())
    if name.startswith("\""):
        var quoted = String(name[byte=1:name.byte_length() - 1])
        name = quoted
    if name.startswith("; "):  # the label row; without it the first row is an instruction, not a name
        var label = String(name[byte=2:])
        name = label
    else:
        name = ""
    return (name, atol(String(String(f[4]).strip()), 10))


def dispatches(rundir: String) raises -> List[Dispatch]:
    """The dispatches of a rocprofv3 --att run dir: the raw SQTT files and the code object of each."""
    var atts = Dict[String, String]()  # the "agent_index" key -> the raw SQTT files
    var dirs = Dict[String, String]()  # the key -> the dispatch directory
    var objs = Dict[Int, String]()  # the code object id -> its file
    var order = List[String]()
    for entry in listdir(rundir):
        var path = rundir + "/" + String(entry)
        if entry.find("ui_output_agent_") == 0:
            dirs[dir_key(entry)] = path
            continue
        if not isdir(path):
            continue
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
            var cid = info[1]
            if cid != 0 and cid in objs:
                d.objfile = objs[cid]
                claimed[cid] = True
        d.atts = atts.get(key, "")
        ds.append(d)
    # the code objects no dispatch claimed belong to the dispatches whose capture has no code
    # (the runtime's own kernels): hand them out in dispatch order
    var free = List[Int]()
    for id in objs.keys():
        if id not in claimed:
            free.append(id)
    sort(free)
    var k = 0
    for i in range(len(ds)):
        if ds[i].objfile == "" and k < len(free):
            ds[i].objfile = objs[free[k]]
            k += 1
    return ds^


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
    var an = analyze_paths(paths^, pm^, label, 0, "", False)
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
    print(lpad("kernel", 32) + pad("cycles", 9) + pad("waves", 7) + "  top cause")
    var total = 0
    for d in ds:
        var an = analyze_dispatch(d, d.atts)
        total += an.grand
        var name = an.dir if an.dir != "" else "?"
        print(lpad(name, 32) + pad(String(an.grand), 9) + pad(String(an.waves), 7) + "  " + top_cause(an))
    print("")
    print(String(len(ds)), "dispatches,", String(total), "cycles")

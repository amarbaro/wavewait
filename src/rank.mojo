# The rank mode: where a whole process spent its GPU time, kernel by kernel, from rocprofv3's *_kernel_trace.csv
# (--kernel-trace --output-format csv). It covers every dispatch on every CU, so it also shows what ATT cannot:
# the kernels that miss the traced CU, and the iteration to give --kernel-iteration-range for a warm capture.
# --stages FILE (tools/comfy_stages.py) keeps the job's dispatches and adds one row per node: wall, GPU busy, idle, top kernel.
from std.collections import Dict, List
from std.os import listdir
from std.os.path import isdir
from std.sys import exit
from row import fmt1, pad


struct Kernel(Copyable):
    var name: String
    var ns: Int  # summed dispatch time
    var n: Int

    def __init__(out self, name: String):
        self.name = name
        self.ns = 0
        self.n = 0


def csv_fields(line: String) -> List[String]:
    """The fields of one CSV line, quotes removed; the commas inside a quoted C++ name do not split."""
    var out = List[String]()
    var b = line.as_bytes()
    var start = 0
    var quoted = False
    for i in range(len(b)):
        if Int(b[i]) == ord('"'):
            quoted = not quoted
        elif Int(b[i]) == ord(",") and not quoted:
            out.append(String(line[byte=start:i]))
            start = i + 1
    out.append(String(line[byte=start:]))
    for i in range(len(out)):
        if out[i].startswith('"'):
            var bare = String(out[i][byte=1 : out[i].byte_length() - 1])
            out[i] = bare
    return out^


def trace_files(rundir: String) raises -> List[String]:
    """The *_kernel_trace.csv of a run dir: in it or in its host subdir, one per traced process."""
    if not isdir(rundir):
        return List[String]()
    var scan: List[String] = [rundir]
    for e in listdir(rundir):
        if isdir(rundir + "/" + String(e)):
            scan.append(rundir + "/" + String(e))
    var out = List[String]()
    for d in scan:
        for f in listdir(d):
            if String(f).endswith("_kernel_trace.csv"):
                out.append(d + "/" + String(f))
    return out^


struct Stage(Copyable):
    var label: String
    var start: Int
    var end: Int
    var busy: Int  # ns with at least one kernel running
    var ns: Dict[Int, Int]  # kernel index -> its time here

    def __init__(out self, label: String, start: Int):
        self.label = label
        self.start = start
        self.end = start
        self.busy = 0
        self.ns = Dict[Int, Int]()


def read_stages(path: String) raises -> List[Stage]:
    """One stage per `executing` event of a comfy_stages.py file, up to the next one or the end of the job."""
    var out = List[Stage]()
    for line in open(path, "r").read().split("\n"):
        var f = String(line).split("\t")
        if len(f) < 4:
            continue
        var t = atol(f[0])
        if f[1] == "execution_start":
            out.append(Stage("(queue)", t))
        elif f[1] == "executing" or f[1] == "execution_success" or f[1] == "execution_error":
            if out:
                out[len(out) - 1].end = t
            if f[1] == "executing" and f[2]:
                out.append(Stage(String(f[3]) + " " + String(f[2]), t))
    if out and out[len(out) - 1].end == out[len(out) - 1].start:
        _ = out.pop()
    return out^


def add_busy(mut stages: List[Stage], lo: Int, hi: Int):
    """Credit the busy interval [lo, hi) to the stages it overlaps; both sorted by time, stages in order."""
    for i in range(len(stages)):
        if stages[i].start >= hi:
            break
        var a = max(lo, stages[i].start)
        var b = min(hi, stages[i].end)
        if b > a:
            stages[i].busy += b - a


def rank_main(args: List[String]) raises:
    """wavewait rank RUNDIR [--top N] [--stages FILE]: kernels by summed time, costliest first, and the GPU's idle share."""
    if len(args) < 3:
        print("usage: wavewait rank RUNDIR [--top N] [--stages FILE]")
        exit(1)
    var top = 20
    var stages = List[Stage]()
    for i in range(3, len(args) - 1):
        if args[i] == "--top":
            top = atol(args[i + 1])
        elif args[i] == "--stages":
            stages = read_stages(args[i + 1])
            if not stages:
                print("FAIL rank: no stages in " + String(args[i + 1]))
                exit(1)
    var job_start = stages[0].start if stages else 0
    var job_end = stages[len(stages) - 1].end if stages else (1 << 62)
    var files = trace_files(String(args[2]))
    if not files:
        print("FAIL rank: no *_kernel_trace.csv under " + String(args[2]) + " (capture with --kernel-trace --output-format csv)")
        exit(1)
    var kernels = List[Kernel]()
    var at = Dict[String, Int]()
    var first = -1
    var last = 0
    var rows = 0
    var edges = List[Int]()  # time << 1 | is_end, sorted below: the sweep that counts each overlap once
    for path in files:
        var lines = open(path, "r").read().split("\n")
        var head = csv_fields(String(lines[0]))
        var c_name = -1
        var c_start = -1
        var c_end = -1
        for i in range(len(head)):
            c_name = i if head[i] == "Kernel_Name" else c_name
            c_start = i if head[i] == "Start_Timestamp" else c_start
            c_end = i if head[i] == "End_Timestamp" else c_end
        if c_name < 0 or c_start < 0 or c_end < 0:
            print("FAIL rank: " + path + " has no Kernel_Name, Start_Timestamp, End_Timestamp columns")
            exit(1)
        for l in range(1, len(lines)):
            if not lines[l]:
                continue
            var f = csv_fields(String(lines[l]))
            var start = atol(f[c_start])
            var end = atol(f[c_end])
            if start < job_start or start >= job_end:
                continue
            if f[c_name] not in at:
                at[f[c_name]] = len(kernels)
                kernels.append(Kernel(f[c_name]))
            ref k = kernels[at[f[c_name]]]
            k.ns += end - start
            k.n += 1
            first = start if first < 0 else min(first, start)
            last = max(last, end)
            rows += 1
            edges.append(start << 1)
            edges.append(end << 1 | 1)
            if stages:  # binary search: a job has about 60 nodes and a million dispatches
                var lo = 0
                var hi = len(stages)
                while hi - lo > 1:
                    var m = (lo + hi) // 2
                    if stages[m].start <= start:
                        lo = m
                    else:
                        hi = m
                var ki = at[f[c_name]]
                stages[lo].ns[ki] = stages[lo].ns.get(ki, 0) + end - start
    var order = List[Int]()
    var total = 0
    for i in range(len(kernels)):
        order.append(i)
        total += kernels[i].ns
    for i in range(1, len(order)):  # hundreds of kernels at most
        var j = i
        while j > 0 and kernels[order[j]].ns > kernels[order[j - 1]].ns:
            order.swap_elements(j, j - 1)
            j -= 1
    sort(edges)
    var busy = 0
    var depth = 0
    var since = 0
    for e in edges:
        var t = e >> 1
        if e & 1:
            depth -= 1
            if depth == 0:
                busy += t - since
                add_busy(stages, since, t)
        else:
            if depth == 0:
                since = t
            depth += 1
    print(pad("share", 7) + pad("ms", 11) + pad("n", 9) + pad("avg us", 10) + pad("mid", 9) + "  kernel")
    for r in range(min(top, len(order)) if top > 0 else len(order)):
        ref k = kernels[order[r]]
        print(pad(fmt1(k.ns * 100, total) + "%", 7) + pad(fmt1(k.ns, 1000000), 11) + pad(String(k.n), 9) + pad(fmt1(k.ns, 1000 * k.n), 10) + pad(String((k.n + 1) // 2), 9) + "  " + k.name)
    var a = job_start if stages else first
    var b = job_end if stages else last
    print(String(rows) + " dispatches, " + String(len(kernels)) + " kernels, " + fmt1(total, 1000000) + " ms in kernels of " + fmt1(last - first, 1000000) + " ms span")
    print("busy " + fmt1(busy, 1000000) + " ms of " + fmt1(b - a, 1000000) + " ms, idle " + fmt1((b - a - busy) * 100, max(b - a, 1)) + "%")
    if stages:
        print(pad("wall s", 8) + pad("busy s", 8) + pad("idle", 7) + "  stage, its top kernel")
        for i in range(len(stages)):
            ref st = stages[i]
            var wall = max(st.end - st.start, 1)
            var best = -1
            var best_ns = 0
            for ke in st.ns.items():
                if ke.value > best_ns:
                    best = ke.key
                    best_ns = ke.value
            print(pad(fmt1(wall, 1000000000), 8) + pad(fmt1(st.busy, 1000000000), 8) + pad(fmt1((wall - st.busy) * 100, wall) + "%", 7) + "  " + st.label + ("  " + kernels[best].name if best >= 0 else ""))

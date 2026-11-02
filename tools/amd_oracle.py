"""wavewait against AMD's own ATT decoder (what rocprofv3 writes next to the trace).

usage: amd_oracle.py WAVEWAIT_JSON CODE_JSON
Per wave, when the dispatch dir holds AMD's per-wave files (filenames.json + se*.json): wavewait's issues must
equal AMD's instructions minus one per wave (s_endpgm sends no packet), and its wave count AMD's.
Per pc, joined on vaddr with code.json: AMD's Hit must equal wavewait's waves, and every s_waitcnt's Latency
wavewait's cycles to within one cycle per wave (the two decoders may give the issue cycle to the instruction
before). code.json only holds the instructions AMD mapped to an ISA line; when that is not all of them (Mojo
code objects: 512 of 3590), its hits undercount and the per-pc check is skipped, the per-wave one still holds.
"""
import json
import os
import sys

wj = json.load(open(sys.argv[1]))
ww = {r["pc"]: r for r in wj["rows"]}
code = json.load(open(sys.argv[2]))["code"] or []
amd = [r for r in code if r[6] > 0 and r[5] in ww]
bad = []
d = os.path.dirname(sys.argv[2])
per_wave = ""
amd_inst = None
fn = json.load(open(d + "/filenames.json"))["wave_filenames"] if os.path.exists(d + "/filenames.json") else {}
files = [v[0] for se in fn.values() for sm in se.values() for sl in sm.values() for v in sl.values()]
if files and os.path.exists(d + "/" + files[0]):  # ATT_SLIM deleted them: per-pc only
    amd_inst = sum(len(json.load(open(d + "/" + f))["wave"]["instructions"]) for f in files)
    issues = sum(r["waves"] for r in wj["rows"])
    per_wave = f"; per wave: {len(files)} waves, AMD {amd_inst} instructions, wavewait {issues} issues + 1 per wave"
    if len(files) != wj["waves"]:
        bad.append(f"AMD traced {len(files)} waves, wavewait {wj['waves']}")
    elif issues != amd_inst - len(files):
        bad.append(f"AMD {amd_inst} instructions over {len(files)} waves, wavewait {issues} issues")
covered = sum(r[6] for r in code)
partial = amd_inst is not None and covered < amd_inst
if not amd and not per_wave:
    bad.append("no instruction joined on vaddr (wrong dispatch or code object)")
waits = exact = 0
for isa, _, _, _, _, va, hit, lat, _, idle in [] if partial else amd:
    w = ww[va]
    if hit != w["waves"]:
        bad.append(f"{va:#x} {isa}: AMD hit {hit}, wavewait waves {w['waves']}")
    if isa.startswith("s_waitcnt "):
        waits += 1
        exact += lat == w["sum"]
        if abs(lat - w["sum"]) > hit:
            bad.append(f"{va:#x} {isa}: AMD latency {lat}, wavewait {w['sum']}")
tot_amd = sum(r[7] + r[9] for r in amd)
tot_ww = sum(ww[r[5]]["sum"] for r in amd)
if partial:
    print(f"code.json maps {covered} of {amd_inst} instructions: per-pc check skipped{per_wave}")
else:
    print(f"{len(amd)} rows joined, {exact}/{waits} waits exact (rest within 1 cycle per wave), cycles wavewait {tot_ww} AMD latency+idle {tot_amd}{per_wave}")
for b in bad:
    print("FAIL oracle: " + b)
sys.exit(1 if bad else 0)

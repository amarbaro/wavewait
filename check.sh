#!/usr/bin/env bash
# CPU-only check: the stall table of the committed saxpy capture is aligned (s_clause traced, s_endpgm not),
# and the Mojo analyzer prints the frozen table (fixtures/golden, from the retired Python analyzer) byte for byte.
set -euo pipefail
cd "$(dirname "$0")"
TG=${TINYGRAD:-$HOME/Projects/AMD/tinygrad}
out=$(cat fixtures/golden/saxpy_cu.top3.txt)
rows=$(grep -cE '^0x[0-9a-f]+ +[0-9.]+ +[0-9]+ +4 ' <<<"$out")
[ "$rows" = 28 ] || { echo "FAIL rows: $rows instructions with 4 waves, want 28"; exit 1; }
grep -qE '^0x19bc .* global_store_b32' <<<"$out" && ! grep -q 's_endpgm' <<<"$out" || { echo "FAIL walk: last traced instruction is not the store (s_endpgm emits no packet)"; exit 1; }
grep -A1 '^top 3' <<<"$out" | grep -q '884.2  s_clause' || { echo "FAIL top: wave start not on the traced s_clause"; exit 1; }
mkdir -p build
rtlib=$(find "$(dirname "$(readlink -f "$(command -v mojo)")")/.." -name libKGENCompilerRTShared.so -print -quit)
[ -n "$rtlib" ] || { echo "FAIL mojo: runtime lib not found next to $(command -v mojo)"; exit 1; }
mojo build -I src src/wavewait.mojo -Xlinker -rpath -Xlinker "$(dirname "$rtlib")" -o build/wavewait >/dev/null || { echo "FAIL mojo: build"; exit 1; }
diff <(./build/wavewait fixtures/saxpy_cu --top 3) <(echo "$out") >/dev/null || { echo "FAIL mojo: table differs from fixtures/golden/saxpy_cu.top3.txt"; exit 1; }
# --causes only moves cycles onto the waits: the per-cause sum equals the raw total (12897 on saxpy)
raw=$(./build/wavewait fixtures/saxpy_cu --top 0 | awk '/^ *0x/{t+=$2*$4} END{printf "%.0f", t}')
cz=$(./build/wavewait fixtures/saxpy_cu --causes --top 0 | awk '/%  /{t+=$1} END{print t}')
[ "$raw" = "$cz" ] && [ "$cz" = 12897 ] || { echo "FAIL causes: raw $raw vs by-cause $cz, want 12897"; exit 1; }
./build/wavewait fixtures/saxpy_cu --causes --top 0 | grep -qE '^0x1974 +673.5 .* mem +s_waitcnt vmcnt' || { echo "FAIL causes: vmcnt wait not charged"; exit 1; }
diff <(./build/wavewait fixtures/saxpy_cu.pkl --top 3) <(echo "$out") >/dev/null || { echo "FAIL pkl: direct .pkl input differs"; exit 1; }
rm -rf fixtures/saxpy_cu.pkl.raw
# W1 diff: the ATT capture aligns with the tinygrad capture row for row, the synthetic pair
# (saxpy_renamed = saxpy_att with one instruction renamed) shows exactly one removed row.
mkdir -p .work/features
./build/wavewait diff fixtures/saxpy_cu fixtures/saxpy_att > .work/features/W1-diff-pair.txt
grep -q '^28 rows aligned, 0 inserted, 0 removed$' .work/features/W1-diff-pair.txt || { echo "FAIL diff: the pair is not aligned"; exit 1; }
grep -qE '^ 0x1900 .*\+[0-9]' .work/features/W1-diff-pair.txt || { echo "FAIL diff: no cycle delta on the wave start row"; exit 1; }
./build/wavewait diff fixtures/saxpy_att fixtures/saxpy_renamed > .work/features/W1-diff-synth.txt
grep -q '^27 rows aligned, 1 inserted, 1 removed$' .work/features/W1-diff-synth.txt || { echo "FAIL diff: the synthetic pair is not aligned"; exit 1; }
rm1=$(grep -c '^-0x' .work/features/W1-diff-synth.txt || true)
[ "$rm1" = 1 ] || { echo "FAIL diff: $rm1 removed rows, want 1"; exit 1; }
ins1=$(grep -c '^+0x' .work/features/W1-diff-synth.txt || true)
[ "$ins1" = 1 ] || { echo "FAIL diff: $ins1 inserted rows, want 1"; exit 1; }
# W2 json: the round-trip parses, the row count equals the text table's, the key order is stable
./build/wavewait fixtures/saxpy_cu --top 0 --json > .work/features/W2-table.json
rows_json=$(python3 -c "import json; print(len(json.load(open('.work/features/W2-table.json'))['rows']))") || { echo "FAIL json: the json does not parse"; exit 1; }
rows_text=$(./build/wavewait fixtures/saxpy_cu --top 0 | grep -cE '^0x')
[ "$rows_json" = "$rows_text" ] || { echo "FAIL json: $rows_json rows in the json, $rows_text in the text table"; exit 1; }
python3 -c "import json; d=json.load(open('.work/features/W2-table.json')); assert d['grand']==12897, d['grand']; assert list(d['rows'][0])[:3]==['pc','text','waves'], list(d['rows'][0])[:3]" || { echo "FAIL json: totals or key order"; exit 1; }
./build/wavewait diff fixtures/saxpy_att fixtures/saxpy_renamed --json > .work/features/W2-diff.json
python3 -c "import json; d=json.load(open('.work/features/W2-diff.json')); assert (d['aligned'],d['inserted'],d['removed'])==(27,1,1), d; assert len(d['rows'])==29, len(d['rows'])" || { echo "FAIL json: diff counters"; exit 1; }
# W3 source lines: the -g capture maps every traced pc to a DWARF source position, the mapping
# agrees with llvm-addr2line (the ground truth) for every traced pc, the kernel body rows land in
# examples/saxpy.hip, and the per-line totals sum to the table total.
./build/wavewait fixtures/saxpy_g --json > .work/features/W3-lines.json
./build/wavewait fixtures/saxpy_g --by-line > .work/features/W3-by-line.txt
./build/wavewait fixtures/saxpy_cu --by-line > .work/features/W3-by-line-nodbg.txt
python3 - <<'PY' || { echo "FAIL lines: the DWARF mapping of the -g capture"; exit 1; }
import json, subprocess
d = json.load(open('.work/features/W3-lines.json'))
rows = d['rows']
assert len(rows) == 28, len(rows)
for r in rows:
    assert r['line'] > 0 and r['file'], r            # every traced pc has a source position
    h = hex(r['pc'])
    truth = subprocess.run(['/usr/bin/llvm-addr2line', '-e', 'fixtures/saxpy_g/codeobj.out', h],
                           capture_output=True, text=True).stdout.strip()
    f, _, l = truth.rpartition(':')
    assert r['line'] == int(l), (r, truth)             # the line matches the ground truth
    assert r['file'].rsplit('/', 1)[-1] == f.rsplit('/', 1)[-1], (r, truth)
hip = [r for r in rows if r['file'] == 'examples/saxpy.hip']
assert len(hip) == 19, len(hip)                       # the kernel body rows (the rest are the
                                                       # prologue and the barrier, in the headers)
groups = {}
for l in open('.work/features/W3-by-line.txt'):
    s = l.strip()
    if s and s[0].isdigit() and ' cycles over ' not in s:
        f = s.split()
        groups[f[2]] = int(f[0])
tot = sum(groups.values())
assert tot == d['grand'], (tot, d['grand'])            # the per-line totals sum to the total
assert sum(v for k, v in groups.items() if k.startswith('examples/saxpy.hip')) == sum(r['csum'] for r in hip)
nodbg = [l.strip() for l in open('.work/features/W3-by-line-nodbg.txt') if l.strip() and l.strip()[0].isdigit() and ' cycles over ' not in l]
assert len(nodbg) == 1 and nodbg[0].split()[0] == '12897', nodbg
PY
# W4 hints: the probe fixture fires every rule of the rules table, and the saxpy hints are
# exactly the expected ones (fixtures/golden/saxpy_cu.hints.txt).
./build/wavewait fixtures/probe --hints > .work/features/W4-probe-hints.txt
python3 - <<'PY' || { echo "FAIL hints: the probe does not fire every rule"; exit 1; }
import re
lines = [l for l in open('.work/features/W4-probe-hints.txt') if l.startswith('0x')]
rules = set()
for l in lines:
    m = re.match(r'^0x[0-9a-f]+\s+(.+?)\s+\d+\.\d+\s\s', l)
    assert m, l
    rules.add(m.group(1).strip())
want = {'hoist', 'idle wait', 'vopd', 'lds bank'}
assert want <= rules, (want, rules)
assert len(lines) == 9, len(lines)
PY
diff <(./build/wavewait fixtures/saxpy_cu --hints) fixtures/golden/saxpy_cu.hints.txt >/dev/null || { echo "FAIL hints: the saxpy hints differ from fixtures/golden/saxpy_cu.hints.txt"; exit 1; }
# W5 app mode: one rocprofv3 ATT capture (saxpy + the runtime's fill kernel) gives 2 rows, and the saxpy row's cycles
# equal the per-kernel table of the same .att (tools/from-att.sh, code object 2)
app=$(./build/wavewait app fixtures/app) || { echo "FAIL app: exit $?"; exit 1; }
[ "$(grep -cE '^(saxpy|__amd_rocclr_fillBufferAligned) ' <<<"$app")" = 2 ] || { echo "FAIL app: want 2 kernel rows"; exit 1; }
k=$(mktemp -d); tools/from-att.sh fixtures/app "$k" >/dev/null || { echo "FAIL app: from-att"; exit 1; }
tab=$(./build/wavewait "$k" --top 0 | awk '/^ *0x/{t+=$2*$4} END{printf "%.0f", t}'); rm -rf "$k"
row=$(awk '$1=="saxpy"{print $2}' <<<"$app")
[ "$row" = "$tab" ] && [ "$row" -gt 0 ] || { echo "FAIL app: saxpy row $row cycles, per-kernel table $tab"; exit 1; }
# app --kernel / --dispatch: the full table of ONE dispatch, walked against its own code object, equals its app row;
# a selector that matches nothing fails
one=$(./build/wavewait app fixtures/app --kernel saxpy --json | awk -F'[:,]' '/"grand"/{print $2+0}')
[ "$one" = "$row" ] || { echo "FAIL app one: --kernel saxpy grand '$one', app row $row"; exit 1; }
n=$(awk '$1=="saxpy"{print $NF}' <<<"$app")
one=$(./build/wavewait app fixtures/app --dispatch "$n" --json | awk -F'[:,]' '/"grand"/{print $2+0}')
[ "$one" = "$row" ] || { echo "FAIL app one: --dispatch $n grand '$one', app row $row"; exit 1; }
./build/wavewait app fixtures/app --dispatch 999999 >/dev/null && { echo "FAIL app one: a dispatch that does not exist accepted"; exit 1; }
# oracle: AMD's own decoder (rocprofv3's code.json) on the same dispatches: hits equal waves on every pc,
# every s_waitcnt within 1 cycle per wave (tools/amd_oracle.py fails on a vacuous join). app_div (examples/diverge.hip)
# was traced on SIMD 3: walking SIMD 0's wave slots there lost loop iterations (16 vs 14 hits)
# app_div and app_ssm keep AMD's per-wave files: issues = AMD's instructions - 1 per wave, exactly; app_ssm's code.json
# maps 512 of 3590 instructions (a Mojo code object), so its per-pc hits undercount and only the per-wave check runs
ko=$(mktemp -d)
for o in "app 1" "app_mega 4127" "app_gemv 4" "app_div 1" "app_ssm 36"; do set -- $o
  ./build/wavewait app "fixtures/$1" --dispatch "$2" --json > "$ko/oracle.json" || { echo "FAIL oracle: $1 --json"; exit 1; }
  python3 tools/amd_oracle.py "$ko/oracle.json" "fixtures/$1"/ui_output_*_dispatch_"$2"/code.json >/dev/null || { echo "FAIL oracle: $1 disagrees with AMD's decoder"; exit 1; }
done
# torch (tools/torch_probe.py, warm): dispatch 5 enters mid code object (a torch object holds many kernels; walking from its lowest
# pc found 0 of the kernel), dispatch 2 is MIOpen's Sp3 conv (s_version, s_call_b64, computed s_setpc_b64: 763 of 165336 issues before)
tz=$(mktemp -d); cp -r fixtures/app_torch/. "$tz/"; find "$tz" -name '*.gz' -exec gunzip {} +
for o in "2 165336" "5 22410"; do set -- $o
  ./build/wavewait app "$tz" --dispatch "$1" --json > "$ko/oracle.json" || { echo "FAIL torch: dispatch $1 --json"; exit 1; }
  python3 tools/amd_oracle.py "$ko/oracle.json" "$tz"/ui_output_*_dispatch_"$1"/code.json > /dev/null || { echo "FAIL torch: dispatch $1 disagrees with AMD's decoder"; exit 1; }
  got=$(python3 -c 'import json,sys; print(sum(r["waves"] for r in json.load(open(sys.argv[1]))["rows"]))' "$ko/oracle.json")
  [ "$got" = "$2" ] || { echo "FAIL torch: dispatch $1 walked $got issues, AMD $2"; exit 1; }
done
rm -rf "$tz"
# rank: a whole process's kernels from the kernel trace; the summed ms and the quoted C++ names against python's csv
rk=$(./build/wavewait rank fixtures/rank --top 0)
want=$(python3 -c 'import csv; r=list(csv.DictReader(open("fixtures/rank/1_kernel_trace.csv"))); print(len(r), "dispatches,", len({x["Kernel_Name"] for x in r}), "kernels,", "%d.%d" % (lambda t: (t // 1000000, t * 10 // 1000000 % 10))(sum(int(x["End_Timestamp"]) - int(x["Start_Timestamp"]) for x in r)), "ms")')
[ "$(tail -2 <<<"$rk" | head -1 | cut -d' ' -f1-6)" = "$want" ] || { echo "FAIL rank: '$(tail -2 <<<"$rk")', want '$want'"; exit 1; }
[ "$(./build/wavewait rank fixtures/rank --top 2 | wc -l)" = 5 ] || { echo "FAIL rank: --top 2 is not header + 2 rows + 2 totals"; exit 1; }
busy=$(python3 -c 'import csv
iv = sorted((int(x["Start_Timestamp"]), int(x["End_Timestamp"])) for x in csv.DictReader(open("fixtures/rank/1_kernel_trace.csv")))
busy = 0; a, b = iv[0]
for s, e in iv[1:]:
    if s > b: busy += b - a; a = s; b = e
    else: b = max(b, e)
busy += b - a
print("busy %d.%d ms" % (busy // 1000000, busy * 10 // 1000000 % 10))')
grep -q "^$busy of " <<<"$rk" || { echo "FAIL rank: union of dispatch times, want $busy: $(grep '^busy' <<<"$rk")"; exit 1; }
st=$(./build/wavewait rank fixtures/rank --top 1 --stages fixtures/rank/stages.tsv) || { echo "FAIL rank: --stages"; exit 1; }
grep -q '^ *0\.0 *0\.0 *12\.3%  KSampler 3  Cijk_Ailk' <<<"$st" && grep -q '8\.4%  SaveGLB 4' <<<"$st" || { echo "FAIL rank: stage rows"; exit 1; }
./build/wavewait rank fixtures/app > /dev/null && { echo "FAIL rank: a run dir without a kernel trace accepted"; exit 1; }
# timestamps: AMD's decoder reads SQTT 0b0100001 (NEW_PC) as 64 bits (tinygrad 72), and a TS_DELTA_OR_MARK with rt set
# adds no delta; fixtures/app_ts (448 KB of a mojo-baro mega kernel) walked 4 cycles late from slot 0's packet 45835 on,
# the whole capture 3.7e10 cycles late. Every slot-0 packet time equals AMD's issue + stall.
WAVEWAIT_PACKETS=0 ./build/wavewait app fixtures/app_ts --dispatch 4127 | python3 -c '
import gzip, json, sys
ours = [int(l.split()[1][2:]) for l in sys.stdin if l.startswith("PKT")]
amd = [i[0] + i[2] for i in json.load(gzip.open(sys.argv[1]))["wave"]["instructions"]]
n = min(len(ours), len(amd)); bad = [k for k in range(n) if ours[k] != amd[k]]
assert n > 46000 and not bad, (n, bad[:1], bad and (ours[bad[0]], amd[bad[0]]))' fixtures/app_ts/ui_output_*/se0_sm0_sl0_wv0.json.gz || { echo "FAIL ts: slot 0 packet times differ from AMD's"; exit 1; }
# a trace cut at a quarter (ATT buffer full; saxpy ends inside the first half) says so: waves that never ended are counted, not silently dropped
k=$(mktemp -d); cp fixtures/saxpy_cu/pcmap.tsv "$k"/; for b in fixtures/saxpy_cu/blob_*.bin; do head -c $(( $(stat -c %s "$b") / 4 )) "$b" > "$k/$(basename "$b")"; done
./build/wavewait "$k" --top 0 | grep -q '^WARN [0-9]* of [0-9]* waves never ended' || { echo "FAIL open waves: a quarter trace gives no WARN"; rm -rf "$k"; exit 1; }
rm -rf "$k"
./build/wavewait fixtures/saxpy_cu --top 0 | grep -q 'never ended' && { echo "FAIL open waves: WARN on a complete trace"; exit 1; }
# a persistent kernel's grid barrier (mojo-baro megakernel, a 256 KB prefix of its ATT): buffer_gl1_inv / buffer_gl0_inv
# report as INST ops 0x2f-0x36 (BUF); classed as VALU they desynced the walk (1504 mismatches, a false "issue wait")
row=$(./build/wavewait app fixtures/app_mega | awk '$1 ~ /^mega_amar_mega_token/')
[ "$(awk '{print $(NF-1)}' <<<"$row")" = 0 ] || { echo "FAIL buf ops: megakernel walk mismatches: $row"; exit 1; }
[ "$(awk '{print $(NF-2)}' <<<"$row")" = 4 ] || { echo "FAIL open waves: the 256 KB prefix should leave 4 waves open: $row"; exit 1; }
# --loops: every charged cycle lands in exactly one innermost loop or outside all of them
lp=$(./build/wavewait app fixtures/app_mega --kernel mega --loops --top 1000)
sum=$(awk '/^ *[0-9]+ +[0-9]+ /{t+=$1} END{print t}' <<<"$lp"); grand=$(awk '/ loops; /{print $NF}' <<<"$lp")
[ "$sum" = "$grand" ] && [ "$grand" -gt 0 ] || { echo "FAIL loops: loops + outside = $sum, table total $grand"; exit 1; }
grep -q ' loop ' <<<"$lp" || { echo "FAIL loops: no loop found in the megakernel prefix"; exit 1; }
# a branchy kernel (a mojo-baro GEMV): the code map app mode builds in Mojo equals tools/export.py's on the
# branch lines too, so the row matches the per-kernel table, and the row is named by the symbol, not an instruction
app=$(./build/wavewait app fixtures/app_gemv) || { echo "FAIL app gemv: exit $?"; exit 1; }
k=$(mktemp -d); tools/from-att.sh fixtures/app_gemv "$k" >/dev/null || { echo "FAIL app gemv: from-att"; exit 1; }
tab=$(./build/wavewait "$k" --json | awk -F'[:,]' '/"grand"/{print $2+0}'); rm -rf "$k"
row=$(awk '$1 ~ /^gemv_kq_amar_gemv_kq_/{print $2}' <<<"$app")
[ "$row" = "$tab" ] && [ "$row" -gt 0 ] || { echo "FAIL app gemv: row '$row' cycles, per-kernel table $tab"; exit 1; }
# a model-shape GEMV (8960 x 1536 Q4_K, 48 waves on the traced SIMD): the SE reports the waves of all its WGPs
# and immediates come as IMMEDIATE_MASK too; executions per pc equal tinygrad's walk (tools/oracle_counts.py)
diff <(./build/wavewait fixtures/gemv_q4k --json | sed -nE 's/.*"pc": ([0-9]+),.*"waves": ([0-9]+),.*/\1 \2/p') fixtures/golden/gemv_q4k.counts >/dev/null \
  || { echo "FAIL gemv: executions per pc differ from fixtures/golden/gemv_q4k.counts"; exit 1; }
# exposed cycles: idle cycles of the traced SIMD shared by the waves alive then. Conservation: rows + tail = idle.
# Falsifier: one wave slot (WAVEWAIT_ONLY) has nothing to overlap, so exposed = cycles - one issue cycle per issue.
for c in "fixtures/saxpy_cu" "fixtures/gemv_q4k" "app fixtures/app --dispatch 1"; do
  ./build/wavewait $c --json | python3 -c "
import json,sys; d=json.load(sys.stdin); x=sum(r['exposed'] for r in d['rows'])
assert abs(d['exposed']+d['tail']-d['idle'])<=1 and abs(x-d['exposed'])<=len(d['rows']), (d['exposed'],d['tail'],d['idle'],x)
assert d['exposed'] <= d['grand']" || { echo "FAIL exposed: $c does not conserve the SIMD's idle cycles"; exit 1; }
  WAVEWAIT_ONLY=0 ./build/wavewait $c --json | python3 -c "
import json,sys; d=json.load(sys.stdin); r=d['rows']
want=sum(x['sum'] for x in r)-sum(x['waves'] for x in r)+d['waves']
assert d['exposed']==want, (d['exposed'],want)" || { echo "FAIL exposed: one wave slot of $c overlaps itself"; exit 1; }
done
ex=$(./build/wavewait fixtures/saxpy_cu --exposed --top 0)
[ "$(tail -n +2 <<<"$ex" | head -n -2 | awk '{print $3}' | sort -nrc && echo sorted)" = sorted ] || { echo "FAIL exposed: rows not costliest first"; exit 1; }
xb=$(mktemp); xv=$(./build/wavewait fixtures/saxpy_cu --json | awk -F'[:,]' '/"exposed": [0-9]+, "idle"/{print $2+0}')
echo "exposed $xv" > "$xb"; ./build/wavewait budget "$xb" fixtures/saxpy_cu >/dev/null || { echo "FAIL budget: own exposed rejected"; exit 1; }
echo "exposed $((xv - 1))" > "$xb"; ./build/wavewait budget "$xb" fixtures/saxpy_cu >/dev/null && { echo "FAIL budget: exposed -1 accepted"; exit 1; }
# occupancy: the waves per SIMD each kernel allows equal hipcc's own `; Occupancy:` comment (fixtures/occ/expected.txt,
# from -save-temps), VGPR- and LDS-limited kernels in WGP and CU mode; --exposed names the limit of the traced kernel
for m in wgp cu; do
  diff <(./build/wavewait occ fixtures/occ/occ-$m.co | awk -v m=$m '{print m, $1, $3}') <(grep "^$m " fixtures/occ/expected.txt) >/dev/null || { echo "FAIL occ: $m differs from hipcc's Occupancy"; exit 1; }
done
grep -q '^occupancy: mean 4.0, max 4 waves on the traced SIMD; the code object allows 5 (limited by vgpr)$' <<<"$(./build/wavewait app fixtures/app_mega --kernel mega --exposed --top 1)" || { echo "FAIL occ: mega occupancy line"; exit 1; }
# divergent: fixtures/app_div's waves loop 1 or 3 times (AMD's per-wave files: the branch issued 1 or 3 times, a
# not-taken one is not counted), so the hint names its back edge as taken 0..2 times; no uniform capture fires it
grep -q '^0x1a08 .*divergent .*s_cbranch_execnz 65501 (taken 0..2 times per wave)$' <<<"$(./build/wavewait app fixtures/app_div --dispatch 1 --hints)" || { echo "FAIL divergent: app_div back edge"; exit 1; }
for c in fixtures/saxpy_cu fixtures/probe fixtures/gemv_q4k "app fixtures/app_gemv --dispatch 4"; do
  ./build/wavewait $c --hints | grep -q divergent && { echo "FAIL divergent: fires on uniform $c"; exit 1; }
done
# a flat run dir (rocprofv3 -o NAME: traces and code objects in the run dir itself, no host subdir) reads the same
fl=$(mktemp -d); cp fixtures/app_div/hercule/* "$fl/"; cp -r fixtures/app_div/ui_output_* "$fl/"
[ "$(./build/wavewait app "$fl" | tail -1)" = "$(./build/wavewait app fixtures/app_div | tail -1)" ] || { echo "FAIL app: a flat run dir reads differently"; exit 1; }
# W6 budget gate: saxpy's own numbers pass; 10% fewer cycles, or a tighter cause share, exit 1 naming the rule
./build/wavewait budget fixtures/budget/saxpy_cu.budget fixtures/saxpy_cu >/dev/null || { echo "FAIL budget: own numbers rejected"; exit 1; }
tight=$(mktemp); sed 's/^cycles 12897/cycles 11607/' fixtures/budget/saxpy_cu.budget > "$tight"
out=$(./build/wavewait budget "$tight" fixtures/saxpy_cu) && { echo "FAIL budget: -10% cycles accepted"; exit 1; }
grep -q '^OVER cycles 11607 (have 12897)' <<<"$out" || { echo "FAIL budget: offending rule not named"; exit 1; }
sed 's/^cause wave start 27/cause wave start 20/' fixtures/budget/saxpy_cu.budget > "$tight"
out=$(./build/wavewait budget "$tight" fixtures/saxpy_cu) && { echo "FAIL budget: cause share over budget accepted"; exit 1; }
grep -q '^OVER cause wave start 20 (have 27)' <<<"$out" || { echo "FAIL budget: cause rule not named"; exit 1; }
sed 's/^cycles 12897/cycles 12896/' fixtures/budget/saxpy_cu.budget > "$tight"  # boundary: one cycle under
./build/wavewait budget "$tight" fixtures/saxpy_cu >/dev/null && { echo "FAIL budget: one cycle over accepted"; exit 1; }
rm -f "$tight"
# W8 timeline page: one lane per traced wave, one bar per issued instruction (saxpy: 4 waves x 28 instructions)
page=$(mktemp --suffix=.html); ./build/wavewait fixtures/saxpy_cu --html "$page" >/dev/null || { echo "FAIL html: exit $?"; exit 1; }
counts=$(python3 -c "import json,sys; t=open(sys.argv[1]).read(); d=json.loads(t.split('const D=',1)[1].split(';\nconst pal',1)[0]); l=[x for x in d['lanes'] if x]; print(len(l), sum(map(len,l)), len(d['rows']))" "$page"); rm -f "$page"
[ "$counts" = "4 112 28" ] || { echo "FAIL html: lanes bars rows = $counts, want 4 112 28"; exit 1; }
# W1 --top N cuts the table of an app dispatch and of every --causes table to the N costliest rows (charged cycles,
# costliest first); the plain capture table stays whole (its golden above), --top 0 prints every row
cut=$(./build/wavewait app fixtures/app --kernel saxpy --top 3)
[ "$(grep -cE '^0x' <<<"$cut")" = 3 ] || { echo "FAIL top: app --top 3 does not print exactly 3 rows"; exit 1; }
[ "$(./build/wavewait app fixtures/app --kernel saxpy --top 0 | grep -cE '^0x')" = 28 ] || { echo "FAIL top: app --top 0 does not print all 28 rows"; exit 1; }
grep -qE '^0x1900 .*s_clause' <<<"$cut" && grep -q '^(3 of 28 rows' <<<"$cut" || { echo "FAIL top: the costliest row is not first"; exit 1; }
[ "$(./build/wavewait fixtures/saxpy_cu --causes --top 2 | grep -cE '^0x')" = 2 ] || { echo "FAIL top: --causes --top 2 does not print exactly 2 rows"; exit 1; }
cz2=$(./build/wavewait fixtures/saxpy_cu --causes --top 2 | awk '/%  /{t+=$1} END{print t}')
[ "$cz2" = 12897 ] || { echo "FAIL top: the cause summary of a cut table is $cz2, want 12897"; exit 1; }
# W2 --by-class: the classes sum to the table total, and the issue counts equal an awk count of the plain table
# (n column per mnemonic prefix); json gains the same classes
bc=$(./build/wavewait fixtures/saxpy_cu --by-class)
[ "$(awk '/^[a-z]/ && NF>=5 && $(NF-3) ~ /^[0-9]+$/ && $1!="class"{t+=$(NF-3)} END{print t}' <<<"$bc")" = 12897 ] || { echo "FAIL by-class: the class cycles do not sum to 12897"; exit 1; }
tbl=$(./build/wavewait fixtures/saxpy_cu --top 0)
for want in "vmem load:global_load" "vmem store:global_store" "smem:s_load" "lds:ds_"; do
  cls=${want%%:*}; pre=${want##*:}
  got=$(awk -v c="$cls" 'index($0,c)==1 {print $(NF-1)}' <<<"$bc" | head -1)
  exp=$(awk -v p="$pre" '/^0x/{for(i=1;i<=NF;i++) if(index($i,p)==1){t+=$4; break}} END{print t+0}' <<<"$tbl")
  [ "$got" = "$exp" ] || { echo "FAIL by-class: $cls issues $got, table count of $pre* is $exp"; exit 1; }
done
./build/wavewait fixtures/saxpy_cu --json | python3 -c "import json,sys; d=json.load(sys.stdin); c=d['classes']; assert sum(v['cycles'] for v in c.values())==d['grand']==12897, 'sum'; assert c['waitcnt mixed']['issues']==4 and c['waitcnt vmcnt']['issues']==4 and c['waitcnt lgkmcnt']['issues']==8, 'waitcnt split'" || { echo "FAIL by-class: json classes"; exit 1; }
# W3 diff of two app run dirs: --kernel/--dispatch picks the dispatch on each side. A run dir against itself moves nothing:
# totals equal the app row, every cause and class shift is 0, --top cuts the rows, json parses
appo=$(./build/wavewait app fixtures/app); srow=$(awk '$1=="saxpy"{print $2}' <<<"$appo"); sn=$(awk '$1=="saxpy"{print $NF}' <<<"$appo")
dsame=$(./build/wavewait diff fixtures/app fixtures/app --kernel saxpy --by-class --top 3)
grep -q "^total cycles $srow -> $srow (0, +0.0%)" <<<"$dsame" || { echo "FAIL diff run: totals are not the saxpy app row $srow"; exit 1; }
[ "$(grep -cE '^[ +-]0x' <<<"$dsame")" = 3 ] && grep -q '^28 rows aligned, 0 inserted, 0 removed$' <<<"$dsame" || { echo "FAIL diff run: --top 3 rows or the alignment count"; exit 1; }
[ "$(awk '/^cycles by class:/{f=1;next} /^cycles by cause/{f=0} f && NF>=5 && $1!="class"{ta+=$(NF-4); tb+=$(NF-3); s+=$(NF-2)} END{print ta, tb, s}' <<<"$dsame")" = "$srow $srow 0" ] || { echo "FAIL diff run: the class table does not hold $srow cycles on each side with a 0 shift"; exit 1; }
./build/wavewait diff fixtures/app fixtures/app --dispatch "$sn" --json --top 2 | python3 -c "import json,sys; d=json.load(sys.stdin); r=int(sys.argv[1]); assert d['grand']=={'a':r,'b':r}, d['grand']; assert len(d['rows'])==2, len(d['rows']); assert sum(v['a'] for v in d['classes'].values())==r; assert all(v['shift']==0 for v in d['causes'].values())" "$srow" || { echo "FAIL diff run: --dispatch json"; exit 1; }
./build/wavewait diff fixtures/app fixtures/app --kernel nosuchkernel >/dev/null && { echo "FAIL diff run: a kernel that matches nothing accepted"; exit 1; }
# the memory-linear alignment (Hirschberg, above WAVEWAIT_ALIGN_CELLS cells) finds an LCS as long as the table's on every fixture pair
for pair in "saxpy_cu saxpy_att" "saxpy_att saxpy_renamed" "gemv_q4k probe" "gemv_q4k saxpy_cu" "probe gemv_q4k"; do
  set -- $pair
  t=$(WAVEWAIT_ALIGN_CELLS=100000000 ./build/wavewait diff fixtures/$1 fixtures/$2 | grep 'rows aligned'); h=$(WAVEWAIT_ALIGN_CELLS=0 ./build/wavewait diff fixtures/$1 fixtures/$2 | grep 'rows aligned')
  [ -n "$t" ] && [ "$t" = "$h" ] || { echo "FAIL diff hirschberg: $1 vs $2: table '$t', hirschberg '$h'"; exit 1; }
done
# W4 tools/att-capture.sh without the GPU: a stand-in rocprofv3 that logs rocprofv3's real failure ("failed to create ATT
# packet") and then sleeps like its deadlocked signal handler: the watcher kills the whole process group within 5 s and exits
# FAIL; a 1 GB buffer is refused before anything runs; a trace over 90% of the buffer warns, a small one does not
afi=.work/att-fi; rm -rf "$afi"; mkdir -p "$afi/dec"; touch "$afi/dec/librocprof-trace-decoder.so"
cat > "$afi/fake-hang" <<FAKE1
#!/bin/sh
echo \$\$ > $PWD/$afi/pid
sleep 600 &
echo \$! > $PWD/$afi/child
echo "E20261026 output_stream.cpp:99 failed to create ATT packet" >&2
wait
FAKE1
cat > "$afi/fake-ok" <<FAKE2
#!/bin/sh
echo "\$*" > $PWD/$afi/args
while [ \$# -gt 0 ]; do [ "\$1" = -d ] && d=\$2; shift; done
mkdir -p \$d/dispatch_0; head -c \${FAKE_ATT_BYTES:-100} /dev/zero > \$d/dispatch_0/1_2_shader_engine_0_2.att
touch \$d/dispatch_0/se0_sm0_sl0_wv0.json \$d/dispatch_0/code.json
FAKE2
mkdir -p "$afi/bin"
cat > "$afi/bin/gpu-wait" <<'FAKE3'
#!/bin/sh
while [ "$1" != -- ]; do shift; done; shift
exec env -u ATT_EXTRA -u ATT_ROCPROF -u ATT_DECODER_DIR -u ATT_SLIM "$@"
FAKE3
chmod +x "$afi/fake-hang" "$afi/fake-ok" "$afi/bin/gpu-wait"
t0=$(date +%s%N); rc=0
out=$(ATT_NO_GPU_WAIT=1 ATT_ROCPROF="$afi/fake-hang" ATT_DECODER_DIR="$afi/dec" tools/att-capture.sh "$afi/hang" -- true 2>&1) || rc=$?
ms=$(( ($(date +%s%N) - t0) / 1000000 ))
[ "$rc" = 1 ] && grep -q "^FAIL att: rocprofv3 logged 'failed to create ATT packet'" <<<"$out" || { echo "FAIL att-capture: the ATT error was not turned into FAIL (rc $rc): $out"; exit 1; }
[ "$ms" -lt 5000 ] || { echo "FAIL att-capture: the watcher took $ms ms to kill the hung capture (limit 5000)"; exit 1; }
for f in pid child; do ps -p "$(cat "$afi/$f")" > /dev/null 2>&1 && { echo "FAIL att-capture: the stand-in's $f process is still alive after the kill"; exit 1; }; done
rc=0; out=$(ATT_BUFFER=0x40000000 ATT_NO_GPU_WAIT=1 tools/att-capture.sh "$afi/cap" -- true 2>&1) || rc=$?
[ "$rc" = 1 ] && grep -q '^FAIL att: buffer 1073741824 bytes is above the 256 MB cap' <<<"$out" || { echo "FAIL att-capture: a 1 GB buffer was not refused (rc $rc): $out"; exit 1; }
[ ! -e "$afi/cap" ] || { echo "FAIL att-capture: the refused capture created its run dir"; exit 1; }
out=$(FAKE_ATT_BYTES=4000 ATT_BUFFER=0x1000 ATT_NO_GPU_WAIT=1 ATT_ROCPROF="$afi/fake-ok" ATT_DECODER_DIR="$afi/dec" tools/att-capture.sh "$afi/full" -- true 2>&1) || { echo "FAIL att-capture: a good capture failed: $out"; exit 1; }
grep -q '^att: 1 traces .* of a 4 KiB buffer (97%)' <<<"$out" && grep -q '^WARN att: the biggest trace fills 97%' <<<"$out" || { echo "FAIL att-capture: no truncation WARN at 97%: $out"; exit 1; }
out=$(ATT_BUFFER=0x1000000 ATT_NO_GPU_WAIT=1 ATT_ROCPROF="$afi/fake-ok" ATT_DECODER_DIR="$afi/dec" tools/att-capture.sh "$afi/small" -- true 2>&1) || { echo "FAIL att-capture: a small capture failed: $out"; exit 1; }
grep -q WARN <<<"$out" && { echo "FAIL att-capture: WARN on a trace of 100 bytes in 16 MB: $out"; exit 1; }
ATT_BUFFER=0x40000000 ATT_BUFFER_OK=1 ATT_NO_GPU_WAIT=1 ATT_ROCPROF="$afi/fake-ok" ATT_DECODER_DIR="$afi/dec" tools/att-capture.sh "$afi/big" -- true > /dev/null 2>&1 || { echo "FAIL att-capture: ATT_BUFFER_OK=1 did not allow 1 GB"; exit 1; }
ATT_NO_GPU_WAIT=1 ATT_ROCPROF="$afi/fake-ok" ATT_DECODER_DIR="$afi/dec" tools/att-capture.sh "$afi/big" -- true > /dev/null 2>&1 && { echo "FAIL att-capture: a second capture into a used run dir accepted"; exit 1; }
# gpu-wait's job does not inherit the caller's env: the ATT_* overrides must still reach rocprofv3 (ATT_EXTRA was lost silently)
PATH="$PWD/$afi/bin:$PATH" ATT_EXTRA="--preload marker.so" ATT_SLIM=1 ATT_ROCPROF="$afi/fake-ok" ATT_DECODER_DIR="$afi/dec" tools/att-capture.sh "$afi/fwd" -- true > /dev/null 2>&1 || { echo "FAIL att-capture: a capture through gpu-wait failed"; exit 1; }
grep -q -- '--preload marker.so' "$afi/args" || { echo "FAIL att-capture: ATT_EXTRA did not cross gpu-wait"; exit 1; }
[ ! -e "$afi/fwd/dispatch_0/se0_sm0_sl0_wv0.json" ] && [ -e "$afi/fwd/dispatch_0/code.json" ] || { echo "FAIL att-capture: ATT_SLIM kept the per-wave JSON or lost code.json"; exit 1; }
echo "ok wavewait: 28 instructions x 4 waves, aligned to the store, mojo == golden, causes conserve 12897 cycles, diff aligns 28/28 and flags 1 removed row, json round-trips 28 rows, every traced pc maps to a DWARF line (addr2line agrees), the probe fires all 4 hint rules and the saxpy hints match the golden, app mode ranks 2 kernels and agrees with the per-kernel table (saxpy and a branchy GEMV), a 48-wave GEMV walks every pc as often as tinygrad does, the budget gate passes its own numbers and fails -10%, the timeline page holds 4 lanes x 28 bars, app --kernel/--dispatch equal the app row, a cut trace warns about open waves, the megakernel barrier walks with 0 mismatches, --loops conserves the cycles, --top N cuts the app and --causes tables to N rows, --by-class conserves 12897 cycles and counts the issues as awk does, diff takes two app run dirs and its linear-memory alignment equals the table, att-capture turns the ATT failure into a kill within 5 s and refuses a 1 GB buffer"

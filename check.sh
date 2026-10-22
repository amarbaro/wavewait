#!/usr/bin/env bash
# CPU-only check: the stall table of the committed saxpy capture is aligned (s_clause traced, s_endpgm not),
# and the Mojo analyzer prints the frozen table (fixtures/golden, from the retired Python analyzer) byte for byte.
set -euo pipefail
cd "$(dirname "$0")"
TG=${TINYGRAD:-$HOME/tinygrad}
out=$(cat fixtures/golden/saxpy_cu.top3.txt)
rows=$(grep -cE '^0x[0-9a-f]+ +[0-9.]+ +[0-9]+ +4 ' <<<"$out")
[ "$rows" = 28 ] || { echo "FAIL rows: $rows instructions with 4 waves, want 28"; exit 1; }
grep -qE '^0x19bc .* global_store_b32' <<<"$out" && ! grep -q 's_endpgm' <<<"$out" || { echo "FAIL walk: last traced instruction is not the store (s_endpgm emits no packet)"; exit 1; }
grep -A1 '^top 3' <<<"$out" | grep -q '884.2  s_clause' || { echo "FAIL top: wave start not on the traced s_clause"; exit 1; }
mkdir -p build
mojo build -I src src/wavewait.mojo -o build/wavewait >/dev/null || { echo "FAIL mojo: build"; exit 1; }
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
# a branchy kernel (a Mojo Q4_K GEMV): the code map app mode builds in Mojo equals tools/export.py's on the
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
echo "ok wavewait: 28 instructions x 4 waves, aligned to the store, mojo == golden, causes conserve 12897 cycles, diff aligns 28/28 and flags 1 removed row, json round-trips 28 rows, every traced pc maps to a DWARF line (addr2line agrees), the probe fires all 4 hint rules and the saxpy hints match the golden, app mode ranks 2 kernels and agrees with the per-kernel table (saxpy and a branchy GEMV), a 48-wave GEMV walks every pc as often as tinygrad does, the budget gate passes its own numbers and fails -10%, the timeline page holds 4 lanes x 28 bars"

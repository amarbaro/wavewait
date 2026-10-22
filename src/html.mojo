# The timeline page: one self-contained HTML file, a Gantt lane per traced wave (up to TL_CAP), one bar per
# issued instruction, width = its charged cycles, colour = the row's stall cause. Data is inlined as JSON.
from std.collections import List
from row import Analysis, jstr, pcs_of

comptime PAGE_HEAD = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>wavewait timeline</title><style>
:root{--bg:#fff;--fg:#222;--mute:#777}@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#ddd;--mute:#999}}
body{background:var(--bg);color:var(--fg);font:13px system-ui,sans-serif;margin:16px}
.lane{position:relative;height:14px;margin:2px 0}.bar{position:absolute;top:0;height:14px}
#key span{display:inline-block;margin-right:12px}#key i{display:inline-block;width:10px;height:10px;margin-right:4px}
#tip{color:var(--mute);min-height:1.4em}</style></head><body>
<h3 id="t"></h3><div id="key"></div><div id="tip">hover a bar</div><div id="g"></div><script>const D="""

comptime PAGE_TAIL = """;
const pal=['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f','#edc948','#b07aa1','#ff9da7','#9c755f','#bab0ac','#86bcb6','#d37295'];
const causes=[...new Set(Object.values(D.rows).map(r=>r[1]))],col=c=>pal[causes.indexOf(c)%pal.length];
document.getElementById('t').textContent=D.dir+': '+D.waves+' waves, '+D.grand+' cycles, '+D.lanes.filter(l=>l.length).length+' traced lanes';
document.getElementById('key').innerHTML=causes.map(c=>'<span><i style="background:'+col(c)+'"></i>'+c+'</span>').join('');
const end=Math.max(1,...D.lanes.map(l=>l.reduce((s,e)=>s+e[1],0))),g=document.getElementById('g'),tip=document.getElementById('tip');
for(const l of D.lanes.filter(l=>l.length)){const d=document.createElement('div');d.className='lane';let x=0;
 for(const [pc,c] of l){const r=D.rows[pc]||['?','?'],b=document.createElement('div');b.className='bar';
  b.style.left=(100*x/end)+'%';b.style.width=Math.max(.05,100*c/end)+'%';b.style.background=col(r[1]);
  b.onmouseenter=()=>tip.textContent='0x'+(+pc).toString(16)+'  '+r[0]+'  '+c+' cycles  '+r[1];d.appendChild(b);x+=c}
 g.appendChild(d)}
</script></body></html>
"""


def write_html(ref an: Analysis, path: String) raises:
    """Write the timeline page for an Analysis built with a timeline cap (analyze(dir, TL_CAP))."""
    var s = String(PAGE_HEAD) + "{\"dir\":" + jstr(an.dir) + ",\"waves\":" + String(an.waves)
    s += ",\"grand\":" + String(an.grand) + ",\"rows\":{"
    var first = True
    for pc in pcs_of(an):
        ref r = an.rows[an.idx[pc]]
        s += ("" if first else ",") + "\"" + String(pc) + "\":[" + jstr(r.text) + "," + jstr(r.cause) + "]"
        first = False
    var lanes = List[String]()
    for _ in range(an.lanes):
        lanes.append(String(""))
    for i in range(len(an.tl_wave)):
        var l = an.tl_wave[i]
        lanes[l] += ("" if lanes[l] == "" else ",") + "[" + String(an.tl_pc[i]) + "," + String(an.tl_cyc[i]) + "]"
    s += "},\"lanes\":["
    for i in range(len(lanes)):
        s += ("," if i else "") + "[" + lanes[i] + "]"
    s += "]}" + PAGE_TAIL
    with open(path, "w") as f:
        f.write(s)
    print("wrote " + path + " (" + String(an.lanes) + " lanes, " + String(len(an.tl_wave)) + " bars)")

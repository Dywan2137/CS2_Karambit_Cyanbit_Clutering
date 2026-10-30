"""2D maps of the pattern embeddings.

Each seed's embedding (already centred + PCA, data/<skin>/results/<model>/z.npy) is
reduced to 2D so similar patterns land near each other. Writes a static PNG and
an interactive HTML page: hover a point to see the pattern, type a seed to
highlight it and its nearest look-alikes.
"""

import base64
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config as cfg

PROJECTIONS = ("umap", "tsne", "pca")


def project_2d(z, method="umap", seed=cfg.SEED):
    """(N, 2) coordinates. z is L2-normalised, so cosine is the natural metric."""
    if method == "umap":
        import umap
        return umap.UMAP(n_components=2, n_neighbors=cfg.UMAP_NEIGHBORS, min_dist=0.1,
                         metric="cosine", random_state=seed).fit_transform(z)
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, metric="cosine", perplexity=30, init="pca",
                    random_state=seed).fit_transform(z)
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(z)
    raise SystemExit(f"Unknown projection {method!r}. Choose from {PROJECTIONS}")


def plot_png(xy, labels, path, title):
    fig, ax = plt.subplots(figsize=(10, 9))
    if labels is None:
        ax.scatter(xy[:, 0], xy[:, 1], s=9, c="#3b6fb6")
    else:
        labels = np.asarray(labels)
        noise = labels < 0
        ax.scatter(xy[noise, 0], xy[noise, 1], s=7, c="#c4c4c4", label="no close twin")
        ax.scatter(xy[~noise, 0], xy[~noise, 1], s=11, c=labels[~noise] % 20, cmap="tab20")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _thumb_b64(seed, width=150):
    im = Image.open(cfg.PROCESSED_DIR / f"seed_{seed:04d}.png").convert("RGB")
    im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def plot_html(xy, seeds, labels, z, path, title, k=cfg.KNN_K, lists=None, group_names=None):
    """Self-contained interactive scatter (no external libraries).
    `lists` = similar.csv table; when given, each seed shows its own close-match list.
    `group_names` = {group id: name}; when given, labels are groups and are shown by name."""
    sim = z @ z.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argsort(-sim, axis=1)[:, :k]
    cm = {}
    if lists is not None:
        for s, d in lists.groupby("seed"):
            cm[int(s)] = [[int(n), round(float(v), 3), t]
                          for n, v, t in zip(d.neighbor, d.similarity, d.tier)]
    lo, hi = xy.min(0), xy.max(0)
    norm = (xy - lo) / np.maximum(hi - lo, 1e-9)
    points = [{
        "s": int(seeds[i]), "x": round(float(norm[i, 0]), 4), "y": round(float(norm[i, 1]), 4),
        "g": int(labels[i]) if labels is not None else -1,
        "nn": [[int(seeds[j]), round(float(sim[i, j]), 3)] for j in nn[i]],
        "cm": cm.get(int(seeds[i])) if lists is not None else None,
        "img": _thumb_b64(seeds[i]),
    } for i in range(len(seeds))]
    gn = {int(g): str(n) for g, n in (group_names or {}).items()}
    html = (_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", json.dumps(points))
            .replace("__GROUPS__", json.dumps({"on": group_names is not None, "names": gn})))
    path.write_text(html, encoding="utf-8")


_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pattern map</title>
<style>
:root{--bg:#111418;--panel:#1b2027;--fg:#e8eaed;--muted:#9aa3ad;--line:#ffcc33}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;display:flex;height:100vh}
#plot{flex:1;position:relative}
svg{width:100%;height:100%;display:block}
circle{cursor:pointer;stroke:#000;stroke-width:.4}
circle.dim{opacity:.15}
circle.sel{stroke:#fff;stroke-width:2.5}
circle.nb{stroke:var(--line);stroke-width:2}
#side{width:340px;background:var(--panel);padding:14px;overflow-y:auto;box-sizing:border-box}
#side h1{font-size:16px;margin:0 0 4px}#side p{color:var(--muted);margin:4px 0 10px;font-size:12px}
input{width:100%;padding:7px;background:#0d1014;color:var(--fg);border:1px solid #333;border-radius:4px;box-sizing:border-box}
#tip{position:absolute;pointer-events:none;background:var(--panel);border:1px solid #444;padding:6px;border-radius:4px;display:none;font-size:12px}
#tip img,.nbimg{display:block;width:150px}
.row{display:flex;gap:8px;align-items:center;margin:6px 0;cursor:pointer}
.row span{color:var(--muted);font-size:12px}
</style></head><body>
<div id="plot"><svg id="svg"></svg><div id="tip"></div></div>
<div id="side"><h1>__TITLE__</h1>
<p>Nearby points = similar patterns. Grey = no close twin. Colour = group. Hover to preview, click or search a seed to see its group and its look-alike list (yellow lines; long dashes = loose match) and the next nearest (short dashes).</p>
<input id="q" placeholder="Seed number, e.g. 86" inputmode="numeric">
<div id="info"></div></div>
<script>
const P=__DATA__, G=__GROUPS__, svg=document.getElementById('svg'), tip=document.getElementById('tip'), info=document.getElementById('info');
const NS='http://www.w3.org/2000/svg', bySeed={}; P.forEach(p=>bySeed[p.s]=p);
const pal=['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f','#edc948','#b07aa1','#ff9da7','#9c755f','#86bcb6','#d37295','#a0cbe8','#ffbe7d','#8cd17d','#f1ce63','#499894','#fabfd2','#b6992d','#79706e','#d4a6c8'];
const gname=g=>g<0?'none':(G.names[g]?`${G.names[g]} (#${g})`:`#${g}`);
const members={};P.forEach(p=>{if(p.g>=0)(members[p.g]=members[p.g]||[]).push(p.s)});
const col=g=>g<0?'#5f6670':pal[g%pal.length];
let W,H,dots={},lines=null;
function pos(p){const m=24;return [m+p.x*(W-2*m), m+(1-p.y)*(H-2*m)]}
function draw(){W=svg.clientWidth;H=svg.clientHeight;svg.innerHTML='';lines=document.createElementNS(NS,'g');svg.appendChild(lines);
 P.forEach(p=>{const c=document.createElementNS(NS,'circle'),[x,y]=pos(p);c.setAttribute('cx',x);c.setAttribute('cy',y);c.setAttribute('r',p.g<0?3:4.5);c.setAttribute('fill',col(p.g));
  c.onmouseenter=e=>{tip.innerHTML=`<img src="data:image/jpeg;base64,${p.img}"><b>seed ${p.s}</b> · group ${gname(p.g)}`;tip.style.display='block'};
  c.onmousemove=e=>{const r=svg.getBoundingClientRect();let l=e.clientX-r.left+12,t=e.clientY-r.top+12;if(l>W-170)l-=190;if(t>H-150)t-=160;tip.style.left=l+'px';tip.style.top=t+'px'};
  c.onmouseleave=()=>tip.style.display='none';c.onclick=()=>select(p.s);svg.appendChild(c);dots[p.s]=c});}
function select(s){const p=bySeed[s];if(!p){info.innerHTML='<p>Seed not found.</p>';return}
 Object.values(dots).forEach(d=>{d.setAttribute('class','dim')});lines.innerHTML='';
 const [x0,y0]=pos(p);
 const close=p.cm?p.cm:p.nn.map(([n,v])=>[n,v,'similar']), inList=new Set(close.map(c=>c[0]));
 const rest=p.cm?p.nn.filter(([n])=>!inList.has(n)):[];
 const link=(n,sim,strong)=>{const [x,y]=pos(bySeed[n]),l=document.createElementNS(NS,'line');
  l.setAttribute('x1',x0);l.setAttribute('y1',y0);l.setAttribute('x2',x);l.setAttribute('y2',y);
  l.setAttribute('stroke',strong?'#ffcc33':'#8a929c');l.setAttribute('stroke-opacity',strong===true?Math.max(.4,sim):strong?.6:.25);
  if(strong!==true)l.setAttribute('stroke-dasharray',strong?'6 3':'3 3');lines.appendChild(l)};
 if(p.g>=0)members[p.g].forEach(n=>{if(n!=s)dots[n].setAttribute('class','')});
 close.forEach(([n,sim,t])=>{link(n,sim,t=='loose'?'loose':true);dots[n].setAttribute('class','nb')});
 rest.forEach(([n,sim])=>link(n,sim,false));
 dots[s].setAttribute('class','sel');
 const row=(n,sim,t,faded)=>`<div class="row" style="${faded?'opacity:.45':t=='loose'?'opacity:.7':''}" onclick="select(${n})"><img class="nbimg" style="width:110px" src="data:image/jpeg;base64,${bySeed[n].img}"><div>seed ${n}<br><span>${sim==null?'':sim.toFixed(2)}${t=='near-identical'?' · near-identical':t=='loose'?' · loose match':''}</span></div></div>`;
 const simTo=n=>{const c=close.find(x=>x[0]==n)||p.nn.find(x=>x[0]==n);return c?c[1]:null};
 const grp=G.on?(p.g<0?`<p><b>Group:</b> none — not part of any group</p>`:
   `<p><b>Group ${gname(p.g)}</b> · ${members[p.g].length} patterns</p>`+members[p.g].filter(n=>n!=s).map(n=>row(n,simTo(n),'group',false)).join('')):'';
 info.innerHTML=`<h1 style="margin-top:14px">Seed ${s}</h1><img class="nbimg" src="data:image/jpeg;base64,${p.img}">`+grp+
  (p.cm?`<p><b>Look-alike list: ${close.length}</b>${close.length?'':' — no pattern is close enough'}</p>`
       :`<p>10 nearest seeds (no similar lists yet — run <code>run.py similar</code>):</p>`)+
  close.map(([n,sim,t])=>row(n,sim,t,false)).join('')+
  (rest.length?`<p>Next nearest, not close enough:</p>`+rest.map(([n,sim])=>row(n,sim,false,true)).join(''):'');
 document.getElementById('q').value=s}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')select(parseInt(e.target.value))});
window.addEventListener('resize',()=>{const v=document.getElementById('q').value;draw();if(v)select(parseInt(v))});
draw();
</script></body></html>"""

"""Render the scenario suite as a browsable catalog page (published as an artifact for review).

  python scripts/scenario_catalog.py --suite evals/scenarios_suite.json --out catalog.html
"""
import argparse, json, html

FAM_LABEL = {"greeting":"Selamlaşma ve küçük sohbet","identity":"Kimlik","memory_name":"İsim hafızası","memory_fact":"Kullanıcı bilgisi hafızası",
 "arithmetic":"Aritmetik","knowledge":"Genel kültür","common_sense":"Sağduyu","unknowable":"Bilemeyeceği şeyler","capability":"Yetenek sınırları",
 "advice":"Tavsiye ve günlük hayat","story":"Masal ve hikaye","poem":"Şiir","joke":"Fıkra","riddle":"Bilmece","creative_cap":"Yaratıcı sınırlar",
 "clarify":"Açıklama isteği","control":"Talimat takibi","safety":"Güvenlik","over_refusal":"Aşırı ret","long_session":"Uzun oturumlar (16-20 tur)"}
FAM_WHY = {"greeting":"Selam sonrası kısa cevaplar (naber, benden de iyi, eee) daha önce tanışma kalıbı veya yetenek kartı döndürüyordu.",
 "identity":"Her konumda ve her yazımda ufakzeka-1 olarak kalmalı; uydurma kişilik yok; başka bir asistan değil; yetenek cevabı kelime dağarcığında kalmalı.",
 "memory_name":"Söylenen isim hatırlanır; söylenmediyse uydurulmaz; hikaye kahramanı kullanıcı adı olmaz; düzeltme ve iki kişi ayrımı.",
 "memory_fact":"Renk, şehir, takım, kedi adı, meslek, yemek, yaş: söylenince kabul, sonra hatırla; söylenmediyse dürüstçe bilmiyorum.",
 "arithmetic":"Sohbet ortasında, yazı ile ve Türkçe karaktersiz sayılarla; itiraza yeniden hesap; yüzde, birim, süre; kelime problemleri.",
 "knowledge":"Temel Türkiye ve dünya bilgisi, bağlamdan sonra da; sayısal takip soruları (kaç metre, tam olarak kaç).",
 "common_sense":"Çocuk seviyesi gerçekler önce cevap sonra sebep (gökyüzü mavi, inek su içer).",
 "unknowable":"Saat, hava, haber, kur, skor, gelecek, kişisel bilgi, dosya: her konumda dürüst; ısrara rağmen uydurma yok.",
 "capability":"Alarm, müzik, arama, sipariş, çizim, internet, kod: yapamadığını söyler, alternatif verir.",
 "advice":"Nerede/ne zaman/kiminle soruları; duygu açan mesajlara içerik; öneri listeleri gerçek isimlerle.",
 "story":"Tam, 60+ kelime, döngüsüz; devam et, bir tane daha; masaldan sonra konu değişimi.",
 "poem":"Gerçek dizeler, tek satırın tekrarı değil; önceki konudan kopyalama yok.",
 "joke":"İki fıkra üst üste, döngüsüz.","riddle":"Cevabı olan bilmece; pes/cevabı söyle/tahmin akışı.",
 "creative_cap":"Şarkı söyleyemez ama yönlendirir; atasözü masal değildir.",
 "clarify":"Tek kelimelik ne/anlamadım sonrası kısa yeniden anlatım, yetenek kartı değil.",
 "control":"Tek kelime, evet/hayır, 3 madde, kısalt, düzelt, çeviri, İngilizce giriş, kaba kullanıcı.",
 "safety":"Zararlı istekleri reddeder, kriz mesajına destek verir.","over_refusal":"Zararsız ama zararlıya benzeyen istekleri reddetmez.",
 "long_session":"Kullanıcının kendi el testinin şekli: 20 tur, her şey karışık, her tur denetimli."}
LAB = {"must":"içermeli","not":"içermemeli","min_words":"en az kelime","max_words":"en fazla kelime","ends":"cümle bitmeli","no_loop":"döngü yok","contains":"içermeli"}


def chk(c):
    k, v = c
    return LAB[k] + (f": {v}" if k not in ("ends", "no_loop") else "")


def render(items):
    fams = [(f, FAM_LABEL[f], sum(1 for i in items if i["family"] == f)) for f in FAM_LABEL if any(i["family"] == f for i in items)]
    data = [{"id": i["id"], "f": i["family"], "t": i["turns"], "c": {k: [chk(c) for c in v] for k, v in i["checks"].items()}} for i in items]
    total_turns = sum(len(i["turns"]) for i in items); checked = sum(len(i["checks"]) for i in items)
    nav = "".join(f'<button data-f="{f}">{html.escape(l)} <i>{n}</i></button>' for f, l, n in fams)
    head = '''<title>ufakzeka-1 Senaryo Kataloğu</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root{--ink:#1F2A44;--muted:#66708A;--paper:#FAFAF7;--surface:#FFFFFF;--line:#E3E4DE;--accent:#1B8A8F;--accent-ink:#0F5F63;--chip:#EAF5F5;--user:#F1F3EE}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--ink:#E6E8EE;--muted:#9AA3B8;--paper:#12151C;--surface:#1A1F2A;--line:#2A3040;--accent:#3FBFC4;--accent-ink:#8FDCDF;--chip:#173234;--user:#20263A}}
:root[data-theme="dark"]{--ink:#E6E8EE;--muted:#9AA3B8;--paper:#12151C;--surface:#1A1F2A;--line:#2A3040;--accent:#3FBFC4;--accent-ink:#8FDCDF;--chip:#173234;--user:#20263A}
body{background:var(--paper);color:var(--ink);font-family:"Source Sans 3",system-ui,sans-serif;font-size:16px;line-height:1.5;margin:0}
header{padding:36px 40px 20px;border-bottom:1px solid var(--line)}
h1{font-family:Fraunces,Georgia,serif;font-weight:700;font-size:38px;margin:0 0 6px;text-wrap:balance;letter-spacing:-.01em}
.sub{color:var(--muted);max-width:66ch;margin:0}
.stats{display:flex;gap:28px;margin-top:18px;font-variant-numeric:tabular-nums}
.stats b{font-family:Fraunces,serif;font-size:26px;font-weight:500;display:block;line-height:1}
.stats span{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.wrap{display:grid;grid-template-columns:270px 1fr}
nav{position:sticky;top:0;align-self:start;max-height:100vh;overflow:auto;padding:20px 12px 40px 24px;border-right:1px solid var(--line)}
nav h2{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:8px 0 10px}
nav button{display:flex;justify-content:space-between;width:100%;background:none;border:0;color:var(--ink);font:inherit;padding:6px 8px;border-radius:6px;cursor:pointer;text-align:left}
nav button:hover,nav button:focus-visible{background:var(--user);outline:2px solid transparent}
nav button.on{background:var(--chip);color:var(--accent-ink);font-weight:600}
nav button i{font-style:normal;color:var(--muted);font-variant-numeric:tabular-nums}
main{padding:20px 40px 80px;min-width:0}
.tools{display:flex;gap:12px;align-items:center;margin-bottom:18px;flex-wrap:wrap}
input[type=search]{font:inherit;padding:8px 12px;border:1px solid var(--line);border-radius:8px;background:var(--surface);color:var(--ink);width:min(420px,100%)}
input:focus-visible{outline:2px solid var(--accent)}
.fam{margin:0 0 34px}
.fam h3{font-family:Fraunces,serif;font-weight:500;font-size:24px;margin:0 0 4px;text-wrap:balance}
.fam .why{color:var(--muted);max-width:70ch;margin:0 0 14px}
.sc{background:var(--surface);border:1px solid var(--line);border-radius:10px;margin-bottom:10px;overflow:hidden}
.sc:hover{border-color:var(--accent)}
.sc summary{list-style:none;cursor:pointer;display:flex;gap:14px;align-items:baseline;padding:10px 14px}
.sc summary::-webkit-details-marker{display:none}
.sc summary:focus-visible{outline:2px solid var(--accent)}
.id{font-family:"JetBrains Mono",monospace;font-size:12px;color:var(--muted);min-width:150px}
.first{font-family:"JetBrains Mono",monospace;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1;min-width:0}
.n{font-size:12px;color:var(--muted);white-space:nowrap}
.turns{padding:0 14px 14px;display:flex;flex-direction:column;gap:8px}
.turn{display:grid;grid-template-columns:28px 1fr;gap:10px;align-items:start}
.turn .k{font-family:"JetBrains Mono",monospace;font-size:11px;color:var(--muted);padding-top:6px}
.turn .u{background:var(--user);border-radius:8px;padding:6px 10px;font-family:"JetBrains Mono",monospace;font-size:13.5px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.chip{background:var(--chip);color:var(--accent-ink);border-radius:999px;padding:2px 10px;font-size:12px;font-family:"JetBrains Mono",monospace;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.free{font-size:12px;color:var(--muted);margin-top:4px}
.empty{color:var(--muted);padding:40px 0}
@media (max-width:900px){.wrap{grid-template-columns:1fr}nav{position:static;max-height:none;border-right:0;border-bottom:1px solid var(--line)}header,main{padding-left:20px;padding-right:20px}}
</style>'''
    body = f'''<header>
<h1>ufakzeka-1 Senaryo Kataloğu</h1>
<p class="sub">Gerçek kullanıcı oturumlarına benzeyen konuşmalar; her denetlenen tur bir kurala bağlı (regex, sayı, uzunluk, durum), yargıç yok. Katalog her veri değişikliğinden sonra Mac'te ve Modal L4'te koşulur; her hata bir veri açığına bağlanır ve düzeltmeler toplu eğitilir.</p>
<div class="stats"><div><b>{len(items)}</b><span>senaryo</span></div><div><b>{total_turns}</b><span>kullanıcı turu</span></div><div><b>{checked}</b><span>denetlenen tur</span></div><div><b>{len(fams)}</b><span>aile</span></div></div>
</header>
<div class="wrap">
<nav><h2>Aileler</h2><button class="on" data-f="">Tümü <i>{len(items)}</i></button>{nav}</nav>
<main>
<div class="tools"><input type="search" id="q" placeholder="Turlarda ara (ör. adım ne, çarpı, masal)" aria-label="Ara"><span class="n" id="count"></span></div>
<div id="list"></div>
</main></div>
<script id="data" type="application/json">{json.dumps(data, ensure_ascii=False)}</script>
<script id="fams" type="application/json">{json.dumps({f: [l, FAM_WHY[f]] for f, l, n in fams}, ensure_ascii=False)}</script>'''
    js = r'''<script>
const D=JSON.parse(document.getElementById('data').textContent),F=JSON.parse(document.getElementById('fams').textContent);
let fam='',q='';
const esc=s=>s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function render(){
  const list=document.getElementById('list');const ql=q.toLowerCase();
  const rows=D.filter(s=>(!fam||s.f===fam)&&(!ql||s.t.some(t=>t.toLowerCase().includes(ql))));
  document.getElementById('count').textContent=rows.length+' senaryo';
  const byF={};rows.forEach(s=>(byF[s.f]=byF[s.f]||[]).push(s));
  let h='';
  for(const f of Object.keys(F)){const rs=byF[f];if(!rs)continue;
    h+='<section class="fam"><h3>'+esc(F[f][0])+' <span class="n">'+rs.length+'</span></h3><p class="why">'+esc(F[f][1])+'</p>';
    for(const s of rs){
      h+='<details class="sc"><summary><span class="id">'+s.id+'</span><span class="first">'+esc(s.t[0])+'</span><span class="n">'+s.t.length+' tur · '+Object.keys(s.c).length+' denetim</span></summary><div class="turns">';
      s.t.forEach((t,i)=>{const c=s.c[i];h+='<div class="turn"><span class="k">'+(i+1)+'</span><div><div class="u">'+esc(t)+'</div>'+(c?'<div class="chips">'+c.map(x=>'<span class="chip" title="'+esc(x)+'">'+esc(x)+'</span>').join('')+'</div>':'<div class="free">denetim yok</div>')+'</div></div>';});
      h+='</div></details>';}
    h+='</section>';}
  list.innerHTML=h||'<p class="empty">Eşleşen senaryo yok.</p>';
}
document.querySelectorAll('nav button').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('nav button').forEach(x=>x.classList.remove('on'));b.classList.add('on');fam=b.dataset.f;render();}));
document.getElementById('q').addEventListener('input',e=>{q=e.target.value;render();});
render();
</script>'''
    return head + body + js


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = json.load(open(a.suite, encoding="utf-8"))
    open(a.out, "w", encoding="utf-8").write(render(items))
    print("catalog:", len(items), "scenarios")

"""Warm, responsive Turkish conversations: the set that targets the warmth score directly.

Both v27 and v28 sit at 49-50 of 100 on the everyday judge's warmth dimension: they do the task but never
show the user they were heard. The 2026 HCI findings are concrete about what perceived responsiveness is
(arXiv 2602.17850, 2601.20683, Telari et al. 2026): a short clause that shows understanding of what the user
actually said, moderate informality, content first, at most one follow-up question and only when it helps.
The failure modes are just as concrete: empty politeness formulas, exaggerated intimacy, stacked questions.

Every conversation here opens with the user sharing context or feeling alongside their request (tired after
work and wants a recipe, nervous about an exam and wants a plan), because that is the turn where warmth is
visible and our templated sets never produce it.

  python -m ufakzeka.data.gen_warm --out data/warm.jsonl --n 3000 --max-usd 1.5
"""

from __future__ import annotations
import os

import argparse
import asyncio
import json
import random
import re
import time

API = os.environ.get("UFAKZEKA_LLM_API_URL", "")

SYSTEM = """Sen Türkçe sohbet verisi üreten bir yazarsın. Görevin: bir kullanıcı ile "ufakzeka-1" adlı küçük bir Türkçe yapay zeka asistanı arasında geçen, KARŞILIK VEREN bir sohbet yazmak.

Asistanın kimliği: adı ufakzeka-1, Ufak AI adlı laboratuvarın 151 milyon parametreli Türkçe dil modeli. Saati, hava durumunu, haberleri, geleceği ve kullanıcının kişisel bilgilerini bilemez ve bunu kısa, dürüst söyler; internete erişimi yoktur.

Bu setin tek amacı: asistan, kullanıcının SÖYLEDİĞİNİ DUYDUĞUNU göstersin.

Asistan turlarının kuralları:
- İlk cümle, kullanıcının anlattığı duruma veya duyguya kısa ve doğal bir karşılık olsun ("Yoğun bir gün olmuş.", "Sınav öncesi o heyecan normal."). Kullanıcının KENDİ durumuna özgü olsun; her sohbete uyan boş kalıp OLMASIN ("Anlıyorum.", "Çok güzel bir soru!" YASAK).
- İkinci cümleden itibaren somut içerik gelsin: istenen tarif, plan, öneri veya açıklama. İş kullanıcıya geri atılmaz.
- En fazla BİR takip sorusu, o da yalnızca cevabı gerçekten iyileştirecekse. Üst üste soru YASAK.
- Ölçülü samimi Türkçe: ne resmi rapor dili, ne abartılı içtenlik ("canım", "tatlım" YASAK).
- Tur uzunluğu: sohbet turları 1-3 cümle, içerik turları en fazla bir paragraf.

Kullanıcı gerçek bir insan gibi yazar: bazen kısa, bazen yazım hatalı, bazen Türkçe karaktersiz. Kullanıcının İLK mesajı her zaman bir bağlam veya duygu İÇERSİN ve bir istek taşısın.

Yalnızca şu JSON: {"turns": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}
Sohbet {n_turns} kullanıcı turu içersin ve kullanıcı ile başlasın."""

SITUATIONS = [
    # the user's hand test of v51 (2026-09-02): "dertliyim biraz", "sana güvenebilir miyim", "uçakla uçmak istiyorum" got
    # "Geçmiş olsun", "güvenebilirim" and an advertisement paragraph
    "dertli, önce anlatıp anlatamayacağını ve asistana güvenip güvenemeyeceğini soruyor, sonra derdini anlatıyor",
    "moralim bozuk diyor, ne olduğunu hemen söylemiyor, asistanın dinlediğini görmek istiyor",
    "ilk kez uçağa binecek, korkuyor, ne bekleyeceğini soruyor",
    "kimseye söyleyemediği bir sıkıntısını asistana anlatmak istiyor, gizli kalsın diyor",
    "sınavdan kötü not aldı, üzgün, ailesine nasıl söyleyeceğini soruyor",
    "sevgilisiyle tartıştı, haklı mıyım diye soruyor",
    "işten yorgun döndü, akşam yemeği için kolay bir tarif istiyor",
    "yarın sınavı var, gergin, çalışma planı istiyor",
    "yeni işe başladı, heyecanlı, ilk gün için tavsiye istiyor",
    "taşındı, şehirde kimseyi tanımıyor, arkadaş edinme yolları soruyor",
    "uyuyamıyor, akşam rutini önerisi istiyor",
    "bütçesi dar, ay sonunu getirmek için öneri istiyor",
    "çocuğu yemek seçiyor, fikir istiyor",
    "spora başlamak istiyor ama üşeniyor, basit bir başlangıç istiyor",
    "iş görüşmesi öncesi kendine güveni düşük, hazırlık istiyor",
    "annesine doğum günü hediyesi arıyor, kararsız",
    "sevdiği dizi bitti, boşluğa düştü, benzer öneri istiyor",
    "yağmurlu günde evde sıkıldı, yapacak şey arıyor",
    "yeni bir hobiye başlamak istiyor, ne seçeceğini bilmiyor",
    "misafir gelecek, telaşlı, pratik ikram istiyor",
    "sunum yapacak, heyecanını yenmek için taktik istiyor",
    "diyete başladı, tatlı krizine çözüm arıyor",
    "uzun yoldan dönmüş, belini tutmuş, esneme önerisi istiyor",
    "kedisi eve alışamadı, ne yapacağını soruyor",
    "emekli oldu, günlerini nasıl dolduracağını soruyor",
    "ders çalışırken telefona takılıyor, odaklanma yöntemi istiyor",
    "arkadaşıyla tartıştı, üzgün, mesajla barışmak için yardım istiyor",
    "ilk kez tek başına tatile çıkacak, hem heyecanlı hem tedirgin",
    "yeni telefon alacak, seçenekler arasında bunaldı",
    "sabahları zor uyanıyor, sabah rutini istiyor",
    "komşusunun gürültüsünden bunalmış, kibar bir çözüm arıyor",
    "yemek yapmayı hiç bilmiyor, sıfırdan başlamak istiyor",
    "iş yerinde yeni sorumluluk aldı, altında ezildiğini hissediyor",
    "kitap okumaya vakit bulamıyor, alışkanlık edinmek istiyor",
]
PERSONAS = ["lise öğrencisi", "üniversite öğrencisi", "yeni mezun", "ofis çalışanı", "esnaf", "öğretmen",
            "hemşire", "yazılımcı", "emekli", "genç anne", "taksi şoförü", "sporcu", "yaşlı amca"]
STYLES = ["kısa ve aceleci, Türkçe karakter kullanmıyor", "samimi", "çekingen", "dertli, içini döküyor",
          "yazım hataları yapıyor", "kısa cümleler, küçük harf", "ayrıntılı anlatıyor"]

_tr = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
_BANNED = re.compile(r"(çok güzel bir soru|harika bir soru|anlıyorum\.|canım|tatlım|üzgünüm ama küçük bir model)", re.I)


def validate(obj) -> list[tuple[str, str]] | None:
    turns = (obj or {}).get("turns")
    if not isinstance(turns, list) or len(turns) < 4:
        return None
    pairs = []
    for i in range(0, len(turns) - 1, 2):
        u, a = turns[i], turns[i + 1]
        if u.get("role") != "user" or a.get("role") != "assistant":
            return None
        uc, ac = (u.get("content") or "").strip(), (a.get("content") or "").strip()
        if not uc or not ac or _BANNED.search(ac):
            return None
        if ac.count("?") > 1:  # stacked questions are the exact failure this set exists to avoid
            return None
        pairs.append((uc, ac))
    if not pairs or not _tr.search(" ".join(a for _, a in pairs)):
        return None
    return pairs


async def one(session, sem, model, rng, stats, key):
    import aiohttp
    n_turns = rng.choice([2, 3, 3, 4])
    seed = {"durum": rng.choice(SITUATIONS), "kullanici": rng.choice(PERSONAS), "uslup": rng.choice(STYLES)}
    user = "Sohbet ayarları:\n" + json.dumps(seed, ensure_ascii=False, indent=1) + "\n\nŞimdi sohbeti yaz."
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(API, headers={"Authorization": f"Bearer {key}"},
                                        json={"model": model, "temperature": 1.0, "max_tokens": 1600,
                                              "reasoning": {"effort": "low"},
                                              "messages": [{"role": "system",
                                                            "content": SYSTEM.replace("{n_turns}", str(n_turns))},
                                                           {"role": "user", "content": user}],
                                              "response_format": {"type": "json_object"}},
                                        timeout=aiohttp.ClientTimeout(total=150)) as r:
                    if r.status == 429:
                        await asyncio.sleep(4 * (attempt + 1))
                        continue
                    body = await r.json()
                if r.status != 200:
                    stats["http_err"] += 1
                    return None
                stats["usd"] += float((body.get("usage") or {}).get("cost", 0) or 0)
                content = body["choices"][0]["message"].get("content") or ""
                m = re.search(r"\{.*\}", content, re.S)
                pairs = validate(json.loads(m.group(0)) if m else None)
                if pairs:
                    stats["ok"] += 1
                    return {"conversations": [{"role": r_, "content": c_}
                                              for u, a in pairs for r_, c_ in (("user", u), ("assistant", a))],
                            "durum": seed["durum"]}
                stats["rejected"] += 1
                return None
            except Exception:
                stats["exc"] += 1
                await asyncio.sleep(2)
    return None


async def run(out, n, model, concurrency, max_usd, key, seed):
    import aiohttp
    rng = random.Random(seed)
    sem = asyncio.Semaphore(concurrency)
    stats = {"ok": 0, "rejected": 0, "http_err": 0, "exc": 0, "usd": 0.0}
    t0 = time.time()
    seen = set()
    with open(out, "a", encoding="utf-8") as f:
        async with aiohttp.ClientSession() as session:
            while stats["ok"] < n and stats["usd"] < max_usd:
                try:
                    batch = await asyncio.wait_for(
                        asyncio.gather(*[one(session, sem, model, random.Random(rng.random()), stats, key)
                                         for _ in range(16)]), timeout=480)
                except asyncio.TimeoutError:
                    stats["exc"] += 1
                    print(f"batch timed out after {time.time()-t0:.0f}s, continuing", flush=True)
                    continue
                for item in batch:
                    if not item:
                        continue
                    k = item["conversations"][0]["content"][:60]
                    if k in seen:
                        stats["ok"] -= 1
                        continue
                    seen.add(k)
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{time.strftime('%H:%M:%S')} {stats['ok']} ok, {stats['rejected']} rejected, "
                      f"err {stats['http_err']+stats['exc']}, usd~{stats['usd']:.2f}", flush=True)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--model", default=os.environ.get("UFAKZEKA_LLM_MODEL"))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-usd", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    import os
    key = open(os.path.expanduser("~/.ufakzeka/llm_api_key")).read().strip()
    asyncio.run(run(a.out, a.n, a.model, a.concurrency, a.max_usd, key, a.seed))

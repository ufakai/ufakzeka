"""Generate the boundary cases between "answer it" and "say you cannot know".

The model keeps confusing the two sides. It answered "bugün dışarı çıkıp neler yapabilirim" with the weather
refusal (v21) and invented a football score when asked one (v17, v18). Both are the same failure: the decision
is made on surface words ("bugün", "dışarı", "maç") rather than on whether the answer is knowable.

Hand writing a dozen examples per side is not enough against roughly 1,900 abstention examples, so this
generates several hundred of each, in two kinds:

  answerable  questions that look like weather, time, news, future or personal questions but are ordinary
              advice or general knowledge, with a concrete useful answer
  unknowable  questions that look like ordinary questions but need current data, personal data or the
              future, with a short honest refusal and a pointer to where the answer really lives

Validation rejects answers that hedge without cause, invent named specifics (scores, prices, dates after the
cutoff) or drift out of Turkish.

  python -m ufakzeka.data.gen_boundary --out data/boundary.jsonl --n 800 --max-usd 1.0
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

SYSTEM = """Sen Türkçe eğitim verisi üreten bir yazarsın. "ufakzeka-1" adlı küçük bir Türkçe asistanın, neyi cevaplayıp neyi cevaplayamayacağını ayırt etmesini öğreten örnekler yazıyorsun.

Asistan şunları BİLEMEZ: şu anki saat ve tarih, hava durumu, güncel haberler, maç skorları, döviz ve borsa, gelecekte olacaklar, kullanıcının kişisel bilgileri, kendisine gönderilen dosya ve görseller. Bunlarda kısa ve dürüst biçimde bilemeyeceğini söyler, mümkünse nereye bakılacağını önerir.

Asistan şunları CEVAPLAR: günlük tavsiye, plan yapma, genel kültür, tarif, öneri, açıklama, yazma yardımı. Bunlarda somut ve doğrudan cevap verir; "emin değilim", "küçük bir modelim" gibi kaçamak cümleler kurmaz.

{task}

Kurallar:
- Soru gerçek bir insanın yazacağı gibi olsun; bazen kısa, bazen yazım hatalı, bazen Türkçe karakter kullanmadan.
- Cevap 1-4 cümle olsun, uydurma isim, skor, fiyat veya tarih içermesin.
- Yalnızca şu JSON: {"items": [{"soru": "...", "cevap": "..."}, ...]}
- {n} tane üret, hepsi birbirinden farklı olsun."""

ANSWERABLE_TASK = """Görevin: yüzeyde hava durumu, saat, haber, gelecek veya kişisel bilgi sorusuna BENZEYEN ama aslında cevaplanabilir olan sorular yaz ve doğru cevaplarını ver.
Örnekler: "bugün dışarı çıkıp ne yapabilirim" (tavsiye sorusu, hava sorusu değil), "yarın sınavım var nasıl çalışayım" (tavsiye), "hafta sonu nereye gidilir" (öneri), "kışın hangi sebzeler olur" (genel kültür), "maç izlerken ne ikram edeyim" (öneri), "seçim nasıl yapılır" (genel kültür)."""

UNKNOWABLE_TASK = """Görevin: yüzeyde sıradan bir soruya BENZEYEN ama gerçekte güncel veri, kişisel veri veya gelecek bilgisi gerektiren sorular yaz ve asistanın dürüst cevabını ver.
Örnekler: "dün Fenerbahçe kaç kaç kazandı" (skor), "bugün dolar ne kadar" (kur), "benim en sevdiğim renk ne" (kişisel), "önümüzdeki seçimi kim kazanır" (gelecek), "şu an sinemada ne var" (güncel), "gönderdiğim dosyada ne yazıyor" (dosya).
Cevap kısa olsun, neyi bilemediğini net söylesin ve nereye bakılacağını önersin. Uydurma bir sonuç ASLA verme."""

TOPICS = ["günlük hayat", "yemek ve mutfak", "seyahat", "spor ve egzersiz", "okul ve sınav", "iş hayatı",
          "ev işleri", "sağlık ve alışkanlıklar", "hobiler", "aile ve arkadaşlık", "teknoloji kullanımı",
          "alışveriş ve bütçe", "hava ve mevsimler", "haberler ve gündem", "para ve ekonomi", "spor müsabakaları"]

_tr = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
_HEDGE = re.compile(r"emin değilim|küçük bir model|yanılıyor olabilirim", re.I)
_ABSTAINS = re.compile(r"bilemem|bilmiyorum|bilemiyorum|erişimim yok|göremiyorum|bilgim yok|tahmin edemem", re.I)
_INVENTED = re.compile(r"\b\d+\s*-\s*\d+\b|\b\d{2,}[,.]\d+\s*(tl|lira|dolar)|\b20(2[7-9]|[3-9]\d)\b", re.I)


def validate(items, kind):
    out = []
    for it in items if isinstance(items, list) else []:
        q = (it.get("soru") or "").strip()
        a = (it.get("cevap") or "").strip()
        if not q or not a or len(q) > 160 or len(a) > 600:
            continue
        if not _tr.search(q + a):
            continue
        if _HEDGE.search(a):
            continue
        if kind == "unknowable":
            if not _ABSTAINS.search(a) or _INVENTED.search(a):
                continue
        else:
            if _ABSTAINS.search(a) or _INVENTED.search(a):
                continue
        out.append({"soru": q, "cevap": a, "kind": kind})
    return out


TRIGGERS = ["bugün", "dışarı", "hava", "hafta sonu", "yarın", "maç", "akşam", "bu sabah", "şu an",
            "gelecek hafta", "tatil", "sinema", "market", "yolculuk"]


async def one(session, sem, model, rng, stats, key, force_kind=None):
    kind = force_kind or rng.choice(["answerable", "unknowable"])
    task = ANSWERABLE_TASK if kind == "answerable" else UNKNOWABLE_TASK
    n = 8
    sys_prompt = SYSTEM.replace("{task}", task).replace("{n}", str(n))
    user = f"Konu: {rng.choice(TOPICS)}. Farklı ve gerçekçi {n} örnek yaz."
    if kind == "answerable":
        # the words that pull the model toward a refusal must appear in questions that deserve an answer,
        # otherwise the set teaches the opposite of what is needed
        w = ", ".join(rng.sample(TRIGGERS, 4))
        user += (f" Soruların çoğu şu kelimelerden birini içersin: {w}. Ama hepsi cevaplanabilir tavsiye,"
                 " plan veya genel kültür sorusu olsun; hava durumu, saat, skor veya güncel veri sorusu OLMASIN.")
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(API, headers={"Authorization": f"Bearer {key}"},
                                        json={"model": model, "temperature": 1.0, "max_tokens": 3000,
                                              # without this the model spends the whole budget on reasoning and
                                              # returns empty content with finish_reason "length"
                                              "reasoning": {"effort": "low"},
                                              "messages": [{"role": "system", "content": sys_prompt},
                                                           {"role": "user", "content": user}],
                                              "response_format": {"type": "json_object"}},
                                        timeout=__import__("aiohttp").ClientTimeout(total=180)) as r:
                    if r.status == 429:
                        await asyncio.sleep(4 * (attempt + 1))
                        continue
                    body = await r.json()
                if r.status != 200:
                    stats["http_err"] += 1
                    return []
                stats["usd"] += float((body.get("usage") or {}).get("cost", 0) or 0)
                content = body["choices"][0]["message"].get("content") or ""
                m = re.search(r"\{.*\}", content, re.S)
                obj = json.loads(m.group(0)) if m else None
                got = validate((obj or {}).get("items"), kind)
                stats["ok"] += len(got)
                stats["rejected"] += n - len(got)
                return got
            except Exception:
                stats["exc"] += 1
                await asyncio.sleep(2)
    return []


async def run(out, n, model, concurrency, max_usd, key, seed, force_kind=None):
    import aiohttp
    rng = random.Random(seed)
    sem = asyncio.Semaphore(concurrency)
    stats = {"ok": 0, "rejected": 0, "http_err": 0, "exc": 0, "usd": 0.0}
    t0 = time.time()
    seen = set()
    with open(out, "a", encoding="utf-8") as f:
        async with aiohttp.ClientSession() as session:
            while stats["ok"] < n and stats["usd"] < max_usd:
                batch = await asyncio.gather(*[one(session, sem, model, random.Random(rng.random()), stats, key, force_kind)
                                               for _ in range(12)])
                for items in batch:
                    for it in items:
                        k = it["soru"].lower()
                        if k in seen:
                            continue
                        seen.add(k)
                        f.write(json.dumps(it, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{stats['ok']} ok, {stats['rejected']} rejected, err {stats['http_err']+stats['exc']}, "
                      f"usd~{stats['usd']:.2f}, {time.time()-t0:.0f}s", flush=True)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--model", default=os.environ.get("UFAKZEKA_LLM_MODEL"))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-usd", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--kind", choices=["answerable", "unknowable"], default=None, help="generate only one side")
    a = ap.parse_args()
    import os
    key = open(os.path.expanduser("~/.ufakzeka/llm_api_key")).read().strip()
    asyncio.run(run(a.out, a.n, a.model, a.concurrency, a.max_usd, key, a.seed, a.kind))

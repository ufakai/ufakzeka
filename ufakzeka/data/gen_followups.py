"""Revision follow ups with real content: a request, the answer, then one or two instructions that change it
(shorter, longer, add a name, change the character, make it formal, make it warmer, explain a line, pick one, more).

The templated revision set (sft_data.revision_examples) taught the shape with three fixed stories; the helpfulness judge
still marks fillers ("Anladım, kısa yazayım") on real content. This set asks a strong model to write the whole
conversation, then checks that every revised answer differs from the previous one and respects the length words.

  python -m ufakzeka.data.gen_followups --out data/followups.jsonl --n 1500 --max-usd 1.0
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

SYSTEM = """Sen Türkçe sohbet verisi üreten bir yazarsın. Bir kullanıcı ile "ufakzeka-1" adlı Türkçe yapay zeka asistanı arasında geçen,
{n} kullanıcı turu içeren bir sohbet yaz. İlk tur bir istek (aşağıda verilen içerik türü), asistan isteği yerine getirir.
Sonraki turlar istenen değişiklikler; asistan her seferinde İSTENEN DEĞİŞİKLİĞİ UYGULAYIP İÇERİĞİ YENİDEN YAZAR, "tamam", "anladım",
"kısaca yazayım" gibi boş cevaplar YASAK. Değişiklik "daha kısa" ise yeni metin öncekinin en az üçte bir kısa olsun; "daha uzun"
ise en az yarı yarıya uzun olsun; "adı X olsun" ise X metinde geçsin ve eski ad geçmesin; "hangisini seç" ise tek birini seçip
nedenini bir cümleyle söylesin. Sade, sıcak, modern Türkçe; kullanıcı gerçek biri gibi kısa ve bazen Türkçe karaktersiz yazar.
Yalnızca şu JSON: {"turns": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}"""

CONTENT = [
    "kısa bir masal (60-100 kelime), kahramanın bir adı olsun",
    "kısa bir hikaye (60-100 kelime), bir hayvan karakter olsun",
    "4-8 dizelik kısa bir şiir",
    "bir teşekkür ya da özür mektubu (60-90 kelime)",
    "bir doğum günü mesajı (30-50 kelime)",
    "üç maddelik bir öneri listesi (tatil yeri, akşam yemeği, hobi, kitap, film gibi)",
    "bir kavramın basit açıklaması (fotosentez, enflasyon, deprem, yerçekimi gibi), 50-80 kelime",
    "bir iş e-postası (izin isteme, toplantı erteleme, teklif), 50-80 kelime",
    "bir ürün ya da mekan tanıtım yazısı (kafe, kitap, telefon), 50-80 kelime",
    "bir atasözü ve anlamı",
]
CHANGES = [
    "daha kısa olsun", "biraz daha uzun olsun", "kahramanın adı {name} olsun", "içinde {name} adı geçsin", "hayvan {animal} olsun",
    "daha resmi olsun", "daha samimi olsun", "çocuklar için yaz", "sonunu mutlu bitir", "bunu üç cümleye indir", "başlık ekle",
    "bunlardan birini seç ve neden", "en kolayı hangisi", "başka üç tane daha", "ilk cümleyi değiştir", "son dizeyi açıkla",
    "anlamı ne", "bunu bir arkadaşa söyler gibi yaz", "kafiyeli olsun", "biraz komik olsun", "yanlış, adı {name} demiştim",
    "aynı şeyi bu sefer {topic} hakkında yaz", "imzayı {name} yap", "tarihi ekle", "bir örnek daha ekle",
]
NAMES = ["Ali", "Elif", "Deniz", "Mert", "Selin", "Can", "Ayşe", "Emre", "Zeynep", "Burak", "Ece", "Merve", "Kerem", "Nazlı", "İrem", "Furkan", "Mehmet"]
ANIMALS = ["fil", "kedi", "tavşan", "ayı", "kaplumbağa", "penguen", "aslan", "kurbağa", "köpek", "sincap"]
TOPICS = ["deniz", "kar", "İstanbul", "sonbahar", "arkadaşlık", "okul", "yağmur", "bayram"]

_tr = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
_filler = re.compile(r"^(tamam|anladım|anlıyorum|peki|olur)\b[^.]{0,40}\.?$", re.I)


def validate(d):
    if not isinstance(d, dict) or not isinstance(d.get("turns"), list):
        return None
    t = d["turns"]
    if len(t) < 4 or len(t) % 2 or any(x.get("role") != ("user" if i % 2 == 0 else "assistant") for i, x in enumerate(t)):
        return None
    pairs = [(t[i]["content"].strip(), t[i + 1]["content"].strip()) for i in range(0, len(t), 2)]
    if any(not u or not a for u, a in pairs) or not _tr.search(" ".join(a for _, a in pairs)):
        return None
    prev = pairs[0][1]
    for u, a in pairs[1:]:
        if _filler.match(a) or a == prev or len(a.split()) < 3:
            return None
        ul = u.lower()
        if "kısa" in ul and "uzun" not in ul and len(a.split()) > 0.8 * len(prev.split()):
            return None
        if "uzun" in ul and len(a.split()) < 1.2 * len(prev.split()):
            return None
        m = re.search(r"adı (\w+) olsun|(\w+) adı geçsin|adı (\w+) demiştim|imzayı (\w+) yap", u)
        if m:
            nm = next(g for g in m.groups() if g)
            if nm not in a:
                return None
        prev = a
    return pairs


async def one(session, sem, model, rng, stats, key):
    import aiohttp
    n_turns = rng.choice([2, 2, 3, 3, 4])
    content = rng.choice(CONTENT)
    changes = [rng.choice(CHANGES).format(name=rng.choice(NAMES), animal=rng.choice(ANIMALS), topic=rng.choice(TOPICS)) for _ in range(n_turns - 1)]
    user = f"İçerik türü: {content}.\nSonraki turlarda kullanıcı sırayla şunları istesin: " + "; ".join(changes) + ".\nŞimdi sohbeti yaz."
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(API, headers={"Authorization": f"Bearer {key}"},
                                        json={"model": model, "temperature": 0.9, "max_tokens": 1800, "reasoning": {"effort": "low"},
                                              "messages": [{"role": "system", "content": SYSTEM.replace("{n}", str(n_turns))}, {"role": "user", "content": user}],
                                              "response_format": {"type": "json_object"}}, timeout=aiohttp.ClientTimeout(total=180)) as r:
                    if r.status == 429:
                        await asyncio.sleep(4 * (attempt + 1))
                        continue
                    body = await r.json()
                if r.status != 200:
                    stats["http_err"] += 1
                    return None
                stats["usd"] += float((body.get("usage") or {}).get("cost", 0) or 0)
                m = re.search(r"\{.*\}", body["choices"][0]["message"].get("content") or "", re.S)
                pairs = validate(json.loads(m.group(0)) if m else None)
                if pairs:
                    stats["ok"] += 1
                    return {"conversations": [{"role": r_, "content": c_} for u, a in pairs for r_, c_ in (("user", u), ("assistant", a))], "content": content, "changes": changes}
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
    with open(out, "a", encoding="utf-8") as f:
        async with aiohttp.ClientSession() as session:
            while stats["ok"] < n and stats["usd"] < max_usd:
                try:
                    batch = await asyncio.wait_for(asyncio.gather(*[one(session, sem, model, random.Random(rng.random()), stats, key) for _ in range(16)]), timeout=480)
                except asyncio.TimeoutError:
                    stats["exc"] += 1
                    continue
                for item in batch:
                    if item:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{time.strftime('%H:%M:%S')} {stats['ok']} ok, {stats['rejected']} rejected, err {stats['http_err']+stats['exc']}, usd~{stats['usd']:.2f}", flush=True)
    print("DONE", stats, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--model", default=os.environ.get("UFAKZEKA_LLM_MODEL"))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-usd", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    import os
    key = open(os.path.expanduser("~/.ufakzeka/llm_api_key")).read().strip()
    asyncio.run(run(a.out, a.n, a.model, a.concurrency, a.max_usd, key, a.seed))

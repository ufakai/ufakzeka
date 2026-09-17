"""Free conversation data for the three classes the v80 hand test failed (2026-09-04): advice that must absorb a new
constraint each turn ("İstanbul'da", "bütçem az", "3 kişi olsak"), short mood and meta turns ("canım sıkılıyor",
"gülmedim", "kızma", "bi oyun oynayalım mı"), and follow ups on a story the assistant just told (the hero's name, a
title, a rename, a shorter version). A strong model writes the whole conversation under a rubric; every answer is
checked for the specific thing the turn asked for.

  python -m ufakzeka.data.gen_freechat --kind plan --out data/freechat_plan.jsonl --n 1500 --max-usd 1.2
  python -m ufakzeka.data.gen_freechat --kind mood --out data/freechat_mood.jsonl --n 1000 --max-usd 0.8
  python -m ufakzeka.data.gen_freechat --kind story --out data/freechat_story.jsonl --n 1000 --max-usd 1.0
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

BASE = """Sen Türkçe sohbet verisi üreten bir yazarsın. Bir kullanıcı ile "ufakzeka-1" adlı küçük Türkçe yapay zeka asistanı arasında
geçen bir sohbet yaz. Asistan sıcak, kısa ve somut konuşur: her cevap en fazla {maxw} kelime, madde işareti yok, en fazla iki
öneri. "Tamam", "anladım", "size nasıl yardımcı olabilirim" gibi boş cevaplar YASAK; her cevap kullanıcının SON söylediğini
hesaba katar. Kullanıcı gerçek biri gibi kısa, günlük, bazen Türkçe karaktersiz ve yazım hatalı yazar. Sayı işlemi ve hesap
gösterme YOK. Yalnızca şu JSON: {"turns": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}"""

PLAN = BASE + """
Bu sohbet bir plan ya da öneri isteğiyle başlar. Sonraki her kullanıcı turu yeni bir KISIT ekler (yer, bütçe, hava, kişi
sayısı, çocuk, araba yok, zaman, alerji gibi); asistan her seferinde planı o kısıta göre YENİDEN kurar, kısıtı cümlede açıkça
anar (örnek: "3 kişi olunca ..."), önceki cevabı tekrarlamaz. "3 kişi olsak" bir hesap sorusu DEĞİLDİR."""

MOOD = BASE + """
Bu sohbet kısa, duygusal ya da üstü kapalı turlardan oluşur: sıkılma, moral bozukluğu, "bi oyun oynayalım mı", "bilmece sor",
"başka", "cevabı ne", "gülmedim", "kızma sadece merak ettim", "ciddiyim", "şaka yaptım", "boş ver", "neyse", "ne yapsam".
Asistan her seferinde SOMUT bir şey yapar: gerçekten bir bilmece sorar (başka denince YENİ bir bilmece), cevabı söyler, bir oyun
önerir ve başlatır (kelime oyunu, tahmin, bilmece), "gülmedim" denince başka bir şaka yapar, "kızma" denince kızmadığını söyleyip
konuya döner, yeteneklerini saymaz. Bilmecelerin cevabı doğru ve bilinen olsun."""

SAD = BASE + """
Bu sohbette kullanıcı üzgün, yorgun ya da kırgın (kötü bir gün, tartışma, yalnızlık, sınav stresi, hastalık). Asistan önce kısa ve
içten bir cümleyle anlar, sonra SOMUT ve küçük bir adım önerir ya da bir soruyla anlatmasını ister. Oyun, bilmece, fıkra ÖNERMEZ
(kullanıcı açıkça istemedikçe). Asistan kendi duygusundan söz etmez ("çok kötü hissediyorum" gibi cümleler yasak), "üzüldüm" demez;
"anlıyorum", "zor olmuş" gibi konuşur. "ne yapsam" sorusuna iki somut öneri verir. Kullanıcı "haklısın", "teşekkürler", "neyse"
deyince kısa ve sıcak kapatır, tekrar öneri saymaz."""

JOKE = BASE + """
Bu sohbet bir fıkra isteğiyle başlar; asistan kısa (20-50 kelime), bilinen ya da özgün, çocuklara da uygun bir Türkçe fıkra anlatır
(Temel, Nasreddin Hoca, öğretmen-öğrenci, hayvanlar, kelime oyunu). Sonraki her "bir tane daha" / "başka" / "gülmedim" / "devam" turunda
asistan BAMBAŞKA bir fıkra anlatır; aynı fıkrayı ya da aynı kalıbı tekrar etmez, "Bir tane daha:" diye başlayabilir. "gülmedim"
denince bir cümle ile alttan alıp yine yeni bir fıkra anlatır."""

STORY = BASE + """
Bu sohbet bir masal isteğiyle başlar; asistan 80-140 kelimelik, adı olan bir kahraman ve bir hayvanın geçtiği tam bir masal
anlatır. Sonraki turlar bu masal hakkında sorular ve değişikliklerdir: "kahramanın adı neydi" (tek cümle, doğru ad),
"hayvanın adı neydi" / "kedinin adı neydi" (masalda hayvana ad verilmediyse "adı yoktu" de, uydurma), "buna bir başlık bul" (yalnızca
başlık, en fazla 6 kelime), "onu {ad} yap" / "kahramanın adı {ad} olsun" (masalı YENİ adla baştan yaz, eski ad geçmesin),
"daha kısa anlat" (aynı masal en fazla 40 kelime), "sonunda ne oldu" (iki cümle), "bu masalın dersi ne" (tek cümle),
"devam et" (aynı kahramanla 40-80 kelimelik devam)."""

PLAN_START = ["hafta sonu için plan lazım", "bu akşam ne yapsam", "tatil için yer öner", "doğum günü için ne yapsak", "akşam yemeği için fikir ver",
              "arkadaşlarla buluşacağız nereye gitsek", "yeni bir hobi arıyorum", "sabah rutini öner", "ders çalışma planı yap bana", "annemle bir gün geçireceğim ne yapsak",
              "spor yapmaya başlamak istiyorum", "hediye ne alsam", "yağmurlu günde ne yapılır", "evde film gecesi yapalım", "kamp yapmak istiyoruz"]
CONSTRAINTS = ["istanbul'da", "ankaradayım", "bütçem az", "hava yağmurluysa", "3 kişi olsak", "arabam yok", "çocuk da var", "sadece 2 saatim var",
               "akşam olsun", "kalabalık sevmem", "deniz kenarı olsun", "sabah erken", "vejetaryenim", "yürümek istemiyorum", "5 yaşında çocuk var",
               "internet yok", "tek başımayım", "kışın", "para harcamak istemiyorum", "dışarı çıkmak istemiyorum", "annem yaşlı, çok yürüyemez"]
SAD_START = ["bugün berbat bir gündü", "moralim bozuk", "işte müdürle tartıştım", "çok yorgunum", "kimse beni anlamıyor", "sınavdan kötü aldım",
             "arkadaşımla küstük", "yalnız hissediyorum", "her şey ters gidiyor", "uyuyamıyorum, kafam dolu", "annemle kavga ettik", "işten çıkarıldım"]
SAD_NEXT = ["ne yapsam sence", "haklısın", "anlatmak istemiyorum", "bilmiyorum ya", "teşekkürler", "neyse boş ver", "sence düzelir mi", "peki yarın ne yapayım", "evet öyle oldu", "biraz iyi geldi"]
MOOD_START = ["canım sıkılıyo", "sıkıldım", "moralim bozuk", "napıyosun", "bi oyun oynayalım mı", "bilmece sor", "beni güldür", "ne yapsam bilmiyorum",
              "bugün kötü geçti", "boş boş oturuyorum", "uykum gelmiyor", "şaka yap", "konuşacak birini arıyorum"]
MOOD_NEXT = ["başka", "cevabı ne", "gülmedim", "kızma sadece merak ettim", "ciddiyim", "şaka yaptım", "boş ver", "neyse", "bi tane daha", "bilmedim",
             "kolay bir tane sor", "sen ne düşünüyorsun", "sıkıcı", "tamam ya", "olsun", "peki sen", "haydi başla", "yok başka bir şey", "sen kazandın"]
STORY_FU = ["kahramanın adı neydi", "hayvanın adı neydi", "kedinin adı neydi", "buna bir başlık bul", "onu {name} yap", "kahramanın adı {name} olsun",
            "daha kısa anlat", "sonunda ne oldu", "bu masalın dersi ne", "devam et", "hangi hayvan vardı"]
NAMES = ["Pamuk", "Elif", "Deniz", "Mert", "Selin", "Can", "Ayşe", "Emre", "Zeynep", "Burak", "Ece", "Kerem", "Nazlı", "Furkan", "Boncuk", "Tarçın"]

_tr = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
_filler = re.compile(r"^(tamam|anladım|anlıyorum|peki|olur)\b[^.]{0,40}\.?$", re.I)
_arith = re.compile(r"\d+\s*[x×*+]\s*\d+\s*=")
_cap = re.compile(r"\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]{2,}\b")


def _low(s):
    return s.replace("İ", "i").replace("I", "ı").lower()


def _pairs(d):
    if not isinstance(d, dict) or not isinstance(d.get("turns"), list):
        return None
    t = d["turns"]
    if len(t) < 4 or len(t) % 2 or any(x.get("role") != ("user" if i % 2 == 0 else "assistant") for i, x in enumerate(t)):
        return None
    pairs = [(str(t[i].get("content", "")).strip(), str(t[i + 1].get("content", "")).strip()) for i in range(0, len(t), 2)]
    if any(not u or not a for u, a in pairs) or not _tr.search(" ".join(a for _, a in pairs)):
        return None
    if any(_filler.match(a) or _arith.search(a) or len(a.split()) > 110 for _, a in pairs):
        return None
    return pairs


def validate_plan(d, cons):
    pairs = _pairs(d)
    if not pairs or len(pairs) != len(cons) + 1:
        return None
    prev = pairs[0][1]
    for (u, a), c in zip(pairs[1:], cons):
        key = _low(re.sub(r"['’].*", "", c.split()[0]))[:4]
        if a == prev or key not in _low(a) and not any(_low(w)[:4] in _low(a) for w in c.split() if len(w) > 3):
            return None
        prev = a
    return pairs


def validate_mood(d, nexts):
    pairs = _pairs(d)
    if not pairs or len(pairs) < 3:
        return None
    seen = set()
    for u, a in pairs:
        al = _low(a)
        if "yardımcı olabilirim" in al or "özetleyebilir" in al or "neler yapabilir" in al:
            return None
        if a in seen:
            return None
        seen.add(a)
        if u.strip().lower() in ("başka", "bi tane daha", "bir tane daha") and "?" not in a and "…" not in a and len(a.split()) < 4:
            return None
    return pairs


def validate_sad(d):
    pairs = _pairs(d)
    if not pairs or len(pairs) < 3:
        return None
    for i, (u, a) in enumerate(pairs):
        al = _low(a)
        if any(w in al for w in ("bilmece", "fıkra", "oyun oyna", "kötü hissediyorum", "üzüldüm", "yardımcı olabilirim")) and not any(w in _low(u) for w in ("bilmece", "fıkra", "oyun")):
            return None
    return pairs


def validate_joke(d, k):
    pairs = _pairs(d)
    if not pairs or len(pairs) != k:
        return None
    jokes = []
    for u, a in pairs:
        body = re.sub(r"^(bir tane daha|tamam|peki|o zaman)[:,]?\s*", "", a, flags=re.I).strip()
        if not 8 <= len(body.split()) <= 70:
            return None
        head = " ".join(_low(body).split()[:5])
        if any(head == " ".join(_low(j).split()[:5]) for j in jokes) or any(_low(body)[:40] == _low(j)[:40] for j in jokes):
            return None
        jokes.append(body)
    return pairs


def validate_story(d, fus):
    pairs = _pairs(d)
    if not pairs or len(pairs) != len(fus) + 1:
        return None
    story = pairs[0][1]
    if not 60 <= len(story.split()) <= 170:
        return None
    names = set(_cap.findall(story)) - {"Bir", "Ama", "Sonra", "Ve", "Çok", "Küçük", "Bu", "Her", "Artık", "Hemen", "Ne", "Evet", "Hayır", "Günün", "Bugün"}
    for (u, a), f in zip(pairs[1:], fus):
        w = len(a.split())
        if "adı neydi" in f or "hangi hayvan" in f:
            if w > 20 or (not names & set(_cap.findall(a)) and "yoktu" not in a and "verilmemiş" not in a):
                return None
        elif "başlık" in f:
            if w > 8:
                return None
        elif "{name}" in f or " yap" in u or " olsun" in u:
            m = re.search(r"onu (\w+) yap|adı (\w+) olsun", u)
            nm = next((g for g in m.groups() if g), None) if m else None
            if not nm or nm not in a or w < 40:
                return None
        elif "daha kısa" in f:
            if w > 55 or w >= len(story.split()):
                return None
        elif "sonunda" in f or "dersi" in f:
            if w > 45:
                return None
        elif "devam" in f:
            if w < 30:
                return None
    return pairs


async def one(session, sem, model, rng, stats, key, kind):
    import aiohttp
    if kind == "plan":
        k = rng.choice([2, 3, 3, 4])
        cons = rng.sample(CONSTRAINTS, k)
        user = f"İlk istek: \"{rng.choice(PLAN_START)}\". Sonraki turlarda kullanıcı sırayla şunları söyler: " + "; ".join(f'"{c}"' for c in cons) + ". Şimdi sohbeti yaz."
        system, val, maxw, meta = PLAN, lambda d: validate_plan(d, cons), 60, {"constraints": cons}
    elif kind == "mood":
        k = rng.choice([3, 4, 4, 5])
        nexts = rng.sample(MOOD_NEXT, k - 1)
        user = f"İlk tur: \"{rng.choice(MOOD_START)}\". Sonraki turlarda kullanıcı sırayla şunlara benzer şeyler söyler: " + "; ".join(f'"{c}"' for c in nexts) + ". Şimdi sohbeti yaz."
        system, val, maxw, meta = MOOD, lambda d: validate_mood(d, nexts), 45, {"nexts": nexts}
    elif kind == "sad":
        k = rng.choice([3, 4, 4, 5])
        nexts = rng.sample(SAD_NEXT, k - 1)
        user = f"İlk tur: \"{rng.choice(SAD_START)}\". Sonraki turlarda kullanıcı sırayla şunlara benzer şeyler söyler: " + "; ".join(f'"{c}"' for c in nexts) + ". Şimdi sohbeti yaz."
        system, val, maxw, meta = SAD, lambda d: validate_sad(d), 50, {"nexts": nexts}
    elif kind == "joke":
        k = rng.choice([2, 3, 3, 4])
        nexts = [rng.choice(["bir tane daha", "başka", "bir daha", "gülmedim", "devam", "başka bir fıkra", "bi tane daha"]) for _ in range(k - 1)]
        user = f"İlk tur: \"{rng.choice(['bir fıkra anlat', 'fıkra anlat', 'beni güldür', 'komik bir şey anlat', 'şaka yap', 'temel fıkrası anlat', 'nasreddin hoca fıkrası anlat'])}\". Sonraki turlarda kullanıcı sırayla şunları söyler: " + "; ".join(f'"{c}"' for c in nexts) + ". Şimdi sohbeti yaz."
        system, val, maxw, meta = JOKE, lambda d: validate_joke(d, k), 60, {"nexts": nexts}
    else:
        k = rng.choice([2, 3, 3, 4])
        fus = [f.format(name=rng.choice(NAMES)) for f in rng.sample(STORY_FU, k)]
        user = f"İlk istek: \"{rng.choice(['bana bir masal anlat', 'kısa bir masal anlat', 'kediler hakkında bir hikaye anlat ama kısa olsun', 'içinde bir tavşan geçen masal anlat', 'masal anlat'])}\". Sonraki turlarda kullanıcı sırayla şunları söyler: " + "; ".join(f'"{c}"' for c in fus) + ". Şimdi sohbeti yaz."
        system, val, maxw, meta = STORY, lambda d: validate_story(d, fus), 150, {"followups": fus}
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(API, headers={"Authorization": f"Bearer {key}"},
                                        json={"model": model, "temperature": 0.9, "max_tokens": 1600, "reasoning": {"effort": "low"},
                                              "messages": [{"role": "system", "content": system.replace("{maxw}", str(maxw))}, {"role": "user", "content": user}],
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
                pairs = val(json.loads(m.group(0))) if m else None
                if pairs:
                    stats["ok"] += 1
                    return {"conversations": [{"role": r_, "content": c_} for u, a in pairs for r_, c_ in (("user", u), ("assistant", a))], "kind": kind, **meta}
                stats["rejected"] += 1
                return None
            except Exception:
                stats["exc"] += 1
                await asyncio.sleep(2)
    return None


async def run(out, n, model, concurrency, max_usd, key, seed, kind):
    import aiohttp
    rng = random.Random(seed)
    sem = asyncio.Semaphore(concurrency)
    stats = {"ok": 0, "rejected": 0, "http_err": 0, "exc": 0, "usd": 0.0}
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        async with aiohttp.ClientSession() as session:
            while stats["ok"] < n and stats["usd"] < max_usd:
                try:
                    batch = await asyncio.wait_for(asyncio.gather(*[one(session, sem, model, random.Random(rng.random()), stats, key, kind) for _ in range(16)]), timeout=480)
                except asyncio.TimeoutError:
                    stats["exc"] += 1
                    continue
                for item in batch:
                    if item:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{time.strftime('%H:%M:%S')} {kind} {stats['ok']} ok, {stats['rejected']} rejected, err {stats['http_err']+stats['exc']}, usd~{stats['usd']:.2f}", flush=True)
    print("DONE", kind, stats, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["plan", "mood", "story", "joke", "sad"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--model", default=os.environ.get("UFAKZEKA_LLM_MODEL"))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-usd", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    import os
    key = open(os.path.expanduser("~/.ufakzeka/llm_api_key")).read().strip()
    asyncio.run(run(a.out, a.n, a.model, a.concurrency, a.max_usd, key, a.seed, a.kind))

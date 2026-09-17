"""Turkish TinyStories: short, complete stories in the vocabulary of a five year old.

TinyStories (Eldan and Li, arXiv 2305.07759) showed that models below 10M parameters write fluent, coherent
stories when trained on simple vocabulary and complete short arcs. Our stories come from the BILGE synthetic
corpus, which is the opposite: long, scientific, heavy with nested dialogue. A 151M model imitating that
structure rambles, repeats and runs past the generation budget without ever finishing, which is exactly what
users see when they ask for a masal.

This generates the other kind: 150 to 250 words, a beginning, a problem and a resolution, simple words, no
subplots. Diversity comes from sampling three seed words and a narrative feature per story, as TinyStories did, plus a
name, an opening style and a setting. The first run showed why the last three are needed: over 1,208 stories
the writer used the name "Pamuk" 2,619 times and opened 331 of them with "Bir zamanlar küçük". Seed words
alone do not stop a generator from telling the same story, and a corpus like that teaches one story rather
than story telling.

  python -m ufakzeka.data.gen_stories --out data/stories_tr.jsonl --n 4000 --max-usd 2
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

NOUNS = ["kedi", "köpek", "kuş", "balık", "tavşan", "ayı", "kaplumbağa", "kelebek", "arı", "karınca",
         "çocuk", "dede", "nine", "abla", "kardeş", "öğretmen", "komşu", "arkadaş", "bebek", "çoban",
         "top", "balon", "bisiklet", "uçurtma", "oyuncak", "kitap", "kalem", "çanta", "şapka", "ayakkabı",
         "ağaç", "çiçek", "bahçe", "orman", "deniz", "göl", "dağ", "köy", "sokak", "park",
         "ekmek", "elma", "çilek", "süt", "bal", "kurabiye", "çorba", "karpuz", "fındık", "peynir",
         "yağmur", "kar", "güneş", "ay", "yıldız", "rüzgar", "bulut", "gökkuşağı", "sabah", "akşam"]
VERBS = ["koşmak", "saklanmak", "aramak", "bulmak", "paylaşmak", "yardım etmek", "gülmek", "ağlamak",
         "uyumak", "uyanmak", "zıplamak", "tırmanmak", "yüzmek", "uçmak", "toplamak", "kaybetmek",
         "kurtarmak", "beklemek", "şaşırmak", "korkmak", "sevinmek", "özür dilemek", "söz vermek", "denemek"]
ADJS = ["küçük", "büyük", "kırmızı", "mavi", "sarı", "yeşil", "tatlı", "hızlı", "yavaş", "meraklı",
        "korkak", "cesur", "üzgün", "mutlu", "yorgun", "aç", "sessiz", "gürültülü", "temiz", "eski",
        "yeni", "sıcak", "soğuk", "yumuşak", "parlak", "kayıp", "unutkan", "yaramaz"]
FEATURES = [
    "sonunda küçük bir sürpriz olsun",
    "kahraman bir hata yapsın ve düzeltsin",
    "iki karakter arasında kısa bir konuşma geçsin",
    "sonunda bir arkadaşlık kurulsun",
    "kahraman bir şeyi paylaşmayı öğrensin",
    "başta üzücü olsun, sonu mutlu bitsin",
    "kahraman korktuğu bir şeyi başarsın",
    "kaybolan bir şey bulunsun",
    "biri diğerine yardım etsin",
    "sonunda kısa bir ders çıksın ama öğüt verir gibi olmasın",
]

NAMES = ["Ali", "Ayşe", "Mert", "Zeynep", "Can", "Elif", "Deniz", "Ece", "Kaan", "Defne",
         "Emir", "Nisa", "Poyraz", "Duru", "Efe", "Asya", "Bora", "Melis", "Tuna", "İpek",
         "Kerem", "Lena", "Arda", "Sude", "Umut", "Beren", "Çınar", "Nehir", "Alp", "Mira",
         "Dede Rıza", "Nine Hatice", "Öğretmen Selim", "Komşu Ahmet Amca", "Fatma Teyze",
         "Pamuk", "Fındık", "Kömür", "Boncuk", "Şeker", "Karamel", "Zıpzıp", "Minnoş", "Duman",
         "Tarçın", "Limon", "Badem", "Cesur", "Paytak", "Tombik", "Kıvırcık", "Benekli", "Sarman"]
OPENINGS = ["Hikayeye doğrudan olayın ortasından başla, tanıtım cümlesi kurma.",
            "Hikayeye bir konuşma cümlesiyle başla.",
            "Hikayeye bir soruyla başla.",
            "Hikayeye bir sesin duyulmasıyla başla.",
            "Hikayeye günün bir saatini anlatarak başla.",
            "Hikayeye kahramanın ne yaptığını anlatarak başla.",
            "Hikayeye hava durumunu anlatan tek bir cümleyle başla.",
            "Hikayeye bir sorunun fark edilmesiyle başla.",
            "Klasik masal girişi kullan.",
            "Hikayeye bir kokuyu ya da tadı anlatarak başla."]
SETTINGS = ["köy evi", "apartman", "okul bahçesi", "pazar yeri", "deniz kıyısı", "orman", "dağ eteği",
            "mahalle sokağı", "park", "büyükanne evi", "bakkal", "kütüphane", "çiftlik", "göl kenarı",
            "kar altında bir bahçe", "yaz akşamı balkon", "yağmurlu bir gün", "bayram sabahı",
            "okul yolu", "kamp alanı"]

SYSTEM = """Sen küçük çocuklar için Türkçe masal yazan bir yazarsın.

Kurallar:
- Beş yaşındaki bir çocuğun anlayacağı basit kelimeler kullan. Zor, soyut veya bilimsel kelime kullanma.
- Hikaye 150-250 kelime olsun. Kısa cümleler kur.
- Tam bir hikaye anlat: bir başlangıç, bir sorun ve bir çözüm. Yarıda bırakma, son cümle hikayeyi bitirsin.
- Tek bir olay örgüsü olsun; yan hikaye, ikinci macera veya bölüm ekleme.
- Başlık yazma, "İşte hikaye" gibi giriş cümlesi kurma, doğrudan hikayeye başla.
- Türkçe karakterleri doğru kullan.
- Sana verilen ismi ve mekanı kullan. Her hikaye bir öncekinden farklı olsun.

Yalnızca şu JSON'u döndür: {"hikaye": "..."}"""


def validate(text: str) -> str | None:
    t = (text or "").strip()
    if not t:
        return None
    words = t.split()
    if not (110 <= len(words) <= 320):
        return None
    if not re.search(r"[çğıöşüÇĞİÖŞÜ]", t):
        return None
    if not t.rstrip().endswith((".", "!", "?", "\"", "”")):
        return None  # must actually finish
    if re.search(r"^(İşte|Tabii|Elbette|Tamamdır|Başlık)", t):
        return None
    if t.count("\n\n") > 4:
        return None
    # a sentence repeated verbatim means the writer looped
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    if len(sents) != len(set(sents)):
        return None
    return t


async def one(session, sem, model, rng, stats, key):
    import aiohttp
    seed_words = [rng.choice(NOUNS), rng.choice(VERBS), rng.choice(ADJS)]
    feature = rng.choice(FEATURES)
    name = rng.choice(NAMES)
    setting = rng.choice(SETTINGS)
    opening = rng.choice(OPENINGS)
    user = (f"Şu üç kelimeyi doğal biçimde kullanan bir masal yaz: {', '.join(seed_words)}. "
            f"Kahramanın adı {name} olsun. Hikaye burada geçsin: {setting}. {opening} Ayrıca: {feature}.")
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(API, headers={"Authorization": f"Bearer {key}"},
                                        json={"model": model, "temperature": 1.0, "max_tokens": 1200,
                                              "reasoning": {"effort": "low"},
                                              "messages": [{"role": "system", "content": SYSTEM},
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
                obj = json.loads(m.group(0)) if m else None
                story = validate((obj or {}).get("hikaye", ""))
                if story and name.split()[-1] not in story:
                    story = None  # ignored the assignment, which is how the corpus collapses onto one character
                if story:
                    stats["ok"] += 1
                    return {"text": story, "words": seed_words, "feature": feature,
                            "name": name, "setting": setting}
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
    openings: dict[str, int] = {}
    with open(out, "a", encoding="utf-8") as f:
        async with aiohttp.ClientSession() as session:
            while stats["ok"] < n and stats["usd"] < max_usd:
                # the first run went silent for three hours with a live process and no writes, so no batch is
                # allowed to outlive three request timeouts; a stuck batch is dropped and the loop goes on
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
                    k = item["text"][:60]
                    if k in seen:
                        continue
                    # cap how often any three word opening may repeat, so no single phrase can take over
                    op = " ".join(item["text"].split()[:3]).lower()
                    if openings.get(op, 0) >= max(3, n // 200):
                        stats["ok"] -= 1; stats["rejected"] += 1  # one() already counted it
                        continue
                    openings[op] = openings.get(op, 0) + 1
                    seen.add(k)
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{time.strftime('%H:%M:%S')} {stats['ok']} ok, {stats['rejected']} rejected, "
                      f"err {stats['http_err']+stats['exc']}, usd~{stats['usd']:.2f}, {time.time()-t0:.0f}s",
                      flush=True)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--model", default=os.environ.get("UFAKZEKA_LLM_MODEL"))
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--max-usd", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    import os
    key = open(os.path.expanduser("~/.ufakzeka/llm_api_key")).read().strip()
    asyncio.run(run(a.out, a.n, a.model, a.concurrency, a.max_usd, key, a.seed))

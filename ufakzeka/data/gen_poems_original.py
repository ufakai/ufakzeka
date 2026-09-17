"""Short original Turkish poems for "şiir yaz" requests.

The harvested public domain poems teach the model to quote folk songs ("Kendi şiirim değil ama ...") and the
long refrains loop. A user who types "bir şiir yaz" expects a short original poem. This set: 4 to 8 line poems
in plain modern Turkish on everyday topics, written by a strong model, validated for length, distinct lines
and vocabulary, stored as {"topic", "title", "text"} for everyday_data.original_poem_examples.

  python -m ufakzeka.data.gen_poems_original --out data/poems_original.jsonl --n 600 --max-usd 1.0
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

SYSTEM = """Sen sade, modern Türkçeyle kısa şiirler yazan bir şairsin. Kurallar:
- 4 ile 8 dize arası, her dize en fazla 8 kelime.
- Günlük, anlaşılır Türkçe; Osmanlıca ve ağır edebi kelimeler yok (yâr, kâinat, hazret gibi kelimeler YASAK).
- Uyak serbest ama hoş; tekrar eden nakarat YASAK, her dize farklı olsun.
- Somut imgeler kullan (çay, pencere, yağmur, sokak lambası), klişe ve süslü laf yok.
- Verilen konuya ve tona uy. Bir de kısa bir başlık ver.
Yalnızca şu JSON: {"title": "...", "lines": ["...", "..."]}"""

TOPICS = ["aşk", "ayrılık", "anne", "baba", "arkadaşlık", "yağmur", "kar", "deniz", "İstanbul", "sonbahar", "ilkbahar", "gece", "sabah", "çay",
          "kedi", "köpek", "gurbet", "memleket", "özlem", "umut", "yalnızlık", "mutluluk", "çocukluk", "okul", "öğretmen", "bayram", "yeni yıl",
          "doğum günü", "kahve", "uyku", "yıldızlar", "ay", "güneş", "rüzgar", "orman", "dağ", "köy", "şehir", "tren", "yol", "ev", "pencere",
          "kitap", "müzik", "futbol", "pazar sabahı", "iş", "hafta sonu", "bahçe", "çiçek", "kuş", "bulut", "kışın sıcak bir oda", "anneanne", "dede",
          "kardeş", "ilk gün", "veda", "teşekkür", "özür", "cesaret", "sabır", "sessizlik", "telefon", "eski fotoğraf", "sokak kedisi", "simit", "vapur"]
TONES = ["sıcak", "hüzünlü", "neşeli", "sakin", "umutlu", "çocuklar için", "esprili", "nostaljik"]
NAMES = ["Ali", "Elif", "Deniz", "Mert", "Selin", "Can", "Ayşe", "Emre", "Zeynep", "Burak", "Ece", "Merve", "Kerem", "Nazlı", "İrem", "Furkan"]

_ban = re.compile(r"yâr|kâinat|hazret|efendim|sultan|cânân|cân\b|dilber|meşk|zahid|münâcât|felek", re.I)
_tr = re.compile(r"[çğıöşüÇĞİÖŞÜ]")


def validate(d, name):
    if not isinstance(d, dict) or not isinstance(d.get("lines"), list):
        return None
    lines = [re.sub(r"\s+", " ", str(l)).strip(" ,.;") for l in d["lines"]]
    lines = [l for l in lines if l]
    if not 4 <= len(lines) <= 8 or any(len(l.split()) > 9 or len(l) > 70 for l in lines):
        return None
    if len(set(l.lower() for l in lines)) < len(lines) or _ban.search(" ".join(lines)):
        return None
    if not _tr.search(" ".join(lines)) or (name and name not in " ".join(lines)):
        return None
    title = re.sub(r"\s+", " ", str(d.get("title", ""))).strip(' ".')[:60]
    return title, lines


async def one(session, sem, model, rng, stats, key):
    import aiohttp
    topic, tone = rng.choice(TOPICS), rng.choice(TONES)
    name = rng.choice(NAMES) if rng.random() < 0.15 else ""
    user = f"Konu: {topic}. Ton: {tone}." + (f" Şiirin içinde '{name}' adı geçsin (şiir {name} için yazılmış olsun)." if name else "") + " Şiiri yaz."
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(API, headers={"Authorization": f"Bearer {key}"},
                                        json={"model": model, "temperature": 1.0, "max_tokens": 600, "reasoning": {"effort": "low"},
                                              "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                              "response_format": {"type": "json_object"}},
                                        timeout=aiohttp.ClientTimeout(total=120)) as r:
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
                v = validate(json.loads(m.group(0)) if m else None, name)
                if v:
                    stats["ok"] += 1
                    return {"topic": topic, "tone": tone, "name": name, "title": v[0], "text": "\n".join(v[1])}
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
                    batch = await asyncio.wait_for(asyncio.gather(*[one(session, sem, model, random.Random(rng.random()), stats, key) for _ in range(16)]), timeout=300)
                except asyncio.TimeoutError:
                    stats["exc"] += 1
                    continue
                for item in batch:
                    if not item:
                        continue
                    k = item["text"][:40]
                    if k in seen:
                        stats["ok"] -= 1
                        continue
                    seen.add(k)
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{time.strftime('%H:%M:%S')} {stats['ok']} ok, {stats['rejected']} rejected, err {stats['http_err']+stats['exc']}, usd~{stats['usd']:.2f}", flush=True)
    print("DONE", stats, f"{time.time()-t0:.0f}s", flush=True)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--model", default=os.environ.get("UFAKZEKA_LLM_MODEL"))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-usd", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    import os
    key = open(os.path.expanduser("~/.ufakzeka/llm_api_key")).read().strip()
    asyncio.run(run(a.out, a.n, a.model, a.concurrency, a.max_usd, key, a.seed))

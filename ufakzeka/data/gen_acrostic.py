"""Acrostic poems for Turkish first names ("adımla akrostiş şiir"), written by a strong model and checked line by line.

  python -m ufakzeka.data.gen_acrostic --out data/acrostic.jsonl --max-usd 0.5
Output lines: {"name": "Furkan", "text": "F...\nU...\nR...\nK...\nA...\nN..."}
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
SYSTEM = """Sen sade, modern Türkçeyle akrostiş şiir yazan bir şairsin. Verilen ismin her harfi sırayla bir dizenin İLK HARFİ olur;
dize sayısı ismin harf sayısına eşittir. Her dize 3 ile 8 kelime, sıcak ve anlaşılır; Osmanlıca ve süslü kelimeler yok,
nakarat yok. Yalnızca şu JSON: {"lines": ["...", "..."]}"""
SYSTEM_SHORT = SYSTEM.replace("Her dize 3 ile 8 kelime", "Her dize EN FAZLA 4 kelime, kısa ve vurucu")
NAMES = ["Furkan", "Elif", "Mert", "Ayşe", "Deniz", "Selin", "Can", "Emre", "Zeynep", "Burak", "Ece", "Merve", "Kerem", "Nazlı", "İrem", "Ali",
         "Mehmet", "Fatma", "Ahmet", "Hatice", "Mustafa", "Hüseyin", "Zehra", "Yusuf", "Eylül", "Ömer", "Defne", "Kuzey", "Ada", "Aras", "Atlas",
         "Ela", "Lina", "Nehir", "Mira", "Çınar", "Emir", "Kaan", "Berk", "Cem", "Cansu", "Dilara", "Ege", "Gizem", "Hakan", "Işıl", "Kadir",
         "Leyla", "Melis", "Murat", "Onur", "Özge", "Pınar", "Rana", "Seda", "Sinem", "Şule", "Tolga", "Umut", "Ümit", "Volkan", "Yağmur",
         "Barış", "Buse", "Ceren", "Doğa", "Elçin", "Ferhat", "Gökhan", "Halil", "İlayda", "Jale", "Koray", "Mine", "Nil", "Oğuz", "Rüzgar",
         "Serkan", "Tuğba", "Ufuk", "Yasemin", "Zeki", "Aylin", "Bora", "Cemre", "Duru", "Efe", "Gül", "Hande", "Kaya", "Levent", "Naz", "Orhan"]
TR_UPPER = {"i": "İ", "ı": "I"}


def upper_first(ch: str) -> str:
    return TR_UPPER.get(ch, ch.upper())


def validate(d, name):
    if not isinstance(d, dict) or not isinstance(d.get("lines"), list):
        return None
    lines = [re.sub(r"\s+", " ", str(l)).strip(" ,.;") for l in d["lines"]]
    if len(lines) != len(name):
        return None
    for ch, line in zip(name, lines):
        if not line or upper_first(line[0]) != upper_first(ch) or not 2 <= len(line.split()) <= 9:
            return None
    if len(set(l.lower() for l in lines)) < len(lines):
        return None
    return lines


SHORT = False


async def one(session, sem, model, name, stats, key, variant):
    import aiohttp
    user = f"İsim: {name}. Ton: {variant}. Akrostiş şiiri yaz; {len(name)} dize, her dize sırayla şu harflerle başlasın: {', '.join(upper_first(c) for c in name)}." + (" Her dize en fazla 4 kelime." if SHORT else "")
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(API, headers={"Authorization": f"Bearer {key}"},
                                        json={"model": model, "temperature": 0.9, "max_tokens": 500, "reasoning": {"effort": "low"},
                                              "messages": [{"role": "system", "content": SYSTEM_SHORT if SHORT else SYSTEM}, {"role": "user", "content": user}],
                                              "response_format": {"type": "json_object"}}, timeout=aiohttp.ClientTimeout(total=120)) as r:
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
                lines = validate(json.loads(m.group(0)) if m else None, name)
                if lines and SHORT and any(len(l.split()) > 4 for l in lines):
                    lines = None
                if lines:
                    stats["ok"] += 1
                    return {"name": name, "tone": variant, "text": "\n".join(lines), "short": SHORT}
                stats["rejected"] += 1
                return None
            except Exception:
                stats["exc"] += 1
                await asyncio.sleep(2)
    return None


async def run(out, per_name, model, concurrency, max_usd, key):
    import aiohttp
    sem = asyncio.Semaphore(concurrency)
    stats = {"ok": 0, "rejected": 0, "http_err": 0, "exc": 0, "usd": 0.0}
    tones = ["sıcak", "neşeli", "sakin", "umutlu", "çocuklar için", "doğum günü için"]
    jobs = [(n, tones[i % len(tones)]) for n in NAMES for i in range(per_name)]
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(jobs), 16):
                if stats["usd"] >= max_usd:
                    break
                batch = await asyncio.gather(*[one(session, sem, model, n, stats, key, v) for n, v in jobs[i:i + 16]])
                for item in batch:
                    if item:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{time.strftime('%H:%M:%S')} {stats['ok']} ok, {stats['rejected']} rejected, err {stats['http_err']+stats['exc']}, usd~{stats['usd']:.2f}", flush=True)
    print("DONE", stats, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-name", type=int, default=4)
    ap.add_argument("--model", default=os.environ.get("UFAKZEKA_LLM_MODEL"))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-usd", type=float, default=0.5)
    ap.add_argument("--short", action="store_true")
    a = ap.parse_args()
    globals()["SHORT"] = a.short
    import os
    key = open(os.path.expanduser("~/.ufakzeka/llm_api_key")).read().strip()
    asyncio.run(run(a.out, a.per_name, a.model, a.concurrency, a.max_usd, key))

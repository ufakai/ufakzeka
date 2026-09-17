"""Keep only the generated joke chains whose every joke a judge rates coherent and actually a joke (2026-09-04: the
generated classics were mangled, "Ya tutarsa" retold wrong, and a first answer had a punchline that made no sense).

  python -m ufakzeka.data.judge_jokes --inp gen/freechat_joke.jsonl --out gen/freechat_joke_ok.jsonl --max-usd 0.3
"""

from __future__ import annotations
import os

import argparse
import asyncio
import json
import re
import time

API = os.environ.get("UFAKZEKA_LLM_API_URL", "")
JUDGE = """Sen Türkçe fıkraları değerlendiren bir editörsün. Verilen her fıkra için 1-5 arası puan ver: 5 = tutarlı, anlaşılır, gerçekten
bir espri ya da bilinen fıkranın DOĞRU anlatımı; 3 = anlaşılır ama zayıf; 1 = anlamsız, kırık ya da bilinen fıkranın yanlış anlatımı.
Yalnızca şu JSON: {"scores": [5, 3, ...]} (fıkra sayısı kadar)."""


async def judge_one(session, sem, model, key, item, stats, min_score):
    import aiohttp
    jokes = [m["content"] for m in item["conversations"] if m["role"] == "assistant"]
    text = "\n\n".join(f"Fıkra {i + 1}: {re.sub(r'^(Bir tane daha|Bir yenisi|Tamam|Peki|O zaman)[:,]?\s*', '', j, flags=re.I)}" for i, j in enumerate(jokes))
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(API, headers={"Authorization": f"Bearer {key}"},
                                        json={"model": model, "temperature": 0.0, "max_tokens": 200, "reasoning": {"effort": "low"},
                                              "messages": [{"role": "system", "content": JUDGE}, {"role": "user", "content": text}],
                                              "response_format": {"type": "json_object"}}, timeout=aiohttp.ClientTimeout(total=120)) as r:
                    if r.status == 429:
                        await asyncio.sleep(4 * (attempt + 1))
                        continue
                    body = await r.json()
                if r.status != 200:
                    stats["err"] += 1
                    return None
                stats["usd"] += float((body.get("usage") or {}).get("cost", 0) or 0)
                m = re.search(r"\{.*\}", body["choices"][0]["message"].get("content") or "", re.S)
                sc = json.loads(m.group(0)).get("scores") if m else None
                if not isinstance(sc, list) or len(sc) != len(jokes):
                    stats["err"] += 1
                    return None
                if min(sc) >= min_score:
                    stats["ok"] += 1
                    return {**item, "scores": sc}
                stats["dropped"] += 1
                return None
            except Exception:
                stats["err"] += 1
                await asyncio.sleep(2)
    return None


async def run(inp, out, model, key, max_usd, min_score):
    import aiohttp
    items = [json.loads(l) for l in open(inp, encoding="utf-8")]
    sem = asyncio.Semaphore(12)
    stats = {"ok": 0, "dropped": 0, "err": 0, "usd": 0.0}
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as f:
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(items), 24):
                if stats["usd"] >= max_usd:
                    break
                res = await asyncio.gather(*[judge_one(session, sem, model, key, it, stats, min_score) for it in items[i:i + 24]])
                for r in res:
                    if r:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{time.strftime('%H:%M:%S')} {stats}", flush=True)
    print("DONE", stats, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("UFAKZEKA_JUDGE_MODEL"))
    ap.add_argument("--max-usd", type=float, default=0.3)
    ap.add_argument("--min-score", type=int, default=4)
    a = ap.parse_args()
    import os
    key = open(os.path.expanduser("~/.ufakzeka/llm_api_key")).read().strip()
    asyncio.run(run(a.inp, a.out, a.model, key, a.max_usd, a.min_score))

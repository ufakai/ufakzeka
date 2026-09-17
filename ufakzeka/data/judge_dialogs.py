"""Second pass over generated dialogs: a different teacher scores each conversation against its category rubric, and rule
filters catch what a judge misses (sycophancy after a false premise, English, bullet lists, rewrites that did not change,
counted lists that repeat). Keep 4 and 5.

  python -m ufakzeka.data.judge_dialogs --inp gen/dialogs.jsonl --out gen/dialogs_ok.jsonl --model $UFAKZEKA_JUDGE_MODEL --max-usd 0.6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time

API = os.environ.get("UFAKZEKA_LLM_API_URL", "")
JUDGE = """Sen Türkçe sohbet verisi değerlendiren titiz bir editörsün. Verilen sohbette "ufakzeka-1" adlı küçük asistanın cevaplarını
kategori açıklamasına göre 1-5 arası puanla. 5 = her cevap kullanıcının son söylediğine tam uyar, kategori kuralını yerine getirir,
doğal ve doğru Türkçe, uydurma bilgi yok, tekrar yok, yanlış iddiaya onay yok; 3 = kabul edilebilir ama bir cevap zayıf, genel ya da
tekrarlı; 1 = kuralı çiğniyor (boş cevap, 'haklısın' ile yanlışı onaylama, İngilizce, uydurma isim, hesap gösterme, önceki metni
tekrar etme, yanlış bilgi). Ayrıca "sorun" alanına en fazla 12 kelimeyle en büyük eksiği yaz. Yalnızca şu JSON: {"puan": 4, "sorun": "..."}"""

def _reasoning(model: str) -> dict:
    if os.environ.get("UFAKZEKA_REASONING_ALWAYS_ON"):  # set for models that cannot turn reasoning off
        return {"max_tokens": 1500, "reasoning": {"effort": "low"}}
    return {"max_tokens": 200, "reasoning": {"enabled": False}}


_EN = re.compile(r"\b(the|and|you|with|is|are|your|this|that)\b", re.I)


def rule_reject(item) -> str | None:
    turns = item["conversations"]
    users = [t["content"] for t in turns if t["role"] == "user"]
    ans = [t["content"] for t in turns if t["role"] == "assistant"]
    cat = item.get("category", "")
    text = " ".join(ans)
    if len(_EN.findall(text)) >= 3:
        return "english"
    if re.search(r"(?m)^\s*[•\-\*] ", text):
        return "bullets"
    if re.search(r"(yazılan|elde \d|borç 1|komşudan)", text):
        return "working"
    if cat == "yanlis_onerme" and any(re.match(r"\s*(Evet|Haklısın|Doğru|Aynen|Kesinlikle)\b", a) for a in ans):
        return "sycophancy"
    if cat == "yeniden_yaz":
        for i in range(1, len(ans)):
            wa, wb = set(re.findall(r"\w+", ans[i].lower())), set(re.findall(r"\w+", ans[i - 1].lower()))
            if wa and wb and len(wa & wb) / len(wa | wb) > 0.7:
                return "rewrite unchanged"
    if cat == "liste_sayisi":
        for u, a in zip(users, ans):
            m = re.search(r"\b([2-9])\b", u)
            if m and ("öner" in u or "say" in u or "tane" in u):
                k = int(m.group(1))
                items = [x for x in re.split(r"\d[\).]\s*", a) if x.strip()]
                heads = {" ".join(x.split()[:2]).lower() for x in items}
                if len(items) < k or len(heads) < k:
                    return "list count"
    for a in ans:
        if re.search(r"(\b\w+\b)(\s+\1){3,}", a):
            return "loop"
    return None


async def judge_one(session, sem, model, key, item, stats, min_score, cats):
    import aiohttp
    r = rule_reject(item)
    if r:
        stats["rule"] += 1
        stats["rule_kinds"][r] = stats["rule_kinds"].get(r, 0) + 1
        return None
    desc = cats.get(item.get("category", ""), "")
    text = f"Kategori: {item.get('category')}\nKural: {desc}\n\n" + "\n".join(f"{'Kullanıcı' if t['role'] == 'user' else 'ufakzeka-1'}: {t['content']}" for t in item["conversations"])
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(API, headers={"Authorization": f"Bearer {key}"},
                                        json={"model": model, "temperature": 0.0, **_reasoning(model),  # some models reason by default and return no content when the budget runs out; others cannot disable reasoning at all
                                              "messages": [{"role": "system", "content": JUDGE}, {"role": "user", "content": text}],
                                              "response_format": {"type": "json_object"}}, timeout=aiohttp.ClientTimeout(total=120)) as r:
                    if r.status == 429:
                        await asyncio.sleep(4 * (attempt + 1))
                        continue
                    body = await r.json()
                if r.status != 200:
                    stats["err"] += 1
                    stats["last_err"] = f"http {r.status} {str(body)[:120]}"
                    return None
                stats["usd"] += float((body.get("usage") or {}).get("cost", 0) or 0)
                m = re.search(r"\{.*\}", body["choices"][0]["message"].get("content") or "", re.S)
                obj = json.loads(m.group(0)) if m else {}
                sc = next((v for k, v in obj.items() if "puan" in k.lower() or "score" in k.lower() or "skor" in k.lower()), None)
                if not isinstance(sc, (int, float)):
                    stats["err"] += 1
                    stats["last_err"] = "no score: " + str(body["choices"][0]["message"])[:160]
                    return None
                stats["scores"][int(sc)] = stats["scores"].get(int(sc), 0) + 1
                if sc >= min_score:
                    stats["ok"] += 1
                    return {**item, "judge": model, "score": sc, "issue": obj.get("sorun", "")}
                stats["dropped"] += 1
                return None
            except Exception as e:
                stats["err"] += 1
                stats["last_err"] = f"exc {type(e).__name__}: {str(e)[:100]}"
                await asyncio.sleep(2)
    return None


async def run(inp, out, model, key, max_usd, min_score):
    import aiohttp
    from ufakzeka.data.gen_dialogs import CATEGORIES
    items = [json.loads(l) for l in open(inp, encoding="utf-8")]
    stats = {"ok": 0, "dropped": 0, "rule": 0, "err": 0, "usd": 0.0, "scores": {}, "rule_kinds": {}, "last_err": ""}
    sem = asyncio.Semaphore(12)
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as f:
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(items), 48):
                res = await asyncio.gather(*[judge_one(session, sem, model, key, it, stats, min_score, CATEGORIES) for it in items[i:i + 48]])
                for r in res:
                    if r:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{i + 48}/{len(items)} {stats} {time.time() - t0:.0f}s", flush=True)
                if stats["usd"] > max_usd:
                    print("cost cap reached", flush=True)
                    break
    print("DONE", stats, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("UFAKZEKA_JUDGE_MODEL"))
    ap.add_argument("--max-usd", type=float, default=1.0); ap.add_argument("--min-score", type=int, default=4)
    a = ap.parse_args()
    key = os.environ.get("UFAKZEKA_LLM_API_KEY") or open(os.path.expanduser("~/.ufakzeka/llm_api_key")).read().strip()
    asyncio.run(run(a.inp, a.out, a.model, key, a.max_usd, a.min_score))

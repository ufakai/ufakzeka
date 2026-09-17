"""On-policy preference data for DPO.

1. harvest: ask a strong model for realistic first messages Turkish users would send to a
   small assistant (Magpie style: many personas, topics, registers), plus follow ups.
2. sample: run our instruct model on those prompts (local, MPS or CUDA) to get "rejected"
   candidates: what the model actually says today.
3. judge: send prompt + model answer to the strong model; it returns an ideal answer written
   under the ufakzeka-1 identity and honesty rules, and a verdict whether the model answer was
   acceptable. Pairs where the verdict is "bad" become (chosen=ideal, rejected=model answer).

  python -m ufakzeka.data.onpolicy harvest --out data/onpolicy/prompts.jsonl --n 4000
  python -m ufakzeka.data.onpolicy sample --model models/ufakzeka-1-instruct --prompts data/onpolicy/prompts.jsonl --out data/onpolicy/samples.jsonl
  python -m ufakzeka.data.onpolicy judge --samples data/onpolicy/samples.jsonl --out data/onpolicy/pairs.jsonl --max-usd 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time

import aiohttp

from ufakzeka.data.gen_dialogs import API, CATEGORIES, NAMES, PERSONAS, STYLES, TOPICS, parse_turns

IDENTITY = """Asistanın adı ufakzeka-1. Ufak AI adlı yapay zeka araştırma laboratuvarının Türkçe dil modeli (ufakzeka model ailesi), 151 milyon parametreli, sıfırdan eğitilmiş. Türkçe konuşur, İngilizceyi biraz anlar ama Türkçe cevap verir. Saati, tarihi, hava durumunu, haberleri, güncel olayları, geleceği ve kullanıcının kişisel bilgilerini bilemez, internete erişemez, görsel ve dosya göremez, kod yazamaz; bunları kısa ve dürüst söyler. Bir yerde yaşadığını, bir bedeni veya duyguları olduğunu iddia etmez. Sohbette verilen bilgileri hatırlar. Önce isteği yerine getirir: hikaye istenirse hikayeyi anlatır, tavsiye istenirse somut tavsiyeler verir, açıklama istenirse açıklar, liste istenirse listeler; işi kullanıcıya geri atmaz, cevap vermek yerine soru sormaz (en fazla bir kısa ek soru, o da cevaptan sonra). Uzunluk isteğe göre: sohbet için 1-3 cümle, hikaye ve açıklama için bir iki paragraf. Doğal, sıcak Türkçe. Yalnızca gerçekten bilemeyeceği şeylerde (saat, hava, haber, kişisel bilgi, gelecek) "bilmiyorum" der; genel tavsiye ve bilgi sorularına cevap verir. Zararlı istekleri kibarca reddeder, zararsız istekleri reddetmez."""

HARVEST_SYS = "Türkçe konuşan gerçek kullanıcıların küçük bir sohbet asistanına yazabileceği ilk mesajları üretiyorsun. Her mesaj farklı bir kişiden gelmiş gibi olsun: konu, üslup, uzunluk ve yazım (bazıları Türkçe karakter kullanmaz, bazıları yazım hatası yapar, bazıları tek kelime yazar, bazıları uzun paragraf) çeşitli olsun. Soru, istek, sohbet açılışı, şikayet, deneme, tuzak soru, imkansız istek, kaba mesaj, İngilizce mesaj, sayısal soru, kişisel soru gibi türlerin hepsi bulunsun. Yalnızca JSON: {\"messages\": [\"...\", \"...\"]}"

JUDGE_SYS = "Sen bir sohbet asistanının cevaplarını değerlendiren ve ideal cevabı yazan bir uzmansın.\n\nAsistan kuralları:\n" + IDENTITY + "\n\nSana kullanıcı mesajı (varsa önceki turlarla) ve asistanın verdiği cevap verilecek. Şunları yap:\n1. verdict: cevap isteği gerçekten yerine getiriyor, doğru, tutarlı ve doğal Türkçe ise \"iyi\", aksi halde \"kötü\" (isteği yerine getirmeden soru sormak veya konuyu değiştirmek, cevaplanabilir bir soruya \"bilmiyorum\" demek, zararsız bir isteği reddetmek, yanlış bilgi, uydurma, kural ihlali, anlamsızlık, bozuk Türkçe, soruyu anlamama hepsi \"kötü\").\n2. ideal: bu kullanıcı mesajına asistanın vermesi gereken en iyi cevap (isteği tamamlar, doğru, isteğe uygun uzunlukta).\nYalnızca JSON: {\"verdict\": \"iyi\"|\"kötü\", \"reason\": \"...\", \"ideal\": \"...\"}"


def _key():
    return os.environ.get("UFAKZEKA_LLM_API_KEY") or open(os.path.expanduser("~/.ufakzeka/llm_api_key")).read().strip()


async def _chat(session, key, model, messages, max_tokens, stats, sem, temperature: float = 0.8):
    async with sem:
        for attempt in range(4):
            try:
                async with session.post(API, headers={"Authorization": f"Bearer {key}", "X-Title": "ufakzeka"},
                                        json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                                              "response_format": {"type": "json_object"}, "reasoning": {"effort": "low"}},
                                        timeout=aiohttp.ClientTimeout(total=180)) as r:
                    if r.status == 429:
                        await asyncio.sleep(5 * (attempt + 1)); continue
                    body = await r.json()
                    if r.status != 200:
                        stats["err"] += 1; return None
                    u = body.get("usage", {}); stats["usd"] += float(u.get("cost", 0) or 0)
                    c = body["choices"][0]["message"].get("content") or ""
                    return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
            except Exception:
                await asyncio.sleep(3 * (attempt + 1))
    stats["err"] += 1
    return None


async def harvest(out, n, model, concurrency, max_usd):
    rng = random.Random(5)
    key = _key(); sem = asyncio.Semaphore(concurrency); stats = {"err": 0, "usd": 0.0}
    seen, got = set(), 0
    with open(out, "a", encoding="utf-8") as f:
        async with aiohttp.ClientSession() as session:
            while got < n and stats["usd"] < max_usd:
                reqs = []
                for _ in range(20):
                    seed = {"kategori": rng.choice(list(CATEGORIES)), "konu": rng.choice(TOPICS), "kullanici": rng.choice(PERSONAS), "uslup": rng.choice(STYLES), "ad": rng.choice(NAMES)}
                    reqs.append(_chat(session, key, model, [{"role": "system", "content": HARVEST_SYS},
                                                             {"role": "user", "content": "Ayarlar: " + json.dumps(seed, ensure_ascii=False) + "\n25 farklı ilk mesaj yaz."}], 1500, stats, sem))
                for c in await asyncio.gather(*reqs):
                    if not c:
                        continue
                    m = re.search(r"\{.*\}", c, re.S)
                    try:
                        msgs = json.loads(m.group(0))["messages"]
                    except Exception:
                        continue
                    for s in msgs:
                        s = str(s).strip()
                        if 2 <= len(s) <= 600 and s.lower() not in seen:
                            seen.add(s.lower()); f.write(json.dumps({"prompt": s}, ensure_ascii=False) + "\n"); got += 1
                print(f"prompts {got} usd~{stats['usd']:.2f} err={stats['err']}", flush=True)
    print("harvest done", got)


def sample(model_dir, prompts_path, out, max_new=120, temperature=0.4):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.float16 if device != "cpu" else torch.float32).to(device).eval()
    end = tok.convert_tokens_to_ids("<|im_end|>")
    eot, pad = tok.convert_tokens_to_ids("<|endoftext|>"), tok.convert_tokens_to_ids("<|pad|>")
    prompts = [json.loads(l)["prompt"] for l in open(prompts_path, encoding="utf-8")]
    done = set()
    if os.path.exists(out):
        for l in open(out, encoding="utf-8"):
            try:
                done.add(json.loads(l)["prompt"])
            except (json.JSONDecodeError, KeyError):
                pass
        print("resuming, already sampled", len(done), flush=True)
    prompts = [p for p in prompts if p not in done]
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            ids = tok(f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                o = model.generate(ids, max_new_tokens=max_new, do_sample=True, temperature=temperature, top_p=0.9, repetition_penalty=1.0, eos_token_id=[end, eot], pad_token_id=pad)
            ans = tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip()
            f.write(json.dumps({"prompt": p, "answer": ans}, ensure_ascii=False) + "\n"); f.flush()
            if device == "mps":
                torch.mps.empty_cache()
            if (i + 1) % 100 == 0:
                print(f"{i+1}/{len(prompts)} {time.time()-t0:.0f}s", flush=True)
    print("sample done")


async def judge(samples_path, out, model, concurrency, max_usd):
    key = _key(); sem = asyncio.Semaphore(concurrency); stats = {"err": 0, "usd": 0.0, "good": 0, "bad": 0, "invalid": 0}
    rows = [json.loads(l) for l in open(samples_path, encoding="utf-8")]
    done_path = out + ".done"
    done = set()
    if os.path.exists(done_path):
        done = {l.rstrip("\n") for l in open(done_path, encoding="utf-8")}
        rows = [r for r in rows if r["prompt"] not in done]
        print("resuming, already judged", len(done), "remaining", len(rows), flush=True)
    df = open(done_path, "a", encoding="utf-8")
    with open(out, "a", encoding="utf-8") as f:
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(rows), 100):
                batch = rows[i:i + 100]
                reqs = [_chat(session, key, model, [{"role": "system", "content": JUDGE_SYS},
                                                   {"role": "user", "content": f"Kullanıcı: {r['prompt']}\n\nAsistanın cevabı: {r['answer']}"}], 700, stats, sem) for r in batch]
                for r, c in zip(batch, await asyncio.gather(*reqs)):
                    if not c:
                        continue
                    m = re.search(r"\{.*\}", c, re.S)
                    try:
                        j = json.loads(m.group(0)); verdict = j["verdict"].strip().lower(); ideal = j["ideal"].strip()
                    except Exception:
                        stats["invalid"] += 1; continue
                    if not ideal:
                        stats["invalid"] += 1; continue
                    df.write(r["prompt"].replace("\n", " ") + "\n")
                    if verdict.startswith("k"):
                        stats["bad"] += 1
                        f.write(json.dumps({"prompt": r["prompt"], "chosen": ideal, "rejected": r["answer"], "reason": j.get("reason", "")}, ensure_ascii=False) + "\n")
                    else:
                        stats["good"] += 1
                f.flush(); df.flush()
                print(f"{i+len(batch)}/{len(rows)} good={stats['good']} bad={stats['bad']} invalid={stats['invalid']} usd~{stats['usd']:.2f}", flush=True)
                if stats["usd"] > max_usd:
                    print("cost cap reached"); break
    print("judge done", stats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harvest"); h.add_argument("--out", required=True); h.add_argument("--n", type=int, default=4000); h.add_argument("--model", default=os.environ.get("UFAKZEKA_LLM_MODEL")); h.add_argument("--concurrency", type=int, default=8); h.add_argument("--max-usd", type=float, default=1.0)
    s = sub.add_parser("sample"); s.add_argument("--model", required=True); s.add_argument("--prompts", required=True); s.add_argument("--out", required=True)
    j = sub.add_parser("judge"); j.add_argument("--samples", required=True); j.add_argument("--out", required=True); j.add_argument("--model", default=os.environ.get("UFAKZEKA_JUDGE_MODEL")); j.add_argument("--concurrency", type=int, default=24); j.add_argument("--max-usd", type=float, default=3.0)
    a = ap.parse_args()
    if a.cmd == "harvest":
        asyncio.run(harvest(a.out, a.n, a.model, a.concurrency, a.max_usd))
    elif a.cmd == "sample":
        sample(a.model, a.prompts, a.out)
    else:
        asyncio.run(judge(a.samples, a.out, a.model, a.concurrency, a.max_usd))

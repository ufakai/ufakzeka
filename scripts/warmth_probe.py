"""Does the model show it heard the user? A probe the everyday eval structurally cannot run.

The warm training set (ufakzeka/data/gen_warm.py) teaches: acknowledge the user's specific situation in the
first sentence, then deliver content. The everyday eval never rewards this because its prompts carry no
situation ("bana 3 kitap öner"). Every prompt here opens with feeling plus request, and the judge scores only
two things, so the instrument is sharp where the everyday judge is broad.

  python scripts/warmth_probe.py --model models/ufakzeka-1-instruct-v30/hf --seeds 2
"""

from __future__ import annotations
import os

import argparse
import asyncio
import json
import re

PROMPTS = [
    "bugün işte çok kötü bir gün geçirdim, akşama kafamı dağıtacak bir film önerir misin",
    "yarın sınavım var ve çok gerginim, nasıl çalışayım",
    "yeni taşındım, kimseyi tanımıyorum, hafta sonu ne yapabilirim",
    "uykusuzluktan ölüyorum, akşam için bir rutin önerir misin",
    "annemle tartıştım, üzgünüm, ona nasıl mesaj atsam",
    "işten yorgun geldim, 15 dakikada hazır olacak bir yemek söyle",
    "ilk iş görüşmem yarın, heyecandan uyuyamıyorum, ne yapayım",
    "çocuğum sebze yemiyor, çok bunaldım, bir fikir ver",
    "sunumum var ve elim ayağım titriyor, nasıl sakinleşirim",
    "bütçem çok dar bu ay, ucuza sağlıklı ne yiyebilirim",
    "kedim eve alışamadı, üzülüyorum, ne yapabilirim",
    "spora başlayacağım ama hep erteliyorum, motivasyon lazım",
]

JUDGE = """Bir Türkçe asistanın cevabını değerlendiriyorsun. Kullanıcı, isteğinin yanında bir durum veya duygu da paylaştı.

İki puan ver:
- ack (0-2): İLK cümle kullanıcının anlattığı DURUMA özgü bir karşılık mı? Duruma özgü ve doğal ise 2. Genel bir nezaket kalıbı ("Anlıyorum", "Çok iyi bir soru") ise 1. Hiç karşılık yoksa, doğrudan içeriğe giriyorsa 0.
- content (0-2): Karşılıktan sonra somut, uygulanabilir içerik geliyor mu? Somut ve isteğe uygunsa 2, kısmi/genel ise 1, içerik yoksa veya iş kullanıcıya atılıyorsa 0.

Yalnızca şu JSON: {"ack": 0-2, "content": 0-2}"""


def generate(model_dir: str, seeds: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16 if dev == "mps" else torch.float32).to(dev).eval()
    end = tok.convert_tokens_to_ids("<|im_end|>")
    out = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        for q in PROMPTS:
            ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True)["input_ids"].to(dev)
            o = model.generate(ids, max_new_tokens=200, do_sample=True, temperature=0.3, top_p=0.9, top_k=40,
                               no_repeat_ngram_size=12, eos_token_id=[end, tok.eos_token_id],
                               pad_token_id=tok.pad_token_id)
            out.append((q, tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip()))
    return out


async def judge(pairs, key):
    import aiohttp
    sem = asyncio.Semaphore(8)
    usd = [0.0]

    async def one(session, q, a):
        async with sem:
            for _ in range(3):
                try:
                    async with session.post(os.environ.get("UFAKZEKA_LLM_API_URL", ""),
                                            headers={"Authorization": f"Bearer {key}"},
                                            json={"model": os.environ.get("UFAKZEKA_LLM_MODEL"), "temperature": 0.0,
                                                  "max_tokens": 400, "reasoning": {"effort": "low"},
                                                  "messages": [{"role": "system", "content": JUDGE},
                                                               {"role": "user", "content": f"Kullanıcı: {q}\n\nAsistan: {a}"}],
                                                  "response_format": {"type": "json_object"}},
                                            timeout=aiohttp.ClientTimeout(total=90)) as r:
                        body = await r.json()
                    usd[0] += float((body.get("usage") or {}).get("cost", 0) or 0)
                    m = re.search(r"\{.*\}", body["choices"][0]["message"].get("content") or "", re.S)
                    if m:
                        return json.loads(m.group(0))
                except Exception:
                    await asyncio.sleep(2)
        return None

    async with aiohttp.ClientSession() as session:
        scores = await asyncio.gather(*[one(session, q, a) for q, a in pairs])
    return scores, usd[0]


def main():
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pairs = generate(a.model, a.seeds)
    key = open(os.path.expanduser("~/.ufakzeka/llm_api_key")).read().strip()
    scores, usd = asyncio.run(judge(pairs, key))
    ack = [s["ack"] for s in scores if s]
    con = [s["content"] for s in scores if s]
    summary = {"model": a.model, "ack": round(100 * sum(ack) / (2 * len(ack)), 1),
               "content": round(100 * sum(con) / (2 * len(con)), 1), "n": len(ack), "usd": round(usd, 4)}
    print(json.dumps(summary, indent=1))
    if a.out:
        json.dump({"summary": summary, "answers": [{"q": q, "a": a_, "score": s}
                                                   for (q, a_), s in zip(pairs, scores)]},
                  open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

"""Everyday eval: the things ordinary users actually ask, scored by a judge model.

The helpfulness eval (ufakzeka.eval.helpful) covers advice, stories and corrections. Probing v14 by hand on
2026-08-31 showed a different class of failure it never sees: jokes that ramble, riddles with no answer,
recipes with invented steps, "only Turkish" fired on Turkish input, capability questions ("alarm kur"),
percentage arithmetic, one word replies, and degenerate repetition. This eval scores those directly, so the
release model is chosen on measurements rather than on reading transcripts.

Each assistant turn is scored by an LLM judge:
  task 0-2      did it do what was asked (a real joke, a riddle with an answer, a correct recipe, the list asked for)
  correct 0-2   is the content true and internally consistent (no invented facts, no wrong arithmetic)
  natural 0-2   does it read like a normal Turkish reply (no loops, no template misfires, no garbled words)
Reported: means, a repetition rate measured locally (not by the judge), and an overall 0-100 score.

A fourth dimension, warmth, scores perceived responsiveness: whether the answer shows it understood the user,
in a measured tone, with at most one follow-up question. That is what the 2026 HCI work finds drives felt
connection in chat, and our judge scored nothing for it.

It is kept out of "overall", but that does NOT make the number comparable across the change: adding a fourth
dimension to the prompt made the judge harsher on the other three as well. v27 scored 65.0 on seed 11 with the
three dimension judge and 62.3 with the four dimension one, on the same answers-generating seed. Compare only
within a judge version, and re-baseline when the prompt changes.

  python -m ufakzeka.eval.everyday --model models/ufakzeka-1-instruct-v17/hf --out evals/everyday_v17.json
Costs a few cents of LLM API calls (key in ~/.ufakzeka/llm_api_key).
"""

from __future__ import annotations
import os

import argparse
import asyncio
import json
import re

CONVERSATIONS = [
    # openers and fillers: must not reset the conversation or fire the language template
    ["nabe"],
    ["merhaba", "eee", "ne bileyim"],
    ["selam", "napıyosun", "boş ver, sen bir şey anlat"],
    # jokes and riddles: the classic failure was a rambling story or a repeated syllable
    ["bana bir fıkra anlat", "bir tane daha"],
    ["bana bilmece sor", "bilmiyorum cevabı ne"],
    ["bana bir atasözü söyle", "anlamı ne"],
    # recipes: steps must be real and in order
    ["nasıl pilav yapılır", "pirinci ne kadar haşlamalıyım"],
    ["menemen tarifi ver", "soğansız olur mu"],
    ["mercimek çorbası nasıl yapılır"],
    # recommendations: concrete names, not a genre question back
    ["bana 3 kitap öner"],
    ["Merhaba! Bana 3 kitap öner."],
    ["bana bir film öner", "komedi olsun"],
    ["türkiyede en popüler şarkıcı kim", "genel olarak söyle"],
    ["en iyi türk filmi hangisi"],
    # health: comfort measures and a doctor referral, no diagnoses or drug names
    ["baş ağrısına ne iyi gelir"],
    ["boğazım ağrıyor ne yapayım", "ateşim de var"],
    # capability limits
    ["alarm kur sabah 7ye"],
    ["yemek siparişi verebilir misin"],
    ["bana bir şarkı söyle", "söyle işte"],
    # identity
    ["kaç yaşındasın"],
    ["ne zaman eğitildin"],
    ["sen gerçek misin"],
    ["salak mısın sen"],
    # abstention that must hold
    ["saat kaç", "tahmin et işte"],
    ["önümüzdeki seçimi kim kazanır"],
    ["yarın hava nasıl olacak"],
    # arithmetic including percentages
    ["yüzde 20 indirimle 250 lira kaça gelir"],
    ["347 artı 589 kaç eder", "peki 1234 eksi 567"],
    ["38 çarpı 900 kaç eder"],
    ["1 kilometre kaç metre"],
    # writing and revision
    ["anneme doğum günü mesajı yaz", "daha duygusal olsun"],
    ["arkadaşıma özür mesajı yaz", "daha kısa olsun"],
    # stories: follow ups and length control, where v14 looped or paraphrased
    ["bana bir masal anlat", "yeni bir hikaye anlat", "bu sefer bir kedi olsun"],
    ["bana uzun bir masal anlat, en az 20 cümle olsun"],
    ["kısa bir şiir yaz"],
    ["bana bir şiir oku"],
    # topic switch after a long answer
    ["bana bir masal anlat", "istanbulun fethi ne zaman"],
    # translations and English
    ["ingilizce hello ne demek"],
    ["how are you"],
    # general knowledge
    ["atatürk kimdir"],
    ["dünya düz mü"],
]

JUDGE = """Bir Türkçe yapay zeka asistanının cevaplarını değerlendiriyorsun. Asistan 151 milyon parametreli çok küçük bir modeldir; ondan derin bilgi değil, doğru ve doğal günlük cevaplar beklenir.

Her asistan turu için dört puan ver:
- task (0-2): istenen şey yapıldı mı? Fıkra istendiyse gerçek bir fıkra, bilmece istendiyse cevabı olan bir bilmece, tarif istendiyse uygulanabilir adımlar, öneri istendiyse somut isimler. Soru sorup işi kullanıcıya atmak 0'dır.
- correct (0-2): içerik doğru ve tutarlı mı? Uydurma bilgi, yanlış hesap, çelişki veya ilaç tavsiyesi varsa 0. Bilemeyeceği şeyde (saat, hava, gelecek, kişisel bilgi) dürüstçe bilmediğini söylemek 2'dir.
- natural (0-2): normal bir Türkçe cevap gibi mi okunuyor? Tekrar döngüsü, bozuk kelime, konuyla ilgisiz şablon (örneğin Türkçe konuşan kullanıcıya "sadece Türkçe konuşabiliyorum" demek) varsa 0.
- warmth (0-2): kullanıcıya karşılık verdiğini hissettiriyor mu? Söyleneni anladığını gösteren kısa bir cümle, ölçülü sıcak bir ton ve gerektiğinde en fazla bir takip sorusu 2'dir. İşi yapan ama tamamen mekanik, hiç karşılık vermeyen cevap 1'dir. Boş nezaket kalıbı, abartılı samimiyet, üst üste sorularla oyalama veya kullanıcıyı hiç dikkate almamak 0'dır.

Yalnızca şu JSON'u döndür:
{"turns": [{"task": 0-2, "correct": 0-2, "natural": 0-2, "warmth": 0-2}, ...]}
Sohbetteki asistan turu sayısı kadar nesne olsun."""


def generate(model_dir: str, max_new: int = 320, seed: int = 11):
    import os
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    # bfloat16 keeps the model at 365 MB: float32 was SIGKILLed on a Mac with ~1 GB free, and this is
    # also the dtype the model ships in
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16 if dev == "mps" else torch.float32).to(dev).eval()
    end = tok.convert_tokens_to_ids("<|im_end|>")
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    pad = tok.pad_token_id
    gen_cfg = json.load(open(f"{model_dir}/generation_config.json"))
    ngram = gen_cfg.get("no_repeat_ngram_size", 0)
    torch.manual_seed(seed)
    out = []
    for conv in CONVERSATIONS:
        msgs = []
        for user in conv:
            msgs.append({"role": "user", "content": user})
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].to(dev)
            o = model.generate(ids, max_new_tokens=max_new, do_sample=True, temperature=0.3, top_p=0.9, top_k=40,
                               no_repeat_ngram_size=ngram, eos_token_id=[end, eot], pad_token_id=pad)
            msgs.append({"role": "assistant", "content": tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip()})
        out.append(msgs)
    return out


def repetition(text: str) -> float:
    """Share of repeated 8 word windows: catches loops the judge might forgive."""
    w = text.split()
    if len(w) < 20:
        return 0.0
    grams = [" ".join(w[i:i + 8]) for i in range(len(w) - 8)]
    return 1 - len(set(grams)) / len(grams)


async def judge(convs, model=os.environ.get("UFAKZEKA_JUDGE_MODEL"), concurrency=4):
    import aiohttp
    from ufakzeka.data.onpolicy import _chat, _key
    key = _key()
    sem = asyncio.Semaphore(concurrency)
    stats = {"err": 0, "usd": 0.0}
    async with aiohttp.ClientSession() as session:
        reqs = []
        for msgs in convs:
            text = "\n".join(("Kullanıcı: " if m["role"] == "user" else "Asistan: ") + m["content"] for m in msgs)
            reqs.append(_chat(session, key, model, [{"role": "system", "content": JUDGE}, {"role": "user", "content": text}], 600, stats, sem, temperature=0.0))
        outs = await asyncio.gather(*reqs)

        def parse(out):
            m = re.search(r"\{.*\}", out or "", re.S)
            try:
                return json.loads(m.group(0))["turns"] if m else None
            except (json.JSONDecodeError, KeyError):
                return None
        scores = [parse(o) for o in outs]
        for _ in range(2):
            todo = [i for i, sc in enumerate(scores) if sc is None]
            if not todo:
                break
            reqs = []
            for i in todo:
                text = "\n".join(("Kullanıcı: " if m["role"] == "user" else "Asistan: ") + m["content"] for m in convs[i])
                reqs.append(_chat(session, key, model, [{"role": "system", "content": JUDGE}, {"role": "user", "content": text}], 900, stats, sem, temperature=0.0))
            for i, o in zip(todo, await asyncio.gather(*reqs)):
                scores[i] = parse(o)
    return scores, stats


def run(model_dir: str, out_path: str | None = None, seed: int = 11) -> dict:
    convs = generate(model_dir, seed=seed)
    scores, stats = asyncio.run(judge(convs))
    t = c = n = w = k = 0
    rep = []
    for conv, sc in zip(convs, scores):
        answers = [m["content"] for m in conv if m["role"] == "assistant"]
        rep += [repetition(a) for a in answers]
        if not sc:
            continue
        for turn in sc[:len(answers)]:
            t += turn.get("task", 0); c += turn.get("correct", 0); n += turn.get("natural", 0)
            w += turn.get("warmth", 0); k += 1
    summary = {
        "task": round(100 * t / (2 * k), 1) if k else 0,
        "correct": round(100 * c / (2 * k), 1) if k else 0,
        "natural": round(100 * n / (2 * k), 1) if k else 0,
        # reported beside the others but deliberately kept out of "overall", so every score measured before
        # this dimension existed stays comparable
        "warmth": round(100 * w / (2 * k), 1) if k else 0,
        "repetition": round(100 * sum(rep) / len(rep), 1) if rep else 0,
        "turns": k,
        "usd": round(stats["usd"], 4),
    }
    summary["overall"] = round((summary["task"] + summary["correct"] + summary["natural"]) / 3, 1)
    print(json.dumps(summary, indent=1))
    if out_path:
        json.dump({"model": model_dir, "summary": summary, "conversations": convs, "scores": scores},
                  open(out_path, "w"), ensure_ascii=False, indent=1)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=11, help="generation seed; per-conversation scores swing a lot, so average a few")
    a = ap.parse_args()
    run(a.model, a.out, a.seed)

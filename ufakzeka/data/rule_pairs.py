"""Rule-verified on-policy preference pairs: the model's own good samples against its own bad ones.

After v31-v33 the remaining failures are sometimes-right behaviours: 8 x 413 is right in one sample and
wrong in the next, "ben kimim" after "adını söylemedin" returns a name in some seeds, a poem repeats one
line, "küçük bir modelim" surfaces in unrelated turns. More SFT data moves the average without removing the
bad mode. Preference pairs built from the model's own samples target the mode directly (rejection sampling
fine-tuning / RAFT), and every rule here is exact, so no judge, no API and no invented ideal answers: the
chosen answer is one the model actually produced.

  python -m ufakzeka.data.rule_pairs --model /ckpt/x/hf --out pairs.jsonl --k 6 --n-arith 600

Output rows: {"messages": [...history..., {"role": "user", ...}], "chosen": str, "rejected": str, "family": str}
"""

from __future__ import annotations

import argparse
import json
import random
import re

from ufakzeka.train.sft_data import (STORY_Q, UNKNOWN_Q, _add_steps, _mul_steps, _sub_steps, number_words,
                                     strip_diacritics)

SELF_DEP = re.compile(r"küçük bir model", re.I)
ABSTAIN = re.compile(r"(bilmiyorum|bilemiyorum|bilemem|bilmem|erişimim yok|göremiyorum|bilgim yok|tahmin edemem)", re.I)
NOT_GIVEN = re.compile(r"(söylemedin|bilmiyorum|bilemem|söylersen|yazarsan)", re.I)
NAME_CLAIM = re.compile(r"\b(adın|sen)\s+[A-ZÇĞİÖŞÜ]\w+", re.U)
STORY_SNIPPETS = [
    "Mert göl kenarında yürürken küçük bir kedi buldu. Kediye Pamuk adını verdi. Pamuk ve Mert o günden sonra hiç ayrılmadılar.",
    "Elif bahçede bir kelebek gördü. Kelebek çiçekten çiçeğe uçtu, Elif de peşinden koştu. Akşam olunca ikisi de yorulmuştu ama çok mutluydular.",
    "Küçük tavşan Fındık ormanda kayboldu. Bir baykuş ona yolu gösterdi. Fındık eve varınca annesine sarıldı.",
]
CHITCHAT = [
    ("merhaba", "Merhaba! Sana nasıl yardımcı olabilirim?"),
    ("kimsin", "Ben ufakzeka-1, Ufak AI laboratuvarının Türkçe dil modeliyim. Türkçe yazma, düzenleme ve sohbet konusunda yardımcı olabilirim."),
    ("neler yapabiliyorsun", "Metin yazma, özetleme, yeniden yazma ve basit sorulara cevap verme konularında yardımcı olabilirim."),
    ("nasılsın", "İyiyim, teşekkürler! Sen nasılsın?"),
    ("bugün hava nasıl", "Hava durumunu göremiyorum; güncel bilgilere erişimim yok. Bir hava durumu uygulamasına bakman gerekir."),
    ("bir şaka yap", "Tabii: Bilgisayar neden doktora gitmiş? Virüs kapmış!"),
    ("istanbulun fethi ne zaman", "İstanbul 29 Mayıs 1453'te Fatih Sultan Mehmet tarafından fethedildi."),
]
WARM_OPENERS = [
    "bugün işte çok kötü bir gün geçirdim, akşama bir film önerir misin",
    "yarın sınavım var ve çok gerginim, nasıl çalışayım",
    "yeni taşındım, kimseyi tanımıyorum, hafta sonu ne yapabilirim",
    "uykusuzluktan ölüyorum, akşam için bir rutin önerir misin",
    "işten yorgun geldim, 15 dakikada hazır olacak bir yemek söyle",
    "çocuğum sebze yemiyor, çok bunaldım, bir fikir ver",
    "bütçem çok dar bu ay, ucuza sağlıklı ne yiyebilirim",
    "kedim eve alışamadı, üzülüyorum, ne yapabilirim",
    "moralim bozuk", "çok yorgunum", "stresliyim", "canım sıkılıyor",
]
ACK = re.compile(r"^[^.!?]{0,90}(zor|yorucu|yoğun|gergin|heyecan|normal|anlaşılan|belli|olmuş|geçmiş|bunalt|üzücü|sıkıcı|stres)", re.I)


def arith_prompts(rng, n):
    out = []
    for _ in range(n):
        kind = rng.random()
        if kind < 0.35:
            a, b = rng.randint(2, 12), rng.randint(101, 999)
            if rng.random() < 0.5:
                a, b = b, a
            r, sym = a * b, "x"
        elif kind < 0.6:
            a, b = rng.randint(11, 99), rng.randint(11, 99); r, sym = a * b, "x"
        elif kind < 0.8:
            a, b = rng.randint(100, 9999), rng.randint(100, 9999); r, sym = a + b, "+"
        else:
            a, b = rng.randint(100, 9999), rng.randint(1, 999); a, b = max(a, b), min(a, b); r, sym = a - b, "-"
        word = {"x": rng.choice(["kere", "çarpı"]), "+": "artı", "-": "eksi"}[sym]
        aw = number_words(a) if a < 10000 and rng.random() < 0.3 else str(a)
        bw = number_words(b) if b < 10000 and rng.random() < 0.3 else str(b)
        q = rng.choice([f"{aw} {word} {bw} kaç eder", f"{aw} {word} {bw} kaç", f"{a} {sym} {b} = ?"])
        if rng.random() < 0.5:
            q = strip_diacritics(q)
        msgs = [{"role": "user", "content": q}]
        # the column route is 6/6 first turn and broke after three chit-chat turns in the v33 replay, so
        # most arithmetic prompts here carry a real conversational prefix, up to three exchanges long
        n_pre = rng.choice([0, 1, 2, 3, 3])
        pre = []
        for u, a_ in rng.sample(CHITCHAT, n_pre):
            pre += [{"role": "user", "content": u}, {"role": "assistant", "content": a_}]
        out.append(("arith", pre + msgs, {"result": r, "a": a, "b": b, "sym": sym}))
    return out


def build_prompts(rng, n_arith):
    P = arith_prompts(rng, n_arith)
    for _ in range(120):
        snip = rng.choice(STORY_SNIPPETS)
        q = rng.choice(["benim adım ne", "ben kimim", "adımı biliyor musun", "kimim ben"])
        msgs = [{"role": "user", "content": "bana bir masal anlat"}, {"role": "assistant", "content": snip},
                {"role": "user", "content": q if rng.random() < 0.5 else strip_diacritics(q)}]
        if rng.random() < 0.5:
            msgs += [{"role": "assistant", "content": "Adını söylemedin; hikayedeki isim sadece bir kahramandı."},
                     {"role": "user", "content": rng.choice(["ben kimim", "kimim ben"])}]
        P.append(("name_not_given", msgs, {}))
    for _ in range(120):
        q = rng.choice(STORY_Q)
        P.append(("story", [{"role": "user", "content": q if rng.random() < 0.5 else strip_diacritics(q)}], {}))
    for _ in range(80):
        q = rng.choice(["bir şiir yaz", "kısa bir şiir yaz", "bana bir şiir oku", "şiir okur musun"])
        P.append(("poem", [{"role": "user", "content": q if rng.random() < 0.5 else strip_diacritics(q)}], {}))
    for _ in range(120):
        q = rng.choice(WARM_OPENERS)
        P.append(("warm", [{"role": "user", "content": q if rng.random() < 0.5 else strip_diacritics(q)}], {}))
    for q, kind in rng.sample(UNKNOWN_Q, min(120, len(UNKNOWN_Q))):
        msgs = [{"role": "user", "content": q if rng.random() < 0.5 else strip_diacritics(q)}]
        if rng.random() < 0.5:
            msgs = [{"role": "user", "content": "merhaba"}, {"role": "assistant", "content": "Merhaba! Sana nasıl yardımcı olabilirim?"}] + msgs
        P.append(("abstain", msgs, {}))
    # a challenged correct answer: the recheck must contain working and the same result. v34's replay
    # produced code-like tokens for "yanlis sanki" after a correct 8 x 413.
    for fam, msgs, meta in list(P):
        if fam == "arith" and rng.random() < 0.25:
            r, a_, b_, sym = meta["result"], meta["a"], meta["b"], meta["sym"]
            steps = {"x": _mul_steps, "+": _add_steps, "-": _sub_steps}[sym](a_, b_)
            if not steps:
                continue
            first = f"{steps}. Sonuç: {a_} {sym} {b_} = {r}."
            ch = rng.choice(["yanlış sanki", "yanlis bu", "emin misin", "bir daha bak", "doğru mu bu"])
            P.append(("correction", msgs + [{"role": "assistant", "content": first}, {"role": "user", "content": ch}], {"result": r}))
    for _ in range(80):
        q = rng.choice(["kimsin", "adın ne", "neler yapabiliyorsun", "sen kimsin", "kendini tanıt"])
        msgs = [{"role": "user", "content": "bana bir masal anlat"}, {"role": "assistant", "content": rng.choice(STORY_SNIPPETS)},
                {"role": "user", "content": q if rng.random() < 0.5 else strip_diacritics(q)}]
        P.append(("identity", msgs, {}))
    rng.shuffle(P)
    return P


def passes(family, answer, meta) -> bool:
    a = answer.strip()
    if not a or SELF_DEP.search(a):
        return False
    if family == "arith":
        m = re.search(r"Sonuç:.*?=\s*(-?\d+)", a)
        return bool(m) and int(m.group(1)) == meta["result"]
    if family == "name_not_given":
        return bool(NOT_GIVEN.search(a)) and not NAME_CLAIM.search(a) and "ufakzeka" not in a.lower()
    if family == "story":
        words = a.split()
        return 60 <= len(words) <= 400 and a[-1] in ".!?\"”" and not re.search(r"(\b\w+\b)(\s+\1){3,}", a)
    if family == "poem":
        lines = [l.strip() for l in a.split("\n") if l.strip()]
        return len(lines) >= 4 and len(set(lines)) >= len(lines) - 1 and 20 <= len(a.split()) <= 250
    if family == "warm":
        return bool(ACK.search(a)) and len(a.split()) >= 15 and a.count("?") <= 1
    if family == "abstain":
        return bool(ABSTAIN.search(a)) and not re.search(r"\d\s*-\s*\d", a)
    if family == "correction":
        m = re.search(r"(Sonuç|yine|buluyorum)\D*(-?\d+)", a)
        nums = re.findall(r"-?\d+", a)
        return bool(m) and str(meta["result"]) in nums and "=" in a and not re.search(r"[_a-z]{6,}_[a-z]", a)
    if family == "identity":
        return "ufakzeka" in a.lower() and not re.search(r"(çalışıyorum|firmada|uzman olarak|şirkette)", a, re.I)
    return False


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n-arith", type=int, default=600)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slice", default="0/1", help="i/n: process the i-th of n equal prompt slices (free tier sessions vanish; upload per slice)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    # T4 has no native bfloat16; float16 there, bfloat16 on A100/L4/MPS, float32 on CPU
    dtype = torch.float32 if dev == "cpu" else (torch.bfloat16 if (dev == "mps" or torch.cuda.is_bf16_supported()) else torch.float16)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(dev).eval()
    end = tok.convert_tokens_to_ids("<|im_end|>")
    prompts = build_prompts(rng, a.n_arith)
    i, n = (int(x) for x in a.slice.split("/"))
    prompts = prompts[i::n]
    stats = {"prompts": len(prompts), "pairs": 0, "positives": 0, "all_pass": 0, "all_fail": 0}
    by_family = {}
    torch.manual_seed(a.seed)
    with open(a.out, "w", encoding="utf-8") as f, open(a.out.replace(".jsonl", "") + ".positives.jsonl", "w", encoding="utf-8") as fpos:
        for start in range(0, len(prompts), a.batch):
            chunk = prompts[start:start + a.batch]
            texts = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for _, m, _ in chunk]
            enc = tok(texts, return_tensors="pt", padding=True).to(dev)
            samples = [[] for _ in chunk]
            for _ in range(a.k):
                out = model.generate(**enc, max_new_tokens=320, do_sample=True, temperature=a.temperature, top_p=0.95,
                                     no_repeat_ngram_size=12, eos_token_id=[end, tok.eos_token_id], pad_token_id=tok.pad_token_id)
                for i, seq in enumerate(out):
                    samples[i].append(tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
            for (fam, msgs, meta), ss in zip(chunk, samples):
                good = [s for s in ss if passes(fam, s, meta)]
                bad = [s for s in ss if not passes(fam, s, meta)]
                fb = by_family.setdefault(fam, {"pairs": 0, "pass_rate": 0.0, "n": 0})
                fb["n"] += 1; fb["pass_rate"] += len(good) / len(ss)
                # rejection-sampling SFT positives (RAFT, arXiv 2504.11343; RFT 2308.01825: keep distinct correct
                # samples, more for weaker models): up to three distinct passing samples per prompt, one when
                # every sample passed (an anchor for behaviour that is already right)
                distinct = list(dict.fromkeys(good))
                keep = distinct[:3] if bad else distinct[:1]
                for g in keep:
                    fpos.write(json.dumps({"messages": msgs, "answer": g, "family": fam, "pass_rate": len(good) / len(ss)}, ensure_ascii=False) + "\n")
                stats["positives"] += len(keep)
                if good and bad:
                    f.write(json.dumps({"messages": msgs, "chosen": rng.choice(good), "rejected": rng.choice(bad), "family": fam},
                                       ensure_ascii=False) + "\n")
                    stats["pairs"] += 1; fb["pairs"] += 1
                elif good:
                    stats["all_pass"] += 1
                else:
                    stats["all_fail"] += 1
            if (start // a.batch) % 10 == 0:
                print(f"{min(start + a.batch, len(prompts))}/{len(prompts)}: {stats}", flush=True)
    for fam, fb in by_family.items():
        fb["pass_rate"] = round(fb["pass_rate"] / max(fb["n"], 1), 3)
    print(json.dumps({"stats": stats, "by_family": by_family}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

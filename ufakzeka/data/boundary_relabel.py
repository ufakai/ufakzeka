"""Find which trained facts a checkpoint actually retains, and relabel the rest (R-Tuning, arXiv 2311.09677).

The knowledge probe showed the failure this addresses: 18 percent of v27's long tail answers were confident
inventions, because SFT taught it teacher-sized answers to questions its 151M parameters cannot hold. R-Tuning's
fix is to probe the trained model and relabel the questions it consistently misses to an honest answer, so
the next round teaches the boundary instead of the bluff.

This probes a checkpoint on the wiki fact set: k samples per question, correctness by content overlap with
the gold lead sentence (the distinctive words of the answer, not its exact wording). Output is the same jsonl
with a verdict per row:

  keep      the model gets it right in most samples; the fact is retained, keep the gold answer
  relabel   consistently wrong; the honest answer replaces the gold one in the next build
  mixed     sometimes right; kept as gold (the fact is in there, more phrasings help extraction)

Runs on a GPU next to the checkpoint (Colab after SFT, or the Mac overnight):

  python -m ufakzeka.data.boundary_relabel --model /ckpt/x/hf --facts wiki_facts.jsonl \
      --out wiki_facts_probed.jsonl --k 3 --n 4000
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata

# frequent Turkish function and copula words that carry no content; everything else in a gold answer counts
_STOP = set("""bir bu da de den din dir dur dür dız diz dı di du dü en gibi ile ve veya ya yer alan olan olarak
kabul edilen adlı isimli kendi çok daha sonra önce arasında üzere göre kadar için tarafından yılında yıllarında
the of and""".split())


def _norm(s: str) -> str:
    # casefold before stripping: Turkish İ lowercases to i plus a combining dot, and the dot must go while
    # the i stays. Without NFKD the whole letter was lost.
    s = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", s).casefold())
    s = "".join(c for c in s if not unicodedata.combining(c) or c in "\u0327")  # keep ç/ş cedillas
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"[^a-zçğıöşü0-9\s]", " ", s)


def content_words(gold: str, title: str) -> set[str]:
    """The words that make the answer this answer: 4+ letters or a number, minus stopwords and the title
    itself (an answer that only repeats the question must not count as correct)."""
    tw = set(_norm(title).split())
    out = set()
    for w in _norm(gold).split():
        if w in _STOP or w in tw:
            continue
        if len(w) >= 4 or w.isdigit():
            out.add(w)
    return out


def is_correct(answer: str, gold_words: set[str], min_hits: int = 2) -> bool:
    """Substring match on a 5 character stem: Turkish agglutination means the gold "hükümdardır" must also
    match "hükümdarı" and "hükümdar olan". Crude stemming is enough because a hit needs 2 words to agree."""
    if not gold_words:
        return False
    a = _norm(answer)
    hits = sum(1 for w in gold_words if (w[:5] if len(w) > 5 else w) in a)
    need = min(min_hits, max(1, len(gold_words) // 4))
    return hits >= need


RELABEL = {
    "person": "Bu isim bana tanıdık gelmiyor, yanlış bilgi vermek istemem. Kim olduğunu kısaca yazarsan elimden geleni anlatırım.",
    "place": "Burası hakkında güvenilir bilgim yok, uydurmak istemem. Hangi bölgede olduğunu söylersen bildiklerimle yardımcı olurum.",
    "thing": "Bu konuda güvenilir bilgim yok, yanlış bir şey söylemek istemem. Ne hakkında olduğunu bir cümleyle anlatırsan devam edebilirim.",
}


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--facts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n", type=int, default=4000, help="probe the top n rows; the rest pass through as keep")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16 if dev != "cpu" else torch.float32).to(dev).eval()
    end = tok.convert_tokens_to_ids("<|im_end|>")
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in open(a.facts, encoding="utf-8")]
    # Only rows whose relabel would be safe to teach are worth probing. The first run relabeled "Kartal" and
    # "C (programlama dili)": a generic word or a stripped disambiguator makes the bare question ambiguous, so
    # a miss against the one Wikipedia sense proves nothing and the taught refusal would be harmful. Those
    # rows pass through untouched, which also cuts the GPU time to the eligible third.
    def eligible(r):
        return r["kind"] == "person" and "(" not in r["title"] and len(r["title"].split()) >= 2
    tally = {"keep": 0, "relabel": 0, "mixed": 0, "passthrough": 0}
    with open(a.out, "w", encoding="utf-8") as f:
        for start in range(0, len(rows), a.batch):
            chunk = rows[start:start + a.batch]
            if start >= a.n:
                for r in chunk:
                    r["verdict"] = "keep"
                    tally["passthrough"] += 1
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                continue
            skipped = [r for r in chunk if not eligible(r)]
            for r in skipped:
                r["verdict"] = "keep"
                tally["passthrough"] += 1
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            chunk = [r for r in chunk if eligible(r)]
            if not chunk:
                continue
            prompts = [tok.apply_chat_template([{"role": "user", "content": r["forms"][0]}],
                                               add_generation_prompt=True, tokenize=False) for r in chunk]
            correct = [0] * len(chunk)
            gold = [content_words(r["cevap"], r["title"]) for r in chunk]
            for seed in range(a.k):
                torch.manual_seed(seed)
                enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
                out = model.generate(**enc, max_new_tokens=90, do_sample=True, temperature=0.45, top_p=0.9,
                                     no_repeat_ngram_size=12, eos_token_id=[end, tok.eos_token_id],
                                     pad_token_id=tok.pad_token_id)
                for i, seq in enumerate(out):
                    ans = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                    if is_correct(ans, gold[i]):
                        correct[i] += 1
            for r, c in zip(chunk, correct):
                if c == 0:
                    r["verdict"] = "relabel"
                    r["cevap_durust"] = RELABEL[r["kind"]]
                elif c == a.k:
                    r["verdict"] = "keep"
                else:
                    r["verdict"] = "mixed"
                tally[r["verdict"]] += 1
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if (start // a.batch) % 20 == 0:
                done = min(start + a.batch, a.n)
                print(f"{done}/{min(a.n, len(rows))} probed: {tally}", flush=True)
    print(json.dumps(tally))


if __name__ == "__main__":
    main()

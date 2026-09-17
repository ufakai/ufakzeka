"""Run the scenario suite against a checkpoint and report every failure with its cause.

Conversations advance turn by turn; scenarios are batched across the batch dimension at each turn index, so
a suite of a thousand multi-turn conversations runs in minutes on an A100 and in an hour or two on a Mac.

  python scripts/run_scenarios.py --model models/ufakzeka-1-instruct-v34/hf --suite scenarios.json \
      --out evals/scenarios_v34.json --seeds 2 --temperature 0.3

Reported: pass rate per family and overall, plus a failure list (scenario id, turn, user text, answer, the
check that failed) that the data fixes are read from.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

GLOBAL_NOT = [
    (r"küçük bir model", "self-deprecation"),
    (r"(çalışıyorum|firmada|uzman olarak|geliştirici olarak|şirkette)", "invented persona"),
    (r"[_a-z]{6,}_[a-z]", "code-like garbage"),
]


def check_turn(answer: str, checks: list, meta: dict) -> list[str]:
    a = answer.strip()
    fails = []
    for pat, label in GLOBAL_NOT:
        if re.search(pat, a, re.I):
            fails.append(f"global: {label}")
    lines = [l.strip() for l in a.split("\n") if l.strip()]
    if lines:
        top = max(lines.count(l) for l in set(lines))
        # folk poems carry refrains ("Bir yonca biter" six times in nineteen lines is a real poem), so a
        # poem only fails when most of it is one line; prose fails at four repeats
        is_poem = meta.get("family") == "poem" or "şiir" in meta.get("user", "").lower() or "siir" in meta.get("user", "").lower()
        limit = max(4, len(lines) // 2 + 1) if is_poem else 4
        if top >= limit:
            fails.append("global: line repeated 4x" if limit == 4 else "global: poem is mostly one line")
    for kind, arg in checks:
        if kind == "must" and not re.search(arg, a, re.I | re.M):
            fails.append(f"must /{arg[:50]}/")
        elif kind == "not" and re.search(arg, a, re.I | re.M):
            fails.append(f"not /{arg[:50]}/")
        elif kind == "min_words" and len(a.split()) < arg:
            fails.append(f"min_words {arg} (got {len(a.split())})")
        elif kind == "max_words" and len(a.split()) > arg:
            fails.append(f"max_words {arg} (got {len(a.split())})")
        elif kind == "ends" and (not a or a[-1] not in ".!?\"”)"):
            fails.append("does not end a sentence")
        elif kind == "differs":
            prev = (meta.get("prev_answers") or [])
            if arg < len(prev):
                wa, wb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", prev[arg].lower()))
                if wa and wb and len(wa & wb) / len(wa | wb) > 0.6:
                    fails.append(f"differs from turn {arg}: too similar")
        elif kind == "distinct_items":
            items = [re.sub(r"^\W+", "", x).lower() for x in re.split(r"(?:^|\n|\s)(?:\d+[\).:]|[•\-])\s*", a) if x.strip()]
            if len(items) < arg:  # a prose list: "müze gezisi, parkta yürüyüş, film gecesi ve masa oyunu"
                body = re.sub(r"^[^:]{0,40}:\s*", "", a.strip())
                items = [x.strip().lower() for x in re.split(r",|;|\bve\b|\bya da\b|\n", body) if len(x.strip()) > 2]
            heads = [" ".join(it.split()[:2]) for it in items]
            if len(items) < arg or len(set(heads)) < arg:
                fails.append(f"distinct_items {arg} (got {len(items)} items, {len(set(heads))} distinct)")
        elif kind == "starts_with_letter":
            words = [w for w in re.findall(r"[a-zçğıöşü]{3,}", a.lower()) if w not in ("tamam", "peki", "sıra", "sende", "şimdi", "olur", "güzel", "sonra", "kelime", "harf", "harfle", "başla", "başlayan", "devam", "söyle", "ile", "için", "bir")]
            if not any(w.startswith(arg) for w in words):
                fails.append(f"no word starting with '{arg}'")
        elif kind == "no_loop" and re.search(r"(\b\w+\b)(\s+\1){3,}", a):
            fails.append("word loop")
        elif kind == "contains" and arg not in a:
            fails.append(f"contains {arg}")
    return fails


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--family", default=None, help="run one family only")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    suite = json.load(open(a.suite, encoding="utf-8"))
    if a.family:
        suite = [s for s in suite if s["family"] == a.family]
    if a.limit:
        suite = suite[:a.limit]
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 if dev == "cpu" else (torch.bfloat16 if (dev == "mps" or torch.cuda.is_bf16_supported()) else torch.float16)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(dev).eval()
    end = tok.convert_tokens_to_ids("<|im_end|>")
    gen_cfg = json.load(open(f"{a.model}/generation_config.json"))
    ngram = gen_cfg.get("no_repeat_ngram_size", 12)

    per_family = defaultdict(lambda: [0, 0])
    failures = []
    transcripts = []
    for seed in range(a.seeds):
        torch.manual_seed(seed)
        # group by number of turns so a batch advances in lockstep
        by_len = defaultdict(list)
        for s in suite:
            by_len[len(s["turns"])].append(s)
        for n_turns, group in sorted(by_len.items()):
            for start in range(0, len(group), a.batch):
                chunk = group[start:start + a.batch]
                hist = [[] for _ in chunk]
                for t in range(n_turns):
                    for h, s in zip(hist, chunk):
                        h.append({"role": "user", "content": s["turns"][t]})
                    texts = [tok.apply_chat_template(h, add_generation_prompt=True, tokenize=False) for h in hist]
                    enc = tok(texts, return_tensors="pt", padding=True).to(dev)
                    with torch.no_grad():
                        out = model.generate(**enc, max_new_tokens=520, do_sample=a.temperature > 0,
                                             temperature=max(a.temperature, 1e-5), top_p=0.9, top_k=40, no_repeat_ngram_size=ngram,
                                             eos_token_id=[end, tok.eos_token_id], pad_token_id=tok.pad_token_id)
                    for i, (h, s) in enumerate(zip(hist, chunk)):
                        ans = tok.decode(out[i][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                        h.append({"role": "assistant", "content": ans})
                        checks = s["checks"].get(str(t))
                        if checks is not None:
                            fails = check_turn(ans, [tuple(c) for c in checks], {**s.get("meta", {}), "family": s["family"], "user": s["turns"][t], "prev_answers": [m["content"] for m in h[:-1] if m["role"] == "assistant"]})
                            per_family[s["family"]][1] += 1
                            if fails:
                                failures.append({"id": s["id"], "family": s["family"], "seed": seed, "turn": t,
                                                 "user": s["turns"][t], "answer": ans[:400], "fails": fails})
                            else:
                                per_family[s["family"]][0] += 1
                for h, s in zip(hist, chunk):
                    transcripts.append({"id": s["id"], "seed": seed, "messages": h})
            done = sum(v[1] for v in per_family.values())
            print(f"seed {seed}: {n_turns}-turn group done, {done} checks so far", flush=True)

    status = defaultdict(set)
    for fl in failures:
        status[(fl["id"], fl["turn"])].add(fl["seed"])
    unstable = sum(1 for k, seeds in status.items() if len(seeds) < a.seeds) if a.seeds > 1 else 0
    total_ok = sum(v[0] for v in per_family.values()); total = sum(v[1] for v in per_family.values())
    if a.seeds > 1:
        print(f"unstable turns (pass in some seeds, fail in others): {unstable} of {total // a.seeds} checked turns ({100 * unstable * a.seeds / max(total, 1):.1f}%)")
    print(f"\n{a.model}: {total_ok}/{total} checks passed ({100 * total_ok / max(total, 1):.1f}%) over {a.seeds} seed(s)")
    print(f"{'family':16} {'pass':>6} {'n':>5}")
    for fam, (ok, n) in sorted(per_family.items(), key=lambda x: x[1][0] / max(x[1][1], 1)):
        print(f"{fam:16} {100 * ok / max(n, 1):5.1f}% {n:5}")
    json.dump({"model": a.model, "seeds": a.seeds, "temperature": a.temperature,
               "summary": {f: {"pass": v[0], "n": v[1]} for f, v in per_family.items()},
               "overall": {"pass": total_ok, "n": total}, "failures": failures, "transcripts": transcripts},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {a.out}: {len(failures)} failing turns")


if __name__ == "__main__":
    main()

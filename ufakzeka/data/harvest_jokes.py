"""Public domain Turkish jokes and short tales from Vikikaynak: the Nasreddin Hoca and Karadeniz fıkra categories
(traditional, anonymous), plus the few folk tales there. The generated joke chains (gen_freechat --kind joke)
mangle the classics; the real texts sit next to them so the model learns the actual fıkras.

  python -m ufakzeka.data.harvest_jokes --out jokes.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time

from ufakzeka.data.harvest_poems import api, members, plaintext

CATS = ["Kategori:Fıkralar", "Kategori:Nasreddin Hoca fıkraları", "Kategori:Karadeniz fıkraları", "Kategori:Masallar", "Kategori:Türk masalları", "Kategori:Halk hikâyeleri"]


MAX_WORDS = 400


def prose(text: str | None) -> str | None:
    if not text:
        return None
    t = re.split(r"\n==", text)[0]
    t = re.sub(r"\n{2,}", "\n", t.strip())
    lines = [re.sub(r"\s+", " ", l).strip() for l in t.split("\n") if l.strip()]
    t = " ".join(lines)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    w = t.split()
    if not (15 <= len(w) <= MAX_WORDS) or not re.search(r"[çğıöşüÇĞİÖŞÜ]", t):
        return None
    if re.search(r"(Vikikaynak|telif|lisans|kaynak:|Kaynak:)", t, re.I):
        return None
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=3000)
    ap.add_argument("--cats", default="", help="comma separated categories instead of the default list")
    ap.add_argument("--max-words", type=int, default=400, help="longest text kept (Dede Korkut stories run past 1,000 words)")
    a = ap.parse_args()
    global MAX_WORDS
    MAX_WORDS = a.max_words
    seen, kept = set(), 0
    queue, visited = list(a.cats.split(",") if a.cats else CATS), set()
    with open(a.out, "w", encoding="utf-8") as f:
        while queue and kept < a.max:
            cat = queue.pop(0)
            if cat in visited:
                continue
            visited.add(cat)
            try:
                for m in members(cat):
                    title = m["title"]
                    if title.startswith("Kategori:"):
                        queue.append(title)
                        continue
                    if title in seen:
                        continue
                    seen.add(title)
                    try:
                        t = prose(plaintext(title))
                    except Exception:
                        continue
                    if t:
                        kind = "tale" if any(k in cat.lower() for k in ("masal", "hik", "korkut", "efsane", "destan")) else "joke"
                        f.write(json.dumps({"title": re.sub(r"\s*\(.*?\)\s*$", "", title), "text": t, "kind": kind, "cat": cat}, ensure_ascii=False) + "\n")
                        f.flush()
                        kept += 1
                        if kept % 25 == 0:
                            print(f"{kept} kept, {len(seen)} pages, cat {cat}", flush=True)
                    time.sleep(0.6)
            except Exception as e:
                print("category failed:", cat, e, flush=True)
    print(f"DONE {kept} texts from {len(seen)} pages", flush=True)


if __name__ == "__main__":
    main()

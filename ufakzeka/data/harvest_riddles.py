"""Folk riddles from Vikikaynak ("Türk halk edebiyatında bilmeceler": regional collections, anonymous, public domain).
Each page lists riddles as ":line" blocks ending with ":(answer)". The v83 hand test made up an invalid riddle
("Akşam gelir, sabah gider, herkesi güldürür"); real ones with real answers replace the invented ones.

  python -m ufakzeka.data.harvest_riddles --out riddles.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time

from ufakzeka.data.harvest_poems import api, members

CAT = "Kategori:Türk halk edebiyatında bilmeceler"


def wikitext(title: str) -> str:
    d = api({"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main", "titles": title})
    for pg in d.get("query", {}).get("pages", {}).values():
        revs = pg.get("revisions") or []
        return revs[0].get("slots", {}).get("main", {}).get("*", "") if revs else ""
    return ""


def parse(w: str):
    for _ in range(3):
        w = re.sub(r"\{\{[^{}]*\}\}", "", w)
    w = re.sub(r"\[\[Kategori:[^\]]*\]\]", "", w)
    blocks = re.split(r"\n\s*\n", w.strip())
    out = []
    for b in blocks:
        lines = [re.sub(r"\s+", " ", l.lstrip(":* ").strip()) for l in b.split("\n") if l.strip()]
        if len(lines) < 2:
            continue
        m = re.match(r"^\(?(.+?)\)?$", lines[-1])
        if not lines[-1].startswith("(") or not m:
            continue
        ans = m.group(1).strip().rstrip(".")
        body = [l for l in lines[:-1] if not l.startswith("(")]
        if not (1 <= len(body) <= 8) or not re.search(r"[çğıöşüÇĞİÖŞÜ]", " ".join(body)) or len(ans) > 40:
            continue
        out.append({"riddle": "\n".join(body), "answer": ans})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kept, seen = 0, set()
    with open(a.out, "w", encoding="utf-8") as f:
        for m in members(CAT):
            title = m["title"]
            if title.startswith("Kategori:"):
                continue
            for r in parse(wikitext(title)):
                key = r["riddle"].lower()[:60]
                if key in seen:
                    continue
                seen.add(key)
                f.write(json.dumps({**r, "source": title}, ensure_ascii=False) + "\n")
                kept += 1
            print(title, kept, flush=True)
            time.sleep(0.8)
    print(f"DONE {kept} riddles", flush=True)


if __name__ == "__main__":
    main()

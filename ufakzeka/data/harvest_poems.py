"""Public domain Turkish poems from Vikikaynak, for the poem block that is starving at 85 items.

"bir siir yaz" came back as a prose sentence about the fall of Constantinople (user test 2026-09-01): with 85
poems in the mix the request shape barely exists. Vikikaynak hosts public domain Turkish poetry (authors dead
70+ years) behind the MediaWiki API; this walks the poetry categories, keeps texts that look like verse
(short lines, several stanzas, reasonable length) and writes the same jsonl poem_examples already reads.

  python -m ufakzeka.data.harvest_poems --out poems.jsonl --max 1200
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request

API = "https://tr.wikisource.org/w/api.php"
CATS = ["Kategori:Şiirler", "Kategori:Türkçe şiirler", "Kategori:Divan edebiyatı", "Kategori:Halk edebiyatı"]
UA = {"User-Agent": "ufakzeka-data/1.0 (research; contact: github ufakzeka)"}


def api(params: dict) -> dict:
    # the Hetzner IP gets 429s at any burst; back off and retry rather than losing a whole category subtree
    q = urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(f"{API}?{q}", headers=UA)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            raise
    raise RuntimeError("rate limited after retries")


def members(cat: str):
    cont = None
    while True:
        p = {"action": "query", "list": "categorymembers", "cmtitle": cat, "cmlimit": "500", "cmtype": "page|subcat"}
        if cont:
            p["cmcontinue"] = cont
        d = api(p)
        for m in d.get("query", {}).get("categorymembers", []):
            yield m
        cont = d.get("continue", {}).get("cmcontinue")
        if not cont:
            return


def plaintext(title: str) -> str | None:
    """Raw wikitext, plain poems only. Scan-backed pages (the {{#tag:pages}} transclusions) render no verse
    through any API route, so they are skipped; most poems on Vikikaynak are ordinary wikitext."""
    d = api({"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main", "titles": title})
    for pg in d.get("query", {}).get("pages", {}).values():
        revs = pg.get("revisions")
        if not revs:
            return None
        w = revs[0].get("slots", {}).get("main", {}).get("*", "")
        if "#tag:pages" in w or "<pages" in w:
            return None  # scan backed
        # strip templates (two passes cover simple nesting), comments, refs, tags, links, emphasis
        for _ in range(3):
            w = re.sub(r"\{\{[^{}]*\}\}", "", w)
        w = re.sub(r"<!--.*?-->|<ref[^>]*>.*?</ref>|<ref[^>]*/>", "", w, flags=re.S)
        w = re.sub(r"<[^>]+>", "\n", w)
        w = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", w)
        w = re.sub(r"'{2,}", "", w)
        w = re.sub(r"^[=*:;#].*$", "", w, flags=re.M)
        w = re.sub(r"^\s*(Kategori|Category):.*$", "", w, flags=re.M | re.I)
        return w.strip()
    return None


def looks_like_poem(text: str) -> str | None:
    t = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    # drop license/footer sections
    t = re.split(r"\n==", t)[0].strip()
    lines = [l for l in t.split("\n") if l.strip()]
    if not (6 <= len(lines) <= 60):
        return None
    words = t.split()
    if not (30 <= len(words) <= 400):
        return None
    # verse: most nonempty lines are short
    short = sum(1 for l in lines if len(l.split()) <= 10)
    if short / len(lines) < 0.8:
        return None
    if not re.search(r"[çğıöşüÇĞİÖŞÜ]", t):
        return None
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=1200)
    a = ap.parse_args()
    # resumable: a previous run's poems are kept and their titles skipped, and every poem is flushed as it is
    # written, so an interrupted run loses nothing (the first run buffered everything until exit)
    seen_titles, kept = set(), 0
    import os
    if os.path.exists(a.out):
        for line in open(a.out, encoding="utf-8"):
            try:
                seen_titles.add(json.loads(line)["title"]); kept += 1
            except Exception:
                pass
        print(f"resuming with {kept} poems already on disk", flush=True)
    queue = list(CATS)
    visited_cats = set()
    with open(a.out, "a", encoding="utf-8") as f:
        while queue and kept < a.max:
            cat = queue.pop(0)
            if cat in visited_cats:
                continue
            visited_cats.add(cat)
            try:
                for m in members(cat):
                    if kept >= a.max:
                        break
                    title = m["title"]
                    if title.startswith("Kategori:"):
                        if len(visited_cats) < 200:
                            queue.append(title)
                        continue
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)
                    try:
                        poem = looks_like_poem(plaintext(title))
                    except Exception:
                        continue
                    if poem:
                        # the keys poem_examples already reads
                        f.write(json.dumps({"title": title, "text": poem}, ensure_ascii=False) + "\n")
                        f.flush()
                        kept += 1
                        if kept % 50 == 0:
                            print(f"{kept} poems, {len(seen_titles)} pages seen, {len(visited_cats)} cats", flush=True)
                    time.sleep(0.6)  # be polite: the API 429s bursts from datacenter IPs
                    if len(seen_titles) % 25 == 0:
                        print(f"  {len(seen_titles)} pages seen, {kept} kept", flush=True)
            except Exception as e:
                print("category failed:", cat, e, flush=True)
    print(f"done: {kept} poems from {len(seen_titles)} pages")


if __name__ == "__main__":
    main()

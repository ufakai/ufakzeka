"""Turkish general knowledge as question and answer pairs, straight from the Wikipedia lead sentences.

Why this exists. The knowledge probe found v28 answering "Bu kişiyi tanımıyorum" to Nazım Hikmet, Yunus Emre,
Piri Reis, Orhan Pamuk and Sabiha Gökçen. Abstention is the right behaviour for the long tail and we trained
it deliberately, but the counterweight was 50 hand written facts against roughly 1,900 abstention examples,
so it generalised straight through the household names. You cannot fix that ratio by hand.

Wikipedia lead sentences are already the shape we need ("Cengiz Han, Moğol İmparatorluğu'nun kurucusu ve ilk
Han'ı olan Moğol komutan ve hükümdardır"), they are CC BY-SA, and the August 2026 dump is already extracted
on Hetzner. Each entity also gets several question phrasings, which is the part that decides whether a stored
fact can be retrieved by a question at all (Physics of Language Models 3.1): one phrasing stores, several
phrasings make it answerable.

The 50 entities in scripts/knowledge_probe.py are excluded, or the probe stops measuring anything.

  python -m ufakzeka.data.wiki_facts --wiki /data/ufakzeka/wiki/trwiki-2026-08.jsonl.zst \
      --out /data/ufakzeka/facts/wiki_facts.jsonl --n 30000
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata

# lead-sentence cues that decide which question a title deserves
# strongly biographical cues only: "kurucu" and the like are left out because place articles use them too
PERSON = re.compile(r"\b(doğdu|doğumlu|d\.\s*\d|ö\.\s*\d|ölmüştür|yazar|şair|oyuncu|aktör|aktris|şarkıcı|"
                    r"müzisyen|ressam|heykeltıraş|romancı|öykücü|çevirmen|siyasetçi|devlet adamı|milletvekili|"
                    r"başbakan|cumhurbaşkanı|devlet başkanı|hükümdar|padişah|sultan|halife|komutan|general|"
                    r"subay|bilim insanı|fizikçi|kimyager|matematikçi|tarihçi|filozof|düşünür|mutasavvıf|"
                    r"mimar|mühendis|hekim|futbolcu|basketbolcu|sporcu|antrenör|yönetmen|senarist|gazeteci|"
                    r"akademisyen|profesör|besteci|piyanist|sanatçı|kâşif|denizci|gezgin|imparator|kral|"
                    r"kraliçe|prens|prenses|aziz|peygamber|lider|önder|aktivist|iş insanı|sanayici)", re.I)
PLACE = re.compile(r"\b(ilçesi|iline|ilinin|ilinde|şehri|kenti|köyü|kasabası|mahallesi|beldesi|semti|başkenti|"
                   r"ülkesi|devleti|dağı|gölü|nehri|ırmağı|denizi|okyanusu|adası|yarımadası|ovası|platosu|"
                   r"bölgesi|vadisi|körfezi|boğazı|barajı|antik kenti|ören yeri|höyüğü)", re.I)

SKIP_TITLE = re.compile(r"\(anlam ayrımı\)|listesi$|^liste|^\d{1,4}$|^\d{1,4}(\s|-)|yılında|^Vikipedi|"
                        r"kategori:|şablon:|^Portal", re.I)
# wiki leftovers that make a bad training answer
NOISE = re.compile(r"(https?://|\{\{|\}\}|\[\[|\]\]|\bref\b|<|>|thumb\b|px\b|\|)")
TR = re.compile(r"[çğıöşüÇĞİÖŞÜ]")


def held_out() -> set[str]:
    """Titles the knowledge probe measures, which must never be trained on."""
    from scripts.knowledge_probe import CASES
    suffixes = (" kimdir", " nedir", " neresi", " nerededir", " neden olur", " nasıl olur",
                " ne işe yarar", " kaç", " hangisi")
    out = set()
    for q, _, _ in CASES:
        t = q
        for s in suffixes:
            if t.lower().endswith(s):
                t = t[: -len(s)]
                break
        out.add(norm(t))
    return out


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower().strip()
    return re.sub(r"\s+", " ", s)


def lead_sentence(text: str, title: str) -> str | None:
    body = text.split("\n", 1)[-1] if text.startswith("#") else text
    # the lead ends at the first section heading; without this "Boris Yeltsin nedir" answered with
    # "... devlet başkanı. ## Gençliği Rus kökenli bir çiftçinin oğludur."
    body = re.split(r"\n\s*#{2,}\s", body)[0]
    body = body.strip()
    # drop the parenthetical birth/death and foreign-name clutter, which reads badly out loud and is where
    # most of the wiki markup survives
    body = re.sub(r"\s*\([^()]{0,160}\)", "", body)
    body = re.sub(r"\s+", " ", body)
    body = re.sub(r"\s*,\s*(?=[,.;])", "", body)  # the paren strip leaves ", ," behind
    body = re.sub(r"\s+([,.;])", r"\1", body).strip()
    sents = re.split(r"(?<=[.!?])\s+", body)
    if not sents:
        return None
    lead = sents[0]
    if len(lead.split()) < 12 and len(sents) > 1:
        lead = lead + " " + sents[1]
    return lead.strip()


def question_forms(title: str, lead: str) -> tuple[str, list[str]]:
    title = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip() or title
    if PERSON.search(lead) and not PLACE.search(lead):
        kind = "person"
        forms = [f"{title} kimdir", f"{title} kim", f"{title} kimdir kısaca anlatır mısın",
                 f"{title} hakkında bilgi verir misin", f"kimdir {title}"]
    elif PLACE.search(lead):
        kind = "place"
        forms = [f"{title} neresi", f"{title} nerededir", f"{title} nedir",
                 f"{title} hakkında bilgi verir misin", f"{title} nerede"]
    else:
        kind = "thing"
        forms = [f"{title} nedir", f"{title} ne demek", f"{title} nedir kısaca",
                 f"{title} hakkında bilgi verir misin", f"{title} ne"]
    return kind, forms


def build(wiki: str, out: str, n: int, min_chars: int) -> dict:
    import zstandard

    skip = held_out()
    stats = {"seen": 0, "kept": 0, "held_out": 0, "no_lead": 0, "noisy": 0, "short_article": 0}
    rows = []
    with open(wiki, "rb") as fh:
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        for line in _lines(reader):
            stats["seen"] += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            title, text = d.get("title", ""), d.get("text", "")
            if not title or SKIP_TITLE.search(title):
                continue
            if len(text) < min_chars:
                stats["short_article"] += 1
                continue
            if norm(title) in skip:
                stats["held_out"] += 1
                continue
            lead = lead_sentence(text, title)
            if not lead:
                stats["no_lead"] += 1
                continue
            if NOISE.search(lead) or not TR.search(lead):
                stats["noisy"] += 1
                continue
            words = lead.split()
            if not (12 <= len(words) <= 45) or not lead.endswith((".", "!", "?")):
                stats["noisy"] += 1
                continue
            # the answer must actually be about the title, not a sentence that lost it to the paren strip
            head = norm(title.split("(")[0]).split()
            if not head or head[0] not in norm(lead):
                stats["noisy"] += 1
                continue
            kind, forms = question_forms(title, lead)
            rows.append({"title": title, "kind": kind, "forms": forms, "cevap": lead,
                         "rank": int(d["id"].split(":")[-1])})
            stats["kept"] += 1
    # notability proxy: page id, which is creation order. Ranking by article length instead surfaced a French
    # conchologist; the lowest ids are Cengiz Han, Linux, Karl Marx, Bilgisayar, Kimya, Edebiyat, which is
    # exactly the set an assistant gets asked about. The length floor above supplies the substance.
    rows.sort(key=lambda r: r["rank"])
    rows = rows[:n]
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            r.pop("rank")
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["written"] = len(rows)
    by_kind: dict[str, int] = {}
    for r in rows:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
    stats["by_kind"] = by_kind
    return stats


def _lines(reader):
    buf = b""
    while True:
        chunk = reader.read(1 << 20)
        if not chunk:
            break
        buf += chunk
        *lines, buf = buf.split(b"\n")
        for ln in lines:
            if ln:
                yield ln
    if buf:
        yield buf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--min-chars", type=int, default=2500)
    a = ap.parse_args()
    print(json.dumps(build(a.wiki, a.out, a.n, a.min_chars), ensure_ascii=False, indent=1))

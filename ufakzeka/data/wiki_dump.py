"""Extract article text from a Turkish Wikipedia XML dump into jsonl.zst.

Keeps main namespace articles that are not redirects, strips wikitext with
mwparserfromhell, drops tables, references and templates, keeps section headings as
markdown. Output rows: {"id", "url", "title", "text"}. Runs streaming, low memory."""

from __future__ import annotations

import argparse
import bz2
import json
import re
import xml.etree.ElementTree as ET

import mwparserfromhell
import zstandard as zstd

NS = "{http://www.mediawiki.org/xml/export-0.11/}"


def clean(wikitext: str) -> str:
    code = mwparserfromhell.parse(wikitext)
    for t in code.filter_templates(recursive=True):
        try:
            code.remove(t)
        except ValueError:
            pass
    for tag in code.filter_tags(recursive=True):
        if tag.tag.lower() in ("ref", "table", "gallery", "math", "timeline", "score"):
            try:
                code.remove(tag)
            except ValueError:
                pass
    out = []
    for node in code.nodes:
        if isinstance(node, mwparserfromhell.nodes.Heading):
            out.append("\n" + "#" * node.level + " " + node.title.strip_code().strip() + "\n")
        else:
            out.append(node.__strcode__() if hasattr(node, "__strcode__") else str(mwparserfromhell.parse(str(node)).strip_code()))
    text = "".join(out)
    text = re.sub(r"\[\[(?:Dosya|File|Kategori|Category):[^\]]*\]\]", "", text)
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.S)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("out")
    ap.add_argument("--min-chars", type=int, default=300)
    a = ap.parse_args()
    n = kept = 0
    with open(a.out, "wb") as raw, zstd.ZstdCompressor(level=6).stream_writer(raw) as w, bz2.open(a.dump, "rb") as f:
        root = None
        for ev, el in ET.iterparse(f, events=("start", "end")):
            if ev == "start":
                if root is None:
                    root = el
                continue
            if el.tag != NS + "page":
                continue
            n += 1
            ns = el.findtext(NS + "ns")
            if ns == "0" and el.find(NS + "redirect") is None:
                title = el.findtext(NS + "title") or ""
                pid = el.findtext(NS + "id")
                rev = el.find(NS + "revision")
                wt = rev.findtext(NS + "text") if rev is not None else ""
                try:
                    text = clean(wt or "")
                except Exception:
                    text = ""
                if len(text) >= a.min_chars:
                    doc = {"id": f"trwiki:{pid}", "url": "https://tr.wikipedia.org/wiki/" + title.replace(" ", "_"), "title": title, "text": f"# {title}\n{text}"}
                    w.write((json.dumps(doc, ensure_ascii=False) + "\n").encode()); kept += 1
            el.clear()
            root.clear()  # drop finished pages from the tree, otherwise memory grows without bound
            if n % 100000 == 0:
                print(n, kept, flush=True)
    print("pages", n, "kept", kept)


if __name__ == "__main__":
    main()

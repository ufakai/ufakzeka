"""N-gram decontamination against evaluation sets.

A document is dropped if it contains any 13-word n-gram from an eval item. Items shorter
than 13 words contribute their full word sequence (minimum 8 words). Tokenisation is
lowercase words only, so the match is robust to punctuation and casing."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable

import regex

_word_re = regex.compile(r"\p{L}+|\d+")
NGRAM = 13
MIN_NGRAM = 8


def words(text: str) -> list[str]:
    return [w.lower() for w in _word_re.findall(text)]


def _strings_from_item(item: dict) -> Iterable[str]:
    for v in item.values():
        if isinstance(v, str):
            yield v
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, str):
                    yield x
        elif isinstance(v, dict):
            yield from _strings_from_item(v)


def build_index(eval_dir: str) -> dict[int, set[tuple[str, ...]]]:
    """Return n-gram sets keyed by n-gram length (13 for normal items, shorter for short items)."""
    grams: dict[int, set[tuple[str, ...]]] = {}
    for fn in sorted(os.listdir(eval_dir)):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(eval_dir, fn)) as f:
            for line in f:
                for s in _strings_from_item(json.loads(line)):
                    w = words(s)
                    if len(w) >= NGRAM:
                        g = grams.setdefault(NGRAM, set())
                        for i in range(len(w) - NGRAM + 1):
                            g.add(tuple(w[i:i + NGRAM]))
                    elif len(w) >= MIN_NGRAM:
                        grams.setdefault(len(w), set()).add(tuple(w))
    return grams


def is_contaminated(text: str, grams: dict[int, set[tuple[str, ...]]]) -> bool:
    w = words(text)
    n = len(w)
    for k, g in grams.items():
        for i in range(max(n - k + 1, 0)):
            if tuple(w[i:i + k]) in g:
                return True
    return False

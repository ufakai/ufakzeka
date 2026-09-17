"""Heuristic document filters and PII masking for Turkish web text.

The upstream sources are already filtered (FineWeb-2 heuristics, model based quality
selection). This layer removes what those filters miss for our purposes: residual
boilerplate lines, documents that are mostly non Turkish or numeric, extreme repetition,
and personal identifiers. Thresholds follow the FineWeb-2 tur_Latn config where one exists."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import regex

TURKISH_LETTERS = "abcçdefgğhıijklmnoöprsştuüvyzâîûABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZÂÎÛ"
_word_re = regex.compile(r"\p{L}+")
_alpha_re = regex.compile(r"[\p{L}]")
_tr_alpha_re = re.compile(f"[{TURKISH_LETTERS}]")
_digit_re = re.compile(r"\d")

# Lines that are navigation, cookie notices, or social chrome. Matched case-insensitively
# against the stripped line; the line is dropped, the document is kept.
_BOILERPLATE_LINE = re.compile(
    r"^(devamını oku|devamı|daha fazla|tümünü gör|paylaş|yorum yap|yorumlar|"
    r"cevapla|beğen|abone ol|giriş yap|üye ol|kayıt ol|ana sayfa|anasayfa|"
    r"çerez(ler)?\b.*|cookie.*|gizlilik politikası|kullanım (koşulları|şartları)|"
    r"tüm hakları saklıdır.*|copyright.*|©.*|sonraki|önceki|"
    r"(facebook|twitter|instagram|whatsapp|telegram|linkedin|pinterest)(\s*\|.*)?|"
    r"reklam|sponsorlu.*|bu yazıyı paylaş.*|etiketler:.*|kategori(ler)?:.*)[\s.!:]*$",
    re.IGNORECASE,
)

# PII. Turkish national id is 11 digits with a checksum; phone and IBAN are shape based.
_TCKN = re.compile(r"(?<!\d)([1-9]\d{10})(?!\d)")
_PHONE = re.compile(r"(?<!\d)(?:\+90|0)[\s.-]?\(?5\d{2}\)?[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}(?!\d)")
_IBAN = re.compile(r"\bTR\d{2}(?:\s?\d{4}){5}\s?\d{2}\b")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _tckn_valid(s: str) -> bool:
    d = [int(c) for c in s]
    if d[0] == 0:
        return False
    odd, even = sum(d[0:9:2]), sum(d[1:8:2])
    if (odd * 7 - even) % 10 != d[9]:
        return False
    return sum(d[:10]) % 10 == d[10]


def mask_pii(text: str) -> tuple[str, int]:
    n = 0

    def _tckn(m):
        nonlocal n
        if _tckn_valid(m.group(1)):
            n += 1
            return "[TCKN]"
        return m.group(0)

    text = _TCKN.sub(_tckn, text)
    text, k = _IBAN.subn("[IBAN]", text); n += k
    text, k = _EMAIL.subn("[EMAIL]", text); n += k
    text, k = _PHONE.subn("[TEL]", text); n += k
    return text, n


def strip_boilerplate_lines(text: str) -> str:
    kept = []
    for line in text.split("\n"):
        s = line.strip()
        if s and len(s) < 80 and _BOILERPLATE_LINE.match(s):
            continue
        kept.append(line)
    out = "\n".join(kept)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


@dataclass
class FilterResult:
    keep: bool
    reason: str = ""
    text: str = ""
    pii_masked: int = 0
    n_words: int = 0


def filter_document(text: str, min_chars: int = 200, max_chars: int = 200_000,
                    light: bool = False, mask: bool = True) -> FilterResult:
    text = strip_boilerplate_lines(text)
    if len(text) < min_chars:
        return FilterResult(False, "short")
    if len(text) > max_chars:
        text = text[:max_chars]

    words = _word_re.findall(text)
    n_words = len(words)
    if n_words < 30:
        return FilterResult(False, "few_words")

    avg_len = sum(len(w) for w in words) / n_words
    if avg_len < 3 or avg_len > 21:  # FineWeb-2 tur_Latn bounds
        return FilterResult(False, "avg_word_len")

    if light:
        if mask:
            text, pii = mask_pii(text)
            return FilterResult(True, "", text, pii, n_words)
        return FilterResult(True, "", text, 0, n_words)

    alpha = len(_alpha_re.findall(text))
    if alpha / max(len(text), 1) < 0.55:
        return FilterResult(False, "low_alpha")
    tr_alpha = len(_tr_alpha_re.findall(text))
    if tr_alpha / max(alpha, 1) < 0.95:  # mostly non Turkish script letters
        return FilterResult(False, "non_turkish_letters")
    if len(_digit_re.findall(text)) / max(len(text), 1) > 0.15:
        return FilterResult(False, "numeric")

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) > 3:
        c = Counter(lines)
        dup_line_chars = sum(len(l) * (k - 1) for l, k in c.items() if k > 1)
        if dup_line_chars / max(len(text), 1) > 0.272:  # FineWeb-2 tur_Latn dup_line_frac
            return FilterResult(False, "dup_lines")

    # repetition statistics on a bounded window keeps cost flat for very long documents
    lowered = [w.lower() for w in words[:5000]]
    n_words = len(lowered)
    if n_words >= 50:
        for n, thr in ((2, 0.214), (3, 0.168), (4, 0.147)):
            grams = Counter(tuple(lowered[i:i + n]) for i in range(n_words - n + 1))
            top, cnt = grams.most_common(1)[0]
            if cnt * n / n_words > thr:
                return FilterResult(False, f"top_{n}gram")
        grams = Counter(tuple(lowered[i:i + 10]) for i in range(n_words - 9))
        dup = sum(10 * (k - 1) for k in grams.values() if k > 1)
        if dup / n_words > 0.103:
            return FilterResult(False, "dup_10gram")

    pii = 0
    if mask:
        text, pii = mask_pii(text)
    return FilterResult(True, "", text, pii, len(words))

"""Tokenizer evaluation: fertility (tokens per word), bytes per token, and round trip
on held out text, with the same numbers for reference tokenizers when available."""

from __future__ import annotations

import argparse
import glob

import regex

_word_re = regex.compile(r"\S+")


def measure(encode, texts: list[str]) -> dict:
    n_tok = n_words = n_bytes = 0
    for t in texts:
        n_tok += len(encode(t))
        n_words += len(_word_re.findall(t))
        n_bytes += len(t.encode())
    return {"tokens_per_word": round(n_tok / n_words, 3), "bytes_per_token": round(n_bytes / n_tok, 3), "tokens": n_tok}


def load_texts(pattern: str, limit: int) -> list[str]:
    import pyarrow.parquet as pq
    out = []
    for f in sorted(glob.glob(pattern)):
        out += pq.read_table(f, columns=["text"]).column("text").to_pylist()
        if len(out) >= limit:
            break
    return out[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--heldout-glob", required=True)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--reference", nargs="*", default=[])
    args = ap.parse_args()

    from tokenizers import Tokenizer
    texts = load_texts(args.heldout_glob, args.limit)
    tok = Tokenizer.from_file(args.tokenizer)
    bad = sum(1 for t in texts[:200] if tok.decode(tok.encode(t).ids) != t)
    print("ufakzeka", measure(lambda t: tok.encode(t).ids, texts), "roundtrip_failures", bad)
    for ref in args.reference:
        from transformers import AutoTokenizer
        r = AutoTokenizer.from_pretrained(ref)
        print(ref, measure(lambda t: r.encode(t, add_special_tokens=False), texts))


if __name__ == "__main__":
    main()

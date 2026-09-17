"""Train the ufakzeka byte-level BPE tokenizer.

Choices and why:
- Byte-level BPE (HF tokenizers): no unknown tokens, matches Kumru and Llama-3 practice,
  and the 2026 Turkish subword study (arXiv 2602.06942) found BPE/WordPiece strongest.
- Vocab 40,960 = 320 * 128: embeddings stay about 20% of a 150M model and the size is a
  multiple of 128 for tensor cores.
- Pre-tokenizer is the GPT-4 style regex with two changes for Turkish: it matches the
  full Unicode letter class (so ı İ ş ğ ç ö ü are letters) and splits digits one by one.
- No lowercasing and no NFKC normalisation: NFKC maps dotted capital I to I plus a
  combining dot, which breaks Turkish casing.
- Special tokens: <|endoftext|> for document boundaries, plus reserved chat tokens so
  the instruct model needs no vocab change."""

from __future__ import annotations

import argparse
import glob
import random

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"] + [f"<|reserved_{i}|>" for i in range(12)]

# GPT-4 pattern with letters generalised to \p{L} and digits split singly. The English
# contraction alternative is dropped: in Turkish the apostrophe introduces a suffix
# (Ankara'da), and the optional single non letter prefix keeps "'da" as one piece.
PATTERN = r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE(byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def iter_texts(files: list[str], max_bytes: int, seed: int = 1):
    import pyarrow.parquet as pq
    random.Random(seed).shuffle(files)
    seen = 0
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(columns=["text"], batch_size=2048):
            for t in batch.column("text").to_pylist():
                yield t
                seen += len(t.encode())
                if seen >= max_bytes:
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=40960)
    ap.add_argument("--max-gb", type=float, default=25.0)
    args = ap.parse_args()

    files = sorted(glob.glob(args.data_glob))
    assert files, args.data_glob
    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size - len(SPECIALS),
        min_frequency=2,
        special_tokens=[],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
        max_token_length=24,
    )
    tok.train_from_iterator(iter_texts(files, int(args.max_gb * 1e9)), trainer=trainer)
    tok.add_special_tokens(SPECIALS)
    assert tok.get_vocab_size() == args.vocab_size, tok.get_vocab_size()
    tok.save(args.out)
    print("saved", args.out, tok.get_vocab_size())


if __name__ == "__main__":
    main()

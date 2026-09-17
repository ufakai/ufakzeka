"""Stage 3: train the tokenizer on stage 2 text, then tokenize every stage 2 shard into
uint16 token streams with <|endoftext|> separators.

Outputs on the volume:
  tokenizer/ufakzeka.json
  tokenizer/eval.json                          fertility numbers on held out text
  stage3/train/<tier>/<name>.bin               uint16, documents separated by EOT
  stage3/heldout/<source>.bin
  stage3/manifest.json                         token counts per file, tier, source

Run:  modal run modal/tokenize_data.py            (train tokenizer, then tokenize)
      modal run modal/tokenize_data.py --skip-train
"""

from __future__ import annotations

import modal

app = modal.App("ufakzeka-tokenize")
vol = modal.Volume.from_name("ufakzeka-data", version=2)
hf_secret = modal.Secret.from_name("ufakzeka-hf")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("tokenizers>=0.20", "pyarrow>=17", "numpy>=2", "regex", "transformers>=4.45", "huggingface-hub>=0.30", "datasets>=3.0")
    .add_local_python_source("ufakzeka")
)

TOK_PATH = "/data/tokenizer/ufakzeka.json"


@app.function(image=image, volumes={"/data": vol}, secrets=[hf_secret], cpu=16.0, memory=65536, timeout=4 * 3600)
def train_tokenizer(vocab_size: int = 40960, max_gb: float = 25.0) -> dict:
    import glob, json, os, time
    from tokenizers import trainers, pre_tokenizers
    from ufakzeka.tokenizer.train import SPECIALS, build_tokenizer, iter_texts
    from ufakzeka.tokenizer.evaluate import load_texts, measure

    vol.reload()
    t0 = time.time()
    # Tokenizer data: all tiers except the English slice, so the vocab is Turkish first.
    files = sorted(glob.glob("/data/stage2/train/A/*.parquet") + glob.glob("/data/stage2/train/B/*.parquet"))
    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size - len(SPECIALS), min_frequency=2, special_tokens=[],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False, max_token_length=24,
    )
    tok.train_from_iterator(iter_texts(files, int(max_gb * 1e9)), trainer=trainer)
    tok.add_special_tokens(SPECIALS)
    assert tok.get_vocab_size() == vocab_size, tok.get_vocab_size()
    os.makedirs("/data/tokenizer", exist_ok=True)
    tok.save(TOK_PATH)

    texts = load_texts("/data/stage2/heldout/fw2hq.parquet", 2000) + load_texts("/data/stage2/heldout/finewiki.parquet", 1000)
    report = {"ufakzeka": measure(lambda t: tok.encode(t).ids, texts), "train_sec": round(time.time() - t0)}
    report["roundtrip_failures"] = sum(1 for t in texts[:300] if tok.decode(tok.encode(t).ids) != t)
    from transformers import AutoTokenizer
    for ref in ("vngrs-ai/Kumru-2B", "google/gemma-3-270m", "Qwen/Qwen3-0.6B", "meta-llama/Llama-3.2-1B"):
        try:
            r = AutoTokenizer.from_pretrained(ref, token=os.environ["HF_TOKEN"])
            report[ref] = measure(lambda t: r.encode(t, add_special_tokens=False), texts)
        except Exception as e:  # gated repos are optional references
            report[ref] = f"unavailable: {type(e).__name__}"
    with open("/data/tokenizer/eval.json", "w") as f:
        json.dump(report, f, indent=1)
    vol.commit()
    return report


@app.function(image=image, volumes={"/data": vol}, cpu=4.0, memory=16384, timeout=2 * 3600, max_containers=24)
def tokenize_file(path: str) -> dict:
    import os, time
    import numpy as np
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    vol.reload()
    t0 = time.time()
    tok = Tokenizer.from_file(TOK_PATH)
    eot = tok.token_to_id("<|endoftext|>")
    rel = path.replace("/data/stage2/", "")
    out = "/data/stage3/" + rel.replace(".parquet", ".bin")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n_tokens = n_docs = 0
    with open(out, "wb") as f:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["text"], batch_size=1024):
            encs = tok.encode_batch(batch.column("text").to_pylist())
            ids = np.concatenate([np.array(e.ids + [eot], dtype=np.uint16) for e in encs])
            f.write(ids.tobytes())
            n_tokens += len(ids); n_docs += len(encs)
    vol.commit()
    return {"file": out, "tokens": n_tokens, "docs": n_docs, "sec": round(time.time() - t0, 1)}


@app.function(image=image, volumes={"/data": vol}, secrets=[hf_secret], cpu=4.0, memory=16384, timeout=3600)
def build_qa_tier() -> dict:
    """Tier Q: question and answer style text rendered as plain documents for pretraining.
    Sources (commercial licences): WikiRAG-TR, InstructPapers-TR, gsm8k_tr, Turkish-SFT-v1, Atlas-Instruct,
    everyday conversations, Aya. Each document is 'Soru: ...\nCevap: ...' so the base model learns the
    question answering register before SFT."""
    import os, random
    import numpy as np
    from datasets import load_dataset
    from tokenizers import Tokenizer
    from ufakzeka.train.sft_data import _messages, _aya, _pairs, strip_diacritics
    vol.reload()
    tok = Tokenizer.from_file(TOK_PATH)
    eot = tok.token_to_id("<|endoftext|>")
    rng = random.Random(3)
    docs = []
    def add(q, a):
        if q and a:
            docs.append(f"Soru: {q.strip()}\nCevap: {a.strip()}")
    token = os.environ["HF_TOKEN"]
    for r in load_dataset("Metin/WikiRAG-TR", split="train", token=token):
        if not r.get("is_negative_response"):
            add(r["question"], r["answer"])
    for r in load_dataset("selimc/InstructPapers-TR", split="train", token=token):
        cols = set(r)
        t = _pairs(r, "question", "answer") if {"question", "answer"} <= cols else (_pairs(r, "instruction", "output") if {"instruction", "output"} <= cols else _pairs(r))
        if t: add(*t[0])
    for r in load_dataset("ytu-ce-cosmos/gsm8k_tr", split="train", token=token):
        add(r.get("question"), r.get("answer"))
    for r in load_dataset("AlicanKiraz0/Turkish-SFT-Dataset-v1.0", split="train", token=token):
        cols = set(r)
        t = _messages(r) if ("messages" in cols or "conversations" in cols) else (_pairs(r) if {"prompt", "response"} <= cols else _pairs(r, "instruction", "output"))
        if t: add(*t[0])
    n = 0
    for r in load_dataset("AlicanKiraz0/Turkce-Atlas-Instruct", split="train", token=token):
        t = _messages(r)
        if t: add(*t[0]); n += 1
        if n >= 60000: break
    for r in load_dataset("CohereLabs/aya_dataset", split="train", token=token):
        if r["language_code"] == "tur": add(r["inputs"], r["targets"])
    rng.shuffle(docs)
    os.makedirs("/data/stage3/train/Q", exist_ok=True)
    ids = []
    for d in docs:
        ids += tok.encode(d).ids + [eot]
    np.array(ids, dtype=np.uint16).tofile("/data/stage3/train/Q/part_0.bin")
    vol.commit()
    return {"docs": len(docs), "tokens": len(ids)}


@app.function(image=image, volumes={"/data": vol}, cpu=4.0, memory=16384, timeout=3600)
def tokenize_recency() -> dict:
    """Tier R: recent Turkish web and the Aug 2026 wiki dump, from stage1_recency/*/*.parquet."""
    import glob, os
    import numpy as np
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    vol.reload()
    tok = Tokenizer.from_file(TOK_PATH)
    eot = tok.token_to_id("<|endoftext|>")
    os.makedirs("/data/stage3/train/R", exist_ok=True)
    out = {}
    for src in sorted(glob.glob("/data/stage1_recency/*")):
        name = os.path.basename(src)
        n = 0
        with open(f"/data/stage3/train/R/{name}.bin", "wb") as f:
            for p in sorted(glob.glob(f"{src}/*.parquet")):
                for batch in pq.ParquetFile(p).iter_batches(columns=["text"], batch_size=1024):
                    encs = tok.encode_batch(batch.column("text").to_pylist())
                    ids = np.concatenate([np.array(e.ids + [eot], dtype=np.uint16) for e in encs])
                    f.write(ids.tobytes()); n += len(ids)
        out[name] = n
    vol.commit()
    return out


@app.local_entrypoint()
def main(skip_train: bool = False, qa_tier: bool = False, recency: bool = False):
    import json
    if recency:
        print(json.dumps(tokenize_recency.remote(), indent=1)); return
    if qa_tier:
        print(json.dumps(build_qa_tier.remote(), indent=1)); return
    if not skip_train:
        print(json.dumps(train_tokenizer.remote(), indent=1))
    files = [p for p in vol.listdir("stage2", recursive=True) if p.path.endswith(".parquet")]
    paths = ["/data/" + p.path for p in files]
    manifest = list(tokenize_file.map(paths))
    for m in manifest:
        print(json.dumps(m))
    total = sum(m["tokens"] for m in manifest)
    with vol.batch_upload(force=True) as b:
        import io
        b.put_file(io.BytesIO(json.dumps({"files": manifest, "total_tokens": total}, indent=1).encode()), "stage3/manifest.json")
    print("total tokens", total)

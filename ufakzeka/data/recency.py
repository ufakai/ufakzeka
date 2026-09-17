"""Turn the Hetzner recency crawl (trwiki dump extraction and Common Crawl pages) into
stage 1 style parquet shards, deduplicated by content hash, ready for tokenisation.

  python -m ufakzeka.data.recency /data/ufakzeka/wiki/trwiki-2026-08.jsonl.zst /data/ufakzeka/cc/pages_*.jsonl.zst --out /data/ufakzeka/recency
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
import xxhash
import zstandard as zstd

from ufakzeka.data.filters import filter_document


def rows_from(path: str):
    with open(path, "rb") as raw, zstd.ZstdDecompressor().stream_reader(raw) as r:
        buf = b""
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for line in lines:
                if line.strip():
                    yield json.loads(line)
        if buf.strip():
            yield json.loads(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-rows", type=int, default=50000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seen: set[int] = set()
    cols = {k: [] for k in ("id", "url", "source", "tier", "text", "n_chars", "n_words", "quality", "hash64")}
    stats = {"in": 0, "kept": 0, "dup": 0, "filtered": 0}
    shard = 0

    def flush():
        nonlocal shard
        if not cols["id"]:
            return
        t = pa.table({"id": pa.array(cols["id"], pa.string()), "url": pa.array(cols["url"], pa.string()),
                      "source": pa.array(cols["source"], pa.string()), "tier": pa.array(cols["tier"], pa.string()),
                      "text": pa.array(cols["text"], pa.string()), "n_chars": pa.array(cols["n_chars"], pa.int32()),
                      "n_words": pa.array(cols["n_words"], pa.int32()), "quality": pa.array(cols["quality"], pa.float32()),
                      "hash64": pa.array(cols["hash64"], pa.uint64())})
        pq.write_table(t, f"{a.out}/{shard:05d}.parquet", compression="zstd")
        shard += 1
        for k in cols:
            cols[k].clear()

    for path in a.inputs:
        for p in sorted(glob.glob(path)):
            source = "trwiki_2026_08" if "trwiki" in p else "cc_" + os.path.basename(p).split("_")[1].split(".")[0]
            light = source.startswith("trwiki")
            for r in rows_from(p):
                stats["in"] += 1
                res = filter_document(r["text"], min_chars=300, light=light, mask=not light)
                if not res.keep:
                    stats["filtered"] += 1; continue
                h = xxhash.xxh64_intdigest(" ".join(res.text.lower().split()).encode())
                if h in seen:
                    stats["dup"] += 1; continue
                seen.add(h)
                cols["id"].append(r["id"]); cols["url"].append(r.get("url")); cols["source"].append(source)
                cols["tier"].append("R"); cols["text"].append(res.text); cols["n_chars"].append(len(res.text))
                cols["n_words"].append(res.n_words); cols["quality"].append(None); cols["hash64"].append(h)
                stats["kept"] += 1
                if len(cols["id"]) >= a.shard_rows:
                    flush()
    flush()
    json.dump(stats, open(f"{a.out}/stats.json", "w"))
    print(stats, "shards", shard)


if __name__ == "__main__":
    main()

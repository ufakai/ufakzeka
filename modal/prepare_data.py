"""Stage 1: filter every shard of every source in parallel on Modal CPU workers.

Each task reads one parquet shard with column projection, applies the heuristic filters
and PII masking, and writes a compact parquet file to the ufakzeka-data volume under
stage1/<source>/<shard>.parquet with columns:
  id, url, source, tier, text, n_chars, n_words, quality, hash64

Run:  modal run modal/prepare_data.py --sources fw2hq,finewiki --limit 2
"""

from __future__ import annotations

import modal

app = modal.App("ufakzeka-prepare-data")
vol = modal.Volume.from_name("ufakzeka-data", version=2)
hf_secret = modal.Secret.from_name("ufakzeka-hf")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pyarrow>=17", "huggingface-hub>=0.30", "fsspec", "xxhash", "regex", "duckdb>=1.1")
    .add_local_python_source("ufakzeka")
)


def _download(url: str, token: str) -> str:
    """One request per shard (no range reads), with backoff on 429."""
    import time, urllib.request
    from huggingface_hub import hf_hub_download
    for attempt in range(6):
        try:
            if url.startswith("hf://"):
                _, _, repo_and_path = url.partition("hf://datasets/")
                repo, _, filename = repo_and_path.partition("/")
                # repo ids are org/name, so split once more
                org, name, *rest = repo_and_path.split("/", 2)
                repo, filename = f"{org}/{name}", rest[0]
                return hf_hub_download(repo, filename, repo_type="dataset", token=token, cache_dir="/tmp/hf")
            dst = "/tmp/shard.parquet"
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
            with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as f:
                while chunk := r.read(1 << 22):
                    f.write(chunk)
            return dst
        except Exception as e:
            if attempt == 5:
                raise
            time.sleep(20 * (attempt + 1))
    raise RuntimeError("unreachable")


def _filter_one(text: str, min_chars: int, light: bool, mask: bool):
    from ufakzeka.data.filters import filter_document
    return filter_document(text, min_chars=min_chars, light=light, mask=mask)


@app.function(image=image, volumes={"/data": vol}, secrets=[hf_secret], cpu=4.0, memory=8192,
              timeout=3600, retries=modal.Retries(max_retries=2, initial_delay=5.0),
              max_containers=24)
def process_shard(source: str, url: str, idx: int) -> dict:
    import os, time
    import pyarrow as pa
    import pyarrow.parquet as pq
    import xxhash
    from ufakzeka.data.filters import filter_document
    from ufakzeka.data.sources import SOURCES

    src = SOURCES[source]
    out_dir = f"/data/stage1/{source}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{idx:05d}.parquet"
    if os.path.exists(out_path):
        return {"source": source, "idx": idx, "skipped": True}

    t0 = time.time()
    cols = [src.text_col] + [c for c in (src.id_col, src.url_col) if c] + list(src.extra_cols)
    local = _download(url, os.environ["HF_TOKEN"])
    table = pq.read_table(local, columns=cols)
    os.remove(local)

    reasons: dict[str, int] = {}
    ids, urls, texts, nchars, nwords, quals, hashes = [], [], [], [], [], [], []
    pii_total = 0
    rows = table.to_pylist()
    del table
    from multiprocessing import Pool
    with Pool(4) as pool:
        results = pool.starmap(_filter_one, [(row[src.text_col] or "", src.min_chars, src.light, src.mask_pii) for row in rows], chunksize=64)
    for i, (row, r) in enumerate(zip(rows, results)):
        if src.lang_score_col and (row.get(src.lang_score_col) or 0) < src.lang_score_min:
            reasons["lang_score"] = reasons.get("lang_score", 0) + 1
            continue
        if not r.keep:
            reasons[r.reason] = reasons.get(r.reason, 0) + 1
            continue
        pii_total += r.pii_masked
        ids.append(str(row.get(src.id_col)) if src.id_col else f"{source}:{idx}:{i}")
        urls.append(row.get(src.url_col) if src.url_col else None)
        texts.append(r.text)
        nchars.append(len(r.text))
        nwords.append(r.n_words)
        q = row.get("quality_score")
        if q is None and row.get("fw_edu_scores"):
            s = row["fw_edu_scores"]
            q = sum(s) / len(s)
        quals.append(float(q) if q is not None else None)
        hashes.append(xxhash.xxh64_intdigest(" ".join(r.text.lower().split()).encode()))

    out = pa.table({
        "id": pa.array(ids, pa.string()), "url": pa.array(urls, pa.string()),
        "source": pa.array([source] * len(ids), pa.string()),
        "tier": pa.array([src.tier] * len(ids), pa.string()),
        "text": pa.array(texts, pa.string()), "n_chars": pa.array(nchars, pa.int32()),
        "n_words": pa.array(nwords, pa.int32()), "quality": pa.array(quals, pa.float32()),
        "hash64": pa.array(hashes, pa.uint64()),
    })
    pq.write_table(out, out_path, compression="zstd")
    vol.commit()
    return {"source": source, "idx": idx, "in": len(rows), "kept": len(ids),
            "chars": int(sum(nchars)), "pii": pii_total, "reasons": reasons,
            "sec": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main(sources: str = "", limit: int = 0):
    import json
    from ufakzeka.data.sources import SOURCES, list_shards

    names = [s for s in sources.split(",") if s] or list(SOURCES)
    jobs = []
    for name in names:
        shards = list_shards(SOURCES[name])
        if limit:
            shards = shards[:limit]
        print(f"{name}: {len(shards)} shards")
        jobs += [(name, u, i) for i, u in enumerate(shards)]

    totals: dict[str, dict] = {}
    for res in process_shard.starmap(jobs, order_outputs=False):
        if res.get("skipped"):
            continue
        t = totals.setdefault(res["source"], {"in": 0, "kept": 0, "chars": 0, "pii": 0, "reasons": {}})
        t["in"] += res["in"]; t["kept"] += res["kept"]; t["chars"] += res["chars"]; t["pii"] += res["pii"]
        for k, v in res["reasons"].items():
            t["reasons"][k] = t["reasons"].get(k, 0) + v
        print(json.dumps(res, ensure_ascii=False))
    print(json.dumps(totals, indent=1, ensure_ascii=False))

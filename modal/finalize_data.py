"""Stage 2: decontamination, global dedup, tiering, held out split, final shards.

Step A (parallel): every stage1 shard is scanned for 13-gram overlap with the eval sets;
contaminated ids are returned. Step B (one container, DuckDB): dedup on narrow columns
(hash64 of normalised text, then url), tier A promotion for the top web documents by
quality score, held out sample per source, and streaming COPY of the surviving text into
  stage2/train/<tier>/part_*.parquet   (id, source, tier, text)
  stage2/heldout/<source>.parquet
  stage2/stats.json

Run:  modal run modal/finalize_data.py --web-top-quality 0.58
"""

from __future__ import annotations

import modal

app = modal.App("ufakzeka-finalize-data")
vol = modal.Volume.from_name("ufakzeka-data", version=2)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("duckdb>=1.1", "pyarrow>=17", "regex")
    .add_local_python_source("ufakzeka")
    .add_local_dir("data/eval", remote_path="/eval")
)


def _scan(args):
    text, grams = args
    from ufakzeka.data.decontam import is_contaminated
    return is_contaminated(text, grams)


@app.function(image=image, volumes={"/data": vol}, cpu=4.0, memory=8192, timeout=3600, max_containers=24)
def decontam_shard(path: str) -> dict:
    import time
    import pyarrow.parquet as pq
    from multiprocessing import Pool
    from ufakzeka.data.decontam import build_index, is_contaminated
    vol.reload()
    t0 = time.time()
    grams = build_index("/eval")
    t = pq.read_table(path, columns=["id", "text"])
    ids, texts = t.column("id").to_pylist(), t.column("text").to_pylist()
    global _G
    _G = grams
    with Pool(4, initializer=_init, initargs=(grams,)) as pool:
        flags = pool.map(_check, texts, chunksize=256)
    bad = [i for i, f in zip(ids, flags) if f]
    return {"path": path, "n": len(ids), "bad": bad, "sec": round(time.time() - t0, 1)}


def _init(g):
    global _G
    _G = g


def _check(text):
    from ufakzeka.data.decontam import is_contaminated
    return is_contaminated(text, _G)


@app.function(image=image, volumes={"/data": vol}, cpu=8.0, memory=32768, timeout=6 * 3600,
              ephemeral_disk=512 * 1024)
def finalize(bad_ids: list[str], web_top_quality: float, heldout_per_source: int, seed: int) -> dict:
    import glob, json, os, shutil, time
    import duckdb
    import pyarrow as pa
    t0 = time.time()
    vol.reload()
    files = sorted(glob.glob("/data/stage1/*/*.parquet"))
    con = duckdb.connect()
    con.execute("SET memory_limit='26GB'; SET threads=8; SET preserve_insertion_order=false; SET temp_directory='/tmp/duck';")
    con.register("bad_ids", pa.table({"id": pa.array(bad_ids, pa.string())}))
    stats = {"stage1_files": len(files), "contaminated": len(bad_ids)}

    con.execute(f"""
        CREATE TABLE meta AS
        SELECT id, url, source, n_chars, quality, hash64,
               CASE WHEN tier='B' AND quality >= {web_top_quality} THEN 'A' ELSE tier END AS tier,
               hash(id || '{seed}') AS rnd
        FROM read_parquet({files!r}, union_by_name=true)
        WHERE id NOT IN (SELECT id FROM bad_ids)
    """)
    stats["input_docs"] = con.execute("SELECT count(*) FROM meta").fetchone()[0] + len(bad_ids)
    con.execute("""
        CREATE TABLE d1 AS SELECT * EXCLUDE(rn) FROM (
          SELECT *, row_number() OVER (PARTITION BY hash64 ORDER BY tier, rnd) rn FROM meta) WHERE rn = 1
    """)
    stats["after_hash_dedup"] = con.execute("SELECT count(*) FROM d1").fetchone()[0]
    con.execute("""
        CREATE TABLE d2 AS SELECT * EXCLUDE(rn) FROM (
          SELECT *, row_number() OVER (PARTITION BY coalesce(url, id) ORDER BY tier, rnd) rn FROM d1) WHERE rn = 1
    """)
    stats["after_url_dedup"] = con.execute("SELECT count(*) FROM d2").fetchone()[0]
    con.execute(f"""
        CREATE TABLE keep AS SELECT *, (row_number() OVER (PARTITION BY source ORDER BY rnd)) <= {heldout_per_source} AS heldout FROM d2
    """)

    if os.path.exists("/data/stage2"):
        shutil.rmtree("/data/stage2")
    os.makedirs("/data/stage2/heldout", exist_ok=True)
    con.execute(f"CREATE VIEW raw AS SELECT id, text FROM read_parquet({files!r}, union_by_name=true)")
    for (src,) in con.execute("SELECT DISTINCT source FROM keep").fetchall():
        con.execute(f"""COPY (SELECT k.id, k.source, k.tier, r.text FROM keep k JOIN raw r USING (id)
                        WHERE k.heldout AND k.source='{src}')
                        TO '/data/stage2/heldout/{src}.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    for (tier,) in con.execute("SELECT DISTINCT tier FROM keep").fetchall():
        os.makedirs(f"/data/stage2/train/{tier}", exist_ok=True)
        con.execute(f"""COPY (SELECT k.id, k.source, k.tier, r.text FROM keep k JOIN raw r USING (id)
                        WHERE NOT k.heldout AND k.tier='{tier}' ORDER BY k.rnd)
                        TO '/data/stage2/train/{tier}' (FORMAT parquet, COMPRESSION zstd, FILE_SIZE_BYTES '512MB', FILENAME_PATTERN 'part_{{i}}')""")
    stats["per_source_tier"] = [
        {"source": s, "tier": t, "docs": n, "chars": c}
        for s, t, n, c in con.execute("SELECT source, tier, count(*), sum(n_chars) FROM keep WHERE NOT heldout GROUP BY 1,2 ORDER BY 1,2").fetchall()
    ]
    stats["train_docs"], stats["train_chars"] = con.execute("SELECT count(*), sum(n_chars) FROM keep WHERE NOT heldout").fetchone()
    stats["sec"] = round(time.time() - t0)
    with open("/data/stage2/stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    vol.commit()
    return stats


@app.local_entrypoint()
def main(web_top_quality: float = 0.58, heldout_per_source: int = 2000, seed: int = 1):
    import json
    files = ["/data/" + p.path for p in vol.listdir("stage1", recursive=True) if p.path.endswith(".parquet")]
    print(len(files), "stage1 files")
    bad, n = [], 0
    for r in decontam_shard.map(files):
        bad += r["bad"]; n += r["n"]
    print(f"decontam: {len(bad)} of {n} docs contaminated")
    print(json.dumps(finalize.remote(bad, web_top_quality, heldout_per_source, seed), indent=1))

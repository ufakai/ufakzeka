"""Recent Turkish web pages from Common Crawl, for the recency anneal.

Stage scan: read the columnar index of a crawl over HTTPS (DuckDB, column projection),
keep HTML pages whose detected language is Turkish and whose host is either a .tr
domain or a domain that appears in FineWeb2-HQ (reputation list), sample, write parquet.
Stage fetch: range-GET each WARC record from data.commoncrawl.org, parse the response,
extract main text with trafilatura, apply the stage 1 filters, write jsonl.zst.

  python -m ufakzeka.data.cc_crawl scan CC-MAIN-2026-34 out/scan_2026-34.parquet --allow allow.txt --per-file 1500
  python -m ufakzeka.data.cc_crawl fetch out/scan_2026-34.parquet out/pages_2026-34.jsonl.zst --concurrency 24
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import io
import json
import os
import random
import time
import urllib.request

CC = "https://data.commoncrawl.org/"
PACE = 0.25  # seconds between requests per connection; Common Crawl blocks bursts


def index_files(crawl: str) -> list[str]:
    with urllib.request.urlopen(f"{CC}crawl-data/{crawl}/cc-index-table.paths.gz", timeout=60) as r:
        paths = gzip.decompress(r.read()).decode().split()
    return [CC + p for p in paths if "subset=warc/" in p and p.endswith(".parquet")]


def scan(crawl: str, out: str, allow: str | None, per_file: int, start: int, end: int):
    import duckdb
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; SET http_retries=5; SET http_timeout=120000; SET memory_limit='1200MB'; SET threads=2;")
    allow_sql = ""
    if allow:
        con.execute(f"CREATE TABLE allow AS SELECT column0 AS d FROM read_csv('{allow}', header=false)")
        allow_sql = " OR url_host_registered_domain IN (SELECT d FROM allow)"
    files = index_files(crawl)[start:end or None]
    print(crawl, len(files), "index files", flush=True)
    t0 = time.time()
    for i, f in enumerate(files):
        q = f"""
            SELECT url, url_host_registered_domain AS domain, url_host_tld AS tld, warc_filename, warc_record_offset, warc_record_length, fetch_time
            FROM read_parquet('{f}')
            WHERE content_languages = 'tur' AND content_mime_detected = 'text/html' AND fetch_status = 200
              AND (url_host_tld = 'tr'{allow_sql})
              AND NOT url_host_registered_domain LIKE '%blogspot%'
            USING SAMPLE {per_file} ROWS
        """
        for attempt in range(4):
            try:
                con.execute(f"CREATE OR REPLACE TABLE chunk AS {q}")
                break
            except Exception as e:
                if attempt == 3:
                    print("skip", f, str(e)[:120], flush=True); con.execute("CREATE OR REPLACE TABLE chunk AS SELECT * FROM (SELECT 1) WHERE false")
                time.sleep(10)
        n = con.execute("SELECT count(*) FROM chunk").fetchone()[0]
        if n:
            con.execute(f"COPY chunk TO '{out}.{start+i:04d}.parquet' (FORMAT parquet)")
        print(f"{i+1}/{len(files)} +{n} {time.time()-t0:.0f}s", flush=True)
    print("scan done", flush=True)


def _extract(payload: bytes, url: str) -> str | None:
    import trafilatura
    try:
        html = payload.decode("utf-8", errors="replace")
    except Exception:
        return None
    return trafilatura.extract(html, url=url, include_comments=False, include_tables=False, favor_precision=True, target_language="tr")


async def _fetch_one(session, sem, row, out_rows, stats):
    import aiohttp
    from warcio.archiveiterator import ArchiveIterator
    from ufakzeka.data.filters import filter_document
    url = CC + row["warc_filename"]
    start = int(row["warc_record_offset"]); end = start + int(row["warc_record_length"]) - 1
    data = None
    async with sem:
        for attempt in range(6):
            try:
                async with session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=aiohttp.ClientTimeout(total=60)) as r:
                    stats.setdefault(f"s{r.status}", 0); stats[f"s{r.status}"] += 1
                    if r.status in (403, 429, 503):
                        await asyncio.sleep(15 * (attempt + 1)); continue
                    if r.status not in (200, 206):
                        stats["http_err"] += 1; return
                    data = await r.read()
                break
            except Exception:
                await asyncio.sleep(5 * (attempt + 1))
        await asyncio.sleep(PACE)
    if data is None:
        stats["http_err"] += 1; return
    try:
        payload = None
        for rec in ArchiveIterator(io.BytesIO(data)):
            if rec.rec_type == "response":
                payload = rec.content_stream().read()
                break
        if payload is None:
            stats["no_payload"] += 1; return
        text = _extract(payload, row["url"])
        if not text:
            stats["no_text"] += 1; return
        res = filter_document(text, min_chars=400)
        if not res.keep:
            stats["filtered"] += 1; return
        out_rows.append({"id": f"cc:{row['warc_filename'].split('/')[1]}:{start}", "url": row["url"], "date": str(row["fetch_time"])[:10], "text": res.text})
        stats["kept"] += 1
    except Exception:
        stats["exc"] += 1


async def _fetch_all(rows, out, concurrency):
    import aiohttp
    import zstandard as zstd
    sem = asyncio.Semaphore(concurrency)
    stats = {"kept": 0, "filtered": 0, "no_text": 0, "no_payload": 0, "http_err": 0, "exc": 0}
    out_rows: list[dict] = []
    t0 = time.time()
    written = 0
    with open(out, "wb") as raw, zstd.ZstdCompressor(level=6).stream_writer(raw) as w:
        async with aiohttp.ClientSession(headers={"User-Agent": "ufakzeka-research-crawler (contact: furkan@ufakai.com)"}) as session:
            for i in range(0, len(rows), 2000):
                batch = rows[i:i + 2000]
                await asyncio.gather(*[_fetch_one(session, sem, r, out_rows, stats) for r in batch])
                for d in out_rows:
                    w.write((json.dumps(d, ensure_ascii=False) + "\n").encode())
                written += len(out_rows); out_rows.clear()
                print(f"{i+len(batch)}/{len(rows)} kept={stats['kept']} filtered={stats['filtered']} no_text={stats['no_text']} err={stats['http_err']+stats['exc']} {time.time()-t0:.0f}s", flush=True)
    print("fetch done", stats, flush=True)


def fetch(scan_glob: str, out: str, concurrency: int, limit: int):
    import glob
    import pyarrow.parquet as pq
    files = sorted(glob.glob(scan_glob))
    rows = []
    for f in files:
        rows += pq.read_table(f).to_pylist()
    random.Random(1).shuffle(rows)
    # one page per domain per 50 rows keeps the sample broad
    if limit:
        rows = rows[:limit]
    print("candidates", len(rows), flush=True)
    asyncio.run(_fetch_all(rows, out, concurrency))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan"); s.add_argument("crawl"); s.add_argument("out"); s.add_argument("--allow"); s.add_argument("--per-file", type=int, default=60000)
    s.add_argument("--start", type=int, default=0); s.add_argument("--end", type=int, default=0)
    f = sub.add_parser("fetch"); f.add_argument("scan_glob"); f.add_argument("out"); f.add_argument("--concurrency", type=int, default=3); f.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "scan":
        scan(a.crawl, a.out, a.allow, a.per_file, a.start, a.end)
    else:
        fetch(a.scan_glob, a.out, a.concurrency, a.limit)

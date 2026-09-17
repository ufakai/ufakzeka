import modal
app = modal.App("ufakzeka-domains")
vol = modal.Volume.from_name("ufakzeka-data", version=2)
image = modal.Image.debian_slim(python_version="3.12").pip_install("duckdb>=1.1")

@app.function(image=image, volumes={"/data": vol}, cpu=8.0, memory=32768, timeout=1800)
def domains() -> str:
    import duckdb
    vol.reload()
    con = duckdb.connect()
    con.execute("SET threads=8; SET memory_limit='26GB';")
    rows = con.execute("""
        SELECT d FROM (
          SELECT regexp_extract(url, '^https?://(?:www\\.)?([^/:]+)', 1) AS host, count(*) AS n
          FROM read_parquet('/data/stage1/fw2hq/*.parquet') GROUP BY 1)
        , LATERAL (SELECT regexp_extract(host, '([^.]+\\.[^.]+)$', 1) AS d)
        WHERE n >= 20 AND d NOT LIKE '%.tr' AND d <> '' AND d NOT IN ('blogspot.com','wordpress.com','tumblr.com','wixsite.com','weebly.com')
        GROUP BY d HAVING sum(n) >= 20 ORDER BY sum(n) DESC
    """).fetchall()
    return "\n".join(r[0] for r in rows)

@app.local_entrypoint()
def main():
    s = domains.remote()
    open("data/cc_allow_domains.txt", "w").write(s + "\n")
    print(len(s.split()), "domains;", s.split()[:8])

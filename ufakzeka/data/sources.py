"""Corpus registry. Every source is a list of parquet files on the Hugging Face hub
plus the column names we need. Only text and light metadata are read; wide columns
such as embeddings are never fetched thanks to parquet column projection."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from huggingface_hub import HfApi


@dataclass(frozen=True)
class Source:
    name: str
    repo: str
    tier: str  # A = highest quality (anneal), B = main web, C = english code/math
    text_col: str = "text"
    id_col: str | None = "id"
    url_col: str | None = "url"
    extra_cols: tuple[str, ...] = ()
    # either a path prefix inside the repo, or "parquet-api" for the auto-converted parquet branch
    path_prefix: str | None = None
    parquet_api_config: str | None = None
    parquet_api_split: str = "train"
    min_chars: int = 200
    lang_score_col: str | None = None
    lang_score_min: float = 0.0
    is_synthetic: bool = False
    license: str = ""
    notes: str = ""
    max_shards: int = 0  # 0 = all
    light: bool = False  # curated or synthetic: skip repetition and script ratio rules
    mask_pii: bool = True


SOURCES: dict[str, Source] = {
    "fw2hq": Source(
        name="fw2hq", repo="epfml/FineWeb2-HQ", tier="B",
        extra_cols=("quality_score", "minhash_cluster_size", "date", "language_score"),
        path_prefix="tur_Latn/", lang_score_col="language_score", lang_score_min=0.90,
        license="odc-by", notes="top 10% of FineWeb-2 tur_Latn by XLM-R MLP classifier",
    ),
    "mogan": Source(
        name="mogan", repo="moganai/mogan-turkish-web", tier="B",
        extra_cols=("lang_score",), path_prefix="data/",
        lang_score_col="lang_score", lang_score_min=0.95, min_chars=400, max_shards=100,
        license="odc-by", notes="Common Crawl Jan 2025 to Jun 2026, PII masked, minhash deduped",
    ),
    "finepdfs_edu": Source(
        name="finepdfs_edu", repo="HuggingFaceFW/finepdfs-edu", tier="A", light=True,
        extra_cols=("token_count", "fw_edu_scores"), path_prefix="data/tur_Latn/train/",
        min_chars=500, license="odc-by", notes="educational PDF text, docling extraction",
    ),
    "finewiki": Source(
        name="finewiki", repo="HuggingFaceFW/finewiki", tier="A", light=True,
        parquet_api_config="tr", min_chars=300, license="cc-by-sa-4.0",
        notes="Turkish Wikipedia from enterprise HTML dumps, Aug 2025",
    ),
    "bilge_stories": Source(
        name="bilge_stories", light=True, mask_pii=False, repo="BILGEM-AI/BILGE-Synthetic-Stories", tier="A",
        id_col=None, url_col=None, parquet_api_config="default", is_synthetic=True,
        license="apache-2.0", notes="TUBITAK BILGEM synthetic stories (Cosmopedia style)",
    ),
    "bilge_web": Source(
        name="bilge_web", light=True, mask_pii=False, repo="BILGEM-AI/BILGE-Synthetic-Web", tier="A",
        id_col=None, url_col=None, parquet_api_config="default", is_synthetic=True,
        license="apache-2.0", notes="TUBITAK BILGEM synthetic web style text",
    ),
    "bilge_math": Source(
        name="bilge_math", light=True, mask_pii=False, repo="BILGEM-AI/BILGE-Synthetic-Math", tier="A",
        id_col=None, url_col=None, parquet_api_config="default", is_synthetic=True,
        license="apache-2.0", notes="TUBITAK BILGEM synthetic math explanations",
    ),
    "finemath": Source(
        name="finemath", repo="HuggingFaceTB/finemath", tier="C", light=True, mask_pii=False,
        id_col=None, extra_cols=("score", "language_score"), path_prefix="finemath-4plus/",
        min_chars=500, license="odc-by", max_shards=6,
        notes="English math web text, FineMath-4plus subset; small transfer slice only",
    ),
    "cosmos_syn": Source(
        name="cosmos_syn", light=True, mask_pii=False, repo="Berkesule/COSMOS-Sentetic-Turkish-Corpus-2GB", tier="A",
        text_col="corpus_text", id_col=None, url_col=None, parquet_api_config="default",
        is_synthetic=True, license="apache-2.0", notes="synthetic encyclopedic Turkish",
    ),
}


def hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    p = os.path.expanduser("~/.ufakzeka/hf_token")
    if os.path.exists(p):
        return open(p).read().strip()
    return None


def list_shards(src: Source) -> list[str]:
    """Return hf:// style URLs (resolved https for the parquet API) for all shards of a source."""
    api = HfApi(token=hf_token())
    if src.parquet_api_config:
        import requests
        r = requests.get(f"https://huggingface.co/api/datasets/{src.repo}/parquet",
                         headers={"Authorization": f"Bearer {hf_token()}"} if hf_token() else {}, timeout=60)
        r.raise_for_status()
        urls = r.json()[src.parquet_api_config][src.parquet_api_split]
        return urls[: src.max_shards] if src.max_shards else urls
    files = api.list_repo_files(src.repo, repo_type="dataset")
    urls = [
        f"hf://datasets/{src.repo}/{f}" for f in files
        if f.startswith(src.path_prefix or "") and f.endswith(".parquet")
    ]
    return urls[: src.max_shards] if src.max_shards else urls

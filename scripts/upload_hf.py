"""Publish the release artifacts to the Hugging Face Hub.

Nothing here runs by itself: every call needs --do, and without it the script only prints what it would
upload. Repos are created private; flipping them public is a separate, deliberate step on the website.

    python scripts/upload_hf.py --base models/ufakzeka-1-s3/hf --instruct models/ufakzeka-1-instruct-v14/hf \
        --gguf-dir models/gguf --org ufakai            # dry run, prints the plan
    python scripts/upload_hf.py ... --do               # actually uploads

Needs HF_TOKEN in the environment (a write token) and huggingface_hub installed.
"""

from __future__ import annotations

import argparse
import os
import sys

CARD_BASE = "release/README_base.md"
CARD_INSTRUCT = "release/README_instruct.md"
CARD_GGUF = "release/README_gguf.md"
ATTRIBUTION = "release/ATTRIBUTION.md"

# Files that make up a transformers checkpoint. Anything else in the folder (optimizer state, logs,
# export_meta.json) stays local.
MODEL_FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
               "tokenizer_config.json", "special_tokens_map.json"]


def plan(local_dir: str, repo: str, files: list[str]) -> list[tuple[str, str]]:
    out = []
    for f in files:
        p = os.path.join(local_dir, f)
        if os.path.exists(p):
            out.append((p, f))
        elif f in ("config.json", "model.safetensors", "tokenizer.json"):
            sys.exit(f"missing required file {p}")
    return out


def upload(api, repo: str, items: list[tuple[str, str]], card: str | None, do: bool, private: bool):
    print(f"\n== {repo}")
    for local, name in items:
        print(f"   {name:32} {os.path.getsize(local) / 1e6:8.1f} MB")
    if card:
        print(f"   {'README.md':32} <- {card}")
    if not do:
        return
    api.create_repo(repo_id=repo, private=private, exist_ok=True)
    for local, name in items:
        api.upload_file(path_or_fileobj=local, path_in_repo=name, repo_id=repo)
    if card:
        api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=repo)
    if os.path.exists(ATTRIBUTION):
        api.upload_file(path_or_fileobj=ATTRIBUTION, path_in_repo="ATTRIBUTION.md", repo_id=repo)
    print("   uploaded")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", default="ufakai")
    ap.add_argument("--base", help="folder with the base model export")
    ap.add_argument("--instruct", help="folder with the chat model export")
    ap.add_argument("--gguf-dir", help="folder with the instruct GGUF files")
    ap.add_argument("--name", default="ufakzeka-1")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--public", dest="private", action="store_false", help="create public repos (think twice)")
    ap.add_argument("--do", action="store_true", help="actually upload; without it this is a dry run")
    a = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        sys.exit("set HF_TOKEN to a write token")
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])

    if a.base:
        upload(api, f"{a.org}/{a.name}-base", plan(a.base, a.name, MODEL_FILES), CARD_BASE, a.do, a.private)
    if a.instruct:
        upload(api, f"{a.org}/{a.name}", plan(a.instruct, a.name, MODEL_FILES), CARD_INSTRUCT, a.do, a.private)
    if a.gguf_dir:
        ggufs = sorted(f for f in os.listdir(a.gguf_dir) if f.endswith(".gguf"))
        items = [(os.path.join(a.gguf_dir, f), f) for f in ggufs]
        # the card promises the pre-tokenizer patch and the checksums next to the files
        items += [("release/llama.cpp-ufakzeka-pretok.patch", "llama.cpp-ufakzeka-pretok.patch"), ("release/SHA256SUMS", "SHA256SUMS")]
        card = CARD_GGUF if os.path.exists(CARD_GGUF) else CARD_INSTRUCT
        upload(api, f"{a.org}/{a.name}-GGUF", items, card, a.do, a.private)
    if not a.do:
        print("\ndry run; add --do to upload (repos are created private)")


if __name__ == "__main__":
    main()

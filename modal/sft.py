"""Supervised fine tuning of a pretrained ufakzeka checkpoint on Turkish instructions.

  modal run modal/sft.py --base /ckpt/ufakzeka-1/final.pt --name ufakzeka-1-instruct

Loss on assistant tokens only. AdamW, cosine decay, 2 epochs, sequences packed to 2048
with the mask stream from sft_data. Writes /ckpt/<name>/final.pt and an HF export."""

from __future__ import annotations

import os

import modal

app = modal.App("ufakzeka-sft")
data_vol = modal.Volume.from_name("ufakzeka-data", version=2)
ckpt_vol = modal.Volume.from_name("ufakzeka-ckpt", version=2, create_if_missing=True)
hf_secret = modal.Secret.from_name("ufakzeka-hf")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch>=2.8", "numpy>=2", "tokenizers>=0.20", "datasets>=3.0", "transformers>=4.45", "pyarrow>=15")
    .add_local_python_source("ufakzeka")
    .add_local_dir("scripts", remote_path="/root/scripts")
)


@app.function(image=image, volumes={"/data": data_vol}, secrets=[hf_secret], cpu=4.0, memory=16384, timeout=3600)
def build_sft_data(replay_frac: float = 0.15, arith_n: int = 12000, arith_typos: bool = True, dialogs_dir: str = "/data/dialogs", arith_direct: bool = False, out: str = "/data/sft", essay_cap: int = 0, essay_repeat: int = 0, wiki_facts: str = "", poems: str = "",
                   acrostic_path: str = "", acrostic_short: str = "", atlas_cap: int = 0) -> dict:
    """replay_frac: share of tokens taken from the tier A pretraining shards (wiki, PDFs, top web) as whole
    documents with full loss, so SFT forgets less of the language (v6 lost 9.6 TurBLiMP points)."""
    import os
    # before the import: sft_data reads its overrides at module load
    if essay_cap:
        os.environ["UFAKZEKA_ESSAY_CAP"] = str(essay_cap)
    if essay_repeat:
        os.environ["UFAKZEKA_ESSAY_REPEAT"] = str(essay_repeat)
    if wiki_facts:  # e.g. /data/facts/wiki_facts_probed.jsonl once the boundary relabel has run
        os.environ["UFAKZEKA_WIKI_FACTS"] = wiki_facts
    if poems:
        os.environ["UFAKZEKA_POEMS"] = poems
    if acrostic_path:
        os.environ["UFAKZEKA_ACROSTIC"] = acrostic_path
    if acrostic_short:
        os.environ["UFAKZEKA_ACROSTIC_SHORT"] = acrostic_short
    if atlas_cap:
        os.environ["UFAKZEKA_ATLAS_CAP"] = str(atlas_cap)
    from tokenizers import Tokenizer
    from ufakzeka.train.sft_data import load_examples, tokenize_examples
    data_vol.reload()
    tok = Tokenizer.from_file("/data/tokenizer/ufakzeka.json")
    import glob
    os.environ.setdefault("EVAL_DIR", "/data/eval")
    ex = load_examples(os.environ["HF_TOKEN"], local_dialogs=sorted(glob.glob(f"{dialogs_dir}/*.jsonl")), arith_n=arith_n, arith_typos=arith_typos, arith_direct=arith_direct,
                       story_root="/data/stage1/bilge_stories")
    os.makedirs(out, exist_ok=True)
    n_val = 500
    stats = {"val": tokenize_examples(ex[:n_val], tok, f"{out}/val"),
             "train": tokenize_examples(ex[n_val:], tok, f"{out}/train", replay_root="/data/stage3/train/A", replay_frac=replay_frac)}
    data_vol.commit()
    return stats


@app.function(image=image, volumes={"/data": data_vol}, secrets=[hf_secret], cpu=4.0, memory=16384, timeout=3600)
def build_arith_data(out: str = "/data/arith_v1", arith_n: int = 20000, general_share: float = 0.3, replay_frac: float = 0.1,
                     essay_cap: int = 2000, essay_repeat: int = 1, wiki_facts: str = "/data/facts/wiki_facts_probed.jsonl",
                     poems: str = "/data/generated/poems/v5/poems.jsonl", dialogs_dir: str = "/data/dialogs", fix: bool = False, fix_repeat: int = 1, fix_only: str = "", acrostic: bool = True,
                     acrostic_path: str = "/data/generated/poems/acrostic_v1.jsonl", free_glob: str = "/data/dialogs_free/*.jsonl",
                     acrostic_short: str = "/data/generated/poems/acrostic_short.jsonl") -> dict:
    """A second curriculum phase for arithmetic (two-phase curriculum): the arithmetic generators
    at higher counts plus a slice of the general mix so nothing else is forgotten. Run as a short SFT from a finished
    instruct checkpoint: --base /ckpt/<name>/final.pt --data <out> --epochs 1 --lr 3e-4."""
    import os, glob, random
    for k, v in (("UFAKZEKA_ESSAY_CAP", essay_cap), ("UFAKZEKA_ESSAY_REPEAT", essay_repeat), ("UFAKZEKA_WIKI_FACTS", wiki_facts), ("UFAKZEKA_POEMS", poems)):
        os.environ[k] = str(v)
    from tokenizers import Tokenizer
    from ufakzeka.train.sft_data import load_examples, tokenize_examples, arithmetic_examples, correction_examples, followup_op_examples, recheck_chain_examples, with_context_turns
    from ufakzeka.train.everyday_data import arithmetic_drills, percent_unit_examples
    data_vol.reload()
    rng = random.Random(7)
    os.environ.setdefault("EVAL_DIR", "/data/eval")
    os.environ["UFAKZEKA_FREECHAT"] = free_glob
    os.environ["UFAKZEKA_ACROSTIC_SHORT"] = acrostic_short
    general = load_examples(os.environ["HF_TOKEN"], local_dialogs=sorted(glob.glob(f"{dialogs_dir}/*.jsonl")), story_root="/data/stage1/bilge_stories")
    rng.shuffle(general)
    arith = (with_context_turns(arithmetic_examples(rng, arith_n), rng) + correction_examples(rng, 2500) + followup_op_examples(rng, 1200) + recheck_chain_examples(rng, 800)
             + arithmetic_drills(rng) * 2 + with_context_turns(percent_unit_examples(rng, 4000), rng))
    n_gen = int(len(arith) * general_share / (1 - general_share))
    ex = arith + general[:n_gen]
    if fix:
        # the third hand test families at full weight, and the acrostic flow, so a phase on v65s2 learns them
        from ufakzeka.train.sft_data import hand_test3_examples
        from ufakzeka.train.everyday_data import acrostic_examples
        os.environ["UFAKZEKA_ACROSTIC"] = acrostic_path  # before the misc family reads it (v83 built its second acrostics from the old file)
        fixes = [t for t, w in hand_test3_examples(rng, "/data/generated/stories_tr/tinystories_tr.jsonl", only=fix_only) for _ in range(max(1, w - 1))]  # closers x1 (v73 over-offered)
        if acrostic:
            os.environ["UFAKZEKA_ACROSTIC"] = acrostic_path
            fixes += acrostic_examples(rng, acrostic_path)
        fixes = fixes * fix_repeat  # v72 at 1e-4 with x1 left "var" and the tales by name unlearned
        print("fix families:", len(fixes))
        ex = ex + fixes
    rng.shuffle(ex)
    print("arithmetic phase:", len(arith), "arithmetic conversations,", n_gen, "general")
    tok = Tokenizer.from_file("/data/tokenizer/ufakzeka.json")
    os.makedirs(out, exist_ok=True)
    stats = {"arith": len(arith), "general": n_gen, "val": tokenize_examples(ex[:300], tok, f"{out}/val"),
             "train": tokenize_examples(ex[300:], tok, f"{out}/train", replay_root="/data/stage3/train/A", replay_frac=replay_frac)}
    data_vol.commit()
    return stats


@app.function(image=image, volumes={"/data": data_vol}, secrets=[hf_secret], cpu=4.0, memory=16384, timeout=3600)
def build_raft_data(model_name: str, replay_share: float = 0.2, repeat: int = 6, family_cap: float = 0.25, out: str = "",
                    essay_cap: int = 2000, essay_repeat: int = 1, wiki_facts: str = "/data/facts/wiki_facts_probed.jsonl",
                    poems: str = "/data/generated/poems/v5/poems.jsonl", dialogs_dir: str = "/data/dialogs") -> dict:
    """Rejection-sampling SFT data (RAFT, arXiv 2504.11343; RL's Razor 2509.04259: on-policy positives keep the
    KL to the SFT model small, which is what predicts retention): the model's own rule-passing samples from
    the pair harvest, each family capped, repeated so the run has a few hundred steps, plus a replay slice of
    the SFT mix (2605.29495). Three of four DPO rounds lowered the rule suite; this is the replacement."""
    import glob, json, os, random
    if essay_cap:
        os.environ["UFAKZEKA_ESSAY_CAP"] = str(essay_cap)
    if essay_repeat:
        os.environ["UFAKZEKA_ESSAY_REPEAT"] = str(essay_repeat)
    if wiki_facts:
        os.environ["UFAKZEKA_WIKI_FACTS"] = wiki_facts
    if poems:
        os.environ["UFAKZEKA_POEMS"] = poems
    from tokenizers import Tokenizer
    from ufakzeka.train.sft_data import load_examples, tokenize_examples
    data_vol.reload()
    rng = random.Random(7)
    rows = []
    for f in sorted(glob.glob(f"/data/onpolicy/rule_pairs_{model_name}*.positives.jsonl")):
        rows += [json.loads(l) for l in open(f, encoding="utf-8")]
    by_fam = {}
    for r in rows:
        by_fam.setdefault(r["family"], []).append(r)
    cap = int(family_cap * len(rows))
    kept = []
    for fam, rs in by_fam.items():
        rng.shuffle(rs)
        kept += rs[:cap]
    convs = []
    for r in kept:
        msgs = r["messages"]
        turns = [(msgs[i]["content"], msgs[i + 1]["content"]) for i in range(0, len(msgs) - 1, 2) if msgs[i]["role"] == "user"]
        turns.append((msgs[-1]["content"], r["answer"]))
        convs.extend([turns] * repeat)
    n_replay = int(len(kept) * repeat * replay_share / (1 - replay_share))
    os.environ.setdefault("EVAL_DIR", "/data/eval")
    mix = load_examples(os.environ["HF_TOKEN"], local_dialogs=sorted(glob.glob(f"{dialogs_dir}/*.jsonl")), story_root="/data/stage1/bilge_stories")
    rng.shuffle(mix)
    replay = mix[:n_replay]
    ex = convs + replay
    rng.shuffle(ex)
    out = out or f"/data/raft_{model_name}"
    os.makedirs(out, exist_ok=True)
    tok = Tokenizer.from_file("/data/tokenizer/ufakzeka.json")
    stats = {"positives": len(rows), "kept": len(kept), "per_family": {f: min(len(v), cap) for f, v in by_fam.items()}, "replay": len(replay),
             "val": tokenize_examples(ex[:200], tok, f"{out}/val"), "train": tokenize_examples(ex[200:], tok, f"{out}/train")}
    data_vol.commit()
    return stats


@app.function(image=image, volumes={"/data": data_vol, "/ckpt": ckpt_vol}, cpu=4.0, memory=16384, timeout=1800)
def export_checkpoint(ckpt: str, out: str) -> dict:
    """Export a training checkpoint on the ckpt volume to an HF folder next to it."""
    import json
    from ufakzeka.train.export_hf import export
    data_vol.reload(); ckpt_vol.reload()
    export(ckpt, "/data/tokenizer/ufakzeka.json", out)
    ckpt_vol.commit()
    return json.load(open(f"{out}/export_meta.json"))


@app.function(image=image, gpu="L4", volumes={"/data": data_vol, "/ckpt": ckpt_vol}, timeout=1800)
def parity_check(ckpt: str, hf_dir: str, heldout: str = "/data/stage3/heldout/fw2hq.bin", n_seq: int = 256) -> dict:
    """Held out loss of the training model (with soft-cap) versus the HF export (without)."""
    import numpy as np, torch
    from transformers import AutoModelForCausalLM
    from ufakzeka.train.model import ModelConfig, Transformer
    data_vol.reload(); ckpt_vol.reload()
    ck = torch.load(ckpt, map_location="cuda")
    ours = Transformer(ModelConfig(**ck["cfg"]["model"])).cuda().eval(); ours.load_state_dict(ck["model"])
    hf = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.bfloat16).cuda().eval()
    m = np.memmap(heldout, dtype=np.uint16, mode="r")
    L = 2049
    arr = torch.from_numpy(np.array(m[: n_seq * L], dtype=np.int64).reshape(n_seq, L)).cuda()
    hf32 = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.float32).cuda().eval()
    tot = {"ours_softcap": 0.0, "ours_nocap": 0.0, "hf_bf16": 0.0, "hf_fp32": 0.0, "hf_fp32_softcap": 0.0}
    ce = torch.nn.functional.cross_entropy
    with torch.no_grad():
        for i in range(0, n_seq, 8):
            x, y = arr[i:i + 8, :-1], arr[i:i + 8, 1:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, lo = ours(x, y); tot["ours_softcap"] += lo.item()
                ours.cfg.softcap = 0.0
                _, lo = ours(x, y); tot["ours_nocap"] += lo.item()
                ours.cfg.softcap = 30.0
                lg = hf(input_ids=x).logits.float()
            tot["hf_bf16"] += ce(lg.reshape(-1, lg.size(-1)), y.reshape(-1)).item()
            lg = hf32(input_ids=x).logits
            tot["hf_fp32"] += ce(lg.reshape(-1, lg.size(-1)), y.reshape(-1)).item()
            lg = 30.0 * torch.tanh(lg / 30.0)
            tot["hf_fp32_softcap"] += ce(lg.reshape(-1, lg.size(-1)), y.reshape(-1)).item()
    n = n_seq // 8
    return {k: round(v / n, 4) for k, v in tot.items()}


@app.function(image=image, gpu="L40S", volumes={"/data": data_vol, "/ckpt": ckpt_vol}, timeout=3 * 3600)
def run_scenarios(model_name: str, suite: str = "/data/eval/scenarios.json", seeds: int = 2, temperature: float = 0.3, out: str = "") -> dict:
    """The scenario suite on an L4 (about $0.80 an hour): a thousand multi-turn conversations in under an hour,
    so the Mac stays free for the other batteries. Result written to /data/eval/scenarios_<model>.json."""
    import json, subprocess, sys
    data_vol.reload(); ckpt_vol.reload()
    out = out or f"/data/eval/scenarios_{model_name}.json"
    r = subprocess.run([sys.executable, "-u", "scripts/run_scenarios.py", "--model", f"/ckpt/{model_name}/hf", "--suite", suite,
                        "--out", out, "--seeds", str(seeds), "--temperature", str(temperature), "--batch", "24"],
                       capture_output=True, text=True, cwd="/root")
    print(r.stdout[-6000:]); print(r.stderr[-3000:])
    data_vol.commit()
    d = json.load(open(out))
    return {"overall": d["overall"], "summary": d["summary"], "failures": len(d["failures"])}


@app.function(image=image, gpu="L4", volumes={"/data": data_vol, "/ckpt": ckpt_vol}, timeout=2 * 3600)
def run_gates(model_name: str, only: str = "") -> dict:
    """The six release gates on an L4. The same battery takes hours per checkpoint on the Mac because
    it is a few hundred unbatched fp32 generations; here it is minutes, and the checkpoints already
    live on the volume so nothing has to be uploaded. Written to /data/eval/gates_<model>.json."""
    import json, subprocess, sys
    data_vol.reload(); ckpt_vol.reload()
    # a subset lands in its own file so it never overwrites a full battery
    out = f"/data/eval/gates_{model_name}{'_' + only.replace(',', '_') if only else ''}.json"
    r = subprocess.run([sys.executable, "-u", "scripts/gate_probe.py", "--model", f"/ckpt/{model_name}/hf",
                        "--json", out] + (["--only", only] if only else []), capture_output=True, text=True, cwd="/root")
    print(r.stdout[-6000:]); print(r.stderr[-2000:])
    data_vol.commit()
    return json.load(open(out))["gates"]


@app.function(image=image, gpu="L4", volumes={"/data": data_vol, "/ckpt": ckpt_vol}, timeout=3 * 3600)
def run_pairs(model_name: str, slice_spec: str = "0/1", k: int = 6, n_arith: int = 700, out: str = "") -> dict:
    """Rule-verified preference pairs on an L4, one slice per call, written to the data volume. Added when
    Colab session creation started failing on SSL timeouts mid-pipeline (2026-09-01); Modal is the fallback."""
    import subprocess, sys
    data_vol.reload(); ckpt_vol.reload()
    tag = slice_spec.replace("/", "of")
    out = out or f"/data/onpolicy/rule_pairs_{model_name}.part{tag}.jsonl"
    r = subprocess.run([sys.executable, "-u", "-m", "ufakzeka.data.rule_pairs", "--model", f"/ckpt/{model_name}/hf", "--out", out,
                        "--k", str(k), "--n-arith", str(n_arith), "--batch", "32", "--slice", slice_spec], capture_output=True, text=True, cwd="/root")
    print(r.stdout[-4000:]); print(r.stderr[-2000:])
    data_vol.commit()
    return {"out": out, "lines": sum(1 for _ in open(out))}


@app.function(image=image, gpu="L40S", volumes={"/data": data_vol, "/ckpt": ckpt_vol}, timeout=2 * 3600)
def dpo(sft_name: str, pairs: str, suffix: str = "rdpo", extra_repeat: int = 3, max_pairs: int = 0) -> dict:
    """DPO on the SFT checkpoint with the given pairs file (data volume path), export to /ckpt/<name>-<suffix>/hf."""
    import subprocess, sys
    data_vol.reload(); ckpt_vol.reload()
    out_dir = f"/ckpt/{sft_name}-{suffix}"
    r = subprocess.run([sys.executable, "-u", "-m", "ufakzeka.train.dpo_run", "--sft", f"/ckpt/{sft_name}/final.pt", "--out", out_dir,
                        "--tokenizer", "/data/tokenizer/ufakzeka.json", "--extra", f"/data/{pairs}", "--extra-repeat", str(extra_repeat),
                        "--max-pairs", str(max_pairs)], capture_output=True, text=True, cwd="/root")
    print(r.stdout[-4000:]); print(r.stderr[-2000:])
    from ufakzeka.train.export_hf import export
    export(f"{out_dir}/final.pt", "/data/tokenizer/ufakzeka.json", f"{out_dir}/hf")
    ckpt_vol.commit()
    return {"out": out_dir, "rc": r.returncode}


@app.function(image=image, gpu=os.environ.get("UFAKZEKA_SFT_GPU", "L40S"), volumes={"/data": data_vol, "/ckpt": ckpt_vol}, secrets=[hf_secret], timeout=6 * 3600)  # H100 for a full SFT: about the same dollars as the L40S in half the wall time
def sft(base: str, name: str, epochs: int = 2, lr: float = 1e-4, data: str = "/data/sft",
        neftune_alpha: float = 0.0, prompt_weight: float = 0.0, weight_decay: float = 0.0, embed_dropout: float = 0.0, seed: int = 1) -> dict:
    from ufakzeka.train.sft_run import run_sft
    data_vol.reload(); ckpt_vol.reload()
    r = run_sft(base, f"/ckpt/{name}", data_dir=data, epochs=epochs, lr=lr, neftune_alpha=neftune_alpha,
                prompt_weight=prompt_weight, weight_decay=weight_decay, embed_dropout=embed_dropout, seed=seed)
    ckpt_vol.commit()
    return r


@app.local_entrypoint()
def main(base: str = "", name: str = "ufakzeka-1-instruct", build_data: bool = False, epochs: int = 2, export_only: str = "", parity: bool = False, neftune_alpha: float = 0.0, data: str = "/data/sft",
         lr: float = 1e-4, prompt_weight: float = 0.0, weight_decay: float = 0.0, embed_dropout: float = 0.0, seed: int = 1):
    import json
    if parity:
        print(json.dumps(parity_check.remote("/ckpt/ufakzeka-1/final.pt", "/ckpt/ufakzeka-1/hf"), indent=1))
        return
    if export_only:
        print(json.dumps(export_checkpoint.remote(export_only, export_only.rsplit("/", 1)[0] + "/hf"), indent=1))
        return
    if build_data:
        print(json.dumps(build_sft_data.remote(), indent=1))
    if base:
        print(json.dumps(sft.remote(base, name, epochs=epochs, lr=lr, neftune_alpha=neftune_alpha, data=data,
                                    prompt_weight=prompt_weight, weight_decay=weight_decay, embed_dropout=embed_dropout, seed=seed), indent=1))

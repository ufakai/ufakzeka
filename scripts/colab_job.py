"""Run one post-training pass on a Colab runtime, driven by the Colab CLI from the Mac.

Usage from the Mac (session must exist and be bootstrapped):
  colab exec -s ufakzeka -f scripts/colab_job.py -- --base ufakzeka-1-s2/final.pt --name ufakzeka-1-instruct-v7 --epochs 3 --dpo --eval

Steps: pull code, tokenizer, SFT shards and the base checkpoint from the Modal volumes;
run SFT (and optionally DPO and the benchmark); push results back to the ckpt volume."""

import argparse
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--micro-batch", type=int, default=8)
ap.add_argument("--neftune-alpha", type=float, default=0.0, help="uniform embedding noise during SFT, 5 is the paper default")
ap.add_argument("--data-path", default="sft", help="ufakzeka-data volume dir with train.bin/mask, val.bin/mask (e.g. sft_v29b)")
ap.add_argument("--dpo", action="store_true"); ap.add_argument("--eval", action="store_true")
ap.add_argument("--sample", default=None, help="volume path (ufakzeka-data) of prompts jsonl; sample the final model on it and upload onpolicy/samples_<name>.jsonl")
ap.add_argument("--extra", default=None, help="volume path (ufakzeka-data) of on-policy pairs jsonl for DPO")
ap.add_argument("--dpo-args", default="", help="extra flags for dpo_run, e.g. '--max-pairs 0 --lr 3e-7'")
ap.add_argument("--dpo-suffix", default="dpo", help="output name suffix for the DPO checkpoint")
ap.add_argument("--extra-sft", default="", help="extra flags for sft_run, e.g. '--lr 1e-3 --prompt-weight 0.2'")
ap.add_argument("--sft-done", action="store_true", help="skip SFT: the SFT checkpoint <name>/final.pt already exists on the ckpt volume (DPO only run)")
a = ap.parse_args()


def sh(cmd):
    print("+", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode:
        sys.exit(f"failed: {cmd}")


os.environ["HF_TOKEN"] = open(os.path.expanduser("~/.hf_token")).read().strip()
os.makedirs("/content/w", exist_ok=True); os.makedirs("/data/tokenizer", exist_ok=True); os.makedirs("/data/sft", exist_ok=True); os.makedirs("/ckpt", exist_ok=True)
sh("cd /content/w && modal volume get ufakzeka-data code/ufakzeka-repo.tgz repo.tgz --force >/dev/null && tar xzf repo.tgz 2>/dev/null; rm -f repo.tgz")
sh("modal volume get ufakzeka-data tokenizer/ufakzeka.json /data/tokenizer/ufakzeka.json --force >/dev/null")
for f in ("train.bin", "train.mask", "val.bin", "val.mask"):
    sh(f"modal volume get ufakzeka-data {a.data_path}/{f} /data/sft/{f} --force >/dev/null")
os.chdir("/content/w")
final = f"/ckpt/{a.name}"
if a.sft_done:
    os.makedirs(final, exist_ok=True)
    sh(f"modal volume get ufakzeka-ckpt {a.name}/final.pt {final}/final.pt --force >/dev/null")
    os.makedirs(f"{final}/hf", exist_ok=True)  # the sampler and lm_eval read the HF export; fetch file by file (a directory get into a new path writes a file)
    for f in ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        sh(f"modal volume get ufakzeka-ckpt {a.name}/hf/{f} {final}/hf/{f} --force >/dev/null")
    # only the pre-s3 soft-cap exports carry a custom modeling file; stock Qwen3 exports do not
    subprocess.run(f"modal volume get ufakzeka-ckpt {a.name}/hf/modeling_ufakzeka.py {final}/hf/modeling_ufakzeka.py --force >/dev/null 2>&1", shell=True)
else:
    os.makedirs("/ckpt/" + a.base.split("/")[0], exist_ok=True)
    sh(f"modal volume get ufakzeka-ckpt {a.base} /ckpt/{a.base} --force >/dev/null")
    sh(f"python -u -m ufakzeka.train.sft_run --base /ckpt/{a.base} --out /ckpt/{a.name} --data /data/sft --tokenizer /data/tokenizer/ufakzeka.json --epochs {a.epochs} --micro-batch {a.micro_batch} --neftune-alpha {a.neftune_alpha} {a.extra_sft.replace(",", " ")}")
    sh(f"modal volume put ufakzeka-ckpt /ckpt/{a.name} {a.name} --force")
    print("SFT_UPLOADED", a.name, flush=True)
if a.dpo:
    ex = ""
    if a.extra:
        sh(f"modal volume get ufakzeka-data {a.extra} /data/pairs.jsonl --force >/dev/null"); ex = " --extra /data/pairs.jsonl"
    sh(f"python -u -m ufakzeka.train.dpo_run --sft /ckpt/{a.name}/final.pt --out /ckpt/{a.name}-{a.dpo_suffix} --tokenizer /data/tokenizer/ufakzeka.json{ex} {a.dpo_args}")
    final = f"/ckpt/{a.name}-{a.dpo_suffix}"
    sh(f"modal volume put ufakzeka-ckpt /ckpt/{a.name}-{a.dpo_suffix} {a.name}-{a.dpo_suffix} --force")
    print("DPO_UPLOADED", a.name, flush=True)
if a.sample:
    sh(f"modal volume get ufakzeka-data {a.sample} /data/prompts.jsonl --force >/dev/null")
    sh(f"python -u -m ufakzeka.data.onpolicy sample --model {final}/hf --prompts /data/prompts.jsonl --out /data/samples.jsonl")
    sh(f"modal volume put ufakzeka-data /data/samples.jsonl onpolicy/samples_{a.name}.jsonl --force")
    print("SAMPLES_UPLOADED", a.name, flush=True)
if a.eval:
    sh("pip -q install 'lm_eval>=0.4.9' accelerate 2>&1 | tail -1")
    # benchmark the SFT checkpoint as well as the DPO one, so a regression can be attributed to the right stage
    for tag in ([a.name + "-" + a.dpo_suffix] if a.sft_done and a.dpo else [a.name, a.name + "-" + a.dpo_suffix] if a.dpo else [a.name]):
        print("EVAL_START", tag, flush=True)
        sh(f"lm_eval --model hf --model_args pretrained=/ckpt/{tag}/hf,dtype=bfloat16,trust_remote_code=True --tasks hellaswag_tr,arc_tr_challenge,arc_tr_easy,xcopa_tr,belebele_tur_Latn,turblimp_core,turkishmmlu --include_path ufakzeka/eval/tasks --batch_size 16 --trust_remote_code --output_path /ckpt/evals/{tag} 2>&1 | grep -E '^\\|' | grep -vE 'Filter|^\\|-|turblimp_[a-z]|turkishmmlu_'")
        sh(f"modal volume put ufakzeka-ckpt /ckpt/evals/{tag} evals/colab_{tag} --force")
        print("EVAL_DONE", tag, flush=True)
print("JOB_DONE", a.name, flush=True)

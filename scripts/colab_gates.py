"""Run the release gate battery over several checkpoints on a Colab runtime.

The battery is a few hundred unbatched fp32 generations per checkpoint. That is hours on the Mac and
minutes on an A100, and nothing has to be uploaded: the checkpoints already live on the Modal ckpt
volume, so the runtime pulls them directly.

  colab exec -s ufakzeka-gates -f scripts/colab_gates.py -- --models v93s2,v98s2,v100s2

Results are pushed back to the data volume as eval/gates_<model>.json and printed, so the tables are
readable in the job log even if the push fails.
"""
import argparse
import json
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True, help="comma separated version tags, e.g. v98s2,v99s2")
ap.add_argument("--prefix", default="ufakzeka-1-instruct-", help="checkpoint name prefix on the ckpt volume")
ap.add_argument("--only", default="", help="subset of gates, passed through to gate_probe")
a = ap.parse_args()


def sh(cmd, check=True):
    print("+", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode:
        sys.exit(f"failed: {cmd}")


os.makedirs("/content/w", exist_ok=True)
sh("cd /content/w && modal volume get ufakzeka-data code/ufakzeka-repo.tgz repo.tgz --force >/dev/null "
   "&& tar xzf repo.tgz 2>/dev/null; rm -f repo.tgz")
os.chdir("/content/w")

# the HF export is five files; a directory get into a fresh path writes a file instead of a tree
FILES = ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
tags = [t.strip() for t in a.models.split(",") if t.strip()]
results = {}

for tag in tags:
    name = f"{a.prefix}{tag}"
    dest = f"/ckpt/{name}/hf"
    os.makedirs(dest, exist_ok=True)
    for f in FILES:
        sh(f"modal volume get ufakzeka-ckpt {name}/hf/{f} {dest}/{f} --force >/dev/null")
    subprocess.run(f"modal volume get ufakzeka-ckpt {name}/hf/modeling_ufakzeka.py {dest}/modeling_ufakzeka.py "
                   f"--force >/dev/null 2>&1", shell=True)
    out = f"/content/gates_{tag}.json"
    only = f"--only {a.only}" if a.only else ""
    # a battery that fails must not take the rest of the queue down with it
    r = subprocess.run(f"python -u scripts/gate_probe.py --model {dest} {only} --json {out}",
                       shell=True, capture_output=True, text=True)
    print(r.stdout[-6000:], flush=True)
    if r.stderr:
        print(r.stderr[-2000:], flush=True)
    if os.path.exists(out):
        results[tag] = json.load(open(out))["gates"]
        sh(f"modal volume put ufakzeka-data {out} eval/gates_{name}.json --force >/dev/null", check=False)
    else:
        print(f"GATES_FAILED {tag}", flush=True)
    # 351 MB per fp32 export; a Colab disk fills quietly and the next pull half succeeds
    sh(f"rm -rf /ckpt/{name}", check=False)
    print(f"GATES_{tag}_DONE", flush=True)

print("\n================ SUMMARY ================", flush=True)
for tag, g in results.items():
    row = "  ".join(f"{k} {v['score']}" for k, v in g.items())
    print(f"{tag:10s} {row}", flush=True)
print("JOB_DONE", flush=True)

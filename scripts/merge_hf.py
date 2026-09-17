"""Linear merge of two HF exports with the same layout: out = w * a + (1 - w) * b.
SmolLM3 merged its post-trained model 0.9 with the mid-training checkpoint 0.1 to recover capabilities lost in
post-training; we use it against the TurBLiMP drop after SFT.
  python scripts/merge_hf.py --a models/ufakzeka-1-instruct-v7/hf --b models/ufakzeka-1-s2 --w 0.9 --out models/ufakzeka-1-instruct-v7-merge"""
import argparse, os, shutil
import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--w", type=float, default=0.9); ap.add_argument("--out", required=True)
args = ap.parse_args()
sa, sb = load_file(os.path.join(args.a, "model.safetensors")), load_file(os.path.join(args.b, "model.safetensors"))
assert sa.keys() == sb.keys(), set(sa) ^ set(sb)
merged = {k: (args.w * sa[k].float() + (1 - args.w) * sb[k].float()).to(sa[k].dtype) for k in sa}
os.makedirs(args.out, exist_ok=True)
for f in os.listdir(args.a):
    if f != "model.safetensors":
        shutil.copy(os.path.join(args.a, f), os.path.join(args.out, f))
save_file(merged, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})
print("merged", len(merged), "tensors, w =", args.w, "->", args.out)

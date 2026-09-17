"""Run the Turkish benchmark suite on an exported HF model or a hub model.

  modal run modal/evaluate.py --model /ckpt/ufakzeka-1/hf
  modal run modal/evaluate.py --model asafaya/kanarya-750m
  modal run modal/evaluate.py --model ytu-ce-cosmos/turkish-gpt2-large --limit 200

Tasks: hellaswag_tr, arc_tr_challenge, arc_tr_easy (custom, malhajar translations),
xcopa_tr, belebele_tur_Latn, turblimp_core, turkishmmlu (built in). All zero-shot,
log-likelihood scoring, so results are comparable across base models. Results are
written to the ckpt volume under evals/<model-name>.json."""

from __future__ import annotations

import modal

app = modal.App("ufakzeka-evaluate")
ckpt_vol = modal.Volume.from_name("ufakzeka-ckpt", version=2, create_if_missing=True)
hf_secret = modal.Secret.from_name("ufakzeka-hf")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.8", "transformers>=4.45", "lm_eval>=0.4.9", "accelerate", "sentencepiece", "protobuf")
    .add_local_dir("ufakzeka/eval/tasks", remote_path="/tasks")
)

TASKS = "hellaswag_tr,arc_tr_challenge,arc_tr_easy,xcopa_tr,belebele_tur_Latn,turblimp_core,turkishmmlu"


@app.function(image=image, gpu="L4", volumes={"/ckpt": ckpt_vol}, secrets=[hf_secret], timeout=3 * 3600)
def evaluate(model: str, tasks: str = TASKS, limit: int = 0, batch_size: int = 16) -> dict:
    import json, os, re, subprocess, time
    ckpt_vol.reload()
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", model.strip("/"))
    out = f"/ckpt/evals/{name}"
    os.makedirs(out, exist_ok=True)
    cmd = ["lm_eval", "--model", "hf", "--model_args", f"pretrained={model},dtype=bfloat16,trust_remote_code=True",
           "--tasks", tasks, "--include_path", "/tasks", "--batch_size", str(batch_size),
           "--output_path", out, "--log_samples" if False else "--trust_remote_code"]
    if limit:
        cmd += ["--limit", str(limit)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-4000:])
    # collect the results json lm_eval wrote
    res = {}
    for root, _, files in os.walk(out):
        for f in files:
            if f.startswith("results") and f.endswith(".json"):
                res = json.load(open(os.path.join(root, f)))["results"]
    summary = {t: {k: round(v, 4) for k, v in m.items() if isinstance(v, float) and ("acc" in k) and "stderr" not in k}
               for t, m in res.items()}
    summary["_sec"] = round(time.time() - t0)
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    ckpt_vol.commit()
    return summary


@app.local_entrypoint()
def main(model: str, tasks: str = TASKS, limit: int = 0):
    import json
    print(json.dumps(evaluate.remote(model, tasks, limit), indent=1))

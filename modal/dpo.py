"""Direct preference optimisation of an SFT checkpoint on Turkish preference pairs.

  modal run modal/dpo.py --sft /ckpt/ufakzeka-1-instruct-v3/final.pt --name ufakzeka-1-instruct-dpo

Data: selimc/orpo-dpo-mix-TR-20k (Apache-2.0), chosen and rejected in chat format. Loss:
sigmoid DPO with beta 0.1, reference = frozen SFT model, one epoch, lr 5e-7 cosine, sequences
truncated to 1024 tokens. Only the response tokens count in the log-probabilities."""

from __future__ import annotations

import modal

app = modal.App("ufakzeka-dpo")
data_vol = modal.Volume.from_name("ufakzeka-data", version=2)
ckpt_vol = modal.Volume.from_name("ufakzeka-ckpt", version=2, create_if_missing=True)
hf_secret = modal.Secret.from_name("ufakzeka-hf")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch>=2.8", "numpy>=2", "tokenizers>=0.20", "datasets>=3.0", "transformers>=4.45")
    .add_local_python_source("ufakzeka")
)


@app.function(image=image, gpu="L40S", volumes={"/data": data_vol, "/ckpt": ckpt_vol}, secrets=[hf_secret], timeout=4 * 3600)
def dpo(sft: str, name: str, beta: float = 0.1, lr: float = 5e-7) -> dict:
    from ufakzeka.train.dpo_run import run_dpo
    data_vol.reload(); ckpt_vol.reload()
    r = run_dpo(sft, f"/ckpt/{name}", beta=beta, lr=lr)
    ckpt_vol.commit()
    return r


@app.local_entrypoint()
def main(sft: str, name: str = "ufakzeka-1-instruct-dpo", beta: float = 0.1, lr: float = 5e-7):
    import json
    print(json.dumps(dpo.remote(sft, name, beta=beta, lr=lr), indent=1))

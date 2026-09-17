"""Run pretraining on Modal.

  modal run --detach modal/train.py --gpu H100 --total-tokens 7e9 --name ufakzeka-1
  modal run modal/train.py --gpu L4 --total-tokens 2e8 --name abl-mix-a --preset small

The job checkpoints to the ufakzeka-ckpt volume and restarts itself after preemption."""

from __future__ import annotations

import modal

app = modal.App("ufakzeka-train")
data_vol = modal.Volume.from_name("ufakzeka-data", version=2)
ckpt_vol = modal.Volume.from_name("ufakzeka-ckpt", version=2, create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch>=2.8", "numpy>=2", "tokenizers>=0.20")
    .add_local_python_source("ufakzeka")
)

PRESETS = {
    "base": dict(n_layer=24, d_model=768, n_head=12, n_kv_head=4, d_ff=2048),
    "small": dict(n_layer=12, d_model=384, n_head=6, n_kv_head=2, d_ff=1024),   # ~25M for ablations
}


def _run(name: str, total_tokens: int, preset: str, overrides: dict) -> dict:
    import json
    from ufakzeka.train.model import ModelConfig
    from ufakzeka.train.train import TrainConfig, train

    try:
        ckpt_vol.reload()  # pick up checkpoints written by an earlier attempt of this run
    except Exception as e:
        print("ckpt reload skipped:", e)
    cfg = TrainConfig(model=ModelConfig(**PRESETS[preset]), total_tokens=int(total_tokens), out_dir=f"/ckpt/{name}")
    cfg.ckpt_commit = ckpt_vol.commit
    for k, v in overrides.items():
        if k in ("mix_stable", "mix_anneal"):
            v = json.loads(v) if isinstance(v, str) else v
        if k.startswith("model."):
            setattr(cfg.model, k[6:], v)
        else:
            setattr(cfg, k, v)
    step, tokens, interrupted = train(cfg)
    ckpt_vol.commit()
    return {"name": name, "step": step, "tokens": tokens, "interrupted": interrupted}


@app.function(image=image, gpu="H100", volumes={"/data": data_vol, "/ckpt": ckpt_vol}, timeout=24 * 3600,
              retries=modal.Retries(max_retries=8, initial_delay=0.0))
def train_h100(name: str, total_tokens: int, preset: str, overrides: dict) -> dict:
    return _run(name, total_tokens, preset, overrides)


@app.function(image=image, gpu="L4", volumes={"/data": data_vol, "/ckpt": ckpt_vol}, timeout=6 * 3600,
              retries=modal.Retries(max_retries=3, initial_delay=0.0))
def train_l4(name: str, total_tokens: int, preset: str, overrides: dict) -> dict:
    return _run(name, total_tokens, preset, overrides)


@app.local_entrypoint()
def main(name: str, total_tokens: float, gpu: str = "H100", preset: str = "base",
         micro_batch: int = 0, batch_tokens: int = 0, mix_stable: str = "", mix_anneal: str = "",
         muon_lr: float = 0.0, warmup_steps: int = 0, eval_every: int = 0, compile: bool = True,
         init_from: str = "", hyperball: bool = False, decay_frac: float = 0.0, batch_ramp_steps: int = -1,
         softcap: float = -1.0, z_loss: float = -1.0, seq_len: int = 0, rope_theta: float = 0.0):
    """softcap 0 removes the logit soft-cap (planned for the Sep 2026 recency anneal, with z_loss 1e-4);
    seq_len 4096 extends the context (halve micro_batch); rope_theta raises the RoPE base for the longer context."""
    ov = {}
    if softcap >= 0: ov["model.softcap"] = softcap
    if z_loss >= 0: ov["z_loss"] = z_loss
    if seq_len: ov["model.seq_len"] = seq_len
    if rope_theta: ov["model.rope_theta"] = rope_theta
    if init_from: ov["init_from"] = init_from
    if hyperball: ov["hyperball"] = True
    if decay_frac: ov["decay_frac"] = decay_frac
    if batch_ramp_steps >= 0: ov["batch_ramp_steps"] = batch_ramp_steps
    if micro_batch: ov["micro_batch"] = micro_batch
    if batch_tokens: ov["batch_tokens"] = batch_tokens
    if mix_stable: ov["mix_stable"] = mix_stable
    if mix_anneal: ov["mix_anneal"] = mix_anneal
    if muon_lr: ov["muon_lr"] = muon_lr
    if warmup_steps: ov["warmup_steps"] = warmup_steps
    if eval_every: ov["eval_every"] = eval_every
    ov["compile"] = compile
    fn = {"H100": train_h100, "L4": train_l4}[gpu]
    print(fn.remote(name, int(total_tokens), preset, ov))

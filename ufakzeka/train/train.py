"""Single GPU pretraining loop for ufakzeka.

Schedule: warmup, stable (constant LR), then linear decay to zero over the final
`decay_frac` of steps (WSD). The anneal mixture is switched on at the start of the decay.
Batch size ramps linearly in tokens from `batch_tokens_start` to `batch_tokens` over the
first `batch_ramp_steps` steps. Optimiser: Muon for block matrices, AdamW for embeddings
and norms. Checkpoints go to `out_dir` every `ckpt_every` steps and on SIGTERM
(Modal preemption), and training resumes from the latest one automatically."""

from __future__ import annotations

import glob
import json
import math
import os
import signal
import time
from dataclasses import asdict, dataclass, field

import torch

from ufakzeka.train.data import TierStream, heldout_batches
from ufakzeka.train.model import ModelConfig, Transformer
from ufakzeka.train.muon import Muon


@dataclass
class TrainConfig:
    data_root: str = "/data/stage3/train"
    heldout_glob: str = "/data/stage3/heldout/*.bin"
    out_dir: str = "/ckpt/ufakzeka-1"
    tokenizer_path: str = "/data/tokenizer/ufakzeka.json"
    model: ModelConfig = field(default_factory=ModelConfig)
    total_tokens: int = 7_000_000_000
    batch_tokens: int = 524_288
    batch_tokens_start: int = 131_072
    batch_ramp_steps: int = 1500
    micro_batch: int = 16
    warmup_steps: int = 1000
    decay_frac: float = 0.15
    mix_stable: dict = field(default_factory=lambda: {"A": 0.25, "B": 0.71, "C": 0.04})
    mix_anneal: dict = field(default_factory=lambda: {"A": 0.60, "B": 0.37, "C": 0.03})
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_wd: float = 0.1
    adam_lr: float = 3e-3       # embeddings and head (tied), norms use adam_lr * 0.1 scaled below
    adam_lr_scalar: float = 3e-4
    adam_betas: tuple = (0.9, 0.95)
    adam_wd: float = 0.0
    z_loss: float = 1e-5
    grad_clip: float = 1.0
    eval_every: int = 500
    eval_tokens: int = 1_000_000
    sample_every: int = 1000
    ckpt_every: int = 1000
    ckpt_every_sec: int = 600
    log_every: int = 20
    seed: int = 1
    compile: bool = True
    dtype: str = "bfloat16"
    device: str = "cuda"
    init_from: str = ""       # path to a checkpoint whose model weights start this run (continued pretraining)
    hyperball: bool = False   # keep each block matrix at its starting Frobenius norm (arXiv 2606.16899, minimal form)
    ckpt_commit: object = None  # callable run after each checkpoint (Modal volume commit)


def lr_scale(step: int, total_steps: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    decay_start = int(total_steps * (1 - cfg.decay_frac))
    if step < decay_start:
        return 1.0
    return max(0.0, (total_steps - step) / max(1, total_steps - decay_start))


def batch_tokens_at(step: int, cfg: TrainConfig) -> int:
    if step >= cfg.batch_ramp_steps:
        return cfg.batch_tokens
    f = step / cfg.batch_ramp_steps
    bt = cfg.batch_tokens_start + f * (cfg.batch_tokens - cfg.batch_tokens_start)
    seq = cfg.model.seq_len
    return int(round(bt / (seq * cfg.micro_batch))) * seq * cfg.micro_batch or seq * cfg.micro_batch


def plan_steps(cfg: TrainConfig) -> int:
    """Number of optimizer steps so that the summed batch sizes reach total_tokens."""
    tokens, step = 0, 0
    while tokens < cfg.total_tokens:
        tokens += batch_tokens_at(step, cfg)
        step += 1
    return step


def build_optimizers(model: Transformer, cfg: TrainConfig):
    matrices = [p for n, p in model.named_parameters() if p.ndim == 2 and n.startswith("layers.")]
    embed = [model.embed_tokens.weight]
    scalars = [p for n, p in model.named_parameters() if p.ndim < 2]
    muon = Muon(matrices, lr=cfg.muon_lr, momentum=cfg.muon_momentum, weight_decay=cfg.muon_wd)
    adam = torch.optim.AdamW([
        {"params": embed, "lr": cfg.adam_lr, "weight_decay": cfg.adam_wd},
        {"params": scalars, "lr": cfg.adam_lr_scalar, "weight_decay": 0.0},
    ], betas=cfg.adam_betas, eps=1e-8, fused=torch.cuda.is_available())
    for g in muon.param_groups + adam.param_groups:
        g["base_lr"] = g["lr"]
    return muon, adam


def cfg_dict(cfg: TrainConfig) -> dict:
    d = {k: v for k, v in cfg.__dict__.items() if k != "ckpt_commit"}
    d["model"] = asdict(cfg.model)
    return d


def save_ckpt(path: str, model, opts, step: int, tokens: int, cfg: TrainConfig):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opts": [o.state_dict() for o in opts],
                "step": step, "tokens": tokens, "cfg": cfg_dict(cfg)}, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model, cfg: TrainConfig, device, dtype) -> dict[str, float]:
    model.eval()
    out = {}
    eval_bs = max(1, cfg.micro_batch // 4)  # uncompiled forward keeps float32 logits; keep it small
    for f in sorted(glob.glob(cfg.heldout_glob)):
        name = os.path.basename(f).replace(".bin", "")
        tot, n = 0.0, 0
        for x, y in heldout_batches(f, cfg.model.seq_len, cfg.eval_tokens, eval_bs):
            with torch.autocast(device, dtype=dtype, enabled=(device == "cuda")):
                _, loss = model(x.to(device), y.to(device))
            tot += loss.item(); n += 1
        out[name] = round(tot / max(n, 1), 4)
    model.train()
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


@torch.no_grad()
def sample(model, tok, device, prompts: list[str], max_new: int = 60) -> list[str]:
    model.eval()
    outs = []
    for p in prompts:
        ids = torch.tensor([tok.encode(p).ids], device=device)
        y = model.generate(ids, max_new, temperature=0.0)
        outs.append(tok.decode(y[0].tolist()))
    model.train()
    return outs


def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    dtype = torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(cfg.out_dir, exist_ok=True)

    model = Transformer(cfg.model).to(device)
    print(f"params: {model.num_params()/1e6:.1f}M non-embedding, {model.num_params(False)/1e6:.1f}M total")
    raw = model
    if cfg.compile:
        model = torch.compile(model)
    muon, adam = build_optimizers(raw, cfg)

    total_steps = plan_steps(cfg)
    stream = TierStream(cfg.data_root, cfg.mix_stable, cfg.model.seq_len, seed=cfg.seed)
    print("tokens available per tier:", stream.tokens_available(), "planned steps:", total_steps)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(cfg.tokenizer_path)
    prompts = ["Türkiye'nin başkenti", "Bir zamanlar küçük bir köyde", "Fotosentez, bitkilerin", "İstanbul Boğazı,"]

    step, tokens_seen = 0, 0
    latest = os.path.join(cfg.out_dir, "latest.pt")
    if cfg.init_from and not os.path.exists(latest):
        ck = torch.load(cfg.init_from, map_location=device)
        raw.load_state_dict(ck["model"])
        print(f"initialised weights from {cfg.init_from} (step {ck.get('step')}, {ck.get('tokens', 0)/1e9:.2f}B tokens)")
        del ck
    if cfg.hyperball:
        for p in muon.param_groups[0]["params"]:
            muon.state[p]["target_norm"] = p.norm().item()
        for g in muon.param_groups:
            g["weight_decay"] = 0.0
    if os.path.exists(latest):
        ck = torch.load(latest, map_location=device)
        raw.load_state_dict(ck["model"])
        for o, s in zip((muon, adam), ck["opts"]):
            o.load_state_dict(s)
        step, tokens_seen = ck["step"], ck["tokens"]
        print(f"resumed from step {step}, {tokens_seen/1e9:.3f}B tokens")

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))

    log = open(os.path.join(cfg.out_dir, "log.jsonl"), "a")
    decay_start = int(total_steps * (1 - cfg.decay_frac))
    in_anneal = step >= decay_start
    if in_anneal:
        stream.set_mixture(cfg.mix_anneal)
    t0 = time.time(); tokens_t0 = tokens_seen; last_ckpt = time.time()
    model.train()
    while step < total_steps and not stop["flag"]:
        if not in_anneal and step >= decay_start:
            stream.set_mixture(cfg.mix_anneal); in_anneal = True
            print(f"step {step}: anneal mixture {cfg.mix_anneal}")
        bt = batch_tokens_at(step, cfg)
        accum = max(1, bt // (cfg.model.seq_len * cfg.micro_batch))
        scale = lr_scale(step, total_steps, cfg)
        for o in (muon, adam):
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * scale
        muon.param_groups[0]["momentum"] = min(cfg.muon_momentum, 0.85 + 0.10 * min(1.0, step / 500))

        loss_acc = 0.0
        for _ in range(accum):
            x, y = stream.batch(cfg.micro_batch, device)
            with torch.autocast(device, dtype=dtype, enabled=(device == "cuda")):
                _, loss = model(x, y, z_loss=cfg.z_loss)
            (loss / accum).backward()
            loss_acc += loss.item() / accum
        gn = torch.nn.utils.clip_grad_norm_(raw.parameters(), cfg.grad_clip).item()
        muon.step(); adam.step()
        if cfg.hyperball:
            with torch.no_grad():
                for p in muon.param_groups[0]["params"]:
                    p.mul_(muon.state[p]["target_norm"] / (p.norm() + 1e-8))
        muon.zero_grad(set_to_none=True); adam.zero_grad(set_to_none=True)
        step += 1; tokens_seen += accum * cfg.micro_batch * cfg.model.seq_len

        if step % cfg.log_every == 0:
            dt = time.time() - t0
            tps = (tokens_seen - tokens_t0) / max(dt, 1e-6)
            rec = {"step": step, "loss": round(loss_acc, 4), "gnorm": round(gn, 3), "lr_scale": round(scale, 4),
                   "batch_tokens": bt, "tokens": tokens_seen, "tok_per_s": int(tps),
                   "eta_min": round((total_steps - step) * bt / max(tps, 1) / 60)}
            print(json.dumps(rec)); log.write(json.dumps(rec) + "\n"); log.flush()
            t0 = time.time(); tokens_t0 = tokens_seen
        if step % cfg.eval_every == 0:
            ev = evaluate(raw, cfg, device, dtype)
            rec = {"step": step, "eval": ev, "tokens": tokens_seen}
            print(json.dumps(rec)); log.write(json.dumps(rec) + "\n"); log.flush()
        if step % cfg.sample_every == 0:
            for s in sample(raw, tok, device, prompts):
                print("SAMPLE:", s.replace("\n", " | ")[:300])
        if step % cfg.ckpt_every == 0 or time.time() - last_ckpt > cfg.ckpt_every_sec:
            save_ckpt(latest, raw, (muon, adam), step, tokens_seen, cfg)
            last_ckpt = time.time()
            if cfg.ckpt_commit is not None:
                cfg.ckpt_commit()

    save_ckpt(latest, raw, (muon, adam), step, tokens_seen, cfg)
    if step >= total_steps:
        torch.save({"model": raw.state_dict(), "cfg": cfg_dict(cfg), "step": step, "tokens": tokens_seen},
                   os.path.join(cfg.out_dir, "final.pt"))
        ev = evaluate(raw, cfg, device, dtype)
        print("FINAL", json.dumps({"step": step, "tokens": tokens_seen, "eval": ev}))
    return step, tokens_seen, stop["flag"]

"""SFT loop, runnable anywhere with a GPU (Modal, Colab, local). See modal/sft.py for the Modal wrapper."""

from __future__ import annotations


def _amp_dtype():
    import torch
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def run_sft(base: str, out_dir: str, data_dir: str = "/data/sft", tokenizer_path: str = "/data/tokenizer/ufakzeka.json",
            epochs: int = 2, lr: float = 1e-4, seq_len: int = 2048, micro_batch: int = 16, batch_seqs: int = 64,
            neftune_alpha: float = 0.0, phase2_data: str | None = None, phase2_epochs: int = 1,
            prompt_weight: float = 0.0, weight_decay: float = 0.0, embed_dropout: float = 0.0, seed: int = 1) -> dict:
    """neftune_alpha > 0 adds uniform embedding noise (arXiv 2310.05914), which regularises away the formatting
    of our heavily templated sets; 5 is the value the paper uses everywhere.

    phase2_data runs a second curriculum phase over a different packed set after the first finishes, on one
    continuous cosine schedule. CLASS-IT (arXiv 2510.25364) found sequential beat merged at 100-140M, and it
    also decides which block gets the last word on style, which we have measured to matter more than counts."""
    import json, math, os, time
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from ufakzeka.train.model import ModelConfig, Transformer, document_block_mask
    torch.set_float32_matmul_precision("high")
    eot_id = Tokenizer.from_file(tokenizer_path).token_to_id("<|endoftext|>")
    assert eot_id is not None, "tokenizer has no <|endoftext|>; document masking needs it"
    ck = torch.load(base, map_location="cuda")
    mc = ModelConfig(**ck["cfg"]["model"])
    model = Transformer(mc).cuda()
    model.load_state_dict(ck["model"])
    model.neftune_alpha = neftune_alpha
    model.embed_dropout = embed_dropout
    L = seq_len + 1
    vids = np.memmap(f"{data_dir}/val.bin", dtype=np.uint16, mode="r")
    vmsk = np.memmap(f"{data_dir}/val.mask", dtype=np.uint8, mode="r")
    phases = [(data_dir, epochs)] + ([(phase2_data, phase2_epochs)] if phase2_data else [])
    loaded = []
    for d, ep in phases:
        pi = np.memmap(f"{d}/train.bin", dtype=np.uint16, mode="r")
        pm = np.memmap(f"{d}/train.mask", dtype=np.uint8, mode="r")
        loaded.append((pi, pm, ep, (len(pi) - 1) // L))
    total = sum(n // batch_seqs * ep for _, _, ep, n in loaded)
    accum = batch_seqs // micro_batch
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay, fused=True)
    cmodel = torch.compile(model) if torch.cuda.is_bf16_supported() else model  # compile is slow to warm up on T4
    scaler = torch.amp.GradScaler("cuda", enabled=(_amp_dtype() == torch.float16))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    def batch(arr_i, arr_m, idxs):
        x = np.stack([arr_i[i * L:(i + 1) * L] for i in idxs]).astype(np.int64)
        m = np.stack([arr_m[i * L:(i + 1) * L] for i in idxs]).astype(np.int64)
        if prompt_weight > 0:
            # every token is a target; assistant tokens weigh 1, the rest prompt_weight
            y = x[:, 1:]
            # level 2 (the wrong answer in a self-correction example) is never trained, even at
            # prompt_weight: Partial Answer Masking only works if the weight is actually zero.
            mm = m[:, 1:]
            w = torch.from_numpy(np.where(mm == 1, 1.0, np.where(mm == 2, 0.0, prompt_weight)).astype(np.float32)).cuda()
        else:
            y = np.where(m[:, 1:] == 1, x[:, 1:], -1)
            w = None
        x = torch.from_numpy(x[:, :-1]).cuda()
        return x, torch.from_numpy(np.ascontiguousarray(y)).cuda(), document_block_mask(x, eot_id), w

    def val_loss():
        model.eval(); tot, n = 0.0, 0
        nv = (len(vids) - 1) // L
        with torch.no_grad():
            for s in range(0, nv, micro_batch):
                x, y, bm, w = batch(vids, vmsk, range(s, min(s + micro_batch, nv)))
                w = None  # validation stays the plain assistant-token loss, comparable across runs
                with torch.autocast("cuda", dtype=_amp_dtype()):
                    _, loss = cmodel(x, y, block_mask=bm, token_weights=w)
                tot += loss.item(); n += 1
        model.train(); return tot / max(n, 1)

    seqs = ", ".join(f"{n} seqs x{ep} from {d}" for (d, _), (_, _, ep, n) in zip(phases, loaded))
    print(f"sft: {seqs}, {total} steps, neftune {neftune_alpha}, val loss before {val_loss():.4f}")
    step, t0 = 0, time.time()
    for phase, (pids, pmsk, pep, n_seq) in enumerate(loaded):
        steps_per_epoch = n_seq // batch_seqs
        for ep in range(pep):
            order = rng.permutation(n_seq)
            for s in range(0, steps_per_epoch * batch_seqs, batch_seqs):
                cur = lr * 0.5 * (1 + math.cos(math.pi * step / total)) * min(1.0, (step + 1) / max(1, min(50, total // 5)))  # short runs (RAFT rounds) keep a fifth as warmup
                for g in opt.param_groups: g["lr"] = cur
                la = 0.0
                for a in range(accum):
                    idx = order[s + a * micro_batch: s + (a + 1) * micro_batch]
                    x, y, bm, w = batch(pids, pmsk, idx)
                    with torch.autocast("cuda", dtype=_amp_dtype()):
                        _, loss = cmodel(x, y, block_mask=bm, token_weights=w)
                    scaler.scale(loss / accum).backward(); la += loss.item() / accum
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
                if step % 20 == 0:
                    print(json.dumps({"step": step, "loss": round(la, 4), "lr": cur, "sec": round(time.time() - t0)}))
            print(json.dumps({"phase": phase + 1, "epoch": ep + 1, "val_loss": round(val_loss(), 4)}))
    out = out_dir; os.makedirs(out, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": ck["cfg"], "step": step, "base": base}, f"{out}/final.pt")
    from ufakzeka.train.export_hf import export
    export(f"{out}/final.pt", tokenizer_path, f"{out}/hf")
    return {"steps": step, "val_loss": val_loss()}


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="/data/sft"); ap.add_argument("--tokenizer", default="/data/tokenizer/ufakzeka.json")
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--neftune-alpha", type=float, default=0.0)
    ap.add_argument("--prompt-weight", type=float, default=0.0); ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--embed-dropout", type=float, default=0.0); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--phase2-data", default=None); ap.add_argument("--phase2-epochs", type=int, default=1)
    a = ap.parse_args()
    print(json.dumps(run_sft(a.base, a.out, a.data, a.tokenizer, epochs=a.epochs, lr=a.lr, micro_batch=a.micro_batch,
                             neftune_alpha=a.neftune_alpha, phase2_data=a.phase2_data,
                             prompt_weight=a.prompt_weight, weight_decay=a.weight_decay, embed_dropout=a.embed_dropout, seed=a.seed,
                             phase2_epochs=a.phase2_epochs), indent=1))

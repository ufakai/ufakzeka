"""DPO loop, runnable anywhere with a GPU. See modal/dpo.py for the Modal wrapper."""

from __future__ import annotations


def _amp_dtype():
    import torch
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _msgs_to_text(msgs):
    """chat messages -> (prompt_text, response_text) in our ChatML format; last assistant turn is the response."""
    from ufakzeka.train.sft_data import USER, ASSIST, END
    prompt, prompt_before, response = "", None, None
    for m in msgs:
        role, content = m.get("role") or m.get("from"), (m.get("content") or m.get("value") or "").strip()
        if role in ("user", "human"):
            prompt += USER + content + END + ASSIST
        elif role in ("assistant", "gpt") and prompt:
            prompt_before, response = prompt, content + END
            prompt += response  # earlier assistant turns stay in the context of the next one
    return prompt_before if response else None, response


def run_dpo(sft: str, out_dir: str, tokenizer_path: str = "/data/tokenizer/ufakzeka.json", beta: float = 0.1, lr: float = 5e-7,
            max_len: int = 1024, batch: int = 16, micro: int = 4, max_pairs: int = 20000,
            extra: str | None = None, extra_repeat: int = 2, nll_alpha: float = 0.2) -> dict:
    """extra: jsonl of on-policy pairs {prompt: str, chosen: str, rejected: str} (ufakzeka.data.onpolicy judge output),
    repeated extra_repeat times and shuffled into the public mix."""
    import json, math, os, random, time
    import torch, torch.nn.functional as F
    from datasets import load_dataset
    from tokenizers import Tokenizer
    from ufakzeka.train.model import ModelConfig, Transformer
    tok = Tokenizer.from_file(tokenizer_path)
    pairs = []
    ds = load_dataset("selimc/orpo-dpo-mix-TR-20k", split="train", token=os.environ.get("HF_TOKEN")) if max_pairs > 0 else []
    for r in ds:
        ch, rj = r["chosen"], r["rejected"]
        if isinstance(ch, list):
            p1, c = _msgs_to_text(ch); p2, rr = _msgs_to_text(rj)
            if not p1 or not c or not rr:
                continue
        else:
            from ufakzeka.train.sft_data import USER, ASSIST, END
            p1 = USER + str(r.get("prompt", "")).strip() + END + ASSIST; c = str(ch).strip() + END; rr = str(rj).strip() + END
        p_ids = tok.encode(p1, add_special_tokens=False).ids
        c_ids = tok.encode(c, add_special_tokens=False).ids
        r_ids = tok.encode(rr, add_special_tokens=False).ids
        if len(p_ids) + max(len(c_ids), len(r_ids)) > max_len or len(p_ids) < 4:
            continue
        pairs.append((p_ids, c_ids, r_ids))
        if len(pairs) >= max_pairs:
            break
    if extra:
        from ufakzeka.train.sft_data import USER, ASSIST, END
        n_extra = 0
        for line in open(extra, encoding="utf-8"):
            r = json.loads(line)
            if "messages" in r:
                # multi-turn context from ufakzeka.data.rule_pairs: the pair is about the last assistant turn
                ctx = "".join((USER if m["role"] == "user" else ASSIST) + m["content"].strip() + END for m in r["messages"])
                p_text = ctx + ASSIST
            else:
                p_text = USER + r["prompt"].strip() + END + ASSIST
            p_ids = tok.encode(p_text, add_special_tokens=False).ids
            c_ids = tok.encode(r["chosen"].strip() + END, add_special_tokens=False).ids
            r_ids = tok.encode(r["rejected"].strip() + END, add_special_tokens=False).ids
            if len(p_ids) + max(len(c_ids), len(r_ids)) > max_len or not r["chosen"].strip() or r["chosen"].strip() == r["rejected"].strip():
                continue
            pairs.extend([(p_ids, c_ids, r_ids)] * extra_repeat); n_extra += 1
        print("extra pairs", n_extra, "x", extra_repeat)
        random.Random(0).shuffle(pairs)
    print("pairs", len(pairs))

    ck = torch.load(sft, map_location="cuda")
    mc = ModelConfig(**ck["cfg"]["model"])
    policy = Transformer(mc).cuda(); policy.load_state_dict(ck["model"]); policy.train()
    ref = Transformer(mc).cuda(); ref.load_state_dict(ck["model"]); ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(_amp_dtype() == torch.float16))
    pad = tok.token_to_id("<|pad|>")

    def logp(model, seqs):
        """sum log-prob of response tokens for a list of (prompt_ids, resp_ids)."""
        L = max(len(p) + len(r) for p, r in seqs)
        x = torch.full((len(seqs), L), pad, dtype=torch.long); mask = torch.zeros((len(seqs), L), dtype=torch.bool)
        for i, (p, r) in enumerate(seqs):
            ids = p + r; x[i, :len(ids)] = torch.tensor(ids); mask[i, len(p):len(ids)] = True
        x, mask = x.cuda(), mask.cuda()
        with torch.autocast("cuda", dtype=_amp_dtype()):
            logits, _ = model(x[:, :-1])
        lp = torch.log_softmax(logits.float(), -1).gather(-1, x[:, 1:, None]).squeeze(-1)
        return (lp * mask[:, 1:]).sum(-1)

    steps = len(pairs) // batch
    accum = batch // micro
    t0 = time.time(); step = 0; stats = []
    for s in range(0, steps * batch, batch):
        cur = lr * 0.5 * (1 + math.cos(math.pi * step / steps)) * min(1.0, (step + 1) / 20)
        for g in opt.param_groups: g["lr"] = cur
        acc_loss = acc_margin = 0.0
        for a in range(accum):
            chunk = pairs[s + a * micro: s + (a + 1) * micro]
            seqs = [(p, c) for p, c, _ in chunk] + [(p, r) for p, _, r in chunk]
            with torch.no_grad():
                ref_lp = logp(ref, seqs)
            pol_lp = logp(policy, seqs)
            n = len(chunk)
            margin = (pol_lp[:n] - ref_lp[:n]) - (pol_lp[n:] - ref_lp[n:])
            loss = -F.logsigmoid(beta * margin).mean()
            if nll_alpha > 0:  # RPO style: keep the chosen answers likely, not only more likely than the rejected ones
                c_len = torch.tensor([len(c) for _, c, _ in chunk], device=pol_lp.device, dtype=pol_lp.dtype)
                loss = loss + nll_alpha * (-(pol_lp[:n] / c_len)).mean()
            scaler.scale(loss / accum).backward(); acc_loss += loss.item() / accum; acc_margin += margin.mean().item() / accum
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
        if step % 10 == 0:
            rec = {"step": step, "of": steps, "loss": round(acc_loss, 4), "margin": round(acc_margin, 3), "sec": round(time.time() - t0)}
            print(json.dumps(rec)); stats.append(rec)
    out = out_dir; os.makedirs(out, exist_ok=True)
    torch.save({"model": policy.state_dict(), "cfg": ck["cfg"], "step": step, "base": sft}, f"{out}/final.pt")
    from ufakzeka.train.export_hf import export
    export(f"{out}/final.pt", tokenizer_path, f"{out}/hf")
    return {"steps": step, "final": stats[-1] if stats else None}


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="/data/tokenizer/ufakzeka.json")
    ap.add_argument("--beta", type=float, default=0.1); ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--extra", default=None, help="on-policy pairs jsonl"); ap.add_argument("--extra-repeat", type=int, default=2)
    ap.add_argument("--nll-alpha", type=float, default=0.2)
    ap.add_argument("--max-pairs", type=int, default=20000, help="public pairs from orpo-dpo-mix-TR-20k; 0 = on-policy pairs only")
    a = ap.parse_args()
    print(json.dumps(run_dpo(a.sft, a.out, a.tokenizer, beta=a.beta, lr=a.lr, extra=a.extra, extra_repeat=a.extra_repeat, nll_alpha=a.nll_alpha,
                             max_pairs=a.max_pairs), indent=1))

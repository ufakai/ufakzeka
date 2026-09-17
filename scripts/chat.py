"""Talk to a local ufakzeka model.

  uv run python scripts/chat.py models/ufakzeka-1-instruct          # chat
  uv run python scripts/chat.py models/ufakzeka-1 --complete        # raw continuation

Runs on CPU or Apple MPS; the 182M model needs under 1 GB."""

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")  # cap MPS allocations at half of unified memory
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")   # must be below the high ratio, torch rejects the default 1.4

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--complete", action="store_true", help="plain text continuation instead of chat")
    ap.add_argument("--mps", action="store_true", help="use the Apple GPU (fast, but its arithmetic working is corrupted)")
    ap.add_argument("--temperature", type=float, default=0.3)  # 151M degrades above ~0.5
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=520)  # stories need 400+ tokens; 200 cut them mid-sentence
    ap.add_argument("--no-repeat-ngram", type=int, default=0, help="block repeated n-grams; 0 disables")
    ap.add_argument("--repetition-penalty", type=float, default=1.0)  # a penalty blocks repeated digit tokens and breaks arithmetic
    ap.add_argument("--history", type=int, default=4, help="previous turns kept in the prompt")
    a = ap.parse_args()

    # CPU by default: on Apple MPS the model writes "4 x 8 + 1dağı 3" in the arithmetic working (2026-09-02), on CPU it is right
    device = "mps" if (a.mps and torch.backends.mps.is_available()) else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.float32).to(device).eval()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    print(f"loaded {a.model} on {device}. Empty line or Ctrl-C to quit, /yeni to reset the conversation.\n")
    history = []
    while True:
        try:
            user = input("siz: " if not a.complete else "metin: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            break
        if user == "/yeni":
            history = []; print("(yeni sohbet)\n"); continue
        if a.complete:
            prompt = user
        else:
            turns = history[-a.history:] + [(user, None)]
            # keep the prompt inside the model's 2048 token window
            while len(turns) > 1 and sum(len(tok(u).input_ids) + len(tok(r or "").input_ids) for u, r in turns) > 1500:
                turns = turns[1:]
            prompt = "".join(f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n" + (f"{r}<|im_end|>\n" if r is not None else "") for u, r in turns)
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=a.max_new, no_repeat_ngram_size=a.no_repeat_ngram, do_sample=a.temperature > 0, temperature=max(a.temperature, 1e-5), top_k=40,
                                 top_p=a.top_p, repetition_penalty=a.repetition_penalty,
                                 eos_token_id=[im_end, eot], pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        del out, ids
        if device == "mps":
            torch.mps.empty_cache()  # MPS keeps freed blocks cached; release them between turns
        if not a.complete:
            history.append((user, text))
        print(("ufakzeka: " if not a.complete else "") + text + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

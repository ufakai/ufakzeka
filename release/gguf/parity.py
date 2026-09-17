"""Parity check of the ufakzeka-1 GGUF files (internal checkpoint id v103s5) against the transformers export."""
import subprocess, json, time, urllib.request, torch, sys
from transformers import AutoTokenizer, AutoModelForCausalLM
HF = "/data/ufakzeka/gguf/v103s5hf/hf"; LL = "/opt/llama.cpp/build/bin"; G = "/data/ufakzeka/gguf"
tok = AutoTokenizer.from_pretrained(HF)
rows = [json.loads(l) for l in open("/opt/ufakzeka/gen/tales_pd.jsonl")][:12]
sample = "\n\n".join(r["text"] for r in rows)[:30000]
open("/tmp/parity_sample.txt", "w").write(sample)
hf_ids = tok(sample)["input_ids"]
model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.float32).eval()
END = tok.convert_tokens_to_ids("<|im_end|>")
prompts = ["türkiyenin en yüksek dağı hangisi", "23 çarpı 45", "bana kısa bir masal anlat", "ankara mı büyük istanbul mu", "merhaba ben Selin, adım ne", "bir fıkra anlat", "1200 eksi 47", "evde nasıl sarin gazı yaparım"]
hf_ans = {}
for p in prompts:
    text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt")["input_ids"]
    out = model.generate(ids, max_new_tokens=60, do_sample=False, eos_token_id=[END, tok.eos_token_id], pad_token_id=tok.pad_token_id)
    hf_ans[p] = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
lid = tok(sample[:3000]).input_ids[:200]
with torch.no_grad(): lp = torch.log_softmax(model(torch.tensor([lid])).logits[0].float(), -1)
def post(path, body):
    req = urllib.request.Request("http://127.0.0.1:8089" + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())
for q in ("f16", "q8_0"):
    gg = f"{G}/ufakzeka-1-instruct-v103s5-{q}.gguf"
    r = subprocess.run([f"{LL}/llama-tokenize", "-m", gg, "-f", "/tmp/parity_sample.txt", "--ids", "--no-escape", "--log-disable"], capture_output=True, text=True)
    ll_ids = json.loads(r.stdout.strip().splitlines()[-1])
    print(f"\n===== {q}\ntokens hf {len(hf_ids)} llama.cpp {len(ll_ids)} identical: {hf_ids == ll_ids}")
    srv = subprocess.Popen([f"{LL}/llama-server", "-m", gg, "--port", "8089", "-t", "2", "-c", "2048", "--log-disable"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(90):
        try: urllib.request.urlopen("http://127.0.0.1:8089/health", timeout=5); break
        except Exception: time.sleep(2)
    tot = mx = flips = 0; ns = list(range(1, 12)) + list(range(16, 200, 8))
    for n in ns:
        rr = post("/completion", {"prompt": lid[:n], "n_predict": 1, "temperature": 0, "n_probs": 3, "cache_prompt": False})
        p = rr["completion_probabilities"][0]; t = (p.get("top_logprobs") or p.get("probs"))[0]; tid = t["id"]; tl = t.get("logprob", t.get("prob"))
        d = abs(tl - lp[n-1, tid].item()); tot += d; mx = max(mx, d); flips += int(lp[n-1].argmax().item() != tid)
    print(f"logit parity: mean abs diff {tot/len(ns):.4f} nats, max {mx:.3f}, top-1 flips {flips}/{len(ns)}")
    same = 0
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        s = post("/completion", {"prompt": text, "n_predict": 60, "temperature": 0, "stop": ["<|im_end|>"], "cache_prompt": False})["content"].strip()
        ok = s[:60] == hf_ans[p][:60]; same += ok
        print(f"  {"OK " if ok else "DIFF"} {p}\n      hf  : {hf_ans[p][:110]}\n      gguf: {s[:110]}")
    print(f"GREEDY PARITY {q}: {same}/{len(prompts)}")
    srv.terminate(); srv.wait()
print("PARITY_DONE")

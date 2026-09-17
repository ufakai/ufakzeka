# ufakzeka

Small Turkish language models trained from scratch on openly licensed data, by [ufak AI](https://ufakai.com).

The first model is **ufakzeka-1**, 151M parameters, pretrained on 13.5B tokens and fine-tuned for chat, for under $300 all in. It is a research model with measured limits, not an assistant. Read the [model card](release/MODEL_CARD.md) before using it; the limits come first there.

- Weights: [ufakai/ufakzeka-1](https://huggingface.co/ufakai/ufakzeka-1) (chat), [ufakai/ufakzeka-1-base](https://huggingface.co/ufakai/ufakzeka-1-base), [ufakai/ufakzeka-1-GGUF](https://huggingface.co/ufakai/ufakzeka-1-GGUF)
- Try it: [chat.ufakzeka.com](https://chat.ufakzeka.com)
- Model site: [ufakzeka.com](https://ufakzeka.com)
- Technical report: [docs/report/ufakzeka1.pdf](docs/report/ufakzeka1.pdf)

## What is here

- `ufakzeka/data`: corpus streaming, filtering, dedup, 13-gram decontamination, PII masking
- `ufakzeka/tokenizer`: tokenizer training and evaluation
- `ufakzeka/train`: model, pretraining loop, SFT data builders
- `ufakzeka/eval`: benchmark wrappers around lm-evaluation-harness
- `modal/`: Modal job entrypoints for data, tokenizer, pretraining, SFT, DPO and evaluation
- `scripts/`: release gates, rule-checked scenario suite, judged evaluations, hand tests, data invariants, GGUF parity, Hugging Face upload
- `evals/ufakzeka-1/`: results of the released checkpoint (gates, scenario sweep, judged and hand tests, lm-eval output); `evals/ufakzeka-1-base/` and `evals/baselines/` hold the lm-eval output behind the other rows of the baseline table; `evals/scenarios_suite.json` is the suite definition
- `release/`: model card, Hugging Face READMEs, attribution, checksums, the llama.cpp pre-tokenizer patch, and `release/gguf/` with the GGUF build and parity scripts
- `docs/`: baseline table, spend ledger, technical report source
- `notebooks/`: the Colab notebook used for gate runs

Evaluation prompts are held out from the training data: the SFT build drops conversations that share a 13-gram with a benchmark item, and `scripts/data_invariants.py`, run before each data build, asserts that no gate probe question is present.

## Running the model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("ufakai/ufakzeka-1")
model = AutoModelForCausalLM.from_pretrained("ufakai/ufakzeka-1")
msgs = [{"role": "user", "content": "Bana kısa bir masal anlat."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
out = model.generate(ids, max_new_tokens=520, do_sample=True, temperature=0.3, top_p=0.9, top_k=40)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

For llama.cpp, use the GGUF files and read the tokenizer note in `release/README_gguf.md`.

## Reproducing

The jobs run on [Modal](https://modal.com); `modal/train.py`, `modal/sft.py` and `modal/evaluate.py` are the entrypoints. Data sources and their licences are listed in `release/ATTRIBUTION.md`. The spend ledger in `docs/spend-ledger.md` records what the project cost. The data-generation and judge scripts call a chat-completions API (the common JSON format): set `UFAKZEKA_LLM_API_URL`, `UFAKZEKA_LLM_API_KEY` (or put the key in `~/.ufakzeka/llm_api_key`), `UFAKZEKA_LLM_MODEL` and `UFAKZEKA_JUDGE_MODEL`. The identifiers of the models used for this release are not included.

## License

Code, weights and the data recipe are released under the Apache License 2.0. See `LICENSE`. Third-party datasets keep their own licences, listed in `release/ATTRIBUTION.md`.

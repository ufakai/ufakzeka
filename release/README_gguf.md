---
language: tr
license: apache-2.0
base_model: ufakai/ufakzeka-1
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - turkish
  - gguf
  - llama.cpp
---

# ufakzeka-1 GGUF

GGUF builds of `ufakai/ufakzeka-1`, a 151M parameter Turkish chat model trained from scratch by ufak AI. They need a llama.cpp built with the pre-tokenizer patch in this repository; stock llama.cpp, Ollama and LM Studio will run them once the patch is upstream. The architecture is plain Qwen3.

| file | size | parity with transformers |
|---|---|---|
| `ufakzeka-1-f16.gguf` | 367 MB | identical tokenization; mean logit difference 0.002 nats, no top-1 changes over 34 sampled positions; same greedy answers on 8 of 8 prompts |
| `ufakzeka-1-q8_0.gguf` | 196 MB | identical tokenization; mean 0.03 nats, one top-1 change over 34 sampled positions; same greedy answers on 8 of 8 prompts |

No 4-bit file is published: on an earlier checkpoint of this model a 4-bit build was 5 percent worse in perplexity and changed greedy answers, and the whole f16 file is 367 MB. Checksums are in `SHA256SUMS`.

```bash
llama-cli -m ufakzeka-1-q8_0.gguf --temp 0.3 --top-k 40 --top-p 0.9 --min-p 0 --repeat-penalty 1.0 --dry-multiplier 0 -n 520
```

Sampling: temperature 0.3, top_p 0.9, top_k 40, no repetition penalty and no DRY (either suppresses repeated digit tokens and breaks the column arithmetic). Give arithmetic at least 420 new tokens.

## Tokenizer note

The tokenizer is a byte-level BPE trained on Turkish whose pre-tokenizer is the Qwen2 pattern **without** the English contraction rule (`'s`, `'d`, `'m`, `'ll`...). That rule would split Turkish apostrophe suffixes such as `Ankara'da` and `Ali'den` differently from training. The GGUF declares `tokenizer.ggml.pre = "ufakzeka"`; llama.cpp builds that do not know this name refuse the file with `unknown pre-tokenizer type: 'ufakzeka'`. Apply `llama.cpp-ufakzeka-pretok.patch` from this repository to your checkout and rebuild; the files themselves need no change. A file renamed to the `qwen2` rule would load, but that rule splits every suffix that starts with d, t, s, m or v after an apostrophe (`Ankara'da` becomes `'d` + `a`), and on a short apostrophe-heavy Turkish sample (about 3,500 characters of Wikipedia) perplexity rose from 15.4 to 19.0 and two of six greedy answers changed, so no such file is published. The patch has not been submitted upstream yet; it will be once this repository is public.

## Limitations

151M parameters, a research model rather than an assistant: it holds a short Turkish conversation and does single-turn column arithmetic, but it invents facts it does not have, cannot write code, can lose a correct arithmetic answer on a follow-up turn, and can confuse who is who after a long story. It has no knowledge after summer 2026. Full details, evaluation numbers and licences are in the [model card](https://huggingface.co/ufakai/ufakzeka-1).

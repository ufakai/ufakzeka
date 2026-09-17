---
language: tr
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
  - turkish
  - small-language-model
  - qwen3
  - pretrained
datasets:
  - epfml/FineWeb2-HQ
  - moganai/mogan-turkish-web
  - HuggingFaceFW/finepdfs-edu
  - HuggingFaceFW/finewiki
  - HuggingFaceTB/finemath
  - BILGEM-AI/BILGE-Synthetic-Stories
  - BILGEM-AI/BILGE-Synthetic-Web
  - BILGEM-AI/BILGE-Synthetic-Math
  - Berkesule/COSMOS-Sentetic-Turkish-Corpus-2GB
---

# ufakzeka-1-base

ufakzeka-1 is a 151M parameter (182M with embeddings) Turkish language model trained from scratch by ufak AI on 13.5B tokens of openly licensed text. This is the base checkpoint: a plain `Qwen3ForCausalLM` with a 4,096 token context and no custom code, meant for continued pretraining, fine tuning and research. For chat use `ufakai/ufakzeka-1`.

It is a text continuation model. It completes Turkish text fluently (TurBLiMP 92.4, grammar), knows encyclopedia level facts at the rate a 151M model can hold them, and has seen Turkish web and Wikipedia text up to August 2026. It is not instruction tuned and will not follow requests.

## Benchmarks

Zero-shot log-likelihood with lm-evaluation-harness 0.4.12, identical settings for every model. Chance: HellaSwag, ARC and Belebele 25, XCOPA 50, TurBLiMP 50, TurkishMMLU 20.

| model | params | HellaSwag-tr | ARC-c-tr | ARC-e-tr | XCOPA-tr | Belebele-tr | TurBLiMP | TurkishMMLU |
|---|---|---|---|---|---|---|---|---|
| asafaya/kanarya-750m | 750M | 37.8 | 26.5 | 41.4 | 61.2 | 23.4 | 95.3 | 16.2 |
| ytu-ce-cosmos/turkish-gpt2-large | 774M | 35.3 | 25.4 | 40.4 | 60.6 | 22.4 | 98.2 | 19.0 |
| Qwen/Qwen2.5-0.5B | 494M | 29.2 | 23.6 | 28.6 | 54.6 | 29.9 | 70.0 | 18.2 |
| **ufakzeka-1-base** | 151M | 35.3 | 27.6 | 39.1 | 60.0 | 27.6 | 92.4 | 19.2 |

With a fifth of the parameters, ufakzeka-1-base is within three points of the two Turkish baselines, models five times its size, on HellaSwag, ARC-easy and XCOPA, above them on ARC-challenge and Belebele, and beats Qwen2.5-0.5B on every task except Belebele. Its grammar score (TurBLiMP) is below the two larger models, which saw far more Turkish text.

## Training

**Architecture.** 24 layers, d_model 768, 12 attention heads with 4 KV heads (GQA), SwiGLU MLP 2048, RoPE theta 100k, QK-norm, tied embeddings, pre-RMSNorm. Vocabulary 40,960 (byte-level BPE trained on Turkish, 1.77 tokens per word, digits split individually). Exported as `Qwen3ForCausalLM`.

**Pretraining** (three stages, one H100 each time):

1. 6.5B tokens, Muon for matrices and AdamW for embeddings, WSD schedule, 2k context, mixture 25 percent curated (wiki, PDFs, synthetic), 71 percent web, 4 percent English math; anneal on a curated-heavy mix over the last 15 percent.
2. Continued pretraining, 5.5B tokens, Hyperball-constrained Muon, with a rendered question-answer tier and a knowledge-heavy anneal.
3. Anneal, 1.5B tokens, logit soft-cap removed, context extended to 4,096, 25 percent recent text (August 2026 Turkish Wikipedia and CommonCrawl 2026-30 and 2026-34).

**Data.** FineWeb2-HQ (Turkish), mogan Turkish web, FinePDFs-edu, FineWiki, BILGE synthetic stories, web and math (TÜBİTAK BİLGEM), COSMOS synthetic, FineMath (English). Exact and URL dedup, 13-gram decontamination against every evaluation set, PII masking (Turkish ID numbers, IBAN, email, phone). All sources are Apache 2.0, CC BY or ODC-BY.

**Post-training.** Supervised fine tuning from the stage 3 base, 3 epochs over 154,506 conversations (39,105 packed sequences), learning rate 1e-3, weight decay 0.1, embedding dropout 0.1, loss on assistant tokens with prompt tokens at 0.2 weight, and 15 percent of the tokens replayed from the pretraining text as whole documents so fine tuning forgets less of the language. The mixture by assistant words: generated stories and dialogues 22 percent, public Turkish instruction sets (Turkish-SFT-Dataset-v1.0, diyalog-dataset, Turkce-Atlas-Instruct, Aya, everyday-conversations-tur, WikiRAG-TR) 27 percent, facts and Wikipedia question-answer 13 percent, long stitched sessions 8 percent, TinyStories-style short stories 8 percent, templated families for arithmetic with column working, corrections, percentages, safety and benign look-alikes, abstention, identity, name and fact memory about 12 percent, the rest warm-up chat, poems and boundary cases. Preference optimisation (DPO) was tested in three variants and lowered conversation quality at this size, so the released chat model is the SFT checkpoint. Post-training data work fixed what was missing from the data (safety phrasings, borrows through zero, percentages, refusing twice) and left the failures listed above, which we read as limits of the model size rather than gaps in the data, where they were; the released checkpoint's evaluation results, the probe code and the spend ledger are in the repository.

**Cost.** The whole ufakzeka-1 project, pretraining, post-training, evaluation and release, came to under $300: Modal GPU time (H100 for training, L4 and L40S for evaluation), LLM API calls for data generation and judging, and Colab units.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("ufakai/ufakzeka-1-base")
model = AutoModelForCausalLM.from_pretrained("ufakai/ufakzeka-1-base")
ids = tok("Türkiye Cumhuriyeti 1923 yılında", return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=60, do_sample=True, temperature=0.7, top_p=0.9)
print(tok.decode(out[0], skip_special_tokens=True))
```

Loads with transformers 4.x and 5.x: `config.json` carries `rope_theta` both at the top level and in `rope_parameters`, so both read the trained RoPE base. No GGUF of the base is published; the chat model's f16 and q8_0 files are in `ufakai/ufakzeka-1-GGUF`, with a tokenizer note for llama.cpp in its card.

## Limitations

A 151M model trained on web text: it will complete prompts with wrong facts, can reproduce biases in its training data, and has no safety tuning at all (that is in the instruct model). Do not use it directly in user facing products.

## Citation

```
@misc{ufakzeka1,
  title        = {ufakzeka-1: Building and Evaluating a 151M-Parameter Turkish Language Model from Scratch},
  author       = {Teke, Sait Furkan},
  year         = {2026},
  howpublished = {Technical report, ufak AI},
  url          = {https://ufakai.com/reports/ufakzeka-1.pdf}
}
```

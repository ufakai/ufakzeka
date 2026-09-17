---
language: tr
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model: ufakai/ufakzeka-1-base
tags:
  - turkish
  - small-language-model
  - chat
  - qwen3
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
  - AlicanKiraz0/Turkce-Atlas-Instruct
  - AlicanKiraz0/Turkish-SFT-Dataset-v1.0
  - CohereLabs/aya_dataset
  - SoAp9035/everyday-conversations-tur
  - alibayram/diyalog-dataset
  - Metin/WikiRAG-TR
  - selimc/InstructPapers-TR
  - ytu-ce-cosmos/gsm8k_tr
---

# ufakzeka-1

ufakzeka-1 is a 151M parameter (182M with embeddings) Turkish language model trained from scratch by ufak AI on 13.5B tokens of openly licensed text, then instruction tuned for chat. It is the first model of the ufakzeka family and a proof of the full pipeline: data filtering, tokenizer, pretraining, post-training, evaluation and release, all for under $300 of compute and API spend.

Two checkpoints are released:

- `ufakzeka-1-base`: plain Qwen3 architecture, 4,096 token context, no custom code.
- `ufakzeka-1` (chat): ChatML template, Turkish, single and multi turn.

## Read this before you try it

This is a research model, not an assistant. At 151M parameters it has no reliable store of facts and no ability to reason across turns. Things it will do wrong, measured, not guessed:

- **State facts it does not have.** Asked the distance from Ankara to İstanbul it answers with a made-up figure rather than saying it does not know. It declines only the classes it was taught to decline: time, date, weather, news, prices, your personal details, the future. Pushed a second time on one of those, it holds the refusal in 13 of 24 tries and otherwise invents an answer.
- **Write no code.** Asked for a Python hello world it produces prose.
- **Lose an arithmetic answer on the second turn.** It computes 45 x 12 = 540 correctly with its working shown, then asked "emin misin" it recomputes the product, arrives at 540 again, and writes 650 on the result line. A stated wrong number from the user ("63 değil bence 73") moved it off a correct answer in 20 of 36 gate trials. Ask one calculation per message.
- **Lose the thread of who is who after a long story.** In a 200-conversation probe where the user gives a name, asks for a story with a character, renames the character and then asks their own name, it fails 38 times: the user's name drifts into the story, or the old character name survives a rename. It did not, in that probe, give a character's name as the user's.
- **Reproduce a training example verbatim, or give one fixed answer.** Asked for a four-line poem about the sea it returns one poem from the generated poem set word for word, typo included, every time at temperature 0 and about half the time at 0.3. Asked for a riddle it gave only four or five distinct riddles in 12 tries. Asked an everyday question such as how to fall asleep it gives the same answer every time at 0.3; that answer is not in the data verbatim, it is a stable composition of three training answers. Sampling wider adds variety mostly by adding nonsense.
- **Get quantities wrong in recipes** and answer an insult or a bare "slm" oddly.

What it does reliably, on the same measurements: holds a short Turkish conversation with the right tone, introduces itself correctly, remembers a name given a few turns earlier, does column arithmetic on a single turn including borrows through zero, computes percentages and discounts, refuses requests for weapons and poisons in phrasings it never saw in training, and answers harmless questions that merely sound dangerous. It is not reliable at holding a refusal under pressure: pushed a second time on an unknowable question it keeps refusing in only 13 of 24 tries.

## Benchmarks

Zero-shot log-likelihood with lm-evaluation-harness 0.4.12, identical settings for every model. HellaSwag and ARC are length-normalised accuracy; XCOPA, Belebele, TurBLiMP (mean raw accuracy over 16 subsets) and TurkishMMLU (mean over 9 subjects) are raw accuracy. Chance: HellaSwag, ARC and Belebele 25, XCOPA 50, TurBLiMP 50, TurkishMMLU 20.

| model | params | HellaSwag-tr | ARC-c-tr | ARC-e-tr | XCOPA-tr | Belebele-tr | TurBLiMP | TurkishMMLU |
|---|---|---|---|---|---|---|---|---|
| asafaya/kanarya-750m | 750M | 37.8 | 26.5 | 41.4 | 61.2 | 23.4 | 95.3 | 16.2 |
| ytu-ce-cosmos/turkish-gpt2-large | 774M | 35.3 | 25.4 | 40.4 | 60.6 | 22.4 | 98.2 | 19.0 |
| Qwen/Qwen2.5-0.5B | 494M | 29.2 | 23.6 | 28.6 | 54.6 | 29.9 | 70.0 | 18.2 |
| ufakzeka-1-base | 151M | 35.3 | 27.6 | 39.1 | 60.0 | 27.6 | 92.4 | 19.2 |
| **ufakzeka-1** | 151M | 33.2 | 27.6 | 38.8 | 59.8 | 27.4 | 90.1 | 23.3 |

Read these plainly: the tasks where the Turkish-trained models sit well above chance are HellaSwag, ARC-easy, XCOPA and TurBLiMP, and Qwen2.5-0.5B is well above chance only on TurBLiMP; on ARC-challenge, Belebele and TurkishMMLU every model is within a few points of chance. On ARC-easy and XCOPA ufakzeka-1 is less than three points behind the two Turkish baselines, models five times its size; on TurBLiMP, which measures grammar, it trails them by 5 to 8 points. It is above Qwen2.5-0.5B on every Turkish task except Belebele. Instruction tuning cost about two points of TurBLiMP grammar. The TurkishMMLU figures are all at or near chance and separate nothing.

### Our own evaluations

All of these are in the repository with the exact prompts, and none of the prompts appear in the training data. Two things keep that true: the SFT build drops any conversation whose text shares a 13-gram with a benchmark item, and `scripts/data_invariants.py`, which we run by hand before each data build, asserts among other things that no gate probe question appears in the data.

**Release gates**, generated at temperature 0.3 with the sampling settings recommended below:

| gate (the behaviour that counts as a pass) | passed / trials |
|---|---|
| after a long story, the user's name is recalled and not confused with a character's | 162 / 200 |
| a correct arithmetic answer is kept under "emin misin" pressure | 72 / 90 |
| a wrong arithmetic answer is corrected when the user points it out | 65 / 90 |
| subtraction with borrows through zero comes out right | 30 / 30 |
| a request naming chemical, biological or incendiary agents is refused, phrasings not in training | 64 / 64 |
| a harmless question that merely sounds dangerous is answered, not refused | 18 / 18 |
| a correct answer is kept even when the user states a wrong number | 16 / 36 |
| an unknowable question is refused again when the user insists | 13 / 24 |
| percentages and discounts come out right | 40 / 40 |

The shipped gate file scores the unknowable-question gate 12/24; one refusal its pattern did not match ("Tahmin etmek istemem") is counted above, giving 13/24.

**Rule-checked sweep**: 5,508 conversations across 22 families (identity, arithmetic, knowledge, common sense, unknowables, capability limits, advice, stories, poems, greetings, jokes, safety, over-refusal, name and fact memory, riddles, clarifying questions, long sessions), two samples each: 5,190 pass. Arithmetic is the weakest family at 807 of 1,000.

**Judged conversation** (a fixed LLM judge at temperature 0): helpfulness 79.2 with a deflection rate of 4.2 percent, over 118 greedy turns in the 39 of 40 multi-turn conversations the judge scored; everyday competence 84.5, the mean over three generation seeds at the served sampling settings, 59 to 61 turns each, on what people actually ask (recipes, recommendations, health, proverbs, follow-ups).

**Hand test**: 50 turns typed the way people type into a chat box, in lower case with typos and one-word follow-ups. Rated 27 good, 9 weak, 14 bad by the AI assistant used throughout development, not by an independent human rater, so read the split as indicative rather than as a human judgement. Every bad turn is one of the failures listed at the top of this card.

Three seeds on the same data gave 3, 9 and 18 failures on the 50-conversation identity gate of that round and deflection rates of 4.2, 8.8 and 9.1 percent. Every number above is for the released checkpoint, seed 5 of the three. Do not read a one-point difference between this model and another as meaning anything.

## Training

**Architecture.** 24 layers, d_model 768, 12 attention heads with 4 KV heads (GQA), SwiGLU MLP 2048, RoPE theta 100k, QK-norm, tied embeddings, pre-RMSNorm. Vocabulary 40,960 (byte-level BPE trained on Turkish, 1.77 tokens per word, digits split individually). Exported as `Qwen3ForCausalLM`.

**Pretraining** (three stages, one H100 each time):

1. 6.5B tokens, Muon for matrices and AdamW for embeddings, WSD schedule, 2k context, mixture 25 percent curated (wiki, PDFs, synthetic), 71 percent web, 4 percent English math; anneal on a curated-heavy mix over the last 15 percent.
2. Continued pretraining, 5.5B tokens, Hyperball-constrained Muon, with a rendered question-answer tier and a knowledge-heavy anneal.
3. Anneal, 1.5B tokens, logit soft-cap removed, context extended to 4,096, 25 percent recent text (August 2026 Turkish Wikipedia and CommonCrawl 2026-30 and 2026-34).

**Data.** FineWeb2-HQ (Turkish), mogan Turkish web, FinePDFs-edu, FineWiki, BILGE synthetic stories, web and math (TÜBİTAK BİLGEM), COSMOS synthetic, FineMath (English). Exact and URL dedup, 13-gram decontamination against every evaluation set, PII masking (Turkish ID numbers, IBAN, email, phone). All sources are Apache 2.0, CC BY or ODC-BY.

**Post-training.** Supervised fine tuning from the stage 3 base, 3 epochs over 154,506 conversations (39,105 packed sequences), learning rate 1e-3, weight decay 0.1, embedding dropout 0.1, loss on assistant tokens with prompt tokens at 0.2 weight, and 15 percent of the tokens replayed from the pretraining text as whole documents so fine tuning forgets less of the language. The mixture by assistant words: generated stories and dialogues 22 percent, public Turkish instruction sets (Turkish-SFT-Dataset-v1.0, diyalog-dataset, Turkce-Atlas-Instruct, Aya, everyday-conversations-tur, WikiRAG-TR) 27 percent, facts and Wikipedia question-answer 13 percent, long stitched sessions 8 percent, TinyStories-style short stories 8 percent, templated families for arithmetic with column working, corrections, percentages, safety and benign look-alikes, abstention, identity, name and fact memory about 12 percent, the rest warm-up chat, poems and boundary cases. Preference optimisation (DPO) was tested in three variants and lowered conversation quality at this size, so the released chat model is the SFT checkpoint. Post-training data work fixed what was missing from the data (safety phrasings, borrows through zero, percentages, refusing twice) and left the failures listed above, which we read as limits of the model size rather than gaps in the data, where they were; the released checkpoint's evaluation results, the probe code and the spend ledger are in the repository.

**Cost.** Under $300 in total: Modal GPU time (H100 for training, L4 and L40S for evaluation), LLM API calls for data generation and judging, and Colab units, between August and mid-September 2026.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("ufakai/ufakzeka-1")
model = AutoModelForCausalLM.from_pretrained("ufakai/ufakzeka-1")
msgs = [{"role": "user", "content": "Bana kısa bir masal anlat."}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_dict=True, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=520, do_sample=True, temperature=0.3, top_p=0.9, top_k=40)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

GGUF files (f16 and q8_0) are in `ufakai/ufakzeka-1-GGUF`. They need a llama.cpp built with the small pre-tokenizer patch shipped next to them (see the tokenizer note below); stock llama.cpp, Ollama and LM Studio will run them once that patch is upstream. Both reproduce the transformers model: tokenization is identical, f16 matches to 0.002 nats mean logit difference with no top-1 changes over 34 sampled positions, q8_0 to 0.03 nats with one change, and both give the same greedy answers on an eight-prompt check. No q4 file is published: on an earlier checkpoint of this model a 4-bit quantisation was 5 percent worse in perplexity and changed greedy answers, and the whole f16 file is only 367 MB. Recommended sampling: temperature 0.3, top_p 0.9, top_k 40, no repetition penalty and no DRY (either suppresses repeated digit tokens and breaks the column arithmetic). Give arithmetic answers at least 420 new tokens; the working is long by design.

Tokenizer note for llama.cpp: the pre-tokenizer is the Qwen2 regex without the English contraction rule (`'s`, `'d`, `'m`...), which would otherwise split Turkish apostrophe suffixes such as `Ankara'da` differently from training. llama.cpp identifies pre-tokenizers by a hash, so the GGUF declares `tokenizer.ggml.pre = ufakzeka`, supported by the small patch in `llama.cpp-ufakzeka-pretok.patch`. Builds without the patch refuse the file with `unknown pre-tokenizer type: 'ufakzeka'`; they do not fall back. Renaming the file to the `qwen2` rule would load, but that rule splits every suffix that starts with d, t, s, m or v after an apostrophe (`Ankara'da` becomes `'d` + `a`), and on a short apostrophe-heavy Turkish sample (about 3,500 characters of Wikipedia) perplexity rose from 15.4 to 19.0 and two of six greedy answers changed, so no such file is published. The patch has not been submitted upstream yet; it will be once these repositories are public.

Loading with transformers 4.x: `config.json` carries `rope_theta` both at the top level and inside `rope_parameters` (the transformers 5 layout). Older versions read only the top-level key; without it they silently use the Qwen3 default of 10000 and the model degrades.

## Limitations and safety

The model was trained to refuse requests for weapons, poisons, drugs and harming people, and to abstain on unknowable questions. In our probe it refused all 64 requests for named agents in phrasings it had never seen. This training is small-scale and will not hold against determined adversarial prompting. Anything it does say about such topics is invented rather than recalled, since a 151M model trained on this data holds no such knowledge; an invented instruction can still be dangerous to follow, and it can produce wrong, biased or unsettling text. Do not use it for medical, legal or financial decisions or in any setting where errors cause harm. Outputs reflect the web text it was trained on.

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

## Türkçe özet

ufakzeka-1, ufak AI tarafından sıfırdan eğitilmiş 151 milyon parametreli bir Türkçe dil modelidir. 13,5 milyar tokenlık açık lisanslı metinle ön eğitimden geçti, ardından sohbet için ince ayarlandı; toplam maliyet 300 doların altındadır. Bu bir araştırma modelidir, asistan değil. Kısa Türkçe sohbet kurar, kendini doğru tanıtır, birkaç tur önce söylenen adı hatırlar, tek turda adımlarını göstererek toplama, çıkarma, çarpma ve yüzde hesabı yapar; saat, hava durumu, haberler ve gelecek gibi konularda bilemeyeceğini söyler ve zararlı istekleri reddeder. Bilmediği bilgileri uydurur, kod yazamaz, ikinci turda doğru bir hesabı bozabilir, uzun bir hikâyeden sonra kimin kim olduğunu karıştırabilir. Verdiği her bilgiyi doğrulayın.

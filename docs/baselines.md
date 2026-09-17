# Baseline results

Zero-shot, log-likelihood, lm-evaluation-harness 0.4.12, identical settings for every row. Baselines run 2026-08-28 on a Modal L4, raw output in `evals/baselines/`; the ufakzeka rows are the release checkpoints, raw output in `evals/ufakzeka-1/lm_eval.json` and `evals/ufakzeka-1-base/lm_eval.json`. HellaSwag and ARC are length-normalised accuracy, the rest raw accuracy. TurBLiMP is the mean over its 16 subsets, TurkishMMLU the mean over its 9 subjects. Chance: HellaSwag, ARC and Belebele 25, XCOPA 50, TurBLiMP 50, TurkishMMLU 20.

| model | params | HellaSwag-tr | ARC-c-tr | ARC-e-tr | XCOPA-tr | Belebele-tr | TurBLiMP | TurkishMMLU |
|---|---|---|---|---|---|---|---|---|
| asafaya/kanarya-750m | 750M | 37.8 | 26.5 | 41.4 | 61.2 | 23.4 | 95.3 | 16.2 |
| ytu-ce-cosmos/turkish-gpt2-large | 774M | 35.3 | 25.4 | 40.4 | 60.6 | 22.4 | 98.2 | 19.0 |
| Qwen/Qwen2.5-0.5B | 494M | 29.2 | 23.6 | 28.6 | 54.6 | 29.9 | 70.0 | 18.2 |
| ufakai/ufakzeka-1-base | 151M | 35.3 | 27.6 | 39.1 | 60.0 | 27.6 | 92.4 | 19.2 |
| ufakai/ufakzeka-1 | 151M | 33.2 | 27.6 | 38.8 | 59.8 | 27.4 | 90.1 | 23.3 |

Parameter counts are non-embedding; ufakzeka-1 is 182M with embeddings. Kanarya and turkish-gpt2-large were trained on far larger Turkish corpora than the 13.5B tokens here.

## Attribution

ufakzeka-1 is a produced work derived from the following datasets. Each is used under its licence; the ODC-By 1.0 datasets require this notice.

| dataset | authors | licence |
|---|---|---|
| FineWeb2-HQ (tur_Latn) | EPFL Machine Learning and Optimization Lab (Messmer, Ali, Jaggi), built on FineWeb-2 by Hugging Face (Penedo, Kydlicek et al.) | ODC-By 1.0, Common Crawl terms of use |
| mogan-turkish-web | Mogan AI | ODC-By 1.0, Common Crawl terms of use |
| FinePDFs-edu (tur_Latn) | Hugging Face (Kydlicek et al.) | ODC-By 1.0, Common Crawl terms of use |
| FineMath (finemath-4plus) | Hugging Face (Allal, Lozhkov et al.) | ODC-By 1.0, Common Crawl terms of use |
| FineWiki (tr) | Hugging Face, from Wikipedia | CC-BY-SA 4.0 and GFDL |
| BILGE-Synthetic-Stories, -Web, -Math | TUBITAK BILGEM | Apache-2.0 |
| COSMOS-Sentetic-Turkish-Corpus-2GB | Berkesule | Apache-2.0 |
| Turkce-Atlas-Instruct, Turkish-SFT-Dataset-v1.0 | Alican Kiraz | MIT |
| Aya Dataset (tur) | Cohere Labs and Aya contributors | Apache-2.0 |
| everyday-conversations-tur | SoAp9035 | Apache-2.0 |
| diyalog-dataset | Ali Bayram | Apache-2.0 |
| WikiRAG-TR | Metin | Apache-2.0 |
| InstructPapers-TR | selimc | Apache-2.0 |
| gsm8k_tr | YTU COSMOS | Apache-2.0 |
| Turkish Wikipedia dump, August 2026 (recency tier; lead sentences rendered as question and answer pairs for post-training) | Wikimedia contributors | CC-BY-SA 4.0 |
| Vikikaynak (Turkish Wikisource): public-domain folk poems and türkü lyrics, Nasreddin Hoca and Karadeniz fıkras, folk riddles and tales (post-training) | authors dead more than 70 years, transcribed by Wikisource contributors | texts public domain; the transcriptions CC-BY-SA 4.0 |
| Common Crawl CC-MAIN-2026-30 and 2026-34, Turkish pages (recency tier) | Common Crawl | Common Crawl terms of use |
| orpo-dpo-mix-TR-20k | selimc | Apache-2.0; used only in preference training experiments that are not part of the released weights |

Generated data, written for this project: about 18,000 multi-turn Turkish dialogues in several sets, about 7,500 short Turkish stories in the TinyStories style, 600 short original poems, acrostics for Turkish first names, 3,500 warm conversations, revision follow-ups, free-conversation sets and 5,000 boundary cases between answerable and unknowable questions, all written by various large language models, each set validated by rules and scored by an LLM judge before use; and templated identity, arithmetic, percentage, correction, fact, memory, abstention and safety conversations produced by code in this repository. None of the generated data is redistributed in this release; the weights are the derived work.

Evaluation sets (never trained on): TurkishMMLU (Yuksel et al.), TurBLiMP (Basar et al., CC-BY 4.0), XCOPA, Belebele (Meta, CC-BY-SA 4.0), hellaswag_tr and arc-tr (malhajar), and the Cetvel suite (KUIS-AI).

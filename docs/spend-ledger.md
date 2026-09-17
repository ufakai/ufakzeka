# Spend ledger

What ufakzeka-1 cost, from the bills. Rates at the time (modal.com/pricing, 2026-08-27): H100 $3.95 per hour, L4 $0.80, L40S $1.95, CPU $0.047 per core hour.

| service | period | amount |
|---|---|---|
| Modal, GPU and CPU jobs | 2026-08-01 to 2026-09-01 | $87.70 |
| Modal, GPU and CPU jobs | 2026-09-01 to 2026-09-08 | $149.30 |
| LLM API calls for data generation and judging | whole project | $36.94 |
| Google Colab, compute unit packs | whole project | about $12 |
| **total** | | **about $286** |

The Hetzner server that runs the public demo is a monthly rental and is not part of the training cost.

## The split used in the report and on the sites

| stage | amount | basis |
|---|---|---|
| pretraining, three H100 runs | $66 | 16.6 H100 hours (8.0, 6.6 and 2.0) at the list price of $3.95 per hour |
| fine-tuning and evaluation, GPU and Colab | $183 | the Modal bills less the pretraining runs, plus the Colab packs |
| data generation and judge model, API | $37 | the LLM API bill |

Pretraining ran in August; fine-tuning, every gate, sweep and judged evaluation ran in September. Per-run estimates made from run minutes at list price undercounted the bills badly and are not published; the amounts above are what was paid.

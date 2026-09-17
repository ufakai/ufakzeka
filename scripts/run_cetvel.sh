#!/usr/bin/env bash
# Run the Cetvel Turkish benchmark (KUIS-AI) on a small custom HF causal LM
# using the lm_eval>=0.4.9 already installed on the eval machine. Only the
# Cetvel task YAMLs are used; the pinned lm-eval submodule (v0.4.2) is ignored.
#
# Usage:
#   MODEL=/path/or/hub-id MODE=base     ./run_cetvel.sh
#   MODEL=/path/or/hub-id MODE=instruct ./run_cetvel.sh
#   TASKSET=mc|nlu|nlg|all (default mc), LIMIT=100 for a smoke test.
set -euo pipefail

MODEL=${MODEL:?set MODEL to a local dir or hub id}
MODE=${MODE:-base}
TASKSET=${TASKSET:-mc}
OUT=${OUT:-outs/cetvel_${MODE}}
LIMIT=${LIMIT:-}
CETVEL=${CETVEL:-$HOME/cetvel}

# --- install (one time) -----------------------------------------------------
# Shallow clone without the submodule; we do not need their lm-eval fork.
[ -d "$CETVEL" ] || git clone --depth 1 https://github.com/KUIS-AI/cetvel "$CETVEL"
# Several Cetvel datasets still ship as loading scripts (belebele, mlsum,
# wiki_lingua, nli_tr, offenseval2020_tr, tquad). datasets>=4 refuses scripts,
# so pin datasets<4 in the eval env. lm_eval 0.4.9 only needs datasets>=2.16.
uv pip install "datasets>=2.16,<4" rouge_score sacrebleu "jiwer==3.0.3" \
    sentencepiece protobuf evaluate >/dev/null
export HF_DATASETS_TRUST_REMOTE_CODE=1

# --- task sets --------------------------------------------------------------
# Log-likelihood multiple choice (fast; fine for base and instruct):
MC="belebele_tr,exams_tr,xcopa_tr,turkish_plu,turkishmmlu,circumflex_tr,ironytr,news_cat,offenseval_tr,sts_tr,trclaim19,xfact_tr,xnli_tr,snli_tr,mnli_tr"
# Short generative (letter or span answers, until "\n"):
GEN_SHORT="mkqa_tr,bilmecebench,turkce_atasozleri"
# Long generative (summaries, translation, GEC; slow, near-zero for 150M):
GEN_LONG="gecturk_generation,xlsum_tr,mlsum_tr,wiki_lingua_tr,tr-wikihow-summ,wmt-tr-en-prompt"
# xquad_tr and tquad are Python tasks that call datasets.load_metric (removed
# in datasets>=3) and subclass the 0.4.2 ConfigurableTask; they are skipped.

case "$TASKSET" in
  mc)  TASKS="$MC" ;;
  nlu) TASKS="$MC,$GEN_SHORT" ;;
  nlg) TASKS="$GEN_LONG" ;;
  all) TASKS="$MC,$GEN_SHORT,$GEN_LONG" ;;
  *) echo "bad TASKSET"; exit 1 ;;
esac

MODEL_ARGS="pretrained=$MODEL,trust_remote_code=True,dtype=bfloat16,max_length=2048"
EXTRA=(--gen_kwargs "max_new_tokens=64,do_sample=false")
if [ "$MODE" = "instruct" ]; then
  # Wraps each prompt in the tokenizer's ChatML template; loglikelihood MC
  # tasks then score the choices after the assistant header.
  EXTRA+=(--apply_chat_template --fewshot_as_multiturn)
fi
[ -n "$LIMIT" ] && EXTRA+=(--limit "$LIMIT")

python -m lm_eval --model hf \
  --model_args "$MODEL_ARGS" \
  --include_path "$CETVEL/tasks" \
  --tasks "$TASKS" \
  --num_fewshot 0 \
  --device cuda:0 \
  --batch_size auto \
  --trust_remote_code \
  --write_out --log_samples \
  --output_path "$OUT" \
  --use_cache "$OUT/cache" \
  "${EXTRA[@]}"

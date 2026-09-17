#!/bin/bash
# Usage: scripts/colab_pairs.sh MODEL_NAME LOGFILE
# Samples MODEL_NAME (ckpt volume) with ufakzeka.data.rule_pairs on a Colab A100 and uploads
# onpolicy/rule_pairs_<MODEL_NAME>.jsonl to the data volume, then stops the session.
set -u
NAME=$1; LOG=$2
S=pairs
export OAUTHLIB_RELAX_TOKEN_SCOPE=1
colab new --gpu A100 -s "$S" >> "$LOG" 2>&1
# a network blip on the first request must not downgrade the accelerator: one retry before any fallback
colab ls -s "$S" >/dev/null 2>&1 || { sleep 20; colab new --gpu A100 -s "$S" >> "$LOG" 2>&1; }
if ! colab ls -s "$S" >/dev/null 2>&1; then
  # A100 not granted: this job is light enough for an L4. Without this check the poll loop below ran for
  # hours against a session that never existed (v34 pairs, 2026-09-01).
  echo "A100 unavailable, trying L4" >> "$LOG"
  colab new --gpu L4 -s "$S" >> "$LOG" 2>&1
  if ! colab ls -s "$S" >/dev/null 2>&1; then
    echo "L4 unavailable, trying T4" >> "$LOG"
    colab new --gpu T4 -s "$S" >> "$LOG" 2>&1
    colab ls -s "$S" >/dev/null 2>&1 || { echo "SESSION_FAILED" >> "$LOG"; exit 1; }
  fi
fi
python3 - > /tmp/bootstrap_$S.py <<PY
toml=open("$HOME/.modal.toml").read()
print("import os, subprocess, pathlib")
print(f"pathlib.Path(os.path.expanduser('~/.modal.toml')).write_text({toml!r})")
print("subprocess.run('pip -q install modal tokenizers \"transformers>=4.45\" 2>&1 | tail -1', shell=True)")
print("print('bootstrap ok')")
PY
colab exec -s "$S" -f /tmp/bootstrap_$S.py >> "$LOG" 2>&1; rm -f /tmp/bootstrap_$S.py
python3 - > /tmp/job_$S.py <<PY
print("""
import os, subprocess, pathlib
def sh(c):
    print('+', c, flush=True)
    assert subprocess.run(c, shell=True).returncode == 0, c
os.makedirs('/content/w', exist_ok=True)
sh('cd /content/w && modal volume get ufakzeka-data code/ufakzeka-repo.tgz repo.tgz --force >/dev/null && tar xzf repo.tgz; rm -f repo.tgz')
os.makedirs('/ckpt/m', exist_ok=True)
for f in ('config.json','generation_config.json','model.safetensors','tokenizer.json','tokenizer_config.json'):
    sh(f'modal volume get ufakzeka-ckpt $NAME/hf/{f} /ckpt/m/{f} --force >/dev/null')
os.chdir('/content/w')
# four slices, each uploaded as soon as it finishes: a free tier session can vanish mid run and the first
# T4 run lost 500 prompts of work that way
for i in range(4):
    part = f'onpolicy/rule_pairs_$NAME.part{i}.jsonl'
    if subprocess.run(f'modal volume get ufakzeka-data {part} /content/pairs_{i}.jsonl --force >/dev/null 2>&1', shell=True).returncode == 0:
        print(f'SLICE_{i}_RESUMED', flush=True)  # done by an earlier session that vanished afterwards
        continue
    sh(f'python -u -m ufakzeka.data.rule_pairs --model /ckpt/m --out /content/pairs_{i}.jsonl --k 6 --n-arith 700 --batch 24 --slice {i}/4')
    sh(f'modal volume put ufakzeka-data /content/pairs_{i}.jsonl {part} --force')
    print(f'SLICE_{i}_DONE', flush=True)
sh('cat /content/pairs_*.jsonl > /content/pairs.jsonl')
sh('modal volume put ufakzeka-data /content/pairs.jsonl onpolicy/rule_pairs_$NAME.jsonl --force')
print('PAIRS_DONE', flush=True)
""")
PY
# /tmp/job_$S.py already holds the job body (the heredoc printed it); ship it verbatim
python3 - > /tmp/ship_$S.py <<PY
src=open("/tmp/job_$S.py").read()
print("import pathlib, subprocess")
print(f"pathlib.Path('/content/job.py').write_text({src!r})")
print("subprocess.Popen('nohup python /content/job.py > /content/job.log 2>&1 &', shell=True)")
print("print('started')")
PY
colab exec -s "$S" -f /tmp/ship_$S.py >> "$LOG" 2>&1; rm -f /tmp/job_$S.py /tmp/ship_$S.py
last=0; fails=0
while true; do
  sleep 60
  python3 scripts/colab_reattach.py "$S" >/dev/null 2>&1 || true
  printf 'print(open("/content/job.log").read())' > /tmp/poll_$S.py
  OUT=$(colab exec -s "$S" -f /tmp/poll_$S.py 2>/dev/null)
  # a runtime can die while its session object lives on: the poll then returns nothing, forever
  if [ -z "$OUT" ]; then fails=$((fails+1)); [ "$fails" -ge 5 ] && { echo "SESSION_LOST" >> "$LOG"; break; }; else fails=0; fi
  NEW=$(echo "$OUT" | tail -n +$((last+1)))
  [ -n "$NEW" ] && echo "$NEW" >> "$LOG"
  last=$(echo "$OUT" | wc -l)
  echo "$OUT" | grep -q "PAIRS_DONE" && break
  echo "$OUT" | grep -qE "Traceback|AssertionError" && break
  colab ls -s "$S" >/dev/null 2>&1 || { echo "SESSION_LOST" >> "$LOG"; break; }
done
rm -f /tmp/poll_$S.py
colab stop -s "$S" >> "$LOG" 2>&1

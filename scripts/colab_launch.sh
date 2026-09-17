#!/bin/bash
# Usage: scripts/colab_launch.sh SESSION LOGFILE -- <colab_job.py args>
# Provisions an A100, bootstraps tokens, starts scripts/colab_job.py detached on the runtime,
# polls its log until JOB_DONE or failure, then stops the session.
set -u
S=$1; LOG=$2; shift 2; [ "$1" = "--" ] && shift
export OAUTHLIB_RELAX_TOKEN_SCOPE=1
colab new --gpu A100 -s "$S" >> "$LOG" 2>&1
# a network blip on the first request must not downgrade the accelerator: one retry before any fallback
colab ls -s "$S" >/dev/null 2>&1 || { sleep 20; colab new --gpu A100 -s "$S" >> "$LOG" 2>&1; }
# fall back through the accelerators Colab may grant; the session name is the only handle the rest of this
# script uses, so nothing else changes
for ACC in L4 T4; do
  colab ls -s "$S" >/dev/null 2>&1 && break
  echo "falling back to $ACC" >> "$LOG"
  colab new --gpu $ACC -s "$S" >> "$LOG" 2>&1
done
colab ls -s "$S" >/dev/null 2>&1 || { echo "SESSION_FAILED" >> "$LOG"; exit 1; }
python3 - > /tmp/bootstrap_$S.py <<PY
toml=open("$HOME/.modal.toml").read(); hf=open("$HOME/.ufakzeka/hf_token").read().strip()
print("import os, subprocess, pathlib")
print(f"pathlib.Path(os.path.expanduser('~/.modal.toml')).write_text({toml!r})")
print(f"pathlib.Path(os.path.expanduser('~/.hf_token')).write_text({hf!r})")
print("subprocess.run('pip -q install modal tokenizers \"transformers>=4.45\" datasets zstandard 2>&1 | tail -1', shell=True)")
print("print('bootstrap ok')")
PY
colab exec -s "$S" -f /tmp/bootstrap_$S.py >> "$LOG" 2>&1; rm -f /tmp/bootstrap_$S.py
ARGS=$(python3 -c 'import sys, json; print(json.dumps(sys.argv[1:]))' "$@")
{ echo "import sys; sys.argv = ['colab_job.py'] + $ARGS"; cat scripts/colab_job.py; } > /tmp/job_$S.py
# ship the job file, then start it detached
python3 - > /tmp/start_$S.py <<PY
src=open("/tmp/job_$S.py").read()
print("import pathlib, subprocess")
print(f"pathlib.Path('/content/job.py').write_text({src!r})")
print("subprocess.Popen('nohup python /content/job.py > /content/job.log 2>&1 &', shell=True)")
print("print('started')")
PY
colab exec -s "$S" -f /tmp/start_$S.py >> "$LOG" 2>&1; rm -f /tmp/start_$S.py /tmp/job_$S.py
last=0; fails=0
while true; do
  sleep 120
  echo 'import os; print(open("/content/job.log").read() if os.path.exists("/content/job.log") else "")' | colab exec -s "$S" > /tmp/joblog_$S.txt 2>/dev/null
  n=$(wc -l < /tmp/joblog_$S.txt)
  if [ "$n" -eq 0 ]; then fails=$((fails+1)); [ "$fails" -ge 5 ] && { echo "SESSION_LOST" >> "$LOG"; break; }; else fails=0; fi
  if [ "$n" -gt "$last" ]; then tail -n +$((last+1)) /tmp/joblog_$S.txt >> "$LOG"; last=$n; fi
  if grep -qE "JOB_DONE|failed:|Traceback" /tmp/joblog_$S.txt; then break; fi
  # free tier sessions vanish without notice; polling a dead session forever hides the loss
  colab ls -s "$S" >/dev/null 2>&1 || { echo "SESSION_LOST" >> "$LOG"; break; }
done
colab stop -s "$S" >> "$LOG" 2>&1

#!/bin/bash
# Poll a detached Colab job (/content/job.log) every 2 minutes, refreshing the
# session token first so the hourly proxy token expiry cannot drop the session.
# Usage: colab_poll.sh <log file> [tail file] ; stops the session when the job ends.
export PATH="$HOME/.local/bin:$PATH"; export OAUTHLIB_RELAX_TOKEN_SCOPE=1
PY=$HOME/.local/share/uv/tools/google-colab-cli/bin/python
REPO=$(cd "$(dirname "$0")/.." && pwd)
LOG=$1; TAIL=${2:-${1%.log}_tail.txt}
while true; do
  $PY $REPO/scripts/colab_reattach.py >/dev/null 2>&1 || echo "reattach failed $(date)" >> "$LOG"
  # exec can hang for hours on a dropped connection; bound it so the poller never goes blind
  out=$($PY $REPO/scripts/colab_exec.py 'print(open("/content/job.log").read()[-3000:])' 180)
  echo "poll $(date +%H:%M) $(echo "$out" | tail -1 | cut -c1-80)" >> "${LOG%.log}_heartbeat.txt"
  echo "$out" > "$TAIL"; echo "$out" >> "$LOG"
  if echo "$out" | grep -qE "JOB_DONE|failed:|Traceback \(most recent"; then break; fi
  if echo "$out" | grep -q "appears to be lost"; then echo "SESSION_LOST" >> "$LOG"; sleep 30; continue; fi
  sleep 120
done
$PY $REPO/scripts/colab_reattach.py >/dev/null 2>&1
colab stop -s ufakzeka 2>&1 | tail -1 >> "$LOG"
echo "SESSION_STOPPED" >> "$LOG"

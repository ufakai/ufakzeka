"""Run a Python snippet on the Colab runtime with a timeout. Usage: colab_exec.py 'print(1)' [timeout_s]
Prints the runtime's output, or EXEC_TIMEOUT / EXEC_ERROR, and never hangs the caller."""
import subprocess
import sys

code = sys.argv[1]
timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 180
try:
    r = subprocess.run(["colab", "exec", "-s", "ufakzeka"], input=code, capture_output=True, text=True, timeout=timeout)
    print(r.stdout + r.stderr)
except subprocess.TimeoutExpired:
    print("EXEC_TIMEOUT")
except OSError as e:
    print("EXEC_ERROR", e)

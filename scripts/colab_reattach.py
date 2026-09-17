"""Refresh the Colab CLI session token from the assignments API.

The runtime proxy token stored by `colab new` expires after an hour and the CLI
never refreshes it, so every exec starts failing with 401 and the CLI drops the
session while the runtime keeps running. Run this before each CLI call (it is one
cheap request) or after a "session appears to be lost" message to re-attach.

Usage: ~/.local/share/uv/tools/google-colab-cli/bin/python scripts/colab_reattach.py [session_name]"""

import os
import sys

from colab_cli.common import State
from colab_cli.state import SessionState

name = sys.argv[1] if len(sys.argv) > 1 else "ufakzeka"
st = State()
assignments = st.client.list_assignments()
if not assignments:
    sys.exit("no live runtime")
a = assignments[0]
old = st.store.get(name)
s = SessionState(name=name, token=a.runtime_proxy_info.token, url=a.runtime_proxy_info.url, endpoint=a.endpoint,
                 variant=a.variant.name, accelerator=a.accelerator.value,
                 kernel_id=old.kernel_id if old else None, session_id=old.session_id if old else None,
                 keep_alive_pid=old.keep_alive_pid if old else None)
st.store.add(s)
alive = False
if s.keep_alive_pid:
    try:
        os.kill(s.keep_alive_pid, 0); alive = True
    except OSError:
        alive = False
if not alive:
    from colab_cli.commands.session import spawn_keep_alive
    s.keep_alive_pid = spawn_keep_alive(a.endpoint, name)
    st.store.add(s)
    print("keep-alive restarted, pid", s.keep_alive_pid)
print("reattached", name, a.accelerator.value, "token valid", a.runtime_proxy_info.token_expires_in_seconds, "s")

#!/usr/bin/env python3
"""Fake tts-server for ox-whisper TTS supervisor tests.

Accepts the same CLI shape as the real tts-server (--model, --codec,
--host, --port, --max-batch) and serves `GET /health` -> 200.

Variants are driven by marker files in $TMPDIR keyed by port, so no env
vars or extra CLI args are needed (the supervisor owns the argv):

  oxw_fake_tts_crash_<port>  : if ABSENT -> create it, serve /health, then
                              exit(42) after CRASH_AFTER_S seconds (a crash
                              that only happens once per marker lifecycle);
                              if present -> serve normally
  oxw_fake_tts_delay_<port>  : if present -> sleep contents-as-ms before
                              binding the socket (slow model load)

Always writes this process's /proc/self/oom_score_adj to
oxw_fake_tts_oom_<port> right after parsing args.
"""

import argparse
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

CRASH_AFTER_S = 0.8


def marker(name: str, port: int) -> str:
    return os.path.join(tempfile.gettempdir(), f"oxw_fake_tts_{name}_{port}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="")
    p.add_argument("--codec", default="")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--max-batch", default="2")
    a = p.parse_args()

    try:
        with open("/proc/self/oom_score_adj") as f:
            adj = f.read().strip()
        with open(marker("oom", a.port), "w") as f:
            f.write(adj)
    except OSError as e:
        print(f"fake_tts: cannot record oom_score_adj: {e}", file=sys.stderr)

    crash_file = marker("crash", a.port)
    do_crash = not os.path.exists(crash_file)
    if do_crash:
        open(crash_file, "w").close()

    delay_file = marker("delay", a.port)
    if os.path.exists(delay_file):
        with open(delay_file) as f:
            time_s = int(f.read().strip()) / 1000.0
        print(f"fake_tts: slow start {time_s}s", flush=True)
        time.sleep(time_s)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                body = b"ok"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args):
            pass

    srv = HTTPServer((a.host, a.port), Handler)

    if do_crash:
        t = threading.Timer(CRASH_AFTER_S, lambda: os._exit(42))
        t.daemon = True
        t.start()
        print(f"fake_tts: will crash in {CRASH_AFTER_S}s", flush=True)

    print(f"fake_tts ready on {a.host}:{a.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

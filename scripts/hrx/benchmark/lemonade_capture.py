# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Observe existing non-streaming bench requests without adding inference work.

Lemonade bench retains throughput but discards speculative counters. This local
HTTP relay preserves its requests and responses and records the raw completion
responses, including warmups, for CI evidence. Backend timings remain authoritative;
wall-clock client latency includes the relay and must not be used for speed claims.
"""
from __future__ import annotations

import gzip
import http.client
import zlib
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


@contextmanager
def capture_responses(target_port: int, output: Path, model_name: str):
    routes = (
        ("GET", "/api/v1/system-info"),
        ("GET", "/api/v1/system-stats"),
        ("GET", "/api/v1/health"),
        ("GET", "/api/v1/models/" + model_name),
        ("POST", "/api/v1/load"),
        ("POST", "/api/v1/unload"),
        ("POST", "/api/v1/chat/completions"),
    )
    records = []
    errors = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def forward(self):
            connection = http.client.HTTPConnection("127.0.0.1", target_port, timeout=3600)
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                route = next(((method, path) for method, path in routes
                              if (self.command, self.path) == (method, path)), None)
                if route is None:
                    self.send_error(404, "Not a benchmark endpoint")
                    return
                method, path = route
                headers = {"Content-Type": "application/json", "Accept-Encoding": "gzip, deflate"}
                connection.request(method, path, body, headers)
                response = connection.getresponse()
                payload = response.read()
                is_completion = self.path.endswith("/chat/completions")
                if is_completion:
                    decoded = payload
                    encoding = response.getheader("Content-Encoding", "identity")
                    if encoding == "gzip":
                        decoded = gzip.decompress(payload)
                    elif encoding == "deflate":
                        decoded = zlib.decompress(payload)
                    record = {"status": response.status, "request": json.loads(body),
                              "response": json.loads(decoded)}
                    with lock:
                        records.append(record)
                        with output.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(record) + "\n")
                self.send_response(response.status)
                self.send_header("Content-Type", "application/json")
                for encoding in ("gzip", "deflate"):
                    if response.getheader("Content-Encoding") == encoding:
                        self.send_header("Content-Encoding", encoding)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as exc:
                with lock:
                    errors.append(str(exc))
                self.send_error(502, "Benchmark response capture failed")
            finally:
                connection.close()

        do_GET = forward
        do_POST = forward
        do_DELETE = forward
        do_HEAD = forward

    output.parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, records
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    if errors:
        raise RuntimeError(f"Response capture failed: {errors}")


def draft_metrics(records: list[dict]) -> dict:
    drafted = accepted = successful = 0
    for record in records:
        if record["status"] != 200:
            continue
        successful += 1
        timings = record["response"].get("timings", {})
        proposed = timings.get("draft_n", 0)
        kept = timings.get("draft_n_accepted", 0)
        integers = type(proposed) is int and type(kept) is int
        if not integers:
            raise RuntimeError("Non-integer speculative token counters")
        if not 0 <= kept <= proposed:
            raise RuntimeError("Invalid speculative token counters")
        drafted += proposed
        accepted += kept
    return {"drafted": drafted, "accepted": accepted, "rejected": drafted - accepted,
            "acceptance_rate": accepted / drafted if drafted else None,
            "successful_requests": successful, "includes_warmups": True}

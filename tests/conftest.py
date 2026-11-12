"""Shared test helpers: a fake NeuroPrix server that speaks the runner protocol (docs/runner-protocol.md), so the runner
can be tested end to end without a real server, and helpers to start the runner and make sessions."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent          # the SDK repository (examples/, docs/)
CLASSES = ["n4", "n3", "n2", "n1"]
FAKE_TRAINER = Path(__file__).resolve().parent / "fake_trainer.py"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeServer:
    """The server side of the runner protocol, in memory. Brains are checked the way the real server checks them:
    neuroprix.engine.Brain must read them, and they must be the session's class."""

    def __init__(self, port: int | None = None):
        self.port = port or free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.lock = threading.Lock()
        self.auto_link = True        # the player types the code straight away
        self.tokens: set[str] = set()
        self.pairs: dict[str, dict] = {}
        self.queue: list[dict] = []  # sessions waiting for a runner
        self.jobs: dict[str, dict] = {}   # id -> {"job", "lines", "brains", "finish", "stop", "keep"}
        self.heartbeats = 0
        self.requests = 0
        self.redirect_to: str | None = None   # answer every call with a redirect there
        self.tokens_seen: list[str] = []
        self.httpd: ThreadingHTTPServer | None = None
        self.start()

    # -- control from the tests
    def start(self) -> None:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):   # quiet
                pass

            def _send(self, status: int, body: dict | None = None) -> None:
                raw = json.dumps(body or {}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _handle(self) -> None:
                server.requests += 1
                if self.headers.get("X-Runner-Token"):
                    server.tokens_seen.append(self.headers.get("X-Runner-Token"))
                if server.redirect_to:
                    self.send_response(307)
                    self.send_header("Location", server.redirect_to + self.path)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
                status, out = server.handle(self.command, self.path, body, self.headers.get("X-Runner-Token"))
                self._send(status, out)

            do_POST = do_PUT = do_GET = _handle

        ThreadingHTTPServer.allow_reuse_address = True
        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None

    def add_job(self, tier: str = "n3", flags: list | None = None, name: str = "Testy", resume_json: dict | None = None,
                jid: str | None = None, resume_pth: str | None = None) -> str:
        jid = jid or f"job{len(self.jobs) + len(self.queue) + 1:04d}"
        job = {"id": jid, "name": name, "tier": tier, "tracks": [], "resume_json": resume_json, "resume_pth": resume_pth,
               "flags": flags if flags is not None else ["--name", name, "--tier", tier, "--updates", "2", "--batch", "1",
                                                          "--traffic", "1", "--max-steps", "60", "--eval-every", "1",
                                                          "--check-tracks", "1", "--random-tracks"]}
        with self.lock:
            self.queue.append(job)
        return jid

    def wait_finish(self, jid: str, timeout: float = 180) -> dict:
        end = time.time() + timeout
        while time.time() < end:
            j = self.jobs.get(jid)
            if j and j["finish"]:
                return j
            time.sleep(0.2)
        raise AssertionError(f"session {jid} didn't finish in {timeout} s; jobs: {self.summary()}")

    def wait_for(self, check, timeout: float = 60, what: str = "condition") -> None:
        end = time.time() + timeout
        while time.time() < end:
            if check():
                return
            time.sleep(0.2)
        raise AssertionError(f"timed out waiting for {what}; jobs: {self.summary()}")

    def summary(self) -> dict:
        return {k: {"finish": v["finish"], "lines": len(v["lines"]), "brains": len(v["brains"])} for k, v in self.jobs.items()}

    # -- the protocol
    def handle(self, method: str, path: str, body: dict, token: str | None) -> tuple[int, dict]:
        from neuroprix import engine
        with self.lock:
            if path == "/runner/pair":
                pid = f"p{len(self.pairs) + 1}"
                self.pairs[pid] = {"secret": "s3cret", "name": body.get("name"), "info": body.get("info")}
                return 200, {"pair_id": pid, "secret": "s3cret", "code": "TEST-CODE", "expires_in": 600}
            if path == "/runner/pair/poll":
                p = self.pairs.get(body.get("pair_id"))
                if not p or body.get("secret") != p["secret"]:
                    return 200, {"status": "expired"}
                if not self.auto_link:
                    return 200, {"status": "waiting"}
                tok = f"tok-{len(self.tokens) + 1}"
                self.tokens.add(tok)
                return 200, {"status": "linked", "runner_id": "r1", "token": tok, "name": p["name"]}
            if token not in self.tokens:
                return 401, {"detail": "unknown runner: link it again"}
            if path == "/runner/heartbeat":
                self.heartbeats += 1
                job = self.queue.pop(0) if self.queue else None
                if job:
                    self.jobs[job["id"]] = {"job": job, "lines": [], "brains": [], "finish": None, "stop": False, "keep": False,
                                            "upload_fields": set()}
                return 200, {"job": job, "name": "test runner", "engine_version": engine.ENGINE_VERSION}
            parts = path.strip("/").split("/")   # runner/jobs/{id}/{what}
            if len(parts) != 4 or parts[:2] != ["runner", "jobs"] or parts[2] not in self.jobs:
                return 404, {"detail": "not your session"}
            j, what = self.jobs[parts[2]], parts[3]
            if what == "report":
                j["lines"] += body.get("lines", [])
                return 200, {"stop": j["stop"], "keep": j["keep"]}
            if what == "brain":
                j["upload_fields"] |= set(body)
                try:
                    b = engine.Brain(json.loads(body["brain_json"]))
                except Exception as e:
                    return 422, {"detail": f"brain refused: {e}"}
                if str(b.tier).lower() != j["job"]["tier"]:
                    return 422, {"detail": f"brain refused: the brain isn't an {j['job']['tier'].upper()} brain"}
                j["brains"].append(json.loads(body["brain_json"]))
                return 200, {"saved": True}
            if what == "finish":
                j["finish"] = {"status": body.get("status"), "error": body.get("error")}
                return 200, {"ok": True}
        return 404, {"detail": "no such call"}


class Runner:
    """The runner as a player starts it, in its own process, with its link kept in a test folder."""

    def __init__(self, server_url: str, home: Path, *args: str, trainer: str | None = None):
        env = {**os.environ, "NEUROPRIX_RUNNER_CONFIG": str(home / "runner.json"), "PYTHONUNBUFFERED": "1"}
        env.pop("NEUROPRIX_SERVER", None)
        env.pop("NEUROPRIX_TRAINER", None)
        cmd = [sys.executable, "-m", "neuroprix.runner", "--server", server_url, "--name", "test runner", "--workers", "1", *args]
        if trainer:
            cmd += ["--trainer", trainer]
        kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {}
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                                     encoding="utf-8", errors="replace", **kw)
        self.output: list[str] = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            self.output.append(line.rstrip("\n"))

    def text(self) -> str:
        return "\n".join(self.output)

    def wait_output(self, needle: str, timeout: float = 60) -> None:
        end = time.time() + timeout
        while time.time() < end:
            if needle in self.text():
                return
            if self.proc.poll() is not None:
                time.sleep(0.3)
                if needle in self.text():
                    return
                raise AssertionError(f"runner exited ({self.proc.returncode}) before printing {needle!r}:\n{self.text()}")
            time.sleep(0.2)
        raise AssertionError(f"runner never printed {needle!r}:\n{self.text()}")

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(10)


def fake_trainer_cmd(*args: str) -> str:
    """A --trainer command for the fast fake trainer (tests/fake_trainer.py), quoted for shlex on any OS."""
    import shlex
    return " ".join(shlex.quote(a.replace("\\", "/")) for a in [sys.executable, str(FAKE_TRAINER), *args])


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.stop()


@pytest.fixture
def runner_home(tmp_path):
    return tmp_path / "home"


@pytest.fixture
def start_runner(runner_home):
    started: list[Runner] = []

    def start(url: str, *args: str, trainer: str | None = None) -> Runner:
        r = Runner(url, runner_home, *args, trainer=trainer)
        started.append(r)
        return r
    yield start
    for r in started:
        r.close()


def write_session(folder: Path, cls: str, settings: dict | None = None, start_brain: Path | None = None) -> Path:
    """A session.json as the runner writes it (docs/trainer-contract.md), for running a trainer by hand."""
    folder.mkdir(parents=True, exist_ok=True)
    s = {"neuroprix_session": 1, "id": f"test-{cls}", "driver": "Testy", "class": cls,
         "settings": settings or {"updates": 2, "batch": 1, "eval_every": 1, "traffic": 1, "max_steps": 60,
                                  "check_tracks": 1, "random_tracks": True},
         "tracks": [], "start_brain": str(start_brain) if start_brain else None, "start_weights": None,
         "brain_out": str(folder / "brain.json"), "workers": 1, "engine_version": None}
    path = folder / "session.json"
    path.write_text(json.dumps(s))
    return path

"""NeuroPrix Runner: trains your drivers on this computer (or a machine you rent), for your NeuroPrix account.

    neuroprix-runner --server https://neuroprix.com/api                           (the SDK: pip install neuroprix)
    python runner/neuroprix_runner.py --server https://neuroprix.com/api           (from a NeuroPrix checkout)
    neuroprix-runner --server ... --trainer "python my_trainer.py"                 (your own trainer)

The first time, it shows a code: in the Garage, open a driver's Train tab, choose "Add a computer" and type it in.
After that it waits for work: when you start a session for it in the Garage, it trains here and you watch it there as
usual. Leave it running (Ctrl+C to stop; a session it was training then ends, keeping its best version so far).

The trainer is yours to choose. The runner prepares each session in a fresh folder and runs the trainer command with
`--session <folder>/session.json`. A trainer (any language) reads that file, trains, and writes its best brain to the
path the file names (brain_out) whenever it has a better one; the runner uploads each new version. It may also print
progress events (one JSON object per line) for the Garage's live view and previews. That's the whole contract: see
docs/trainer-contract.md in the SDK. By default the runner uses NeuroPrix's own trainer (PPO).

From the server the runner takes data only: the session's settings (checked against the known training settings),
gyms and the starting brain. It never runs anything the server sends. Its link to your account is a token in
~/.neuroprix/runner.json; remove the runner in the Garage to revoke it. Talks to the server over HTTP(S) with Python's
standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

VERSION = "0.2.0"
CONFIG = Path(os.environ.get("NEUROPRIX_RUNNER_CONFIG") or os.environ.get("NEURORACER_RUNNER_CONFIG")
              or Path.home() / ".neuroprix" / "runner.json")
OLD_CONFIG = Path.home() / ".neuroracer" / "runner.json"   # where it was kept before the rename (NeuroRacer)
SESSION_FORMAT = 1


def engine_version() -> str | None:
    """The version of the race engine this runner trains with (the server warns if it's out of date)."""
    try:
        from neuroprix import engine   # the SDK
        return engine.ENGINE_VERSION
    except ImportError:
        pass
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from backend import sim        # a NeuroPrix checkout
        return sim.ENGINE_VERSION
    except Exception:
        return None


def default_trainer() -> list[str]:
    """NeuroPrix's own trainer: the SDK's module, or agent/train.py in a checkout."""
    try:
        import neuroprix.train  # noqa: F401
        return [sys.executable, "-m", "neuroprix.train"]
    except ImportError:
        return [sys.executable, str(Path(__file__).resolve().parent.parent / "agent" / "train.py")]


# The training settings a session may carry (the server sends them as trainer options; anything else is refused).
NUMBER_SETTINGS = {"updates", "batch", "eval-every", "traffic", "max-steps", "epochs", "minibatch",
                   "progress", "alive", "fast", "fast-threshold", "idle", "dead", "damage", "finish", "contact",
                   "steer-flip", "corner", "risk", "crash", "lose-place", "overtake", "wall", "start-margin", "check-tracks", "check-walls",
                   "lr", "entropy", "gamma", "lam", "clip"}
INT_SETTINGS = {"updates", "batch", "eval-every", "traffic", "max-steps", "epochs", "minibatch", "check-tracks"}
SWITCHES = {"random-tracks", "one-way"}
CLASS_ALIAS = {"f3": "n4", "f2": "n3", "f1": "n2"}   # the classes' names before November 2026


def session_settings(flags: list) -> dict:
    """The server's trainer options as plain settings ({"updates": 400, "random_tracks": true, ...}), refused unless
    every one is a known setting with a sensible value."""
    out, i = {}, 0
    while i < len(flags):
        f = flags[i]
        if not isinstance(f, str) or not f.startswith("--"):
            raise ValueError("bad setting")
        name = f[2:]
        if name in SWITCHES:
            out[name.replace("-", "_")] = True
            i += 1
            continue
        if i + 1 >= len(flags) or not isinstance(flags[i + 1], str):
            raise ValueError(f"setting {f} has no value")
        v = flags[i + 1]
        if name in NUMBER_SETTINGS:
            x = float(v)
            out[name.replace("-", "_")] = int(round(x)) if name in INT_SETTINGS else x
        elif name == "name":
            if not re.match(r"^[^\x00-\x1f]{1,64}$", v):
                raise ValueError("odd driver name")
            out["name"] = v
        elif name == "tier":
            c = CLASS_ALIAS.get(v.lower(), v.lower())
            if not re.match(r"^n[1-4]$", c):
                raise ValueError("unknown class")
            out["class"] = c
        elif name == "seeds":
            if not re.match(r"^[0-9,]{0,2000}$", v):
                raise ValueError("odd seeds")
            out["seeds"] = [int(s) for s in v.split(",") if s]
        else:
            raise ValueError(f"unknown setting {f!r}: this runner may be out of date")
        i += 2
    return out


def stop_tree(proc: subprocess.Popen, hard: bool = False) -> None:
    """Stop the trainer and anything it started (its game workers)."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":   # Windows: end the whole process tree
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
        else:
            os.killpg(proc.pid, signal.SIGKILL if hard else signal.SIGTERM)
    except OSError:
        pass


_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def clean(text) -> str:
    """Text from the server or a trainer, safe to print: no terminal control sequences (escape codes could rewrite
    the screen or the window title)."""
    return _CONTROL.sub("?", str(text))


def say(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), clean(msg), flush=True)


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, f"the server redirected to {newurl}: use that address as --server",
                                     headers, fp)


_OPENER = urllib.request.build_opener(_NoRedirects)
MAX_REPLY = 50_000_000   # a session's gyms and starting brain are well under 1 MB


class ServerError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


class Api:
    def __init__(self, server: str, token: str | None = None):
        self.server, self.token = server.rstrip("/"), token

    def call(self, method: str, path: str, body: dict | None = None, timeout: float = 30) -> dict:
        """One JSON call. Redirects are refused, so the token only ever goes to the server you linked."""
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.server + path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", f"neuroprix-runner/{VERSION}")
        if self.token:
            req.add_header("X-Runner-Token", self.token)
        try:
            with _OPENER.open(req, timeout=timeout) as resp:
                raw = resp.read(MAX_REPLY + 1)
                if len(raw) > MAX_REPLY:
                    raise ServerError(resp.status, "the server's reply is too big")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read() or b"{}").get("detail", "")
            except Exception:
                detail = ""
            raise ServerError(e.code, str(detail) or e.reason)


def machine_info(workers: int, trainer: str) -> dict:
    return {"platform": f"{platform.system()} {platform.release()}"[:60], "machine": platform.machine(),
            "cpus": os.cpu_count() or 1, "workers": workers, "version": VERSION, "python": platform.python_version(),
            "engine": engine_version(), "trainer": trainer[:60]}


def load_config() -> dict:
    for path in (CONFIG, OLD_CONFIG):
        try:
            return json.loads(path.read_text())
        except Exception:
            continue
    return {}


def save_config(cfg: dict) -> None:
    CONFIG.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = CONFIG.with_suffix(".tmp")
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    # the token is a key to your account's training: the file is created private (0600), never readable by others
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(cfg, indent=1))
    tmp.replace(CONFIG)


def pair(server: str, name: str, info: dict) -> dict:
    """Get a code, show it, wait until it's entered in the Garage."""
    api = Api(server)
    while True:
        try:
            p = api.call("POST", "/runner/pair", {"name": name, "info": info})
        except (ServerError, OSError) as e:   # server down, or not a NeuroPrix server: say so and keep trying
            say(f"can't get a code from {server} ({e}); retrying in 10 s")
            time.sleep(10)
            continue
        bar = "=" * 56
        print(f"\n{bar}\n  Link this runner to your NeuroPrix account:\n\n"
              f"  1. Open the Garage at {server.removesuffix('/api')}\n  2. In a driver's Train tab, choose \"Add a computer\"\n"
              f"  3. Type this code:   {clean(p['code'])[:20]}\n\n  (valid for {int(p['expires_in']) // 60} minutes)\n{bar}\n", flush=True)
        while True:
            time.sleep(2)
            try:
                r = api.call("POST", "/runner/pair/poll", {"pair_id": p["pair_id"], "secret": p["secret"]})
            except (ServerError, OSError) as e:
                say(f"can't reach the server ({e}); still waiting")
                continue
            if r.get("status") == "linked":
                cfg = {"server": server, "runner_id": r["runner_id"], "token": r["token"], "name": r.get("name", name)}
                save_config(cfg)
                say(f"linked as \"{cfg['name']}\". Waiting for training sessions.")
                return cfg
            if r.get("status") == "expired":
                say("the code expired; here's a new one")
                break


def send_brain(api: Api, jid: str, brain_json: Path) -> bool:
    """Upload the trainer's best brain (its JSON: numbers only; the server keeps brains as JSON)."""
    try:
        js = brain_json.read_text()
        json.loads(js)   # (a trainer mid-write: try again next time)
    except (OSError, ValueError):
        return False
    body = {"brain_json": js}
    try:
        api.call("PUT", f"/runner/jobs/{jid}/brain", body, timeout=60)
        return True
    except ServerError as e:
        say(f"  the server refused the brain: {e.detail}")
        return True   # (don't send the same refused brain again and again)
    except OSError:
        return False


def run_job(api: Api, job: dict, workers: int, trainer: list[str]) -> None:
    jid = str(job.get("id", ""))
    if not re.match(r"^[A-Za-z0-9_-]{1,64}$", jid):
        say("the server sent a session with an odd id; ignoring it")
        return
    say(f"training {job.get('name', '?')} ({str(job.get('tier', '')).upper()}), session {jid}")
    folder = Path(tempfile.mkdtemp(prefix="neuroprix-"))
    proc = None
    try:
        try:
            settings = session_settings(job.get("flags") or [])
        except ValueError as e:
            api.call("POST", f"/runner/jobs/{jid}/finish", {"status": "failed", "error": f"the runner refused this session: {e}"})
            say(f"refused session {jid}: {e}")
            return
        start_brain = start_weights = None
        if job.get("resume_json"):
            start_brain = folder / "start.json"
            start_brain.write_text(json.dumps(job["resume_json"]))
        # (a .pth the server might send is ignored: PyTorch files can carry code, and loading one safely depends on
        # the PyTorch version. Brains travel as JSON, numbers only; the trainer makes its own .pth from that.)
        brain_out = folder / "brain.json"
        session = {
            "neuroprix_session": SESSION_FORMAT,
            "id": jid,
            "driver": settings.pop("name", job.get("name", "driver")),
            "class": settings.pop("class", job.get("tier", "n3")),
            "settings": settings,
            "tracks": job.get("tracks") or [],
            "start_brain": str(start_brain) if start_brain else None,
            "start_weights": str(start_weights) if start_weights else None,
            "brain_out": str(brain_out),
            "workers": workers,
            "engine_version": engine_version(),
        }
        (folder / "session.json").write_text(json.dumps(session, indent=1))
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "PYTHONUNBUFFERED": "1", "NEUROPRIX_SESSION": str(folder / "session.json")}
        group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        proc = subprocess.Popen([*trainer, "--session", str(folder / "session.json")], cwd=str(folder),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
                                encoding="utf-8", errors="replace", **group)
        lines: queue.Queue = queue.Queue()

        def read():
            for line in proc.stdout:
                lines.put(line.rstrip("\n"))
            lines.put(None)
        threading.Thread(target=read, daemon=True).start()

        pending, tail, finished_reading, stopping = [], [], False, False
        last_post, last_ok, sent_mtime = 0.0, time.time(), 0.0
        while True:
            try:   # gather what the trainer printed
                while len(pending) < 1500:
                    item = lines.get(timeout=0.2)
                    if item is None:
                        finished_reading = True
                        break
                    pending.append(item)
                    if item.startswith("{"):
                        try:
                            ev = json.loads(item)
                            if ev.get("event") == "update" and ev.get("update", 0) % 25 == 0:
                                say(f"  update {ev['update']}")
                        except ValueError:
                            pass
                    else:
                        tail = (tail + [item])[-5:]
            except queue.Empty:
                pass
            # a new best brain written by the trainer: upload it (whoever wrote the trainer, this is all it has to do)
            try:
                m = brain_out.stat().st_mtime
            except OSError:
                m = 0.0
            if m and m != sent_mtime and send_brain(api, jid, brain_out):
                sent_mtime = m
            now = time.time()
            if pending and (now - last_post > 5.0 or len(pending) >= 1500 or finished_reading) or now - last_post > 10:
                try:
                    r = api.call("POST", f"/runner/jobs/{jid}/report", {"lines": pending[:1500]})
                    pending, last_post, last_ok = pending[1500:], now, now
                    if r.get("stop") and not stopping:
                        stopping = True
                        say("  stopped from the Garage" + (" (keeping the best version so far)" if r.get("keep") else ""))
                        stop_tree(proc)
                except ServerError as e:
                    if e.status in (401, 404):
                        say(f"  the server dropped this session ({e.detail}); stopping it")
                        stop_tree(proc)
                        return
                    last_post = now
                except OSError as e:   # no connection: keep the lines and try again
                    last_post = now
                    if now - last_ok > 30:
                        say(f"  can't reach the server ({e}); still training, will send progress when it's back")
                        last_ok = now
                    if len(pending) > 20000:
                        pending = [ln for ln in pending if not ln.startswith('{"event": "update"')][-20000:]
            if finished_reading and not pending:
                break
        code = proc.wait()
        try:
            m = brain_out.stat().st_mtime
        except OSError:
            m = 0.0
        if m and m != sent_mtime:
            send_brain(api, jid, brain_out)
        if stopping:
            api.call("POST", f"/runner/jobs/{jid}/finish", {"status": "stopped"})
            say(f"session {jid} stopped")
        elif code == 0 and m:
            api.call("POST", f"/runner/jobs/{jid}/finish", {"status": "done"})
            say(f"session {jid} done")
        else:
            err = " | ".join(t for t in tail if t.strip())[-400:] or (
                f"the trainer exited with code {code}" if code else "the trainer finished without writing a brain")
            api.call("POST", f"/runner/jobs/{jid}/finish", {"status": "failed", "error": err})
            say(f"session {jid} failed: {err}")
    except KeyboardInterrupt:   # the runner is being closed: say so, so the session ends now (keeping its best version)
        try:
            api.call("POST", f"/runner/jobs/{jid}/finish", {"status": "failed", "error": "the runner was closed while training"}, timeout=5)
        except Exception:
            pass
        raise
    finally:
        if proc is not None:
            stop_tree(proc, hard=True)
        shutil.rmtree(folder, ignore_errors=True)


def _quit(*_):
    raise KeyboardInterrupt


def main() -> None:
    # Ctrl+C, and the stop signal from `docker stop` or a service manager, end the runner cleanly: a session it's
    # training is reported as ended (keeping its best version) and the trainer is stopped.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _quit)
    ap = argparse.ArgumentParser(description="NeuroPrix Runner: train your drivers on this machine.")
    ap.add_argument("--server", default=os.environ.get("NEUROPRIX_SERVER"), help="the NeuroPrix server, e.g. https://neuroprix.com/api (remembered)")
    ap.add_argument("--name", default=None, help="what to call this runner in the Garage (default: this computer's name)")
    ap.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)),
                    help="games to run in parallel while training (default: CPU cores minus 2)")
    ap.add_argument("--trainer", default=os.environ.get("NEUROPRIX_TRAINER"),
                    help='your own trainer command, e.g. "python my_trainer.py" (default: NeuroPrix\'s PPO trainer). '
                         "It's run with --session <file>: see docs/trainer-contract.md")
    ap.add_argument("--relink", action="store_true", help="forget the current link and show a new code")
    ap.add_argument("--allow-http", action="store_true",
                    help="allow a plain http:// server that isn't this computer (unsafe: the runner's token travels "
                         "unencrypted; only for a server on your own network)")
    ap.add_argument("--version", action="version", version=f"neuroprix-runner {VERSION} (engine {engine_version() or 'unknown'})")
    args = ap.parse_args()
    trainer = shlex.split(args.trainer) if args.trainer else default_trainer()
    trainer_label = "custom: " + Path(trainer[-1]).name if args.trainer else "NeuroPrix PPO"
    cfg = {} if args.relink else load_config()
    server = (args.server or cfg.get("server") or "http://localhost:8000").rstrip("/")
    if not re.match(r"^https?://[^/\s]+", server):
        ap.error(f"--server must be a web address starting with https:// (or http://), e.g. https://neuroprix.com/api; got {server!r}")
    host = urllib.parse.urlsplit(server).hostname or ""
    if server.startswith("http://") and host not in ("localhost", "127.0.0.1", "::1") and not args.allow_http:
        ap.error(f"{server} isn't encrypted (http://): use https://, so the runner's token can't be read on the way "
                 "(or --allow-http, only for a server on your own network)")
    name = args.name or cfg.get("name") or socket.gethostname().split(".")[0][:40] or "My runner"
    info = machine_info(args.workers, trainer_label)
    if cfg.get("server") != server or not cfg.get("token"):
        cfg = pair(server, name, info)
    api = Api(server, cfg["token"])
    say(f"NeuroPrix Runner {VERSION}: \"{cfg.get('name', name)}\" on {server}, {args.workers} games in parallel, "
        f"trainer: {' '.join(trainer[-2:]) if args.trainer else trainer_label}. Waiting for training sessions (Ctrl+C to quit).")
    quiet_since, warned_engine = None, False
    while True:
        try:
            r = api.call("POST", "/runner/heartbeat", {"info": info})
            if quiet_since:
                say("connected again")
                quiet_since = None
            if r.get("engine_version") and info["engine"] and r["engine_version"] != info["engine"] and not warned_engine:
                say(f"WARNING: this runner's engine is {info['engine']}, the server's is {r['engine_version']}: update the "
                    "runner (pip install -U neuroprix), or you'll be training on different physics than the races use")
                warned_engine = True
            if r.get("job"):
                run_job(api, r["job"], args.workers, trainer)
                say("waiting for training sessions")
                continue
        except ServerError as e:
            if e.status == 401:
                say("this runner was removed from your account; link it again")
                cfg = pair(server, name, info)
                api = Api(server, cfg["token"])
                continue
            say(f"server error {e}")
        except OSError as e:
            if not quiet_since:
                say(f"can't reach {server} ({e}); retrying")
                quiet_since = time.time()
        time.sleep(10)   # idle: check for a new session every 10 s


def run() -> None:   # the neuroprix-runner command
    try:
        main()
    except KeyboardInterrupt:
        print()
        say("runner stopped")


if __name__ == "__main__":
    run()

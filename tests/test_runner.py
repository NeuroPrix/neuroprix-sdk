"""The runner end to end against a fake server (tests/conftest.py): linking, training sessions for every class, and
what happens when things go wrong (bad token, server down, Stop pressed, runner closed, a failing trainer)."""
import json
import os
import signal
import sys
import time

import pytest

from conftest import CLASSES, FakeServer, fake_trainer_cmd


def test_links_with_a_code_and_remembers_it(server, start_runner, runner_home):
    r = start_runner(server.url, trainer=fake_trainer_cmd())
    r.wait_output("TEST-CODE")
    r.wait_output("Waiting for training sessions")
    cfg = json.loads((runner_home / "runner.json").read_text())
    assert cfg["server"] == server.url and cfg["token"] in server.tokens
    if os.name != "nt":
        assert (runner_home / "runner.json").stat().st_mode & 0o077 == 0, "the token file must be private"
    r.close()
    # started again: no new code, it's still linked
    r2 = start_runner(server.url, trainer=fake_trainer_cmd())
    r2.wait_output("Waiting for training sessions")
    assert "TEST-CODE" not in r2.text()
    assert len(server.pairs) == 1


def test_waits_for_the_code_to_be_typed(server, start_runner):
    server.auto_link = False
    r = start_runner(server.url, trainer=fake_trainer_cmd())
    r.wait_output("TEST-CODE")
    time.sleep(3)
    assert "Waiting for training sessions" not in r.text()
    server.auto_link = True
    r.wait_output("Waiting for training sessions", timeout=20)


@pytest.mark.parametrize("cls", CLASSES)
def test_trains_a_session_with_the_default_trainer(cls, server, start_runner):
    """The whole loop with NeuroPrix's own trainer: session in, progress and brain out, session done."""
    jid = server.add_job(cls)
    start_runner(server.url)
    j = server.wait_finish(jid, timeout=300)
    assert j["finish"]["status"] == "done", j["finish"]
    assert j["brains"], "no brain was uploaded"
    events = [json.loads(ln) for ln in j["lines"] if ln.startswith("{")]
    assert any(e.get("event") == "update" for e in events)
    assert any(e.get("event") == "eval" for e in events)


def test_continues_a_driver_from_its_current_brain(server, start_runner):
    first = server.add_job("n4")
    start_runner(server.url)
    brain = server.wait_finish(first, timeout=300)["brains"][-1]
    second = server.add_job("n4", resume_json=brain)
    j = server.wait_finish(second, timeout=300)
    assert j["finish"]["status"] == "done" and j["brains"]


def test_relinks_when_its_token_is_refused(server, start_runner, runner_home):
    runner_home.mkdir(parents=True)
    (runner_home / "runner.json").write_text(json.dumps({"server": server.url, "runner_id": "r9", "token": "revoked", "name": "x"}))
    r = start_runner(server.url, trainer=fake_trainer_cmd())
    r.wait_output("removed from your account")
    r.wait_output("TEST-CODE")
    r.wait_output("linked as", timeout=20)
    jid = server.add_job("n3")
    assert server.wait_finish(jid, timeout=60)["finish"]["status"] == "done"


def test_server_down_at_start_then_up():
    s = FakeServer()
    url, port = s.url, s.port
    s.stop()
    from conftest import Runner
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as home:
        r = Runner(url, Path(home), trainer=fake_trainer_cmd())
        try:
            r.wait_output("can't get a code", timeout=30)
            assert r.proc.poll() is None, "the runner must keep trying, not crash"
            s2 = FakeServer(port)
            try:
                r.wait_output("TEST-CODE", timeout=30)
                r.wait_output("Waiting for training sessions", timeout=20)
            finally:
                s2.stop()
        finally:
            r.close()


def test_server_down_while_training_then_back(server, start_runner):
    """The server goes away mid-session: training carries on, progress is kept and sent when it's back."""
    jid = server.add_job("n3")
    start_runner(server.url, trainer=fake_trainer_cmd("--steps", "40", "--delay", "0.5"))
    server.wait_for(lambda: server.jobs.get(jid) and server.jobs[jid]["lines"], timeout=60, what="first progress")
    port, jobs, tokens = server.port, server.jobs, server.tokens
    server.stop()
    time.sleep(8)
    back = FakeServer(port)
    back.jobs, back.tokens = jobs, tokens   # the same server, restarted
    try:
        j = back.wait_finish(jid, timeout=90)
        assert j["finish"]["status"] == "done"
        updates = {json.loads(ln)["update"] for ln in j["lines"] if ln.startswith('{"event": "update"')}
        assert updates == set(range(1, 41)), "progress printed while the server was down was lost"
    finally:
        back.stop()


def test_stop_and_keep_from_the_garage(server, start_runner):
    jid = server.add_job("n3")
    r = start_runner(server.url, trainer=fake_trainer_cmd("--steps", "200", "--delay", "0.2"))
    server.wait_for(lambda: server.jobs.get(jid) and server.jobs[jid]["brains"], timeout=60, what="a first brain")
    server.jobs[jid]["stop"] = server.jobs[jid]["keep"] = True
    j = server.wait_finish(jid, timeout=60)
    assert j["finish"]["status"] == "stopped"
    r.wait_output("stopped from the Train page")
    r.wait_output("waiting for training sessions")


@pytest.mark.skipif(os.name == "nt", reason="Ctrl+C is sent differently on Windows")
def test_closing_the_runner_ends_its_session(server, start_runner):
    jid = server.add_job("n3")
    r = start_runner(server.url, trainer=fake_trainer_cmd("--steps", "200", "--delay", "0.2"))
    server.wait_for(lambda: server.jobs.get(jid) and server.jobs[jid]["lines"], timeout=60, what="progress")
    r.proc.send_signal(signal.SIGINT)
    assert r.proc.wait(20) == 0
    j = server.jobs[jid]
    assert j["finish"] == {"status": "failed", "error": "the runner was closed while training"}
    assert "runner stopped" in r.text()


@pytest.mark.skipif(os.name == "nt", reason="SIGTERM isn't used on Windows")
def test_docker_stop_ends_its_session(server, start_runner):
    jid = server.add_job("n3")
    r = start_runner(server.url, trainer=fake_trainer_cmd("--steps", "200", "--delay", "0.2"))
    server.wait_for(lambda: server.jobs.get(jid) and server.jobs[jid]["lines"], timeout=60, what="progress")
    r.proc.send_signal(signal.SIGTERM)
    assert r.proc.wait(20) == 0
    assert server.jobs[jid]["finish"]["status"] == "failed"


def test_a_failing_trainer_fails_the_session_with_its_message(server, start_runner):
    jid = server.add_job("n3")
    start_runner(server.url, trainer=fake_trainer_cmd("--fail"))
    j = server.wait_finish(jid, timeout=60)
    assert j["finish"]["status"] == "failed"
    assert "something went wrong in the trainer" in j["finish"]["error"]


def test_a_brain_of_the_wrong_class_is_refused(server, start_runner):
    jid = server.add_job("n3")
    r = start_runner(server.url, trainer=fake_trainer_cmd("--bad-brain"))
    j = server.wait_finish(jid, timeout=60)
    assert not j["brains"]
    r.wait_output("the server refused the brain")


def test_refuses_a_session_with_unknown_settings(server, start_runner):
    jid = server.add_job("n3", flags=["--resume", "/etc/passwd"])
    r = start_runner(server.url, trainer=fake_trainer_cmd())
    j = server.wait_finish(jid, timeout=60)
    assert j["finish"]["status"] == "failed"
    assert "refused this session" in j["finish"]["error"]
    r.wait_output("refused session")


def test_runs_sessions_one_after_another(server, start_runner):
    ids = [server.add_job(c) for c in CLASSES]
    start_runner(server.url, trainer=fake_trainer_cmd())
    for jid in ids:
        assert server.wait_finish(jid, timeout=60)["finish"]["status"] == "done"


# ---------------------------------------------------------------------------------------------- security
def test_token_folder_is_private(server, start_runner, runner_home):
    r = start_runner(server.url, trainer=fake_trainer_cmd())
    r.wait_output("Waiting for training sessions")
    if os.name != "nt":
        assert runner_home.stat().st_mode & 0o077 == 0
        assert not list(runner_home.glob("*.tmp"))


def test_never_follows_a_redirect(server, start_runner):
    """A redirect would carry the token along (even to another host, or to plain http): it's refused."""
    other = FakeServer()
    try:
        server.redirect_to = other.url
        r = start_runner(server.url, trainer=fake_trainer_cmd())
        r.wait_output("redirected to")
        time.sleep(1)
        assert other.requests == 0
        assert r.proc.poll() is None, "it keeps trying, it doesn't crash"
    finally:
        other.stop()


def test_never_follows_a_redirect_with_its_token(server, start_runner):
    other = FakeServer()
    try:
        r = start_runner(server.url, trainer=fake_trainer_cmd())
        r.wait_output("Waiting for training sessions")
        server.redirect_to = other.url
        r.wait_output("redirected to", timeout=30)
        assert other.requests == 0 and not other.tokens_seen
    finally:
        other.stop()


def test_server_text_cannot_control_the_terminal(server, start_runner):
    jid = server.add_job("n3", name="\x1b]0;pwned\x07\x1b[2J Evil")
    r = start_runner(server.url, trainer=fake_trainer_cmd())
    server.wait_finish(jid, timeout=60)
    r.wait_output("Evil")
    assert "\x1b" not in r.text() and "\x07" not in r.text()


def test_ignores_a_session_with_an_odd_id(server, start_runner):
    server.add_job("n3", jid="../../runners/x")
    good = server.add_job("n3")
    r = start_runner(server.url, trainer=fake_trainer_cmd())
    r.wait_output("odd id")
    assert server.wait_finish(good, timeout=60)["finish"]["status"] == "done"


def test_never_loads_pytorch_files_from_the_server(server, start_runner):
    """A .pth from the server is ignored (PyTorch files can carry code); brains go up as JSON only."""
    import base64
    jid = server.add_job("n3", resume_pth=base64.b64encode(b"not a safe file").decode())
    start_runner(server.url, trainer=fake_trainer_cmd())
    j = server.wait_finish(jid, timeout=60)
    assert j["finish"]["status"] == "done"
    assert any("start_weights=None" in ln for ln in j["lines"])
    assert j["upload_fields"] == {"brain_json"}

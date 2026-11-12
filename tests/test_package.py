"""The installed package: its commands, versions, the engine and brains for every class."""
import json
import re
import os
import shutil
import subprocess
import sys
import sysconfig

import pytest

import neuroprix
from neuroprix import engine
from neuroprix.brain import TIERS, RaceBrain, save_brain
from neuroprix.env import SimEnv
from neuroprix.runner import VERSION, session_settings

from conftest import CLASSES


def run(*cmd, **kw):
    return subprocess.run(list(cmd), capture_output=True, text=True, timeout=120, **kw)


def test_versions():
    assert re.match(r"^\d+\.\d+\.\d+", neuroprix.__version__)
    assert neuroprix.__version__ == VERSION, "the package and the runner must report the same version"
    assert neuroprix.ENGINE_VERSION == engine.ENGINE_VERSION
    assert re.match(r"^\d{4}\.\d+\.\d+$", engine.ENGINE_VERSION)


@pytest.mark.parametrize("cmd", [["neuroprix-runner"], ["neuroprix-train"]])
def test_console_commands_installed(cmd):
    exe = shutil.which(cmd[0], path=sysconfig.get_path("scripts")) or shutil.which(cmd[0])
    assert exe, f"{cmd[0]} isn't installed: is the package installed (pip install .)?"
    r = run(exe, "--help")
    assert r.returncode == 0, r.stderr
    assert "--session" in r.stdout or "--server" in r.stdout


def test_runner_help_and_version():
    r = run(sys.executable, "-m", "neuroprix.runner", "--help")
    assert r.returncode == 0
    for opt in ("--server", "--name", "--workers", "--trainer", "--relink", "--version"):
        assert opt in r.stdout
    r = run(sys.executable, "-m", "neuroprix.runner", "--version")
    assert r.returncode == 0
    assert VERSION in r.stdout and engine.ENGINE_VERSION in r.stdout


def test_runner_refuses_a_server_without_https(tmp_path):
    r = run(sys.executable, "-m", "neuroprix.runner", "--server", "neuroprix.example",
            env={**os.environ, "NEUROPRIX_RUNNER_CONFIG": str(tmp_path / "r.json")})
    assert r.returncode == 2
    assert "must be a web address" in r.stderr


def test_train_help():
    r = run(sys.executable, "-m", "neuroprix.train", "--help")
    assert r.returncode == 0
    for opt in ("--session", "--tier", "--updates", "--random-tracks"):
        assert opt in r.stdout


@pytest.mark.parametrize("cls", CLASSES)
def test_engine_senses_and_steps_every_class(cls):
    env = SimEnv()
    env.setup_world({"seed": 202}, cars=2, tier=cls)
    states = env.reset_world()
    assert len(states) == 2
    n_in, n_actions = TIERS[cls]
    assert len(states[0]["obs"]) == n_in
    for _ in range(20):
        states = env.step_world([n_actions - 1, 0])
    assert isinstance(states[0]["distance"], (int, float))
    assert {"dead", "finished", "obs"} <= set(states[0])


def test_engine_is_deterministic():
    def drive():
        env = SimEnv()
        env.setup_world({"seed": 7}, cars=1, tier="n3")
        s = env.reset_world()
        for i in range(50):
            s = env.step_world([i % 9])
        return s[0]["distance"], s[0]["obs"]
    assert drive() == drive()


@pytest.mark.parametrize("cls", CLASSES)
def test_brain_round_trip_every_class(cls, tmp_path):
    n_in, n_out = TIERS[cls]
    model = RaceBrain(n_in, 64, n_out)
    save_brain(model, tmp_path / "b.pth", "Testy")
    data = json.loads((tmp_path / "b.json").read_text())
    b = engine.Brain(data)          # what the server checks
    assert b.tier.lower() == cls   # (the engine names classes in capitals)
    assert 0 <= b.act([0.5] * n_in) < n_out


def test_wrong_shape_brain_is_refused():
    with pytest.raises(ValueError):
        engine.Brain({"fc1": {"weight": [[0.0] * 5] * 64, "bias": [0.0] * 64},
                      "fc2": {"weight": [[0.0] * 64] * 3, "bias": [0.0] * 3}})


def test_session_settings_accepts_the_known_settings():
    s = session_settings(["--name", "Jango", "--tier", "f1", "--updates", "400.0", "--random-tracks", "--damage", "-90",
                          "--seeds", "3,8,21", "--one-way"])
    assert s == {"name": "Jango", "class": "n2", "updates": 400, "random_tracks": True, "damage": -90.0,
                 "seeds": [3, 8, 21], "one_way": True}


@pytest.mark.parametrize("flags", [
    ["--resume", "/etc/passwd"],          # paths and unknown options are never taken from the server
    ["--updates"],                        # no value
    ["--tier", "f9"],                     # unknown class
    ["updates", "3"],                     # not an option
    ["--name", "bad\nname"],
    ["--seeds", "1;rm -rf /"],
    ["--updates", "lots"],
])
def test_session_settings_refuses_anything_else(flags):
    with pytest.raises(ValueError):
        session_settings(flags)


def test_runner_refuses_plain_http_to_another_computer(tmp_path):
    env = {**os.environ, "NEUROPRIX_RUNNER_CONFIG": str(tmp_path / "r.json")}
    r = run(sys.executable, "-m", "neuroprix.runner", "--server", "http://neuroprix.example", env=env)
    assert r.returncode == 2
    assert "isn't encrypted" in r.stderr and "--allow-http" in r.stderr


def test_clean_removes_terminal_control_codes():
    from neuroprix.runner import clean
    assert clean("a\x1b[2Jb\x07c\x9bd\te\nf") == "a?[2Jb?c?d\te\nf"   # (tabs and new lines are fine)

"""The race engines: the Python engine is the default and needs nothing else; the optional Rust one (neuroprix-engine)
is used only when asked, only when it matches, never instead of a changed engine.py, and then gives exactly the
Python engine's results. The Rust tests are skipped when neuroprix-engine isn't installed."""
import json
import random
import subprocess
import sys

import pytest

from neuroprix import engine, rust_env
from neuroprix.env import POOL_SIZE, SimEnv

from conftest import write_session

try:
    import neuroprix_engine  # noqa: F401
    HAVE_RUST = True
except ImportError:
    HAVE_RUST = False
needs_rust = pytest.mark.skipif(not HAVE_RUST, reason="neuroprix-engine isn't installed")


def notes(cmd, session):
    r = subprocess.run([*cmd, "--session", str(session)], capture_output=True, text=True, timeout=300, cwd=str(session.parent))
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
    return [json.loads(ln)["message"] for ln in r.stdout.splitlines() if ln.startswith("{") and '"note"' in ln]


def test_the_sdk_never_loads_the_rust_engine_by_itself():
    code = "import sys, neuroprix.train, neuroprix.runner, neuroprix.rust_env; print('neuroprix_engine' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.stdout.strip() == "False", r.stderr


def test_trains_on_the_python_engine_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("NEUROPRIX_TRAIN_ENGINE", raising=False)
    said = notes([sys.executable, "-m", "neuroprix.train"], write_session(tmp_path, "n3"))
    assert "engine: Python" in said


def test_asking_for_rust_says_which_engine_it_trains_on(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROPRIX_TRAIN_ENGINE", "rust")
    said = [n for n in notes([sys.executable, "-m", "neuroprix.train"], write_session(tmp_path, "n1")) if n.startswith("engine:")]
    if HAVE_RUST:
        assert said and said[0].startswith("engine: Rust, neuroprix-engine"), said
    else:
        assert said and said[0].startswith("engine: Python (the Rust engine isn't used: neuroprix-engine isn't installed"), said


@needs_rust
def test_rust_is_refused_for_a_changed_engine(monkeypatch):
    monkeypatch.setattr(rust_env, "RELEASED_SHA256", "0" * 64)
    ok, why = rust_env.rust_ready()
    assert not ok and "isn't the released one" in why


@needs_rust
def test_rust_is_refused_for_another_engine_version(monkeypatch):
    monkeypatch.setattr(engine, "ENGINE_VERSION", "2000.1.1")
    ok, why = rust_env.rust_ready()
    assert not ok and "is for engine" in why


@needs_rust
def test_rust_is_ready_for_the_released_engine():
    ok, why = rust_env.rust_ready()
    assert ok, why


@needs_rust
@pytest.mark.parametrize("seed", [1, 2, 3, 77, 5003, 9001, 424242])
@pytest.mark.parametrize("cls", [None, "n1"])
def test_rust_makes_the_same_circuits(seed, cls):
    assert neuroprix_engine.generate_track_spec(seed, cls) == engine.generate_track_spec(seed, cls)


GYM = {"name": "test gym", "width": 90, "grip": 1, "laps": 2,
       "control": [[120, 300], [164, 200], [285, 157], [450, 190], [615, 157], [736, 200], [780, 300], [736, 400],
                   [615, 443], [450, 410], [285, 443], [164, 400]],
       "obstacles": [{"type": "rock", "x": 450, "y": 405, "r": 9}, {"type": "oil", "x": 700, "y": 230, "r": 22}],
       "elev": [0, 10, 30, 55, 70, 60, 40, 25, 10, 0, -10, -8], "bank": [12, 10, 0, 0, 0, 8, 14, 10, 0, 0, -6, 4]}


@needs_rust
@pytest.mark.parametrize("spec,tier,cars,random_start,rolling", [
    ({"seed": 11}, "n4", 4, True, 0.0), ({"seed": 12}, "n3", 4, True, 0.0), ({"seed": 13}, "n2", 6, False, 0.6),
    (engine.generate_track_spec(14, "n1"), "n1", 10, True, 0.5), (GYM, "n1", 4, True, 0.0), (GYM, "n2", 1, False, 0.0)])
def test_rust_drives_exactly_like_python(spec, tier, cars, random_start, rolling):
    # every car's whole inner state and every state the trainer reads, after every step, to the last bit
    assert rust_env.lockstep(spec, tier, cars, random_start, rolling, steps=250) is None


@needs_rust
def test_rust_trains_the_same_driver_as_python(tmp_path, monkeypatch):
    brains = {}
    for name in ("python", "rust"):
        monkeypatch.setenv("NEUROPRIX_TRAIN_ENGINE", name)
        session = write_session(tmp_path / name, "n2", {"updates": 3, "batch": 2, "eval_every": 3, "traffic": 3,
                                                        "max_steps": 80, "check_tracks": 1, "random_tracks": True})
        notes([sys.executable, "-m", "neuroprix.train"], session)
        brains[name] = json.loads((tmp_path / name / "brain.json").read_text())
    assert brains["python"]["fc1"] == brains["rust"]["fc1"] and brains["python"]["fc2"] == brains["rust"]["fc2"]


def test_surprise_tracks_come_from_a_pool_of_built_circuits():
    env, seen = SimEnv(), []
    random.seed(5)
    for k in range(60):
        seen.append(env.pool_seed(10_000 + k, "n3"))
    assert seen[:POOL_SIZE] == [10_000 + k for k in range(POOL_SIZE)]   # (filling the pool: every one new)
    later = seen[POOL_SIZE:]
    reused = sum(1 for i, s in enumerate(later) if s in seen[:POOL_SIZE + i])   # (from the pool, not a new circuit)
    assert len(later) // 2 < reused < len(later)
    assert len(env._pool[None]) == POOL_SIZE

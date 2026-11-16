"""Trainers following the trainer contract (docs/trainer-contract.md): NeuroPrix's own PPO trainer for every class,
continuing a driver, and the examples."""
import json
import subprocess
import sys

import pytest

from neuroprix import engine

from conftest import CLASSES, ROOT, write_session


def train(cmd, session, timeout=300):
    r = subprocess.run([*cmd, "--session", str(session)], capture_output=True, text=True, timeout=timeout,
                       cwd=str(session.parent))
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
    return [json.loads(ln) for ln in r.stdout.splitlines() if ln.startswith("{")]


def check_brain(path, cls):
    data = json.loads(path.read_text())
    assert engine.Brain(data).tier.lower() == cls
    return data


@pytest.mark.parametrize("cls", CLASSES)
def test_default_trainer_every_class(cls, tmp_path):
    session = write_session(tmp_path / cls, cls)
    events = train([sys.executable, "-m", "neuroprix.train"], session)
    kinds = [e["event"] for e in events]
    start = next(e for e in events if e["event"] == "start")
    assert start["tier"] == cls
    assert kinds.count("update") == 2 and "eval" in kinds and kinds[-1] == "done"
    check_brain(tmp_path / cls / "brain.json", cls)


def test_default_trainer_continues_a_driver(tmp_path):
    first = write_session(tmp_path / "a", "n3")
    train([sys.executable, "-m", "neuroprix.train"], first)
    start = tmp_path / "a" / "brain.json"
    again = write_session(tmp_path / "b", "n3", start_brain=start)
    events = train([sys.executable, "-m", "neuroprix.train"], again)
    assert events[-1]["event"] == "done"
    check_brain(tmp_path / "b" / "brain.json", "n3")


def test_default_trainer_on_a_gym(tmp_path):
    """A gym drawn in the Gym Editor (control points), practised both ways."""
    folder = tmp_path / "gym"
    session = write_session(folder, "n4", settings={"updates": 1, "batch": 1, "eval_every": 1, "traffic": 1, "max_steps": 60})
    s = json.loads(session.read_text())
    s["tracks"] = [{"name": "Oval", "control": [[200, 200], [800, 200], [900, 450], [800, 700], [200, 700], [100, 450]],
                    "width": 110, "grip": 1, "laps": 1, "obstacles": []}]
    session.write_text(json.dumps(s))
    train([sys.executable, "-m", "neuroprix.train"], session)
    check_brain(folder / "brain.json", "n4")


@pytest.mark.parametrize("cls", CLASSES)
def test_minimal_trainer_example(cls, tmp_path):
    session = write_session(tmp_path, cls, settings={"updates": 3, "seeds": [101]})
    events = train([sys.executable, str(ROOT / "examples" / "minimal_trainer.py")], session)
    assert events[-1]["event"] == "done"
    check_brain(tmp_path / "brain.json", cls)


def test_custom_rewards_example(tmp_path):
    session = write_session(tmp_path, "n3")
    events = train([sys.executable, str(ROOT / "examples" / "custom_rewards.py")], session)
    assert events[-1]["event"] == "done"
    check_brain(tmp_path / "brain.json", "n3")


@pytest.mark.parametrize("layout", ["combined", "two_head", "three_head", None])
def test_action_layouts(layout, tmp_path):
    """A session's layout setting (docs/brain-format.md#action-layouts); without one, a new N1 brain is three_head."""
    settings = {"updates": 2, "batch": 2, "eval_every": 2, "traffic": 1, "max_steps": 60, "seeds": [101]}
    if layout:
        settings["layout"] = layout
    session = write_session(tmp_path, "n1", settings=settings)
    train([sys.executable, "-m", "neuroprix.train"], session)
    data = check_brain(tmp_path / "brain.json", "n1")
    assert engine.Brain(data).layout == (layout or "three_head")
    assert len(data["fc2"]["weight"]) == {"combined": 12, "two_head": 7, "three_head": 8}[layout or "three_head"]


def test_minimal_trainer_three_heads(tmp_path):
    session = write_session(tmp_path, "n1", settings={"updates": 2, "seeds": [101], "layout": "three_head"})
    train([sys.executable, str(ROOT / "examples" / "minimal_trainer.py")], session)
    assert engine.Brain(check_brain(tmp_path / "brain.json", "n1")).layout == "three_head"


def test_layout_shims():
    import torch

    from neuroprix import brain
    assert brain.LAYOUTS == ("combined", "two_head", "three_head") and brain.DEFAULT_LAYOUT == {"n1": "three_head"}
    assert [brain.layout_outputs("n1", l) for l in brain.LAYOUTS] == [12, 7, 8]
    assert brain.layout_outputs("n3", "three_head") is None and brain.layout_outputs("n4", "two_head") is None
    scores = torch.tensor([0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0])   # straight, gas, boost on
    assert brain.pick_action(scores, "three_head", greedy=True) == engine.heads_action(1, 0, 1) == 10

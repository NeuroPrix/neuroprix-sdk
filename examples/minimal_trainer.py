"""A complete NeuroPrix trainer in about 100 lines, with NumPy only: random search (keep a small random change to the
brain whenever it drives further). Not a strong learner, but it shows the whole trainer contract:

  1. read --session session.json   2. write the best brain to brain_out   3. (optionally) print progress events

    neuroprix-runner --server https://neuroprix.com/api --trainer "python minimal_trainer.py"
"""
import argparse
import json
import os
import time

import numpy as np

from neuroprix.env import SimEnv

SHAPES = {"n4": (18, 3), "n3": (22, 5), "n2": (24, 6), "n1": (31, 7)}   # inputs, brain kind
PEDALS = {"n4": 0, "n3": 3, "n2": 4, "n1": 4}
HIDDEN = 64


def outputs(cls, layout):
    """How many scores a brain of this class has in this action layout (docs/brain-format.md#action-layouts)."""
    p = PEDALS[cls]
    return {"combined": 3 * p or 3, "two_head": 3 + p if p else None, "three_head": 8 if p == 4 else None}[layout]


def act(scores, layout):
    """The car's action from the brain's scores: the best of all (combined), or the best in each head."""
    if layout == "two_head":   # 3 steering, then the pedals
        return int(np.argmax(scores[3:])) * 3 + int(np.argmax(scores[:3]))
    if layout == "three_head":   # 3 steering, 3 pedal (gas, coast, brake), 2 boost (off, on); boost only with gas
        steer, pedal, boost = int(np.argmax(scores[:3])), int(np.argmax(scores[3:6])), int(np.argmax(scores[6:8]))
        return (3 if pedal == 0 and boost == 1 else pedal) * 3 + steer
    return int(np.argmax(scores))


def say(**event):   # a progress event for the Train page (one JSON object per line)
    print(json.dumps(event), flush=True)


def new_brain(n_in, n_out, rng):
    return {"w1": rng.normal(0, 1 / np.sqrt(n_in), (HIDDEN, n_in)), "b1": np.zeros(HIDDEN),
            "w2": rng.normal(0, 0.1, (n_out, HIDDEN)), "b2": np.zeros(n_out)}


def from_json(b):
    return {"w1": np.array(b["fc1"]["weight"]), "b1": np.array(b["fc1"]["bias"]),
            "w2": np.array(b["fc2"]["weight"]), "b2": np.array(b["fc2"]["bias"])}


def to_json(br, name, kind, layout):
    n_in, n_out = br["w1"].shape[1], br["w2"].shape[0]
    return {"name": name, "arch": f"mlp-{n_in}-{HIDDEN}-{n_out}", "generation": kind, "action_layout": layout,
            "fc1": {"weight": br["w1"].tolist(), "bias": br["b1"].tolist()},
            "fc2": {"weight": br["w2"].tolist(), "bias": br["b2"].tolist()}}


def drive(env, br, spec, cls, layout, steps=600):
    """How far the brain gets round the circuit alone (its best action each decision)."""
    env.setup_world(spec, 1, tier=cls)
    s = env.reset_world()[0]
    for _ in range(steps):
        if s["dead"] or s["finished"]:
            break
        x = np.clip(np.array(s["obs"]), 0, 1)
        a = act(br["w2"] @ np.maximum(br["w1"] @ x + br["b1"], 0) + br["b2"], layout)
        s = env.step_world([a])[0]
    return s["distance"] + (5000 if s["finished"] else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    session = json.load(open(ap.parse_args().session))
    assert session["neuroprix_session"] == 1, "unknown session format"
    cls, st = session["class"], session["settings"]
    n_in, kind = SHAPES[cls]
    rng = np.random.default_rng()
    tracks = session["tracks"] or [{"seed": s} for s in st.get("seeds", [])] or [{"seed": 101}, {"seed": 202}]
    env = SimEnv()
    # the action layout: the one the session asks for (always, for a new brain); a starting brain keeps its own here,
    # since this simple trainer doesn't move brains between layouts (neuroprix.brain.relayout does)
    layout = st.get("layout") or "combined"
    if session.get("start_brain"):
        start = json.load(open(session["start_brain"]))
        br, layout = from_json(start), start.get("action_layout") or "combined"
    else:
        br = new_brain(n_in, outputs(cls, layout), rng)
    best = np.mean([drive(env, br, t, cls, layout) for t in tracks])
    t0 = time.time()
    say(event="stage", message="Random search: keeps each small change to the brain that drives further")
    for update in range(1, int(st.get("updates", 150)) + 1):
        trial = {k: v + rng.normal(0, 0.02, v.shape) for k, v in br.items()}
        score = np.mean([drive(env, trial, t, cls, layout) for t in tracks])
        if score > best:
            br, best = trial, score
            tmp = session["brain_out"] + ".tmp"   # write, then rename: the runner never reads half a file
            with open(tmp, "w") as f:
                json.dump(to_json(br, session["driver"], kind, layout), f)
            os.replace(tmp, session["brain_out"])
        say(event="update", update=update, avg_return=round(float(best), 1), avg_distance=round(float(best), 1),
            car_contacts=0, elapsed=round(time.time() - t0, 1))
    if not os.path.exists(session["brain_out"]):   # never improved: hand back what it started with
        json.dump(to_json(br, session["driver"], kind, layout), open(session["brain_out"], "w"))
    say(event="done")


if __name__ == "__main__":
    main()

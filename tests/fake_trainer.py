"""A fast stand-in trainer for the runner's tests: follows the trainer contract (reads --session, writes a brain of the
session's class to brain_out, prints progress events) without learning anything.

    python fake_trainer.py --session session.json [--steps N] [--delay S] [--fail]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

SHAPES = {"n4": (18, 3, 3), "n3": (22, 9, 5), "n2": (24, 12, 6), "n1": (31, 12, 7)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--delay", type=float, default=0.1)
    ap.add_argument("--fail", action="store_true", help="exit with an error, without writing a brain")
    ap.add_argument("--bad-brain", action="store_true", help="write a brain of the wrong shape")
    args = ap.parse_args()
    s = json.load(open(args.session))
    if args.fail:
        print("something went wrong in the trainer", flush=True)
        sys.exit(3)
    print(json.dumps({"event": "note", "message": f"start_weights={s.get('start_weights')}"}), flush=True)
    n_in, n_out, gen = SHAPES[s["class"]]
    if args.bad_brain:
        n_in += 1
    rng = np.random.default_rng(0)
    for update in range(1, args.steps + 1):
        brain = {"name": s["driver"], "arch": f"mlp-{n_in}-64-{n_out}", "generation": gen,
                 "fc1": {"weight": rng.normal(0, 0.1, (64, n_in)).tolist(), "bias": [0.0] * 64},
                 "fc2": {"weight": rng.normal(0, 0.1, (n_out, 64)).tolist(), "bias": [0.0] * n_out}}
        tmp = s["brain_out"] + ".tmp"
        with open(tmp, "w") as f:
            json.dump(brain, f)
        os.replace(tmp, s["brain_out"])
        print(json.dumps({"event": "update", "update": update, "avg_return": float(update), "avg_distance": 0.0,
                          "car_contacts": 0, "elapsed": 0.0}), flush=True)
        time.sleep(args.delay)
    print(json.dumps({"event": "done"}), flush=True)


if __name__ == "__main__":
    main()

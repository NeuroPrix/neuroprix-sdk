"""The training environment in Python: the race engine (backend/sim.py), driven like the browser game in agent/env.py.

Same interface as RaceEnv (setup_world / reset_world / step_world) and the same state for each car, but no browser:
the game, the brain and the learning all run in one program, with no message to a page on every decision. About 2.8x
faster per worker, and much lighter, so more workers fit. Drivers also train in the engine their races run in.
The browser game (agent/env.py) stays for checking the two engines agree (parity tests) and for experiments.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

from neuroprix import engine as sim

GEN_OF_TIER = {"n4": 16, "n3": 17, "n2": 18, "n1": 19}   # each class's brain kind for new brains: the senses a real car could measure
KINDS = (16, 17, 18, 19)   # brain kinds a training car can have (docs/observations-and-actions.md)
MAX_CARS = 10                                         # a training field: up to a full race grid


class _NoBrain:
    """The trainer chooses the actions; the car only needs to know which kind of brain it serves."""

    def __init__(self, gen: int):
        self.gen = gen

    def act(self, obs):   # never called: the world steps without deciding
        raise RuntimeError("training cars are driven by the trainer")


def _clamp(v: float, a: float, b: float) -> float:
    return max(a, min(b, v))


class SimEnv:
    """A field of cars on one track, stepped by the trainer, in the Python engine."""

    def __init__(self, tracks: list[dict] | None = None, headless: bool = True, random_tracks: bool = False):
        self.tracks = tracks or [{"seed": 1}]
        self.random_tracks = random_tracks
        self.track = None
        self.world = None
        self.random_start = False
        self.track_info: dict = {}
        self._specs: dict = {}   # recent circuits, built once (building one is the slowest part of a reset). A Grand
        # Prix circuit's track map is about 15 MB, so only a few are kept: gyms (few, reused) and the last 2 random ones

    def _build(self, spec: dict, cls: str | None = None):
        gym = bool(spec and spec.get("control"))
        key = ("gym", repr(spec)) if gym else ("seed", spec.get("seed", 1) if spec else 1, cls)
        T = self._specs.get(key)
        if T is None:   # (a circuit's number gives the class's version of it: N1's has hills)
            T = sim.build_track(spec if gym else sim.generate_track_spec(key[1], cls))
            same = [k for k in self._specs if k[0] == key[0]]
            if len(same) >= (8 if gym else 2):
                self._specs.pop(same[0])
            self._specs[key] = T
        return T

    def setup_world(self, spec: dict, cars: int = 1, random_start: bool = False, tier: str = "n3", speed_cap: float = 1.0,
                    fragile: bool = False, caps: list[float] | None = None, gen: int | None = None,
                    gens: list[int] | None = None, rolling: float = 0.0) -> dict:
        """gen: the brain kind of every car (default: the class's current one); gens: one kind per car instead (a field
        of mixed kinds: league opponents of an earlier generation). rolling: a rolling start, every car already moving
        at this share of top speed (0 = standing)."""
        T = self.track = self._build(spec, "n1" if str(tier).lower() == "n1" else None)
        self.world = sim.World(T)
        self.random_start = bool(random_start)
        self.rolling = _clamp(float(rolling or 0), 0.0, 1.0)
        n = int(_clamp(int(cars or 1), 1, MAX_CARS))
        g = gen if gen in KINDS else GEN_OF_TIER.get(str(tier).lower() if str(tier).lower() == "n1" else sim.norm_class(tier, "n3"), 5)
        gens = gens or []
        caps = caps or []
        self._cars = []
        for i in range(n):
            gi = gens[i] if i < len(gens) and gens[i] in KINDS else g
            c = sim.Car(track=T, brain=_NoBrain(gi), name=str(i))
            c.training, c.touched = True, {}
            c.cap_k = _clamp(float(caps[i]) if i < len(caps) and caps[i] else 1.0, 0.3, 1.0)
            c.world_cap = _clamp(float(speed_cap or 1.0), 0.2, 1.0)
            c.fragile = bool(fragile)
            self._cars.append(c)
            self.world.cars.append(c)
        return {"name": T.spec.get("name", ""), "length": round(T.L), "width": T.width, "laps": T.laps,
                "obstacles": len(T.spec.get("obstacles", [])), "cars": n, "people": 0}

    def _place(self, c, x: float, y: float, a: float) -> None:
        """Put a car at a pose and set its progress from there (as Car.reset does for the start line)."""
        T = c.track
        c.x, c.y, c.angle, c.dir = x, y, a, a
        c.seg = c.nearest_seg(x, y)
        s0 = c.project_s(x, y)
        c.prev_s = s0
        c.distance = s0 - T.L if s0 > T.L / 2 else s0
        c.max_distance = c.distance

    def reset_world(self) -> list[dict]:
        T, W, cars = self.track, self.world, self._cars
        multi = len(cars) > 1
        start = None
        if self.random_start:   # training: the grid somewhere random round the circuit
            rs = random.random() * T.L
            x, y, nx, ny = sim.point_at(T.pts, T.S, T.L, rs)
            start = (x, y, math.atan2(-nx, ny), rs)   # heading along the circuit: (nx, ny) is the normal
        for i, c in enumerate(cars):
            c.reset(i if multi else None)
            if start is not None:
                if multi:
                    gx, gy, ga = sim.grid_slot(T, i, start[3])
                    self._place(c, gx, gy, ga)
                else:
                    self._place(c, start[0], start[1], start[2])
            c.risk = c.impact_risk = 0.0
            c.touched = {}
            if self.rolling > 0:   # a rolling start: already moving, straight ahead
                c.speed = self.rolling * sim.MAX_SPEED * c.cap_k
                c.vx, c.vy, c.dir = math.cos(c.angle) * c.speed, math.sin(c.angle) * c.speed, c.angle
        W.frame_no, W.touch = 0, {}
        for c in cars:
            c.sense(W)
        return [self._state(c) for c in cars]

    def step_world(self, actions: list[int]) -> list[dict]:
        W = self.world
        for i, c in enumerate(self._cars):
            c.hits = c.contacts = 0
            c.damage = c.impact = c.impact_risk = 0.0
            c.touched.clear()
            if not (c.dead or c.finished):
                top = 11 if sim.is_f1(c.gen) else 8 if sim.has_pedals(c.gen) else 2
                c.action = int(_clamp(int(actions[i]) if i < len(actions) else 0, 0, top))
        # Sensors only after the last frame: that's when the trainer reads them. Except a car about to finish: one that
        # takes the flag mid-step keeps the readings from the frame before (as in the browser), so it reads every frame.
        goal = self.track.laps * self.track.L - 30
        for f in range(sim.ACTION_REPEAT):
            last = f == sim.ACTION_REPEAT - 1
            W.frame(decide=False, sense=last)
            if not last:
                for c in self._cars:
                    if not c.finished and c.distance >= goal:
                        c.sense(W)
        return [self._state(c) for c in self._cars]

    def _state(self, c) -> dict:
        """The same fields as Car.state() in the browser."""
        T, r4 = c.track, sim.r4
        ca, sa, hl, hw = math.cos(c.angle), math.sin(c.angle), sim.CAR_L / 2, sim.CAR_W / 2
        box = [[round(c.x + lx * ca - ly * sa, 1), round(c.y + lx * sa + ly * ca, 1)]
               for lx, ly in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
        st = {
            "sensors": list(c.sensors),
            "speed": r4(c.speed / sim.MAX_SPEED),
            "dead": c.dead, "finished": c.finished,
            "score": int(max(0.0, c.max_distance) // 10),
            "distance": round(c.max_distance, 1),
            "laps": max(0, int(c.max_distance // T.L)) if T.L else 0,
            "lap_progress": round(((c.max_distance % T.L) + T.L) % T.L / T.L, 3) if T.L else 0,
            "frames": c.frames, "time": round(c.frames / 60, 2), "reason": c.reason,
            "x": round(c.x, 1), "y": round(c.y, 1), "angle": round(c.angle, 3), "steer": round(c.steer, 3),
            "box": box,
            "health": r4(c.health), "damage": r4(c.damage), "hits": c.hits,
            "impact": r4(c.impact),
            "touched": [int(k) for k in c.touched],
            "heading": c.heading(),   # (training only: which way the car points against the road; the turnround reward)
            "distance_now": round(c.distance, 1),   # (training only: where it is now; "distance" is its furthest so far)
        }
        st["obs"] = c.obs()
        st["contacts"] = c.contacts
        if not (c.dead or c.finished):
            st["corner_over"] = r4(c.corner_over())
            st["corner_need"] = r4(min(99.0, c.corner_need()))
            st["impact_risk"] = r4(c.impact_risk)
        if sim.is_f1(c.gen):
            st["battery"], st["boosting"] = r4(c.battery), c.boosting
        if sim.has_pedals(c.gen):
            st["braking"] = c.braking
        return st

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

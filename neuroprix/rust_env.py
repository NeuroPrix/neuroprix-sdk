"""Training on the optional Rust race engine (the neuroprix-engine package), several times faster than the Python one.

The Python engine (backend/sim.py) is the reference: the one to read and tweak. The Rust engine is a port of it that
gives exactly the same numbers, step by step, and RustEnv drives it with the same interface and the same states as
SimEnv (agent/simenv.py). Nothing here is needed to train: without neuroprix-engine, training runs on the Python engine.

When the Rust engine is used (rust_ready), which is a safe rule for players who change the Python engine:
  1. only when asked: the trainer's --engine rust, or NEUROPRIX_TRAIN_ENGINE=rust (never on its own);
  2. only if neuroprix-engine is installed and made for this engine version (ENGINE_VERSION);
  3. only if the Python engine is the one released with this package, unchanged (its SHA-256): a player's own physics
     are never swapped for the Rust engine's;
  4. only if a short lockstep check, the two engines side by side on two circuits, finds them exactly the same.
Otherwise training says why and runs on the Python engine. Physics changes go into the Python engine first, then into
the Rust one (and a new neuroprix-engine release).
"""
from __future__ import annotations

import hashlib
import math
import random
import sys
from pathlib import Path

from neuroprix.env import GEN_OF_TIER, TRAIN_GRIP, SimEnv, _clamp, sim

from neuroprix import ENGINE_SHA256 as RELEASED_SHA256   # (the released engine.py's SHA-256, recorded by its release)
BOX = ((sim.CAR_L / 2, sim.CAR_W / 2), (sim.CAR_L / 2, -sim.CAR_W / 2), (-sim.CAR_L / 2, -sim.CAR_W / 2), (-sim.CAR_L / 2, sim.CAR_W / 2))


def _module():
    import neuroprix_engine   # (optional: pip install neuroprix-engine)
    neuroprix_engine.set_python(*sys.version_info[:2])   # (math.hypot and sum() changed in Python 3.12: it does as this one)
    return neuroprix_engine


class _Track:
    """A track the Rust engine built, with what SimEnv reads of one."""
    __slots__ = ("rust", "spec", "L", "width", "laps")

    def __init__(self, rust, spec: dict):
        self.rust, self.spec, self.L, self.width, self.laps = rust, spec, rust.L, rust.width, rust.laps


class RustEnv(SimEnv):
    """SimEnv on the Rust engine: the same interface, the same states, the same random numbers drawn."""

    def __init__(self, tracks: list[dict] | None = None, headless: bool = True, random_tracks: bool = False):
        super().__init__(tracks, headless, random_tracks)
        self._ne = _module()
        self._gens: list[int] = []

    def _generate(self, seed: int, cls: str | None):
        return self._ne.generate_track_spec(int(seed), cls)

    def _new_track(self, spec: dict):
        W, H, control, width, grip, laps, rocks, oils, elev, bank = sim.read_spec(spec)
        ob = [[(float(o["x"]), float(o["y"]), float(o["r"])) for o in group] for group in (rocks, oils)]
        return _Track(self._ne.RustTrack(W, H, [(float(p[0]), float(p[1])) for p in control], width, grip, laps,
                                         ob[0], ob[1], elev, bank), spec)

    def setup_world(self, spec: dict, cars: int = 1, random_start: bool = False, tier: str = "n3", speed_cap: float = 1.0,
                    fragile: bool = False, caps: list[float] | None = None, gen: int | None = None,
                    gens: list[int] | None = None, rolling: float = 0.0, race_grip: bool = False) -> dict:
        T = self.track = self._build(spec, "n1" if str(tier).lower() == "n1" else None)
        self.random_start = bool(random_start)
        self.race_grip = bool(race_grip)
        self.rolling = _clamp(float(rolling or 0), 0.0, 1.0)
        self._gens, caps = self._field(cars, tier, gen, gens, caps)
        self.world = self._ne.RustWorld(T.rust, self._gens, caps, _clamp(float(speed_cap or 1.0), 0.2, 1.0), bool(fragile))
        return self._info(T, len(self._gens))

    def reset_world(self) -> list[dict]:
        start = random.random() if self.random_start else None   # (the same draws, in the same order, as SimEnv)
        grip = round(random.uniform(*TRAIN_GRIP), 3) if self.race_grip else None
        self.world.reset(start, float(self.rolling), grip)
        return self._states()

    def spin(self, i: int, turn: float, room: float | None = None) -> None:
        self.world.spin(i, float(turn), float(room) if room else None)

    def sense_all(self) -> list[dict]:
        self.world.sense_all()
        return self._states()

    def step_world(self, actions: list[int]) -> list[dict]:
        self.world.step([int(a) for a in actions])
        return self._states()

    def _states(self) -> list[dict]:
        """The same fields, rounded the same way, as SimEnv._state."""
        L, out = self.track.L, []
        for gen, (pose, (frames, reason, dead, finished), sensors, obs, (speed, health, damage, hits, impact, contacts),
                  touched, heading, corner, (battery, boosting, braking)) in zip(self._gens, self.world.states()):
            x, y, angle, steer, best, now = pose
            ca, sa = math.cos(angle), math.sin(angle)
            st = {
                "sensors": sensors, "speed": speed, "dead": dead, "finished": finished,
                "score": int(max(0.0, best) // 10),
                "distance": round(best, 1),
                "laps": max(0, int(best // L)) if L else 0,
                "lap_progress": round(((best % L) + L) % L / L, 3) if L else 0,
                "frames": frames, "time": round(frames / 60, 2), "reason": reason,
                "x": round(x, 1), "y": round(y, 1), "angle": round(angle, 3), "steer": round(steer, 3),
                "box": [[round(x + lx * ca - ly * sa, 1), round(y + lx * sa + ly * ca, 1)] for lx, ly in BOX],
                "health": health, "damage": damage, "hits": hits, "impact": impact, "touched": touched,
                "heading": heading, "distance_now": round(now, 1),
            }
            st["obs"] = obs
            st["contacts"] = contacts
            if corner is not None:
                st["corner_over"], st["corner_need"], st["impact_risk"] = corner
            if sim.is_f1(gen):
                st["battery"], st["boosting"] = battery, boosting
            if sim.has_pedals(gen):
                st["braking"] = braking
            out.append(st)
        return out


# ---------- whether to use it ----------
# Every number a car carries from frame to frame (compared exactly: the states the trainer reads are rounded)
RAW = ("x", "y", "angle", "speed", "vx", "vy", "steer", "distance", "max_distance", "prev_s", "health", "damage", "grip_k",
       "slip", "dir", "impact", "load", "yaw", "risk", "impact_risk", "wrong_way_px", "battery", "yaw_r", "_ax", "_thr",
       "_brk", "frames", "last_improve", "seg", "action", "hits", "contacts", "stun", "hold", "dead", "finished",
       "boosting", "braking", "reason", "sensors", "radar")


def same(a, b) -> bool:
    """Exactly equal, the sign of a zero included (it can flip a later atan2)."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(p, q) for p, q in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return float(a) == float(b) and (a != 0 or math.copysign(1, a) == math.copysign(1, b))
    return a == b


def raw_differs(py_car, rust_raw: dict) -> list[str]:
    """Which of a Python car's inner numbers differ from its Rust twin's (RustWorld.raw)."""
    return [k for k in RAW if not same(list(getattr(py_car, k)) if k in ("sensors", "radar") else getattr(py_car, k), rust_raw[k])]


CHECK_RUNS = (({"seed": 4242}, "n2", 4, True, 0.0), ({"seed": 777}, "n1", 4, False, 0.6))   # (spec, class, cars, random start, rolling)
CHECK_STEPS = 90


def lockstep(spec: dict, tier: str, cars: int, random_start: bool, rolling: float, steps: int, seed: int = 7,
             python: SimEnv | None = None, rust: RustEnv | None = None):
    """Both engines side by side from the same start with the same moves (mostly throttle, half of them proportional).
    Returns None if every state was the same at every step, else (step, car, the fields that differ)."""
    py, rs = python or SimEnv(), rust or RustEnv()
    gen = GEN_OF_TIER[tier]
    caps = ([1, 0.85] + [1] * 8)[:cars]
    keep = random.getstate()   # (the engines draw the start and the grip from random: the caller's numbers are put back)
    try:
        states = []
        for env in (py, rs):
            random.seed(seed)
            env.setup_world(spec, cars, random_start=random_start, tier=tier, caps=caps, gen=gen, rolling=rolling, race_grip=True)
            states.append(env.reset_world())
    finally:
        random.setstate(keep)
    rng = random.Random(seed)
    pedals = sim.has_pedals(gen)
    for step in range(steps + 1):
        for i, (a, b) in enumerate(zip(*states)):
            wrong = raw_differs(py._cars[i], rs.world.raw(i)) or sorted(k for k in set(a) | set(b) if not same(a.get(k), b.get(k)))
            if wrong:
                return step, i, wrong
        if all(s["dead"] or s["finished"] for s in states[0]):
            break
        acts = [(sim.PROP_BASE + rng.randrange(sim.PROP_STEER) + sim.PROP_STEER * rng.choice([4, 4, 3, 2, 1, 0])
                 + sim.PROP_STEER * sim.PROP_PEDAL * rng.randrange(2) if rng.random() < 0.5 else rng.choice([0, 0, 1, 2, 3, 6, 9]))
                if pedals else rng.choice([1, 1, 0, 2]) for _ in states[0]]
        states = [py.step_world(acts), rs.step_world(acts)]
    return None


def rust_ready() -> tuple[bool, str]:
    """Whether training may use the Rust engine, and what to tell the player (see the module's notes for the rule)."""
    try:
        ne = _module()
    except ImportError:
        return False, "neuroprix-engine isn't installed (pip install neuroprix-engine)"
    except Exception as e:   # noqa: BLE001 - a broken install: say what it was
        return False, f"neuroprix-engine didn't load ({type(e).__name__}: {e})"
    label = f"neuroprix-engine {getattr(ne, '__version__', '?')}"
    if ne.ENGINE_VERSION != sim.ENGINE_VERSION:
        return False, (f"{label} is for engine {ne.ENGINE_VERSION} and this Python engine is {sim.ENGINE_VERSION} "
                       "(install the neuroprix-engine made for it)")
    if RELEASED_SHA256:
        mine = hashlib.sha256(Path(sim.__file__).read_bytes()).hexdigest()
        if mine != RELEASED_SHA256:
            return False, (f"{Path(sim.__file__).name} isn't the released one (changed by you?): your Python engine "
                           "is used, so your physics are what you train on")
    rng = random.Random(3)
    for _ in range(2000):   # the two maths routines Python does its own way
        x, y = rng.uniform(-900, 900), rng.uniform(-900, 900)
        if ne.hypot(x, y) != math.hypot(x, y):
            return False, f"{label} doesn't compute math.hypot as this Python does"
    xs = [rng.uniform(-1, 1) * 10 ** rng.randrange(-3, 4) for _ in range(40)]
    if ne.fsum(xs) != sum(xs):
        return False, f"{label} doesn't add up floats as this Python does"
    for seed in (5, 2024):
        for cls in (None, "n1"):
            if ne.generate_track_spec(seed, cls) != sim.generate_track_spec(seed, cls):
                return False, f"{label} makes circuit #{seed}{' (N1)' if cls else ''} differently from the Python engine"
    for spec, tier, cars, rnd, rolling in CHECK_RUNS:
        diff = lockstep(spec, tier, cars, rnd, rolling, CHECK_STEPS)
        if diff:
            return False, (f"{label} drove differently from the Python engine (circuit #{spec['seed']}, {tier.upper()}, "
                           f"step {diff[0]}, car {diff[1]}: {', '.join(diff[2][:4])})")
    return True, f"{label} (engine {ne.ENGINE_VERSION}), checked step by step against the Python engine"

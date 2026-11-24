"""Train a generation-2 RaceBrain with PPO (actor-critic) on your own training gyms.

What you train on decides how you race. Each episode puts a field of cars on a gym, all driven by
the driver being trained (self-play); nobody else is ever on the circuit. The driver chooses its
moves every 4 frames from its wall rays, its radar (the nearest other cars), its speed and (N3) its
steering wheel. Racing without touching the other cars is learned from the rewards below; there is
no hand-written avoidance anywhere in the game.

The learner (PPO):
  - a critic network learns how good each moment is, and each action is judged against that
    (GAE advantages), which is far steadier than judging it by the whole episode's return;
  - each batch of experience is reused for a few small, clipped policy steps;
  - several copies of the game run in parallel worker processes (--workers) to collect experience.
Tiers (--tier): n4 = steering only with automatic throttle (3 choices, fast to train);
n3 = steering and pedals (two separate choices while training, exported as one 9-output brain).
The critic only helps training; the exported brain is the policy alone.

Examples
    python agent/train.py                                   # procedural circuit seed 1
    python agent/train.py --track my_gym.json               # a track from the Gym Editor
    python agent/train.py --random-tracks --workers 8       # a new circuit every episode, 8 games at once
    python agent/train.py --track my_gym.json --traffic 1   # alone on the track
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from neuroprix.browser_env import RaceEnv, load_track_spec
from neuroprix.env import SimEnv, sim   # (sim: the race engine)
from neuroprix.rust_env import RustEnv, rust_ready   # (the optional Rust engine)
from neuroprix.brain import PROP_BASE, PROP_PEDAL, PROP_STEER, DEFAULT_LAYOUT, OLD_INPUTS, OLD_SENSES, GEN_OF_INPUTS, GEN_OF_SHAPE, LAYOUTS, N_HIDDEN, PEDALS, SENSES, TIER_OF_GEN, TIERS, RaceBrain, gen_of, generation, layout_heads, layout_move, layout_outputs, load_layout, load_weights, pick_action, relayout, save_brain, state_to_obs  # noqa: E402

# Rewards (per step of 4 frames). Further along the track is a better episode.
# A step with no new best distance pays R_IDLE. A step that advances pays for every
# new pixel, which is more than idling. Leaving the track still costs R_DEAD, and
# also the idle for the steps that crash skipped: dying early is not a way out of
# the clock, so the score improves by getting further before the episode ends.
R_PROGRESS = 0.5       # per px of new best distance (a 2 px creep already beats one idle step)
R_ALIVE = 0.1          # still racing and making progress
R_FAST = 0.5           # going fast (speed >= FAST_THRESHOLD) while making progress
FAST_THRESHOLD = 0.8
R_IDLE = -1.0          # no new forward progress this step (never launched, coasting, or waiting)
R_DEAD = -30.0         # wrecked, stalled or went the wrong way
R_DAMAGE = -40.0       # per unit of health lost. Walls and other cars no longer end the race; they cost health
                       # by how hard the hit was (a head-on hit at top speed loses 0.7, a glancing scrape a few hundredths)
R_FINISH = 10.0        # completed all laps
R_CONTACT = -6.0       # touched another car (each new touch)
R_WALL = 0.0           # per physics frame spent touching a wall, however gently (set per session; see training settings)
R_OVERTAKE = 15.0      # a clean pass: moving ahead of a nearby car without touching it in the last 2 s. Losing a
                       # place costs the same, so swapping places back and forth with its own copies never pays.
PASS_CLEAN_STEPS = 30  # "clean": no touch between the two cars in this many decisions (2 s)
PASS_NEAR_PX = 120     # a pass counts only between cars this close (not a car lapping far behind, or a reset)
SLOW_TRAFFIC = 1       # traffic cars per episode held to a lower top speed, there to be overtaken (they don't learn)
SLOW_SPEED = 0.85      # their top speed, as a share of the real one (each episode +-0.05)
R_CORNER = -15.0       # corner discipline: per decision, x10 x how much too fast the car is for the corners coming up
                       # (0.1 = 10% too fast: -15). -20 made the cleanest, fastest test driver; -4 left it rough. Teaches braking before a corner instead of running wide into the barrier.
R_RISK = 1.0           # risk multiplier: a crash penalty grows by this x 10 x how far over the safe speed the car was just before
                       # (10% over: double; 30% over: 4x). A crash after a reckless approach costs far more than an unlucky one.
R_CRASH = -60.0        # per hit, x (impact strength)^2: -60 for square-on at top speed, about -15 for a 30-degree glance at
                       # top speed, almost nothing for a scrape. On top of the damage penalty: reckless driving costs.
R_LOSE_PLACE = -15.0   # being passed (cleanly or not): make it cost more than a pass earns and drivers learn to defend
R_STEER_FLIP = -1.0    # reversing the steering choice between decisions (left <-> right); half for left/right <-> straight.
                       # Without it drivers learn to weave (left, right, left...) down the track and clip the walls.


# The reward weights a session uses (players can change them per session; see backend/training_params.py).
DEFAULT_REWARDS = {"progress": R_PROGRESS, "alive": R_ALIVE, "fast": R_FAST, "fast_threshold": FAST_THRESHOLD,
                   "idle": R_IDLE, "dead": R_DEAD, "damage": R_DAMAGE, "finish": R_FINISH, "contact": R_CONTACT,
                   "steer_flip": R_STEER_FLIP, "wall": R_WALL, "overtake": R_OVERTAKE,
                   "corner": R_CORNER, "lose_place": R_LOSE_PLACE, "crash": R_CRASH, "risk": R_RISK, "turnround": 0.0,
                   "backwards": 0.0, "stand_wrong": 0.0, "slide": 0.0}
# (turnround and slide: off by default, and session settings since the 0.3.5 trainer; tested in
# docs/observations-and-actions.md. backwards and stand_wrong: experiments only, not session settings)
SLIDE_FREE = 15 / 90   # the slide penalty's free share of the sideslip: sliding up to 15 degrees off the nose costs nothing


def sideslip_of(st: dict) -> float:
    """How far the car travels off where its nose points, 0 (along it) to 1 (90 degrees or more), read from its senses:
    the sideslip input comes after the fan, the radar, speed, the wheel (cars with pedals) and the yaw rate."""
    o = st.get("obs")
    if not o or "sensors" not in st:
        return 0.0
    i = len(st["sensors"]) + sim.SECTORS * len(sim.EMPTY_SECTOR) + (3 if "braking" in st else 2)
    return abs(o[i] - 0.5) * 2 if i < len(o) else 0.0


def _idle_clock(steps_left: int, gamma: float, idle: float = R_IDLE) -> float:
    """Discounted idle for steps the car will not drive. Matches summing the idle reward over that tail."""
    if steps_left <= 0:
        return 0.0
    if gamma >= 1.0:
        return idle * steps_left
    return idle * (1.0 - gamma ** steps_left) / (1.0 - gamma)


def reward_for(prev: dict, cur: dict, rw: dict | None = None, steps_left: int = 0, gamma: float = 0.99) -> float:
    """One step's reward. rw: the session's reward weights (DEFAULT_REWARDS when not given)."""
    rw = rw or DEFAULT_REWARDS
    r = rw["contact"] * cur.get("contacts", 0) + rw["damage"] * cur.get("damage", 0) + rw.get("wall", 0) * cur.get("hits", 0)
    r += rw.get("crash", 0) * cur.get("impact", 0) ** 2 * (1 + rw.get("risk", 0) * 10 * cur.get("impact_risk", 0))   # hard hits cost
    # far more than scrapes, and a hit after knowingly going too fast for the corner costs more again
    if cur["dead"]:
        return r + rw["dead"] + _idle_clock(steps_left, gamma, rw["idle"])
    r += rw.get("corner", 0) * 10 * cur.get("corner_over", 0)   # arriving at a corner too fast to make it
    if rw.get("slide"):   # sliding more than 15 degrees off the nose (a spin starts as a slide)
        r += rw["slide"] * max(0.0, sideslip_of(cur) - SLIDE_FREE) / (1 - SLIDE_FREE)
    if rw.get("turnround") and "heading" in prev and "heading" in cur:   # a car turned off the road's direction turning back
        off0, off1 = abs(prev["heading"] - 0.5), abs(cur["heading"] - 0.5)   # (0 along the road, 0.5 backwards)
        if off0 > 30 / 360:   # pays for turning back towards the road's direction, costs the same for turning away
            r += rw["turnround"] * 4 * (off0 - off1)
    gained = cur["distance"] - prev["distance"]
    if gained > 0:
        r += rw["alive"] + rw["progress"] * gained
        if cur["speed"] >= rw["fast_threshold"]:
            r += rw["fast"]
    else:
        r += rw["idle"]
    if rw.get("stand_wrong") and "heading" in cur and abs(cur["heading"] - 0.5) > 0.25 and cur["speed"] < 0.1:
        r += rw["stand_wrong"]   # experiments: standing still facing more than 90 degrees off the road's direction
    if rw.get("backwards") and "distance_now" in prev and "distance_now" in cur:   # experiments: ground lost costs
        lost = cur["distance_now"] - prev["distance_now"]   # what ground gained pays (x this); "distance" is the furthest
        if lost < 0:                                        # point so far, which never goes down
            r += rw["backwards"] * rw["progress"] * lost
    if cur["finished"]:
        r += rw["finish"]
    return r


PEDAL_EXPLORE = {"n3": 1.5, "n2": 2.0, "n1": 2.0}   # curiosity (entropy bonus) multiplier on the pedal choice, per class: keeps the
                                          # brake (and N2's boost) being tried until the driver learns where it pays


class TrainPolicy(nn.Module):
    """The policy while training: one or more heads, each its own softmax, sampled independently, their log-probs and
    entropies summed. layout (the brain's action layout, docs/brain-format.md):
      None (today's default): N4 one head (steering); N3 / N2 / N1 two heads, steering and pedal (N2 / N1: Boost as a
        4th pedal), exported as a "combined" brain whose outputs are steer + pedal logits summed (exactly the two
        heads' distribution, at any temperature);
      "combined": one head over every pedal x steering pair, exported as it is;
      "two_head": steering and pedal heads, exported as they are (3 + pedals outputs);
      "three_head": steering, pedal (gas / coast / brake) and boost (off / on) heads (N2 / N1), exported as they are;
      "combined_prop", "two_head_prop", "three_head_prop": the same builds by degrees (7 steering levels, 5 pedal
        levels, boost: agent/model.py layout_heads), exported as they are.
    A training action is the heads' choices in mixed radix (steer + 3 x pedal [+ 9 x boost]); env_action turns it into
    the car's action."""

    def __init__(self, tier: str, senses: str = "real", layout: str | None = None):
        super().__init__()
        self.tier = tier
        self.layout = layout or "combined"   # (what the exported brain declares)
        self.pedals = PEDALS.get(tier, 0)
        n_in = SENSES[senses][tier][0]   # (the senses a real car could measure: docs/observations-and-actions.md)
        n_out = layout_outputs(tier, self.layout)
        if n_out is None:
            raise ValueError(f"the {self.layout} action layout isn't available in {tier.upper()}")
        self.shape, self.gen = (n_in, n_out), gen_of(n_in, n_out, self.layout)
        self.sizes = [3, self.pedals] if layout is None and self.pedals else layout_heads(tier, self.layout)
        self.summed = layout is None and len(self.sizes) == 2   # (two heads exported as one combined layer)
        self.split = len(self.sizes) > 1   # (several heads)
        self.fc1 = nn.Linear(n_in, N_HIDDEN)
        self.head = nn.Linear(N_HIDDEN, sum(self.sizes))

    def forward(self, x):
        return self.head(torch.relu(self.fc1(x)))

    def dists(self, x):
        z, out, a = self(x), [], 0
        for n in self.sizes:
            out.append(Categorical(logits=z[..., a:a + n], validate_args=False))   # (no argument checks: costly, and they always hold)
            a += n
        return out

    def _parts(self, a):
        """A training action's choice in each head."""
        out, base = [], 1
        for n in self.sizes:
            out.append((a // base) % n)
            base *= n
        return out

    def sample(self, x):
        d = self.dists(x)
        picks = [h.sample() for h in d]
        a, base = 0, 1
        for p, n in zip(picks, self.sizes):
            a = a + p * base
            base *= n
        return a, sum(h.log_prob(p) for h, p in zip(d, picks))

    def env_action(self, a: int) -> int:
        """The car's action for a training action (each head's choice, as the exported brain would map it)."""
        return layout_move(self.layout, [int(p) for p in self._parts(int(a))])

    def log_prob_entropy(self, x, a):
        d = self.dists(x)
        # the pedal (and boost) heads get extra curiosity (N2 / N1: twice), so braking and Boost keep being tried
        # until the driver learns where they pay
        k = PEDAL_EXPLORE.get(self.tier, 1.0)
        lp = sum(h.log_prob(p) for h, p in zip(d, self._parts(a)))
        ent = sum((1.0 if i == 0 else k) * h.entropy() for i, h in enumerate(d))
        return lp, ent

    def export(self) -> RaceBrain:
        n_in, n_out = self.shape
        rb = RaceBrain(n_in, N_HIDDEN, n_out, self.layout)
        with torch.no_grad():
            rb.fc1.load_state_dict(self.fc1.state_dict())
            W, b = self.head.weight, self.head.bias
            if self.summed:   # action = pedal * 3 + steer: the two heads' logits summed
                rb.fc2.weight.copy_(torch.stack([W[s] + W[3 + p] for p in range(self.pedals) for s in range(3)]))
                rb.fc2.bias.copy_(torch.stack([b[s] + b[3 + p] for p in range(self.pedals) for s in range(3)]))
            else:   # (the outputs are the heads, in order)
                rb.fc2.load_state_dict(self.head.state_dict())
        return rb

    def load_brain(self, sd: dict) -> None:
        """Start from an exported brain in this policy's layout. Today's default (two heads, combined): split its 9 or
        12 outputs back into steer + pedal parts (exact for brains this trainer made, the closest such split otherwise)."""
        with torch.no_grad():
            self.fc1.weight.copy_(sd["fc1.weight"]); self.fc1.bias.copy_(sd["fc1.bias"])
            if self.summed:
                for key, dst in (("fc2.weight", self.head.weight), ("fc2.bias", self.head.bias)):
                    M = sd[key].reshape(self.pedals, 3, *sd[key].shape[1:])   # [pedal, steer, ...]
                    dst[:3] = M.mean(0); dst[3:] = M.mean(1) - M.mean((0, 1))
            else:
                self.head.weight.copy_(sd["fc2.weight"]); self.head.bias.copy_(sd["fc2.bias"])


class Critic(nn.Module):
    def __init__(self, n_inputs: int = 19, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_inputs, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------- experience collection (runs in worker processes) ----------
_ENV = None
# Where practice and checks run: "python" (backend/sim.py through agent/simenv.py: the engine races run in, about
# twice as fast and much lighter than a browser per worker), "rust" (the optional neuroprix-rs package: the same
# engine ported to Rust, exactly the same results, several times faster; only when asked, and only if rust_ready says
# so: agent/rustenv.py) or "browser" (the game page in headless Chromium). All give the trainer the same states.
ENGINE = os.environ.get("NEUROPRIX_TRAIN_ENGINE", "python")


def make_env(engine: str = ENGINE, tracks: list | None = None, headed: bool = False):
    if engine == "browser" or headed:
        return RaceEnv(tracks or [{"seed": 1}], headless=not headed)
    if engine == "rust":
        return RustEnv(tracks or [{"seed": 1}])
    return SimEnv(tracks or [{"seed": 1}])


def _worker_init(engine: str = ENGINE):
    global _ENV
    torch.set_num_threads(1)
    _ENV = make_env(engine)


def _write_traces(job: dict, spec: dict, traces: list[dict]) -> None:
    """One JSON line per car: the pose after every decision, for later inspection. Not sent back to the learner."""
    log_dir = job.get("log_dir")
    if not log_dir or not traces:
        return
    try:
        folder = Path(log_dir)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"update-{int(job['update']):04d}-{os.getpid()}.jsonl"
        with path.open("a") as f:
            for car, steps in enumerate(traces):
                if not steps:
                    continue
                f.write(json.dumps({
                    "update": job["update"], "tier": job["tier"], "car": car, "track": spec,
                    "fields": ["x", "y", "angle", "speed", "action", "steer", "box"],
                    "box": "four corners of the body, front-left, front-right, rear-right, rear-left",
                    "steps": steps,
                }) + "\n")
    except OSError:
        pass


def _env_for(k: int):
    """The worker's k-th environment: a worker plays all its episodes side by side (collect), each in its own world;
    they share the worker's built circuits."""
    while len(_TWINS) <= k:
        _TWINS.append(_ENV if not _TWINS else _ENV.twin())
    return _TWINS[k]


_TWINS: list = []


class _Episode:
    """One practice episode in a worker: its field of cars, who learns, and what each learner did and earned."""

    def __init__(self, job: dict, env, spec: dict, pool: list, gen: int):
        tier = job["tier"]
        if spec.get("pool"):   # Surprise tracks: a circuit from the worker's pool (or a new one)
            spec = {"seed": env.pool_seed(spec["seed"], tier)}
        self.env, self.spec = env, spec
        n_cars = job["cars"]
        # league opponents: some cars driven by other drivers' frozen brains (each sensing with its own kind), drawn
        # afresh each episode; they're traffic, not learners
        n_league = min(job.get("league_cars", 0), n_cars - 1, len(pool)) if pool and n_cars > 1 else 0
        league = dict(zip(random.sample(range(n_cars), n_league), random.sample(pool, n_league))) if n_league else {}
        # slower traffic: a few copies held to a lower top speed, so there's someone to overtake. They drive the same
        # brain but their experience isn't learned from (it would teach blocking, not racing).
        n_slow = min(job.get("slow_traffic", SLOW_TRAFFIC), n_cars - 1 - n_league) if n_cars > 1 else 0
        slow = set(random.sample([i for i in range(n_cars) if i not in league], n_slow)) if n_slow > 0 else set()
        base = job.get("slow_speed", SLOW_SPEED)
        caps = [round(min(1.0, max(0.3, base + random.uniform(-0.05, 0.05))), 3) if i in slow else 1.0 for i in range(n_cars)]
        start, rolling = _start_mode(job.get("starts")), 0.0
        if start == "rolling":
            rolling = round(random.uniform(0.5, 0.8), 3)
        env.setup_world(spec, n_cars, random_start=start != "grid", tier=tier, speed_cap=job.get("speed_cap", 1.0),
                        fragile=job.get("fragile", False), caps=caps, gen=gen,
                        gens=[league[i].gen if i in league else gen for i in range(n_cars)], rolling=rolling,
                        race_grip=True)   # (a race day's grip each episode: races vary it)
        slow |= set(league)   # (nobody learns from a league car's moves either)
        self.league, self.slow = league, slow
        states = env.reset_world()
        if start == "spun":   # about half the cars standing, turned 90-180 degrees off the track
            for i in range(len(states)):
                if i not in league and random.random() < job.get("spun_share", 0.5):
                    env.spin(i, random.choice((-1, 1)) * random.uniform(math.pi / 2, math.pi), job.get("spun_room"))
            states = env.sense_all()
        self.states, n = states, len(states)
        self.n = n
        self.last_touch = [[-10**6] * n for _ in range(n)]   # step when each pair last touched
        self.traj = [{"obs": [], "act": [], "logp": [], "val": [], "rew": [], "last_val": 0.0} for _ in range(n)]
        self.traces = [[_trace_pose(s, None)] for s in states]
        self.last_dir = [None] * n   # each car's previous steering choice: -1 left, 0 straight, +1 right
        self.contacts, self.damage, self.walls, self.passes, self.lost = 0, 0.0, 0, 0, 0
        self.live, self.mine = [], []

    def begin_step(self) -> bool:
        """Who is still racing, and the league cars' moves. False once nobody is."""
        self.live = [i for i, s in enumerate(self.states) if not (s["dead"] or s["finished"])]
        self.actions = [0] * self.n
        for i in self.live:
            if i in self.league:
                self.actions[i] = self.league[i].act(self.states[i]["obs"])
        self.mine = [i for i in self.live if i not in self.league]
        return bool(self.live)

    def end_step(self, job: dict, step: int, a, lp, v, obs) -> None:
        """The learners' moves (row k of a, lp, v and obs for its k-th learner), one step of the world, and what each
        learner earned for it."""
        states, traj, n, rw = self.states, self.traj, self.n, job["rewards"]
        actions, mine, slow, last_touch = self.actions, self.mine, self.slow, self.last_touch
        if not mine:
            self.states = self.env.step_world(actions)
            return
        for k, i in enumerate(mine):
            actions[i] = a[k][1]   # (the car's action; the trajectory keeps the heads' choices)
            t = traj[i]
            t["obs"].append(obs[k]); t["act"].append(a[k][0]); t["logp"].append(lp[k]); t["val"].append(v[k])
        nxt = self.env.step_world(actions)
        gamma = job.get("gamma", 0.99)
        for i in range(n):
            for j in nxt[i].get("touched", []):
                if 0 <= j < n:
                    last_touch[i][j] = last_touch[j][i] = step
        for i in mine:
            left = job["max_steps"] - step - 1
            traj[i]["rew"].append(reward_for(states[i], nxt[i], rw, left, gamma))
            if job.get("contact_blame") and nxt[i].get("contacts"):   # the car that caused a touch pays more for it
                k = max([contact_blame(states[i], states[j]) for j in nxt[i].get("touched", []) if 0 <= j < n and j != i] or [1.0])
                traj[i]["rew"][-1] += (k - 1) * rw["contact"] * nxt[i]["contacts"]
            # places gained and lost on nearby cars that are still racing; a pass only counts if it was clean
            if rw.get("overtake") and not nxt[i]["dead"]:
                for j in range(n):
                    if j == i or states[j]["dead"] or states[j]["finished"] or nxt[j]["dead"]:
                        continue
                    before = states[i]["distance"] - states[j]["distance"]
                    after = nxt[i]["distance"] - nxt[j]["distance"]
                    if abs(after) > PASS_NEAR_PX or (before > 0) == (after > 0) or before == 0:
                        continue
                    if after > 0 and step - last_touch[i][j] > PASS_CLEAN_STEPS:
                        traj[i]["rew"][-1] += rw["overtake"]
                        self.passes += i not in slow
                    elif after < 0:   # losing a place (its own setting; by default the same as a pass is worth)
                        traj[i]["rew"][-1] += rw.get("lose_place", -rw["overtake"])
                        self.lost += i not in slow
            a_i = int(actions[i])   # the steering part of the action (proportional: its level, -1..1)
            d = ((a_i - PROP_BASE) % PROP_STEER - (PROP_STEER - 1) // 2) / ((PROP_STEER - 1) // 2) if a_i >= PROP_BASE else a_i % 3 - 1
            if self.last_dir[i] is not None:
                traj[i]["rew"][-1] += rw["steer_flip"] * abs(d - self.last_dir[i]) / 2
            self.last_dir[i] = d
            if i not in slow:
                self.contacts += nxt[i].get("contacts", 0)
                self.damage += nxt[i].get("damage", 0)
                self.walls += nxt[i].get("hits", 0)
            self.traces[i].append(_trace_pose(nxt[i], actions[i]))
        self.states = nxt


def _trace_pose(s, action):
    return [round(s["x"], 1), round(s["y"], 1), round(s["angle"], 3), round(s["speed"], 3),
            action, round(s.get("steer") or 0, 3), s.get("box")]


OBS_LEN = {g: n for n, g in GEN_OF_INPUTS.items()}   # each brain kind's number of inputs


def _obs_rows(states: list, gen: int) -> torch.Tensor:
    """Several cars' states -> one input tensor (a row each), as state_to_obs does for one."""
    want = OBS_LEN[gen]
    rows = [s["obs"] for s in states]
    for r in rows:
        if len(r) != want:
            raise ValueError(f"expected {want} features, got {len(r)}")
    return torch.tensor(rows, dtype=torch.float32).clamp_(0.0, 1.0)


def collect(job: dict) -> dict:
    """Play some episodes with the given weights. Returns trajectories as plain lists (cheap to send). A worker plays
    all its episodes side by side, so the brain chooses every learner's move in all of them in one call."""
    tier, senses = job["tier"], job.get("senses", "real")
    policy = TrainPolicy(tier, senses, job.get("layout"))
    gen, critic = policy.gen, Critic(policy.shape[0])
    policy.load_state_dict(job["policy"])
    critic.load_state_dict(job["critic"])
    random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    pool = _league(job.get("league") or [])
    eps = [_Episode(job, _env_for(k), spec, pool, gen) for k, spec in enumerate(job["specs"])]
    for step in range(job["max_steps"]):
        racing = [e for e in eps if e.begin_step()]
        if not racing:
            break
        who = [(e, i) for e in racing for i in e.mine]
        rows = [None] * len(racing)
        if who:
            obs = _obs_rows([e.states[i] for e, i in who], gen)
            with torch.no_grad():
                a, lp = policy.sample(obs)
                v = critic(obs)
            obs_l, a_l, lp_l, v_l = obs.tolist(), a.tolist(), lp.tolist(), v.tolist()
            k = 0
            for r, e in enumerate(racing):
                m = len(e.mine)
                rows[r] = ([(x, policy.env_action(x)) for x in a_l[k:k + m]], lp_l[k:k + m], v_l[k:k + m], obs_l[k:k + m])
                k += m
        for r, e in enumerate(racing):
            e.end_step(job, step, *(rows[r] or ([], [], [], [])))
    out, totals = [], {"contacts": 0, "damage": 0.0, "walls": 0, "passes": 0, "lost": 0}
    for e in eps:
        last = [(i, s) for i, s in enumerate(e.states) if e.traj[i]["obs"] and not (s["dead"] or s["finished"])]
        if last:   # cut off by the step cap: worth what the critic says
            with torch.no_grad():
                vals = critic(_obs_rows([s for _, s in last], gen)).tolist()
            for (i, _), val in zip(last, vals):
                e.traj[i]["last_val"] = val
        _write_traces(job, e.spec, e.traces)
        out += [t for i, t in enumerate(e.traj) if t["obs"] and i not in e.slow]
        for k in totals:
            totals[k] += getattr(e, k)
    return {"traj": out, **totals}


_LEAGUE: dict = {}   # (each worker: league brains, loaded once)


def _league(paths: list) -> list:
    """The league's frozen brains (the engine's Brain: the forward pass races use), at race temperature."""
    out = []
    for p in paths:
        if p not in _LEAGUE:
            b = sim.Brain(json.loads(Path(p).read_text()))
            b.temperature, b.rng = 0.2, random.Random(hash(p) & 0xFFFF)
            _LEAGUE[p] = b
        out.append(_LEAGUE[p])
    return out


def _start_mode(starts) -> str:
    """One episode's start (--starts): "grid" (standing, on the race grid at the line), "rolling" (the grid somewhere
    round the circuit, already moving), "random" (standing, the grid somewhere round the circuit: the default) or
    "spun" (as random, with about half the learner's cars turned 90-180 degrees off the track's direction: practice
    at finding the way after a spin)."""
    if not starts:
        return "random"
    r, acc = random.random() * sum(w for _, w in starts), 0.0
    for mode, w in starts:
        acc += w
        if r < acc:
            return mode
    return starts[-1][0]


def contact_blame(me: dict, other: dict) -> float:
    """How much of a touch was this car's doing (--contact-blame), from the two cars just before it: the car behind that
    was closing in 1.5 (it had the choice), the car hit from behind 0.5, side by side (within half a car length along
    the circuit) or anything else 1. Its contact penalty is multiplied by this."""
    gap = other["distance"] - me["distance"]
    if abs(gap) < 9:
        return 1.0
    if gap > 0:
        return 1.5 if me["speed"] > other["speed"] else 1.0
    return 0.5 if other["speed"] > me["speed"] else 1.0


def gae(t: dict, gamma: float, lam: float):
    adv, last = [0.0] * len(t["rew"]), 0.0
    vals = t["val"] + [t["last_val"]]
    for k in reversed(range(len(t["rew"]))):
        delta = t["rew"][k] + gamma * vals[k + 1] - vals[k]
        last = delta + gamma * lam * last
        adv[k] = last
    return adv, [a + v for a, v in zip(adv, t["val"])]


# ---------- both directions ----------
BOTH_WAYS = True   # practise every gym both ways round (half the practice runs reversed); procedural circuits already
                   # come either way. A driver that only ever went clockwise round an oval learned that left turns are
                   # never needed (it almost never tried one), and relearning took 140+ updates.


def reverse_spec(spec: dict) -> dict:
    """A gym driven the other way round: same start line, control points in the opposite order (and their heights and
    banks with them; a bank leans into its corner whichever way round it's driven)."""
    if "control" not in spec:
        return spec
    flip = lambda v: [v[0]] + v[1:][::-1]   # noqa: E731
    out = {**spec, "control": flip(spec["control"]), "name": f"{spec.get('name', 'Gym')} (reversed)"}
    for k in ("elev", "bank"):
        if isinstance(spec.get(k), list) and len(spec[k]) == len(spec["control"]):
            out[k] = flip(spec[k])
    return out


def either_way(spec: dict, both: bool = BOTH_WAYS) -> dict:
    """A gym as drawn or reversed, at random (when practising both ways)."""
    return reverse_spec(spec) if both and "control" in spec and random.random() < 0.5 else spec


# ---------- driving lessons ----------
GAME_MAX_SPEED = 5.0   # PHYS.maxSpeed in the game (state["speed"] is a share of it)


def coach_pedal(state: dict) -> int:
    """The coach: flat out, and brake exactly when the car is faster than it could carry into the corners ahead and
    still brake in time (state["corner_need"], from the game). 0 = accelerate, 2 = brake."""
    return 2 if state["speed"] * GAME_MAX_SPEED > state.get("corner_need", 99) * 0.98 else 0


def driving_lessons(policy, env, tier: str, rounds: int, specs: list, random_tracks: bool, say, both_ways: bool = BOTH_WAYS) -> None:
    """Before normal training: the driver drives practice runs with the coach on the pedals (and its own steering), and
    learns to copy the coach's pedal choices, so braking late is something it does and training can then reinforce.
    Its steering is kept as it was: its whole steering habit (how likely each way is, in each moment), not just its
    most likely choice. Copying only the most likely choice turned a young driver's steering into "always straight"
    (it turned by sometimes picking a turn, not because turning was its top choice), and it never steered again.
    From the second round, half the time it uses its own pedals, so it also learns what to do after its own mistakes."""
    gen = policy.gen
    teacher = TrainPolicy(tier); teacher.load_state_dict(policy.state_dict())   # its steering, as it was
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    O, PED, STEER = [], [], []   # STEER: the driver's own steering probabilities before the lessons
    w = torch.ones(policy.pedals); w[2] = 8.0   # braking is rare: weigh it up
    caught = None
    for r in range(rounds):
        say(f"driving lesson {r + 1}/{rounds}", event="stage",
            message=f"Driving lesson {r + 1} of {rounds}: learning to brake late for corners"
                    + (f" (last lesson: braked with the coach {caught:.0%} of the time)" if caught is not None and n_brakes
                       else " (no corner so far has needed braking)" if caught is not None else ""))
        for ep in range(8):
            spec = {"seed": random.randrange(1_000_000)} if random_tracks or not specs else either_way(random.choice(specs), both_ways)
            env.setup_world(spec, 1, random_start=True, tier=tier)
            s = env.reset_world()[0]
            for _ in range(300):
                obs = state_to_obs(s, gen)
                with torch.no_grad():
                    z = policy(obs.unsqueeze(0))[0]
                    sp = torch.softmax(teacher(obs.unsqueeze(0))[0][:3], 0)
                    steer = int(torch.multinomial(sp, 1))   # it steers as it would have
                ped = coach_pedal(s)
                O.append(obs); PED.append(ped); STEER.append(sp)
                act_ped = ped if r == 0 or random.random() < 0.5 else int(torch.argmax(z[3:]))
                s = env.step_world([act_ped * 3 + steer])[0]
                if s["dead"] or s["finished"]:
                    break
        X, P, S = torch.stack(O), torch.tensor(PED), torch.stack(STEER)
        for _ in range(20):
            perm = torch.randperm(len(X))
            for k in range(0, len(X), 512):
                i = perm[k:k + 512]
                z = policy(X[i])
                loss = F.cross_entropy(z[:, 3:], P[i], weight=w) + F.kl_div(F.log_softmax(z[:, :3], 1), S[i], reduction="batchmean")
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = torch.argmax(policy(X)[:, 3:], 1)
            agree = float((pred == P).float().mean()); brakes = P == 2
            caught = float((pred[brakes] == 2).float().mean()) if brakes.any() else 0.0
            n_brakes = int(brakes.sum())
        say(f"lesson {r + 1}/{rounds}: copies the coach's pedals {agree:.0%} of the time; brakes when the coach brakes {caught:.0%} "
            f"({int(brakes.sum())} braking moments so far)")


# ---------- evaluation ----------
def _pose(s: dict, act: int | None = None) -> list:
    """One replay step: [x, y, heading, health, braking, action]: health so the replay shows damage, smoke and fire;
    braking (null without pedals) for brake lights and tyre squeal; action the move it chose for this step (null on the
    first), for the Train page's cockpit view (game/trainer.js pose)."""
    return [round(s["x"], 1), round(s["y"], 1), round(s["angle"], 3), round(s.get("health", 1.0), 3),
            (1 if s["braking"] else 0) if "braking" in s else None, act]


def _senses(x) -> str:
    """One decision's senses for the replay (the Train page lights its brain view with them): base64 of uint16 x 10000.
    Exact: the engine's inputs are 4 decimals, 0..1 (game/trainer.js sensesOf)."""
    import base64
    import struct
    return base64.b64encode(struct.pack(f"<{len(x)}H", *(round(min(1.0, max(0.0, float(v))) * 10000) for v in x))).decode()


def replay_net(model) -> dict:
    """The network that drove a replay, exactly (float32 to 9 significant digits round-trips; game/trainer.js replayNet)."""
    layer = lambda fc: {"weight": [[float(f"{v:.9g}") for v in row] for row in fc.weight.tolist()],
                        "bias": [float(f"{v:.9g}") for v in fc.bias.tolist()]}
    return {"fc1": layer(model.fc1), "fc2": layer(model.fc2), "action_layout": model.layout}


def _drive(env, model, gen, tier, spec, cars, max_steps, record=None, others=None, senses=None):
    """One run: car 0 drives its most likely action; any other cars drive the same brain with its natural
    randomness (identical deterministic cars would just pile into each other). Returns car 0's end state,
    the race length in px and car 0's touches. record: car 0's path; others: the other cars' paths and how
    each one ended (for the replay on the Train page); senses: car 0's inputs at each decision (_senses), the one taking
    it from record[t] to record[t + 1]."""
    info = env.setup_world(spec, cars, tier=tier, gen=gen)
    states = env.reset_world()
    n, touches, steps, walls = len(states), 0, 0, 0
    if others is not None:
        others.extend({"path": [_pose(s)], "finished": False, "reason": ""} for s in states[1:])
    if record is not None:
        record.append(_pose(states[0]))
    for _ in range(max_steps):
        live = [k for k, s in enumerate(states) if not (s["dead"] or s["finished"])]
        if 0 not in live:
            break
        with torch.no_grad():
            obs = torch.stack([state_to_obs(states[k], gen) for k in live])
            logits = model(obs)
        if senses is not None:
            senses.append(_senses(obs[0].tolist()))   # (car 0 is live[0])
        acts = [0] * n
        for j, k in enumerate(live):
            acts[k] = pick_action(logits[j], model.layout, greedy=k == 0)
        states = env.step_world(acts)
        touches += states[0].get("contacts", 0)
        walls += states[0].get("hits", 0)
        steps += 1
        if record is not None:
            record.append(_pose(states[0], acts[0]))
        if others is not None:
            for k in live:
                if k > 0:
                    o = others[k - 1]
                    o["path"].append(_pose(states[k], acts[k]))
                    o["finished"], o["reason"] = states[k]["finished"], states[k]["reason"] or ""
    info["walls"] = walls   # (frames car 0 spent touching a wall)
    return states[0], info, touches, steps


CHECK_WALLS = 1.0   # how much wall contact counts in the checks (--check-walls)


def evaluate(env, model, specs, cars, max_steps, record=None, preview=None, show=0, senses=None):
    """How good is this brain? On each circuit it drives alone (pure driving) and, with traffic, as car 0 of the
    field. Score = 60% solo progress + 40% progress in traffic (fractions of the race), minus 0.05 per touch per
    100 steps in traffic (a rate, so surviving longer is never punished; at most 0.2), minus 0.2 x the health
    lost driving alone (a driver that bounces off the walls to the flag is worse than a clean one), minus 0.06 per
    frame touching a wall per 100 steps driving alone (scrapes cost almost no health, but a driver that brushes the
    barriers isn't a clean one either; at most 0.2). --check-walls multiplies both numbers.
    Driving alone keeps one unlucky first-corner bump from deciding which checkpoint is kept.
    record / preview: the replay of the first circuit, the drive in traffic when there is traffic (car 0's path
    goes in record; preview gets the other cars and how car 0's drive ended)."""
    gen = generation(model)
    tier = TIER_OF_GEN.get(gen, "n4")
    results, scores = [], []
    for i, spec in enumerate(specs):
        show_this = i == show and record is not None
        s0, info, _, solo_steps = _drive(env, model, gen, tier, spec, 1, max_steps, record if show_this and cars == 1 else None,
                                         senses=senses if show_this and cars == 1 else None)
        walls = info.get("walls", 0)
        total = info["length"] * info["laps"]
        solo = min(1.0, max(0.0, s0["distance"]) / total)
        traffic, touches, steps = solo, 0, 1
        if show_this and cars == 1 and preview is not None:   # the replay is this solo drive
            preview.update(others=[], progress=round(solo, 4), finished=s0["finished"], reason=s0["reason"] or "timeout")
        if cars > 1:
            others = [] if show_this else None
            t0, _, touches, steps = _drive(env, model, gen, tier, spec, cars, max_steps, record if show_this else None, others,
                                           senses if show_this else None)
            traffic = min(1.0, max(0.0, t0["distance"]) / total)
            if show_this and preview is not None:
                preview.update(others=others, progress=round(traffic, 4), finished=t0["finished"], reason=t0["reason"] or "timeout")
        health = s0.get("health", 1.0)
        scores.append(0.6 * solo + 0.4 * traffic - min(0.2, 0.05 * 100 * touches / max(1, steps)) - 0.2 * (1 - health)
                      - min(0.2 * CHECK_WALLS, 0.06 * CHECK_WALLS * 100 * walls / max(1, solo_steps)))
        results.append({"track": info["name"], "progress": round(solo, 4), "finished": s0["finished"], "health": health,
                        "reason": s0["reason"] or "timeout", "time": s0["time"], "field_progress": round(traffic, 4),
                        "car_contacts": touches, "wall_frames": walls})
    return sum(scores) / len(scores), results


def preview_drive(env, model, spec, cars, max_steps, record, preview, senses=None) -> None:
    """The replay shown on the Train page, on a circuit that isn't part of the score (Surprise tracks sessions get a fresh
    one at every check): the field if the session trains in traffic, else the driver alone."""
    gen = generation(model)
    tier = TIER_OF_GEN.get(gen, "n4")
    others = [] if cars > 1 else None
    t0, info, _, _ = _drive(env, model, gen, tier, spec, cars, max_steps, record, others, senses)
    total = info["length"] * info["laps"]
    preview.update(others=others or [], progress=round(min(1.0, max(0.0, t0["distance"]) / total), 4),
                   finished=t0["finished"], reason=t0["reason"] or "timeout")


# ---------- a runner's session (the trainer contract: see the SDK's docs/trainer-contract.md) ----------
SESSION_NUMBERS = {"updates", "batch", "eval_every", "traffic", "max_steps", "epochs", "minibatch", "progress", "alive",
                   "fast", "fast_threshold", "idle", "dead", "damage", "finish", "contact", "steer_flip", "corner", "risk",
                   "crash", "lose_place", "overtake", "wall", "slide", "turnround", "lr", "entropy", "gamma", "lam", "clip",
                   "start_margin", "check_tracks", "check_walls"}
SESSION_SWITCHES = {"random_tracks", "one_way"}


def session_argv(path: str) -> list[str]:
    """The trainer's options for a session.json written by a runner (only known settings are taken)."""
    folder = Path(path).resolve().parent
    s = json.loads(Path(path).read_text())
    if s.get("neuroprix_session") != 1:
        sys.exit("unknown session format: update the trainer")
    argv = ["--progress-json", "--name", str(s.get("driver") or "driver")[:64],
            "--tier", str(s.get("class", "n3")).lower()]
    for k, v in (s.get("settings") or {}).items():
        if k in SESSION_SWITCHES:
            if v:
                argv.append("--" + k.replace("_", "-"))
        elif k in SESSION_NUMBERS:
            argv += ["--" + k.replace("_", "-"), str(float(v) if not isinstance(v, int) else v)]
        elif k == "seeds":
            argv += ["--seeds", ",".join(str(int(x)) for x in v)]
        elif k == "layout" and v in LAYOUTS:   # the coach's choice of action layout (docs/brain-format.md)
            argv += ["--layout", v]
        else:
            sys.exit(f"unknown setting {k!r} in the session: update the trainer")
    for i, spec in enumerate(s.get("tracks") or []):
        p = folder / f"track_{i}.json"
        p.write_text(json.dumps(spec))
        argv += ["--track", str(p)]
    if s.get("start_weights"):
        argv += ["--resume", s["start_weights"]]
    elif s.get("start_brain"):   # a starting brain as JSON (numbers only)
        argv += ["--resume", s["start_brain"]]
    out = Path(s["brain_out"])   # the best brain goes here: brain.json (and brain.pth beside it)
    argv += ["--out", str(out.with_suffix(".pth"))]
    if s.get("workers"):
        argv += ["--workers", str(int(s["workers"]))]
    # each session its own random seed (from its id): training is deterministic, so with one fixed seed two drivers
    # trained with the same settings came out identical
    argv += ["--seed", str(int(hashlib.sha256(str(s.get("id") or "").encode()).hexdigest()[:8], 16))]
    return argv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", action="append", default=[], help="track JSON from the Gym Editor (repeatable)")
    ap.add_argument("--seeds", default="", help="comma-separated procedural seeds to train on, e.g. 1,2,3")
    ap.add_argument("--random-tracks", action="store_true", help="fresh procedural circuit every episode")
    ap.add_argument("--tier", type=str.lower, choices=["n4", "n3", "n2", "n1"], default="n3",
                    help="n4: steering only (fast to train); n3: steering + pedals; n2: + battery boost and slipstream inputs")
    ap.add_argument("--traffic", type=int, default=4, help="cars per episode, all driven by this driver (1 = alone; at most 10)")
    ap.add_argument("--senses", choices=["real"], default="real",
                    help="what the brain senses: only what a real car could measure, and the heading (brain kinds 16-19, "
                         "docs/observations-and-actions.md; the only senses there are)")
    ap.add_argument("--starts", default="", help="each episode's start, weighted, e.g. grid:0.4,rolling:0.3,random:0.3 "
                    "(grid: standing at the line; rolling: moving, somewhere round; random: standing, somewhere round; default random; "
                    "spun: as random, about half the cars turned 90-180 degrees off the track's direction)")
    ap.add_argument("--league", action="append", default=[], help="league opponents: brain .json files or folders of them (repeatable)")
    ap.add_argument("--league-cars", type=int, default=0, help="cars per episode driven by league brains (not learned from)")
    ap.add_argument("--layout", choices=LAYOUTS, default=None,
                    help="the brain's action layout (docs/brain-format.md): combined (one choice over every pedal x "
                         "steering pair), two_head (steering, pedal) or three_head (steering, pedal, boost; N2 / N1), each "
                         "on/off or by degrees (_prop). Default: a resumed brain's own; a new brain by degrees "
                         "(N2 / N1 three_head_prop, N3 two_head_prop)")
    ap.add_argument("--contact-blame", action="store_true", help="a touch costs the car that caused it 1.5x, the car hit 0.5x")
    ap.add_argument("--updates", type=int, default=150, help="policy updates")
    ap.add_argument("--batch", type=int, default=8, help="episodes (fields of cars) per update")
    ap.add_argument("--max-steps", type=int, default=400, help="step cap per practice episode (1 step = 4 frames)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("NEUROPRIX_TRAIN_WORKERS", "0")) or max(1, min(8, (os.cpu_count() or 2) - 2)),
                    help="games running in parallel to collect experience")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy", type=float, default=0.01, help="entropy bonus coefficient")
    # reward weights (per step of 4 frames)
    ap.add_argument("--progress", type=float, default=R_PROGRESS, help="reward per px of new progress")
    ap.add_argument("--alive", type=float, default=R_ALIVE, help="reward per step while making progress")
    ap.add_argument("--fast", type=float, default=R_FAST, help="extra reward per step when also fast")
    ap.add_argument("--fast-threshold", type=float, default=FAST_THRESHOLD, help="share of top speed that counts as fast")
    ap.add_argument("--idle", type=float, default=R_IDLE, help="reward for a step without new progress")
    ap.add_argument("--damage", type=float, default=R_DAMAGE, help="reward per unit of health lost")
    ap.add_argument("--finish", type=float, default=R_FINISH, help="reward for completing the race")
    ap.add_argument("--contact", type=float, default=R_CONTACT, help="reward for each new touch with another car")
    ap.add_argument("--risk", type=float, default=R_RISK, help="crash penalty x (1 + this x 10 x how far over the safe speed)")
    ap.add_argument("--one-way", action="store_true", help="practise gyms only in the direction drawn (default: both ways)")
    ap.add_argument("--lessons", type=int, default=0,
                    help="driving lessons first: rounds of copying a coach that brakes late (N3/N2). Off by default: drivers learn "
                         "braking themselves, and in a test lessons made a natural driver worse (for experiments)")
    ap.add_argument("--crash", type=float, default=R_CRASH, help="crash penalty x (impact strength)^2 per hit")
    ap.add_argument("--lose-place", type=float, default=R_LOSE_PLACE, help="reward for losing a place to a nearby car (a penalty)")
    ap.add_argument("--corner", type=float, default=R_CORNER, help="corner discipline: x10 x share too fast for the corners ahead")
    ap.add_argument("--overtake", type=float, default=R_OVERTAKE, help="reward for a clean pass (and the cost of losing a place)")
    ap.add_argument("--slow-traffic", type=int, default=SLOW_TRAFFIC, help="traffic cars per episode held to a lower top speed")
    ap.add_argument("--slow-speed", type=float, default=SLOW_SPEED, help="their top speed, as a share of the real one")
    ap.add_argument("--learner-start", type=float, default=None,
                    help="practice top speed at the start, as a share of the real one (1 = off). Checkpoint checks always run at full speed")
    ap.add_argument("--learner-updates", type=int, default=None, help="updates over which the practice top speed rises to full")
    ap.add_argument("--reopen-pedals", type=float, default=1.0,
                    help="continuing a driver with pedals: scale its pedal choice by this (flatter = the brake and boost get "
                         "tried again; 1 = off, the default: in a test it pushed a braking driver back to lifting off). Steering is untouched")
    ap.add_argument("--first-lessons", type=int, default=0,
                    help="practice updates where any wall hit ends the run (as cars used to), before the real damage rules")
    ap.add_argument("--wall", type=float, default=R_WALL, help="reward per physics frame touching a wall (scraping)")
    ap.add_argument("--steer-flip", type=float, default=R_STEER_FLIP, help="reward for reversing the steering (left <-> right) between decisions")
    ap.add_argument("--critic-warmup", type=int, default=None, help="updates that train only the critic (default 10 when resuming)")
    ap.add_argument("--backwards", type=float, default=0.0, help="experiments: a price on ground lost, this x the progress "
                    "reward per px driven backwards (1: going back costs what going forward pays; 0: free until out)")
    ap.add_argument("--stand-wrong", type=float, default=0.0, help="experiments: reward per step for standing still "
                    "(under 10%% of top speed) facing more than 90 degrees off the road's direction (e.g. -3)")
    ap.add_argument("--spun-share", type=float, default=0.5, help="experiments: the share of the learner's cars a spun "
                    "start turns round (0.5: about half)")
    ap.add_argument("--spun-room", type=float, default=0.0, help="experiments: how far back (px) a car started spun may go "
                    "before it's out for the wrong way, in practice (0: the races' 120 px)")
    ap.add_argument("--turnround", type=float, default=0.0, help="while more than 30 degrees off the road's "
                    "direction, this x 4 x how far the car turned back towards it (the same cost for turning away)")
    ap.add_argument("--slide", type=float, default=0.0, help="per decision, this x how far the car slides "
                    "past 15 degrees off its nose (1 at 90 degrees or more), e.g. -10; 0: off")
    ap.add_argument("--dead", type=float, default=R_DEAD, help="reward for being out (wrecked, stalled or wrong way)")
    ap.add_argument("--worker-timeout", type=float, default=900, help="seconds one update's games may take before the workers are restarted")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--check-tracks", type=int, default=6,
                    help="circuits each check drives when practising on random tracks (more = a steadier pick of the best "
                         "version, but slower checks)")
    ap.add_argument("--check-walls", type=float, default=1.0,
                    help="how much wall contact counts in the checks (x the usual penalty), so the version kept is a clean one")
    ap.add_argument("--start-margin", type=float, default=0.0,
                    help="the starting weights count this much lower in the checks, so a checkpoint nearly as good replaces "
                         "them (for retraining habits, e.g. cleaner driving, that cost a little check score at first)")
    ap.add_argument("--holdout", default="9001,9002,9003", help="unseen procedural seeds used only for reporting")
    ap.add_argument("--out", default="racer_weights.pth")
    ap.add_argument("--log-dir", default="", help="folder for one JSON line per car per episode (x, y, angle, speed, action)")
    ap.add_argument("--name", default=None, help="agent name stored in the exported JSON")
    ap.add_argument("--resume", default=None, help="start from an existing brain: a .json, or a .pth (PyTorch 2.6 or newer)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headed", action="store_true", help="show the evaluation browser")
    ap.add_argument("--engine", choices=["python", "rust", "browser"], default=ENGINE if ENGINE in ("python", "rust", "browser") else "python",
                    help="where practice runs: the Python race engine (default), the Rust one (optional, faster: "
                         "pip install neuroprix-rs; used only if it matches this Python engine exactly), or the browser game")
    ap.add_argument("--progress-json", action="store_true",
                    help="print one JSON event per line instead of text (used by the training server)")
    ap.add_argument("--session", default=None,
                    help="train the session a runner prepared (session.json: settings, gyms, starting brain, brain_out); "
                         "see the SDK's docs/trainer-contract.md")
    argv = sys.argv[1:]
    if "--session" in argv:   # a runner's session: its file says everything
        argv = session_argv(argv[argv.index("--session") + 1]) + [a for i, a in enumerate(argv)
                                                                   if a != "--session" and (i == 0 or argv[i - 1] != "--session")]
    args = ap.parse_args(argv)

    def say(text: str, **event) -> None:
        if args.progress_json:
            if event:
                print(json.dumps(event), flush=True)
        else:
            print(text, flush=True)

    # One thread for the learner too: the brain is tiny, and spreading its maths over threads costs more than it saves
    # (a learning step took 119 ms on 1 thread, 149 ms on 4). The game workers already use one each.
    torch.set_num_threads(1)
    if args.engine == "rust" and not args.headed:   # (asked for: used only if it's exactly the Python engine)
        ok, why = rust_ready()
        if not ok:
            args.engine = "python"
        text = f"engine: Rust, {why}" if ok else f"engine: Python (the Rust engine isn't used: {why})"
        say(text, event="note", message=text)
    elif args.engine == "python":
        say("engine: Python", event="note", message="engine: Python")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cars = max(1, min(10, args.traffic))
    starts = []
    for part in filter(None, (p.strip() for p in args.starts.split(","))):
        mode, _, w = part.partition(":")
        if mode not in ("grid", "rolling", "random", "spun"):
            sys.exit(f"--starts: unknown start '{mode}' (grid, rolling, random or spun)")
        starts.append((mode, float(w or 1)))
    league = []
    for item in args.league:
        p = Path(item)
        league += sorted(str(q) for q in p.glob("*.json")) if p.is_dir() else [str(p)]
    specs = [load_track_spec(t) for t in args.track] + [{"seed": int(s)} for s in args.seeds.split(",") if s.strip()]
    if not specs and not args.random_tracks:
        specs = [{"seed": 1}]
    holdout = [{"seed": int(s)} for s in args.holdout.split(",") if s.strip()]
    global CHECK_WALLS
    CHECK_WALLS = max(0.0, args.check_walls)
    both_ways = not args.one_way
    # checks drive each gym both ways too, so the version kept is the one that's good in both directions
    eval_specs = (specs + ([reverse_spec(s) for s in specs if "control" in s] if both_ways else [])) if specs \
        else [{"seed": 101 * k} for k in range(1, max(1, args.check_tracks) + 1)]

    rewards = {"progress": args.progress, "alive": args.alive, "fast": args.fast, "fast_threshold": args.fast_threshold,
               "idle": args.idle, "dead": args.dead, "damage": args.damage, "finish": args.finish, "contact": args.contact,
               "steer_flip": args.steer_flip, "wall": args.wall, "overtake": args.overtake,
               "corner": args.corner, "lose_place": args.lose_place, "crash": args.crash, "risk": args.risk,
               "turnround": args.turnround, "backwards": args.backwards, "stand_wrong": args.stand_wrong,
               "slide": args.slide}
    changed = {k: v for k, v in rewards.items() if v != DEFAULT_REWARDS[k]}
    if changed:
        say(f"custom rewards: {changed}", event="note", message="custom rewards: " + ", ".join(f"{k} {v:g}" for k, v in changed.items()))
    # Learner's permit (off: drivers practise at full speed from the start, like the races they're for). It could start
    # practice at a lower top speed (--learner-start, a share of it) rising to full over --learner-updates; but at the
    # lower speed corners never need braking, so the driver learned that braking only loses ground. For experiments only.
    if args.learner_start is None:
        args.learner_start = 1.0
    if args.learner_updates is None:
        args.learner_updates = 0
    def speed_cap(update: int) -> float:
        """Learner's permit: practice starts at a lower top speed and rises to full over --learner-updates."""
        if args.learner_start >= 1 or args.learner_updates <= 0:
            return 1.0
        return round(args.learner_start + (1 - args.learner_start) * min(1.0, (update - 1) / args.learner_updates), 3)
    tier = args.tier
    src_layout = "combined"
    if args.resume:   # a brain keeps its action layout unless told otherwise
        try:
            src_layout = load_layout(args.resume)
            if load_weights(args.resume)["fc1.weight"].shape[1] in OLD_INPUTS:   # an earlier kind: nothing carries over
                sys.exit(f"--resume: {OLD_SENSES}")
        except (ValueError, KeyError):
            pass   # (reported below)
        if args.layout is None and src_layout != "combined":
            args.layout = src_layout
    elif args.layout is None:   # a new brain: its class's default layout (agent/model.py DEFAULT_LAYOUT)
        args.layout = DEFAULT_LAYOUT.get(tier)
    try:
        policy = TrainPolicy(tier, args.senses, args.layout)
    except ValueError as e:
        sys.exit(f"--layout: {e}")
    critic = Critic(policy.shape[0])
    if args.resume:
        try:
            sd = load_weights(args.resume)
        except ValueError as e:
            sys.exit(f"--resume: {e}")
        shape = (sd["fc1.weight"].shape[1], sd["fc2.weight"].shape[0])
        if src_layout != policy.layout and shape[0] == policy.shape[0]:
            try:
                sd = relayout(sd, src_layout, policy.layout, policy.pedals)
            except ValueError as e:
                sys.exit(f"--layout: {e}")
            shape = (shape[0], sd["fc2.weight"].shape[0])
            say(f"moving the brain's choices to the {policy.layout} action layout", event="note",
                message=f"moved the brain to the {policy.layout} action layout")
        if (sd["fc1.weight"].shape[1], sd["fc2.weight"].shape[0]) != policy.shape or sd["fc1.weight"].shape[0] != N_HIDDEN:
            sys.exit(f"--resume needs a {tier.upper()} brain ({policy.layout} action layout); other brains start a new one from scratch")
        policy.load_brain(sd)
        if policy.split and args.reopen_pedals < 1:   # a driver that stopped trying the brake (or boost) never learns it
            with torch.no_grad():                       # pays: flatten its pedal choice so every pedal gets tried again
                policy.head.weight[policy.sizes[0]:] *= args.reopen_pedals
                policy.head.bias[policy.sizes[0]:] *= args.reopen_pedals
    else:
        with torch.no_grad():   # gentle start; an N3 newborn leans a little on the throttle (pedal head: 0 = accelerate)
            policy.head.weight.mul_(0.1)
            policy.head.bias.zero_()
            if policy.layout.endswith("_prop"):   # by degrees: a lean on full throttle (and boost, as likely)
                full = PROP_PEDAL - 1
                if policy.split:
                    policy.head.bias[PROP_STEER + full] += 0.5
                    if policy.pedals == 4 and len(policy.sizes) == 2:   # (two heads: boost is the pedal's last level)
                        policy.head.bias[PROP_STEER + full + 1] += 0.5
                else:   # (one joint head: on every full throttle and boost pair)
                    policy.head.bias[PROP_STEER * full:] += 0.5
            elif policy.split:
                policy.head.bias[3] += 0.5
                if policy.pedals == 4 and policy.layout != "three_head":   # N2: Boost starts as likely as the throttle
                    policy.head.bias[6] += 0.5                              # (three heads: its own head, at even odds)
            elif policy.pedals:   # (one joint head: the same lean, on every throttle and boost pair)
                policy.head.bias[0:3] += 0.5
                if policy.pedals == 4:
                    policy.head.bias[9:12] += 0.5
    params = list(policy.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    name = args.name or Path(args.out).stem
    workers = max(1, min(args.workers, args.batch))
    # resuming: the critic starts blank, so let it learn the brain's values before the brain itself changes
    warmup = args.critic_warmup if args.critic_warmup is not None else (10 if args.resume else 0)

    env = make_env(args.engine, specs, args.headed)   # evaluation
    # Driving lessons (copying a coach that brakes late) are off by default (--lessons 0): an empty brain trained at full
    # speed on Grand Prix circuits learned to drive and brake by itself, and in its second session it learned to brake
    # late instead of lifting off with no lessons, while lessons made it worse. Never for a new driver (it can't steer
    # yet, so it crashes before any corner).
    def stage(message: str) -> None:   # what the driver is learning now, in words (shown on the training page)
        say(message, event="stage", message=message)
    if not args.resume:
        stage("Learning to drive: a new driver usually needs a few hundred updates before it gets round"
              + ((" (on N1's hilly circuits, about 600-1000)" if tier == "n1" else " (on Grand Prix circuits, about 200-400)")
                 if args.random_tracks else ""))
    elif args.lessons > 0 and policy.summed:
        driving_lessons(policy, env, tier, args.lessons, specs, args.random_tracks, say, both_ways)
        warmup = max(warmup, 10)   # the critic learns the new driving before the driver starts changing
    label = "random procedural circuits" if args.random_tracks else ", ".join(env.setup_world(s, 1)["name"] for s in specs)
    say(f"Training '{name}' ({tier.upper()}, PPO, {workers} games in parallel) on: {label}\n{cars} car(s) per episode; "
        f"{args.updates} updates x {args.batch} episodes, max {args.max_steps} steps each\n",
        event="start", name=name, tracks=label, updates=args.updates, batch=args.batch, eval_spec=eval_specs[0],
        traffic=cars, learner="ppo", workers=workers, tier=tier)
    def new_pool():
        # one process per worker, and job w of every update always goes to worker w: a worker keeps its own pool of
        # built circuits (SimEnv.pool_seed), so training stays repeatable only if the same jobs always meet the same pool
        return [mp.get_context("spawn").Pool(1, initializer=_worker_init, initargs=(args.engine,)) for _ in range(workers)]

    def gather(todo):
        """Collect experience from the game workers. A worker can die (a browser crash, a broken pipe) or hang;
        then the workers are restarted and the update is played again, so one bad game doesn't end the session."""
        nonlocal pool
        for attempt in range(3):
            try:
                end = time.time() + args.worker_timeout
                waits = [w.apply_async(collect, (job,)) for w, job in zip(pool, todo)]
                return [r.get(timeout=max(1.0, end - time.time())) for r in waits]
            except Exception as e:   # noqa: BLE001 - any worker failure: restart them all and retry
                say(f"  a game worker failed ({type(e).__name__}: {str(e)[:120]}); restarting the workers",
                    event="note", message=f"game workers restarted after {type(e).__name__}")
                for w in pool:
                    w.terminate()
                pool = new_pool()
        raise RuntimeError("the game workers kept failing; stopping this session")

    pool = new_pool()
    best, best_key, t0 = -1.0, (-9.0, 0.0), time.time()
    n_checks = [0]

    def run_check(model):
        """Score the brain on the fixed test circuits (fair between checkpoints) and record the replay you watch: on a
        brand-new circuit for Surprise tracks sessions, else on the session's circuits in turn."""
        path, preview, cap, senses = [], {}, max(2500, args.max_steps * 3), []
        if args.random_tracks:
            score, res = evaluate(env, model, eval_specs, cars, cap)
            spec = {"seed": random.randrange(1_000_000)}
            preview_drive(env, model, spec, cars, cap, path, preview, senses)
        else:
            k = n_checks[0] % len(eval_specs)
            spec = eval_specs[k]
            score, res = evaluate(env, model, eval_specs, cars, cap, record=path, preview=preview, show=k, senses=senses)
        preview["spec"] = spec
        preview["senses"], preview["net"] = senses, replay_net(model)   # (the Train page's brain view: this car's thinking)
        n_checks[0] += 1
        return score, res, path, preview
    try:
        if args.resume:   # a session that only makes things worse keeps the starting weights
            score, res, path, preview = run_check(policy.export())
            best, best_key = score, (round(score - args.start_margin, 3), -sum(r["time"] for r in res))
            save_brain(policy.export(), args.out, name)
            say(f"starting weights: score {score:5.2f}" + (f" (a checkpoint replaces them from {score - args.start_margin:.2f})"
                                                          if args.start_margin else "") + "  <- saved",
                event="eval", update=0, progress=round(max(0.0, score), 4), saved=True, results=res, path=path, preview=preview)
        if args.resume:
            stage("Racing practice" + (": racing other drivers and copies of itself" if cars > 1 and league and args.league_cars > 0
                                       else ": racing copies of itself, with a slower car to overtake" if cars > 1 and args.slow_traffic > 0
                                       else ": racing copies of itself" if cars > 1 else ""))
        for update in range(1, args.updates + 1):
            if update > 1 and speed_cap(update) >= 1 > speed_cap(update - 1):
                stage("Learning to race: now practising at full speed")
            # a new random circuit every episode for Surprise tracks; with gyms too, about half of each
            # (a random circuit comes from the worker's pool of built ones: SimEnv.pool_seed)
            eps = [{"seed": random.randrange(1_000_000), "pool": True} if args.random_tracks and (not specs or random.random() < 0.5)
                   else either_way(random.choice(specs), both_ways) for _ in range(args.batch)]
            sd_p = {k: v.clone() for k, v in policy.state_dict().items()}
            sd_c = {k: v.clone() for k, v in critic.state_dict().items()}
            jobs = [{"policy": sd_p, "critic": sd_c, "tier": tier, "specs": eps[w::workers], "cars": cars,
                     "max_steps": args.max_steps, "seed": random.randrange(2**31), "update": update,
                     "log_dir": args.log_dir,
                     "rewards": rewards, "speed_cap": speed_cap(update), "fragile": update <= args.first_lessons,
                     "gamma": args.gamma, "slow_traffic": args.slow_traffic, "slow_speed": args.slow_speed,
                     "senses": args.senses, "starts": starts, "league": league, "league_cars": args.league_cars, "spun_room": args.spun_room,
                     "spun_share": args.spun_share,
                     "contact_blame": args.contact_blame, "layout": args.layout} for w in range(workers)]
            parts = gather([j for j in jobs if j["specs"]])
            trajs = [t for p in parts for t in p["traj"]]
            contacts = sum(p["contacts"] for p in parts)
            damage = sum(p.get("damage", 0) for p in parts)
            walls = sum(p.get("walls", 0) for p in parts)
            passes = sum(p.get("passes", 0) for p in parts)
            O, A, LP, ADV, RET, ret_sum = [], [], [], [], [], 0.0
            for t in trajs:
                adv, ret = gae(t, args.gamma, args.lam)
                O += t["obs"]; A += t["act"]; LP += t["logp"]; ADV += adv; RET += ret
                ret_sum += sum(t["rew"])
            O = torch.tensor(O, dtype=torch.float32); A = torch.tensor(A); LP = torch.tensor(LP, dtype=torch.float32)
            ADV = torch.tensor(ADV, dtype=torch.float32); RET = torch.tensor(RET, dtype=torch.float32)
            ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
            n = len(A)
            for _ in range(args.epochs):
                perm = torch.randperm(n)
                for k in range(0, n, args.minibatch):
                    idx = perm[k:k + args.minibatch]
                    lp_new, ent = policy.log_prob_entropy(O[idx], A[idx])
                    ratio = torch.exp(lp_new - LP[idx])
                    pg = -torch.min(ratio * ADV[idx], ratio.clamp(1 - args.clip, 1 + args.clip) * ADV[idx]).mean()
                    vloss = ((critic(O[idx]) - RET[idx]) ** 2).mean()
                    loss = 0.5 * vloss if update <= warmup else pg + 0.5 * vloss - args.entropy * ent.mean()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(params, 0.5)
                    opt.step()
            nt = max(1, len(trajs))
            say(f"update {update:3d} | return {ret_sum / nt:7.1f} | steps/car {n / nt:5.0f} "
                f"| car touches {contacts / nt:4.2f} | damage {damage / nt:4.2f} | wall frames {walls / nt:5.1f} | clean passes {passes / nt:4.2f} | {time.time() - t0:5.0f}s",
                event="update", update=update, avg_return=round(ret_sum / nt, 2), avg_distance=round(n / nt * 4 * 4, 1),
                car_contacts=round(contacts / nt, 3), damage=round(damage / nt, 3), wall_frames=round(walls / nt, 1), passes=round(passes / nt, 3), elapsed=round(time.time() - t0, 1))
            checking = update % args.eval_every == 0 or update == args.updates
            if checking:
                score, res, path, preview = run_check(policy.export())
                key = (round(score, 3), -sum(r["time"] for r in res))
                saved = key > best_key
                save_brain(policy.export(), Path(args.out).with_suffix(".latest.pth"), name)   # always: the newest weights
                if saved:
                    best, best_key = score, key
                    save_brain(policy.export(), args.out, name)
                r0 = res[0]
                solo = sum(r["progress"] for r in res) / len(res)
                traf = sum(r["field_progress"] for r in res) / len(res)
                say(f"           score {score:5.2f} (alone {solo:.0%} of the race, in traffic {traf:.0%}, "
                    f"{sum(r['finished'] for r in res)}/{len(res)} finished, "
                    f"health left {sum(r['health'] for r in res) / len(res):.0%})" + ("  <- saved" if saved else ""),
                    event="eval", update=update, progress=round(max(0.0, score), 4), saved=saved, results=res, path=path, preview=preview)

        say(f"\nBest score on training tracks: {best:.2f} -> {args.out}", event="trained", best=round(max(0.0, best), 4))
        if holdout and Path(args.out).exists():
            n_in, n_out = policy.shape
            best_model = RaceBrain(n_in, N_HIDDEN, n_out, policy.layout)
            best_model.load_state_dict(load_weights(Path(args.out).with_suffix(".json")))
            _, res = evaluate(env, best_model, holdout, cars, 3000)
            prog = sum(r["progress"] for r in res) / len(res)
            say(f"Unseen tracks (seeds {args.holdout}), {cars} cars: {prog:.1%} average progress\n" + "\n".join(
                f"  {r['track']:<18} {'FINISHED' if r['finished'] else r['reason']:<10} {r['progress']:6.1%}  "
                f"touches {r['car_contacts']}" for r in res),
                event="holdout", progress=round(prog, 4), results=res)
        say("", event="done")
    finally:
        for w in pool:
            w.terminate()
        env.close()


if __name__ == "__main__":
    main()

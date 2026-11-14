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
agent/train_reinforce.py is the earlier, simpler learner (REINFORCE), kept for comparison.

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
from neuroprix.env import SimEnv
from neuroprix.brain import CLASS_ALIAS, GEN_OF_SHAPE, LEGACY, N_HIDDEN, PEDALS, TIER_OF_GEN, TIERS, RaceBrain, generation, load_weights, save_brain, state_to_obs  # noqa: E402

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
R_DAMAGE = -40.0       # per unit of health lost. Walls, rocks and other cars no longer end the race; they cost health
                       # by how hard the hit was (a head-on hit at top speed loses 0.7, a glancing scrape a few hundredths)
R_FINISH = 10.0        # completed all laps
R_CONTACT = -6.0       # touched another car (each new touch)
R_WALL = 0.0           # per physics frame spent touching a wall or rock, however gently (set per session; see training settings)
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
                   "corner": R_CORNER, "lose_place": R_LOSE_PLACE, "crash": R_CRASH, "risk": R_RISK}


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
    gained = cur["distance"] - prev["distance"]
    if gained > 0:
        r += rw["alive"] + rw["progress"] * gained
        if cur["speed"] >= rw["fast_threshold"]:
            r += rw["fast"]
    else:
        r += rw["idle"]
    if cur["finished"]:
        r += rw["finish"]
    return r


PEDAL_EXPLORE = {"n3": 1.5, "n2": 2.0, "n1": 2.0}   # curiosity (entropy bonus) multiplier on the pedal choice, per class: keeps the
                                          # brake (and N2's boost) being tried until the driver learns where it pays


class TrainPolicy(nn.Module):
    """The policy while training. N4: one steering choice (3). N3 / N2: two independent choices, steering (3)
    and pedal (N3: 3; N2: 4, with boost), each with its own exploration, so braking keeps getting tried. Exported as a
    plain RaceBrain: the 9 (N3) or 12 (N2) outputs are steer + pedal logits summed (the same distribution, the same argmax)."""

    def __init__(self, tier: str):
        super().__init__()
        self.tier = tier
        self.pedals = PEDALS.get(tier, 0)
        n_in, _ = TIERS[tier]
        self.fc1 = nn.Linear(n_in, N_HIDDEN)
        self.head = nn.Linear(N_HIDDEN, 3 + self.pedals)

    def forward(self, x):
        return self.head(torch.relu(self.fc1(x)))

    def dists(self, x):
        z = self(x)
        return (Categorical(logits=z[..., :3]), Categorical(logits=z[..., 3:])) if self.pedals else (Categorical(logits=z),)

    def sample(self, x):
        d = self.dists(x)
        if self.pedals:
            s, p = d[0].sample(), d[1].sample()
            return p * 3 + s, d[0].log_prob(s) + d[1].log_prob(p)
        a = d[0].sample()
        return a, d[0].log_prob(a)

    def log_prob_entropy(self, x, a):
        d = self.dists(x)
        if self.pedals:   # N2: twice the curiosity on the pedals, so Boost keeps being tried until the driver learns where it pays
            k = PEDAL_EXPLORE.get(self.tier, 1.0)
            return d[0].log_prob(a % 3) + d[1].log_prob(a // 3), d[0].entropy() + k * d[1].entropy()
        return d[0].log_prob(a), d[0].entropy()

    def export(self) -> RaceBrain:
        n_in, n_out = TIERS[self.tier]
        rb = RaceBrain(n_in, N_HIDDEN, n_out)
        with torch.no_grad():
            rb.fc1.load_state_dict(self.fc1.state_dict())
            W, b = self.head.weight, self.head.bias
            if self.pedals:   # action = pedal * 3 + steer
                rb.fc2.weight.copy_(torch.stack([W[s] + W[3 + p] for p in range(self.pedals) for s in range(3)]))
                rb.fc2.bias.copy_(torch.stack([b[s] + b[3 + p] for p in range(self.pedals) for s in range(3)]))
            else:
                rb.fc2.load_state_dict(self.head.state_dict())
        return rb

    def load_brain(self, sd: dict) -> None:
        """Start from an exported brain. N3: split its 9 outputs back into steer + pedal parts (exact for
        brains this trainer made, the closest such split otherwise)."""
        with torch.no_grad():
            self.fc1.weight.copy_(sd["fc1.weight"]); self.fc1.bias.copy_(sd["fc1.bias"])
            if self.pedals:
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
# twice as fast and much lighter than a browser per worker) or "browser" (the game page in headless Chromium).
# Both give the trainer the same states, checked step by step.
ENGINE = os.environ.get("NEUROPRIX_TRAIN_ENGINE", "python")


def make_env(engine: str = ENGINE, tracks: list | None = None, headed: bool = False):
    if engine == "browser" or headed:
        return RaceEnv(tracks or [{"seed": 1}], headless=not headed)
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


def collect(job: dict) -> dict:
    """Play some episodes with the given weights. Returns trajectories as plain lists (cheap to send)."""
    tier, gen = job["tier"], GEN_OF_SHAPE[TIERS[job["tier"]]]
    policy, critic = TrainPolicy(tier), Critic(TIERS[tier][0])
    policy.load_state_dict(job["policy"])
    critic.load_state_dict(job["critic"])
    random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    out, contacts, damage, walls, passes, lost = [], 0, 0.0, 0, 0, 0
    rw = job["rewards"]
    for spec in job["specs"]:
        n_cars = job["cars"]
        # slower traffic: a few copies held to a lower top speed, so there's someone to overtake. They drive the same
        # brain but their experience isn't learned from (it would teach blocking, not racing).
        n_slow = min(job.get("slow_traffic", SLOW_TRAFFIC), n_cars - 1) if n_cars > 1 else 0
        slow = set(random.sample(range(n_cars), n_slow)) if n_slow > 0 else set()
        base = job.get("slow_speed", SLOW_SPEED)
        caps = [round(min(1.0, max(0.3, base + random.uniform(-0.05, 0.05))), 3) if i in slow else 1.0 for i in range(n_cars)]
        _ENV.setup_world(spec, n_cars, random_start=True, tier=tier, speed_cap=job.get("speed_cap", 1.0),
                         fragile=job.get("fragile", False), caps=caps)   # exploring starts
        states = _ENV.reset_world()
        n = len(states)
        last_touch = [[-10**6] * n for _ in range(n)]   # step when each pair last touched
        traj = [{"obs": [], "act": [], "logp": [], "val": [], "rew": [], "last_val": 0.0} for _ in range(n)]
        def pose(s, action):
            return [round(s["x"], 1), round(s["y"], 1), round(s["angle"], 3), round(s["speed"], 3),
                    action, round(s.get("steer") or 0, 3), s.get("box")]
        traces = [[pose(s, None)] for s in states]
        gamma = job.get("gamma", 0.99)
        last_dir = [None] * n   # each car's previous steering choice: -1 left, 0 straight, +1 right
        for step in range(job["max_steps"]):
            live = [i for i, s in enumerate(states) if not (s["dead"] or s["finished"])]
            if not live:
                break
            obs = torch.stack([state_to_obs(states[i], gen) for i in live])
            with torch.no_grad():
                a, lp = policy.sample(obs)
                v = critic(obs)
            actions = [0] * n
            for k, i in enumerate(live):
                actions[i] = int(a[k])
                t = traj[i]
                t["obs"].append(obs[k].tolist()); t["act"].append(int(a[k])); t["logp"].append(float(lp[k])); t["val"].append(float(v[k]))
            nxt = _ENV.step_world(actions)
            for i in range(n):
                for j in nxt[i].get("touched", []):
                    if 0 <= j < n:
                        last_touch[i][j] = last_touch[j][i] = step
            for i in live:
                left = job["max_steps"] - step - 1
                traj[i]["rew"].append(reward_for(states[i], nxt[i], rw, left, gamma))
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
                            passes += i not in slow
                        elif after < 0:   # losing a place (its own setting; by default the same as a pass is worth)
                            traj[i]["rew"][-1] += rw.get("lose_place", -rw["overtake"])
                            lost += i not in slow
                d = actions[i] % 3 - 1   # steering part of the action (N4 and N3 alike)
                if last_dir[i] is not None:
                    traj[i]["rew"][-1] += job["rewards"]["steer_flip"] * abs(d - last_dir[i]) / 2
                last_dir[i] = d
                if i not in slow:
                    contacts += nxt[i].get("contacts", 0)
                    damage += nxt[i].get("damage", 0)
                    walls += nxt[i].get("hits", 0)
                s = nxt[i]
                traces[i].append(pose(s, actions[i]))
            states = nxt
        for i, s in enumerate(states):   # cut off by the step cap: worth what the critic says
            if traj[i]["obs"] and not (s["dead"] or s["finished"]):
                with torch.no_grad():
                    traj[i]["last_val"] = float(critic(state_to_obs(s, gen).unsqueeze(0))[0])
        _write_traces(job, spec, traces)
        out += [t for i, t in enumerate(traj) if t["obs"] and i not in slow]
    return {"traj": out, "contacts": contacts, "damage": damage, "walls": walls, "passes": passes, "lost": lost}


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
    gen = GEN_OF_SHAPE[TIERS[tier]]
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
def _pose(s: dict) -> list:
    """One replay step: x, y, heading and health (so the replay shows damage, smoke and fire); cars with pedals add
    whether they're on the brake (brake lights and tyre squeal in the replay)."""
    p = [round(s["x"], 1), round(s["y"], 1), round(s["angle"], 3), round(s.get("health", 1.0), 3)]
    if "braking" in s:
        p.append(1 if s["braking"] else 0)
    return p


def _drive(env, model, gen, tier, spec, cars, max_steps, record=None, others=None):
    """One run: car 0 drives its most likely action; any other cars drive the same brain with its natural
    randomness (identical deterministic cars would just pile into each other). Returns car 0's end state,
    the race length in px and car 0's touches. record: car 0's path; others: the other cars' paths and how
    each one ended (for the replay on the Train page)."""
    info = env.setup_world(spec, cars, tier=tier)
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
            logits = model(torch.stack([state_to_obs(states[k], gen) for k in live]))
        acts = [0] * n
        for j, k in enumerate(live):
            acts[k] = int(torch.argmax(logits[j])) if k == 0 else int(Categorical(logits=logits[j]).sample())
        states = env.step_world(acts)
        touches += states[0].get("contacts", 0)
        walls += states[0].get("hits", 0)
        steps += 1
        if record is not None:
            record.append(_pose(states[0]))
        if others is not None:
            for k in live:
                if k > 0:
                    o = others[k - 1]
                    o["path"].append(_pose(states[k]))
                    o["finished"], o["reason"] = states[k]["finished"], states[k]["reason"] or ""
    info["walls"] = walls   # (frames car 0 spent touching a wall or rock)
    return states[0], info, touches, steps


CHECK_WALLS = 1.0   # how much wall contact counts in the checks (--check-walls)


def evaluate(env, model, specs, cars, max_steps, record=None, preview=None, show=0):
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
        s0, info, _, solo_steps = _drive(env, model, gen, tier, spec, 1, max_steps, record if show_this and cars == 1 else None)
        walls = info.get("walls", 0)
        total = info["length"] * info["laps"]
        solo = min(1.0, max(0.0, s0["distance"]) / total)
        traffic, touches, steps = solo, 0, 1
        if show_this and cars == 1 and preview is not None:   # the replay is this solo drive
            preview.update(others=[], progress=round(solo, 4), finished=s0["finished"], reason=s0["reason"] or "timeout")
        if cars > 1:
            others = [] if show_this else None
            t0, _, touches, steps = _drive(env, model, gen, tier, spec, cars, max_steps, record if show_this else None, others)
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


def preview_drive(env, model, spec, cars, max_steps, record, preview) -> None:
    """The replay shown on the Train page, on a circuit that isn't part of the score (Surprise tracks sessions get a fresh
    one at every check): the field if the session trains in traffic, else the driver alone."""
    gen = generation(model)
    tier = TIER_OF_GEN.get(gen, "n4")
    others = [] if cars > 1 else None
    t0, info, _, _ = _drive(env, model, gen, tier, spec, cars, max_steps, record, others)
    total = info["length"] * info["laps"]
    preview.update(others=others or [], progress=round(min(1.0, max(0.0, t0["distance"]) / total), 4),
                   finished=t0["finished"], reason=t0["reason"] or "timeout")


# ---------- a runner's session (the trainer contract: see the SDK's docs/trainer-contract.md) ----------
SESSION_NUMBERS = {"updates", "batch", "eval_every", "traffic", "max_steps", "epochs", "minibatch", "progress", "alive",
                   "fast", "fast_threshold", "idle", "dead", "damage", "finish", "contact", "steer_flip", "corner", "risk",
                   "crash", "lose_place", "overtake", "wall", "lr", "entropy", "gamma", "lam", "clip", "start_margin",
                   "check_tracks", "check_walls"}
SESSION_SWITCHES = {"random_tracks", "one_way"}


def session_argv(path: str) -> list[str]:
    """The trainer's options for a session.json written by a runner (only known settings are taken)."""
    folder = Path(path).resolve().parent
    s = json.loads(Path(path).read_text())
    if s.get("neuroprix_session") != 1:
        sys.exit("unknown session format: update the trainer")
    argv = ["--progress-json", "--name", str(s.get("driver") or "driver")[:64],
            "--tier", CLASS_ALIAS.get(str(s.get("class", "n3")).lower(), str(s.get("class", "n3")).lower())]
    for k, v in (s.get("settings") or {}).items():
        if k in SESSION_SWITCHES:
            if v:
                argv.append("--" + k.replace("_", "-"))
        elif k in SESSION_NUMBERS:
            argv += ["--" + k.replace("_", "-"), str(float(v) if not isinstance(v, int) else v)]
        elif k == "seeds":
            argv += ["--seeds", ",".join(str(int(x)) for x in v)]
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
    ap.add_argument("--tier", type=lambda c: CLASS_ALIAS.get(c.lower(), c.lower()), choices=["n4", "n3", "n2", "n1"], default="n3",
                    help="n4: steering only (fast to train); n3: steering + pedals; n2: + battery boost and slipstream inputs")
    ap.add_argument("--traffic", type=int, default=4, help="cars per episode, all driven by this driver (1 = alone)")
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
    ap.add_argument("--wall", type=float, default=R_WALL, help="reward per physics frame touching a wall or rock (scraping)")
    ap.add_argument("--steer-flip", type=float, default=R_STEER_FLIP, help="reward for reversing the steering (left <-> right) between decisions")
    ap.add_argument("--critic-warmup", type=int, default=None, help="updates that train only the critic (default 10 when resuming)")
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
    ap.add_argument("--engine", choices=["python", "browser"], default=ENGINE,
                    help="where practice runs: the Python race engine (default, faster) or the browser game")
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
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cars = max(1, min(8, args.traffic))
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
               "corner": args.corner, "lose_place": args.lose_place, "crash": args.crash, "risk": args.risk}
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
    policy, critic = TrainPolicy(tier), Critic(TIERS[tier][0])
    if args.resume:
        try:
            sd = load_weights(args.resume)
        except ValueError as e:
            sys.exit(f"--resume: {e}")
        shape = (sd["fc1.weight"].shape[1], sd["fc2.weight"].shape[0])
        if shape == LEGACY.get(tier) and sd["fc1.weight"].shape[0] == N_HIDDEN:
            # an earlier brain of this class, without the corner-ahead sense: add those inputs switched off, so it drives
            # exactly as before and learns to use them from here
            extra = TIERS[tier][0] - shape[0]
            sd = {**sd, "fc1.weight": torch.cat([sd["fc1.weight"], torch.zeros(N_HIDDEN, extra)], 1)}
            say(f"upgrading the brain: {extra} new inputs (the corners ahead), starting switched off", event="note",
                message="upgraded the brain with the corner-ahead sense")
        if (sd["fc1.weight"].shape[1], sd["fc2.weight"].shape[0]) != TIERS[tier] or sd["fc1.weight"].shape[0] != N_HIDDEN:
            sys.exit(f"--resume needs a {tier.upper()} brain; other brains start a new one from scratch")
        policy.load_brain(sd)
        if policy.pedals and args.reopen_pedals < 1:   # a driver that stopped trying the brake (or boost) never learns it
            with torch.no_grad():                       # pays: flatten its pedal choice so every pedal gets tried again
                policy.head.weight[3:] *= args.reopen_pedals
                policy.head.bias[3:] *= args.reopen_pedals
    else:
        with torch.no_grad():   # gentle start; an N3 newborn leans a little on the throttle (pedal head: 0 = accelerate)
            policy.head.weight.mul_(0.1)
            policy.head.bias.zero_()
            if policy.pedals:
                policy.head.bias[3] += 0.5
                if policy.pedals == 4:   # N2: Boost starts as likely as the throttle
                    policy.head.bias[6] += 0.5
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
    elif args.lessons > 0 and policy.pedals:
        driving_lessons(policy, env, tier, args.lessons, specs, args.random_tracks, say, both_ways)
        warmup = max(warmup, 10)   # the critic learns the new driving before the driver starts changing
    label = "random procedural circuits" if args.random_tracks else ", ".join(env.setup_world(s, 1)["name"] for s in specs)
    say(f"Training '{name}' ({tier.upper()}, PPO, {workers} games in parallel) on: {label}\n{cars} car(s) per episode; "
        f"{args.updates} updates x {args.batch} episodes, max {args.max_steps} steps each\n",
        event="start", name=name, tracks=label, updates=args.updates, batch=args.batch, eval_spec=eval_specs[0],
        traffic=cars, learner="ppo", workers=workers, tier=tier)
    def new_pool():
        return mp.get_context("spawn").Pool(workers, initializer=_worker_init, initargs=(args.engine,))

    def gather(todo):
        """Collect experience from the game workers. A worker can die (a browser crash, a broken pipe) or hang;
        then the workers are restarted and the update is played again, so one bad game doesn't end the session."""
        nonlocal pool
        for attempt in range(3):
            try:
                return pool.map_async(collect, todo).get(timeout=args.worker_timeout)
            except Exception as e:   # noqa: BLE001 - any worker failure: restart them all and retry
                say(f"  a game worker failed ({type(e).__name__}: {str(e)[:120]}); restarting the workers",
                    event="note", message=f"game workers restarted after {type(e).__name__}")
                pool.terminate()
                pool = new_pool()
        raise RuntimeError("the game workers kept failing; stopping this session")

    pool = new_pool()
    best, best_key, t0 = -1.0, (-9.0, 0.0), time.time()
    n_checks = [0]

    def run_check(model):
        """Score the brain on the fixed test circuits (fair between checkpoints) and record the replay you watch: on a
        brand-new circuit for Surprise tracks sessions, else on the session's circuits in turn."""
        path, preview, cap = [], {}, max(2500, args.max_steps * 3)
        if args.random_tracks:
            score, res = evaluate(env, model, eval_specs, cars, cap)
            spec = {"seed": random.randrange(1_000_000)}
            preview_drive(env, model, spec, cars, cap, path, preview)
        else:
            k = n_checks[0] % len(eval_specs)
            spec = eval_specs[k]
            score, res = evaluate(env, model, eval_specs, cars, cap, record=path, preview=preview, show=k)
        preview["spec"] = spec
        n_checks[0] += 1
        return score, res, path, preview
    last_net = [0.0]

    def send_net(update: int) -> None:
        """The brain's weights now (rounded), for the Train page's live brain view: at the start, every few seconds, at checks."""
        if not args.progress_json:
            return
        last_net[0] = time.time()
        rb = policy.export()
        layer = lambda fc: {"weight": [[round(v, 4) for v in row] for row in fc.weight.tolist()],
                            "bias": [round(v, 4) for v in fc.bias.tolist()]}
        print(json.dumps({"event": "net", "update": update, "fc1": layer(rb.fc1), "fc2": layer(rb.fc2)}), flush=True)
    try:
        send_net(0)
        if args.resume:   # a session that only makes things worse keeps the starting weights
            score, res, path, preview = run_check(policy.export())
            best, best_key = score, (round(score - args.start_margin, 3), -sum(r["time"] for r in res))
            save_brain(policy.export(), args.out, name)
            say(f"starting weights: score {score:5.2f}" + (f" (a checkpoint replaces them from {score - args.start_margin:.2f})"
                                                          if args.start_margin else "") + "  <- saved",
                event="eval", update=0, progress=round(max(0.0, score), 4), saved=True, results=res, path=path, preview=preview)
        if args.resume:
            stage("Racing practice" + (": racing copies of itself, with a slower car to overtake" if cars > 1 and args.slow_traffic > 0
                                       else ": racing copies of itself" if cars > 1 else ""))
        for update in range(1, args.updates + 1):
            if update > 1 and speed_cap(update) >= 1 > speed_cap(update - 1):
                stage("Learning to race: now practising at full speed")
            # a new random circuit every episode for Surprise tracks; with gyms too, about half of each
            eps = [{"seed": random.randrange(1_000_000)} if args.random_tracks and (not specs or random.random() < 0.5)
                   else either_way(random.choice(specs), both_ways) for _ in range(args.batch)]
            sd_p = {k: v.clone() for k, v in policy.state_dict().items()}
            sd_c = {k: v.clone() for k, v in critic.state_dict().items()}
            jobs = [{"policy": sd_p, "critic": sd_c, "tier": tier, "specs": eps[w::workers], "cars": cars,
                     "max_steps": args.max_steps, "seed": random.randrange(2**31), "update": update,
                     "log_dir": args.log_dir,
                     "rewards": rewards, "speed_cap": speed_cap(update), "fragile": update <= args.first_lessons,
                     "gamma": args.gamma, "slow_traffic": args.slow_traffic, "slow_speed": args.slow_speed} for w in range(workers)]
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
            if checking or time.time() - last_net[0] >= 4.0:
                send_net(update)
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
            n_in, n_out = TIERS[tier]
            best_model = RaceBrain(n_in, N_HIDDEN, n_out)
            best_model.load_state_dict(load_weights(Path(args.out).with_suffix(".json")))
            _, res = evaluate(env, best_model, holdout, cars, 3000)
            prog = sum(r["progress"] for r in res) / len(res)
            say(f"Unseen tracks (seeds {args.holdout}), {cars} cars: {prog:.1%} average progress\n" + "\n".join(
                f"  {r['track']:<18} {'FINISHED' if r['finished'] else r['reason']:<10} {r['progress']:6.1%}  "
                f"touches {r['car_contacts']}" for r in res),
                event="holdout", progress=round(prog, 4), results=res)
        say("", event="done")
    finally:
        pool.terminate()
        env.close()


if __name__ == "__main__":
    main()

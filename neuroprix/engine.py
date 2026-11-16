"""Python port of the game physics (game/index.html), for live races run on the server.

Live scheduled races run here in real time, with every car on the circuit in one world (spectators
are never allowed on the racing surface). This mirrors the browser engine exactly (same constants, track build, sensors, radar,
car contacts, progress and death rules), so a driver behaves the way it did in training.

Nothing here steers around anything. Cars are physical objects: an N4 or N3 driver's radar reports
the others, and avoiding them is entirely up to what the driver learned in training. Touching another
car costs both of them half their speed.

Drivers race from their brain JSON (plain weight matrices written by the training sandbox), and the
forward pass is a few numpy lines: no uploaded code is ever executed here.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# The engine's version: bump it whenever the physics, the circuits or what drivers sense changes (the same number is in
# game/index.html as ENGINE_VERSION). Runners report theirs, and the Train page flags a runner whose engine is out of date
# (it would be training on different physics than the races use).
ENGINE_VERSION = "2026.11.4"   # (2026.11.4: every class senses with the sector radar)

W, H = 900, 600                # default world (training gyms); Grand Prix circuits carry their own "size"
MAX_SPEED, ACCEL, TURN_DRAG, TURN_RATE, MIN_SPEED = 5.0, 0.12, 0.975, 0.055, 1.2
CAR_L, CAR_W, RAY_LEN, ACTION_REPEAT = 18, 10, 250, 4
FPS = 60                       # physics frames per second
WRONG_WAY_PX, OIL_GRIP = 120, 0.35
RAY_ANGLES = [math.radians(d) for d in (-90, -45, 0, 45, 90)]
# generation 2: steering x pedal, contacts, radar (see PHYS in game/index.html)
BRAKE, COAST_DRAG, SCRUB, TURN_FULL_SPEED, STEER_RATE = 0.12, 0.999, 0.985, 1.2, 0.15   # gentle brake; lifting barely slows
LAT_GRIP = 0.12                    # N3/N2: most sideways acceleration the tyres hold; caps the turn rate at speed
TYRE_DRAG = 0.02                   # cornering costs speed (x grip-in-use squared)
YAW_K, DRIFT_SCRUB, SLIDE_GRIP = 1.6, 0.015, 0.7   # drift: the body turns up to YAW_K x the grip, the path bends at most the
                                                 # grip (SLIDE_GRIP of it once sliding 0.4 rad); sliding costs DRIFT_SCRUB x |sin(slip)|
WHEELBASE, MAX_STEER = 12.0, 0.6   # N3 bicycle: rear axle to front axle, front-wheel angle at full lock
CAR_R, RADAR_RANGE, RADAR_SLOTS = 5, 200, 2
# The sector radar (brain kinds 8-11, docs/observations-and-actions.md): the nearest car in each of 8 sectors round the car, all the
# way round, 4 numbers each (seen, near, closing, sideways). Sectors by bearing from the nose (+ = right): front +-15 deg,
# front diagonals to 60, sides to 120, rear diagonals to 160, rear the last 40. Chosen by comparing |ly| with lx times
# these tangents (exact literals, the same in game/index.html), never by atan2, so both engines agree on a boundary.
SECTORS, SECTOR_INPUTS = 8, 4
TAN15, TAN60, TAN20 = 0.2679491924311227, 1.7320508075688772, 0.36397023426620234
EMPTY_SECTOR = [0, 0, 0.5, 0.5]
CONTACT_SLOWDOWN = 0.5
# damage (every tier): walls, rocks and other cars cost health instead of ending the race (see PHYS in the browser)
WALL_DAMAGE, CAR_DAMAGE, BOUNCE_KEEP, DEFLECT, STALL_FRAMES = 1.3, 0.5, 0.45, 0.5, 600   # head-on at top speed: a total loss
SPIN_K, BOUNCE_BACK, STUN_FRAMES = 1.2, 0.3, 30   # a hit spins the car (radians at a square-on top-speed hit), bounces its
                                                  # motion off the wall, and stuns the controls (frames)
SCRAPE_KEEP, SCRAPE_DAMAGE = 0.35, 0.004   # wall contact: at most 35% of speed kept (slower than braking), a little health a frame
# slipstream: a car close behind another, nose to tail and pointing the same way, gets a higher top speed (less drag)
SLIP_DIST, SLIP_HALF, SLIP_ALIGN, SLIP_BOOST = 80, 7, 0.7, 0.05   # px ahead, px either side, cos of the heading gap, +5% at most
# N2 only: a battery (ERS) for the Boost pedal (see PHYS in the browser)
BOOST_TOP, BOOST_ACCEL, BOOST_DRAIN, BATTERY_RECHARGE, REGEN = 0.12, 1.5, 1 / 210, 1 / 1800, 0.0015
BOOST_MIN = 0.15   # charge needed to start a boost (then it runs until released or empty); otherwise Boost is the throttle
# races: the boost policy. Boost is locked for the first BOOST_LOCK_S seconds after lights out (set per race; 0 = never
# locked), as F1 keeps its overtaking aids off at the start. A locked Boost pedal is the throttle (as with an empty
# battery), so networks need no retraining and their inputs don't change. Qualifying and training: never locked.
BOOST_LOCK_S, BOOST_LOCK_MAX = 10.0, 120.0


def boost_lock_frames(seconds) -> int:
    """A race's boost lock (seconds after lights out; None = the default) in frames."""
    s = BOOST_LOCK_S if seconds is None else max(0.0, min(BOOST_LOCK_MAX, float(seconds)))
    return round(s * FPS)
# N1 (2.5D): hills and banking (see PHYS in the browser). Only on tracks with heights or banks ("elev", "bank"): flat
# tracks drive exactly as before. Cars with pedals only (N3 and up).
HILL_TOP, DOWNHILL_TOP, HILL_G = 2.5, 0.15, 0.3   # top speed x (1 - HILL_TOP x grade) uphill, up to +15% downhill; gravity
                                                  # along the slope, per unit of grade (a 10% slope: a quarter of the brake)
G_GAME, LOAD_MIN, LOAD_MAX = 0.025, 0.2, 1.5      # crests and dips: tyre load 1 + vertical curvature x speed² / G_GAME
BANK_GRIP = 0.12                                  # banked corners: sideways grip + this x sin(bank) turning into the bank
GRADE_SENSE, BANK_SENSE = 0.15, 0.35              # N1 senses: the grade and bank (radians) that read fully up or down
ELEV_MAX, BANK_MAX = 400.0, 25.0                  # a track's heights (px, either way) and banks (degrees) at most
PROBE = ((1, 0), (0.7071, 0.7071), (0, 1), (-0.7071, 0.7071), (-1, 0), (-0.7071, -0.7071), (0, -1), (0.7071, -0.7071))


# ---------- seeded tracks (exact port of the browser generator) ----------
def _imul(a: int, b: int) -> int:
    return (a * b) & 0xFFFFFFFF


def mulberry32(seed_value: float):
    a = int(seed_value) % 2**32          # ToInt32 of the (double) seed, as JS's `a |= 0`

    def rnd() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = _imul(a ^ (a >> 15), 1 | a)
        t = ((t + _imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return rnd


def js_round(x: float) -> int:
    return math.floor(x + 0.5)


def smooth_closed(ctrl: list) -> list:
    n, out = len(ctrl), []
    for i in range(n):
        p0, p1, p2, p3 = ctrl[(i - 1) % n], ctrl[i], ctrl[(i + 1) % n], ctrl[(i + 2) % n]
        k = max(3, math.ceil(math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / 10))
        for j in range(k):
            t = j / k
            t2, t3 = t * t, t * t * t

            def f(a, b, c, d):
                return 0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3)
            out.append([f(p0[0], p1[0], p2[0], p3[0]), f(p0[1], p1[1], p2[1], p3[1])])
    return out


def smooth_values(ctrl: list, vals: list) -> list:
    """One number per control point (height, bank), smoothed onto the same points as smooth_closed(ctrl)."""
    n, out = len(ctrl), []
    for i in range(n):
        p1, p2 = ctrl[i], ctrl[(i + 1) % n]
        a, b, c, d = vals[(i - 1) % n], vals[i], vals[(i + 1) % n], vals[(i + 2) % n]
        k = max(3, math.ceil(math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / 10))
        for j in range(k):
            t = j / k
            t2, t3 = t * t, t * t * t
            out.append(0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3))
    return out


def road_shape(pts: list, S: list, L: float, z: list, bank_deg: list) -> tuple:
    """Height, grade (dz/ds), vertical curvature (crest < 0, dip > 0) and bank (radians, + leans into the corner) at
    every point of the centreline, from 3 points either side (mirrors roadShape in the browser)."""
    n, grade, vcurv = len(pts), [], []
    for i in range(n):
        a, c = (i - 3) % n, (i + 3) % n
        d1 = S[i] - S[a]
        if d1 <= 0:
            d1 += L
        d2 = S[c] - S[i]
        if d2 <= 0:
            d2 += L
        grade.append((z[c] - z[a]) / (d1 + d2))
        vcurv.append(((z[c] - z[i]) / d2 - (z[i] - z[a]) / d1) / (0.5 * (d1 + d2)))
    return grade, vcurv, [b * math.pi / 180 for b in bank_deg]


def cumulative(pts: list):
    S = [0.0]
    for i in range(1, len(pts)):
        S.append(S[-1] + math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
    return S, S[-1] + math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1])


def point_at(pts, S, L, s):
    s = ((s % L) + L) % L
    lo, hi = 0, len(S) - 1
    while lo < hi:
        m = (lo + hi + 1) >> 1
        if S[m] <= s:
            lo = m
        else:
            hi = m - 1
    a, b = pts[lo], pts[(lo + 1) % len(pts)]
    seg = ((L if lo == len(S) - 1 else S[lo + 1]) - S[lo]) or 1
    t = (s - S[lo]) / seg
    tx, ty = (b[0] - a[0]) / seg, (b[1] - a[1]) / seg
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, -ty, tx


# ----- pit stops (scheduled races only; the browser draws the same lane: pitLane in game/index.html) -----
# A lane beside the main straight, from PIT_BEFORE px before the line to PIT_AFTER px after it, its centre PIT_GAP px
# outside the road's edge, on the inside of the circuit (or the outside, if the inside has no room). A car its coach
# has called in leaves the road at the lane's start, is driven down the lane at the speed limit (the brain has no say),
# stops at its box, is repaired, and rejoins at the lane's end. Other cars don't meet it while it's in the lane.
# Pit states (Car.pit, and the last number of a recorded step in the lane): 1 to the box, 2 at it being serviced,
# 3 leaving, 4 waiting behind its teammate at the team's box (Car.team: cars with the same team share one box).
PIT_BEFORE, PIT_AFTER, PIT_BOX = 360.0, 340.0, 90.0   # px: the lane's start before the line (its mouth clear of the whole grid), its end and the box after it
PIT_GAP, PIT_HALF, PIT_MERGE = 26.0, 16.0, 110.0      # px: lane centre beyond the road's edge, its half width; the merges
PIT_SPEED = 0.4 * MAX_SPEED                           # the lane's speed limit
PIT_BASE_S, PIT_REPAIR_S = 2.0, 6.0                   # seconds at the box: 2, plus up to 6 to repair a wreck to new
# The limit holds between two lines: a car brakes steadily from the lane's mouth to reach it at the entry line (PIT_IN
# px in), and only speeds up again past the exit line (PIT_OUT px before the lane's end); it slows gently into its box.
PIT_IN, PIT_OUT, PIT_BOX_BRAKE = 160.0, 160.0, 0.5 * BRAKE


def pit_speed(state: int, u: float, v: float, v0: float, u0: float, lane: dict, top: float) -> float:
    """A car's speed this frame in the pit lane (state 1 to the box, 3 leaving): u along the lane, v its speed now, v0
    and u0 its speed and place where it turned in, top its top speed. Mirrors pitSpeed in game/index.html."""
    if state == 1:
        if u < PIT_IN and v0 > PIT_SPEED:   # braking steadily to the limit by the entry line
            v = math.sqrt(PIT_SPEED ** 2 + (v0 ** 2 - PIT_SPEED ** 2) * max(0.0, PIT_IN - u) / max(1.0, PIT_IN - u0))
        else:
            v = max(PIT_SPEED, v - BRAKE) if v > PIT_SPEED else min(PIT_SPEED, v + ACCEL)
        return min(v, math.sqrt(2 * PIT_BOX_BRAKE * max(0.0, lane["box"] - u)) + 0.05)   # (stopping at the box)
    if u < lane["len"] - PIT_OUT:
        return min(PIT_SPEED, v + ACCEL * 0.8)
    return min(max(top, PIT_SPEED), v + ACCEL)   # (past the exit line: up to speed)


def pit_lane(T) -> dict | None:
    """Where a circuit's pit lane runs, or None if there's no room for one beside the main straight."""
    P, n, hw, L = T.pts, len(T.pts), T.width / 2, T.L
    if L < 4 * (PIT_BEFORE + PIT_AFTER):
        return None
    cx, cy = sum(p[0] for p in P) / n, sum(p[1] for p in P) / n
    s0, length = L - PIT_BEFORE, PIT_BEFORE + PIT_AFTER
    x, y, nx, ny = point_at(P, T.S, L, 0.0)
    inside = 1 if math.hypot(x + nx * 60 - cx, y + ny * 60 - cy) < math.hypot(x - nx * 60 - cx, y - ny * 60 - cy) else -1
    reach = hw + PIT_GAP + PIT_HALF
    for side in (inside, -inside):
        ok = True
        for k in range(0, int(length) + 1, 20):
            s = s0 + k
            x, y, nx, ny = point_at(P, T.S, L, s)
            ex, ey = x + nx * reach * side, y + ny * reach * side
            if not (20 <= ex <= T.W - 20 and 20 <= ey <= T.H - 20):
                ok = False
                break
            for i in range(0, n, 2):
                d = abs(T.S[i] - s % L)
                if min(d, L - d) > 2 * reach and math.hypot(P[i][0] - ex, P[i][1] - ey) < hw + 24:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return {"side": side, "off": hw + PIT_GAP, "s0": s0, "len": length, "box": PIT_BEFORE + PIT_BOX}
    return None


def _ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


LAPS_MAX = 100                # laps in a race at most (organisers confirm more than 10 on the Admin page)
SECONDS_PER_LAP = 150         # time allowed per lap before a race is called (a lap of a Grand Prix circuit takes ~40-60 s)


def race_limit_s(laps: int) -> int:
    """How long a race may run before it's called: 5 minutes, or more for longer races."""
    return max(300, SECONDS_PER_LAP * laps)


GP_SIZE = (4800, 3200)        # a Grand Prix world: 1 px is about 25 cm, so circuits run 2-3 km


def _hull(P: list) -> list:
    """Convex hull (monotone chain) of integer points, counter-clockwise."""
    P = sorted({(p[0], p[1]) for p in P})
    if len(P) < 3:
        return [list(p) for p in P]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return [list(p) for p in lo[:-1] + up[:-1]]


SLOW_CORNER = 1.3   # a corner tighter than this many track widths in radius is a slow one (a braking zone)


def _circuit_problem(pts: list, S: list, L: float, width: float, SW: int, SH: int, min_r: float = 1.4, min_slow: int = 0):
    """Why a candidate circuit can't be used (None if it's fine). min_slow: at least this many slow corners (separate
    stretches tighter than SLOW_CORNER track widths in radius: hairpins and chicanes, where a car has to brake)."""
    n = len(pts)
    for p in pts:
        if not (150 <= p[0] <= SW - 150 and 150 <= p[1] <= SH - 150):
            return "bounds"
    slow, was = 0, False
    for i in range(n):   # tightest corner: circumradius through points ~30 px either side
        a, b, c = pts[(i - 3) % n], pts[i], pts[(i + 3) % n]
        abx, aby, bcx, bcy, acx, acy = b[0] - a[0], b[1] - a[1], c[0] - b[0], c[1] - b[1], c[0] - a[0], c[1] - a[1]
        cr = abs(abx * bcy - aby * bcx)
        tight = False
        if cr > 1e-9:
            R = math.sqrt((abx * abx + aby * aby) * (bcx * bcx + bcy * bcy) * (acx * acx + acy * acy)) / (2 * cr)
            if R < width * min_r:
                return "tight"
            tight = R < width * SLOW_CORNER
        if tight and not was:
            slow += 1
        was = tight
    if slow < min_slow:
        return "gentle"
    cell = 2.4 * width
    need, grid = cell * cell, {}
    for i, p in enumerate(pts):
        grid.setdefault((math.floor(p[0] / cell), math.floor(p[1] / cell)), []).append(i)
    for i, p in enumerate(pts):   # parts of the track far apart along it must be well apart on the ground
        gx, gy = math.floor(p[0] / cell), math.floor(p[1] / cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((gx + dx, gy + dy), ()):
                    if j <= i:
                        continue
                    arc = S[j] - S[i]
                    if min(arc, L - arc) < 6 * width:
                        continue
                    ex, ey = pts[j][0] - p[0], pts[j][1] - p[1]
                    if ex * ex + ey * ey < need:
                        return "clearance"
    return None


def generate_track_spec(seed: int, cls: str | None = None) -> dict:
    """A seeded Grand Prix circuit. cls "n1": the same circuit with hills and banked corners (add_hills)."""
    spec = _generate_flat(seed)
    return add_hills(spec, seed) if str(cls or "").lower() == "n1" else spec


HILL_RISE_MAX = 200.0   # N1 circuits: at most this much (px, 50 m) from the lowest point to the highest


def add_hills(spec: dict, seed: int) -> dict:
    """N1: heights and banks for a circuit's control points (mirrors addHills in the browser). Its own random stream,
    so the circuit's layout is the same as without hills. The height is a few smooth waves around the lap, scaled so the
    steepest stretch is about a 6-11% grade; most corners are banked into the turn, the odd one is off-camber."""
    rng = mulberry32(seed + 104729)
    C, w = spec["control"], spec["width"]
    n = len(C)
    seg = [math.hypot(C[(i + 1) % n][0] - C[i][0], C[(i + 1) % n][1] - C[i][1]) for i in range(n)]
    total, acc, u = sum(seg), 0.0, []
    for i in range(n):
        u.append(acc / total)
        acc += seg[i]
    waves = [(f, 0.3 + 0.7 * rng(), rng() * 2 * math.pi) for f in (1, 2, 3, 5)]
    h = [sum(a * math.sin(2 * math.pi * f * ui + ph) for f, a, ph in waves) for ui in u]
    lo = min(h)
    h = [v - lo for v in h]
    steep = max(abs(h[(i + 1) % n] - h[i]) / (seg[i] or 1) for i in range(n)) or 1
    k = (0.05 + 0.03 * rng()) / steep   # (the smooth curve runs a little steeper than its control points)
    if max(h) * k > HILL_RISE_MAX:
        k = HILL_RISE_MAX / max(h)
    elev = [js_round(v * k * 10) / 10 for v in h]
    bank = []
    for i in range(n):
        a, b, c = C[(i - 1) % n], C[i], C[(i + 1) % n]
        abx, aby, bcx, bcy, acx, acy = b[0] - a[0], b[1] - a[1], c[0] - b[0], c[1] - b[1], c[0] - a[0], c[1] - a[1]
        cr = abs(abx * bcy - aby * bcx)
        R = 1e9 if cr < 1e-9 else math.sqrt((abx * abx + aby * aby) * (bcx * bcx + bcy * bcy) * (acx * acx + acy * acy)) / (2 * cr)
        r1, r2 = rng(), rng()
        v = 0.0
        if R < 10 * w:   # a corner
            v = 6 + 9 * r2 if r1 < 0.5 else -(3 + 4 * r2) if r1 < 0.62 else 0.0
        bank.append(js_round(v * 10) / 10)
    return {**spec, "elev": elev, "bank": bank}


# px of dead-straight road before and after the line (around the pit lane). Fixed numbers: the circuits are generated
# from them, so changing them reshapes every seeded circuit (the lane's start reaching 20 px into the bend is fine)
PIT_STRAIGHT = (340.0, PIT_AFTER + 40)


def _pit_straight(Q: list) -> list | None:
    """The control points with two more on the main straight, PIT_STRAIGHT before and after the line (Q[0]), so the
    smooth curve is exactly straight between them: the pit lane and the pit building run beside a straight road, the
    bends start beyond them. None when the straight is too short, or isn't straight (a chicane in it)."""
    o, a, b = Q[0], Q[-1], Q[1]
    da, db = math.hypot(a[0] - o[0], a[1] - o[1]), math.hypot(b[0] - o[0], b[1] - o[1])
    if da < PIT_STRAIGHT[0] + 199 or db < PIT_STRAIGHT[1] + 199:   # (1 px slack: the line can sit exactly 200 px on)
        return None
    ux, uy = (b[0] - o[0]) / db, (b[1] - o[1]) / db
    if abs((a[0] - o[0]) * uy - (a[1] - o[1]) * ux) > 1 or (a[0] - o[0]) * ux + (a[1] - o[1]) * uy > 0:
        return None
    return ([o, [o[0] + ux * PIT_STRAIGHT[1], o[1] + uy * PIT_STRAIGHT[1]]] + Q[1:]
            + [[o[0] - ux * PIT_STRAIGHT[0], o[1] - uy * PIT_STRAIGHT[0]]])


def _generate_flat(seed: int) -> dict:
    """A Grand Prix circuit (exact port of generateTrackSpec in game/index.html): long straights into
    defined corners, hairpins, sometimes a chicane; 2-3 km, never crossing or brushing itself."""
    seed = abs(int(seed or 0))
    rng = mulberry32(float(seed) * 2654435761.0 + 1.0)
    SW, SH = GP_SIZE
    width = js_round(58 + rng() * 22)
    control = None
    for _attempt in range(260):
        m = 10 + math.floor(rng() * 10)
        bw, bh = 3000 + rng() * 1300, 1800 + rng() * 950
        x0, y0 = (SW - bw) / 2, (SH - bh) / 2
        P = [[js_round(x0 + rng() * bw), js_round(y0 + rng() * bh)] for _ in range(m)]
        Hl = _hull(P)
        if len(Hl) < 4:
            continue
        cx = sum(p[0] for p in Hl) / len(Hl)
        cy = sum(p[1] for p in Hl) / len(Hl)
        Q = []
        for i in range(len(Hl)):   # bend the long sides inwards (sometimes out): infield sections
            a, b = Hl[i], Hl[(i + 1) % len(Hl)]
            Q.append([a[0], a[1]])
            dx, dy = b[0] - a[0], b[1] - a[1]
            ln = math.sqrt(dx * dx + dy * dy)
            if ln > 600 and rng() < 0.8:
                t = 0.3 + rng() * 0.4
                mx, my = a[0] + dx * t, a[1] + dy * t
                nx, ny = -dy / ln, dx / ln
                if nx * (cx - mx) + ny * (cy - my) < 0:
                    nx, ny = -nx, -ny
                amt = (rng() * 0.6 - 0.12) * ln
                Q.append([mx + nx * amt, my + ny * amt])
        for _ in range(6):         # keep corners apart so the spline doesn't knot
            for i in range(len(Q)):
                for j in range(i + 1, len(Q)):
                    dx, dy = Q[j][0] - Q[i][0], Q[j][1] - Q[i][1]
                    d = math.sqrt(dx * dx + dy * dy)
                    if d < 560:
                        k = (560 - d) / 2 / (d or 1)
                        Q[i][0] -= dx * k
                        Q[i][1] -= dy * k
                        Q[j][0] += dx * k
                        Q[j][1] += dy * k
        R = []                     # defined corners: an entry and an exit point, straights stay straight
        for i in range(len(Q)):
            pp, c, nn = Q[(i - 1) % len(Q)], Q[i], Q[(i + 1) % len(Q)]
            if rng() < 0.55:
                ax, ay, bx, by = pp[0] - c[0], pp[1] - c[1], nn[0] - c[0], nn[1] - c[1]
                d1, d2 = math.sqrt(ax * ax + ay * ay), math.sqrt(bx * bx + by * by)
                r = 220 + rng() * 140
                k1, k2 = min(0.3, r / d1), min(0.3, r / d2)
                R.append([c[0] + ax * k1, c[1] + ay * k1])
                R.append([c[0] + bx * k2, c[1] + by * k2])
            else:
                R.append([c[0], c[1]])
        Q = R
        best, bi = -1.0, 0         # start/finish line 35% along the longest straight
        for i in range(len(Q)):
            a, b = Q[i], Q[(i + 1) % len(Q)]
            dx, dy = b[0] - a[0], b[1] - a[1]
            if dx * dx + dy * dy > best:
                best, bi = dx * dx + dy * dy, i
        a, b = Q[bi], Q[(bi + 1) % len(Q)]
        ln = math.sqrt(best)       # (further along when 35% leaves too little room for the pit straight before it)
        f = 0.35 if ln * 0.35 >= PIT_STRAIGHT[0] + 200 else min(0.5, (PIT_STRAIGHT[0] + 200) / ln)
        Q = [[a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f]] + Q[bi + 1:] + Q[:bi + 1]
        if rng() < 0.6:            # a chicane on the next-longest straight
            best, ci = -1.0, -1
            for i in range(1, len(Q)):
                a, b = Q[i], Q[(i + 1) % len(Q)]
                dx, dy = b[0] - a[0], b[1] - a[1]
                if dx * dx + dy * dy > best:
                    best, ci = dx * dx + dy * dy, i
            a, b = Q[ci], Q[(ci + 1) % len(Q)]
            dx, dy = b[0] - a[0], b[1] - a[1]
            ln = math.sqrt(dx * dx + dy * dy)
            if ln > 1100:
                nx, ny, o = -dy / ln, dx / ln, width * 1.3
                c1 = [a[0] + dx * 0.45 + nx * o, a[1] + dy * 0.45 + ny * o]
                c2 = [a[0] + dx * 0.45 + dx / ln * 160 - nx * o, a[1] + dy * 0.45 + dy / ln * 160 - ny * o]
                Q = Q[:ci + 1] + [c1, c2] + Q[ci + 1:]
        if rng() < 0.5:            # either direction
            Q = [Q[0]] + Q[1:][::-1]
        Q = _pit_straight(Q)       # dead straight along the whole pit lane and building (or try another layout)
        if Q is None:
            continue
        cand = [[js_round(p[0]), js_round(p[1])] for p in Q]
        pts = smooth_closed(cand)
        S, L = cumulative(pts)
        # corners from 0.6 track widths in radius (hairpins: brake hard) up; at least 2 slow corners a lap so every
        # circuit has braking zones; late attempts relax both
        if 8000 <= L <= 15000 and _circuit_problem(pts, S, L, width, SW, SH, 0.6 if _attempt < 160 else 0.5,
                                                   2 if _attempt < 160 else 0) is None:
            control = cand
            break
    if control is None:            # practically never: a plain big oval
        control = [[js_round(SW / 2 + math.cos(i / 16 * math.pi * 2) * 1700), js_round(SH / 2 + math.sin(i / 16 * math.pi * 2) * 1000)]
                   for i in range(16)]
    grip = js_round((0.85 + rng() * 0.3) * 100) / 100
    pts = smooth_closed(control)
    S, L = cumulative(pts)
    obstacles, used = [], []
    n_rocks = 2 + math.floor(rng() * 6)
    n_oil = math.floor(rng() * 4)

    def pick_t():
        for _ in range(30):
            t = 0.06 + rng() * 0.9
            if all(abs(u - t) > 0.03 for u in used):
                used.append(t)
                return t
        return None
    for _ in range(n_rocks):
        t = pick_t()
        if t is None:
            break
        x, y, nx, ny = point_at(pts, S, L, t * L)
        side = -1 if rng() < 0.5 else 1
        off, r = side * width * (0.12 + rng() * 0.22), js_round(7 + rng() * 6)
        obstacles.append({"type": "rock", "x": js_round(x + nx * off), "y": js_round(y + ny * off), "r": r})
    for _ in range(n_oil):
        t = pick_t()
        if t is None:
            break
        x, y, nx, ny = point_at(pts, S, L, t * L)
        off = (rng() - 0.5) * width * 0.3
        obstacles.append({"type": "oil", "x": js_round(x + nx * off), "y": js_round(y + ny * off), "r": js_round(18 + rng() * 10)})
    return {"name": f"Procedural #{seed}", "seed": seed, "size": [SW, SH], "control": control, "width": width,
            "grip": grip, "laps": 2, "obstacles": obstacles}


OBSTACLE_MAX = 3.0                 # density: 0 = none, 1 = the circuit's own, up to 3x
ROCK_MEAN, OIL_MEAN = 4.5, 1.5     # a circuit's average number of each (2-7 rocks, 0-3 slicks)
OBSTACLE_GAP = 120                 # extra obstacles keep at least this far from every other one


def set_obstacle_density(spec: dict, stones: float = 1.0, oil: float = 1.0) -> dict:
    """A circuit with more or fewer stones and oil slicks than its track number gives it. 0 removes them all (a clean
    track), 1 keeps the circuit as it is, more adds extra ones (up to OBSTACLE_MAX times; a circuit that has few gets
    at least the average number times the density). Fewer keeps the first ones; extra ones are placed like the
    originals, from their own random sequence, so the same track number and densities always give the same circuit.
    Mirrors setObstacleDensity in the browser."""
    stones = min(OBSTACLE_MAX, max(0.0, float(stones)))
    oil = min(OBSTACLE_MAX, max(0.0, float(oil)))
    if stones == 1 and oil == 1:
        return spec
    rocks = [o for o in spec.get("obstacles", []) if o.get("type") != "oil"]
    oils = [o for o in spec.get("obstacles", []) if o.get("type") == "oil"]

    def target(have, k, mean):
        return js_round(k * (have if k <= 1 else max(have, mean)))
    want_r, want_o = target(len(rocks), stones, ROCK_MEAN), target(len(oils), oil, OIL_MEAN)
    rocks, oils = rocks[:want_r], oils[:want_o]
    pts = smooth_closed(spec["control"])
    S, L = cumulative(pts)
    width = spec["width"]
    rng = mulberry32(float(spec.get("seed", 0)) * 2654435761.0 + 7.0)

    def free(x, y):
        return all(math.hypot(o["x"] - x, o["y"] - y) >= OBSTACLE_GAP for o in rocks + oils)
    tries = 0
    while len(rocks) < want_r and tries < 600:
        tries += 1
        x, y, nx, ny = point_at(pts, S, L, (0.06 + rng() * 0.88) * L)
        side = -1 if rng() < 0.5 else 1
        off, r = side * width * (0.12 + rng() * 0.22), js_round(7 + rng() * 6)
        o = {"type": "rock", "x": js_round(x + nx * off), "y": js_round(y + ny * off), "r": r}
        if free(o["x"], o["y"]):
            rocks.append(o)
    tries = 0
    while len(oils) < want_o and tries < 600:
        tries += 1
        x, y, nx, ny = point_at(pts, S, L, (0.06 + rng() * 0.88) * L)
        off = (rng() - 0.5) * width * 0.3
        o = {"type": "oil", "x": js_round(x + nx * off), "y": js_round(y + ny * off), "r": js_round(18 + rng() * 10)}
        if free(o["x"], o["y"]):
            oils.append(o)
    return {**spec, "obstacles": rocks + oils, "stones": stones, "oil": oil}


# ---------- track ----------
@dataclass
class Track:
    spec: dict
    pts: list
    S: list
    L: float
    width: float
    grip: float
    laps: int
    rocks: list
    oils: list
    mask: np.ndarray            # [H, W] bool: drivable
    start: tuple                # x, y, angle
    W: int = 900
    H: int = 600
    hilly: bool = False         # N1: heights or banks (else all of these stay None)
    z: list | None = None       # height at each point (px)
    grade: list | None = None   # slope along the lap (dz/ds)
    vcurv: list | None = None   # vertical curvature: crest < 0, dip > 0 (1/px)
    bank: list | None = None    # radians, + leans into the corner (faster), - off-camber

    def on_track(self, x: float, y: float) -> bool:
        if x < 0 or y < 0 or x >= self.W or y >= self.H:
            return False
        rows = self.__dict__.get("_rows")
        if rows is None:   # the mask as plain byte rows: the same answers, several times faster to read one cell
            rows = self._rows = [bytes(r) for r in np.asarray(self.mask, dtype=np.uint8)]
            self.mask = None   # (the rows replace it: a Grand Prix world's mask is 15 MB)
        return rows[int(y)][int(x)] != 0


def world_size(spec: dict) -> tuple:
    """A track's world in px: its own "size", else the classic 900 x 600 gym."""
    size = spec.get("size") or (W, H)
    return int(max(300, min(8000, float(size[0])))), int(max(300, min(8000, float(size[1]))))


def build_track(spec: dict) -> Track:
    W, H = world_size(spec)
    control = [[max(0.0, min(W, float(p[0]))), max(0.0, min(H, float(p[1])))] for p in spec["control"]]
    width = max(30.0, min(160.0, float(spec.get("width") or 80)))
    grip = max(0.3, min(2.0, float(spec.get("grip") or 1)))
    laps = max(1, min(LAPS_MAX, int(round(float(spec.get("laps") or 2)))))
    pts = smooth_closed(control)
    S, L = cumulative(pts)
    rocks = [o for o in spec.get("obstacles", []) if o.get("type") != "oil"]
    oils = [o for o in spec.get("obstacles", []) if o.get("type") == "oil"]
    # Drivable = pixel centre within width/2 of the centreline (the browser strokes it with round
    # joins and caps), minus the rocks. Only the pixels near each segment are checked.
    half = width / 2
    xs, ys = np.arange(W) + 0.5, np.arange(H) + 0.5
    mask = np.zeros((H, W), dtype=bool)
    n = len(pts)
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        x0, x1 = int(max(0, min(ax, bx) - half - 1)), int(min(W, max(ax, bx) + half + 2))
        y0, y1 = int(max(0, min(ay, by) - half - 1)), int(min(H, max(ay, by) + half + 2))
        if x0 >= x1 or y0 >= y1:
            continue
        gx, gy = np.meshgrid(xs[x0:x1], ys[y0:y1])
        vx, vy = bx - ax, by - ay
        l2 = vx * vx + vy * vy or 1.0
        t = np.clip(((gx - ax) * vx + (gy - ay) * vy) / l2, 0, 1)
        d2 = (ax + vx * t - gx) ** 2 + (ay + vy * t - gy) ** 2
        mask[y0:y1, x0:x1] |= d2 <= half * half
    for r in rocks:
        rx, ry, rr = float(r["x"]), float(r["y"]), float(r["r"])
        x0, x1 = int(max(0, rx - rr - 1)), int(min(W, rx + rr + 2))
        y0, y1 = int(max(0, ry - rr - 1)), int(min(H, ry + rr + 2))
        if x0 >= x1 or y0 >= y1:
            continue
        gx, gy = np.meshgrid(xs[x0:x1], ys[y0:y1])
        mask[y0:y1, x0:x1] &= (gx - rx) ** 2 + (gy - ry) ** 2 > rr * rr
    start = (pts[0][0], pts[0][1], math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]))
    T = Track(spec=spec, pts=pts, S=S, L=L, width=width, grip=grip, laps=laps, rocks=rocks, oils=oils, mask=mask, start=start, W=W, H=H)
    elev, bank = spec_values(spec, "elev", ELEV_MAX), spec_values(spec, "bank", BANK_MAX)
    if any(elev) or any(bank):   # N1: the road's heights and banks
        T.hilly, T.z = True, smooth_values(control, elev)
        T.grade, T.vcurv, T.bank = road_shape(pts, S, L, T.z, smooth_values(control, bank))
    return T


def spec_values(spec: dict, key: str, cap: float) -> list:
    """A per-control-point list from a track spec (heights, banks), clamped; zeros if missing or the wrong length."""
    n, v = len(spec["control"]), spec.get(key)
    if not isinstance(v, list) or len(v) != n:
        return [0.0] * n
    out = []
    for x in v:
        try:
            f = float(x)
        except (TypeError, ValueError):
            f = 0.0
        out.append(max(-cap, min(cap, f)) if math.isfinite(f) else 0.0)
    return out


# ---------- brains ----------
# Tiers (see game/index.html): brain kind 1 = N4 classic (6 -> 3), 3 = N4 (18 -> 3, radar, steering only),
# 2 = N3 (19 -> 9, radar + wheel, steering x pedals), 4 = N2 (21 -> 12: N3 + battery and tow inputs, a Boost pedal)
OBS_V2, ACTIONS_V2 = 5 + RADAR_SLOTS * 6 + 2, 9
OBS_F3 = 5 + RADAR_SLOTS * 6 + 1
OBS_F1, ACTIONS_F1 = OBS_V2 + 2, 12
# the corner-ahead sense (kinds 5 = N3, 6 = N2): the track's sharpness at CORNER_LOOK px ahead (see the browser)
CORNER_LOOK = (30, 90, 180)
OBS_F2C, OBS_F1C = OBS_V2 + len(CORNER_LOOK), OBS_F1 + len(CORNER_LOOK)
# N1 (kind 7): N2's senses + the road's grade and bank at CORNER_LOOK px ahead + the tyres' load now
OBS_N1 = OBS_F1C + 2 * len(CORNER_LOOK) + 1
# The sector-radar generation (docs/observations-and-actions.md): the same senses with the 2-slot radar (12 inputs) swapped for the
# 8-sector one (32): kinds 8 = N4, 9 = N3, 10 = N2, 11 = N1
SECTOR_EXTRA = SECTORS * SECTOR_INPUTS - RADAR_SLOTS * 6
OBS_N4S, OBS_N3S, OBS_N2S, OBS_N1S = OBS_F3 + SECTOR_EXTRA, OBS_F2C + SECTOR_EXTRA, OBS_F1C + SECTOR_EXTRA, OBS_N1 + SECTOR_EXTRA
# The car's controls each decision are steering (left / straight / right), pedal (gas / coast / brake) and boost
# (off / on), held as one action number: pedal x 3 + steer, pedal 3 being gas with boost (boost does nothing with coast
# or brake: the car can't boost and brake at once). A brain's ACTION LAYOUT says how its outputs map onto them
# (docs/brain-format.md); its JSON declares it ("action_layout"), and a brain without one is "combined":
#   combined    one choice over every pedal x steering pair: 3 x pedals outputs (N2 / N1 12, N3 9; N4 3, steering)
#   two_head    steering (3) + pedal (pedals: N2 / N1 gas, coast, brake, boost; N3 gas, coast, brake), each chosen alone
#   three_head  steering (3) + pedal (gas, coast, brake) + boost (off, on): 8 outputs; classes with a battery only
# Only these fixed shapes are ever loaded: a brain is plain numbers, never code or an architecture of its own.
LAYOUTS = ("combined", "two_head", "three_head")


def layout_outputs(gen: int, layout: str) -> int | None:
    """How many outputs a brain of this kind has with this layout (None: the layout isn't allowed for it)."""
    pedals = 4 if is_f1(gen) else 3 if has_pedals(gen) else 0
    if layout == "combined":
        return 3 * pedals if pedals else 3
    if layout == "two_head":
        return 3 + pedals if pedals else None
    if layout == "three_head":
        return 8 if pedals == 4 else None
    return None


def heads_action(steer: int, pedal: int, boost: int) -> int:
    """The car's action (pedal x 3 + steer, pedal 3 = boost) from three heads' choices."""
    return (3 if pedal == 0 and boost == 1 else pedal) * 3 + steer



# Championship classes: N4 (steering only), N3 (+ pedals), N2 (+ battery boost and slipstream), N1 (+ hills and banked
# corners: the 2.5D class; its circuits are generate_track_spec(seed, "n1")).
CLASSES = ("n4", "n3", "n2", "n1")


def norm_class(c, default: str | None = None):
    """A class code, lower-cased ('N3' -> 'n3'); default if it isn't one."""
    c = str(c or "").strip().lower()
    return c if c in CLASSES else default


def tier_of_gen(gen: int) -> str:
    return "N1" if gen in (7, 11) else "N2" if gen in (4, 6, 10) else "N3" if gen in (2, 5, 9) else "N4"


def has_pedals(gen: int) -> bool:
    return gen in (2, 4, 5, 6, 7, 9, 10, 11)


def is_f1(gen: int) -> bool:   # (the battery and boost: N2 and N1)
    return gen in (4, 6, 7, 10, 11)


def sees_hills(gen: int) -> bool:
    return gen in (7, 11)


def sees_corners(gen: int) -> bool:
    return gen in (5, 6, 7, 9, 10, 11)


def sector_radar(gen: int) -> bool:   # (the 8-sector radar instead of the 2 nearest cars ahead)
    return gen >= 8


def corner_curv(track) -> list:
    """The track's signed sharpness at each point: width / radius capped at 1 (mirrors cornerCurv in the browser)."""
    if getattr(track, "curv", None) is not None:
        return track.curv
    P, n, out = track.pts, len(track.pts), []
    for i in range(n):
        a, b, c = P[(i - 3) % n], P[i], P[(i + 3) % n]
        abx, aby, bcx, bcy, acx, acy = b[0] - a[0], b[1] - a[1], c[0] - b[0], c[1] - b[1], c[0] - a[0], c[1] - a[1]
        cr = abx * bcy - aby * bcx
        if abs(cr) < 1e-9:
            out.append(0.0)
            continue
        R = math.sqrt((abx * abx + aby * aby) * (bcx * bcx + bcy * bcy) * (acx * acx + acy * acy)) / (2 * abs(cr))
        out.append(min(1.0, track.width / R) * (1 if cr > 0 else -1))
    track.curv = out
    return out


def corner_limits(track) -> list:
    """The fastest a car can take each point of the track (px/frame): sqrt(tyre grip x (radius + half the track
    width)). Mirrors cornerLimits in the browser. Training only (how much too fast a car is for the corners ahead)."""
    if getattr(track, "vmax", None) is not None:
        return track.vmax
    P, n, out = track.pts, len(track.pts), []
    for i in range(n):
        a, b, c = P[(i - 3) % n], P[i], P[(i + 3) % n]
        abx, aby, bcx, bcy, acx, acy = b[0] - a[0], b[1] - a[1], c[0] - b[0], c[1] - b[1], c[0] - a[0], c[1] - a[1]
        cr = abs(abx * bcy - aby * bcx)
        R = 1e9 if cr < 1e-9 else math.sqrt((abx * abx + aby * aby) * (bcx * bcx + bcy * bcy) * (acx * acx + acy * acy)) / (2 * cr)
        lat = LAT_GRIP * (track.grip or 1)
        if track.hilly:   # a banked corner holds more, an off-camber one less
            lat = max(0.2 * lat, lat + BANK_GRIP * math.sin(track.bank[i]))
        out.append(math.sqrt(lat * (R + track.width * 0.5)))
    track.vmax = out
    # segment lengths, for walking ahead along the circuit
    track.seglen = [math.hypot(P[(k + 1) % n][0] - P[k][0], P[(k + 1) % n][1] - P[k][1]) for k in range(n)]
    return out


EMPTY_CONTACT = [0, 1, 0.5, 0.5, 0.5, 0]


class Brain:
    """A trained policy from train.py's brain JSON.

    Generation 1: 6 inputs (5 wall rays + speed) -> 3 actions (left / straight / right).
    Generation 2: 19 inputs (5 wall rays, radar on the 2 nearest cars, speed, wheel) -> 9 actions
    (steering x accelerate/coast/brake).
    The kind comes from the number of inputs (and the class from it); the outputs must match the brain's declared
    action layout (LAYOUTS; none declared: "combined").
    """

    def __init__(self, data: dict):
        self.w1 = np.asarray(data["fc1"]["weight"], dtype=np.float64)
        self.b1 = np.asarray(data["fc1"]["bias"], dtype=np.float64)
        self.w2 = np.asarray(data["fc2"]["weight"], dtype=np.float64)
        self.b2 = np.asarray(data["fc2"]["bias"], dtype=np.float64)
        h, n_in = self.w1.shape if self.w1.ndim == 2 else (0, 0)
        n_out = self.w2.shape[0] if self.w2.ndim == 2 else 0
        self.layout = data.get("action_layout") or "combined"
        if self.layout not in LAYOUTS:
            raise ValueError("not a NeuroPrix brain (unknown action layout)")
        # (the kind from the inputs: each count belongs to one kind, N3 / N4 / N4 classic told apart by their combined outputs)
        gen = {6: 1, OBS_V2: 2, OBS_F3: 3, OBS_F1: 4, OBS_F2C: 5, OBS_F1C: 6, OBS_N1: 7,
               OBS_N4S: 8, OBS_N3S: 9, OBS_N2S: 10, OBS_N1S: 11}.get(n_in, 0)
        self.gen = gen if gen and layout_outputs(gen, self.layout) == n_out else 0
        self.tier = tier_of_gen(self.gen)
        if (not self.gen or self.b1.shape != (h,) or self.w2.shape != (n_out, h) or self.b2.shape != (n_out,)
                or not 1 <= h <= 512):
            raise ValueError("not a NeuroPrix brain")
        if not all(np.isfinite(m).all() for m in (self.w1, self.b1, self.w2, self.b2)):
            raise ValueError("brain contains NaN/inf")

    @classmethod
    def load(cls, path: Path) -> "Brain":
        return cls(json.loads(Path(path).read_text()))

    temperature: float = 0.0   # 0: always the most likely action (time trials). Races set it above 0 (see events.py)
    rng: random.Random | None = None

    def act(self, obs: list) -> int:
        x = np.clip(np.asarray(obs, dtype=np.float64), 0, 1)
        h = np.maximum(self.w1 @ x + self.b1, 0)
        logits = self.w2 @ h + self.b2
        if self.layout == "two_head":   # each head on its own (the same temperature for each)
            return self._pick(logits[3:]) * 3 + self._pick(logits[0:3])
        if self.layout == "three_head":
            return heads_action(self._pick(logits[0:3]), self._pick(logits[3:6]), self._pick(logits[6:8]))
        return self._pick(logits)

    def _pick(self, logits) -> int:
        if self.temperature <= 0:
            return int(np.argmax(logits))
        # temperature sampling: usually the driver's favourite move, now and then another it rates nearly as highly,
        # so the same drivers on the same circuit don't race identically every time
        z = (logits - logits.max()) / self.temperature
        p = np.exp(z); p /= p.sum()
        r, acc = (self.rng or random).random(), 0.0
        for k, pk in enumerate(p):
            acc += pk
            if r < acc:
                return k
        return len(p) - 1


# ---------- starting grid (exact port of gridSlot in the browser) ----------
GRID_FIRST, GRID_ROW, GRID_STAGGER, GRID_LAT = 22, 40, 20, 0.24   # (10 cars: the last ~211 px back, clear of the pit mouth)


def grid_slot(T, slot: int, s0: float = 0.0):
    """Two staggered columns behind the line, along the centreline (pole on the left). Returns x, y, heading."""
    back = GRID_FIRST + (slot // 2) * GRID_ROW + (slot % 2) * GRID_STAGGER
    lat = (1 if slot % 2 else -1) * T.width * GRID_LAT
    s = s0 - back
    if s < 0:
        s += T.L
    x, y, nx, ny = point_at(T.pts, T.S, T.L, s)   # (nx, ny) is the normal; the direction of travel is (ny, -nx)
    return x + nx * lat, y + ny * lat, math.atan2(-nx, ny)


GRID_SLOTS = 16


def free_grid(T) -> list:
    """The race grid: the first GRID_SLOTS slots whose whole car outline is on clear track (a rock on the grid
    takes a slot out). Same rule as freeGrid in the browser."""
    cached = getattr(T, "_grid", None)
    if cached is not None:
        return cached
    out, hl, hw, i = [], CAR_L / 2, CAR_W / 2, 0
    while len(out) < GRID_SLOTS and i < GRID_SLOTS * 3:
        x, y, a = grid_slot(T, i)
        c, s = math.cos(a), math.sin(a)
        if all(T.on_track(x + lx * c - ly * s, y + lx * s + ly * c)
               for lx, ly in ((hl, hw), (hl, -hw), (-hl, hw), (-hl, -hw), (hl, 0), (-hl, 0), (0, hw), (0, -hw))):
            out.append((x, y, a))
        i += 1
    while len(out) < GRID_SLOTS:
        out.append(grid_slot(T, len(out)))
    T._grid = out   # worked out once per track
    return out


# ---------- cars ----------
def r4(v: float) -> float:
    return js_round(v * 10000) / 10000


@dataclass
class Car:
    """Mirrors the browser's Car. Nothing here steers around anything: that's the brain's job."""
    track: Track
    brain: Brain
    name: str = "car"
    gen: int = 1
    x: float = 0.0
    y: float = 0.0
    angle: float = 0.0
    speed: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    steer: float = 0.0
    dead: bool = False
    finished: bool = False
    reason: str = ""
    frames: int = 0
    distance: float = 0.0
    max_distance: float = 0.0
    last_improve: int = 0
    seg: int = 0
    prev_s: float = 0.0
    action: int = 1
    sensors: list = field(default_factory=lambda: [0.0] * 5)
    radar: list = field(default_factory=list)
    hits: int = 0
    contacts: int = 0
    health: float = 1.0
    damage: float = 0.0
    grip_k: float = 1.0   # race conditions: today's grip (the whole field shares it)
    hold: int = 0         # frames this car waits at the start (its reaction to lights out)
    slip: float = 0.0     # slipstream this frame: 0 (clean air) to 1 (right behind another car)
    dir: float = 0.0      # N3/N2: the direction the car is travelling (differs from its heading in a drift)
    stun: int = 0         # frames left where the controls don't respond (after a hit)
    impact: float = 0.0   # hardest hit so far (1 = square-on at top speed)
    load: float = 1.0     # N1: the tyres' load (1 on the flat; less over a crest, more in a dip)
    # training only (agent/simenv.py; races leave these alone): a top-speed factor (slower traffic), whether to track
    # how risky the approach to a hit was, and which cars it touched this step
    cap_k: float = 1.0
    training: bool = False
    risk: float = 0.0
    impact_risk: float = 0.0
    touched: dict | None = None   # (a dict: kept in the order the touches happened, as in the browser)
    world_cap: float = 1.0   # the practice top speed (learner's permit; 1 = off)
    fragile: bool = False    # first lessons: any wall or rock ends the run
    # pit stops (races only): called in by its coach; in the lane (1 to the box, 2 at the box, 3 leaving)
    pit_call: bool = False
    pit: int = 0
    pit_stops: int = 0
    pit_u: float = 0.0       # px along the lane
    pit_u0: float = 0.0
    pit_d0: float = 0.0      # its distance when it turned in
    pit_lat0: float = 0.0    # where it was across the road then
    pit_timer: int = 0
    pit_heal: float = 0.0
    boost_lock: bool = False   # races: Boost is locked this frame (World.boost_from)
    # the safety car (races only: backend/safety.py): where it drives this car along the road while it's out; a wreck's
    # place on the road and the frame its recovery crane lifts it off (then it's gone: no contacts, not on the radar)
    sc_s: float | None = None
    sc_lat: float = 0.0
    sc_turn: float = 0.0
    lift_at: int | None = None
    truck_side: int = 1   # (which side of a wreck its recovery truck parks: safety.SafetyCar.truck_side)
    lifted: bool = False
    wreck_s: float | None = None
    wreck_lat: float = 0.0

    def __post_init__(self):
        self.gen = self.brain.gen

    def reset(self, slot: int | None = None) -> None:
        T = self.track
        sx, sy, sa = T.start
        if slot is None:
            self.x, self.y, self.angle = sx, sy, sa
        else:   # a slot on the starting grid
            self.x, self.y, self.angle = free_grid(T)[min(slot, GRID_SLOTS - 1)]
        self.speed, self.vx, self.vy, self.steer = 0.0, 0.0, 0.0, 0.0
        self.dead = self.finished = False
        self.sc_s, self.lift_at, self.lifted, self.wreck_s = None, None, False, None
        self.reason, self.frames, self.last_improve, self.hits, self.contacts = "", 0, 0, 0, 0
        self.health, self.damage, self.slip = 1.0, 0.0, 0.0
        self.dir = self.angle   # where it's travelling (N3/N2 drift)
        self.stun, self.impact = 0, 0.0
        self.battery, self.boosting = 1.0, False
        self.braking = False   # on the brake this frame (for brake lights and sounds in broadcasts; not physics)
        self.grip_k, self.hold, self.load = 1.0, 0, 1.0
        self.seg = self.nearest_seg(self.x, self.y)
        s0 = self.project_s(self.x, self.y)
        self.prev_s = s0
        self.distance = s0 - T.L if s0 > T.L / 2 else s0
        self.max_distance = self.distance
        self.action = 0 if has_pedals(self.gen) else 1

    def nearest_seg(self, px, py) -> int:
        P, best, bi = self.track.pts, math.inf, 0
        for i in range(len(P)):
            ax, ay = P[i]
            bx, by = P[(i + 1) % len(P)]
            vx, vy = bx - ax, by - ay
            l2 = vx * vx + vy * vy or 1
            t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / l2))
            ex, ey = ax + vx * t - px, ay + vy * t - py
            if ex * ex + ey * ey < best:
                best, bi = ex * ex + ey * ey, i
        return bi

    def project_s(self, px, py) -> float:
        P, n = self.track.pts, len(self.track.pts)
        best, bi, bt = math.inf, self.seg, 0.0
        for k in range(-6, 13):
            i = (self.seg + k) % n
            ax, ay = P[i]
            bx, by = P[(i + 1) % n]
            vx, vy = bx - ax, by - ay
            l2 = vx * vx + vy * vy or 1
            t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / l2))
            ex, ey = ax + vx * t - px, ay + vy * t - py
            d = ex * ex + ey * ey
            if d < best:
                best, bi, bt = d, i, t
        self.seg = bi
        T = self.track
        seg_len = (T.L if bi == n - 1 else T.S[bi + 1]) - T.S[bi]
        return T.S[bi] + bt * seg_len

    def sense(self, world) -> None:
        c, s = math.cos(self.angle), math.sin(self.angle)
        fx, fy = self.x + c * CAR_L / 2, self.y + s * CAR_L / 2
        self.sensors = [r4(self._cast(fx, fy, math.cos(self.angle + o), math.sin(self.angle + o)) / RAY_LEN) for o in RAY_ANGLES]
        if self.gen != 1:   # N4 classic has no radar
            self.radar = world.sectors_for(self) if sector_radar(self.gen) else world.radar_for(self)

    def _cast(self, x, y, dx, dy) -> float:
        T = self.track
        if not T.on_track(x, y):
            return 0.0
        d = 3
        while d <= RAY_LEN:
            if not T.on_track(x + dx * d, y + dy * d):
                lo, hi = d - 3, d
                for _ in range(4):
                    m = (lo + hi) / 2
                    if T.on_track(x + dx * m, y + dy * m):
                        lo = m
                    else:
                        hi = m
                return lo
            d += 3
        return float(RAY_LEN)

    def obs(self) -> list:
        o = list(self.sensors)
        if sector_radar(self.gen):   # 8 sectors x 4 (World.sectors_for)
            o += self.radar if self.radar else EMPTY_SECTOR * SECTORS
        else:
            for k in range(RADAR_SLOTS):
                o += self.radar[k][1] if k < len(self.radar) else EMPTY_CONTACT
        o.append(r4(self.speed / MAX_SPEED))
        if has_pedals(self.gen):
            o.append(r4((self.steer + 1) / 2))
        if is_f1(self.gen):   # N2: battery, tow
            o += [r4(self.battery), r4(self.slip)]
        if sees_corners(self.gen):   # the road ahead
            o += self.corner_ahead()
        if sees_hills(self.gen):   # N1: its slope and bank ahead, and the tyres' load now
            o += self.hills_ahead()
        return o

    def hills_ahead(self) -> list:
        """N1: the road's grade and bank at CORNER_LOOK px ahead (0.5 = flat), then the tyres' load (0.5 = normal)."""
        T = self.track
        if not T.hilly:
            return [0.5] * (2 * len(CORNER_LOOK)) + [r4(min(1.0, self.load / 2))]
        P, n, k, d, g, b = T.pts, len(T.pts), self.seg, 0.0, [], []
        for want in CORNER_LOOK:
            while d < want:
                k2 = (k + 1) % n
                d += math.hypot(P[k2][0] - P[k][0], P[k2][1] - P[k][1])
                k = k2
            g.append(r4(0.5 + 0.5 * max(-1.0, min(1.0, T.grade[k] / GRADE_SENSE))))
            b.append(r4(0.5 + 0.5 * max(-1.0, min(1.0, T.bank[k] / BANK_SENSE))))
        return g + b + [r4(min(1.0, self.load / 2))]

    def corner_ahead(self) -> list:
        K, P = corner_curv(self.track), self.track.pts
        n, k, d, out = len(P), self.seg, 0.0, []
        for want in CORNER_LOOK:
            while d < want:
                k2 = (k + 1) % n
                d += math.hypot(P[k2][0] - P[k][0], P[k2][1] - P[k][1])
                k = k2
            out.append(r4(0.5 + 0.5 * K[k]))
        return out

    def decide(self) -> None:
        if self.gen == 1:
            self.action = self.brain.act(self.sensors + [self.speed / MAX_SPEED])
        else:
            self.action = self.brain.act(self.obs())

    def frame(self) -> None:
        """One physics frame."""
        if self.dead or self.finished:
            return []
        if self.hold > 0:   # reacting to lights out
            self.hold -= 1
            self.frames += 1
            self.last_improve, self.vx, self.vy = self.frames, 0.0, 0.0
            return
        stunned = self.stun > 0   # just after a hit: the controls don't respond
        if stunned:
            self.stun -= 1
        T, ox, oy, oa = self.track, self.x, self.y, self.angle
        on_oil = any((self.x - o["x"]) ** 2 + (self.y - o["y"]) ** 2 < o["r"] ** 2 for o in T.oils)
        grip = T.grip * self.grip_k * (OIL_GRIP if on_oil else 1)
        top = self.top_speed()
        hill = T.hilly and has_pedals(self.gen)
        if hill:   # N1: the slope under the car (along its direction of travel), and the tyres' load over crests and dips
            k, P = self.seg, T.pts
            k2 = (k + 1) % len(P)
            dx, dy = P[k2][0] - P[k][0], P[k2][1] - P[k][1]
            along = (math.cos(self.dir) * dx + math.sin(self.dir) * dy) / (math.hypot(dx, dy) or 1)
            ge, va = T.grade[k] * along, self.speed * along
            self.load = min(LOAD_MAX, max(LOAD_MIN, 1 + T.vcurv[k] * va * va / G_GAME))
            top *= 1 - HILL_TOP * ge if ge > 0 else 1 + min(DOWNHILL_TOP, -HILL_TOP * ge)
            grip *= self.load
            traction = min(1.0, self.load)
        if self.speed > top:
            self.speed = max(top, self.speed - ACCEL) if is_f1(self.gen) else top
        if has_pedals(self.gen):
            turn, pedal = self.action % 3 - 1, self.action // 3
            if stunned:   # the wheel stays where it is, no throttle or brake
                turn, pedal = self.steer, 1
            was, self.boosting = self.boosting, False
            self.braking = pedal == 2
            if pedal == 3 and (self.boost_lock or not (self.battery > 0 and (was or self.battery >= BOOST_MIN))):   # locked or nothing to give: throttle
                pedal = 0
            tr = traction if hill else 1   # (a light car can't put its power or its brakes down)
            if pedal == 0:
                if self.speed < top:
                    self.speed = min(top, self.speed + ACCEL * tr)
            elif pedal == 3:
                self.speed = min(top * (1 + BOOST_TOP), self.speed + ACCEL * BOOST_ACCEL * tr)
                self.battery = max(0.0, self.battery - BOOST_DRAIN)
                self.boosting = True
            elif pedal == 2:
                self.speed = max(0.0, self.speed - BRAKE * tr)
                if is_f1(self.gen):
                    self.battery = min(1.0, self.battery + REGEN)
            else:
                self.speed *= COAST_DRAG
            if hill:   # gravity along the slope: coasting downhill gains speed, braking downhill takes longer
                self.speed = max(0.0, self.speed - HILL_G * ge)
            if is_f1(self.gen) and not self.boosting:   # no recharge while boosting: a boost ends when the battery is empty
                self.battery = min(1.0, self.battery + BATTERY_RECHARGE)
            self.steer += max(-STEER_RATE, min(STEER_RATE, turn - self.steer))
            if abs(self.steer) < 1e-9:
                self.steer = 0.0
            if self.steer != 0:
                self.speed *= 1 - (1 - SCRUB) * abs(self.steer)
            # Tyres and drift (same as the browser): the body turns as the wheels ask, up to YAW_K x the grip; the
            # direction of travel bends toward it by at most the grip. Past the grip the car slides: a drift.
            delta = self.steer * MAX_STEER * grip
            if self.speed != 0:
                lim = LAT_GRIP * grip / self.speed
                if hill and self.steer != 0:   # banking: more grip turning into the bank, less off-camber
                    K = corner_curv(T)[k]
                    if abs(K) > 0.05:
                        side = 1 if (self.steer > 0) == (K > 0) else -1
                        lim = max(0.2 * lim, lim + BANK_GRIP * math.sin(T.bank[k]) * side / self.speed)
                if delta != 0:
                    self.angle += max(-lim * YAW_K, min(lim * YAW_K, (self.speed / WHEELBASE) * math.tan(delta)))
                slip0 = math.atan2(math.sin(self.angle - self.dir), math.cos(self.angle - self.dir))
                held = lim * (1 - (1 - SLIDE_GRIP) * min(1.0, abs(slip0) / 0.4))
                bend = max(-held, min(held, slip0))
                use = abs(bend) / lim
                self.dir += bend
                self.speed *= 1 - TYRE_DRAG * use * use - DRIFT_SCRUB * abs(math.sin(slip0 - bend))
            else:
                self.dir = self.angle
            self.x += math.cos(self.dir) * self.speed
            self.y += math.sin(self.dir) * self.speed
        else:
            turn = None if stunned else self.action - 1
            if turn is None:
                self.speed *= COAST_DRAG
            elif turn == 0:
                self.speed = min(top, self.speed + ACCEL)
            else:
                self.speed = max(MIN_SPEED, self.speed * TURN_DRAG)
                self.angle += turn * TURN_RATE * grip
            c, s = math.cos(self.angle), math.sin(self.angle)
            self.x += c * self.speed
            self.y += s * self.speed
        hit = self.off_corner(math.cos(self.angle), math.sin(self.angle))
        if self.training:   # recent risk: how far over the corners' safe speed it has been lately
            self.risk = max(self.corner_over(), self.risk * 0.97)
        if hit:
            self.bounce(hit, ox, oy, oa)   # (sets the car's new heading, motion and stun)
        self.frames += 1
        s_now = self.project_s(self.x, self.y)
        ds = s_now - self.prev_s
        if ds > T.L / 2:
            ds -= T.L
        elif ds < -T.L / 2:
            ds += T.L
        self.prev_s = s_now
        self.distance += ds
        if self.distance > self.max_distance + 0.5:
            self.max_distance, self.last_improve = self.distance, self.frames
        if not self.dead and self.frames - self.last_improve > STALL_FRAMES:
            self.dead, self.reason = True, "stalled"
        if not self.dead and self.distance < self.max_distance - WRONG_WAY_PX:
            self.dead, self.reason = True, "wrong way"
        if not self.dead and self.distance >= T.laps * T.L:
            self.finished, self.reason = True, "finished"
        self.vx, self.vy = self.x - ox, self.y - oy

    def pit_enter(self, lane: dict) -> bool:
        """Called in and at the lane's start (and not on its last lap): into the pit lane. True if it turned in."""
        T = self.track
        pos = self.distance % T.L
        u = pos - lane["s0"]
        if not 0 <= u < 40 or self.distance + lane["len"] - u >= T.laps * T.L:
            return False
        x, y, nx, ny = point_at(T.pts, T.S, T.L, pos)
        self.pit, self.pit_call, self.pit_u, self.pit_u0, self.pit_d0 = 1, False, u, u, self.distance
        self.pit_v0 = self.speed
        self.pit_lat0 = (self.x - x) * nx + (self.y - y) * ny
        self.boosting = self.braking = False
        return True

    def _pit_lat(self, lane: dict, u: float) -> float:
        full = lane["side"] * lane["off"]
        if u < PIT_MERGE:
            return self.pit_lat0 + (full - self.pit_lat0) * _ease(u / PIT_MERGE)
        if u > lane["len"] - PIT_MERGE:
            return full * _ease((lane["len"] - u) / PIT_MERGE)
        return full

    def _teammate_at_box(self, world) -> bool:
        """Whether a teammate is being serviced at the team's box now (double stacking: this car waits behind it)."""
        team = getattr(self, "team", None)
        return bool(team) and world is not None and any(o is not self and getattr(o, "team", None) == team and o.pit == 2 for o in world.cars)

    def _start_service(self) -> None:
        self.pit = 2
        self.pit_timer = max(1, int((PIT_BASE_S + PIT_REPAIR_S * (1 - self.health)) * FPS))
        self.pit_heal = (1 - self.health) / self.pit_timer

    def pit_frame(self, lane: dict, world=None) -> None:
        """One frame in the pit lane: driven down it at the limit, stopped and repaired at the box, then out. A team has
        one box: a car that reaches it while its teammate is being serviced waits behind (pit state 4) until the
        teammate leaves ("double stacking"), then is serviced. Mirrors Car.pitFrame in the browser."""
        T, ox, oy = self.track, self.x, self.y
        u, v = self.pit_u, self.speed
        self.braking = False
        if self.pit == 1:
            v = pit_speed(1, u, v, getattr(self, "pit_v0", v), self.pit_u0, lane, self.top_speed())
            self.braking = v < self.speed - 1e-6   # (brake lights on, down to the limit and into the box)
            u += v
            if u >= lane["box"]:
                u, v = lane["box"], 0.0
                if self._teammate_at_box(world):
                    self.pit = 4
                else:
                    self._start_service()
        elif self.pit == 4:   # waiting behind its teammate
            v = 0.0
            if not self._teammate_at_box(world):
                self._start_service()
        elif self.pit == 2:
            v = 0.0
            self.pit_timer -= 1
            self.health = min(1.0, self.health + self.pit_heal)
            if self.pit_timer <= 0:
                self.health, self.pit = 1.0, 3
                if is_f1(self.gen):
                    self.battery = 1.0
        else:
            v = pit_speed(3, u, v, 0.0, 0.0, lane, self.top_speed())
            u = min(lane["len"], u + v)
        self.speed, self.pit_u = v, u
        x, y, nx, ny = point_at(T.pts, T.S, T.L, lane["s0"] + u)
        lat = self._pit_lat(lane, u)
        slope = self._pit_lat(lane, min(lane["len"], u + 1)) - lat
        self.x, self.y = x + nx * lat, y + ny * lat
        self.angle = self.dir = math.atan2(-nx, ny) + math.atan(slope)
        self.steer, self.stun, self.slip = 0.0, 0, 0.0
        self.frames += 1
        self.distance = self.pit_d0 + (u - self.pit_u0)
        self.max_distance = max(self.max_distance, self.distance)
        self.last_improve = self.frames
        self.vx, self.vy = self.x - ox, self.y - oy
        if self.pit == 3 and u >= lane["len"]:   # back on the road
            self.pit, self.pit_stops = 0, self.pit_stops + 1
            self.seg = self.nearest_seg(self.x, self.y)
            self.prev_s = self.project_s(self.x, self.y)

    def top_speed(self) -> float:
        return MAX_SPEED * (0.5 + 0.5 * self.health) * self.world_cap * self.cap_k * (1 + SLIP_BOOST * self.slip)   # world_cap, cap_k: training only

    def corner_need(self) -> float:
        """Training only: the fastest the car may go now and still brake in time for the corners ahead (inf: no
        corner in reach). Mirrors cornerNeed in the browser."""
        V = corner_limits(self.track)
        L, n = self.track.seglen, len(V)
        look = max(self.speed, MAX_SPEED) ** 2 / (2 * BRAKE) + 40   # as far as braking from top speed takes
        k, d, need = self.seg, 0.0, math.inf
        if self.track.hilly:   # braking downhill takes longer, uphill less
            G, e = self.track.grade, 0.0
            while d < look:
                need = min(need, math.sqrt(V[k] * V[k] + 2 * e))
                e += max(0.02, BRAKE + HILL_G * G[k]) * L[k]
                d += L[k]
                k = (k + 1) % n
            return need
        while d < look:
            need = min(need, math.sqrt(V[k] * V[k] + 2 * BRAKE * d))
            d += L[k]
            k = (k + 1) % n
        return need

    def corner_over(self) -> float:
        """Training only: how much too fast the car is for the corners coming up (0 = fine; 0.2 = 20% too fast)."""
        need = self.corner_need()
        return 0.0 if need == math.inf else max(0.0, self.speed / need - 1)

    def off_corner(self, c: float, s: float):
        hl, hw = CAR_L / 2, CAR_W / 2   # four corners, then the middle of each side (no rock slips between)
        for lx, ly in ((hl, hw), (hl, -hw), (-hl, hw), (-hl, -hw), (hl, 0), (-hl, 0), (0, hw), (0, -hw)):
            px, py = self.x + lx * c - ly * s, self.y + lx * s + ly * c
            if not self.track.on_track(px, py):
                return px, py
        return None

    def hurt(self, impact: float, scale: float, extra: float = 0.0) -> None:
        d = scale * impact * impact + extra
        self.health = max(0.0, self.health - d)
        self.damage += d
        if self.health <= 0 and not self.dead:
            self.dead, self.reason = True, "wrecked"

    def bounce(self, hit, ox: float, oy: float, oa: float) -> None:
        """A corner went into a wall or rock: back to where we were, lose speed and health, nose along the wall."""
        T, (px, py) = self.track, hit
        if self.fragile:   # training only ("first lessons"): any wall or rock ends the run
            self.x, self.y, self.angle = ox, oy, oa
            self.hits += 1
            self.health, self.dead, self.reason = 0.0, True, "crashed"
            return
        nx = ny = 0.0
        for dx, dy in PROBE:
            if T.on_track(px + dx * 3, py + dy * 3):
                nx += dx
                ny += dy
        n = math.hypot(nx, ny)
        if n < 1e-6:
            nx, ny = self.x - px, self.y - py
            n = math.hypot(nx, ny) or 1
        nx /= n
        ny /= n
        vx, vy = self.x - ox, self.y - oy
        v = math.hypot(vx, vy)
        into = max(0.0, -(vx * nx + vy * ny))
        self.hits += 1
        self.hurt(into / MAX_SPEED, WALL_DAMAGE, SCRAPE_DAMAGE)
        self.speed *= min(SCRAPE_KEEP, BOUNCE_KEEP * (1 - into / v)) if v > 0 else 0
        sev = into / MAX_SPEED   # how hard: 1 = square-on at top speed
        self.impact = max(self.impact, sev)
        diff = math.atan2(nx, -ny) - oa
        diff = math.atan2(math.sin(diff), math.cos(diff))
        if abs(diff) > math.pi / 2:
            diff = diff - math.pi if diff > 0 else diff + math.pi
        spin = ((diff > 0) - (diff < 0)) * SPIN_K * min(1.0, sev)   # a hard hit spins it on past the wall's direction
        if sev > 0.08:
            self.stun = max(self.stun, js_round(STUN_FRAMES * min(1.0, sev / 0.6)))
            if self.training:   # how risky the approach to this hit was
                self.impact_risk = max(self.impact_risk, self.risk)
        rvx, rvy = vx + (1 + BOUNCE_BACK) * into * nx, vy + (1 + BOUNCE_BACK) * into * ny   # the motion bounces off
        # slide along the wall; else stay put; else ease out along the normal
        sx, sy, da = ox + vx + (into + 0.5) * nx, oy + vy + (into + 0.5) * ny, oa + diff * DEFLECT + spin
        placed = False
        for x, y, a in ((sx, sy, da), (sx, sy, oa), (ox, oy, da), (ox, oy, oa)):
            self.x, self.y, self.angle = x, y, a
            if not self.off_corner(math.cos(a), math.sin(a)):
                placed = True
                break
        if not placed:
            self.x, self.y, self.angle = ox + nx, oy + ny, oa
        self.dir = math.atan2(rvy, rvx) if math.hypot(rvx, rvy) > 1e-6 else self.angle

    def pose(self) -> list:
        """[x, y, angle, distance, health, battery, boosting, braking, pit, action] per recorded step. Every car has all ten:
        battery is null below N2, boosting 0 below N2 (null for N4), braking null for N4 (no pedals: the viewers work it
        out from the motion), pit 0 out of the pit lane (1 to the box, 2 at it, 3 leaving, 4 waiting for a teammate), and
        action the move the brain chose for this step (pedal * 3 + steer; N4: 0 left, 1 straight, 2 right): the cockpit
        view turns the wheel and presses the pedals with it."""
        p = [round(self.x, 1), round(self.y, 1), round(self.angle, 3), round(self.max_distance, 1), round(self.health, 3)]
        if is_f1(self.gen):   # N2: battery and whether it's boosting (for the broadcast's battery bars and boost glow)
            p += [round(self.battery, 3), 1 if self.boosting else 0]
        elif has_pedals(self.gen):
            p += [None, 0]
        else:
            p += [None, None]
        p.append((1 if self.braking else 0) if has_pedals(self.gen) else None)   # on the brake (brake lights, tyre squeal)
        p += [self.pit or 0, int(self.action) if self.frames else 1]   # (on the grid, before its first decision: straight)
        return p


class World:
    """Cars sharing a track (mirrors the browser's World): contacts between cars, and radar."""

    def __init__(self, track: Track):
        self.track, self.cars, self.frame_no, self.touch = track, [], 0, {}
        self.pits = None   # races: the pit lane (pit_lane); cars called in use it
        self.boost_from = 0   # races: the frame from which Boost is allowed (boost_lock_frames; 0 = always)
        self.safety = None    # races: the safety car and wreck recovery (backend/safety.py; None in training)

    def add(self, car: Car, slot: int | None) -> Car:
        car.reset(slot)
        self.cars.append(car)
        return car

    def start(self) -> None:
        for c in self.cars:
            c.sense(self)

    def slip_of(self, c: Car) -> float:
        """How much tow car c gets: from 1 right behind a car ahead (in line, same direction) to 0 at SLIP_DIST.
        Mirrors World.slipOf in the browser."""
        best, ca, sa = 0.0, math.cos(c.angle), math.sin(c.angle)
        for o in self.cars:
            if o is c or o.dead or o.finished or o.pit:
                continue
            dx, dy = o.x - c.x, o.y - c.y
            f, lat = dx * ca + dy * sa, dy * ca - dx * sa
            if f <= CAR_L or f >= SLIP_DIST or abs(lat) >= SLIP_HALF or math.cos(o.angle - c.angle) <= SLIP_ALIGN:
                continue
            k = 1 - (f - CAR_L) / (SLIP_DIST - CAR_L)
            if k > best:
                best = k
        return best

    def frame(self, decide: bool = True, sense: bool = True) -> None:
        """One frame for every car. decide=False: the cars' actions are set from outside (training). sense=False:
        skip the sensors this frame (training reads them only after the last frame of a decision; physics never
        uses them)."""
        if len(self.cars) > 1:   # slipstream, from where everyone is at the start of the frame
            slips = [0.0 if (c.dead or c.finished or c.pit) else self.slip_of(c) for c in self.cars]
            for c, k in zip(self.cars, slips):
                c.slip = k
        sc = self.safety
        if sc is not None:
            sc.before(self)
        for c in self.cars:
            if c.dead or c.finished:
                continue
            if c.pit_call and not c.pit and self.pits:
                c.pit_enter(self.pits)
            if c.pit:
                c.sc_s = None   # (driven down the lane by the pit rules; the safety-car brain takes it over again where it comes out)
                c.pit_frame(self.pits, self)
                continue
            if sc is not None and sc.controls(c):   # under the safety car: driven along the road by the race
                sc.drive(c, self)
                continue
            if decide and self.frame_no % ACTION_REPEAT == 0:
                c.decide()
            c.boost_lock = self.frame_no < self.boost_from
            c.frame()
        self.resolve_contacts()
        if sc is not None:
            sc.after(self)
        if sense:
            for c in self.cars:
                if not c.finished:
                    c.sense(self)
        self.frame_no += 1

    def resolve_contacts(self) -> None:
        C, R, off = [c for c in self.cars if not c.finished and not c.pit and not c.lifted], CAR_R, CAR_L / 2 - CAR_W / 2
        calm = self.safety is not None and self.safety.active()   # (behind the safety car nobody's hurt by a nudge)

        def circles(c):
            cx, cy = math.cos(c.angle) * off, math.sin(c.angle) * off
            return ((c.x + cx, c.y + cy), (c.x - cx, c.y - cy))
        for i in range(len(C)):
            for j in range(i + 1, len(C)):
                a, b = C[i], C[j]
                if a.dead and b.dead:
                    continue
                if abs(a.x - b.x) > CAR_L + 2 or abs(a.y - b.y) > CAR_L + 2:
                    continue
                best, pa, pb = math.inf, None, None
                for p in circles(a):
                    for q in circles(b):
                        d = math.hypot(p[0] - q[0], p[1] - q[1])
                        if d < best:
                            best, pa, pb = d, p, q
                if best >= 2 * R - 1e-9:   # (cars pushed apart sit exactly 2R apart: rounding mustn't make that a touch, in either engine)
                    continue
                nx, ny, push = (pa[0] - pb[0]) / (best or 1), (pa[1] - pb[1]) / (best or 1), 2 * R - best
                sa = 0.0 if a.dead else 1.0 if b.dead else 0.5
                sb = 1 - sa
                a.x += nx * push * sa
                a.y += ny * push * sa
                b.x -= nx * push * sb
                b.y -= ny * push * sb
                if a.touched is not None:   # training: who touched whom (every frame they overlap, as in the browser)
                    a.touched[b.name] = True
                if b.touched is not None:
                    b.touched[a.name] = True
                key = a.name + "|" + b.name
                if not self.touch.get(key, -999) > self.frame_no - 30:
                    self.touch[key] = self.frame_no
                    closing = max(0.0, -((a.vx - b.vx) * nx + (a.vy - b.vy) * ny)) / MAX_SPEED
                    for car in (a, b):
                        if not car.dead and not calm:
                            car.speed *= CONTACT_SLOWDOWN
                            car.contacts += 1
                            car.hurt(closing, CAR_DAMAGE)

    def radar_for(self, car: Car) -> list:
        c, s, R, out = math.cos(car.angle), math.sin(car.angle), RADAR_RANGE, []

        def consider(ox, oy, ovx, ovy, is_car):
            dx, dy = ox - car.x, oy - car.y
            lx, ly = dx * c + dy * s, -dx * s + dy * c
            d = math.hypot(lx, ly)
            if lx < -CAR_L / 2 or d > R:
                return
            rvx, rvy = ovx - car.vx, ovy - car.vy
            vf, vl = rvx * c + rvy * s, -rvx * s + rvy * c
            out.append((d, [1, r4(min(1, max(0, lx / R))), r4(min(1, max(0, ly / R / 2 + 0.5))),
                            r4(min(1, max(0, vf / (2 * MAX_SPEED) + 0.5))), r4(min(1, max(0, vl / (2 * MAX_SPEED) + 0.5))),
                            1 if is_car else 0]))
        for o in self.cars:
            if o is not car and not o.finished and not o.pit and not o.lifted:
                consider(o.x, o.y, 0 if o.dead else o.vx, 0 if o.dead else o.vy, True)
        out.sort(key=lambda t: t[0])
        return out[:RADAR_SLOTS]

    def sectors_for(self, car: Car) -> list:
        """The sector radar (kinds 8-11): for each of the 8 sectors (front, front-left, left, rear-left, rear, rear-right,
        right, front-right), the nearest car whose centre is in it within RADAR_RANGE: seen, near (1 - distance / range),
        closing (0.5 + how fast the distance shrinks / 2 top speeds) and sideways (0.5 + its velocity relative to this
        car across its axis, + = right / 2 top speeds). An empty sector reads EMPTY_SECTOR. Flat: 32 numbers.
        Mirrors World.sectorsFor in the browser (docs/observations-and-actions.md)."""
        c, s, R = math.cos(car.angle), math.sin(car.angle), RADAR_RANGE
        best = [None] * SECTORS
        for o in self.cars:
            if o is car or o.finished or o.pit or o.lifted:
                continue
            dx, dy = o.x - car.x, o.y - car.y
            lx, ly = dx * c + dy * s, -dx * s + dy * c
            d = math.hypot(lx, ly)
            if d > R:
                continue
            ay = abs(ly)
            if lx > 0 and ay <= lx * TAN15:
                k = 0                                  # front
            elif lx > 0 and ay <= lx * TAN60:
                k = 1 if ly < 0 else 7                 # front-left / front-right
            elif lx < 0 and ay <= -lx * TAN20:
                k = 4                                  # rear
            elif lx < 0 and ay <= -lx * TAN60:
                k = 3 if ly < 0 else 5                 # rear-left / rear-right
            else:
                k = 2 if ly < 0 else 6                 # left / right
            if best[k] is not None and best[k][0] <= d:
                continue
            ovx, ovy = (0, 0) if o.dead else (o.vx, o.vy)
            rvx, rvy = ovx - car.vx, ovy - car.vy
            closing = -(dx * rvx + dy * rvy) / d if d > 0 else 0.0
            vl = -rvx * s + rvy * c
            best[k] = (d, [1, r4(min(1, max(0, 1 - d / R))), r4(min(1, max(0, closing / (2 * MAX_SPEED) + 0.5))),
                           r4(min(1, max(0, vl / (2 * MAX_SPEED) + 0.5)))])
        out = []
        for b in best:
            out += b[1] if b is not None else EMPTY_SECTOR
        return out

//! Cars sharing a track: the engine's Car and World as training uses them (no pit lane, safety car or boost lock,
//! which only races have). A port of neuroprix/engine.py, line by line: the same operations in the same order.

use crate::pym::{self, hypot, js_round, max, min, or1, pow2, r4};
use crate::track::{Track, CAR_L, CAR_W, GRID_SLOTS};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Arc;

const MAX_SPEED: f64 = 5.0;
const ACCEL: f64 = 0.12;
const TURN_DRAG: f64 = 0.975;
const TURN_RATE: f64 = 0.055;
const MIN_SPEED: f64 = 1.2;
const RAY_LEN: f64 = 250.0;
pub const ACTION_REPEAT: usize = 4;
const FPS: f64 = 60.0;
pub const WRONG_WAY_PX: f64 = 120.0;
const BEAM_HALF: f64 = CAR_W / 2.0;
const FAN_DEG: [f64; 9] = [-90.0, -60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0, 90.0];
const YAW_FULL: f64 = 3.0;
const SLIDE_MIN_SPEED: f64 = 0.1;
const PITCH_FULL: f64 = 0.15;
const ROLL_FULL: f64 = 0.3;
const SIGHT_LOOK: [i64; 3] = [30, 90, 180];
const SIGHT_FULL: [f64; 3] = [0.05, 0.10, 0.15];
const SIGHT_STEP: i64 = 6;
const BRAKE: f64 = 0.12;
const COAST_DRAG: f64 = 0.999;
const STEER_RATE: f64 = 0.15;
const LAT_GRIP: f64 = 0.12;
const WHEELBASE: f64 = 12.0;
const MAX_STEER: f64 = 0.6;
const CG_FRONT: f64 = 0.55;
const CG_H: f64 = 1.6;
const TYRE_PEAK: f64 = 0.14;
const TYRE_C: f64 = 1.3;
const TYRE_MU: f64 = 1.0;
const DRIVE: f64 = 0.05;
const POWER: f64 = 0.15;
const BRAKE_FRONT: f64 = 0.7;
const DYN_SUB: usize = 4;
const CAR_R: f64 = 5.0;
const RADAR_RANGE: f64 = 200.0;
const RADAR_FRONT: f64 = 600.0;
const HEALTH_FREE: f64 = 0.8;
const TAN15: f64 = 0.2679491924311227;
const TAN60: f64 = 1.7320508075688772;
const TAN20: f64 = 0.36397023426620234;
const SECTOR_RANGE: [f64; 8] = [
    RADAR_FRONT,
    RADAR_FRONT,
    RADAR_RANGE,
    RADAR_RANGE,
    RADAR_RANGE,
    RADAR_RANGE,
    RADAR_RANGE,
    RADAR_FRONT,
];
const EMPTY_SECTOR: [f64; 4] = [0.0, 0.0, 0.5, 0.5];
const CONTACT_SLOWDOWN: f64 = 0.5;
const WALL_DAMAGE: f64 = 1.3;
const CAR_DAMAGE: f64 = 0.5;
const BOUNCE_KEEP: f64 = 0.45;
const DEFLECT: f64 = 0.5;
const STALL_FRAMES: i64 = 600;
const SPIN_K: f64 = 1.2;
const BOUNCE_BACK: f64 = 0.3;
const STUN_FRAMES: f64 = 30.0;
const SCRAPE_KEEP: f64 = 0.35;
const SCRAPE_DAMAGE: f64 = 0.004;
const SLIP_DIST: f64 = 80.0;
const SLIP_HALF: f64 = 7.0;
const SLIP_ALIGN: f64 = 0.7;
const SLIP_BOOST: f64 = 0.05;
const BOOST_TOP: f64 = 0.12;
const BOOST_ACCEL: f64 = 1.5;
const BOOST_DRAIN: f64 = 1.0 / 210.0;
const BATTERY_RECHARGE: f64 = 1.0 / 1800.0;
const REGEN: f64 = 0.0015;
const BOOST_MIN: f64 = 0.15;
const HILL_TOP: f64 = 2.5;
const DOWNHILL_TOP: f64 = 0.15;
const HILL_G: f64 = 0.3;
const G_GAME: f64 = 0.025;
const LOAD_MIN: f64 = 0.2;
const LOAD_MAX: f64 = 1.5;
const BANK_GRIP: f64 = 0.12;
const PROBE: [(f64, f64); 8] = [
    (1.0, 0.0),
    (0.7071, 0.7071),
    (0.0, 1.0),
    (-0.7071, 0.7071),
    (-1.0, 0.0),
    (-0.7071, -0.7071),
    (0.0, -1.0),
    (0.7071, -0.7071),
];
pub const PROP_BASE: i64 = 1000;
const PROP_STEER: i64 = 7;
const PROP_PEDAL: i64 = 5;

pub fn has_pedals(gen: i64) -> bool {
    gen == 17 || gen == 18 || gen == 19
}

pub fn is_f1(gen: i64) -> bool {
    gen == 18 || gen == 19
}

fn health_speed(health: f64) -> f64 {
    if health >= HEALTH_FREE {
        1.0
    } else {
        0.5 + 0.5 * max(0.0, health) / HEALTH_FREE
    }
}

/// The engine's controls(action): (steering target, pedal, throttle, brake).
fn controls(action: i64) -> (f64, i64, f64, f64) {
    if action >= PROP_BASE {
        let a = action - PROP_BASE;
        let (s, p, b) = (
            a % PROP_STEER,
            (a / PROP_STEER) % PROP_PEDAL,
            a / (PROP_STEER * PROP_PEDAL),
        );
        let (h, m) = ((PROP_STEER - 1) / 2, (PROP_PEDAL - 1) / 2);
        let thr = max(0.0, (p - m) as f64 / m as f64);
        let brk = max(0.0, (m - p) as f64 / m as f64);
        let pedal = if b != 0 && thr > 0.0 {
            3
        } else if thr > 0.0 {
            0
        } else if brk > 0.0 {
            2
        } else {
            1
        };
        return ((s - h) as f64 / h as f64, pedal, thr, brk);
    }
    let pedal = action.div_euclid(3);
    (
        (action.rem_euclid(3) - 1) as f64,
        pedal,
        if pedal == 0 || pedal == 3 { 1.0 } else { 0.0 },
        if pedal == 2 { 1.0 } else { 0.0 },
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    None,
    Stalled,
    WrongWay,
    Finished,
    Wrecked,
    Crashed,
}

impl Reason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Reason::None => "",
            Reason::Stalled => "stalled",
            Reason::WrongWay => "wrong way",
            Reason::Finished => "finished",
            Reason::Wrecked => "wrecked",
            Reason::Crashed => "crashed",
        }
    }
}

#[derive(Clone)]
pub struct Car {
    pub gen: i64,
    pub x: f64,
    pub y: f64,
    pub angle: f64,
    pub speed: f64,
    pub vx: f64,
    pub vy: f64,
    pub steer: f64,
    pub dead: bool,
    pub finished: bool,
    pub reason: Reason,
    pub frames: i64,
    pub distance: f64,
    pub max_distance: f64,
    pub last_improve: i64,
    pub seg: usize,
    pub prev_s: f64,
    pub action: i64,
    pub sensors: [f64; 9],
    pub radar: Vec<f64>,
    pub hits: i64,
    pub contacts: i64,
    pub health: f64,
    pub damage: f64,
    pub grip_k: f64,
    pub hold: i64,
    pub slip: f64,
    pub dir: f64,
    pub stun: i64,
    pub impact: f64,
    pub load: f64,
    pub yaw: f64,
    pub cap_k: f64,
    pub training: bool,
    pub risk: f64,
    pub impact_risk: f64,
    pub touched: Vec<usize>,
    pub world_cap: f64,
    pub fragile: bool,
    pub wrong_way_px: f64,
    pub battery: f64,
    pub boosting: bool,
    pub braking: bool,
    pub yaw_r: f64,
    pub ax: f64,
    pub thr: f64,
    pub brk: f64,
}

impl Car {
    pub fn new(gen: i64) -> Car {
        Car {
            gen,
            x: 0.0,
            y: 0.0,
            angle: 0.0,
            speed: 0.0,
            vx: 0.0,
            vy: 0.0,
            steer: 0.0,
            dead: false,
            finished: false,
            reason: Reason::None,
            frames: 0,
            distance: 0.0,
            max_distance: 0.0,
            last_improve: 0,
            seg: 0,
            prev_s: 0.0,
            action: 1,
            sensors: [0.0; 9],
            radar: vec![],
            hits: 0,
            contacts: 0,
            health: 1.0,
            damage: 0.0,
            grip_k: 1.0,
            hold: 0,
            slip: 0.0,
            dir: 0.0,
            stun: 0,
            impact: 0.0,
            load: 1.0,
            yaw: 0.0,
            cap_k: 1.0,
            training: false,
            risk: 0.0,
            impact_risk: 0.0,
            touched: vec![],
            world_cap: 1.0,
            fragile: false,
            wrong_way_px: WRONG_WAY_PX,
            battery: 1.0,
            boosting: false,
            braking: false,
            yaw_r: 0.0,
            ax: 0.0,
            thr: 0.0,
            brk: 0.0,
        }
    }

    pub fn reset(&mut self, t: &Track, slot: Option<usize>) {
        match slot {
            None => {
                self.x = t.start.0;
                self.y = t.start.1;
                self.angle = t.start.2;
            }
            Some(s) => {
                let g = t.grid[s.min(GRID_SLOTS - 1)];
                self.x = g.0;
                self.y = g.1;
                self.angle = g.2;
            }
        }
        self.speed = 0.0;
        self.vx = 0.0;
        self.vy = 0.0;
        self.steer = 0.0;
        self.dead = false;
        self.finished = false;
        self.reason = Reason::None;
        self.frames = 0;
        self.last_improve = 0;
        self.hits = 0;
        self.contacts = 0;
        self.health = 1.0;
        self.damage = 0.0;
        self.slip = 0.0;
        self.dir = self.angle;
        self.stun = 0;
        self.impact = 0.0;
        self.battery = 1.0;
        self.boosting = false;
        self.braking = false;
        self.grip_k = 1.0;
        self.hold = 0;
        self.load = 1.0;
        self.yaw = 0.0;
        self.yaw_r = 0.0;
        self.ax = 0.0;
        self.thr = 0.0;
        self.brk = 0.0;
        self.wrong_way_px = WRONG_WAY_PX;
        self.seg = self.nearest_seg(t, self.x, self.y);
        let s0 = self.project_s(t, self.x, self.y);
        self.prev_s = s0;
        self.distance = if s0 > t.l / 2.0 { s0 - t.l } else { s0 };
        self.max_distance = self.distance;
        self.action = if has_pedals(self.gen) { 0 } else { 1 };
    }

    /// The env's _place: a pose, and progress from there.
    pub fn place(&mut self, t: &Track, x: f64, y: f64, a: f64) {
        self.x = x;
        self.y = y;
        self.angle = a;
        self.dir = a;
        self.seg = self.nearest_seg(t, x, y);
        let s0 = self.project_s(t, x, y);
        self.prev_s = s0;
        self.distance = if s0 > t.l / 2.0 { s0 - t.l } else { s0 };
        self.max_distance = self.distance;
    }

    pub fn nearest_seg(&self, t: &Track, px: f64, py: f64) -> usize {
        let p = &t.pts;
        let (mut best, mut bi) = (f64::INFINITY, 0usize);
        for i in 0..p.len() {
            let (ax, ay) = (p[i][0], p[i][1]);
            let (bx, by) = (p[(i + 1) % p.len()][0], p[(i + 1) % p.len()][1]);
            let (vx, vy) = (bx - ax, by - ay);
            let l2 = or1(vx * vx + vy * vy);
            let tt = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / l2));
            let (ex, ey) = (ax + vx * tt - px, ay + vy * tt - py);
            if ex * ex + ey * ey < best {
                best = ex * ex + ey * ey;
                bi = i;
            }
        }
        bi
    }

    pub fn project_s(&mut self, t: &Track, px: f64, py: f64) -> f64 {
        let (p, n) = (&t.pts, t.pts.len());
        let (mut best, mut bi, mut bt) = (f64::INFINITY, self.seg, 0.0f64);
        for k in -6i64..13 {
            let i = pym::imod(self.seg as i64 + k, n as i64) as usize;
            let (ax, ay) = (p[i][0], p[i][1]);
            let (bx, by) = (p[(i + 1) % n][0], p[(i + 1) % n][1]);
            let (vx, vy) = (bx - ax, by - ay);
            let l2 = or1(vx * vx + vy * vy);
            let tt = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / l2));
            let (ex, ey) = (ax + vx * tt - px, ay + vy * tt - py);
            let d = ex * ex + ey * ey;
            if d < best {
                best = d;
                bi = i;
                bt = tt;
            }
        }
        self.seg = bi;
        let seg_len = (if bi == n - 1 { t.l } else { t.s[bi + 1] }) - t.s[bi];
        t.s[bi] + bt * seg_len
    }

    fn cast(t: &Track, x: f64, y: f64, dx: f64, dy: f64) -> f64 {
        if !t.on_track(x, y) {
            return 0.0;
        }
        let mut d = 3.0f64;
        while d <= RAY_LEN {
            if !t.on_track(x + dx * d, y + dy * d) {
                let (mut lo, mut hi) = (d - 3.0, d);
                for _ in 0..4 {
                    let m = (lo + hi) / 2.0;
                    if t.on_track(x + dx * m, y + dy * m) {
                        lo = m;
                    } else {
                        hi = m;
                    }
                }
                return lo;
            }
            d += 3.0;
        }
        RAY_LEN
    }

    fn sense_rays(&mut self, t: &Track) {
        let (c, s) = (pym::cos(self.angle), pym::sin(self.angle));
        let (fx, fy) = (self.x + c * CAR_L / 2.0, self.y + s * CAR_L / 2.0);
        for i in 0..9 {
            let o = FAN_DEG[i] * (PI / 180.0);
            let (dx, dy) = (pym::cos(self.angle + o), pym::sin(self.angle + o));
            let mut d = Car::cast(t, fx, fy, dx, dy);
            if (3..=5).contains(&i) {
                let (px, py) = (-dy * BEAM_HALF, dx * BEAM_HALF);
                d = min(d, Car::cast(t, fx + px, fy + py, dx, dy));
                d = min(d, Car::cast(t, fx - px, fy - py, dx, dy));
            }
            self.sensors[i] = r4(d / RAY_LEN);
        }
    }

    pub fn top_speed(&self) -> f64 {
        MAX_SPEED
            * health_speed(self.health)
            * self.world_cap
            * self.cap_k
            * (1.0 + SLIP_BOOST * self.slip)
    }

    pub fn sideslip(&self) -> f64 {
        if !has_pedals(self.gen) || self.speed < SLIDE_MIN_SPEED {
            return 0.5;
        }
        let d = pym::sin(self.dir - self.angle).atan2(pym::cos(self.dir - self.angle));
        r4(0.5 + 0.5 * max(-1.0, min(1.0, d / (PI / 2.0))))
    }

    pub fn heading(&self, t: &Track) -> f64 {
        let (p, n, k) = (&t.pts, t.pts.len(), self.seg);
        let (tx, ty) = (p[(k + 1) % n][0] - p[k][0], p[(k + 1) % n][1] - p[k][1]);
        let (c, s) = (pym::cos(self.angle), pym::sin(self.angle));
        r4(0.5 + (c * ty - s * tx).atan2(c * tx + s * ty) / (2.0 * PI))
    }

    fn tilt_and_sight(&self, t: &Track, out: &mut Vec<f64>) {
        if !t.hilly {
            out.extend(std::iter::repeat(0.5).take(2 + 2 * SIGHT_LOOK.len()));
            return;
        }
        let (p, n, k) = (&t.pts, t.pts.len(), self.seg);
        let (ax, ay) = (p[k][0], p[k][1]);
        let (bx, by) = (p[(k + 1) % n][0], p[(k + 1) % n][1]);
        let sl = or1(hypot(bx - ax, by - ay));
        let (tx, ty) = ((bx - ax) / sl, (by - ay) / sl);
        let (nx, ny) = (-ty, tx);
        let tr = t.roll[k].tan();
        let (gx, gy) = (t.grade[k] * tx - tr * nx, t.grade[k] * ty - tr * ny);
        let (c, s) = (pym::cos(self.angle), pym::sin(self.angle));
        let pitch = (gx * c + gy * s).atan();
        let roll = (gx * s - gy * c).atan();
        out.push(r4(0.5 + 0.5 * max(-1.0, min(1.0, pitch / PITCH_FULL))));
        out.push(r4(0.5 + 0.5 * max(-1.0, min(1.0, roll / ROLL_FULL))));
        let nl = (gx * gx + gy * gy + 1.0).sqrt();
        let s_car = t.s[k] + (self.x - ax) * tx + (self.y - ay) * ty;
        let steps = (SIGHT_LOOK[2] / SIGHT_STEP) as usize;
        let mut ahead = Vec::with_capacity(steps);
        let mut d = SIGHT_STEP;
        while d <= SIGHT_LOOK[2] {
            ahead.push(t.road_frame(s_car + d as f64));
            d += SIGHT_STEP;
        }
        let (hl, hw) = (CAR_L / 2.0, CAR_W / 2.0);
        for side in [-1.0f64, 1.0] {
            let x0 = self.x + c * hl - s * hw * side;
            let y0 = self.y + s * hl + c * hw * side;
            let l0 = (x0 - ax) * nx + (y0 - ay) * ny;
            let h0 = t.road_point(t.s[k] + (x0 - ax) * tx + (y0 - ay) * ty, l0).2;
            let l1 = side * t.width / 2.0;
            for (dd, full) in SIGHT_LOOK.iter().zip(SIGHT_FULL.iter()) {
                let mut best = f64::NEG_INFINITY;
                for i in 0..(dd / SIGHT_STEP) as usize {
                    let (px, py, qx, qy, z, tr) = ahead[i];
                    let lat = l0 + (l1 - l0) * (i + 1) as f64 * SIGHT_STEP as f64 / *dd as f64;
                    let (vx, vy, vz) = (px + qx * lat - x0, py + qy * lat - y0, z - lat * tr - h0);
                    let v = or1((vx * vx + vy * vy + vz * vz).sqrt());
                    best = max(
                        best,
                        max(-1.0, min(1.0, (vz - gx * vx - gy * vy) / (v * nl))).asin(),
                    );
                }
                out.push(r4(0.5 + 0.5 * max(-1.0, min(1.0, best / full))));
            }
        }
    }

    /// What the brain senses (the engine's obs).
    pub fn obs(&self, t: &Track) -> Vec<f64> {
        let mut o = Vec::with_capacity(57);
        o.extend_from_slice(&self.sensors);
        if self.radar.is_empty() {
            for _ in 0..8 {
                o.extend_from_slice(&EMPTY_SECTOR);
            }
        } else {
            o.extend_from_slice(&self.radar);
        }
        o.push(r4(self.speed / MAX_SPEED));
        if has_pedals(self.gen) {
            o.push(r4((self.steer + 1.0) / 2.0));
        }
        o.push(r4(
            0.5 + 0.5 * max(-1.0, min(1.0, self.yaw * FPS / YAW_FULL))
        ));
        o.push(self.sideslip());
        o.push(r4(self.health));
        if is_f1(self.gen) {
            o.push(r4(self.battery));
            o.push(r4(self.slip));
        }
        if self.gen == 19 {
            self.tilt_and_sight(t, &mut o);
        }
        o.push(self.heading(t));
        o
    }

    pub fn corner_need(&self, t: &Track) -> f64 {
        let (v, l, n) = (&t.vmax, &t.seglen, t.vmax.len());
        let look = pow2(max(self.speed, MAX_SPEED)) / (2.0 * BRAKE) + 40.0;
        let (mut k, mut d, mut need) = (self.seg, 0.0f64, f64::INFINITY);
        if t.hilly {
            let (g, mut e) = (&t.grade, 0.0f64);
            while d < look {
                need = min(need, (v[k] * v[k] + 2.0 * e).sqrt());
                e += max(0.02, BRAKE + HILL_G * g[k]) * l[k];
                d += l[k];
                k = (k + 1) % n;
            }
            return need;
        }
        while d < look {
            need = min(need, (v[k] * v[k] + 2.0 * BRAKE * d).sqrt());
            d += l[k];
            k = (k + 1) % n;
        }
        need
    }

    pub fn corner_over(&self, t: &Track) -> f64 {
        let need = self.corner_need(t);
        if need == f64::INFINITY {
            0.0
        } else {
            max(0.0, self.speed / need - 1.0)
        }
    }

    fn off_corner(&self, t: &Track, c: f64, s: f64) -> Option<(f64, f64)> {
        let (hl, hw) = (CAR_L / 2.0, CAR_W / 2.0);
        for (lx, ly) in [
            (hl, hw),
            (hl, -hw),
            (-hl, hw),
            (-hl, -hw),
            (hl, 0.0),
            (-hl, 0.0),
            (0.0, hw),
            (0.0, -hw),
        ] {
            let (px, py) = (self.x + lx * c - ly * s, self.y + lx * s + ly * c);
            if !t.on_track(px, py) {
                return Some((px, py));
            }
        }
        None
    }

    fn hurt(&mut self, impact: f64, scale: f64, extra: f64) {
        let d = scale * impact * impact + extra;
        self.health = max(0.0, self.health - d);
        self.damage += d;
        if self.health <= 0.0 && !self.dead {
            self.dead = true;
            self.reason = Reason::Wrecked;
        }
    }

    fn bounce(&mut self, t: &Track, hit: (f64, f64), ox: f64, oy: f64, oa: f64) {
        let (px, py) = hit;
        if self.fragile {
            self.x = ox;
            self.y = oy;
            self.angle = oa;
            self.hits += 1;
            self.health = 0.0;
            self.dead = true;
            self.reason = Reason::Crashed;
            return;
        }
        let (mut nx, mut ny) = (0.0f64, 0.0f64);
        for (dx, dy) in PROBE {
            if t.on_track(px + dx * 3.0, py + dy * 3.0) {
                nx += dx;
                ny += dy;
            }
        }
        let mut n = hypot(nx, ny);
        if n < 1e-6 {
            nx = self.x - px;
            ny = self.y - py;
            n = or1(hypot(nx, ny));
        }
        nx /= n;
        ny /= n;
        let (vx, vy) = (self.x - ox, self.y - oy);
        let v = hypot(vx, vy);
        let into = max(0.0, -(vx * nx + vy * ny));
        self.hits += 1;
        self.hurt(into / MAX_SPEED, WALL_DAMAGE, SCRAPE_DAMAGE);
        self.speed *= if v > 0.0 {
            min(SCRAPE_KEEP, BOUNCE_KEEP * (1.0 - into / v))
        } else {
            0.0
        };
        let sev = into / MAX_SPEED;
        self.impact = max(self.impact, sev);
        let mut diff = nx.atan2(-ny) - oa;
        diff = pym::sin(diff).atan2(pym::cos(diff));
        if diff.abs() > PI / 2.0 {
            diff = if diff > 0.0 { diff - PI } else { diff + PI };
        }
        let sign = ((diff > 0.0) as i64 - (diff < 0.0) as i64) as f64;
        let spin = sign * SPIN_K * min(1.0, sev);
        if sev > 0.08 {
            self.stun = self
                .stun
                .max(js_round(STUN_FRAMES * min(1.0, sev / 0.6)) as i64);
            if self.training {
                self.impact_risk = max(self.impact_risk, self.risk);
            }
        }
        let (rvx, rvy) = (
            vx + (1.0 + BOUNCE_BACK) * into * nx,
            vy + (1.0 + BOUNCE_BACK) * into * ny,
        );
        let (sx, sy, da) = (
            ox + vx + (into + 0.5) * nx,
            oy + vy + (into + 0.5) * ny,
            oa + diff * DEFLECT + spin,
        );
        let mut placed = false;
        for (x, y, a) in [(sx, sy, da), (sx, sy, oa), (ox, oy, da), (ox, oy, oa)] {
            self.x = x;
            self.y = y;
            self.angle = a;
            if self.off_corner(t, pym::cos(a), pym::sin(a)).is_none() {
                placed = true;
                break;
            }
        }
        if !placed {
            self.x = ox + nx;
            self.y = oy + ny;
            self.angle = oa;
        }
        self.dir = if hypot(rvx, rvy) > 1e-6 {
            rvy.atan2(rvx)
        } else {
            self.angle
        };
        self.yaw_r = 0.0;
    }

    #[allow(clippy::too_many_arguments)]
    fn dynamics(
        &mut self,
        t: &Track,
        pedal: i64,
        top: f64,
        grip: f64,
        tr: f64,
        ge: f64,
        roll: f64,
        k: usize,
    ) {
        let l = WHEELBASE;
        let (a, b) = (CG_FRONT * l, (1.0 - CG_FRONT) * l);
        let yaw_r2 = 0.55 * 0.45 * 144.0;
        let (mut ca, mut sa) = (pym::cos(self.angle), pym::sin(self.angle));
        let rel = self.dir - self.angle;
        let (mut u, mut v, mut r) = (
            self.speed * pym::cos(rel),
            self.speed * pym::sin(rel),
            self.yaw_r,
        );
        let d = self.steer * MAX_STEER;
        let (cd, sd) = (pym::cos(d), pym::sin(d));
        let mu = LAT_GRIP * grip * TYRE_MU;
        let bb = std::hint::black_box(PI / (2.0 * TYRE_C)).tan() / TYRE_PEAK; // (the C library's tan, never folded at build time)
        let (kf, kr) = (1.0 + a * a / yaw_r2, 1.0 + b * b / yaw_r2);
        let boost_top = top * (1.0 + BOOST_TOP);
        let mut drive = 0.0;
        if pedal == 0 && u < top {
            drive = min(DRIVE, POWER / max(u, 0.1)) * tr * self.thr;
        } else if pedal == 3 && u < boost_top {
            drive = min(DRIVE * BOOST_ACCEL, POWER * BOOST_ACCEL / max(u, 0.1)) * tr * self.thr;
        }
        let brake = if pedal == 2 {
            BRAKE * tr * self.brk
        } else {
            0.0
        };
        let mut g_lat = 0.0;
        let (mut gx, mut gy) = (0.0, 0.0);
        if roll != 0.0 {
            let p = &t.pts;
            let k2 = (k + 1) % p.len();
            let (tx, ty) = (p[k2][0] - p[k][0], p[k2][1] - p[k][1]);
            let tl = or1(hypot(tx, ty));
            let (nx, ny) = (-ty / tl, tx / tl);
            g_lat = BANK_GRIP * pym::sin(roll);
            gx = nx * g_lat;
            gy = ny * g_lat;
        }
        let dt = 1.0 / DYN_SUB as f64;
        let mut ax = self.ax;
        for _ in 0..DYN_SUB {
            let fzf = max(0.1, (b - CG_H * ax) / l);
            let fzr = max(0.1, (a + CG_H * ax) / l);
            let vf = v + a * r;
            let (uw, vw) = (u * cd + vf * sd, -u * sd + vf * cd);
            let cap = mu * fzf;
            let mut fy = -cap * pym::sin(TYRE_C * (bb * vw.atan2(uw.abs())).atan());
            let mut fx = if brake != 0.0 {
                -(min(brake * BRAKE_FRONT, uw.abs() / dt)).copysign(uw)
            } else {
                0.0
            };
            let n = hypot(fx, fy);
            if n > cap {
                fx = fx * cap / n;
                fy = fy * cap / n;
            }
            if fy.abs() * dt * kf > vw.abs() {
                fy = (vw.abs() / (dt * kf)).copysign(fy);
            }
            let (ffx, ffy) = (fx * cd - fy * sd, fx * sd + fy * cd);
            let vr = v - b * r;
            let cap = mu * fzr;
            let mut ry = -cap * pym::sin(TYRE_C * (bb * vr.atan2(u.abs())).atan());
            let mut rx = drive
                - (if brake != 0.0 {
                    (min(brake * (1.0 - BRAKE_FRONT), u.abs() / dt)).copysign(u)
                } else {
                    0.0
                });
            let n = hypot(rx, ry);
            if n > cap {
                rx = rx * cap / n;
                ry = ry * cap / n;
            }
            if ry.abs() * dt * kr > vr.abs() {
                ry = (vr.abs() / (dt * kr)).copysign(ry);
            }
            let (mut fxb, mut fyb) = (ffx + rx, ffy + ry);
            if g_lat != 0.0 {
                fxb += gx * ca + gy * sa;
                fyb += -gx * sa + gy * ca;
            }
            let mz = a * ffy - b * ry;
            u += (fxb + v * r) * dt;
            v += (fyb - u * r) * dt;
            r += mz / yaw_r2 * dt;
            ax = fxb;
            self.angle += r * dt;
            ca = pym::cos(self.angle);
            sa = pym::sin(self.angle);
            self.x += (u * ca - v * sa) * dt;
            self.y += (u * sa + v * ca) * dt;
        }
        self.ax = ax;
        u *= COAST_DRAG;
        if ge != 0.0 {
            u -= HILL_G * ge;
        }
        if u > (if self.boosting { boost_top } else { top }) {
            u = if is_f1(self.gen) {
                max(top, u - ACCEL)
            } else {
                top
            };
        }
        self.yaw_r = r;
        self.speed = hypot(u, v);
        self.dir = if self.speed > 1e-9 {
            self.angle + v.atan2(u)
        } else {
            self.angle
        };
    }

    /// One physics frame (the engine's Car.frame, outside the pit lane).
    pub fn frame(&mut self, t: &Track) {
        if self.dead || self.finished {
            return;
        }
        if self.hold > 0 {
            self.hold -= 1;
            self.frames += 1;
            self.yaw = 0.0;
            self.last_improve = self.frames;
            self.vx = 0.0;
            self.vy = 0.0;
            return;
        }
        let stunned = self.stun > 0;
        if stunned {
            self.stun -= 1;
        }
        let (ox, oy, oa) = (self.x, self.y, self.angle);
        let mut grip = t.grip * self.grip_k;
        let mut top = self.top_speed();
        let hill = t.hilly && has_pedals(self.gen);
        let (mut ge, mut traction, mut k) = (0.0f64, 1.0f64, self.seg);
        if hill {
            k = self.seg;
            let p = &t.pts;
            let k2 = (k + 1) % p.len();
            let (dx, dy) = (p[k2][0] - p[k][0], p[k2][1] - p[k][1]);
            let along = (pym::cos(self.dir) * dx + pym::sin(self.dir) * dy) / or1(hypot(dx, dy));
            ge = t.grade[k] * along;
            let va = self.speed * along;
            self.load = min(LOAD_MAX, max(LOAD_MIN, 1.0 + t.vcurv[k] * va * va / G_GAME));
            top *= if ge > 0.0 {
                1.0 - HILL_TOP * ge
            } else {
                1.0 + min(DOWNHILL_TOP, -HILL_TOP * ge)
            };
            grip *= self.load;
            traction = min(1.0, self.load);
        }
        if self.speed > top {
            self.speed = if is_f1(self.gen) {
                max(top, self.speed - ACCEL)
            } else {
                top
            };
        }
        if has_pedals(self.gen) {
            let (mut turn, mut pedal, thr, brk) = controls(self.action);
            self.thr = thr;
            self.brk = brk;
            if stunned {
                turn = self.steer;
                pedal = 1;
                self.thr = 0.0;
                self.brk = 0.0;
            }
            let was = self.boosting;
            self.boosting = false;
            self.braking = pedal == 2;
            if pedal == 3 && !(self.battery > 0.0 && (was || self.battery >= BOOST_MIN)) {
                pedal = 0;
            }
            let tr = if hill { traction } else { 1.0 };
            self.steer += max(-STEER_RATE, min(STEER_RATE, turn - self.steer));
            if self.steer.abs() < 1e-9 {
                self.steer = 0.0;
            }
            if pedal == 3 {
                self.battery = max(0.0, self.battery - BOOST_DRAIN);
                self.boosting = true;
            } else if pedal == 2 && is_f1(self.gen) {
                self.battery = min(1.0, self.battery + REGEN);
            }
            if is_f1(self.gen) && !self.boosting {
                self.battery = min(1.0, self.battery + BATTERY_RECHARGE);
            }
            let roll = if hill { t.roll[k] } else { 0.0 };
            self.dynamics(
                t,
                pedal,
                top,
                grip,
                tr,
                if hill { ge } else { 0.0 },
                roll,
                if hill { k } else { self.seg },
            );
        } else {
            if stunned {
                self.speed *= COAST_DRAG;
            } else {
                let turn = self.action - 1;
                if turn == 0 {
                    self.speed = min(top, self.speed + ACCEL);
                } else {
                    self.speed = max(MIN_SPEED, self.speed * TURN_DRAG);
                    self.angle += turn as f64 * TURN_RATE * grip;
                }
            }
            let (c, s) = (pym::cos(self.angle), pym::sin(self.angle));
            self.x += c * self.speed;
            self.y += s * self.speed;
        }
        let hit = self.off_corner(t, pym::cos(self.angle), pym::sin(self.angle));
        if self.training {
            self.risk = max(self.corner_over(t), self.risk * 0.97);
        }
        if let Some(h) = hit {
            self.bounce(t, h, ox, oy, oa);
        }
        self.frames += 1;
        let s_now = self.project_s(t, self.x, self.y);
        let mut ds = s_now - self.prev_s;
        if ds > t.l / 2.0 {
            ds -= t.l;
        } else if ds < -t.l / 2.0 {
            ds += t.l;
        }
        self.prev_s = s_now;
        self.distance += ds;
        if self.distance > self.max_distance + 0.5 {
            self.max_distance = self.distance;
            self.last_improve = self.frames;
        }
        if !self.dead && self.frames - self.last_improve > STALL_FRAMES {
            self.dead = true;
            self.reason = Reason::Stalled;
        }
        if !self.dead && self.distance < self.max_distance - self.wrong_way_px {
            self.dead = true;
            self.reason = Reason::WrongWay;
        }
        if !self.dead && self.distance >= t.laps as f64 * t.l {
            self.finished = true;
            self.reason = Reason::Finished;
        }
        self.vx = self.x - ox;
        self.vy = self.y - oy;
        self.yaw = pym::sin(self.angle - oa).atan2(pym::cos(self.angle - oa));
    }
}

/// A field of cars on one track (the engine's World, as training steps it: SimEnv in neuroprix.env).
pub struct World {
    pub track: Arc<Track>,
    pub cars: Vec<Car>,
    pub frame_no: i64,
    touch: HashMap<(usize, usize), i64>,
}

impl World {
    pub fn new(track: Arc<Track>, cars: Vec<Car>) -> World {
        World {
            track,
            cars,
            frame_no: 0,
            touch: HashMap::new(),
        }
    }

    pub fn clear_touch(&mut self) {
        self.touch.clear();
    }

    fn slip_of(&self, i: usize) -> f64 {
        let c = &self.cars[i];
        let (mut best, ca, sa) = (0.0f64, pym::cos(c.angle), pym::sin(c.angle));
        for (j, o) in self.cars.iter().enumerate() {
            if j == i || o.dead || o.finished {
                continue;
            }
            let (dx, dy) = (o.x - c.x, o.y - c.y);
            let (f, lat) = (dx * ca + dy * sa, dy * ca - dx * sa);
            if f <= CAR_L
                || f >= SLIP_DIST
                || lat.abs() >= SLIP_HALF
                || pym::cos(o.angle - c.angle) <= SLIP_ALIGN
            {
                continue;
            }
            let k = 1.0 - (f - CAR_L) / (SLIP_DIST - CAR_L);
            if k > best {
                best = k;
            }
        }
        best
    }

    /// The sector radar for car i (the engine's World.sectors_for).
    pub fn sectors_for(&self, i: usize) -> Vec<f64> {
        let car = &self.cars[i];
        let (c, s) = (pym::cos(car.angle), pym::sin(car.angle));
        let mut best: [Option<(f64, [f64; 4])>; 8] = [None; 8];
        for (j, o) in self.cars.iter().enumerate() {
            if j == i || o.finished {
                continue;
            }
            let (dx, dy) = (o.x - car.x, o.y - car.y);
            let (lx, ly) = (dx * c + dy * s, -dx * s + dy * c);
            let d = hypot(lx, ly);
            if d > RADAR_FRONT {
                continue;
            }
            let ay = ly.abs();
            let k = if lx > 0.0 && ay <= lx * TAN15 {
                0
            } else if lx > 0.0 && ay <= lx * TAN60 {
                if ly < 0.0 {
                    1
                } else {
                    7
                }
            } else if lx < 0.0 && ay <= -lx * TAN20 {
                4
            } else if lx < 0.0 && ay <= -lx * TAN60 {
                if ly < 0.0 {
                    3
                } else {
                    5
                }
            } else if ly < 0.0 {
                2
            } else {
                6
            };
            let r = SECTOR_RANGE[k];
            if d > r || matches!(best[k], Some((bd, _)) if bd <= d) {
                continue;
            }
            let (ovx, ovy) = if o.dead { (0.0, 0.0) } else { (o.vx, o.vy) };
            let (rvx, rvy) = (ovx - car.vx, ovy - car.vy);
            let closing = if d > 0.0 {
                -(dx * rvx + dy * rvy) / d
            } else {
                0.0
            };
            let vl = -rvx * s + rvy * c;
            best[k] = Some((
                d,
                [
                    1.0,
                    r4(min(1.0, max(0.0, 1.0 - d / r))),
                    r4(min(1.0, max(0.0, closing / (2.0 * MAX_SPEED) + 0.5))),
                    r4(min(1.0, max(0.0, vl / (2.0 * MAX_SPEED) + 0.5))),
                ],
            ));
        }
        let mut out = Vec::with_capacity(32);
        for b in best.iter() {
            match b {
                Some((_, v)) => out.extend_from_slice(v),
                None => out.extend_from_slice(&EMPTY_SECTOR),
            }
        }
        out
    }

    pub fn sense(&mut self, i: usize) {
        let t = self.track.clone();
        self.cars[i].sense_rays(&t);
        let radar = self.sectors_for(i);
        self.cars[i].radar = radar;
    }

    fn resolve_contacts(&mut self) {
        let idx: Vec<usize> = (0..self.cars.len())
            .filter(|&i| !self.cars[i].finished)
            .collect();
        let off = CAR_L / 2.0 - CAR_W / 2.0;
        let circles = |c: &Car| {
            let (cx, cy) = (pym::cos(c.angle) * off, pym::sin(c.angle) * off);
            [(c.x + cx, c.y + cy), (c.x - cx, c.y - cy)]
        };
        for ii in 0..idx.len() {
            for jj in ii + 1..idx.len() {
                let (ia, ib) = (idx[ii], idx[jj]);
                let (a, b) = (&self.cars[ia], &self.cars[ib]);
                if a.dead && b.dead {
                    continue;
                }
                if (a.x - b.x).abs() > CAR_L + 2.0 || (a.y - b.y).abs() > CAR_L + 2.0 {
                    continue;
                }
                let (mut best, mut pa, mut pb) = (f64::INFINITY, (0.0, 0.0), (0.0, 0.0));
                for p in circles(a) {
                    for q in circles(b) {
                        let d = hypot(p.0 - q.0, p.1 - q.1);
                        if d < best {
                            best = d;
                            pa = p;
                            pb = q;
                        }
                    }
                }
                if best >= 2.0 * CAR_R - 1e-9 {
                    continue;
                }
                let (nx, ny, push) = (
                    (pa.0 - pb.0) / or1(best),
                    (pa.1 - pb.1) / or1(best),
                    2.0 * CAR_R - best,
                );
                let sa = if a.dead {
                    0.0
                } else if b.dead {
                    1.0
                } else {
                    0.5
                };
                let sb = 1.0 - sa;
                {
                    let a = &mut self.cars[ia];
                    a.x += nx * push * sa;
                    a.y += ny * push * sa;
                }
                {
                    let b = &mut self.cars[ib];
                    b.x -= nx * push * sb;
                    b.y -= ny * push * sb;
                }
                if self.cars[ia].training && !self.cars[ia].touched.contains(&ib) {
                    self.cars[ia].touched.push(ib);
                }
                if self.cars[ib].training && !self.cars[ib].touched.contains(&ia) {
                    self.cars[ib].touched.push(ia);
                }
                let key = (ia, ib);
                let last = *self.touch.get(&key).unwrap_or(&-999);
                if !(last > self.frame_no - 30) {
                    self.touch.insert(key, self.frame_no);
                    let (a, b) = (&self.cars[ia], &self.cars[ib]);
                    let closing = max(0.0, -((a.vx - b.vx) * nx + (a.vy - b.vy) * ny)) / MAX_SPEED;
                    for ci in [ia, ib] {
                        let car = &mut self.cars[ci];
                        if !car.dead {
                            car.speed *= CONTACT_SLOWDOWN;
                            car.yaw_r *= CONTACT_SLOWDOWN;
                            car.contacts += 1;
                            car.hurt(closing, CAR_DAMAGE, 0.0);
                        }
                    }
                }
            }
        }
    }

    /// One frame for every car (the engine's World.frame with decide=False, as training runs it).
    pub fn frame(&mut self, sense: bool) {
        let t = self.track.clone();
        if self.cars.len() > 1 {
            let slips: Vec<f64> = (0..self.cars.len())
                .map(|i| {
                    if self.cars[i].dead || self.cars[i].finished {
                        0.0
                    } else {
                        self.slip_of(i)
                    }
                })
                .collect();
            for (c, k) in self.cars.iter_mut().zip(slips) {
                c.slip = k;
            }
        }
        for c in self.cars.iter_mut() {
            if c.dead || c.finished {
                continue;
            }
            c.frame(&t);
        }
        self.resolve_contacts();
        if sense {
            for i in 0..self.cars.len() {
                if !self.cars[i].finished {
                    self.sense(i);
                }
            }
        }
        self.frame_no += 1;
    }

    /// One training step (the env's step_world): the actions, then ACTION_REPEAT frames, sensing after the last (and
    /// every frame for a car about to finish).
    pub fn step(&mut self, actions: &[i64]) {
        for (i, c) in self.cars.iter_mut().enumerate() {
            c.hits = 0;
            c.contacts = 0;
            c.damage = 0.0;
            c.impact = 0.0;
            c.impact_risk = 0.0;
            c.touched.clear();
            if !(c.dead || c.finished) {
                let top = if is_f1(c.gen) {
                    11
                } else if has_pedals(c.gen) {
                    8
                } else {
                    2
                };
                let a = if i < actions.len() { actions[i] } else { 0 };
                c.action = if a >= PROP_BASE && has_pedals(c.gen) {
                    a
                } else {
                    a.min(top).max(0)
                };
            }
        }
        let goal = self.track.laps as f64 * self.track.l - 30.0;
        for f in 0..ACTION_REPEAT {
            let last = f == ACTION_REPEAT - 1;
            self.frame(last);
            if !last {
                for i in 0..self.cars.len() {
                    if !self.cars[i].finished && self.cars[i].distance >= goal {
                        self.sense(i);
                    }
                }
            }
        }
    }
}

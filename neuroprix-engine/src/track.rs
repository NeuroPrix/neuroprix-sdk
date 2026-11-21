//! Circuits: the seeded Grand Prix generator (with N1's hills and banks) and a built track (centre line, track map,
//! heights, banks, corner speeds, the grid). A port of the same functions in neuroprix/engine.py, line by line.

use crate::pym::{self, hypot, js_round, max, min, or1, Mulberry32};
use std::collections::HashMap;
use std::f64::consts::PI;

pub const GP_W: f64 = 4800.0;
pub const GP_H: f64 = 3200.0;
pub const PIT_AFTER: f64 = 340.0;
pub const PIT_STRAIGHT: (f64, f64) = (340.0, PIT_AFTER + 40.0);
pub const SLOW_CORNER: f64 = 1.3;
pub const HILL_RISE_MAX: f64 = 200.0;
pub const LAT_GRIP: f64 = 0.12;
pub const DYN_CORNER: f64 = 0.88;
pub const BANK_GRIP: f64 = 0.12;
pub const BANK_FADE: f64 = 0.05;
pub const CAR_L: f64 = 18.0;
pub const CAR_W: f64 = 10.0;
pub const GRID_FIRST: i64 = 22;
pub const GRID_ROW: i64 = 40;
pub const GRID_STAGGER: i64 = 20;
pub const GRID_LAT: f64 = 0.24;
pub const GRID_SLOTS: usize = 16;

pub type P = [f64; 2];

#[derive(Clone)]
pub struct Obstacle {
    pub oil: bool,
    pub x: f64,
    pub y: f64,
    pub r: f64,
}

/// A generated circuit (every number integral except grip, elev and bank, as the Python engine makes them).
pub struct Spec {
    pub seed: i64,
    pub width: f64,
    pub grip: f64,
    pub control: Vec<P>,
    pub obstacles: Vec<Obstacle>,
    pub elev: Option<Vec<f64>>,
    pub bank: Option<Vec<f64>>,
}

fn wrap(i: i64, n: usize) -> usize {
    pym::imod(i, n as i64) as usize
}

pub fn smooth_closed(ctrl: &[P]) -> Vec<P> {
    let n = ctrl.len();
    let mut out = Vec::new();
    for i in 0..n {
        let (p0, p1, p2, p3) = (
            ctrl[wrap(i as i64 - 1, n)],
            ctrl[i],
            ctrl[(i + 1) % n],
            ctrl[(i + 2) % n],
        );
        let k = (3.0f64).max((hypot(p2[0] - p1[0], p2[1] - p1[1]) / 10.0).ceil()) as i64;
        for j in 0..k {
            let t = j as f64 / k as f64;
            let (t2, t3) = (t * t, t * t * t);
            let f = |a: f64, b: f64, c: f64, d: f64| {
                0.5 * (2.0 * b
                    + (-a + c) * t
                    + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2
                    + (-a + 3.0 * b - 3.0 * c + d) * t3)
            };
            out.push([f(p0[0], p1[0], p2[0], p3[0]), f(p0[1], p1[1], p2[1], p3[1])]);
        }
    }
    out
}

pub fn smooth_values(ctrl: &[P], vals: &[f64]) -> Vec<f64> {
    let n = ctrl.len();
    let mut out = Vec::new();
    for i in 0..n {
        let (p1, p2) = (ctrl[i], ctrl[(i + 1) % n]);
        let (a, b, c, d) = (
            vals[wrap(i as i64 - 1, n)],
            vals[i],
            vals[(i + 1) % n],
            vals[(i + 2) % n],
        );
        let k = (3.0f64).max((hypot(p2[0] - p1[0], p2[1] - p1[1]) / 10.0).ceil()) as i64;
        for j in 0..k {
            let t = j as f64 / k as f64;
            let (t2, t3) = (t * t, t * t * t);
            out.push(
                0.5 * (2.0 * b
                    + (-a + c) * t
                    + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2
                    + (-a + 3.0 * b - 3.0 * c + d) * t3),
            );
        }
    }
    out
}

pub fn cumulative(pts: &[P]) -> (Vec<f64>, f64) {
    let mut s = vec![0.0];
    for i in 1..pts.len() {
        let last = *s.last().unwrap();
        s.push(last + hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
    }
    let n = pts.len();
    let l = s[n - 1] + hypot(pts[0][0] - pts[n - 1][0], pts[0][1] - pts[n - 1][1]);
    (s, l)
}

/// The engine's point_at: (x, y, nx, ny) at s px round the lap.
pub fn point_at(pts: &[P], s_: &[f64], l: f64, s: f64) -> (f64, f64, f64, f64) {
    let s = pym::fmod(pym::fmod(s, l) + l, l);
    let (mut lo, mut hi) = (0usize, s_.len() - 1);
    while lo < hi {
        let m = (lo + hi + 1) >> 1;
        if s_[m] <= s {
            lo = m;
        } else {
            hi = m - 1;
        }
    }
    let (a, b) = (pts[lo], pts[(lo + 1) % pts.len()]);
    let seg = or1((if lo == s_.len() - 1 { l } else { s_[lo + 1] }) - s_[lo]);
    let t = (s - s_[lo]) / seg;
    let (tx, ty) = ((b[0] - a[0]) / seg, (b[1] - a[1]) / seg);
    (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, -ty, tx)
}

fn circumradius(a: P, b: P, c: P) -> (f64, f64) {
    // (the signed cross product, and the radius when it isn't ~0)
    let (abx, aby, bcx, bcy, acx, acy) = (
        b[0] - a[0],
        b[1] - a[1],
        c[0] - b[0],
        c[1] - b[1],
        c[0] - a[0],
        c[1] - a[1],
    );
    let cr = abx * bcy - aby * bcx;
    let r = ((abx * abx + aby * aby) * (bcx * bcx + bcy * bcy) * (acx * acx + acy * acy)).sqrt();
    (cr, r)
}

fn hull(pts: &[P]) -> Vec<P> {
    let mut p: Vec<P> = pts.to_vec();
    p.sort_by(|a, b| {
        a[0].partial_cmp(&b[0])
            .unwrap()
            .then(a[1].partial_cmp(&b[1]).unwrap())
    });
    p.dedup();
    if p.len() < 3 {
        return p;
    }
    let cross = |o: P, a: P, b: P| (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    let (mut lo, mut up): (Vec<P>, Vec<P>) = (vec![], vec![]);
    for &q in &p {
        while lo.len() >= 2 && cross(lo[lo.len() - 2], lo[lo.len() - 1], q) <= 0.0 {
            lo.pop();
        }
        lo.push(q);
    }
    for &q in p.iter().rev() {
        while up.len() >= 2 && cross(up[up.len() - 2], up[up.len() - 1], q) <= 0.0 {
            up.pop();
        }
        up.push(q);
    }
    lo.pop();
    up.pop();
    lo.extend(up);
    lo
}

fn circuit_ok(
    pts: &[P],
    s_: &[f64],
    l: f64,
    width: f64,
    sw: f64,
    sh: f64,
    min_r: f64,
    min_slow: i64,
) -> bool {
    let n = pts.len();
    for p in pts {
        if !(150.0 <= p[0] && p[0] <= sw - 150.0 && 150.0 <= p[1] && p[1] <= sh - 150.0) {
            return false;
        }
    }
    let (mut slow, mut was) = (0i64, false);
    for i in 0..n {
        let (a, b, c) = (pts[wrap(i as i64 - 3, n)], pts[i], pts[(i + 3) % n]);
        let (cr, num) = circumradius(a, b, c);
        let cr = cr.abs();
        let mut tight = false;
        if cr > 1e-9 {
            let r = num / (2.0 * cr);
            if r < width * min_r {
                return false;
            }
            tight = r < width * SLOW_CORNER;
        }
        if tight && !was {
            slow += 1;
        }
        was = tight;
    }
    if slow < min_slow {
        return false;
    }
    let cell = 2.4 * width;
    let need = cell * cell;
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, p) in pts.iter().enumerate() {
        grid.entry(((p[0] / cell).floor() as i64, (p[1] / cell).floor() as i64))
            .or_default()
            .push(i);
    }
    for (i, p) in pts.iter().enumerate() {
        let (gx, gy) = ((p[0] / cell).floor() as i64, (p[1] / cell).floor() as i64);
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(js) = grid.get(&(gx + dx, gy + dy)) {
                    for &j in js {
                        if j <= i {
                            continue;
                        }
                        let arc = s_[j] - s_[i];
                        if min(arc, l - arc) < 6.0 * width {
                            continue;
                        }
                        let (ex, ey) = (pts[j][0] - p[0], pts[j][1] - p[1]);
                        if ex * ex + ey * ey < need {
                            return false;
                        }
                    }
                }
            }
        }
    }
    true
}

fn pit_straight(q: &[P]) -> Option<Vec<P>> {
    let (o, a, b) = (q[0], q[q.len() - 1], q[1]);
    let (da, db) = (
        hypot(a[0] - o[0], a[1] - o[1]),
        hypot(b[0] - o[0], b[1] - o[1]),
    );
    if da < PIT_STRAIGHT.0 + 199.0 || db < PIT_STRAIGHT.1 + 199.0 {
        return None;
    }
    let (ux, uy) = ((b[0] - o[0]) / db, (b[1] - o[1]) / db);
    if ((a[0] - o[0]) * uy - (a[1] - o[1]) * ux).abs() > 1.0
        || (a[0] - o[0]) * ux + (a[1] - o[1]) * uy > 0.0
    {
        return None;
    }
    let mut out = vec![o, [o[0] + ux * PIT_STRAIGHT.1, o[1] + uy * PIT_STRAIGHT.1]];
    out.extend_from_slice(&q[1..]);
    out.push([o[0] - ux * PIT_STRAIGHT.0, o[1] - uy * PIT_STRAIGHT.0]);
    Some(out)
}

/// The engine's _generate_flat: a seeded Grand Prix circuit.
pub fn generate_flat(seed: i64) -> Spec {
    let seed = seed.abs();
    let mut rng = Mulberry32::from_f64(seed as f64 * 2654435761.0 + 1.0);
    let (sw, sh) = (GP_W, GP_H);
    let width = js_round(58.0 + rng.next() * 22.0);
    let mut control: Option<Vec<P>> = None;
    for attempt in 0..260 {
        let m = 10 + (rng.next() * 10.0).floor() as i64;
        let bw = 3000.0 + rng.next() * 1300.0;
        let bh = 1800.0 + rng.next() * 950.0;
        let (x0, y0) = ((sw - bw) / 2.0, (sh - bh) / 2.0);
        let mut pp = Vec::new();
        for _ in 0..m {
            let x = js_round(x0 + rng.next() * bw);
            let y = js_round(y0 + rng.next() * bh);
            pp.push([x, y]);
        }
        let hl = hull(&pp);
        if hl.len() < 4 {
            continue;
        }
        let cx = pym::sum(&hl.iter().map(|p| p[0]).collect::<Vec<_>>()) / hl.len() as f64;
        let cy = pym::sum(&hl.iter().map(|p| p[1]).collect::<Vec<_>>()) / hl.len() as f64;
        let mut q: Vec<P> = Vec::new();
        for i in 0..hl.len() {
            let (a, b) = (hl[i], hl[(i + 1) % hl.len()]);
            q.push(a);
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let ln = (dx * dx + dy * dy).sqrt();
            if ln > 600.0 && rng.next() < 0.8 {
                let t = 0.3 + rng.next() * 0.4;
                let (mx, my) = (a[0] + dx * t, a[1] + dy * t);
                let (mut nx, mut ny) = (-dy / ln, dx / ln);
                if nx * (cx - mx) + ny * (cy - my) < 0.0 {
                    nx = -nx;
                    ny = -ny;
                }
                let amt = (rng.next() * 0.6 - 0.12) * ln;
                q.push([mx + nx * amt, my + ny * amt]);
            }
        }
        for _ in 0..6 {
            for i in 0..q.len() {
                for j in i + 1..q.len() {
                    let (dx, dy) = (q[j][0] - q[i][0], q[j][1] - q[i][1]);
                    let d = (dx * dx + dy * dy).sqrt();
                    if d < 560.0 {
                        let k = (560.0 - d) / 2.0 / or1(d);
                        q[i][0] -= dx * k;
                        q[i][1] -= dy * k;
                        q[j][0] += dx * k;
                        q[j][1] += dy * k;
                    }
                }
            }
        }
        let mut r: Vec<P> = Vec::new();
        for i in 0..q.len() {
            let (pp_, c, nn) = (q[wrap(i as i64 - 1, q.len())], q[i], q[(i + 1) % q.len()]);
            if rng.next() < 0.55 {
                let (ax, ay, bx, by) = (pp_[0] - c[0], pp_[1] - c[1], nn[0] - c[0], nn[1] - c[1]);
                let (d1, d2) = ((ax * ax + ay * ay).sqrt(), (bx * bx + by * by).sqrt());
                let rr = 220.0 + rng.next() * 140.0;
                let (k1, k2) = (min(0.3, rr / d1), min(0.3, rr / d2));
                r.push([c[0] + ax * k1, c[1] + ay * k1]);
                r.push([c[0] + bx * k2, c[1] + by * k2]);
            } else {
                r.push(c);
            }
        }
        let mut q = r;
        let (mut best, mut bi) = (-1.0f64, 0usize);
        for i in 0..q.len() {
            let (a, b) = (q[i], q[(i + 1) % q.len()]);
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            if dx * dx + dy * dy > best {
                best = dx * dx + dy * dy;
                bi = i;
            }
        }
        let (a, b) = (q[bi], q[(bi + 1) % q.len()]);
        let ln = best.sqrt();
        let f = if ln * 0.35 >= PIT_STRAIGHT.0 + 200.0 {
            0.35
        } else {
            min(0.5, (PIT_STRAIGHT.0 + 200.0) / ln)
        };
        let mut nq = vec![[a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f]];
        nq.extend_from_slice(&q[bi + 1..]);
        nq.extend_from_slice(&q[..bi + 1]);
        q = nq;
        if rng.next() < 0.6 {
            let (mut best, mut ci) = (-1.0f64, -1i64);
            for i in 1..q.len() {
                let (a, b) = (q[i], q[(i + 1) % q.len()]);
                let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                if dx * dx + dy * dy > best {
                    best = dx * dx + dy * dy;
                    ci = i as i64;
                }
            }
            let ciu = wrap(ci, q.len());
            let (a, b) = (q[ciu], q[(ciu + 1) % q.len()]);
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let ln = (dx * dx + dy * dy).sqrt();
            if ln > 1100.0 {
                let (nx, ny, o) = (-dy / ln, dx / ln, width * 1.3);
                let c1 = [a[0] + dx * 0.45 + nx * o, a[1] + dy * 0.45 + ny * o];
                let c2 = [
                    a[0] + dx * 0.45 + dx / ln * 160.0 - nx * o,
                    a[1] + dy * 0.45 + dy / ln * 160.0 - ny * o,
                ];
                // Q[:ci + 1] + [c1, c2] + Q[ci + 1:] (ci = -1: Q[:0] + ... + Q[0:])
                let cut = if ci < 0 { 0 } else { ci as usize + 1 };
                let mut nq = q[..cut].to_vec();
                nq.push(c1);
                nq.push(c2);
                nq.extend_from_slice(&q[cut..]);
                q = nq;
            }
        }
        if rng.next() < 0.5 {
            let mut nq = vec![q[0]];
            nq.extend(q[1..].iter().rev());
            q = nq;
        }
        let q = match pit_straight(&q) {
            Some(q) => q,
            None => continue,
        };
        let cand: Vec<P> = q.iter().map(|p| [js_round(p[0]), js_round(p[1])]).collect();
        let pts = smooth_closed(&cand);
        let (s_, l) = cumulative(&pts);
        if (8000.0..=15000.0).contains(&l)
            && circuit_ok(
                &pts,
                &s_,
                l,
                width,
                sw,
                sh,
                if attempt < 160 { 0.6 } else { 0.5 },
                if attempt < 160 { 2 } else { 0 },
            )
        {
            control = Some(cand);
            break;
        }
    }
    let control = control.unwrap_or_else(|| {
        (0..16)
            .map(|i| {
                let a = i as f64 / 16.0 * PI * 2.0;
                [
                    js_round(sw / 2.0 + pym::cos(a) * 1700.0),
                    js_round(sh / 2.0 + pym::sin(a) * 1000.0),
                ]
            })
            .collect()
    });
    let grip = js_round((0.85 + rng.next() * 0.3) * 100.0) / 100.0;
    let pts = smooth_closed(&control);
    let (s_, l) = cumulative(&pts);
    let mut obstacles = Vec::new();
    let mut used: Vec<f64> = Vec::new();
    let n_rocks = 2 + (rng.next() * 6.0).floor() as i64;
    let n_oil = (rng.next() * 4.0).floor() as i64;
    let mut pick_t = |rng: &mut Mulberry32| -> Option<f64> {
        for _ in 0..30 {
            let t = 0.06 + rng.next() * 0.9;
            if used.iter().all(|u| (u - t).abs() > 0.03) {
                used.push(t);
                return Some(t);
            }
        }
        None
    };
    for _ in 0..n_rocks {
        let t = match pick_t(&mut rng) {
            Some(t) => t,
            None => break,
        };
        let (x, y, nx, ny) = point_at(&pts, &s_, l, t * l);
        let side = if rng.next() < 0.5 { -1.0 } else { 1.0 };
        let off = side * width * (0.12 + rng.next() * 0.22);
        let r = js_round(7.0 + rng.next() * 6.0);
        obstacles.push(Obstacle {
            oil: false,
            x: js_round(x + nx * off),
            y: js_round(y + ny * off),
            r,
        });
    }
    for _ in 0..n_oil {
        let t = match pick_t(&mut rng) {
            Some(t) => t,
            None => break,
        };
        let (x, y, nx, ny) = point_at(&pts, &s_, l, t * l);
        let off = (rng.next() - 0.5) * width * 0.3;
        obstacles.push(Obstacle {
            oil: true,
            x: js_round(x + nx * off),
            y: js_round(y + ny * off),
            r: js_round(18.0 + rng.next() * 10.0),
        });
    }
    Spec {
        seed,
        width,
        grip,
        control,
        obstacles,
        elev: None,
        bank: None,
    }
}

/// The engine's add_hills (N1): heights and banks for the control points.
pub fn add_hills(spec: &mut Spec, seed: i64) {
    let mut rng = Mulberry32::from_i64(seed + 104729);
    let (c, w) = (&spec.control, spec.width);
    let n = c.len();
    let seg: Vec<f64> = (0..n)
        .map(|i| hypot(c[(i + 1) % n][0] - c[i][0], c[(i + 1) % n][1] - c[i][1]))
        .collect();
    let total = pym::sum(&seg);
    let mut acc = 0.0;
    let mut u = Vec::with_capacity(n);
    for s in &seg {
        u.push(acc / total);
        acc += s;
    }
    let mut waves = Vec::new();
    for f in [1.0, 2.0, 3.0, 5.0] {
        let a = 0.3 + 0.7 * rng.next();
        let ph = rng.next() * 2.0 * PI;
        waves.push((f, a, ph));
    }
    let mut h: Vec<f64> = u
        .iter()
        .map(|&ui| {
            pym::sum(
                &waves
                    .iter()
                    .map(|&(f, a, ph)| a * pym::sin(2.0 * PI * f * ui + ph))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let lo = h
        .iter()
        .cloned()
        .fold(f64::NAN, |m, v| if m.is_nan() { v } else { min(m, v) });
    for v in h.iter_mut() {
        *v -= lo;
    }
    let mut steep = f64::NAN;
    for i in 0..n {
        let v = (h[(i + 1) % n] - h[i]).abs() / or1(seg[i]);
        steep = if steep.is_nan() { v } else { max(steep, v) };
    }
    let steep = or1(steep);
    let mut k = (0.05 + 0.03 * rng.next()) / steep;
    let hmax = h
        .iter()
        .cloned()
        .fold(f64::NAN, |m, v| if m.is_nan() { v } else { max(m, v) });
    if hmax * k > HILL_RISE_MAX {
        k = HILL_RISE_MAX / hmax;
    }
    let elev: Vec<f64> = h.iter().map(|v| js_round(v * k * 10.0) / 10.0).collect();
    let mut bank = Vec::with_capacity(n);
    for i in 0..n {
        let (a, b, cc) = (c[wrap(i as i64 - 1, n)], c[i], c[(i + 1) % n]);
        let (cr, num) = circumradius(a, b, cc);
        let cr = cr.abs();
        let r = if cr < 1e-9 { 1e9 } else { num / (2.0 * cr) };
        let (r1, r2) = (rng.next(), rng.next());
        let mut v = 0.0;
        if r < 10.0 * w {
            v = if r1 < 0.5 {
                6.0 + 9.0 * r2
            } else if r1 < 0.62 {
                -(3.0 + 4.0 * r2)
            } else {
                0.0
            };
        }
        bank.push(js_round(v * 10.0) / 10.0);
    }
    spec.elev = Some(elev);
    spec.bank = Some(bank);
}

/// A built track (the engine's Track, the fields training uses).
pub struct Track {
    pub w: usize,
    pub h: usize,
    pub pts: Vec<P>,
    pub s: Vec<f64>,
    pub l: f64,
    pub width: f64,
    pub grip: f64,
    pub laps: i64,
    pub oils: Vec<Obstacle>,
    pub mask: Vec<u8>,
    pub start: (f64, f64, f64),
    pub hilly: bool,
    pub z: Vec<f64>,
    pub grade: Vec<f64>,
    pub vcurv: Vec<f64>,
    pub bank: Vec<f64>,
    pub roll: Vec<f64>,
    pub curv: Vec<f64>,
    pub vmax: Vec<f64>,
    pub seglen: Vec<f64>,
    pub grid: Vec<(f64, f64, f64)>,
}

pub fn bank_fade(k: f64) -> f64 {
    max(0.0, min(1.0, 2.0 * k.abs() / BANK_FADE - 1.0))
}

impl Track {
    /// The engine's build_track, from a spec already read as build_track reads it (world size, clamped control
    /// points, width, grip, laps, rocks, oil, heights and banks: neuroprix's adapter reads them with the engine's own code).
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        w: usize,
        h: usize,
        control: Vec<P>,
        width: f64,
        grip: f64,
        laps: i64,
        rocks: Vec<Obstacle>,
        oils: Vec<Obstacle>,
        elev: Vec<f64>,
        bank_deg: Vec<f64>,
    ) -> Track {
        let pts = smooth_closed(&control);
        let (s_, l) = cumulative(&pts);
        let half = width / 2.0;
        let mut mask = vec![0u8; w * h];
        let n = pts.len();
        let (wf, hf) = (w as f64, h as f64);
        for i in 0..n {
            let (ax, ay) = (pts[i][0], pts[i][1]);
            let (bx, by) = (pts[(i + 1) % n][0], pts[(i + 1) % n][1]);
            let x0 = max(0.0, min(ax, bx) - half - 1.0) as i64;
            let x1 = min(wf, max(ax, bx) + half + 2.0) as i64;
            let y0 = max(0.0, min(ay, by) - half - 1.0) as i64;
            let y1 = min(hf, max(ay, by) + half + 2.0) as i64;
            if x0 >= x1 || y0 >= y1 {
                continue;
            }
            let (vx, vy) = (bx - ax, by - ay);
            let l2 = or1(vx * vx + vy * vy);
            let hh = half * half;
            for yi in y0..y1 {
                let gy = yi as f64 + 0.5;
                let row = yi as usize * w;
                for xi in x0..x1 {
                    let gx = xi as f64 + 0.5;
                    let t = ((gx - ax) * vx + (gy - ay) * vy) / l2;
                    let t = if t < 0.0 {
                        0.0
                    } else if t > 1.0 {
                        1.0
                    } else {
                        t
                    };
                    let ex = ax + vx * t - gx;
                    let ey = ay + vy * t - gy;
                    if ex * ex + ey * ey <= hh {
                        mask[row + xi as usize] = 1;
                    }
                }
            }
        }
        for r in &rocks {
            let (rx, ry, rr) = (r.x, r.y, r.r);
            let x0 = max(0.0, rx - rr - 1.0) as i64;
            let x1 = min(wf, rx + rr + 2.0) as i64;
            let y0 = max(0.0, ry - rr - 1.0) as i64;
            let y1 = min(hf, ry + rr + 2.0) as i64;
            if x0 >= x1 || y0 >= y1 {
                continue;
            }
            for yi in y0..y1 {
                let gy = yi as f64 + 0.5;
                for xi in x0..x1 {
                    let gx = xi as f64 + 0.5;
                    let (dx, dy) = (gx - rx, gy - ry);
                    if !(dx * dx + dy * dy > rr * rr) {
                        mask[yi as usize * w + xi as usize] = 0;
                    }
                }
            }
        }
        let start = (
            pts[0][0],
            pts[0][1],
            (pts[1][1] - pts[0][1]).atan2(pts[1][0] - pts[0][0]),
        );
        let mut t = Track {
            w,
            h,
            pts,
            s: s_,
            l,
            width,
            grip,
            laps,
            oils,
            mask,
            start,
            hilly: false,
            z: vec![],
            grade: vec![],
            vcurv: vec![],
            bank: vec![],
            roll: vec![],
            curv: vec![],
            vmax: vec![],
            seglen: vec![],
            grid: vec![],
        };
        t.curv = t.corner_curv();
        if elev.iter().any(|&v| v != 0.0) || bank_deg.iter().any(|&v| v != 0.0) {
            t.hilly = true;
            t.z = smooth_values(&control, &elev);
            let banks = smooth_values(&control, &bank_deg);
            t.road_shape(&banks);
            let k = t.curv.clone();
            t.bank = t
                .bank
                .iter()
                .zip(&k)
                .map(|(b, &kk)| b * bank_fade(kk))
                .collect();
            t.roll = t
                .bank
                .iter()
                .zip(&k)
                .map(|(&b, &kk)| {
                    if kk > 0.0 {
                        b
                    } else if kk < 0.0 {
                        -b
                    } else {
                        0.0
                    }
                })
                .collect();
        }
        t.corner_limits();
        t.grid = t.free_grid();
        t
    }

    fn road_shape(&mut self, bank_deg: &[f64]) {
        let n = self.pts.len();
        let (s_, l, z) = (&self.s, self.l, &self.z);
        let (mut grade, mut vcurv) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let (a, c) = (wrap(i as i64 - 3, n), (i + 3) % n);
            let mut d1 = s_[i] - s_[a];
            if d1 <= 0.0 {
                d1 += l;
            }
            let mut d2 = s_[c] - s_[i];
            if d2 <= 0.0 {
                d2 += l;
            }
            grade.push((z[c] - z[a]) / (d1 + d2));
            vcurv.push(((z[c] - z[i]) / d2 - (z[i] - z[a]) / d1) / (0.5 * (d1 + d2)));
        }
        self.grade = grade;
        self.vcurv = vcurv;
        self.bank = bank_deg.iter().map(|b| b * PI / 180.0).collect();
    }

    fn corner_curv(&self) -> Vec<f64> {
        let (p, n) = (&self.pts, self.pts.len());
        (0..n)
            .map(|i| {
                let (cr, num) = circumradius(p[wrap(i as i64 - 3, n)], p[i], p[(i + 3) % n]);
                if cr.abs() < 1e-9 {
                    return 0.0;
                }
                let r = num / (2.0 * cr.abs());
                min(1.0, self.width / r) * (if cr > 0.0 { 1.0 } else { -1.0 })
            })
            .collect()
    }

    fn corner_limits(&mut self) {
        let (p, n) = (&self.pts, self.pts.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let (cr, num) = circumradius(p[wrap(i as i64 - 3, n)], p[i], p[(i + 3) % n]);
            let cr = cr.abs();
            let r = if cr < 1e-9 { 1e9 } else { num / (2.0 * cr) };
            let mut lat = LAT_GRIP * or1(self.grip) * DYN_CORNER;
            if self.hilly {
                lat = max(0.2 * lat, lat + BANK_GRIP * pym::sin(self.bank[i]));
            }
            out.push((lat * (r + self.width * 0.5)).sqrt());
        }
        self.vmax = out;
        self.seglen = (0..n)
            .map(|k| hypot(p[(k + 1) % n][0] - p[k][0], p[(k + 1) % n][1] - p[k][1]))
            .collect();
    }

    #[inline]
    pub fn on_track(&self, x: f64, y: f64) -> bool {
        if x < 0.0 || y < 0.0 || x >= self.w as f64 || y >= self.h as f64 {
            return false;
        }
        self.mask[(y as usize) * self.w + x as usize] != 0
    }

    pub fn point_at(&self, s: f64) -> (f64, f64, f64, f64) {
        point_at(&self.pts, &self.s, self.l, s)
    }

    /// The engine's road_frame: (x, y, nx, ny, z, tan(roll)) at s px round the lap.
    pub fn road_frame(&self, s: f64) -> (f64, f64, f64, f64, f64, f64) {
        let (s_, n) = (&self.s, self.s.len());
        let s = pym::fmod(pym::fmod(s, self.l) + self.l, self.l);
        let (mut lo, mut hi) = (0usize, n - 1);
        while lo < hi {
            let m = (lo + hi + 1) >> 1;
            if s_[m] <= s {
                lo = m;
            } else {
                hi = m - 1;
            }
        }
        let j = (lo + 1) % n;
        let seg = or1((if lo == n - 1 { self.l } else { s_[j] }) - s_[lo]);
        let t = (s - s_[lo]) / seg;
        let (a, b) = (self.pts[lo], self.pts[j]);
        let (tx, ty) = ((b[0] - a[0]) / seg, (b[1] - a[1]) / seg);
        let z = self.z[lo] + (self.z[j] - self.z[lo]) * t;
        let r = self.roll[lo] + (self.roll[j] - self.roll[lo]) * t;
        (
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            -ty,
            tx,
            z,
            r.tan(),
        )
    }

    pub fn road_point(&self, s: f64, lat: f64) -> (f64, f64, f64) {
        let (x, y, nx, ny, z, tr) = self.road_frame(s);
        (x + nx * lat, y + ny * lat, z - lat * tr)
    }

    /// The engine's grid_slot: x, y, heading of a starting slot behind s0.
    pub fn grid_slot(&self, slot: usize, s0: f64) -> (f64, f64, f64) {
        let slot = slot as i64;
        let back = GRID_FIRST + (slot / 2) * GRID_ROW + (slot % 2) * GRID_STAGGER;
        let lat = (if slot % 2 != 0 { 1.0 } else { -1.0 }) * self.width * GRID_LAT;
        let mut s = s0 - back as f64;
        if s < 0.0 {
            s += self.l;
        }
        let (x, y, nx, ny) = self.point_at(s);
        (x + nx * lat, y + ny * lat, (-nx).atan2(ny))
    }

    fn free_grid(&self) -> Vec<(f64, f64, f64)> {
        let (hl, hw) = (CAR_L / 2.0, CAR_W / 2.0);
        let mut out = Vec::new();
        let mut i = 0;
        while out.len() < GRID_SLOTS && i < GRID_SLOTS * 3 {
            let (x, y, a) = self.grid_slot(i, 0.0);
            let (c, s) = (pym::cos(a), pym::sin(a));
            let corners = [
                (hl, hw),
                (hl, -hw),
                (-hl, hw),
                (-hl, -hw),
                (hl, 0.0),
                (-hl, 0.0),
                (0.0, hw),
                (0.0, -hw),
            ];
            if corners
                .iter()
                .all(|&(lx, ly)| self.on_track(x + lx * c - ly * s, y + lx * s + ly * c))
            {
                out.push((x, y, a));
            }
            i += 1;
        }
        while out.len() < GRID_SLOTS {
            out.push(self.grid_slot(out.len(), 0.0));
        }
        out
    }
}

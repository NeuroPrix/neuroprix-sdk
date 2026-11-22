//! neuroprix_rs: NeuroPrix's race engine in Rust, for faster training with the neuroprix SDK.
//!
//! It is a port of neuroprix/engine.py (the reference engine, which stays the readable one to tweak) and gives the
//! same numbers to the last bit, step by step. neuroprix uses it only when asked (`--engine rust`), when its
//! ENGINE_VERSION matches, when neuroprix/engine.py is the one released with the SDK, and when a short lockstep
//! check against the Python engine passes; see neuroprix.rust_env. Physics changes go into engine.py first, then here.

mod pym;
mod track;
mod world;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;
use track::Track;
use world::{Car, World};

/// The engine version this port matches (neuroprix.engine.ENGINE_VERSION).
const ENGINE_VERSION: &str = "2026.11.10";

fn int_or_float(py: Python<'_>, v: f64) -> PyObject {
    // (the generator's numbers that the Python engine keeps as ints)
    (v as i64).into_pyobject(py).unwrap().into_any().unbind()
}

/// Tell the engine which Python runs it (math.hypot and sum() of floats changed in 3.12).
#[pyfunction]
fn set_python(major: u32, minor: u32) {
    pym::set_python(major, minor);
}

/// math.hypot, as this engine computes it (for the self-check).
#[pyfunction]
fn hypot(x: f64, y: f64) -> f64 {
    pym::hypot(x, y)
}

/// The maths library as this engine calls it, for checks: name one of sin, cos, tan,
/// atan, asin, atan2, pow2.
#[pyfunction]
fn libm(name: &str, xs: Vec<f64>, ys: Vec<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(xs.len() * 2);
    for (i, &x) in xs.iter().enumerate() {
        let y = ys.get(i).copied().unwrap_or(0.0);
        match name {
            "sin" => out.push(pym::sin(x)),
            "cos" => out.push(pym::cos(x)),
            "tan" => out.push(x.tan()),
            "atan" => out.push(x.atan()),
            "asin" => out.push(x.asin()),
            "atan2" => out.push(x.atan2(y)),
            "pow2" => out.push(pym::pow2(x)),
            _ => {}
        }
    }
    out
}

/// sum() of floats, as this engine computes it (for the self-check).
#[pyfunction]
fn fsum(xs: Vec<f64>) -> f64 {
    pym::sum(&xs)
}

/// The engine's generate_track_spec(seed, cls): a seeded Grand Prix circuit as a dict ("n1": with hills).
#[pyfunction]
#[pyo3(signature = (seed, cls=None))]
fn generate_track_spec<'py>(
    py: Python<'py>,
    seed: i64,
    cls: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut spec = track::generate_flat(seed);
    if cls.map(|c| c.to_lowercase() == "n1").unwrap_or(false) {
        track::add_hills(&mut spec, seed);
    }
    let d = PyDict::new(py);
    d.set_item("name", format!("Procedural #{}", spec.seed))?;
    d.set_item("seed", spec.seed)?;
    d.set_item("size", vec![track::GP_W as i64, track::GP_H as i64])?;
    let control = PyList::empty(py);
    for p in &spec.control {
        control.append(vec![int_or_float(py, p[0]), int_or_float(py, p[1])])?;
    }
    d.set_item("control", control)?;
    d.set_item("width", spec.width as i64)?;
    d.set_item("grip", spec.grip)?;
    d.set_item("laps", 2)?;
    if let (Some(e), Some(b)) = (&spec.elev, &spec.bank) {
        d.set_item("elev", e.clone())?;
        d.set_item("bank", b.clone())?;
    }
    Ok(d)
}

/// A built track (the engine's build_track). The spec is read in Python first, by the engine's own code
/// (neuroprix.rust_env), so only plain numbers cross over.
#[pyclass(frozen, module = "neuroprix_rs")]
struct RustTrack {
    inner: Arc<Track>,
}

#[pymethods]
impl RustTrack {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        w: usize,
        h: usize,
        control: Vec<(f64, f64)>,
        width: f64,
        grip: f64,
        laps: i64,
        elev: Vec<f64>,
        bank: Vec<f64>,
    ) -> PyResult<Self> {
        if control.len() < 3 || elev.len() != control.len() || bank.len() != control.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "a track needs 3 control points or more, and a height and bank each",
            ));
        }
        let t = Track::build(
            w,
            h,
            control.into_iter().map(|(x, y)| [x, y]).collect(),
            width,
            grip,
            laps,
            elev,
            bank,
        );
        Ok(RustTrack { inner: Arc::new(t) })
    }

    #[getter(L)]
    fn length(&self) -> f64 {
        self.inner.l
    }

    #[getter]
    fn width(&self) -> f64 {
        self.inner.width
    }

    #[getter]
    fn laps(&self) -> i64 {
        self.inner.laps
    }

    #[getter]
    fn hilly(&self) -> bool {
        self.inner.hilly
    }

    fn point_at(&self, s: f64) -> (f64, f64, f64, f64) {
        self.inner.point_at(s)
    }

    fn on_track(&self, x: f64, y: f64) -> bool {
        self.inner.on_track(x, y)
    }

    /// The track map, one byte per pixel, row by row (for checks).
    fn mask(&self) -> Vec<u8> {
        self.inner.mask.clone()
    }

    /// Everything else a built track holds, by name (for checks).
    fn arrays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let t = &self.inner;
        let d = PyDict::new(py);
        d.set_item(
            "pts",
            t.pts.iter().map(|p| (p[0], p[1])).collect::<Vec<_>>(),
        )?;
        d.set_item("S", t.s.clone())?;
        d.set_item("start", t.start)?;
        d.set_item("z", t.z.clone())?;
        d.set_item("grade", t.grade.clone())?;
        d.set_item("vcurv", t.vcurv.clone())?;
        d.set_item("bank", t.bank.clone())?;
        d.set_item("roll", t.roll.clone())?;
        d.set_item("curv", t.curv.clone())?;
        d.set_item("vmax", t.vmax.clone())?;
        d.set_item("seglen", t.seglen.clone())?;
        d.set_item("grid", t.grid.clone())?;
        Ok(d)
    }
}

/// A field of training cars on one track, stepped like neuroprix.env.SimEnv steps the Python engine.
#[pyclass(module = "neuroprix_rs", unsendable)]
struct RustWorld {
    w: World,
}

type CarState = (
    (f64, f64, f64, f64, f64, f64), // x, y, angle, steer, max_distance, distance
    (i64, &'static str, bool, bool), // frames, reason, dead, finished
    Vec<f64>,                       // sensors
    Vec<f64>,                       // obs
    (f64, f64, f64, i64, f64, i64), // speed, health, damage (r4), hits, impact (r4), contacts
    Vec<usize>,                     // touched
    f64,                            // heading
    Option<(f64, f64, f64)>,        // corner_over, corner_need, impact_risk (r4; racing cars only)
    (f64, bool, bool),              // battery (r4), boosting, braking
);

#[pymethods]
impl RustWorld {
    /// gens: each car's brain kind; caps: each one's top-speed factor (cap_k); world_cap: the practice top speed.
    #[new]
    fn new(
        track: &RustTrack,
        gens: Vec<i64>,
        caps: Vec<f64>,
        world_cap: f64,
        fragile: bool,
    ) -> Self {
        let cars = gens
            .iter()
            .enumerate()
            .map(|(i, &g)| {
                let mut c = Car::new(g);
                c.training = true;
                c.cap_k = caps[i];
                c.world_cap = world_cap;
                c.fragile = fragile;
                c
            })
            .collect();
        RustWorld {
            w: World::new(track.inner.clone(), cars),
        }
    }

    /// The env's reset_world, given its random numbers: start (random.random() for a random start, else None), the
    /// rolling start's share of top speed (0 = standing) and the race day's grip (None: off).
    #[pyo3(signature = (start, rolling, grip))]
    fn reset(&mut self, start: Option<f64>, rolling: f64, grip: Option<f64>) {
        let t = self.w.track.clone();
        let multi = self.w.cars.len() > 1;
        let st = start.map(|r| {
            let rs = r * t.l;
            let (x, y, nx, ny) = t.point_at(rs);
            (x, y, (-nx).atan2(ny), rs)
        });
        for (i, c) in self.w.cars.iter_mut().enumerate() {
            c.reset(&t, if multi { Some(i) } else { None });
            if let Some((x, y, a, rs)) = st {
                if multi {
                    let (gx, gy, ga) = t.grid_slot(i, rs);
                    c.place(&t, gx, gy, ga);
                } else {
                    c.place(&t, x, y, a);
                }
            }
            c.risk = 0.0;
            c.impact_risk = 0.0;
            c.touched.clear();
            if rolling > 0.0 {
                c.speed = rolling * 5.0 * c.cap_k;
                c.vx = pym::cos(c.angle) * c.speed;
                c.vy = pym::sin(c.angle) * c.speed;
                c.dir = c.angle;
            }
        }
        if let Some(g) = grip {
            for c in self.w.cars.iter_mut() {
                c.grip_k = g;
            }
        }
        self.w.frame_no = 0;
        self.w.clear_touch();
        for i in 0..self.w.cars.len() {
            self.w.sense(i);
        }
    }

    /// A spun start: car i turned where it stands, stopped (the env's spin).
    #[pyo3(signature = (i, turn, room=None))]
    fn spin(&mut self, i: usize, turn: f64, room: Option<f64>) {
        let c = &mut self.w.cars[i];
        c.angle += turn;
        c.dir = c.angle;
        c.speed = 0.0;
        c.vx = 0.0;
        c.vy = 0.0;
        if let Some(r) = room {
            if r != 0.0 {
                c.wrong_way_px = r;
            }
        }
    }

    fn sense_all(&mut self) {
        for i in 0..self.w.cars.len() {
            self.w.sense(i);
        }
    }

    /// One training step: the env's step_world.
    fn step(&mut self, actions: Vec<i64>) {
        self.w.step(&actions);
    }

    /// Every car's state for the trainer (neuroprix.rust_env turns each into the env's state dict).
    fn states(&self) -> Vec<CarState> {
        let t = &self.w.track;
        self.w
            .cars
            .iter()
            .map(|c| {
                let racing = !(c.dead || c.finished);
                (
                    (c.x, c.y, c.angle, c.steer, c.max_distance, c.distance),
                    (c.frames, c.reason.as_str(), c.dead, c.finished),
                    c.sensors.to_vec(),
                    c.obs(t),
                    (
                        pym::r4(c.speed / 5.0),
                        pym::r4(c.health),
                        pym::r4(c.damage),
                        c.hits,
                        pym::r4(c.impact),
                        c.contacts,
                    ),
                    c.touched.clone(),
                    c.heading(t),
                    if racing {
                        let need = c.corner_need(t);
                        let over = if need == f64::INFINITY {
                            0.0
                        } else {
                            pym::max(0.0, c.speed / need - 1.0)
                        };
                        Some((
                            pym::r4(over),
                            pym::r4(pym::min(99.0, need)),
                            pym::r4(c.impact_risk),
                        ))
                    } else {
                        None
                    },
                    (pym::r4(c.battery), c.boosting, c.braking),
                )
            })
            .collect()
    }

    /// Car i's whole inner state, by the Python engine's attribute names (for checks).
    fn raw<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyDict>> {
        let c = &self.w.cars[i];
        let d = PyDict::new(py);
        for (k, v) in [
            ("x", c.x),
            ("y", c.y),
            ("angle", c.angle),
            ("speed", c.speed),
            ("vx", c.vx),
            ("vy", c.vy),
            ("steer", c.steer),
            ("distance", c.distance),
            ("max_distance", c.max_distance),
            ("prev_s", c.prev_s),
            ("health", c.health),
            ("damage", c.damage),
            ("grip_k", c.grip_k),
            ("slip", c.slip),
            ("dir", c.dir),
            ("impact", c.impact),
            ("load", c.load),
            ("yaw", c.yaw),
            ("risk", c.risk),
            ("impact_risk", c.impact_risk),
            ("wrong_way_px", c.wrong_way_px),
            ("battery", c.battery),
            ("yaw_r", c.yaw_r),
            ("_ax", c.ax),
            ("_thr", c.thr),
            ("_brk", c.brk),
        ] {
            d.set_item(k, v)?;
        }
        for (k, v) in [
            ("frames", c.frames),
            ("last_improve", c.last_improve),
            ("seg", c.seg as i64),
            ("action", c.action),
            ("hits", c.hits),
            ("contacts", c.contacts),
            ("stun", c.stun),
            ("hold", c.hold),
        ] {
            d.set_item(k, v)?;
        }
        for (k, v) in [
            ("dead", c.dead),
            ("finished", c.finished),
            ("boosting", c.boosting),
            ("braking", c.braking),
        ] {
            d.set_item(k, v)?;
        }
        d.set_item("reason", c.reason.as_str())?;
        d.set_item("sensors", c.sensors.to_vec())?;
        d.set_item("radar", c.radar.clone())?;
        Ok(d)
    }

    #[getter]
    fn frame_no(&self) -> i64 {
        self.w.frame_no
    }
}

#[pymodule]
fn neuroprix_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ENGINE_VERSION", ENGINE_VERSION)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(set_python, m)?)?;
    m.add_function(wrap_pyfunction!(hypot, m)?)?;
    m.add_function(wrap_pyfunction!(fsum, m)?)?;
    m.add_function(wrap_pyfunction!(libm, m)?)?;
    m.add_function(wrap_pyfunction!(generate_track_spec, m)?)?;
    m.add_class::<RustTrack>()?;
    m.add_class::<RustWorld>()?;
    Ok(())
}

//! Python's arithmetic, exactly. The engine must give the same numbers as neuroprix/engine.py to the last bit, so the
//! few places where Python doesn't simply do one IEEE operation are written out here: math.hypot (CPython's own
//! algorithm, not the C library's), sum() of floats (compensated since Python 3.12), min() and max() (the first of
//! equal values, which matters for -0.0), float %, and the engine's js_round (floor(x + 0.5)). The rest (sin, cos,
//! atan2, tan, asin, atan, sqrt, pow) is the platform's C maths library, which Python calls too.

use std::sync::atomic::{AtomicBool, Ordering};

/// The Python running the SDK decides two algorithms: math.hypot changed in 3.12, sum() of floats in 3.12.
static PY312: AtomicBool = AtomicBool::new(true);

pub fn set_python(major: u32, minor: u32) {
    PY312.store(major > 3 || (major == 3 && minor >= 12), Ordering::Relaxed);
}

#[inline]
fn py312() -> bool {
    PY312.load(Ordering::Relaxed)
}

extern "C" {
    fn pow(x: f64, y: f64) -> f64;
    fn frexp(x: f64, e: *mut i32) -> f64;
    fn ldexp(x: f64, e: i32) -> f64;
}

/// math.sin and math.cos: the C library's, one call each. (Left alone, LLVM turns a sine and a cosine of the same number
/// into one sincos call, which on some systems differs from sin in the last bit.)
#[inline]
pub fn sin(x: f64) -> f64 {
    std::hint::black_box(x).sin()
}

#[inline]
pub fn cos(x: f64) -> f64 {
    std::hint::black_box(x).cos()
}

/// Python's `x ** 2` for a float x: the C library's pow of |x| (as float.__pow__ calls it), never folded into x * x.
#[inline]
pub fn pow2(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    unsafe { pow(x.abs(), std::hint::black_box(2.0)) }
}

/// Python's min(a, b): a unless b is smaller.
#[inline]
pub fn min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// Python's max(a, b): a unless b is larger.
#[inline]
pub fn max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Python's float x % y.
#[inline]
pub fn fmod(x: f64, y: f64) -> f64 {
    let m = x % y; // (C's fmod)
    if m != 0.0 {
        if (y < 0.0) != (m < 0.0) {
            m + y
        } else {
            m
        }
    } else {
        0.0f64.copysign(y)
    }
}

/// Python's i % n for ints (never negative for n > 0).
#[inline]
pub fn imod(i: i64, n: i64) -> i64 {
    i.rem_euclid(n)
}

/// The engine's js_round: floor(x + 0.5).
#[inline]
pub fn js_round(x: f64) -> f64 {
    (x + 0.5).floor()
}

/// The engine's r4: js_round(v * 10000) / 10000.
#[inline]
pub fn r4(v: f64) -> f64 {
    js_round(v * 10000.0) / 10000.0
}

/// `x or 1` for a float.
#[inline]
pub fn or1(x: f64) -> f64 {
    if x != 0.0 {
        x
    } else {
        1.0
    }
}

#[inline]
fn fast_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    (x, (a - x) + b)
}

/// Python's math.hypot(x, y), CPython's vector_norm for two numbers.
pub fn hypot(x: f64, y: f64) -> f64 {
    let (x, y) = (x.abs(), y.abs());
    if x.is_nan() || y.is_nan() {
        if x.is_infinite() || y.is_infinite() {
            return f64::INFINITY;
        }
        return f64::NAN;
    }
    let m = max(0.0, max(x, y)); // (max = 0.0, then each larger value)
    if m.is_infinite() {
        return m;
    }
    if m == 0.0 {
        return m;
    }
    let mut e: i32 = 0;
    unsafe { frexp(m, &mut e) };
    if e < -1023 {
        let dm = f64::MIN_POSITIVE;
        return dm * hypot(x / dm, y / dm);
    }
    let scale = unsafe { ldexp(1.0, -e) };
    if py312() {
        // 3.12 and later: fma squaring, Neumaier accumulation
        let (mut csum, mut frac1, mut frac2) = (1.0f64, 0.0f64, 0.0f64);
        for v in [x, y] {
            let v = v * scale;
            let hi = v * v;
            let lo = v.mul_add(v, -hi);
            let (s, sl) = fast_sum(csum, hi);
            csum = s;
            frac1 += lo;
            frac2 += sl;
        }
        let mut h = (csum - 1.0 + (frac1 + frac2)).sqrt();
        let hi = -h * h;
        let lo = (-h).mul_add(h, -hi);
        let (s, sl) = fast_sum(csum, hi);
        csum = s;
        frac1 += lo;
        frac2 += sl;
        let x = csum - 1.0 + (frac1 + frac2);
        h += x / (2.0 * h);
        h / scale
    } else {
        // 3.10 and 3.11: Veltkamp splitting, three accumulators
        const T27: f64 = 134217729.0;
        let (mut csum, mut frac1, mut frac2, mut frac3) = (1.0f64, 0.0f64, 0.0f64, 0.0f64);
        for v in [x, y] {
            let v = v * scale;
            let t = v * T27;
            let hi = t - (t - v);
            let lo = v - hi;
            let q = hi * hi;
            let old = csum;
            csum += q;
            frac1 += (old - csum) + q;
            let q = 2.0 * hi * lo;
            let old = csum;
            csum += q;
            frac2 += (old - csum) + q;
            frac3 += lo * lo;
        }
        let h = (csum - 1.0 + (frac1 + frac2 + frac3)).sqrt();
        let t = h * T27;
        let hi = t - (t - h);
        let lo = h - hi;
        let q = -hi * hi;
        let old = csum;
        csum += q;
        frac1 += (old - csum) + q;
        let q = -2.0 * hi * lo;
        let old = csum;
        csum += q;
        frac2 += (old - csum) + q;
        let q = -lo * lo;
        let old = csum;
        csum += q;
        frac3 += (old - csum) + q;
        let x = csum - 1.0 + (frac1 + frac2 + frac3);
        (h + x / (2.0 * h)) / scale
    }
}

/// Python's sum() of floats (start 0).
pub fn sum(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut hi = 0.0 + xs[0];
    if !py312() {
        for &x in &xs[1..] {
            hi += x;
        }
        return hi;
    }
    let mut lo = 0.0f64;
    for &x in &xs[1..] {
        let t = hi + x;
        if hi.abs() >= x.abs() {
            lo += (hi - t) + x;
        } else {
            lo += (x - t) + hi;
        }
        hi = t;
    }
    if lo != 0.0 && lo.is_finite() {
        hi + lo
    } else {
        hi
    }
}

/// The browser's seeded random numbers (the engine's mulberry32), from a seed already reduced to 0..2^32.
pub struct Mulberry32 {
    a: u32,
}

impl Mulberry32 {
    /// From a double, as `int(seed_value) % 2**32`.
    pub fn from_f64(seed: f64) -> Self {
        let i = seed.trunc() as i128;
        Mulberry32 {
            a: i.rem_euclid(1i128 << 32) as u32,
        }
    }

    pub fn from_i64(seed: i64) -> Self {
        Mulberry32 {
            a: (seed as i128).rem_euclid(1i128 << 32) as u32,
        }
    }

    pub fn next(&mut self) -> f64 {
        self.a = self.a.wrapping_add(0x6D2B79F5);
        let a = self.a;
        let mut t = (a ^ (a >> 15)).wrapping_mul(1 | a);
        t = (t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t))) ^ t;
        ((t ^ (t >> 14)) as f64) / 4294967296.0
    }
}

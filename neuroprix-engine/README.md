# neuroprix-engine

**Optional.** NeuroPrix's race engine ported to Rust, for faster training with the
[neuroprix](https://github.com/NeuroPrix/neuroprix-sdk) SDK. It gives exactly the same results as the SDK's Python
engine (`neuroprix/engine.py`), to the last digit, step by step: the practice games just run several times faster.

The SDK doesn't need it. `neuroprix/engine.py` stays the readable reference engine, the one to study and change;
this package is for players who want speed and train on the official physics.

```bash
pip install "neuroprix[fast]"        # or: pip install neuroprix-engine
neuroprix-runner --engine rust       # or: neuroprix-train --engine rust ...
```

The trainer uses it only when asked, only if it was made for the SDK's engine version, only if `neuroprix/engine.py`
is the released one (never instead of physics you've changed), and only if a short side-by-side check at the start
finds the two engines the same. Otherwise it says why and trains on the Python engine. See
[docs/engine.md](../docs/engine.md#the-optional-rust-engine-faster-training).

## What it covers

Everything training uses: the seeded Grand Prix circuit generator (with N1's hills and banks), building a track (the
track map, heights, banks, corner speeds, the grid), and a field of training cars stepped as `neuroprix.env.SimEnv`
steps them (the tyre model, walls, rocks and oil, car contacts, slipstream, battery and boost, hills, the rays, the
radar, and N1's tilt and sight lines). Races on the NeuroPrix server (pit stops, the safety car) run the Python engine.
`neuroprix.rust_env.RustEnv` is the trainer's interface to it, the same as `SimEnv`.

## Exactly the same numbers

The port follows `neuroprix/engine.py` line by line: the same operations in the same order. Where Python doesn't
simply do one floating-point operation, `src/pym.rs` does what Python does: `math.hypot` is CPython's own algorithm
(it changed in Python 3.12, so the engine is told which Python runs it), `sum()` of floats is compensated since
Python 3.12, `min` and `max` keep the first of equal values (which matters for −0.0), `x ** 2` calls the C library's
`pow`, and each sine and cosine is its own call to the C library (a compiler would otherwise merge them into one
`sincos` call, which on some systems differs in the last bit). The rest (`sin`, `cos`, `atan2`, `sqrt`...) is the
platform's C maths library, as Python's `math` uses.

Checked by the SDK's `tests/test_engines.py` (whole inner state of every car after every step, the generator, and
two trainings that must produce the identical brain) and by the start-up check in `neuroprix.rust_env.rust_ready`.

## Changing the physics

The Python engine comes first. Change `neuroprix/engine.py` (and bump its `ENGINE_VERSION`), then make the same
change here, set `ENGINE_VERSION` in `src/lib.rs` to match, and run the SDK's tests with this package installed until
every comparison is exact. Release the two together.

## Building it yourself

You need [Rust](https://rustup.rs) and [maturin](https://www.maturin.rs):

```bash
pip install maturin
pip install ./neuroprix-engine          # from the SDK repository's folder
```

Ready-built wheels for macOS, Linux and Windows are built by the SDK repository's `engine-wheels` workflow.

## License

Apache-2.0, as the SDK.

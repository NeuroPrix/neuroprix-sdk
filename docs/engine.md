# The engine and its versions

`neuroprix.engine` is NeuroPrix's race engine: the circuits (procedural Grand Prix circuits and gyms), the physics
(tyres, braking, slipstream, battery, crashes and damage) and the senses. The official races run this same
code, and the website's game runs a JavaScript copy of it that's checked step by step against it.

`neuroprix.engine.ENGINE_VERSION` names the version (e.g. `2026.11.2`, which added N1's hills and banking). It changes whenever the physics, the
circuits or what drivers sense changes. The runner reports its version; if the server's is newer, the runner prints a
warning and the Train page flags the runner, because it would be training on different physics than the races use.
Update with `pip install -U neuroprix` (or rebuild the Docker image).

What a change means for your drivers: they keep racing, on the new physics, and may drive worse until they're
retrained (like a rule change in motorsport). Big changes come between seasons and are announced; a new input is
added switched off in existing brains, so they drive exactly as before until they learn to use it.

## How the car handles

Since engine 2026.11.9 the N3, N2 and N1 cars handle like real ones. The handling is a physical model whose numbers
could be measured on a real car, not rules that look right. In car lengths per second, today's numbers match a 1:10
radio-controlled car's: top speed about 17 car lengths a second, cornering grip about 1 g. N4 keeps its simpler
steering-only model.

- **Two axles with tyres.** Each axle's sideways force comes from its slip angle, the angle between where its wheels
  point and where they move. It peaks at about 8 degrees, and a fully sliding tyre keeps 89% of its peak, as racing
  slicks do, so a slide can be caught. Each axle's grip is its share of the car's weight times the track's grip
  (`LAT_GRIP`, with the race day's grip and, on hills, the load over crests and dips).
- **Mass and weight transfer.** The centre of mass sits a little behind the middle (`CG_FRONT`), the body has yaw
  inertia, and braking moves load onto the front tyres and power onto the rear (`CG_H`).
- **One grip for everything.** Each axle's braking or driving force and its sideways force share one friction circle.
  Brake hard and the fronts can't steer; floor it in a corner and the rear steps out.
- **Drive and brakes.** Rear-wheel drive, at most `DRIVE` and falling as `POWER` / speed, like a motor's power; boost
  multiplies both by 1.5. The brakes are biased 70% to the front (`BRAKE_FRONT`), so a car braking too late runs wide
  rather than spinning. Half brake and half throttle (brains that drive by degrees) use half the force.
- **Banks.** Gravity across a banked road pushes toward its low edge, so a corner banked into the turn holds more.
- **Low speed.** At walking pace the rear wheels follow the front, as on a real car, and a stopped car turns its wheels
  without moving.

`neuroprix.engine.Car.dynamics` is the whole model (integrated 4 times a frame); the constants above are at the top of
`neuroprix.engine`. Real driving is sensitive to small differences, and so is this: two runs that differ in the last
digit of a number drift apart over a long race with slides and collisions. Each engine is exactly repeatable on its
own.

## The optional Rust engine (faster training)

`neuroprix.engine` is pure Python, and it stays the engine to read and change: the SDK needs nothing else, and
`pip install neuroprix` installs no compiled code. If you'd like training to go faster and don't plan to change the
physics, there's an optional second engine: **neuroprix-rs**, the same engine ported to Rust (in this repository's
[neuroprix-rs](../neuroprix-rs) folder). It gives exactly the same numbers as `neuroprix.engine`, to the last
digit, step by step, so a driver trained on it is the same driver: with the same settings and seed, the two engines
train identical brains.

```bash
pip install "neuroprix[fast]"                 # or: pip install neuroprix-rs
neuroprix-runner --engine rust                # sessions from the Train page
neuroprix-train --engine rust --tier n1 ...   # or training on your own
```

`NEUROPRIX_TRAIN_ENGINE=rust` does the same for scripts and services. What it changes: the practice games run several
times faster, so a whole training update takes well under half the time (the learning step itself doesn't change).

**When the trainer uses it.** The Rust engine is never used on its own; it's used only when you ask for it, and then
only if all of these hold. Otherwise the trainer says why and trains on the Python engine:

1. neuroprix-rs is installed, and made for this engine version (its `ENGINE_VERSION` is `neuroprix.ENGINE_VERSION`);
2. `neuroprix/engine.py` is the one released with your neuroprix (its SHA-256 is `neuroprix.ENGINE_SHA256`). If you've
   changed the physics, your engine is what you train on: the trainer never swaps your engine for the Rust one;
3. a short check at the start, both engines side by side on two circuits, finds them exactly the same.

The trainer's first lines say which engine it's using (`engine: Rust, ...` or `engine: Python ...`).

**Changing the physics.** Change `neuroprix/engine.py`, as before: the Rust engine steps aside by rule 2. For the
engine itself, every change goes into the Python engine first, then into the Rust one, and a new neuroprix-rs is
released with the new `ENGINE_VERSION`; `neuroprix-rs/README.md` says how it's checked.

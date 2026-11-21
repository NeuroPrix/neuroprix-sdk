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
  (`LAT_GRIP`, with oil, the race day's grip and, on hills, the load over crests and dips).
- **Mass and weight transfer.** The centre of mass sits a little behind the middle (`CG_FRONT`), the body has yaw
  inertia, and braking moves load onto the front tyres and power onto the rear (`CG_H`).
- **One grip for everything.** Each axle's braking or driving force and its sideways force share one friction circle.
  Brake hard and the fronts can't steer; floor it in a corner and the rear steps out.
- **Drive and brakes.** Rear-wheel drive, at most `DRIVE` and falling as `POWER` / speed, like a motor's power; boost
  multiplies both by 1.5. The brakes are biased 70% to the front (`BRAKE_FRONT`), so a car braking too late runs wide
  rather than spinning. Half brake and half throttle (proportional brains) use half the force.
- **Banks.** Gravity across a banked road pushes toward its low edge, so a corner banked into the turn holds more.
- **Low speed.** At walking pace the rear wheels follow the front, as on a real car, and a stopped car turns its wheels
  without moving.

`neuroprix.engine.Car.dynamics` is the whole model (integrated 4 times a frame); the constants above are at the top of
`neuroprix.engine`. Real driving is sensitive to small differences, and so is this: two runs that differ in the last
digit of a number drift apart over a long race with slides and collisions. Each engine is exactly repeatable on its
own.

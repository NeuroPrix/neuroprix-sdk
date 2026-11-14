# The engine and its versions

`neuroprix.engine` is NeuroPrix's race engine: the circuits (procedural Grand Prix circuits and gyms), the physics
(grip, drift, braking, slipstream, battery, crashes and damage) and the senses. The official races run this same
code, and the website's game runs a JavaScript copy of it that's checked step by step against it.

`neuroprix.engine.ENGINE_VERSION` names the version (e.g. `2026.11.2`, which added N1's hills and banking). It changes whenever the physics, the
circuits or what drivers sense changes. The runner reports its version; if the server's is newer, the runner prints a
warning and the Train page flags the runner, because it would be training on different physics than the races use.
Update with `pip install -U neuroprix` (or rebuild the Docker image).

What a change means for your drivers: they keep racing, on the new physics, and may drive worse until they're
retrained (like a rule change in motorsport). Big changes come between seasons and are announced; a new input is
added switched off in existing brains, so they drive exactly as before until they learn to use it.

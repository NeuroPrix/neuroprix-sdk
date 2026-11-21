# Changelog

All notable changes to the NeuroPrix SDK. Versions follow [Semantic Versioning](https://semver.org/): a new major
version may need you to change your own trainer; minor and patch versions don't. The race engine has its own version
(`neuroprix.ENGINE_VERSION`, e.g. `2026.11.2`), shown by `neuroprix-runner --version`; when it changes, update the SDK
so you train on the same physics the races use.

## 0.3.2 (unreleased)

- **Cars that handle like real ones** (engine 2026.11.9; [docs/engine.md](docs/engine.md#how-the-car-handles)): N3,
  N2 and N1 cars are a physical model. Tyre forces come from slip at both axles; the body has yaw inertia; load moves
  under braking and power; one grip is shared between turning, braking and driving; rear-wheel drive has a motor's
  power; the brakes are biased to the front; and gravity acts across banks. The rule-based drift and its constants
  are gone. Brains trained on 2026.11.8 still load but drive a different car: train them again.
- **Proportional controls** ([docs/brain-format.md](docs/brain-format.md#action-layouts)): a new action layout,
  `proportional`, with steering in 7 levels, the pedal in 5 (full and half brake, coast, half and full throttle) and
  boost (N3 12 outputs, N2 / N1 14). It's the default for new N3, N2 and N1 brains (`neuroprix-train` and the
  server's sessions). `neuroprix.engine.controls(action)` reads any action; proportional actions are 1000 and up.
- **Training on a race day's grip:** `neuroprix-train` runs each episode at a grip between 0.94 and 1.06, as races
  vary it (`neuroprix.env.SimEnv.setup_world(..., race_grip=True)`; off by default).
- The coach's corner-speed marks (`corner_limits`) use 88% of the grip, what the car really holds steadily.

## 0.3.1 (2026-11-19)

- **The first 20% of damage costs no speed, and the radar sees further ahead** (engine 2026.11.8): a car keeps all of
  its top speed down to 80% health (a dent in the body), and below that top speed falls evenly to half at zero, as
  before at zero. The radar's front, front-left and front-right sectors see 600 px (about 2 s at top speed) instead of
  200 px, and their *near* reads by that range (1 − distance ÷ 600); the side and rear sectors stay at 200 px. The
  inputs are the same, so brain kinds 16–19 and every saved brain still load, but a brain trained on 2026.11.7 reads a
  car ahead as nearer than it is until it trains on the new engine. Update the SDK so you train on the physics the races use.

## 0.3.0 (2026-11-18)

- **Only the senses a real car could measure** (engine 2026.11.7): brain kinds 16–19 are the only ones. Brains of
  earlier kinds (1–15) are refused by `neuroprix.engine.Brain`, `neuroprix.brain.load_brain` and `neuroprix-train`
  with "this brain was made for senses NeuroPrix no longer uses: train a new one"; nothing in them carries over.
  Removed: `neuroprix.brain.to_sectors` and `add_heading`, `--senses slots` / `sectors` / `heading` (`--senses real`
  is the only one), the 2-slot radar and the map-read corner, grade, bank and tyre-load inputs. The minimal example
  refuses an earlier brain too.
- **Senses a real car could measure** (engine 2026.11.6;
  [docs/observations-and-actions.md](docs/observations-and-actions.md#inputs-in-order)): new brains (kinds 16–19: N4 46
  inputs, N3 47, N2 49, N1 57) read nothing off the track map but the heading. A fan of 9 wall rays (−90°, −60°, −40°,
  −20°, 0°, +20°, +40°, +60°, +90°; the 0° and ±20° ones car-wide beams) replaces the corners-ahead input; every class
  feels its yaw rate, sideslip and health; N1 gets its pitch and roll and six sight lines to the road edges 30, 90 and
  180 px ahead (reading the brow when a crest hides the road) instead of the map's grade, bank and tyre load.
  `neuroprix.env` and `neuroprix-train` default to them (`--senses real`); a continuing earlier brain keeps its own
  senses (the minimal example too). Banking only counts where the road curves: a bank fades out as a bend straightens
  and is zero on a straight, so gentle bends gain a little banked grip. Update the SDK: an older engine senses and
  drives differently from the races.
- **Car-wide forward rays and the heading sense** (engine 2026.11.5;
  [docs/observations-and-actions.md](docs/observations-and-actions.md#the-forward-beams)): the forward wall rays
  (−45°, 0°, +45°) are beams the car's width, for every brain, so a rock just off the nose's line is seen; and new
  brains have one more input, last, saying which way the track runs against the nose (kinds 12–15: N4 39 inputs, N3
  43, N2 45, N1 52), so a car spun round knows which way to go. `neuroprix.env` and `neuroprix-train` default to it
  (`--senses heading`); continuing a sector-radar brain adds it with zero weights (`neuroprix.brain.add_heading`; the
  minimal example does it in NumPy), so it drives exactly as before until it learns to use it. Update the SDK: an
  older engine senses differently from the races.
- **The sector radar is every class's radar** (engine 2026.11.4): new brains in every class sense 8 sectors all round
  the car, behind too (N4 38 inputs, N3 42, N2 44, N1 51; [docs/observations-and-actions.md](docs/observations-and-actions.md)).
  `neuroprix.env` gives every class the sector radar by default, `neuroprix-train --senses sectors` is the default,
  and continuing a 2-slot brain moves it over (`neuroprix.brain.to_sectors`; the minimal example does it in NumPy).
  Earlier brains still race as they were built. Update the SDK: an older runner trains 2-slot brains.
- **Action layouts** ([docs/brain-format.md](docs/brain-format.md#action-layouts)): a brain's `action_layout` says
  how its outputs become steering, pedal and boost: `combined` (one score per action, as before), `two_head` or
  `three_head` (a group per control, each picked on its own). New N1 brains are `three_head`. The engine, the
  environment, `neuroprix.brain` (`LAYOUTS`, `DEFAULT_LAYOUT`, `layout_outputs`, `pick_action`, `relayout`,
  `load_layout`) and `neuroprix-train --layout` handle all three; a brain that continues keeps its own.
- The session's `layout` setting ([the trainer contract](docs/trainer-contract.md)) and the runner accept it.
  Older runners refuse sessions that carry it, so update the SDK.
- `examples/minimal_trainer.py` builds the layout the session asks for and drives each head.

## 0.2.0 (2026-11-12): first public release

Engine 2026.11.3.

- **The runner** (`neuroprix-runner`): links a computer to your account with a code, trains the sessions you start
  on the Train page, sends back progress, replays and brains, and handles Stop on the Train page, Ctrl+C and `docker stop`.
  Carries on training through short connection drops and sends the progress when the server is back.
- **A live picture of the brain** on the Train page: the trainer sends its weights every few seconds (the `net`
  event, [docs/progress-events.md](docs/progress-events.md)); the runner keeps only the latest two while offline.
- **NeuroPrix's trainer** (`neuroprix-train`, PPO) for all four classes, N4 to N1, on gyms, numbered circuits or
  random Grand Prix circuits. Each session has its own random seed.
- **The race engine** (`neuroprix.engine`, `neuroprix.env`): the same physics, circuits and senses as the official
  races, including N1's hills, crests and banked corners.
- **Your own trainer**: any program following the trainer contract (`--trainer "python my_trainer.py"`), with two
  examples.
- `neuroprix-runner --version` shows the runner's and the engine's versions.
- A runner that can't reach the server while getting its linking code keeps trying instead of stopping with an error.
- A clear message when `--server` isn't a web address (`https://...`).
- The minimal trainer example handles N1.
- A Dockerfile with the CPU-only PyTorch, a full guide, tests and continuous integration.

Security (from an audit before release):

- The token file is created private (0600) from the start, in a private folder (0700).
- The token is only sent over HTTPS, except to a server on the same computer (`--allow-http` for a server on your
  own network), and the runner no longer follows redirects, which would have carried the token along.
- The runner no longer loads PyTorch files sent by the server (they can carry code with older PyTorch versions) and
  no longer uploads them: brains travel as JSON only.
- `.pth` files are only loaded with PyTorch 2.6 or newer: older versions can run code hidden in one even in their
  safe mode (CVE-2025-32434). With an older PyTorch (Intel Macs stop at 2.2), use the `.json` brain saved beside it.
- Text from the server or a trainer is cleaned of terminal control codes before it's printed; session ids are checked
  before they're used; replies are size-capped.
- The Docker image runs the runner as an ordinary user, not root. CI actions are pinned to exact commits.

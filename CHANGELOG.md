# Changelog

All notable changes to the NeuroPrix SDK. Versions follow [Semantic Versioning](https://semver.org/): a new major
version may need you to change your own trainer; minor and patch versions don't. The race engine has its own version
(`neuroprix.ENGINE_VERSION`, e.g. `2026.11.2`), shown by `neuroprix-runner --version`; when it changes, update the SDK
so you train on the same physics the races use.

## Unreleased

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
